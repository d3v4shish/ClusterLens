from __future__ import annotations

"""Resource-aware scheduling for visible, cancellable desktop work.

The coordinator intentionally schedules *ownership*, not implementation.  A
workspace keeps its existing worker implementation but declares the resources
it needs before it starts.  This makes independent jobs concurrent while
giving users an explicit, recoverable choice when CUDA work would contend.
"""

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from PyQt6.QtCore import QObject, pyqtSignal

from .job_manager import JobManager


GpuConflictChoice = Literal["ask", "queue", "cpu", "cancel"]


@dataclass(frozen=True)
class SourceScope:
    """A caller-prepared source identity safe to compare on the UI thread."""

    path: str
    identity: tuple[int, int] | None = None


def prepare_source_scope(path: str | os.PathLike[str]) -> SourceScope:
    """Resolve aliases and inode identity before submitting work.

    This function can touch the filesystem and must therefore run in request
    preparation, not in a Qt input/paint callback.
    """

    resolved = Path(path).expanduser().resolve(strict=False)
    identity: tuple[int, int] | None = None
    try:
        stat_result = resolved.stat()
        identity = (int(stat_result.st_dev), int(stat_result.st_ino))
    except OSError:
        pass
    return SourceScope(path=os.path.normcase(os.path.normpath(str(resolved))), identity=identity)


def _normalize_source_scope(value: str | SourceScope) -> SourceScope:
    if isinstance(value, SourceScope):
        return SourceScope(
            path=os.path.normcase(os.path.normpath(os.path.abspath(str(value.path)))),
            identity=value.identity,
        )
    return SourceScope(path=os.path.normcase(os.path.normpath(os.path.abspath(str(value)))))


def _normalize_source_scopes(values: tuple[str | SourceScope, ...]) -> tuple[SourceScope, ...]:
    scopes = {
        _normalize_source_scope(value)
        for value in values
        if str(value.path if isinstance(value, SourceScope) else value).strip()
    }
    return tuple(sorted(scopes, key=lambda item: (item.path, item.identity or (-1, -1))))


@dataclass(frozen=True)
class JobSpec:
    label: str
    origin: str = "Workspace"
    foreground: bool = True
    cpu_slots: int = 1
    io_bound: bool = False
    uses_gpu: bool = False
    cpu_fallback: bool = False
    source_locks: tuple[str | SourceScope, ...] = ()
    source_reads: tuple[str | SourceScope, ...] = ()
    source_writes: tuple[str | SourceScope, ...] = ()
    data_home_read: bool = False
    data_home_write: bool = False
    model_cache_read: bool = False
    model_cache_write: bool = False
    depends_on: tuple[int, ...] = ()

    def normalized(self) -> "JobSpec":
        return JobSpec(
            label=str(self.label or "Background work"),
            origin=str(self.origin or "Workspace"),
            foreground=bool(self.foreground),
            cpu_slots=max(1, int(self.cpu_slots or 1)),
            io_bound=bool(self.io_bound),
            uses_gpu=bool(self.uses_gpu),
            cpu_fallback=bool(self.cpu_fallback),
            source_locks=_normalize_source_scopes(self.source_locks),
            source_reads=_normalize_source_scopes(self.source_reads),
            source_writes=_normalize_source_scopes(self.source_writes),
            data_home_read=bool(self.data_home_read),
            data_home_write=bool(self.data_home_write),
            model_cache_read=bool(self.model_cache_read),
            model_cache_write=bool(self.model_cache_write),
            depends_on=tuple(dict.fromkeys(int(job_id) for job_id in self.depends_on if int(job_id) > 0)),
        )


@dataclass
class _QueuedWork:
    job_id: int
    spec: JobSpec
    starter: Callable[[int, bool], None]
    cancel: Callable[[], None] | None = None
    use_cpu_fallback: bool = False
    waiting_for_decision: bool = False
    gpu_decision: GpuConflictChoice | None = None
    queued_reason: str = ""


class WorkCoordinator(QObject):
    """Schedule visible workspace jobs without doing work on the Qt thread.

    ``starter(job_id, use_cpu_fallback)`` must start non-UI-thread work and
    eventually call :meth:`finish`.  A cancellation requested before start
    removes queued work; after start it delegates to the registered cancel
    callback.  The coordinator only holds locks while the job is active.
    """

    gpu_conflict_requested = pyqtSignal(int, str)
    job_queued = pyqtSignal(int, str)
    job_started = pyqtSignal(int, bool)
    job_finished = pyqtSignal(int, str)

    def __init__(
        self,
        job_manager: JobManager,
        parent=None,
        *,
        gpu_policy: GpuConflictChoice = "ask",
        cpu_capacity: int = 4,
        io_capacity: int = 2,
        max_pending: int = 256,
    ) -> None:
        super().__init__(parent)
        self.job_manager = job_manager
        self.gpu_policy: GpuConflictChoice = self._normalize_policy(gpu_policy)
        self.cpu_capacity = max(1, int(cpu_capacity))
        self.io_capacity = max(1, int(io_capacity))
        self.max_pending = max(1, int(max_pending))
        self._queued: dict[int, _QueuedWork] = {}
        self._running: dict[int, _QueuedWork] = {}

    def _emit_safely(self, signal_name: str, *args) -> None:
        try:
            getattr(self, signal_name).emit(*args)
        except RuntimeError:
            pass

    def submit(
        self,
        spec: JobSpec,
        starter: Callable[[int, bool], None],
        *,
        cancel: Callable[[], None] | None = None,
    ) -> int:
        normalized = spec.normalized()
        rejection = ""
        if normalized.cpu_slots > self.cpu_capacity:
            rejection = (
                f"Requires {normalized.cpu_slots} CPU slots, but scheduler capacity is "
                f"{self.cpu_capacity}."
            )
        elif len(self._queued) >= self.max_pending:
            rejection = f"Scheduler queue capacity ({self.max_pending}) is full."
        job_id = self.job_manager.register_job(
            normalized.label,
            cancel_fn=(
                None
                if rejection
                else lambda job_id_ref=None: self.cancel(
                    job_id if job_id_ref is None else int(job_id_ref)
                )
            ),
            origin=normalized.origin,
            foreground=normalized.foreground,
        )
        if rejection:
            self.job_manager.finish(job_id, status="failed", error=rejection)
            self._emit_safely("job_finished", job_id, "failed")
            return job_id
        work = _QueuedWork(job_id=job_id, spec=normalized, starter=starter, cancel=cancel)
        self._queued[job_id] = work
        self.job_manager.set_queued(job_id)
        self._schedule()
        return job_id

    def submit_async_job(
        self,
        spec: JobSpec,
        job: object,
        starter: Callable[[bool], object],
    ) -> int:
        """Submit one AsyncJob-like worker with a single visible lifecycle.

        The worker is not started until its declared resources are owned.
        Public progress and terminal signals update the same JobManager row
        registered by :meth:`submit`; callers must not bind a second row.
        ``starter`` receives the coordinator's CPU-fallback decision.
        """

        lifecycle = {"started": False}
        job_id_holder: dict[str, int] = {}

        def _emit_unstarted_terminal(status: str) -> None:
            if lifecycle["started"]:
                return
            if status == "cancelled":
                signal = getattr(job, "cancelled", None)
                if signal is not None:
                    signal.emit()
                return
            if status == "failed":
                signal = getattr(job, "failed", None)
                state = self.job_manager.get(job_id_holder.get("job_id", 0))
                if signal is not None:
                    signal.emit(str(state.error if state is not None else "Scheduling failed"))

        def _coordinator_finished(finished_job_id: int, status: str) -> None:
            if int(finished_job_id) != job_id_holder.get("job_id"):
                return
            try:
                self.job_finished.disconnect(_coordinator_finished)
            except (TypeError, RuntimeError):
                pass
            _emit_unstarted_terminal(str(status))

        def _start(job_id: int, use_cpu_fallback: bool) -> None:
            lifecycle["started"] = True
            progress_signal = getattr(job, "progress", None)
            completed_signal = getattr(job, "completed", None)
            failed_signal = getattr(job, "failed", None)
            cancelled_signal = getattr(job, "cancelled", None)
            if progress_signal is not None:
                progress_signal.connect(
                    lambda value, text, job_id=job_id: self.job_manager.update(
                        job_id,
                        progress=int(value),
                        text=str(text),
                    )
                )
            if completed_signal is not None:
                completed_signal.connect(
                    lambda _result, job_id=job_id: self.finish(job_id, status="finished")
                )
            if failed_signal is not None:
                failed_signal.connect(
                    lambda message, job_id=job_id: self.finish(
                        job_id,
                        status="failed",
                        error=str(message),
                    )
                )
            if cancelled_signal is not None:
                cancelled_signal.connect(
                    lambda job_id=job_id: self.finish(job_id, status="cancelled")
                )
            try:
                starter(bool(use_cpu_fallback))
            except Exception as exc:
                if failed_signal is None:
                    raise
                failed_signal.emit(str(exc))

        cancel = getattr(job, "cancel", None)
        job_id = self.submit(
            spec,
            _start,
            cancel=cancel if callable(cancel) else None,
        )
        job_id_holder["job_id"] = job_id
        self.job_finished.connect(_coordinator_finished)
        state = self.job_manager.get(job_id)
        if state is not None and state.status in {"failed", "cancelled", "finished"}:
            _coordinator_finished(job_id, state.status)
        return job_id

    def finish(self, job_id: int, *, status: str = "finished", error: str = "") -> None:
        job_id = int(job_id)
        normalized_status = status if status in {"finished", "failed", "cancelled"} else "failed"
        queued = self._queued.pop(job_id, None)
        running = self._running.pop(job_id, None)
        if queued is not None or running is not None:
            self.job_manager.finish(job_id, status=normalized_status, error=error)
            self._emit_safely("job_finished", job_id, normalized_status)
        self._schedule()

    def cancel(self, job_id: int) -> None:
        job_id = int(job_id)
        queued = self._queued.pop(job_id, None)
        if queued is not None:
            self.job_manager.finish(job_id, status="cancelled")
            self._emit_safely("job_finished", job_id, "cancelled")
            self._schedule()
            return
        running = self._running.get(job_id)
        if running is None:
            return
        self.job_manager.update(job_id, text="Cancellation requested at the next safe checkpoint")
        if running.cancel is not None:
            running.cancel()

    def resolve_gpu_conflict(self, job_id: int, choice: GpuConflictChoice, *, remember: bool = False) -> None:
        work = self._queued.get(int(job_id))
        if work is None:
            return
        normalized = self._normalize_policy(choice)
        if remember and normalized in {"queue", "cpu"}:
            self.gpu_policy = normalized
        work.waiting_for_decision = False
        if normalized == "cancel":
            self.cancel(work.job_id)
            return
        work.gpu_decision = normalized
        work.use_cpu_fallback = normalized == "cpu" and work.spec.cpu_fallback
        self.job_manager.update(
            work.job_id,
            text="Will use CPU fallback when scheduled." if work.use_cpu_fallback else "Waiting for the GPU resource.",
            cache_status="cpu-fallback" if work.use_cpu_fallback else "queued-for-gpu",
        )
        self._schedule()

    def _schedule(self) -> None:
        progressed = True
        while progressed:
            progressed = False
            for work in tuple(self._queued.values()):
                dependency_error = self._dependency_error(work.spec)
                if dependency_error:
                    self._queued.pop(work.job_id, None)
                    self.job_manager.finish(work.job_id, status="cancelled", error=dependency_error)
                    self._emit_safely("job_finished", work.job_id, "cancelled")
                    progressed = True
                    break
                if work.waiting_for_decision or not self._dependencies_finished(work.spec):
                    continue
                conflict = self._resource_conflict_for(work)
                if conflict == "GPU" and work.spec.cpu_fallback and not work.use_cpu_fallback:
                    decision = work.gpu_decision or self.gpu_policy
                    if decision == "ask":
                        work.waiting_for_decision = True
                        self.job_manager.update(work.job_id, text="GPU is busy. Choose Queue, CPU fallback, or Cancel.")
                        self._emit_safely("gpu_conflict_requested", work.job_id, work.spec.label)
                        continue
                    if decision == "cancel":
                        self._queued.pop(work.job_id, None)
                        self.job_manager.finish(work.job_id, status="cancelled")
                        self._emit_safely("job_finished", work.job_id, "cancelled")
                        progressed = True
                        break
                    if decision == "cpu":
                        work.use_cpu_fallback = True
                        conflict = self._resource_conflict_for(work)
                if conflict is not None:
                    self._mark_queued(work, conflict)
                    continue
                capacity_conflict = self._capacity_conflict_for(work)
                if capacity_conflict is not None:
                    self._mark_queued(work, capacity_conflict)
                    if capacity_conflict == "CPU capacity":
                        return
                    continue
                self._queued.pop(work.job_id, None)
                self._running[work.job_id] = work
                mode = "CPU fallback" if work.use_cpu_fallback else "running"
                self.job_manager.start(work.job_id, text=mode, cache_status="cpu-fallback" if work.use_cpu_fallback else "")
                self._emit_safely("job_started", work.job_id, work.use_cpu_fallback)
                try:
                    work.starter(work.job_id, work.use_cpu_fallback)
                except Exception as exc:
                    self.finish(work.job_id, status="failed", error=str(exc))
                progressed = True
                break

    def _mark_queued(self, work: _QueuedWork, reason: str) -> None:
        if work.queued_reason == reason:
            return
        work.queued_reason = reason
        self.job_manager.set_queued(
            work.job_id,
            text=f"Queued: waiting for {reason}.",
            cache_status="queued",
        )
        self._emit_safely("job_queued", work.job_id, reason)

    def _dependencies_finished(self, spec: JobSpec) -> bool:
        for dependency in spec.depends_on:
            job = self.job_manager.get(dependency)
            if job is None or job.status in {"running", "cancelling"}:
                return False
            if job.status != "finished":
                return False
        return True

    def _dependency_error(self, spec: JobSpec) -> str:
        for dependency in spec.depends_on:
            job = self.job_manager.get(dependency)
            if job is None:
                return f"Dependency {dependency} is unavailable."
            if job.status in {"failed", "cancelled"}:
                return f"Dependency {dependency} did not complete."
        return ""

    def _capacity_conflict_for(self, candidate: _QueuedWork) -> str | None:
        used_cpu = sum(work.spec.cpu_slots for work in self._running.values())
        if used_cpu + candidate.spec.cpu_slots > self.cpu_capacity:
            return "CPU capacity"
        if candidate.spec.io_bound:
            used_io = sum(1 for work in self._running.values() if work.spec.io_bound)
            if used_io >= self.io_capacity:
                return "I/O capacity"
        return None

    def _resource_conflict_for(self, candidate: _QueuedWork) -> str | None:
        for running in self._running.values():
            if (
                candidate.spec.data_home_write
                and (running.spec.data_home_read or running.spec.data_home_write)
            ) or (
                candidate.spec.data_home_read and running.spec.data_home_write
            ):
                return "Data Home access"
            if (
                candidate.spec.model_cache_write
                and (running.spec.model_cache_read or running.spec.model_cache_write)
            ) or (
                candidate.spec.model_cache_read and running.spec.model_cache_write
            ):
                return "model cache access"
            if self._source_access_conflicts(candidate.spec, running.spec):
                return "the same source files"
            if candidate.spec.uses_gpu and not candidate.use_cpu_fallback and running.spec.uses_gpu and not running.use_cpu_fallback:
                return "GPU"
        return None

    @classmethod
    def _source_access_conflicts(cls, candidate: JobSpec, running: JobSpec) -> bool:
        candidate_reads = tuple(candidate.source_reads)
        candidate_writes = tuple(candidate.source_locks) + tuple(candidate.source_writes)
        running_reads = tuple(running.source_reads)
        running_writes = tuple(running.source_locks) + tuple(running.source_writes)
        return (
            cls._scope_sets_overlap(candidate_writes, running_reads + running_writes)
            or cls._scope_sets_overlap(candidate_reads, running_writes)
        )

    @classmethod
    def _scope_sets_overlap(
        cls,
        left: tuple[SourceScope, ...],
        right: tuple[SourceScope, ...],
    ) -> bool:
        return any(cls._scopes_overlap(first, second) for first in left for second in right)

    @staticmethod
    def _scopes_overlap(first: SourceScope, second: SourceScope) -> bool:
        if first.identity is not None and second.identity is not None and first.identity == second.identity:
            return True
        try:
            common = os.path.commonpath((first.path, second.path))
        except ValueError:
            return False
        return common == first.path or common == second.path

    @staticmethod
    def _normalize_policy(value: str) -> GpuConflictChoice:
        normalized = str(value or "ask").strip().lower()
        return normalized if normalized in {"ask", "queue", "cpu", "cancel"} else "ask"  # type: ignore[return-value]
