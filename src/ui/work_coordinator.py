from __future__ import annotations

"""Resource-aware scheduling for visible, cancellable desktop work.

The coordinator intentionally schedules *ownership*, not implementation.  A
workspace keeps its existing worker implementation but declares the resources
it needs before it starts.  This makes independent jobs concurrent while
giving users an explicit, recoverable choice when CUDA work would contend.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from PyQt6.QtCore import QObject, pyqtSignal

from .job_manager import JobManager


GpuConflictChoice = Literal["ask", "queue", "cpu", "cancel"]


@dataclass(frozen=True)
class JobSpec:
    label: str
    origin: str = "Workspace"
    foreground: bool = True
    cpu_slots: int = 1
    io_bound: bool = False
    uses_gpu: bool = False
    cpu_fallback: bool = False
    source_locks: tuple[str, ...] = ()
    data_home_write: bool = False
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
            source_locks=tuple(sorted({str(path) for path in self.source_locks if str(path or "").strip()})),
            data_home_write=bool(self.data_home_write),
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

    def __init__(self, job_manager: JobManager, parent=None, *, gpu_policy: GpuConflictChoice = "ask") -> None:
        super().__init__(parent)
        self.job_manager = job_manager
        self.gpu_policy: GpuConflictChoice = self._normalize_policy(gpu_policy)
        self._queued: dict[int, _QueuedWork] = {}
        self._running: dict[int, _QueuedWork] = {}

    def submit(
        self,
        spec: JobSpec,
        starter: Callable[[int, bool], None],
        *,
        cancel: Callable[[], None] | None = None,
    ) -> int:
        normalized = spec.normalized()
        job_id = self.job_manager.register_job(
            normalized.label,
            cancel_fn=lambda job_id_ref=None: self.cancel(job_id if job_id_ref is None else int(job_id_ref)),
            origin=normalized.origin,
            foreground=normalized.foreground,
        )
        work = _QueuedWork(job_id=job_id, spec=normalized, starter=starter, cancel=cancel)
        self._queued[job_id] = work
        self._schedule()
        return job_id

    def finish(self, job_id: int, *, status: str = "finished", error: str = "") -> None:
        job_id = int(job_id)
        self._queued.pop(job_id, None)
        if self._running.pop(job_id, None) is not None:
            self.job_manager.finish(job_id, status=status, error=error)
            self.job_finished.emit(job_id, status)
        self._schedule()

    def cancel(self, job_id: int) -> None:
        job_id = int(job_id)
        queued = self._queued.pop(job_id, None)
        if queued is not None:
            self.job_manager.finish(job_id, status="cancelled")
            self.job_finished.emit(job_id, "cancelled")
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
        work.use_cpu_fallback = normalized == "cpu"
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
                    self.job_finished.emit(work.job_id, "cancelled")
                    progressed = True
                    break
                if work.waiting_for_decision or not self._dependencies_finished(work.spec):
                    continue
                conflict = self._conflict_for(work)
                if conflict == "GPU" and work.spec.cpu_fallback and not work.use_cpu_fallback:
                    if self.gpu_policy == "ask":
                        work.waiting_for_decision = True
                        self.job_manager.update(work.job_id, text="GPU is busy. Choose Queue, CPU fallback, or Cancel.")
                        self.gpu_conflict_requested.emit(work.job_id, work.spec.label)
                        continue
                    if self.gpu_policy == "cpu":
                        work.use_cpu_fallback = True
                        conflict = None
                if conflict is not None:
                    self.job_manager.update(work.job_id, text=f"Queued: waiting for {conflict}.")
                    self.job_queued.emit(work.job_id, conflict)
                    continue
                self._queued.pop(work.job_id, None)
                self._running[work.job_id] = work
                mode = "CPU fallback" if work.use_cpu_fallback else "running"
                self.job_manager.update(work.job_id, text=mode, cache_status="cpu-fallback" if work.use_cpu_fallback else "")
                self.job_started.emit(work.job_id, work.use_cpu_fallback)
                try:
                    work.starter(work.job_id, work.use_cpu_fallback)
                except Exception as exc:
                    self.finish(work.job_id, status="failed", error=str(exc))
                progressed = True
                break

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

    def _conflict_for(self, candidate: _QueuedWork) -> str | None:
        for running in self._running.values():
            if candidate.spec.data_home_write and running.spec.data_home_write:
                return "Data Home write"
            if set(candidate.spec.source_locks).intersection(running.spec.source_locks):
                return "the same source files"
            if candidate.spec.uses_gpu and not candidate.use_cpu_fallback and running.spec.uses_gpu and not running.use_cpu_fallback:
                return "GPU"
        return None

    @staticmethod
    def _normalize_policy(value: str) -> GpuConflictChoice:
        normalized = str(value or "ask").strip().lower()
        return normalized if normalized in {"ask", "queue", "cpu", "cancel"} else "ask"  # type: ignore[return-value]
