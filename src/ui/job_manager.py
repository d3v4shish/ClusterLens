from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class JobState:
    job_id: int
    label: str
    progress: int | None = None  # None => indeterminate
    text: str = ""
    status: str = "running"  # queued|running|cancelling|finished|failed|cancelled
    started_at_s: float = 0.0
    finished_at_s: float | None = None
    error: str = ""
    cache_status: str = ""
    cancel_fn: Callable[[], None] | None = None
    origin: str = "Workspace"
    foreground: bool = True
    updated_at_s: float = 0.0

    @property
    def cancellable(self) -> bool:
        return self.cancel_fn is not None and self.status in {"queued", "running"}


class JobManager(QObject):
    """
    Lightweight UI job registry for long-running background work.
    """

    job_added = pyqtSignal(int)
    job_updated = pyqtSignal(int)
    job_finished = pyqtSignal(int)

    def __init__(self, parent=None, *, max_history: int = 500):
        super().__init__(parent)
        self._next_id = 1
        self._jobs: dict[int, JobState] = {}
        self._history: list[int] = []
        self._max_history = max(10, int(max_history))

    def _emit_safely(self, signal_name: str, *args) -> None:
        """Ignore queued lifecycle delivery after the owning window is gone."""

        try:
            getattr(self, signal_name).emit(*args)
        except RuntimeError:
            pass

    def register_job(
        self,
        label: str,
        cancel_fn: Callable[[], None] | None = None,
        *,
        origin: str = "Workspace",
        foreground: bool = True,
    ) -> int:
        """Register visible background work without blocking the Qt event loop."""
        job_id = self._next_id
        self._next_id += 1
        now = time()
        self._jobs[job_id] = JobState(
            job_id=job_id,
            label=str(label),
            cancel_fn=cancel_fn,
            origin=str(origin or "Workspace"),
            foreground=bool(foreground),
            started_at_s=now,
            updated_at_s=now,
        )
        self._history.append(job_id)
        self._prune_history()
        self._emit_safely("job_added", job_id)
        return job_id

    def update(
        self,
        job_id: int,
        progress: int | None = None,
        text: str | None = None,
        cache_status: str | None = None,
    ) -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"queued", "running", "cancelling"}:
            return
        if progress is not None:
            job.progress = None if progress < 0 else int(progress)
        if text is not None:
            job.text = str(text)
        if cache_status is not None:
            job.cache_status = str(cache_status)
        job.updated_at_s = time()
        self._emit_safely("job_updated", job.job_id)

    def set_queued(self, job_id: int, *, text: str = "Waiting to schedule.", cache_status: str = "queued") -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"queued", "running"}:
            return
        job.status = "queued"
        job.progress = None
        job.text = str(text)
        job.cache_status = str(cache_status)
        job.updated_at_s = time()
        self._emit_safely("job_updated", job.job_id)

    def start(self, job_id: int, *, text: str = "Running.", cache_status: str = "") -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"queued", "running"}:
            return
        job.status = "running"
        job.text = str(text)
        job.cache_status = str(cache_status)
        job.updated_at_s = time()
        self._emit_safely("job_updated", job.job_id)

    def finish(self, job_id: int, status: str = "finished", error: str = "", cache_status: str = "") -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"queued", "running", "cancelling"}:
            return
        normalized_status = str(status)
        if normalized_status not in {"finished", "failed", "cancelled"}:
            normalized_status = "failed"
        job.status = normalized_status
        job.error = str(error or "")
        if cache_status:
            job.cache_status = str(cache_status)
        job.finished_at_s = time()
        job.updated_at_s = job.finished_at_s
        job.cancel_fn = None
        self._emit_safely("job_finished", job.job_id)
        self._prune_history()

    def cancel(self, job_id: int) -> None:
        job = self._jobs.get(int(job_id))
        if job is None or not job.cancellable:
            return
        try:
            job.status = "cancelling"
            job.updated_at_s = time()
            self._emit_safely("job_updated", job.job_id)
            job.cancel_fn()
        except Exception as exc:
            self.finish(job_id, status="failed", error=str(exc))

    def get(self, job_id: int) -> JobState | None:
        return self._jobs.get(int(job_id))

    def active_jobs(self) -> list[JobState]:
        return [job for job in self._jobs.values() if job.status in {"queued", "running", "cancelling"}]

    def history(self, limit: int = 100) -> list[JobState]:
        ids = list(self._history)[-max(1, int(limit)) :]
        out: list[JobState] = []
        for job_id in reversed(ids):
            job = self._jobs.get(job_id)
            if job is not None:
                out.append(job)
        return out

    def most_recent_active(self, *, foreground_only: bool = False) -> JobState | None:
        """Return the newest active job, independent of progress frequency.

        ``updated_at_s`` is deliberately excluded: otherwise two concurrent
        workers can steal footer/cancellation ownership on every progress
        callback. Registration order is the deterministic tie-breaker.
        """

        active = [
            job
            for job in self._jobs.values()
            if job.status in {"queued", "running", "cancelling"} and (not foreground_only or job.foreground)
        ]
        return max(active, key=lambda job: (job.started_at_s, job.job_id), default=None)

    def active_count(self, *, foreground_only: bool = False) -> int:
        return len(
            [
                job
                for job in self._jobs.values()
                if job.status in {"queued", "running", "cancelling"} and (not foreground_only or job.foreground)
            ]
        )

    def bind_async_job(
        self,
        job: object,
        label: str,
        *,
        origin: str,
        foreground: bool = True,
    ) -> int:
        """Bind an AsyncJob-like object to the job registry lifecycle."""
        cancel_fn = getattr(job, "cancel", None)
        job_id = self.register_job(
            label,
            cancel_fn=cancel_fn if callable(cancel_fn) else None,
            origin=origin,
            foreground=foreground,
        )
        progress_signal = getattr(job, "progress", None)
        completed_signal = getattr(job, "completed", None)
        failed_signal = getattr(job, "failed", None)
        cancelled_signal = getattr(job, "cancelled", None)
        if progress_signal is not None:
            progress_signal.connect(
                lambda value, text, job_id=job_id: self.update(job_id, progress=value, text=str(text))
            )
        if completed_signal is not None:
            completed_signal.connect(lambda _result, job_id=job_id: self.finish(job_id, status="finished"))
        if failed_signal is not None:
            failed_signal.connect(
                lambda message, job_id=job_id: self.finish(job_id, status="failed", error=str(message))
            )
        if cancelled_signal is not None:
            cancelled_signal.connect(lambda job_id=job_id: self.finish(job_id, status="cancelled"))
        return job_id

    def _prune_history(self) -> None:
        if len(self._history) <= self._max_history:
            return
        retained: list[int] = []
        removable = len(self._history) - self._max_history
        for job_id in self._history:
            job = self._jobs.get(job_id)
            if removable > 0 and job is not None and job.status not in {"queued", "running", "cancelling"}:
                self._jobs.pop(job_id, None)
                removable -= 1
            else:
                retained.append(job_id)
        self._history = retained
