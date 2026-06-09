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
    status: str = "running"  # running|cancelling|finished|failed|cancelled
    started_at_s: float = 0.0
    finished_at_s: float | None = None
    error: str = ""
    cancel_fn: Callable[[], None] | None = None

    @property
    def cancellable(self) -> bool:
        return self.cancel_fn is not None and self.status in {"running", "cancelling"}


class JobManager(QObject):
    """
    Lightweight UI job registry for long-running background work.
    """

    job_added = pyqtSignal(int)
    job_updated = pyqtSignal(int)
    job_finished = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._next_id = 1
        self._jobs: dict[int, JobState] = {}
        self._history: list[int] = []

    def register_job(self, label: str, cancel_fn: Callable[[], None] | None = None) -> int:
        job_id = self._next_id
        self._next_id += 1
        self._jobs[job_id] = JobState(job_id=job_id, label=str(label), cancel_fn=cancel_fn, started_at_s=time())
        self._history.append(job_id)
        self.job_added.emit(job_id)
        return job_id

    def update(self, job_id: int, progress: int | None = None, text: str | None = None) -> None:
        job = self._jobs.get(int(job_id))
        if job is None:
            return
        if progress is not None:
            job.progress = None if progress < 0 else int(progress)
        if text is not None:
            job.text = str(text)
        self.job_updated.emit(job.job_id)

    def finish(self, job_id: int, status: str = "finished", error: str = "") -> None:
        job = self._jobs.get(int(job_id))
        if job is None:
            return
        job.status = str(status)
        job.error = str(error or "")
        job.finished_at_s = time()
        self.job_finished.emit(job.job_id)

    def cancel(self, job_id: int) -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.cancel_fn is None:
            return
        try:
            job.status = "cancelling"
            self.job_updated.emit(job.job_id)
            job.cancel_fn()
        except Exception as exc:
            self.finish(job_id, status="failed", error=str(exc))

    def get(self, job_id: int) -> JobState | None:
        return self._jobs.get(int(job_id))

    def active_jobs(self) -> list[JobState]:
        return [job for job in self._jobs.values() if job.status in {"running", "cancelling"}]

    def history(self, limit: int = 100) -> list[JobState]:
        ids = list(self._history)[-max(1, int(limit)) :]
        out: list[JobState] = []
        for job_id in reversed(ids):
            job = self._jobs.get(job_id)
            if job is not None:
                out.append(job)
        return out

    def most_recent_active(self) -> JobState | None:
        for job_id in reversed(self._history):
            job = self._jobs.get(job_id)
            if job is not None and job.status in {"running", "cancelling"}:
                return job
        return None

