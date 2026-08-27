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
    cache_status: str = ""
    cancel_fn: Callable[[], None] | None = None

    @property
    def cancellable(self) -> bool:
        return self.cancel_fn is not None and self.status == "running"


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

    def register_job(self, label: str, cancel_fn: Callable[[], None] | None = None) -> int:
        job_id = self._next_id
        self._next_id += 1
        self._jobs[job_id] = JobState(job_id=job_id, label=str(label), cancel_fn=cancel_fn, started_at_s=time())
        self._history.append(job_id)
        self._prune_history()
        self.job_added.emit(job_id)
        return job_id

    def update(
        self,
        job_id: int,
        progress: int | None = None,
        text: str | None = None,
        cache_status: str | None = None,
    ) -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"running", "cancelling"}:
            return
        if progress is not None:
            job.progress = None if progress < 0 else int(progress)
        if text is not None:
            job.text = str(text)
        if cache_status is not None:
            job.cache_status = str(cache_status)
        self.job_updated.emit(job.job_id)

    def finish(self, job_id: int, status: str = "finished", error: str = "", cache_status: str = "") -> None:
        job = self._jobs.get(int(job_id))
        if job is None or job.status not in {"running", "cancelling"}:
            return
        normalized_status = str(status)
        if normalized_status not in {"finished", "failed", "cancelled"}:
            normalized_status = "failed"
        job.status = normalized_status
        job.error = str(error or "")
        if cache_status:
            job.cache_status = str(cache_status)
        job.finished_at_s = time()
        job.cancel_fn = None
        self.job_finished.emit(job.job_id)
        self._prune_history()

    def cancel(self, job_id: int) -> None:
        job = self._jobs.get(int(job_id))
        if job is None or not job.cancellable:
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

    def _prune_history(self) -> None:
        if len(self._history) <= self._max_history:
            return
        retained: list[int] = []
        removable = len(self._history) - self._max_history
        for job_id in self._history:
            job = self._jobs.get(job_id)
            if removable > 0 and job is not None and job.status not in {"running", "cancelling"}:
                self._jobs.pop(job_id, None)
                removable -= 1
            else:
                retained.append(job_id)
        self._history = retained
