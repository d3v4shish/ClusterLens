from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer
from PyQt6 import sip

from .footer_bar import WorkspaceFooter
from .job_manager import JobManager, JobState


class JobPresentationController(QObject):
    """Mirror the latest foreground job into the shared workspace footer.

    Individual views keep their own inline loading feedback. This controller is
    the sole writer of footer progress, preventing one background operation
    from erasing another operation's visible state.
    """

    def __init__(self, job_manager: JobManager, footer: WorkspaceFooter, parent=None):
        super().__init__(parent)
        self._job_manager = job_manager
        self._footer = footer
        self._last_job_id: int | None = None
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        job_manager.job_added.connect(self._schedule_refresh)
        job_manager.job_updated.connect(self._schedule_refresh)
        job_manager.job_finished.connect(self._schedule_refresh)
        self.refresh()

    def _schedule_refresh(self, *_args) -> None:
        if not self._footer_is_alive():
            return
        if not self._refresh_timer.isActive():
            self._refresh_timer.start(50)

    def _footer_is_alive(self) -> bool:
        if self._footer is None:
            return False
        try:
            return not bool(sip.isdeleted(self._footer))
        except Exception:
            return True

    def shutdown(self) -> None:
        """Stop queued footer updates before its parent window is deleted."""

        self._refresh_timer.stop()
        self._footer = None

    @staticmethod
    def _text_for(job: JobState, active_count: int) -> str:
        text = f"{job.origin}: {job.label}"
        if job.text:
            text += f" — {job.text}"
        if job.status == "cancelling":
            text += " (cancelling)"
        if active_count > 1:
            text += f" ({active_count} active jobs)"
        return text

    def refresh(self) -> None:
        if not self._footer_is_alive():
            return
        job = self._job_manager.most_recent_active(foreground_only=True)
        if job is None:
            if self._last_job_id is not None:
                self._footer.set_progress(None)
            self._last_job_id = None
            return
        self._last_job_id = job.job_id
        self._footer.set_progress(
            -1 if job.progress is None else job.progress,
            self._text_for(job, self._job_manager.active_count()),
        )
