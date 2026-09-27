from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QApplication, QDialog, QHeaderView, QHBoxLayout, QLabel, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .job_manager import JobManager, JobState


def _set_visual_state(widget: QWidget, state: str) -> None:
    """Refresh a semantic state selector only when the state changes."""

    normalized = str(state or "")
    if str(widget.property("state") or "") == normalized:
        return
    widget.setProperty("state", normalized)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class JobIndicatorWidget(QWidget):
    def __init__(self, job_manager: JobManager, parent=None):
        super().__init__(parent)
        self.job_manager = job_manager
        self._compact = False
        self._shutting_down = False
        self._jobs_dialog: JobsDialog | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)

        self.label = QLabel("")
        self.label.setMaximumWidth(260)
        self.label.setAccessibleName("Active job status")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(96)
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Active job progress")
        self.progress.setVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setAccessibleName("Cancel active job")
        self.cancel_btn.setVisible(False)
        self.jobs_btn = QPushButton("Jobs…")
        self.jobs_btn.setAccessibleName("Open all visible jobs and job history")
        self.jobs_btn.setToolTip("Open every active, queued, completed, cancelled, and failed job.")

        layout.addWidget(self.label)
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel_btn)
        layout.addWidget(self.jobs_btn)

        self.cancel_btn.clicked.connect(self._cancel_active)
        self.jobs_btn.clicked.connect(self._open_jobs_dialog)

        self.job_manager.job_added.connect(self._refresh)
        self.job_manager.job_updated.connect(self._refresh)
        self.job_manager.job_finished.connect(self._refresh)
        self._refresh()

    def _active_job(self) -> JobState | None:
        return self.job_manager.most_recent_active(foreground_only=True)

    def _refresh(self, *_args) -> None:
        if self._shutting_down:
            return
        job = self._active_job()
        active_count = self.job_manager.active_count()
        self.jobs_btn.setText(f"Jobs ({active_count})" if active_count else "Jobs")
        active = self._active_job()
        active_detail = f" Current: {active.origin}: {active.label}." if active is not None else ""
        self.jobs_btn.setToolTip(
            f"Open all visible jobs and job history. {active_count} job(s) are currently running or cancelling."
            f"{active_detail}"
        )
        self.jobs_btn.setAccessibleDescription(self.jobs_btn.toolTip())
        if job is None:
            if active_count:
                self.label.setText(f"{active_count} background job(s) running — open Jobs to view or cancel")
                self.label.setToolTip("Background work is visible and cancellable from Jobs.")
                self.label.setVisible(not self._compact)
            else:
                self.label.clear()
                self.label.hide()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setAccessibleDescription("No foreground job is currently running.")
            _set_visual_state(self.progress, "")
            self.progress.hide()
            self.cancel_btn.setVisible(False)
            return

        text = f"{job.origin}: {job.label}"
        if job.text:
            text += f" | {job.text}"
        if active_count > 1:
            text += f" ({active_count} jobs)"
        if job.status == "cancelling":
            text += " (cancelling)"
        elif job.status == "queued":
            text += " (queued)"
        if job.status == "cancelling":
            compact_text = f"{job.origin}: cancelling"
        elif job.status == "queued":
            compact_text = f"{job.origin}: queued"
        else:
            compact_text = f"{job.origin}: {job.progress}%" if job.progress is not None else f"{job.origin}: working"
        self.label.setText(compact_text if self._compact else text)
        self.label.setToolTip(text)
        self.label.setAccessibleDescription(text)
        self.label.setMaximumWidth(130 if self._compact else 260)
        self.label.setVisible(True)
        if job.status in {"queued", "cancelling"} or job.progress is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(job.progress))))
        _set_visual_state(self.progress, job.status)
        if job.status == "queued":
            progress_description = f"{job.label} is queued and waiting to start."
        elif job.status == "cancelling":
            progress_description = f"{job.label} is cancelling; waiting for the worker to stop safely."
        elif job.progress is None:
            progress_description = f"{job.label} is running; completion total is not known yet."
        else:
            progress_description = f"{job.label} is {max(0, min(100, int(job.progress)))} percent complete."
        if job.text:
            progress_description += f" {job.text}"
        self.progress.setAccessibleDescription(progress_description)
        self.progress.setVisible(not self._compact)

        self.cancel_btn.setText(f"Cancel {job.origin}" if self._compact else "Cancel")
        self.cancel_btn.setAccessibleName(f"Cancel {job.origin} job: {job.label}")
        self.cancel_btn.setToolTip(f"Cancel {job.origin}: {job.label}")
        self.cancel_btn.setVisible(bool(job.cancellable))

    def set_compact(self, compact: bool) -> None:
        """Collapse verbose job text while keeping Jobs and cancellation reachable."""
        self._compact = bool(compact)
        self._refresh()

    def _cancel_active(self) -> None:
        job = self._active_job()
        if job is None:
            return
        self.job_manager.cancel(job.job_id)

    def _open_jobs_dialog(self) -> None:
        if self._shutting_down:
            return
        dialog = self._jobs_dialog
        if dialog is None:
            dialog = JobsDialog(self.job_manager, parent=self)
            dialog.setModal(False)
            self._jobs_dialog = dialog
        dialog.set_focus_return(QApplication.focusWidget() or self.jobs_btn)
        dialog.refresh()
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def shutdown(self) -> None:
        """Detach the modeless monitor before its production shell is torn down."""

        if self._shutting_down:
            return
        self._shutting_down = True
        for signal in (
            self.job_manager.job_added,
            self.job_manager.job_updated,
            self.job_manager.job_finished,
        ):
            try:
                signal.disconnect(self._refresh)
            except (TypeError, RuntimeError):
                pass
        if self._jobs_dialog is not None:
            self._jobs_dialog.shutdown()


class JobsDialog(QDialog):
    def __init__(self, job_manager: JobManager, parent=None):
        super().__init__(parent)
        self.job_manager = job_manager
        self._shutting_down = False
        self._focus_return: QWidget | None = None
        self.setWindowTitle("Jobs")
        self.resize(860, 420)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setToolTip("All registered work is retained here, including background work and completed/cancelled jobs.")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setAccessibleName("Jobs history")
        self.table.setAccessibleDescription(
            "Running, queued, completed, cancelled, and failed work. Select a running job to cancel it."
        )
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["Origin", "Task", "Priority", "Status", "Progress", "Cache", "Details", "Started", "Finished"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setTabKeyNavigation(False)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        # Details must remain readable at the real 860 px launch size. A
        # stretch section is allowed to shrink below an explicit width when
        # the other columns consume the viewport, so keep it interactive and
        # let the table provide horizontal scrolling when necessary.
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Interactive)
        self._job_ids: list[int] = []
        self._job_snapshots: dict[int, tuple[object, ...]] = {}
        layout.addWidget(self.table)

        close_row = QHBoxLayout()
        self.cancel_selected_btn = QPushButton("Cancel selected")
        self.cancel_selected_btn.setAccessibleName("Cancel selected job")
        self.cancel_selected_btn.clicked.connect(self._cancel_selected)
        close_row.addWidget(self.cancel_selected_btn)
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.setAccessibleName("Close Jobs history")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.job_manager.job_added.connect(self.refresh)
        self.job_manager.job_updated.connect(self.refresh)
        self.job_manager.job_finished.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self._update_cancel_state)
        self.refresh()

    def set_focus_return(self, widget: QWidget | None) -> None:
        self._focus_return = widget

    def done(self, result: int) -> None:
        super().done(result)
        self._queue_focus_restore()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().closeEvent(event)
        self._queue_focus_restore()

    def _queue_focus_restore(self) -> None:
        target = self._focus_return
        self._focus_return = None
        if target is None:
            return

        def _restore() -> None:
            try:
                if target.isEnabled() and target.isVisible():
                    target.window().activateWindow()
                    target.setFocus(Qt.FocusReason.OtherFocusReason)
            except RuntimeError:
                return

        _restore()
        QTimer.singleShot(0, _restore)

    @staticmethod
    def _snapshot(job: JobState) -> tuple[object, ...]:
        return (
            job.origin, job.label, job.foreground, job.status, job.progress,
            job.cache_status, job.text, job.started_at_s, job.finished_at_s, job.error,
        )

    def _set_item(self, row: int, column: int, value: object, *, alignment: bool = False, tooltip: str = "") -> None:
        item = self.table.item(row, column)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(row, column, item)
        item.setText(str(value))
        if alignment:
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
        if tooltip:
            item.setToolTip(tooltip)

    def _update_row(self, row: int, job: JobState) -> None:
        started = datetime.fromtimestamp(job.started_at_s).strftime("%H:%M:%S")
        finished = "" if job.finished_at_s is None else datetime.fromtimestamp(job.finished_at_s).strftime("%H:%M:%S")
        priority = "Visible" if job.foreground else "Background"
        status = {
            "finished": "Done",
            "cancelled": "Cancelled",
            "failed": "Failed",
            "cancelling": "Cancelling",
            "queued": "Queued",
            "running": "Running",
        }.get(job.status, str(job.status).title())
        values = (job.origin, job.label, priority, status, job.cache_status, job.text, started, finished)
        for column, value in enumerate(values):
            target_column = column if column < 4 else column + 1
            self._set_item(
                row,
                target_column,
                value,
                alignment=target_column == 3,
                tooltip=job.error if target_column == 1 else "",
            )
        task = self.table.item(row, 1)
        if task is not None:
            task.setData(Qt.ItemDataRole.UserRole, int(job.job_id))
        progress = self.table.cellWidget(row, 4)
        if not isinstance(progress, QProgressBar):
            progress = QProgressBar(self.table)
            progress.setAccessibleName(f"{job.label} progress")
            self.table.setCellWidget(row, 4, progress)
        _set_visual_state(progress, job.status)
        if job.status == "finished":
            progress.setRange(0, 100)
            progress.setValue(100)
            progress.setFormat("Done")
            progress.setTextVisible(True)
            progress.setAccessibleDescription("Completed successfully.")
        elif job.status in {"cancelled", "failed"}:
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setFormat("Cancelled" if job.status == "cancelled" else "Failed")
            progress.setTextVisible(True)
            if job.status == "failed" and job.error:
                progress.setAccessibleDescription(f"Failed: {job.error}")
            else:
                progress.setAccessibleDescription("Cancelled before completion." if job.status == "cancelled" else "Failed.")
        elif job.status in {"queued", "cancelling"} or job.progress is None:
            progress.setRange(0, 0)
            progress.setFormat("")
            progress.setTextVisible(False)
            if job.status == "queued":
                progress.setAccessibleDescription("Queued; waiting to start. Completion total is not known yet.")
            elif job.status == "cancelling":
                progress.setAccessibleDescription("Cancelling; waiting for the worker to stop safely.")
            else:
                progress.setAccessibleDescription("In progress; completion total is not known yet.")
        else:
            progress.setRange(0, 100)
            progress.setValue(max(0, min(100, int(job.progress or 0))))
            progress.setFormat("%p%")
            progress.setTextVisible(True)
            progress.setAccessibleDescription(f"{progress.value()} percent complete.")

    def refresh(self, *_args) -> None:
        if self._shutting_down:
            return
        jobs = self.job_manager.history(limit=500)
        active_count = self.job_manager.active_count()
        self.summary_label.setText(
            f"{active_count} active job(s). Select any cancellable running job to stop it safely; "
            "completed, cancelled, and failed work remains visible below."
        )
        selected = self._selected_job()
        job_ids = [job.job_id for job in jobs]
        rebuilt = job_ids != self._job_ids
        if rebuilt:
            self.table.setRowCount(len(jobs))
            self._job_ids = job_ids
            self._job_snapshots = {}
        for row, job in enumerate(jobs):
            snapshot = self._snapshot(job)
            if rebuilt or self._job_snapshots.get(job.job_id) != snapshot:
                self._update_row(row, job)
                self._job_snapshots[job.job_id] = snapshot
        if rebuilt:
            self.table.resizeColumnsToContents()
            self.table.setColumnWidth(6, max(260, self.table.columnWidth(6)))
        if selected is not None and selected.job_id in self._job_ids:
            self.table.selectRow(self._job_ids.index(selected.job_id))
        self._update_cancel_state()

    def _selected_job(self) -> JobState | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 1)
        if item is None:
            return None
        job_id = item.data(Qt.ItemDataRole.UserRole)
        return self.job_manager.get(int(job_id)) if job_id is not None else None

    def _update_cancel_state(self) -> None:
        job = self._selected_job()
        self.cancel_selected_btn.setEnabled(bool(job is not None and job.cancellable))

    def _cancel_selected(self) -> None:
        job = self._selected_job()
        if job is not None and job.cancellable:
            self.job_manager.cancel(job.job_id)

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        for signal in (
            self.job_manager.job_added,
            self.job_manager.job_updated,
            self.job_manager.job_finished,
        ):
            try:
                signal.disconnect(self.refresh)
            except (TypeError, RuntimeError):
                pass
        self.close()

