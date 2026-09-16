from __future__ import annotations

from datetime import datetime

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QProgressBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from .job_manager import JobManager, JobState


class JobIndicatorWidget(QWidget):
    def __init__(self, job_manager: JobManager, parent=None):
        super().__init__(parent)
        self.job_manager = job_manager
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
        job = self._active_job()
        active_count = self.job_manager.active_count()
        self.jobs_btn.setText(f"Jobs ({active_count})" if active_count else "Jobs")
        self.jobs_btn.setToolTip(
            f"Open all visible jobs and job history. {active_count} job(s) are currently running or cancelling."
        )
        if job is None:
            if active_count:
                self.label.setText(f"{active_count} background job(s) running — open Jobs to view or cancel")
                self.label.setToolTip("Background work is visible and cancellable from Jobs.")
                self.label.show()
            else:
                self.label.clear()
                self.label.hide()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
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
        self.label.setText(text)
        self.label.setToolTip(text)
        self.label.show()
        if job.progress is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(job.progress))))
        self.progress.show()

        self.cancel_btn.setVisible(bool(job.cancellable))

    def _cancel_active(self) -> None:
        job = self._active_job()
        if job is None:
            return
        self.job_manager.cancel(job.job_id)

    def _open_jobs_dialog(self) -> None:
        dialog = JobsDialog(self.job_manager, parent=self)
        dialog.exec()


class JobsDialog(QDialog):
    def __init__(self, job_manager: JobManager, parent=None):
        super().__init__(parent)
        self.job_manager = job_manager
        self.setWindowTitle("Jobs")
        self.resize(860, 420)
        layout = QVBoxLayout(self)

        self.summary_label = QLabel("", self)
        self.summary_label.setWordWrap(True)
        self.summary_label.setToolTip("All registered work is retained here, including background work and completed/cancelled jobs.")
        layout.addWidget(self.summary_label)

        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels(["Origin", "Task", "Priority", "Status", "Progress", "Cache", "Details", "Started", "Finished"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        close_row = QHBoxLayout()
        self.cancel_selected_btn = QPushButton("Cancel selected")
        self.cancel_selected_btn.setAccessibleName("Cancel selected job")
        self.cancel_selected_btn.clicked.connect(self._cancel_selected)
        close_row.addWidget(self.cancel_selected_btn)
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.job_manager.job_added.connect(self.refresh)
        self.job_manager.job_updated.connect(self.refresh)
        self.job_manager.job_finished.connect(self.refresh)
        self.table.itemSelectionChanged.connect(self._update_cancel_state)
        self.refresh()

    def refresh(self, *_args) -> None:
        jobs = self.job_manager.history(limit=500)
        active_count = self.job_manager.active_count()
        self.summary_label.setText(
            f"{active_count} active job(s). Select any cancellable running job to stop it safely; "
            "completed, cancelled, and failed work remains visible below."
        )
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            started = datetime.fromtimestamp(job.started_at_s).strftime("%H:%M:%S")
            finished = "" if job.finished_at_s is None else datetime.fromtimestamp(job.finished_at_s).strftime("%H:%M:%S")
            priority = "Visible" if job.foreground else "Background"
            values = [job.origin, job.label, priority, job.status, "", job.cache_status, job.text, started, finished]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 3:
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                if col == 1 and job.error:
                    item.setToolTip(job.error)
                if col == 1:
                    item.setData(Qt.ItemDataRole.UserRole, int(job.job_id))
                self.table.setItem(row, col, item)
            progress = QProgressBar(self.table)
            progress.setTextVisible(job.progress is not None)
            progress.setAccessibleName(f"{job.label} progress")
            if job.progress is None and job.status in {"running", "cancelling"}:
                progress.setRange(0, 0)
            else:
                progress.setRange(0, 100)
                progress.setValue(max(0, min(100, int(job.progress or 0))))
            self.table.setCellWidget(row, 4, progress)
        self.table.resizeColumnsToContents()
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

