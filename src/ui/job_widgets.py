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
        self.label.setMinimumWidth(220)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        self.progress.setTextVisible(False)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setVisible(False)
        self.jobs_btn = QPushButton("Jobs…")

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
        return self.job_manager.most_recent_active()

    def _refresh(self, *_args) -> None:
        job = self._active_job()
        if job is None:
            self.label.setText("Idle")
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.cancel_btn.setVisible(False)
            return

        text = job.label
        if job.text:
            text += f" | {job.text}"
        if job.status == "cancelling":
            text += " (cancelling)"
        self.label.setText(text)

        if job.progress is None:
            self.progress.setRange(0, 0)  # indeterminate
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(job.progress))))

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

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Label", "Status", "Progress", "Text", "Started", "Finished"])
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        self.job_manager.job_added.connect(self.refresh)
        self.job_manager.job_updated.connect(self.refresh)
        self.job_manager.job_finished.connect(self.refresh)
        self.refresh()

    def refresh(self, *_args) -> None:
        jobs = self.job_manager.history(limit=200)
        self.table.setRowCount(len(jobs))
        for row, job in enumerate(jobs):
            started = datetime.fromtimestamp(job.started_at_s).strftime("%H:%M:%S")
            finished = "" if job.finished_at_s is None else datetime.fromtimestamp(job.finished_at_s).strftime("%H:%M:%S")
            progress = "" if job.progress is None else str(job.progress)
            values = [job.label, job.status, progress, job.text, started, finished]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col in (1, 2):
                    item.setTextAlignment(int(Qt.AlignmentFlag.AlignCenter))
                if col == 0 and job.error:
                    item.setToolTip(job.error)
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

