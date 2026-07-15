from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QResizeEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QProgressBar, QPushButton, QSizePolicy, QWidget


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full_text = str(text or "")
        self.setWordWrap(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._full_text = str(text or "")
        self.setToolTip(self._full_text)
        self._apply_elision()

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        width = max(32, self.contentsRect().width())
        text = self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, width)
        super().setText(text)


class WorkspaceFooter(QWidget):
    clear_storage_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workspaceFooter")
        self.setFixedHeight(38)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.status_label = ElidedLabel("Idle", self)
        self.metrics_label = ElidedLabel("", self)
        self.performance_dashboard_label = self.metrics_label
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.storage_label = ElidedLabel("Storage: scanning...", self)
        self.storage_label.setMinimumWidth(220)
        self.clear_storage_button = QPushButton("Clear Caches / Temp", self)
        self.clear_storage_button.setFixedHeight(24)
        self.clear_storage_button.setToolTip(
            "Clear rebuildable caches, runtime temp files, support bundles, and benchmark artifacts. "
            "This action stays unavailable while clustering or other background work is active."
        )
        self.clear_storage_button.clicked.connect(self.clear_storage_requested.emit)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        layout.addWidget(self.status_label, stretch=4)
        layout.addWidget(self.metrics_label, stretch=5)
        layout.addWidget(self.storage_label, stretch=3)
        layout.addWidget(self.clear_storage_button, stretch=0)
        layout.addWidget(self.progress_bar, stretch=0)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or "Idle")

    def set_metrics(self, text: str) -> None:
        self.metrics_label.setText(text or "")

    def set_performance_dashboard(self, text: str) -> None:
        self.performance_dashboard_label.setText(text or "")

    def set_storage_usage(self, text: str, *, tooltip: str = "") -> None:
        full_text = str(text or "").strip() or "Storage: unavailable"
        self.storage_label.setText(full_text)
        self.storage_label.setToolTip(tooltip or full_text)

    def set_clear_storage_enabled(self, enabled: bool, *, reason: str = "") -> None:
        self.clear_storage_button.setEnabled(bool(enabled))
        if reason:
            self.clear_storage_button.setToolTip(reason)
        elif self.clear_storage_button.toolTip().startswith("Unavailable"):
            self.clear_storage_button.setToolTip(
                "Clear rebuildable caches, runtime temp files, support bundles, and benchmark artifacts. "
                "This action stays unavailable while clustering or other background work is active."
            )

    def set_progress(self, value: int | None, text: str = "") -> None:
        if text:
            self.set_status(text)
        if value is None:
            self.progress_bar.hide()
            self.progress_bar.reset()
            return
        self.progress_bar.show()
        if int(value) < 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(value))))
