from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
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

    def changeEvent(self, event) -> None:
        result = super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.StyleChange}:
            self._apply_elision()
        return result

    def _apply_elision(self) -> None:
        width = max(32, self.contentsRect().width())
        text = self.fontMetrics().elidedText(self._full_text, Qt.TextElideMode.ElideRight, width)
        super().setText(text)


class WorkspaceFooter(QWidget):
    clear_storage_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._progress_owner_active = False
        self._deferred_status = ""
        self.setObjectName("workspaceFooter")
        self._sync_height()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(10)

        self.folder_label = ElidedLabel("Folder: none", self)
        self.folder_label.setObjectName("footerFolderChip")
        self.folder_label.setAccessibleName("Selected folder")
        self.folder_label.setMinimumWidth(220)
        self.status_label = ElidedLabel("Idle", self)
        self.metrics_label = ElidedLabel("", self)
        self.performance_dashboard_label = self.metrics_label
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.storage_label = ElidedLabel("Storage: scanning...", self)
        self.storage_label.setMinimumWidth(220)
        self.clear_storage_button = QPushButton("Clear rebuildable data", self)
        self.clear_storage_button.setFixedHeight(24)
        self.clear_storage_button.setToolTip(
            "Clear rebuildable caches, runtime temp files, support bundles, and benchmark artifacts. "
            "This action stays unavailable while clustering or other background work is active."
        )
        self.clear_storage_button.clicked.connect(self.clear_storage_requested.emit)
        self.clear_storage_button.hide()

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()

        layout.addWidget(self.folder_label, stretch=3)
        layout.addWidget(self.status_label, stretch=4)
        layout.addWidget(self.metrics_label, stretch=5)
        layout.addWidget(self.storage_label, stretch=3)
        layout.addWidget(self.clear_storage_button, stretch=0)
        layout.addWidget(self.progress_bar, stretch=0)

    def _sync_height(self) -> None:
        self.setFixedHeight(max(38, self.fontMetrics().lineSpacing() + 12))

    def changeEvent(self, event) -> None:
        result = super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.StyleChange}:
            self._sync_height()
        return result

    def set_status(self, text: str) -> None:
        value = str(text or "Idle")
        if self._progress_owner_active:
            self._deferred_status = value
            return
        self.status_label.setText(value)

    def set_selected_folder(self, path: str) -> None:
        directory = str(path or "").strip()
        self.folder_label.setText(f"Folder: {directory}" if directory else "Folder: none")
        self.folder_label.setToolTip(directory or "No folder selected")

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
        if value is None:
            self._progress_owner_active = False
            self.progress_bar.hide()
            self.progress_bar.reset()
            if text:
                self._deferred_status = ""
                self.status_label.setText(str(text))
            elif self._deferred_status:
                deferred = self._deferred_status
                self._deferred_status = ""
                self.status_label.setText(deferred)
            return
        self._progress_owner_active = True
        if text:
            self._deferred_status = ""
            self.status_label.setText(str(text))
        self.progress_bar.show()
        if int(value) < 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(value))))
