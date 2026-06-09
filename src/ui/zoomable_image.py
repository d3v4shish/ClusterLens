from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea


class ZoomableImageView(QScrollArea):
    zoom_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._user_zoomed = False
        # "Fit" should be a true fit-to-viewport operation (including scaling up),
        # otherwise small/preview pixmaps can look like thin strips.
        self._fit_scale_up = True

    def set_fit_scale_up(self, enabled: bool) -> None:
        self._fit_scale_up = bool(enabled)
        if not self._user_zoomed:
            self.fit_to_window()

    def set_pixmap(self, pixmap: QPixmap | None, preserve_zoom: bool = False) -> None:
        previous_zoom = self._zoom
        previous_user_zoomed = self._user_zoomed
        self._pixmap = pixmap
        if preserve_zoom and previous_user_zoomed and pixmap is not None and not pixmap.isNull():
            self._user_zoomed = True
            self._zoom = previous_zoom
            self._apply_zoom()
            self.zoom_changed.emit(float(self._zoom))
            return
        self._user_zoomed = False
        self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._user_zoomed = True
        self._zoom = max(0.05, min(25.0, self._zoom * factor))
        self._apply_zoom()
        self.zoom_changed.emit(float(self._zoom))
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if not self._user_zoomed:
            self.fit_to_window()

    def fit_to_window(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._zoom = 1.0
            self._apply_zoom()
            self.zoom_changed.emit(float(self._zoom))
            return
        viewport = self.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            self._zoom = 1.0
            self._apply_zoom()
            self.zoom_changed.emit(float(self._zoom))
            return
        scale_w = viewport.width() / max(1, self._pixmap.width())
        scale_h = viewport.height() / max(1, self._pixmap.height())
        scale = min(scale_w, scale_h)
        if not self._fit_scale_up:
            scale = min(1.0, scale)
        self._zoom = max(0.05, min(25.0, float(scale)))
        self._apply_zoom()
        self.zoom_changed.emit(float(self._zoom))

    def _apply_zoom(self) -> None:
        if self._pixmap is None or self._pixmap.isNull():
            self._label.clear()
            self._label.adjustSize()
            return
        scaled = self._pixmap.scaled(
            int(self._pixmap.width() * self._zoom),
            int(self._pixmap.height() * self._zoom),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
