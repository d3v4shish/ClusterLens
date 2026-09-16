from __future__ import annotations

from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QWheelEvent
from PyQt6.QtWidgets import QLabel, QScrollArea


class ZoomableImageView(QScrollArea):
    zoom_changed = pyqtSignal(float)
    face_box_clicked = pyqtSignal(int)
    face_box_drawn = pyqtSignal(tuple)
    face_box_moved = pyqtSignal(int, tuple)
    face_box_resized = pyqtSignal(int, tuple)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWidget(self._label)
        self._pixmap: QPixmap | None = None
        self._face_boxes: tuple[tuple[float, float, float, float], ...] = ()
        self._face_boxes_normalized = False
        self._selected_face_indexes: set[int] = set()
        self._face_boxes_dirty = False
        self._draw_mode = False
        self._drag_origin: tuple[int, int] | None = None
        self._draft_box: tuple[int, int, int, int] | None = None
        self._drag_face_index: int | None = None
        self._drag_face_origin: tuple[int, int] | None = None
        self._drag_face_box: tuple[float, float, float, float] | None = None
        self._drag_face_started = False
        self._resize_face_index: int | None = None
        self._resize_face_handle: str | None = None
        self._resize_face_box: tuple[float, float, float, float] | None = None
        self._resize_face_started = False
        self._zoom = 1.0
        self._user_zoomed = False
        self._pan_origin: QPoint | None = None
        self._pan_scroll_origin: tuple[int, int] | None = None
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

    def set_face_boxes(
        self,
        face_boxes: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...] | None,
        *,
        normalized: bool = False,
        dirty: bool = False,
    ) -> None:
        self._face_boxes = tuple(
            (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            )
            for box in (face_boxes or ())
            if len(box) == 4
        )
        self._face_boxes_normalized = bool(normalized)
        self._face_boxes_dirty = bool(dirty)
        self._apply_zoom()

    def set_selected_face_indexes(self, indexes) -> None:
        self._selected_face_indexes = {int(index) for index in (indexes or ()) if int(index) >= 0}
        self._apply_zoom()

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        if not self._draw_mode:
            self._drag_origin = None
            self._draft_box = None
            self._drag_face_index = None
            self._drag_face_origin = None
            self._drag_face_box = None
            self._drag_face_started = False
            self._resize_face_index = None
            self._resize_face_handle = None
            self._resize_face_box = None
            self._resize_face_started = False
        self._apply_zoom()

    def draw_mode_enabled(self) -> bool:
        return bool(self._draw_mode)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        position = event.position()
        old_zoom = float(self._zoom)
        old_width = max(1.0, float(self._pixmap.width() if self._pixmap is not None else 1) * old_zoom)
        old_height = max(1.0, float(self._pixmap.height() if self._pixmap is not None else 1) * old_zoom)
        anchor_x = (float(self.horizontalScrollBar().value()) + float(position.x())) / old_width
        anchor_y = (float(self.verticalScrollBar().value()) + float(position.y())) / old_height
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._user_zoomed = True
        self._zoom = max(0.02, min(64.0, self._zoom * factor))
        self._apply_zoom()
        if self._pixmap is not None and not self._pixmap.isNull():
            self.horizontalScrollBar().setValue(
                int(round(anchor_x * self._pixmap.width() * self._zoom - float(position.x())))
            )
            self.verticalScrollBar().setValue(
                int(round(anchor_y * self._pixmap.height() * self._zoom - float(position.y())))
            )
        self.zoom_changed.emit(float(self._zoom))
        event.accept()

    def zoom_in(self) -> None:
        self._set_zoom(self._zoom * 1.25)

    def zoom_out(self) -> None:
        self._set_zoom(self._zoom / 1.25)

    def actual_size(self) -> None:
        self._set_zoom(1.0)

    def _set_zoom(self, zoom: float) -> None:
        self._user_zoomed = True
        self._zoom = max(0.02, min(64.0, float(zoom)))
        self._apply_zoom()
        self.zoom_changed.emit(float(self._zoom))

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
        self._zoom = max(0.02, min(64.0, float(scale)))
        self._apply_zoom()
        self.zoom_changed.emit(float(self._zoom))

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = event.position().toPoint()
            self._pan_scroll_origin = (
                int(self.horizontalScrollBar().value()),
                int(self.verticalScrollBar().value()),
            )
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._viewport_point_to_image(event.position())
            if point is not None:
                if self._draw_mode:
                    self._drag_origin = point
                    self._draft_box = (point[0], point[1], point[0], point[1])
                    self._apply_zoom()
                    event.accept()
                    return
                resize_hit = self._face_box_resize_handle_at_point(point)
                if resize_hit is not None:
                    self._resize_face_index = int(resize_hit[0])
                    self._resize_face_handle = str(resize_hit[1])
                    self._resize_face_box = tuple(self._face_boxes[int(resize_hit[0])])
                    self._resize_face_started = False
                    event.accept()
                    return
                clicked_index = self._face_box_index_at_point(point)
                if clicked_index is not None:
                    self._drag_face_index = int(clicked_index)
                    self._drag_face_origin = point
                    self._drag_face_box = tuple(self._face_boxes[int(clicked_index)])
                    self._drag_face_started = False
                    event.accept()
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_origin is not None and self._pan_scroll_origin is not None:
            delta = event.position().toPoint() - self._pan_origin
            self.horizontalScrollBar().setValue(int(self._pan_scroll_origin[0] - delta.x()))
            self.verticalScrollBar().setValue(int(self._pan_scroll_origin[1] - delta.y()))
            event.accept()
            return
        if self._draw_mode and self._drag_origin is not None:
            point = self._viewport_point_to_image(event.position())
            if point is not None:
                self._draft_box = (
                    self._drag_origin[0],
                    self._drag_origin[1],
                    point[0],
                    point[1],
                )
                self._apply_zoom()
                event.accept()
                return
        if (
            self._resize_face_index is not None
            and self._resize_face_handle is not None
            and self._resize_face_box is not None
        ):
            point = self._viewport_point_to_image(event.position())
            if point is not None:
                resized_box = self._resize_face_box_to_point(self._resize_face_box, self._resize_face_handle, point)
                if resized_box is not None and resized_box != tuple(self._face_boxes[self._resize_face_index]):
                    face_boxes = list(self._face_boxes)
                    face_boxes[self._resize_face_index] = resized_box
                    self._face_boxes = tuple(face_boxes)
                    self._resize_face_started = True
                    self._apply_zoom()
                    event.accept()
                    return
        if self._drag_face_index is not None and self._drag_face_origin is not None and self._drag_face_box is not None:
            point = self._viewport_point_to_image(event.position())
            if point is not None:
                moved_box = self._move_face_box(self._drag_face_box, self._drag_face_origin, point)
                if moved_box is not None and moved_box != tuple(self._face_boxes[self._drag_face_index]):
                    face_boxes = list(self._face_boxes)
                    face_boxes[self._drag_face_index] = moved_box
                    self._face_boxes = tuple(face_boxes)
                    self._drag_face_started = True
                    self._apply_zoom()
                    event.accept()
                    return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_origin is not None and event.button() == Qt.MouseButton.MiddleButton:
            self._pan_origin = None
            self._pan_scroll_origin = None
            self.viewport().unsetCursor()
            event.accept()
            return
        if self._draw_mode and self._drag_origin is not None and event.button() == Qt.MouseButton.LeftButton:
            point = self._viewport_point_to_image(event.position()) or self._drag_origin
            x1 = min(self._drag_origin[0], point[0])
            y1 = min(self._drag_origin[1], point[1])
            x2 = max(self._drag_origin[0], point[0])
            y2 = max(self._drag_origin[1], point[1])
            self._drag_origin = None
            self._draft_box = None
            self._draw_mode = False
            self._apply_zoom()
            if (x2 - x1) >= 2 and (y2 - y1) >= 2:
                self.face_box_drawn.emit((x1, y1, x2, y2))
            event.accept()
            return
        if self._resize_face_index is not None and event.button() == Qt.MouseButton.LeftButton:
            resized_index = int(self._resize_face_index)
            started = bool(self._resize_face_started)
            resized_box = None
            if started and 0 <= resized_index < len(self._face_boxes):
                resized_box = self._denormalized_box(self._face_boxes[resized_index])
            self._resize_face_index = None
            self._resize_face_handle = None
            self._resize_face_box = None
            self._resize_face_started = False
            if started and resized_box is not None:
                self.face_box_resized.emit(resized_index, resized_box)
            event.accept()
            return
        if self._drag_face_index is not None and event.button() == Qt.MouseButton.LeftButton:
            clicked_index = int(self._drag_face_index)
            started = bool(self._drag_face_started)
            moved_box = None
            if started and 0 <= clicked_index < len(self._face_boxes):
                moved_box = self._denormalized_box(self._face_boxes[clicked_index])
            self._drag_face_index = None
            self._drag_face_origin = None
            self._drag_face_box = None
            self._drag_face_started = False
            if started and moved_box is not None:
                self.face_box_moved.emit(clicked_index, moved_box)
            elif not started:
                self.face_box_clicked.emit(clicked_index)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _viewport_point_to_image(self, position) -> tuple[int, int] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        label_point = self._label.mapFrom(self.viewport(), QPoint(int(position.x()), int(position.y())))
        if label_point.x() < 0 or label_point.y() < 0:
            return None
        label_pixmap = self._label.pixmap()
        if label_pixmap is None or label_pixmap.isNull():
            return None
        if label_point.x() >= label_pixmap.width() or label_point.y() >= label_pixmap.height():
            return None
        scale_x = self._pixmap.width() / max(1, label_pixmap.width())
        scale_y = self._pixmap.height() / max(1, label_pixmap.height())
        return (
            max(0, min(self._pixmap.width(), int(round(label_point.x() * scale_x)))),
            max(0, min(self._pixmap.height(), int(round(label_point.y() * scale_y)))),
        )

    def _face_box_index_at_point(self, point: tuple[int, int]) -> int | None:
        for index in range(len(self._face_boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self._denormalized_box(self._face_boxes[index])
            if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
                return index
        return None

    def _face_box_resize_handle_at_point(self, point: tuple[int, int]) -> tuple[int, str] | None:
        threshold = max(4, int(round(8.0 / max(0.05, float(self._zoom)))))
        prioritized_indexes = list(self._selected_face_indexes) + [
            index for index in range(len(self._face_boxes)) if index not in self._selected_face_indexes
        ]
        for index in reversed(prioritized_indexes):
            if index < 0 or index >= len(self._face_boxes):
                continue
            x1, y1, x2, y2 = self._denormalized_box(self._face_boxes[index])
            handle_points = {
                "nw": (x1, y1),
                "ne": (x2, y1),
                "sw": (x1, y2),
                "se": (x2, y2),
            }
            for handle_name, handle_point in handle_points.items():
                if abs(point[0] - handle_point[0]) <= threshold and abs(point[1] - handle_point[1]) <= threshold:
                    return int(index), str(handle_name)
        return None

    def _denormalized_box(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        if self._face_boxes_normalized and self._pixmap is not None and not self._pixmap.isNull():
            return (
                int(round(float(box[0]) * self._pixmap.width())),
                int(round(float(box[1]) * self._pixmap.height())),
                int(round(float(box[2]) * self._pixmap.width())),
                int(round(float(box[3]) * self._pixmap.height())),
            )
        return (
            int(round(float(box[0]))),
            int(round(float(box[1]))),
            int(round(float(box[2]))),
            int(round(float(box[3]))),
        )

    def _move_face_box(
        self,
        original_box: tuple[float, float, float, float],
        origin: tuple[int, int],
        point: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        x1, y1, x2, y2 = self._denormalized_box(original_box)
        delta_x = int(point[0]) - int(origin[0])
        delta_y = int(point[1]) - int(origin[1])
        width = max(2, x2 - x1)
        height = max(2, y2 - y1)
        next_x1 = max(0, min(self._pixmap.width() - width, x1 + delta_x))
        next_y1 = max(0, min(self._pixmap.height() - height, y1 + delta_y))
        next_x2 = next_x1 + width
        next_y2 = next_y1 + height
        if self._face_boxes_normalized:
            return (
                float(next_x1) / max(1, self._pixmap.width()),
                float(next_y1) / max(1, self._pixmap.height()),
                float(next_x2) / max(1, self._pixmap.width()),
                float(next_y2) / max(1, self._pixmap.height()),
            )
        return (float(next_x1), float(next_y1), float(next_x2), float(next_y2))

    def _resize_face_box_to_point(
        self,
        original_box: tuple[float, float, float, float],
        handle: str,
        point: tuple[int, int],
    ) -> tuple[float, float, float, float] | None:
        if self._pixmap is None or self._pixmap.isNull():
            return None
        x1, y1, x2, y2 = self._denormalized_box(original_box)
        min_size = 2
        next_x1 = x1
        next_y1 = y1
        next_x2 = x2
        next_y2 = y2
        if "w" in handle:
            next_x1 = max(0, min(int(point[0]), x2 - min_size))
        if "e" in handle:
            next_x2 = min(self._pixmap.width(), max(int(point[0]), x1 + min_size))
        if "n" in handle:
            next_y1 = max(0, min(int(point[1]), y2 - min_size))
        if "s" in handle:
            next_y2 = min(self._pixmap.height(), max(int(point[1]), y1 + min_size))
        if self._face_boxes_normalized:
            return (
                float(next_x1) / max(1, self._pixmap.width()),
                float(next_y1) / max(1, self._pixmap.height()),
                float(next_x2) / max(1, self._pixmap.width()),
                float(next_y2) / max(1, self._pixmap.height()),
            )
        return (float(next_x1), float(next_y1), float(next_x2), float(next_y2))

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
        if self._face_boxes:
            canvas = QPixmap(scaled)
            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            scale_x = scaled.width() / max(1, self._pixmap.width())
            scale_y = scaled.height() / max(1, self._pixmap.height())
            for index, box in enumerate(self._face_boxes):
                if self._face_boxes_normalized:
                    x1 = int(round(float(box[0]) * scaled.width()))
                    y1 = int(round(float(box[1]) * scaled.height()))
                    x2 = int(round(float(box[2]) * scaled.width()))
                    y2 = int(round(float(box[3]) * scaled.height()))
                else:
                    x1 = int(round(float(box[0]) * scale_x))
                    y1 = int(round(float(box[1]) * scale_y))
                    x2 = int(round(float(box[2]) * scale_x))
                    y2 = int(round(float(box[3]) * scale_y))
                color = QColor("#EF4444") if self._face_boxes_dirty else QColor("#22C55E")
                pen = QPen(color, 4 if index in self._selected_face_indexes else 2)
                painter.setPen(pen)
                painter.drawRect(x1, y1, max(2, x2 - x1), max(2, y2 - y1))
                if index in self._selected_face_indexes:
                    handle_size = 8
                    half_handle = handle_size // 2
                    for handle_x, handle_y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                        painter.fillRect(
                            handle_x - half_handle,
                            handle_y - half_handle,
                            handle_size,
                            handle_size,
                            QColor("#FFFFFF"),
                        )
                        painter.setPen(QPen(color, 1))
                        painter.drawRect(
                            handle_x - half_handle,
                            handle_y - half_handle,
                            handle_size,
                            handle_size,
                        )
            if self._draft_box is not None:
                x1 = int(round(min(self._draft_box[0], self._draft_box[2]) * scale_x))
                y1 = int(round(min(self._draft_box[1], self._draft_box[3]) * scale_y))
                x2 = int(round(max(self._draft_box[0], self._draft_box[2]) * scale_x))
                y2 = int(round(max(self._draft_box[1], self._draft_box[3]) * scale_y))
                painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.PenStyle.DashLine))
                painter.drawRect(x1, y1, max(2, x2 - x1), max(2, y2 - y1))
            painter.end()
            scaled = canvas
        elif self._draft_box is not None:
            canvas = QPixmap(scaled)
            painter = QPainter(canvas)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            scale_x = scaled.width() / max(1, self._pixmap.width())
            scale_y = scaled.height() / max(1, self._pixmap.height())
            x1 = int(round(min(self._draft_box[0], self._draft_box[2]) * scale_x))
            y1 = int(round(min(self._draft_box[1], self._draft_box[3]) * scale_y))
            x2 = int(round(max(self._draft_box[0], self._draft_box[2]) * scale_x))
            y2 = int(round(max(self._draft_box[1], self._draft_box[3]) * scale_y))
            painter.setPen(QPen(QColor("#F59E0B"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(x1, y1, max(2, x2 - x1), max(2, y2 - y1))
            painter.end()
            scaled = canvas
        self._label.setPixmap(scaled)
        self._label.resize(scaled.size())
