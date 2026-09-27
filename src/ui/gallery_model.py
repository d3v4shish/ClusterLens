from __future__ import annotations

from bisect import bisect_left
from collections import OrderedDict
from dataclasses import dataclass

from PyQt6.QtCore import QAbstractListModel, QEvent, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from PyQt6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionButton, QStyleOptionViewItem, QApplication

from ui.theme import COLORS, theme_revision

MAX_FACE_BOXES_PER_TILE = 16


@dataclass
class GalleryItem:
    image_path: str
    checked: bool = False
    failed: bool = False
    error: str = ""
    overlay: str = ""
    subtitle: str = ""
    face_boxes: tuple[tuple[float, float, float, float], ...] = ()
    face_box_state: str = "saved"


class GalleryImageModel(QAbstractListModel):
    PathRole = Qt.ItemDataRole.UserRole + 1
    PixmapRole = Qt.ItemDataRole.UserRole + 2
    FailedRole = Qt.ItemDataRole.UserRole + 3
    ErrorRole = Qt.ItemDataRole.UserRole + 4
    OverlayRole = Qt.ItemDataRole.UserRole + 5
    SubtitleRole = Qt.ItemDataRole.UserRole + 6
    FaceBoxesRole = Qt.ItemDataRole.UserRole + 7
    FaceBoxStateRole = Qt.ItemDataRole.UserRole + 8

    def __init__(self, parent=None, *, image_cache_capacity: int = 256):
        super().__init__(parent)
        self.items: list[GalleryItem] = []
        self._image_cache_capacity = max(32, int(image_cache_capacity))
        self._images_by_row: OrderedDict[int, QImage] = OrderedDict()
        self._preview_face_boxes_by_path: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
        self._rows_by_path: dict[str, list[int]] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self.items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return None
        item = self.items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.image_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if role == Qt.ItemDataRole.AccessibleTextRole:
            name = item.image_path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
            selection = "selected" if item.checked else "not selected"
            state = "failed to load" if item.failed else ("loaded" if index.row() in self._images_by_row else "loading")
            context = f", {item.overlay}" if item.overlay else ""
            return f"{name}, {selection}, {state}, photo {index.row() + 1} of {len(self.items)}{context}"
        if role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked
        if role == self.PathRole:
            return item.image_path
        if role == self.PixmapRole:
            image = self._images_by_row.get(index.row())
            if image is not None:
                self._images_by_row.move_to_end(index.row())
            return image
        if role == self.FailedRole:
            return item.failed
        if role == self.ErrorRole:
            return item.error
        if role == self.OverlayRole:
            return item.overlay
        if role == self.SubtitleRole:
            return item.subtitle
        if role == self.FaceBoxesRole:
            return self._preview_face_boxes_by_path.get(item.image_path, item.face_boxes)
        if role == self.FaceBoxStateRole:
            return item.face_box_state
        return None

    def flags(self, index: QModelIndex):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return (
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsUserCheckable
        )

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole):
        if not index.isValid() or not (0 <= index.row() < len(self.items)):
            return False
        if role == Qt.ItemDataRole.CheckStateRole:
            self.items[index.row()].checked = value == Qt.CheckState.Checked
            self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
            return True
        return False

    def set_images(self, image_paths: list[str]) -> None:
        self.beginResetModel()
        self.items = [GalleryItem(image_path=path) for path in image_paths]
        self._images_by_row.clear()
        self._preview_face_boxes_by_path = {}
        self._rows_by_path = self._build_rows_by_path(self.items)
        self.endResetModel()

    def image_paths(self) -> list[str]:
        return [item.image_path for item in self.items]

    @staticmethod
    def _build_rows_by_path(items: list[GalleryItem]) -> dict[str, list[int]]:
        rows_by_path: dict[str, list[int]] = {}
        for row, item in enumerate(items):
            rows_by_path.setdefault(str(item.image_path), []).append(int(row))
        return rows_by_path

    def rows_for_path(self, image_path: str) -> list[int]:
        return list(self._rows_by_path.get(str(image_path or ""), ()))

    def _target_rows(self, limit_paths: set[str] | None = None) -> list[int]:
        if not self.items:
            return []
        targets = {str(path) for path in (limit_paths or set()) if str(path)}
        if not targets:
            return list(range(len(self.items)))
        rows: list[int] = []
        for path in sorted(targets):
            rows.extend(self._rows_by_path.get(path, ()))
        return sorted(set(int(row) for row in rows if 0 <= int(row) < len(self.items)))

    def _emit_changed_rows(self, rows: list[int], roles: list[int]) -> None:
        if not rows:
            return
        normalized_rows = sorted(set(int(row) for row in rows if 0 <= int(row) < len(self.items)))
        if not normalized_rows:
            return
        range_start = normalized_rows[0]
        range_end = normalized_rows[0]
        for row in normalized_rows[1:]:
            if row == (range_end + 1):
                range_end = row
                continue
            top_left = self.index(range_start, 0)
            bottom_right = self.index(range_end, 0)
            if top_left.isValid() and bottom_right.isValid():
                self.dataChanged.emit(top_left, bottom_right, roles)
            range_start = row
            range_end = row
        top_left = self.index(range_start, 0)
        bottom_right = self.index(range_end, 0)
        if top_left.isValid() and bottom_right.isValid():
            self.dataChanged.emit(top_left, bottom_right, roles)

    def append_images(self, image_paths: list[str]) -> None:
        appended = [str(path) for path in image_paths if str(path or "")]
        if not appended:
            return
        start_row = len(self.items)
        end_row = start_row + len(appended) - 1
        self.beginInsertRows(QModelIndex(), start_row, end_row)
        for offset, path in enumerate(appended):
            self.items.append(GalleryItem(image_path=path))
            self._rows_by_path.setdefault(path, []).append(start_row + offset)
        self.endInsertRows()

    def set_image(self, row: int, image: QImage) -> set[int]:
        if not (0 <= row < len(self.items)):
            return set()
        self._images_by_row[row] = QImage(image)
        self._images_by_row.move_to_end(row)
        self.items[row].failed = False
        self.items[row].error = ""
        evicted: set[int] = set()
        while len(self._images_by_row) > self._image_cache_capacity:
            evicted_row, _evicted_image = self._images_by_row.popitem(last=False)
            evicted.add(int(evicted_row))
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.PixmapRole, self.FailedRole, self.ErrorRole])
        self._emit_changed_rows(sorted(evicted - {row}), [self.PixmapRole])
        return evicted

    def set_image_cache_capacity(self, capacity: int) -> set[int]:
        self._image_cache_capacity = max(32, int(capacity))
        evicted: set[int] = set()
        while len(self._images_by_row) > self._image_cache_capacity:
            row, _image = self._images_by_row.popitem(last=False)
            evicted.add(int(row))
        self._emit_changed_rows(sorted(evicted), [self.PixmapRole])
        return evicted

    def cached_rows(self) -> set[int]:
        return set(self._images_by_row)

    def failed_rows(self) -> set[int]:
        return {row for row, item in enumerate(self.items) if item.failed}

    def evict_images_except(self, retained_rows: set[int]) -> set[int]:
        retained = {int(row) for row in retained_rows if 0 <= int(row) < len(self.items)}
        evicted = {row for row in self._images_by_row if row not in retained}
        for row in evicted:
            self._images_by_row.pop(row, None)
        self._emit_changed_rows(sorted(evicted), [self.PixmapRole])
        return evicted

    def set_failed(self, row: int, error: str) -> None:
        if not (0 <= row < len(self.items)):
            return
        self.items[row].failed = True
        self.items[row].error = error
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.FailedRole, self.ErrorRole])

    def clear_pixmaps(self) -> None:
        if not self.items or not self._images_by_row:
            return
        self._images_by_row.clear()
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.items) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [self.PixmapRole])

    def clear_pixmaps_for_paths(self, image_paths: set[str]) -> list[int]:
        targets = {str(path) for path in image_paths if str(path)}
        if not self.items or not targets:
            return []
        changed_rows: list[int] = []
        for row in self._target_rows(targets):
            item = self.items[row]
            self._images_by_row.pop(row, None)
            item.failed = False
            item.error = ""
            changed_rows.append(int(row))
        self._emit_changed_rows(changed_rows, [self.PixmapRole, self.FailedRole, self.ErrorRole])
        return changed_rows

    def set_loading(self, row: int) -> None:
        if not (0 <= row < len(self.items)):
            return
        self.items[row].failed = False
        self.items[row].error = ""
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.FailedRole, self.ErrorRole])

    def set_overlays_by_path(
        self,
        overlay_by_path: dict[str, str] | None = None,
        subtitle_by_path: dict[str, str] | None = None,
        *,
        limit_paths: set[str] | None = None,
    ) -> None:
        overlay_by_path = overlay_by_path or {}
        subtitle_by_path = subtitle_by_path or {}
        if not self.items:
            return
        changed_rows: list[int] = []
        for row in self._target_rows(limit_paths):
            item = self.items[row]
            new_overlay = str(overlay_by_path.get(item.image_path, ""))
            new_subtitle = str(subtitle_by_path.get(item.image_path, ""))
            if item.overlay != new_overlay or item.subtitle != new_subtitle:
                item.overlay = new_overlay
                item.subtitle = new_subtitle
                changed_rows.append(int(row))
        self._emit_changed_rows(changed_rows, [self.OverlayRole, self.SubtitleRole])

    def set_face_boxes_by_path(
        self,
        face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]] | None = None,
        *,
        limit_paths: set[str] | None = None,
    ) -> None:
        face_boxes_by_path = face_boxes_by_path or {}
        if not self.items:
            return
        changed_rows: list[int] = []
        for row in self._target_rows(limit_paths):
            item = self.items[row]
            next_boxes = tuple(
                (
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                )
                for box in face_boxes_by_path.get(item.image_path, ())
                if len(box) == 4
            )
            if item.face_boxes != next_boxes:
                item.face_boxes = next_boxes
                self._preview_face_boxes_by_path.pop(item.image_path, None)
                changed_rows.append(int(row))
        self._emit_changed_rows(changed_rows, [self.FaceBoxesRole])

    def set_face_box_states_by_path(
        self,
        state_by_path: dict[str, str] | None = None,
        *,
        limit_paths: set[str] | None = None,
    ) -> None:
        state_by_path = state_by_path or {}
        if not self.items:
            return
        changed_rows: list[int] = []
        for row in self._target_rows(limit_paths):
            item = self.items[row]
            next_state = "draft" if str(state_by_path.get(item.image_path, "saved")).strip().lower() == "draft" else "saved"
            if item.face_box_state != next_state:
                item.face_box_state = next_state
                changed_rows.append(int(row))
        self._emit_changed_rows(changed_rows, [self.FaceBoxStateRole])

    def checked_paths(self) -> list[str]:
        return [item.image_path for item in self.items if item.checked]

    def set_all_checked(self, checked: bool) -> None:
        if not self.items:
            return
        for item in self.items:
            item.checked = checked
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.items) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.CheckStateRole])

    def set_checked_paths(self, checked_paths: set[str]) -> None:
        if not self.items:
            return
        changed = False
        checked_paths = {str(path) for path in checked_paths if path}
        for item in self.items:
            next_state = item.image_path in checked_paths
            if item.checked != next_state:
                item.checked = next_state
                changed = True
        if not changed:
            return
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.items) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.CheckStateRole])

    def remove_paths(self, removed_paths: set[str]) -> set[int]:
        removed = {str(path) for path in removed_paths if str(path)}
        removed_rows = [row for row, item in enumerate(self.items) if item.image_path in removed]
        if not removed_rows:
            return self.cached_rows()
        old_images = OrderedDict(self._images_by_row)
        removed_row_set = set(removed_rows)
        ranges: list[tuple[int, int]] = []
        start = end = removed_rows[0]
        for row in removed_rows[1:]:
            if row == end + 1:
                end = row
                continue
            ranges.append((start, end))
            start = end = row
        ranges.append((start, end))
        for start, end in reversed(ranges):
            self.beginRemoveRows(QModelIndex(), start, end)
            del self.items[start : end + 1]
            self.endRemoveRows()
        self._rows_by_path = self._build_rows_by_path(self.items)
        self._preview_face_boxes_by_path = {
            path: boxes for path, boxes in self._preview_face_boxes_by_path.items() if path not in removed
        }
        self._images_by_row.clear()
        for old_row, image in old_images.items():
            if old_row in removed_row_set:
                continue
            self._images_by_row[int(old_row) - bisect_left(removed_rows, int(old_row))] = image
        return self.cached_rows()

    def set_preview_face_boxes(self, image_path: str, face_boxes: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...] | None) -> None:
        path = str(image_path or "")
        if not path:
            return
        next_boxes = tuple(
            (
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
            )
            for box in (face_boxes or ())
            if len(box) == 4
        )
        if self._preview_face_boxes_by_path.get(path) == next_boxes:
            return
        self._preview_face_boxes_by_path[path] = next_boxes
        self._emit_changed_rows(self.rows_for_path(path), [self.FaceBoxesRole])

    def clear_preview_face_boxes(self, image_path: str) -> None:
        path = str(image_path or "")
        if not path or path not in self._preview_face_boxes_by_path:
            return
        self._preview_face_boxes_by_path.pop(path, None)
        self._emit_changed_rows(self.rows_for_path(path), [self.FaceBoxesRole])


class GalleryItemDelegate(QStyledItemDelegate):
    copy_requested = pyqtSignal(str)

    def __init__(self, image_size: int, parent=None):
        super().__init__(parent)
        self.image_size = image_size
        self.card_width = image_size + 32
        self.card_height = image_size + 74
        self._copy_icon_size = 18
        self._theme_revision = -1
        self._sync_theme_colors()

    def _sync_theme_colors(self) -> None:
        revision = theme_revision()
        if revision == self._theme_revision:
            return
        self._theme_revision = revision
        # Custom painting bypasses QSS, so use the shared design tokens.
        self._card = QColor(COLORS["surface"])
        self._card_checked = QColor(COLORS["surface_checked"])
        self._card_selected = QColor(COLORS["surface_selected"])
        self._border = QColor(COLORS["border"])
        self._border_checked = QColor(COLORS["success_bright"])
        self._border_selected = QColor(COLORS["info"])
        self._selected_accent = QColor(COLORS["info"])
        self._checked_accent = QColor(COLORS["success_bright"])
        self._text = QColor(COLORS["text"])
        self._muted = QColor(COLORS["text_muted"])
        self._danger = QColor(COLORS["danger_surface"])
        self._danger_border = QColor(COLORS["danger"])
        self._copy_bg = QColor(COLORS["surface_raised"])
        self._copy_border = QColor(COLORS["border"])
        self._copy_glyph = QColor(COLORS["text_muted"])
        self._face_box = QColor(COLORS["success_bright"])
        self._face_box_draft = QColor(COLORS["danger_hover"])
        self._placeholder = QColor(COLORS["surface_sunken"])
        self._danger_text = QColor(COLORS["danger_text"])
        self._overlay = QColor(COLORS["overlay"])

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(self.card_width, self.card_height)

    @staticmethod
    def card_rect_for(option_rect) -> QRect:
        return option_rect.adjusted(4, 4, -4, -4)

    def image_slot_rect_for(self, option_rect) -> QRect:
        card_rect = self.card_rect_for(option_rect)
        return QRect(card_rect.x() + 12, card_rect.y() + 34, self.image_size, self.image_size)

    @staticmethod
    def draw_rect_for_image(image: QImage, slot_rect: QRect) -> QRect | None:
        if image.isNull():
            return None
        if image.width() == slot_rect.width() and image.height() == slot_rect.height():
            return slot_rect
        scaled_size = image.size()
        scaled_size.scale(slot_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        if not scaled_size.isValid() or scaled_size.width() <= 0 or scaled_size.height() <= 0:
            return None
        offset_x = slot_rect.x() + ((slot_rect.width() - scaled_size.width()) // 2)
        offset_y = slot_rect.y() + ((slot_rect.height() - scaled_size.height()) // 2)
        return QRect(offset_x, offset_y, scaled_size.width(), scaled_size.height())

    def _copy_rect(self, option: QStyleOptionViewItem):
        rect = self.card_rect_for(option.rect)
        x = rect.right() - (self._copy_icon_size + 10)
        y = rect.top() + 10
        return rect.__class__(x, y, self._copy_icon_size, self._copy_icon_size)

    @staticmethod
    def _normalized_face_rect(
        box: tuple[float, float, float, float],
        *,
        origin_x: int,
        origin_y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int] | None:
        try:
            x1 = float(box[0])
            y1 = float(box[1])
            x2 = float(box[2])
            y2 = float(box[3])
        except Exception:
            return None
        if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
        left = int(round(origin_x + (min(x1, x2) * width)))
        top = int(round(origin_y + (min(y1, y2) * height)))
        right = int(round(origin_x + (max(x1, x2) * width)))
        bottom = int(round(origin_y + (max(y1, y2) * height)))
        left = max(origin_x, min(origin_x + width, left))
        top = max(origin_y, min(origin_y + height, top))
        right = max(origin_x, min(origin_x + width, right))
        bottom = max(origin_y, min(origin_y + height, bottom))
        rect_width = max(2, right - left)
        rect_height = max(2, bottom - top)
        if rect_width <= 0 or rect_height <= 0:
            return None
        return left, top, rect_width, rect_height

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        self._sync_theme_colors()
        painter.save()
        rect = self.card_rect_for(option.rect)
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        is_checked = index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
        card_color = self._card_selected if is_selected else self._card_checked if is_checked else self._card
        border_color = self._border_selected if is_selected else self._border_checked if is_checked else self._border
        border_width = 3 if is_selected else 2 if is_checked else 1
        painter.fillRect(rect, card_color)
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        painter.setPen(Qt.PenStyle.NoPen)
        if is_selected:
            painter.setBrush(self._selected_accent)
            painter.drawRoundedRect(rect.adjusted(6, 6, -6, -(rect.height() - 12)), 3, 3)
        elif is_checked:
            painter.setBrush(self._checked_accent)
            painter.drawRoundedRect(rect.adjusted(6, 6, -6, -(rect.height() - 12)), 3, 3)

        checkbox_option = QStyleOptionButton()
        checkbox_option.rect = rect.adjusted(8, 8, -rect.width() + 28, -rect.height() + 28)
        checkbox_option.state = QStyle.StateFlag.State_Enabled
        if index.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked:
            checkbox_option.state |= QStyle.StateFlag.State_On
        else:
            checkbox_option.state |= QStyle.StateFlag.State_Off
        QApplication.style().drawControl(QStyle.ControlElement.CE_CheckBox, checkbox_option, painter)

        # Copy icon (top-right).
        copy_rect = self._copy_rect(option)
        painter.setPen(self._copy_border)
        painter.setBrush(self._copy_bg)
        painter.drawRoundedRect(copy_rect.adjusted(-2, -2, 2, 2), 4, 4)
        painter.setPen(self._copy_border)
        # Simple "two sheets" glyph.
        back = copy_rect.adjusted(3, 2, -5, -6)
        front = copy_rect.adjusted(5, 5, -3, -3)
        painter.setBrush(self._card)
        painter.drawRect(back)
        painter.setBrush(self._placeholder)
        painter.drawRect(front)
        painter.setPen(self._copy_glyph)
        font = QFont(option.font)
        font.setPointSize(max(7, font.pointSize() - 2))
        painter.setFont(font)
        painter.drawText(front, Qt.AlignmentFlag.AlignCenter, "C")

        image_rect = self.image_slot_rect_for(option.rect)
        image = index.data(GalleryImageModel.PixmapRole)
        failed = bool(index.data(GalleryImageModel.FailedRole))
        if isinstance(image, QImage) and not image.isNull():
            draw_rect = self.draw_rect_for_image(image, image_rect)
            if draw_rect is not None:
                if draw_rect == image_rect:
                    painter.drawImage(image_rect.topLeft(), image)
                else:
                    painter.drawImage(draw_rect, image)
            face_boxes = index.data(GalleryImageModel.FaceBoxesRole) or ()
            if draw_rect is not None and face_boxes:
                painter.save()
                painter.setClipRect(draw_rect)
                face_box_state = str(index.data(GalleryImageModel.FaceBoxStateRole) or "saved").strip().lower()
                box_color = self._face_box_draft if face_box_state == "draft" else self._face_box
                painter.setPen(QPen(box_color, 2 if is_selected else 1))
                for box in tuple(face_boxes)[:MAX_FACE_BOXES_PER_TILE]:
                    normalized_rect = self._normalized_face_rect(
                        box,
                        origin_x=draw_rect.x(),
                        origin_y=draw_rect.y(),
                        width=draw_rect.width(),
                        height=draw_rect.height(),
                    )
                    if normalized_rect is None:
                        continue
                    painter.drawRect(*normalized_rect)
                painter.restore()
        else:
            painter.fillRect(image_rect, self._placeholder if not failed else self._danger)
            painter.setPen(self._muted if not failed else self._danger_text)
            painter.drawText(
                image_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Loading" if not failed else "Failed",
            )

        overlay = index.data(GalleryImageModel.OverlayRole) or ""
        if overlay:
            overlay = str(overlay)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._overlay)
            pad_x = 6
            pad_y = 3
            font = QFont(option.font)
            font.setPointSize(max(7, font.pointSize() - 2))
            painter.setFont(font)
            metrics = QFontMetrics(font)
            max_w = max(20, image_rect.width() - 8)
            text = metrics.elidedText(overlay, Qt.TextElideMode.ElideRight, max_w)
            text_w = metrics.horizontalAdvance(text)
            text_h = metrics.height()
            bubble_w = min(max_w, text_w + pad_x * 2)
            bubble_h = text_h + pad_y * 2
            bubble_x = image_rect.x() + 4
            bubble_y = image_rect.y() + image_rect.height() - bubble_h - 4
            painter.drawRoundedRect(bubble_x, bubble_y, bubble_w, bubble_h, 6, 6)
            painter.setPen(self._text)
            painter.drawText(
                bubble_x + pad_x,
                bubble_y + pad_y + metrics.ascent(),
                text,
            )
            painter.restore()

        subtitle = str(index.data(GalleryImageModel.SubtitleRole) or "")
        if subtitle:
            subtitle_rect = rect.adjusted(8, rect.height() - 42, -8, -22)
            subtitle_font = QFont(option.font)
            subtitle_font.setPointSize(max(7, subtitle_font.pointSize() - 2))
            painter.setFont(subtitle_font)
            subtitle_metrics = QFontMetrics(subtitle_font)
            painter.setPen(self._muted)
            painter.drawText(
                subtitle_rect,
                Qt.AlignmentFlag.AlignVCenter,
                subtitle_metrics.elidedText(subtitle, Qt.TextElideMode.ElideMiddle, subtitle_rect.width()),
            )
        text_rect = rect.adjusted(8, rect.height() - 24, -8, -6)
        name = index.data(Qt.ItemDataRole.DisplayRole) or ""
        metrics = QFontMetrics(option.font)
        painter.setFont(option.font)
        painter.setPen(self._text)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, text_rect.width()))
        painter.restore()

    def editorEvent(self, event, model, option, index):
        if event.type() not in (QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            return super().editorEvent(event, model, option, index)
        # Handle copy icon click before checkbox/selection.
        copy_rect = self._copy_rect(option)
        if copy_rect.contains(event.position().toPoint()):
            image_path = index.data(GalleryImageModel.PathRole)
            if image_path:
                self.copy_requested.emit(str(image_path))
            return True
        checkbox_rect = option.rect.adjusted(12, 12, -(option.rect.width() - 32), -(option.rect.height() - 32))
        if checkbox_rect.contains(event.position().toPoint()):
            current = index.data(Qt.ItemDataRole.CheckStateRole)
            next_state = Qt.CheckState.Unchecked if current == Qt.CheckState.Checked else Qt.CheckState.Checked
            return model.setData(index, next_state, Qt.ItemDataRole.CheckStateRole)
        return super().editorEvent(event, model, option, index)

