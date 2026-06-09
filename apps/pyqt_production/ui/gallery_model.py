from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QAbstractListModel, QEvent, QModelIndex, QRect, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QStyle, QStyledItemDelegate, QStyleOptionButton, QStyleOptionViewItem, QApplication


@dataclass
class GalleryItem:
    image_path: str
    pixmap: QPixmap | None = None
    checked: bool = False
    failed: bool = False
    error: str = ""
    overlay: str = ""
    subtitle: str = ""


class GalleryImageModel(QAbstractListModel):
    PathRole = Qt.ItemDataRole.UserRole + 1
    PixmapRole = Qt.ItemDataRole.UserRole + 2
    FailedRole = Qt.ItemDataRole.UserRole + 3
    ErrorRole = Qt.ItemDataRole.UserRole + 4
    OverlayRole = Qt.ItemDataRole.UserRole + 5
    SubtitleRole = Qt.ItemDataRole.UserRole + 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.items: list[GalleryItem] = []

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
        if role == Qt.ItemDataRole.CheckStateRole:
            return Qt.CheckState.Checked if item.checked else Qt.CheckState.Unchecked
        if role == self.PathRole:
            return item.image_path
        if role == self.PixmapRole:
            return item.pixmap
        if role == self.FailedRole:
            return item.failed
        if role == self.ErrorRole:
            return item.error
        if role == self.OverlayRole:
            return item.overlay
        if role == self.SubtitleRole:
            return item.subtitle
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
        self.endResetModel()

    def set_pixmap(self, row: int, pixmap: QPixmap) -> None:
        if not (0 <= row < len(self.items)):
            return
        self.items[row].pixmap = pixmap
        self.items[row].failed = False
        self.items[row].error = ""
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.PixmapRole, self.FailedRole, self.ErrorRole])

    def set_failed(self, row: int, error: str) -> None:
        if not (0 <= row < len(self.items)):
            return
        self.items[row].failed = True
        self.items[row].error = error
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.FailedRole, self.ErrorRole])

    def clear_pixmaps(self) -> None:
        if not self.items:
            return
        for item in self.items:
            item.pixmap = None
        top_left = self.index(0, 0)
        bottom_right = self.index(len(self.items) - 1, 0)
        self.dataChanged.emit(top_left, bottom_right, [self.PixmapRole])

    def set_loading(self, row: int) -> None:
        if not (0 <= row < len(self.items)):
            return
        self.items[row].failed = False
        self.items[row].error = ""
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [self.FailedRole, self.ErrorRole])

    def set_overlays_by_path(self, overlay_by_path: dict[str, str] | None = None, subtitle_by_path: dict[str, str] | None = None) -> None:
        overlay_by_path = overlay_by_path or {}
        subtitle_by_path = subtitle_by_path or {}
        if not self.items:
            return
        changed = False
        for item in self.items:
            new_overlay = str(overlay_by_path.get(item.image_path, ""))
            new_subtitle = str(subtitle_by_path.get(item.image_path, ""))
            if item.overlay != new_overlay or item.subtitle != new_subtitle:
                item.overlay = new_overlay
                item.subtitle = new_subtitle
                changed = True
        if changed:
            top_left = self.index(0, 0)
            bottom_right = self.index(len(self.items) - 1, 0)
            self.dataChanged.emit(top_left, bottom_right, [self.OverlayRole, self.SubtitleRole])

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

    def remove_paths(self, removed_paths: set[str]) -> None:
        self.set_images([item.image_path for item in self.items if item.image_path not in removed_paths])


class GalleryItemDelegate(QStyledItemDelegate):
    def __init__(self, image_size: int, parent=None):
        super().__init__(parent)
        self.image_size = image_size
        self.card_width = image_size + 32
        self.card_height = image_size + 120
        self._copy_icon_size = 18
        self._header_height = 34
        self._footer_height = 72
        self._file_meta_cache: dict[str, tuple[int, int, str]] = {}
        # Matte black theme colors (custom paint bypasses QSS).
        self._bg = QColor("#0B0B0B")
        self._card = QColor("#111111")
        self._card_checked = QColor("#111A14")
        self._card_selected = QColor("#102238")
        self._border = QColor("#2A2A2A")
        self._border_checked = QColor("#22C55E")
        self._border_selected = QColor("#38BDF8")
        self._selected_accent = QColor("#38BDF8")
        self._checked_accent = QColor("#22C55E")
        self._text = QColor("#EDEDED")
        self._muted = QColor("#B0B0B0")
        self._subtle = QColor("#7F8792")
        self._footer = QColor("#0D0D0D")
        self._footer_line = QColor("#232323")
        self._danger = QColor("#3A1313")
        self._danger_border = QColor("#6A2A2A")
        self._copy_bg = QColor("#151515")
        self._copy_border = QColor("#2A2A2A")
        self._copy_glyph = QColor("#D0D0D0")

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        return QSize(self.card_width, self.card_height)

    def _copy_rect(self, option: QStyleOptionViewItem):
        rect = option.rect.adjusted(4, 4, -4, -4)
        x = rect.right() - (self._copy_icon_size + 10)
        y = rect.top() + 10
        return rect.__class__(x, y, self._copy_icon_size, self._copy_icon_size)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        rect = option.rect.adjusted(4, 4, -4, -4)
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
        painter.setBrush(QColor("#1A1A1A"))
        painter.drawRect(back)
        painter.setBrush(QColor("#101010"))
        painter.drawRect(front)
        painter.setPen(self._copy_glyph)
        font = QFont(option.font)
        font.setPointSize(max(7, font.pointSize() - 2))
        painter.setFont(font)
        painter.drawText(front, Qt.AlignmentFlag.AlignCenter, "C")

        pixmap_rect = QRect(
            rect.left() + 12,
            rect.top() + self._header_height,
            max(1, rect.width() - 24),
            self.image_size,
        )
        pixmap = index.data(GalleryImageModel.PixmapRole)
        failed = bool(index.data(GalleryImageModel.FailedRole))
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            scaled = pixmap.scaled(
                pixmap_rect.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = pixmap_rect.x() + (pixmap_rect.width() - scaled.width()) // 2
            y = pixmap_rect.y() + (pixmap_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.fillRect(pixmap_rect, QColor("#0E0E0E" if not failed else self._danger))
            painter.setPen(self._muted if not failed else QColor("#FFB4B4"))
            painter.drawText(
                pixmap_rect,
                Qt.AlignmentFlag.AlignCenter,
                "Loading" if not failed else "Failed",
            )

        overlay = index.data(GalleryImageModel.OverlayRole) or ""
        if overlay:
            overlay = str(overlay)
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(0, 0, 0, 160))
            pad_x = 6
            pad_y = 3
            font = QFont(option.font)
            font.setPointSize(max(7, font.pointSize() - 2))
            painter.setFont(font)
            metrics = QFontMetrics(font)
            max_w = max(20, pixmap_rect.width() - 8)
            text = metrics.elidedText(overlay, Qt.TextElideMode.ElideRight, max_w)
            text_w = metrics.horizontalAdvance(text)
            text_h = metrics.height()
            bubble_w = min(max_w, text_w + pad_x * 2)
            bubble_h = text_h + pad_y * 2
            bubble_x = pixmap_rect.x() + 4
            bubble_y = pixmap_rect.y() + pixmap_rect.height() - bubble_h - 4
            painter.drawRoundedRect(bubble_x, bubble_y, bubble_w, bubble_h, 6, 6)
            painter.setPen(QColor("#FFFFFF"))
            painter.drawText(
                bubble_x + pad_x,
                bubble_y + pad_y + metrics.ascent(),
                text,
            )
            painter.restore()

        self._paint_footer(painter, option, index, rect, pixmap_rect, failed=failed)
        painter.restore()

    def _paint_footer(
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index: QModelIndex,
        rect,
        pixmap_rect,
        *,
        failed: bool,
    ) -> None:
        footer_top = pixmap_rect.bottom() + 7
        footer_rect = QRect(rect.left() + 8, footer_top, max(1, rect.width() - 16), max(1, rect.bottom() - footer_top - 4))
        painter.fillRect(footer_rect, self._footer)
        painter.setPen(QPen(self._footer_line, 1))
        painter.drawLine(footer_rect.left(), footer_rect.top(), footer_rect.right(), footer_rect.top())

        image_path = str(index.data(GalleryImageModel.PathRole) or "")
        name = str(index.data(Qt.ItemDataRole.DisplayRole) or Path(image_path).name or "Untitled image")
        subtitle = str(index.data(GalleryImageModel.SubtitleRole) or "").strip()
        folder = subtitle or self._folder_text(image_path)
        meta = self._file_meta_text(image_path)
        if failed:
            meta = f"{meta} | thumbnail failed"

        name_font = QFont(option.font)
        name_font.setPointSize(max(9, option.font.pointSize()))
        name_font.setWeight(QFont.Weight.DemiBold)
        name_metrics = QFontMetrics(name_font)
        name_rect = QRect(footer_rect.left() + 4, footer_rect.top() + 6, footer_rect.width() - 8, name_metrics.height() + 2)
        painter.setFont(name_font)
        painter.setPen(self._text)
        painter.drawText(
            name_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            name_metrics.elidedText(name, Qt.TextElideMode.ElideMiddle, name_rect.width()),
        )

        small_font = QFont(option.font)
        small_font.setPointSize(max(7, option.font.pointSize() - 2))
        small_metrics = QFontMetrics(small_font)
        painter.setFont(small_font)
        folder_rect = QRect(footer_rect.left() + 4, name_rect.bottom() + 1, footer_rect.width() - 8, small_metrics.height() + 1)
        painter.setPen(self._muted)
        painter.drawText(
            folder_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            small_metrics.elidedText(folder, Qt.TextElideMode.ElideMiddle, folder_rect.width()),
        )
        meta_rect = QRect(footer_rect.left() + 4, folder_rect.bottom() + 1, footer_rect.width() - 8, small_metrics.height() + 1)
        painter.setPen(self._subtle)
        painter.drawText(
            meta_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            small_metrics.elidedText(meta, Qt.TextElideMode.ElideRight, meta_rect.width()),
        )

    @staticmethod
    def _folder_text(image_path: str) -> str:
        if not image_path:
            return "Folder unavailable"
        parent = Path(image_path).parent
        folder_name = parent.name or str(parent)
        return f"in {folder_name}" if folder_name else "Folder unavailable"

    def _file_meta_text(self, image_path: str) -> str:
        path = Path(str(image_path or ""))
        extension = path.suffix.lower().lstrip(".").upper() or "FILE"
        try:
            stat = path.stat()
        except OSError:
            return f"{extension} | file unavailable"
        cached = self._file_meta_cache.get(str(path))
        if cached is not None and cached[0] == int(stat.st_mtime_ns) and cached[1] == int(stat.st_size):
            return cached[2]
        try:
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d")
        except Exception:
            modified = "date unavailable"
        text = f"{extension} | {self._format_bytes(int(stat.st_size))} | {modified}"
        self._file_meta_cache[str(path)] = (int(stat.st_mtime_ns), int(stat.st_size), text)
        if len(self._file_meta_cache) > 4096:
            self._file_meta_cache.clear()
        return text

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        size = float(max(0, int(size_bytes)))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{int(size_bytes)} B"

    def editorEvent(self, event, model, option, index):
        if event.type() not in (QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick):
            return super().editorEvent(event, model, option, index)
        # Handle copy icon click before checkbox/selection.
        copy_rect = self._copy_rect(option)
        if copy_rect.contains(event.position().toPoint()):
            image_path = index.data(GalleryImageModel.PathRole)
            pixmap = index.data(GalleryImageModel.PixmapRole)
            qimage: QImage | None = None
            if isinstance(pixmap, QPixmap) and not pixmap.isNull():
                qimage = pixmap.toImage()
            if qimage is None or qimage.isNull():
                qimage = QImage(str(image_path))
            if qimage is not None and not qimage.isNull():
                QGuiApplication.clipboard().setImage(qimage)
                parent = self.parent()
                if parent is not None and hasattr(parent, "status_label"):
                    try:
                        parent.status_label.setText("Copied image to clipboard.")
                    except Exception:
                        pass
            return True
        checkbox_rect = option.rect.adjusted(12, 12, -(option.rect.width() - 32), -(option.rect.height() - 32))
        if checkbox_rect.contains(event.position().toPoint()):
            current = index.data(Qt.ItemDataRole.CheckStateRole)
            next_state = Qt.CheckState.Unchecked if current == Qt.CheckState.Checked else Qt.CheckState.Checked
            return model.setData(index, next_state, Qt.ItemDataRole.CheckStateRole)
        return super().editorEvent(event, model, option, index)

