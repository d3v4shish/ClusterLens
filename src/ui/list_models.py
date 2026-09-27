from __future__ import annotations

from dataclasses import dataclass, replace

from PyQt6.QtCore import QAbstractListModel, QModelIndex, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter
from PyQt6.QtWidgets import QApplication, QStyle, QStyledItemDelegate, QStyleOptionViewItem

from ui.theme import COLORS


@dataclass(frozen=True)
class ListEntry:
    title: str
    tooltip: str = ""
    payload: object | None = None
    icon: QIcon | None = None
    subtitle: str = ""


class ListEntryModel(QAbstractListModel):
    PayloadRole = Qt.ItemDataRole.UserRole + 1
    SubtitleRole = Qt.ItemDataRole.UserRole + 2

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items: list[ListEntry] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.title
        if role == self.SubtitleRole:
            return item.subtitle
        if role == Qt.ItemDataRole.ToolTipRole:
            return item.tooltip
        if role == Qt.ItemDataRole.DecorationRole:
            return item.icon
        if role == self.PayloadRole:
            return item.payload
        return None

    def set_items(self, items: list[ListEntry]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self.endResetModel()

    def append_items(self, items: list[ListEntry]) -> None:
        if not items:
            return
        start_row = len(self._items)
        end_row = start_row + len(items) - 1
        self.beginInsertRows(QModelIndex(), start_row, end_row)
        self._items.extend(list(items))
        self.endInsertRows()

    def item_at(self, row: int) -> ListEntry | None:
        if 0 <= int(row) < len(self._items):
            return self._items[int(row)]
        return None

    def set_item_icon(self, row: int, icon: QIcon | None) -> None:
        """Update a late-loading icon without resetting selection or scroll."""
        row = int(row)
        if not (0 <= row < len(self._items)):
            return
        self._items[row] = replace(self._items[row], icon=icon)
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.DecorationRole])

    def clear(self) -> None:
        self.set_items([])


class PagedListEntryModel(ListEntryModel):
    """A bounded, filterable list model that exposes data one page at a time."""

    def __init__(self, parent=None, *, page_size: int = 50) -> None:
        super().__init__(parent)
        self.page_size = max(1, int(page_size))
        self._all_items: list[ListEntry] = []
        self._matching_items: list[ListEntry] = []
        self._query = ""
        self._sort_descending = False

    @property
    def total_count(self) -> int:
        return len(self._matching_items)

    @property
    def source_count(self) -> int:
        return len(self._all_items)

    def set_source_items(self, items: list[ListEntry], *, preserve_payload: object | None = None) -> None:
        self.beginResetModel()
        self._all_items = list(items)
        self._rebuild_matches()
        self._items = self._matching_items[: self.page_size]
        self.endResetModel()
        _ = preserve_payload

    def set_filter_text(self, query: str) -> None:
        normalized = str(query or "").strip().casefold()
        if normalized == self._query:
            return
        self.beginResetModel()
        self._query = normalized
        self._rebuild_matches()
        self._items = self._matching_items[: self.page_size]
        self.endResetModel()

    def set_sort_descending(self, descending: bool) -> None:
        descending = bool(descending)
        if descending == self._sort_descending:
            return
        self.beginResetModel()
        self._sort_descending = descending
        self._rebuild_matches()
        self._items = self._matching_items[: self.page_size]
        self.endResetModel()

    def canFetchMore(self, parent: QModelIndex = QModelIndex()) -> bool:
        return not parent.isValid() and len(self._items) < len(self._matching_items)

    def fetchMore(self, parent: QModelIndex = QModelIndex()) -> None:
        if parent.isValid() or not self.canFetchMore(parent):
            return
        start = len(self._items)
        end = min(start + self.page_size, len(self._matching_items))
        self.beginInsertRows(QModelIndex(), start, end - 1)
        self._items.extend(self._matching_items[start:end])
        self.endInsertRows()

    def row_for_payload(self, payload: object) -> int:
        for row, item in enumerate(self._matching_items):
            if item.payload == payload:
                while row >= len(self._items) and self.canFetchMore():
                    self.fetchMore()
                return row
        return -1

    def _rebuild_matches(self) -> None:
        if self._query:
            items = [
                item for item in self._all_items
                if self._query in f"{item.title} {item.tooltip}".casefold()
            ]
        else:
            items = list(self._all_items)
        self._matching_items = sorted(items, key=lambda item: item.title.casefold(), reverse=self._sort_descending)


class SidebarListEntryDelegate(QStyledItemDelegate):
    """Draw dense sidebar rows with separate label and supporting metadata."""

    @staticmethod
    def _text_parts(index) -> tuple[str, str]:
        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "").strip()
        subtitle = str(index.data(ListEntryModel.SubtitleRole) or "").strip()
        if subtitle:
            return title, subtitle
        lines = [line.strip() for line in title.splitlines() if line.strip()]
        if len(lines) > 1:
            return lines[0], " · ".join(lines[1:])
        return title, ""

    def sizeHint(self, option, index):  # type: ignore[override]
        base = super().sizeHint(option, index)
        _title, subtitle = self._text_parts(index)
        minimum_height = 46 if subtitle else 32
        return QSize(base.width(), max(base.height(), minimum_height))

    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        title, subtitle = self._text_parts(index)
        if not subtitle:
            super().paint(painter, option, index)
            return

        style_option = QStyleOptionViewItem(option)
        self.initStyleOption(style_option, index)
        style_option.text = ""
        style_option.icon = QIcon()
        style = style_option.widget.style() if style_option.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, style_option, painter, style_option.widget)

        content = style_option.rect.adjusted(10, 5, -10, -5)
        title_font = QFont(style_option.font)
        title_font.setWeight(QFont.Weight.DemiBold)
        title_metrics = QFontMetrics(title_font)
        subtitle_font = QFont(style_option.font)
        subtitle_font.setPointSize(max(9, subtitle_font.pointSize() - 1))
        subtitle_metrics = QFontMetrics(subtitle_font)
        selected = bool(style_option.state & QStyle.StateFlag.State_Selected)
        title_color = style_option.palette.highlightedText().color() if selected else QColor(COLORS["text"])
        subtitle_color = style_option.palette.highlightedText().color() if selected else QColor(COLORS["text_muted"])
        painter.save()
        painter.setFont(title_font)
        painter.setPen(title_color)
        painter.drawText(
            content.x(),
            content.y() + title_metrics.ascent(),
            title_metrics.elidedText(title, Qt.TextElideMode.ElideRight, max(1, content.width())),
        )
        painter.setFont(subtitle_font)
        painter.setPen(subtitle_color)
        painter.drawText(
            content.x(),
            content.bottom() - subtitle_metrics.descent(),
            subtitle_metrics.elidedText(subtitle, Qt.TextElideMode.ElideRight, max(1, content.width())),
        )
        painter.restore()
