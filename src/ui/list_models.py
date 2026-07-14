from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QAbstractListModel, QModelIndex, Qt
from PyQt6.QtGui import QIcon


@dataclass(frozen=True)
class ListEntry:
    title: str
    tooltip: str = ""
    payload: object | None = None
    icon: QIcon | None = None


class ListEntryModel(QAbstractListModel):
    PayloadRole = Qt.ItemDataRole.UserRole + 1

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

    def clear(self) -> None:
        self.set_items([])
