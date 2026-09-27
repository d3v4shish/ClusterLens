from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6 import sip
from PyQt6.QtCore import QAbstractTableModel, QEvent, QItemSelectionModel, QModelIndex, QObject, QPoint, QRect, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QProgressBar,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.selection import SelectionTarget
from app.services.photo_metadata import PhotoMetadataService
from app.services.thumbnails import ThumbnailService
from infra.cancel import raise_if_cancelled
from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.count_copy import showing, thumbnail_progress
from ui.gallery_model import GalleryImageModel
from ui.gallery_pane import GalleryPane
from ui.job_manager import JobManager
from ui.photo_inspector_dialog import PhotoInspectorDialog
from ui.theme import COLORS, get_theme_manager
from ui.work_coordinator import JobSpec, WorkCoordinator


@dataclass(frozen=True)
class GallerySection:
    """A presentation-only group. `section_id` is never shown to the user."""

    section_id: str
    paths: tuple[str, ...]
    kind: str = "photos"
    title: str = ""
    anchor_path: str = ""
    anchor_face_index: int = -1
    children: tuple["GallerySection", ...] = ()


@dataclass(frozen=True)
class _GridRow:
    kind: str
    section_id: str
    paths: tuple[str, ...] = ()
    depth: int = 0


@dataclass(frozen=True)
class _PreparedSectionedGalleryState:
    """Pure, worker-safe hierarchy data ready for one model-reset commit."""

    sections: tuple[GallerySection, ...]
    columns: int
    sections_by_id: dict[str, GallerySection]
    paths_by_section: dict[str, tuple[str, ...]]
    depth_by_section: dict[str, int]
    all_paths: tuple[str, ...]
    rows: tuple[_GridRow, ...]
    path_locations: dict[str, tuple[int, int]]
    header_rows: tuple[int, ...]
    collapsed_section_ids: frozenset[str]


class SectionedGalleryModel(QAbstractTableModel):
    HeaderRole = Qt.ItemDataRole.UserRole + 41
    PathRole = GalleryImageModel.PathRole
    PixmapRole = GalleryImageModel.PixmapRole
    TitleRole = Qt.ItemDataRole.UserRole + 42
    FailedRole = GalleryImageModel.FailedRole

    def __init__(self, parent=None, *, cache_capacity: int = 512) -> None:
        super().__init__(parent)
        self._columns = 4
        self._sections: list[GallerySection] = []
        self._rows: list[_GridRow] = []
        self._header_rows: tuple[int, ...] = ()
        self._path_locations: dict[str, tuple[int, int]] = {}
        self._sections_by_id: dict[str, GallerySection] = {}
        self._paths_by_section: dict[str, tuple[str, ...]] = {}
        self._depth_by_section: dict[str, int] = {}
        self._all_paths: tuple[str, ...] = ()
        self._paths: set[str] = set()
        self._collapsed: set[str] = set()
        self._checked_sections: set[str] = set()
        self._checked_paths: set[str] = set()
        self._images: OrderedDict[str, QImage] = OrderedDict()
        self._labels: dict[str, str] = {}
        self._failed: dict[str, str] = {}
        self._cache_capacity = max(64, int(cache_capacity))

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else self._columns

    def flags(self, index: QModelIndex):  # type: ignore[override]
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        row = self._rows[index.row()]
        if row.kind == "header":
            return Qt.ItemFlag.ItemIsEnabled
        if not self.path_at(index):
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        if row.kind == "header":
            section = self._sections_by_id.get(row.section_id)
            if section is None:
                return None
            if role == self.HeaderRole:
                return section
            if role == Qt.ItemDataRole.AccessibleTextRole:
                state = "collapsed" if section.section_id in self._collapsed else "expanded"
                return f"{self.header_text(section)}, {state}, {len(self.paths_for_section(section.section_id))} photos"
            return None
        path = self.path_at(index)
        if not path:
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return Path(path).name
        if role == self.PathRole:
            return path
        if role == self.PixmapRole:
            image = self._images.get(path)
            if image is not None:
                self._images.move_to_end(path)
            return image
        if role == self.TitleRole:
            return self._labels.get(path, "")
        if role == self.FailedRole:
            return path in self._failed
        if role == Qt.ItemDataRole.AccessibleTextRole:
            title = self._labels.get(path, "")
            selection = "selected" if path in self._checked_paths else "not selected"
            return f"{title or Path(path).name}, {Path(path).name}, {selection}"
        return None

    @staticmethod
    def header_text(section: GallerySection) -> str:
        return section.title or ("Other photos" if section.kind == "other" else "Photos")

    def path_at(self, index: QModelIndex) -> str:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return ""
        row = self._rows[index.row()]
        if row.kind != "photos" or not (0 <= index.column() < len(row.paths)):
            return ""
        return str(row.paths[index.column()] or "")

    def section_at_header(self, index: QModelIndex) -> GallerySection | None:
        if not index.isValid() or not (0 <= index.row() < len(self._rows)):
            return None
        row = self._rows[index.row()]
        return self._sections_by_id.get(row.section_id) if row.kind == "header" else None

    def sections(self) -> list[GallerySection]:
        """Return every visible hierarchy node in presentation order."""

        return list(self._sections_by_id.values())

    def root_sections(self) -> list[GallerySection]:
        """Return only top-level sections for hierarchy-preserving updates."""

        return list(self._sections)

    def all_paths(self) -> list[str]:
        return list(self._all_paths)

    def paths_for_sections(self, section_ids: set[str]) -> list[str]:
        return list(
            dict.fromkeys(
                path
                for section_id in section_ids
                for path in self.paths_for_section(section_id)
            )
        )

    def paths_for_section(self, section_id: str) -> tuple[str, ...]:
        """Return direct and descendant paths for one selectable header."""

        return self._paths_by_section.get(str(section_id), ())

    def section_depth(self, section_id: str) -> int:
        return int(self._depth_by_section.get(str(section_id), 0))

    def checked_section_ids(self) -> set[str]:
        return set(self._checked_sections)

    def checked_paths(self) -> set[str]:
        return set(self._checked_paths)

    def has_image(self, path: str) -> bool:
        return str(path) in self._images

    def image_for_path(self, path: str) -> QImage | None:
        image = self._images.get(str(path))
        return QImage(image) if image is not None else None

    @staticmethod
    def prepare_sections(
        sections: list[GallerySection] | tuple[GallerySection, ...],
        *,
        columns: int,
        collapsed_section_ids: set[str] | frozenset[str] = frozenset(),
        cancel_check: Callable[[], bool] | None = None,
    ) -> _PreparedSectionedGalleryState:
        """Normalize and lay out sections without touching Qt model state.

        This is intentionally a pure operation so a large hierarchy can be
        prepared by an owned worker.  Scope/path validation belongs to the
        caller that builds ``GallerySection``; this function only preserves
        the gallery's stable ID and first-path-wins presentation contract.
        """

        effective_columns = max(1, int(columns))
        unique: list[GallerySection] = []
        seen_paths: set[str] = set()
        seen_ids: set[str] = set()
        checks = 0

        def _check_cancelled() -> None:
            nonlocal checks
            checks += 1
            if checks % 128 == 0:
                raise_if_cancelled(cancel_check)

        def _normalize(section: GallerySection) -> GallerySection | None:
            _check_cancelled()
            section_id = str(section.section_id or "").strip()
            if not section_id or section_id in seen_ids:
                return None
            seen_ids.add(section_id)
            paths: list[str] = []
            for path in section.paths:
                _check_cancelled()
                normalized_path = str(path or "").strip()
                if not normalized_path or normalized_path in seen_paths:
                    continue
                seen_paths.add(normalized_path)
                paths.append(normalized_path)
            children = tuple(child for child in (_normalize(item) for item in section.children) if child is not None)
            if not paths and not children:
                return None
            return GallerySection(
                section_id,
                tuple(paths),
                section.kind,
                section.title,
                section.anchor_path,
                section.anchor_face_index,
                children,
            )

        for section in sections:
            normalized = _normalize(section)
            if normalized is not None:
                unique.append(normalized)
        raise_if_cancelled(cancel_check)
        sections_by_id: dict[str, GallerySection] = {}
        paths_by_section: dict[str, tuple[str, ...]] = {}
        depth_by_section: dict[str, int] = {}

        def _index(section: GallerySection, depth: int) -> tuple[str, ...]:
            _check_cancelled()
            sections_by_id[section.section_id] = section
            depth_by_section[section.section_id] = depth
            paths = list(section.paths)
            for child in section.children:
                paths.extend(_index(child, depth + 1))
            unique_paths = tuple(dict.fromkeys(paths))
            paths_by_section[section.section_id] = unique_paths
            return unique_paths

        all_paths: list[str] = []
        for section in unique:
            all_paths.extend(_index(section, 0))
        normalized_all_paths = tuple(dict.fromkeys(all_paths))
        collapsed = frozenset(str(section_id) for section_id in collapsed_section_ids if str(section_id) in sections_by_id)
        rows: list[_GridRow] = []

        def _append(section: GallerySection, depth: int) -> None:
            _check_cancelled()
            rows.append(_GridRow("header", section.section_id, depth=depth))
            if section.section_id in collapsed:
                return
            for offset in range(0, len(section.paths), effective_columns):
                rows.append(
                    _GridRow(
                        "photos",
                        section.section_id,
                        section.paths[offset : offset + effective_columns],
                        depth,
                    )
                )
            for child in section.children:
                _append(child, depth + 1)

        for section in unique:
            _append(section, 0)
        path_locations = {
            path: (row_index, column)
            for row_index, row in enumerate(rows)
            if row.kind == "photos"
            for column, path in enumerate(row.paths)
        }
        header_rows = tuple(index for index, row in enumerate(rows) if row.kind == "header")
        raise_if_cancelled(cancel_check)
        return _PreparedSectionedGalleryState(
            sections=tuple(unique),
            columns=effective_columns,
            sections_by_id=sections_by_id,
            paths_by_section=paths_by_section,
            depth_by_section=depth_by_section,
            all_paths=normalized_all_paths,
            rows=tuple(rows),
            path_locations=path_locations,
            header_rows=header_rows,
            collapsed_section_ids=collapsed,
        )

    def apply_prepared_sections(self, prepared: _PreparedSectionedGalleryState) -> None:
        """Commit one already-normalized hierarchy in a single Qt reset."""

        valid_paths = set(prepared.all_paths)
        self.beginResetModel()
        self._sections = list(prepared.sections)
        self._columns = max(1, int(prepared.columns))
        self._sections_by_id = dict(prepared.sections_by_id)
        self._paths_by_section = dict(prepared.paths_by_section)
        self._depth_by_section = dict(prepared.depth_by_section)
        self._all_paths = tuple(prepared.all_paths)
        self._paths = valid_paths
        self._collapsed = set(prepared.collapsed_section_ids)
        self._checked_sections.intersection_update(self._sections_by_id)
        self._checked_paths.intersection_update(valid_paths)
        self._images = OrderedDict((path, image) for path, image in self._images.items() if path in valid_paths)
        self._labels = {path: label for path, label in self._labels.items() if path in valid_paths}
        self._failed = {path: error for path, error in self._failed.items() if path in valid_paths}
        self._rows = list(prepared.rows)
        self._path_locations = dict(prepared.path_locations)
        self._header_rows = tuple(prepared.header_rows)
        self.endResetModel()

    def set_sections(self, sections: list[GallerySection], *, reset_scroll_state: bool = True) -> None:
        _ = reset_scroll_state
        self.apply_prepared_sections(
            self.prepare_sections(
                sections,
                columns=self._columns,
                collapsed_section_ids=self._collapsed,
            )
        )

    def set_column_count(self, columns: int) -> None:
        columns = max(1, int(columns))
        if columns == self._columns:
            return
        self.beginResetModel()
        self._columns = columns
        self._rebuild_rows()
        self.endResetModel()

    def _rebuild_rows(self) -> None:
        rows: list[_GridRow] = []

        def _append(section: GallerySection, depth: int) -> None:
            rows.append(_GridRow("header", section.section_id, depth=depth))
            if section.section_id in self._collapsed:
                return
            for offset in range(0, len(section.paths), self._columns):
                rows.append(_GridRow("photos", section.section_id, section.paths[offset : offset + self._columns], depth))
            for child in section.children:
                _append(child, depth + 1)

        for section in self._sections:
            _append(section, 0)
        self._rows = rows
        self._header_rows = tuple(index for index, row in enumerate(rows) if row.kind == "header")
        self._path_locations = {
            path: (row_index, column)
            for row_index, row in enumerate(rows)
            if row.kind == "photos"
            for column, path in enumerate(row.paths)
        }

    def toggle_collapsed(self, section_id: str) -> None:
        if section_id not in self._sections_by_id:
            return
        self.beginResetModel()
        if section_id in self._collapsed:
            self._collapsed.remove(section_id)
        else:
            self._collapsed.add(section_id)
        self._rebuild_rows()
        self.endResetModel()

    def set_all_collapsed(self, collapsed: bool) -> bool:
        target = set(self._sections_by_id) if collapsed else set()
        if target == self._collapsed:
            return False
        self.beginResetModel()
        self._collapsed = target
        self._rebuild_rows()
        self.endResetModel()
        return True

    def set_collapsed_sections(self, section_ids: set[str]) -> bool:
        """Set a deliberate initial hierarchy state without rebuilding tiles."""

        target = {str(section_id) for section_id in section_ids if str(section_id) in self._sections_by_id}
        if target == self._collapsed:
            return False
        self.beginResetModel()
        self._collapsed = target
        self._rebuild_rows()
        self.endResetModel()
        return True

    def toggle_checked_section(self, section_id: str) -> None:
        if section_id not in self._sections_by_id:
            return
        if section_id in self._checked_sections:
            self._checked_sections.remove(section_id)
        else:
            self._checked_sections.add(section_id)
        for row_index, row in enumerate(self._rows):
            if row.kind == "header" and row.section_id == section_id:
                first = self.index(row_index, 0)
                last = self.index(row_index, max(0, self._columns - 1))
                self.dataChanged.emit(first, last, [self.HeaderRole])
                break

    def set_checked_paths(self, paths: set[str]) -> None:
        self._checked_paths = {path for path in paths if path in self._paths}

    def set_image(self, path: str, image: QImage) -> None:
        if not path or path not in self._paths:
            return
        self._images[path] = QImage(image)
        self._images.move_to_end(path)
        self._failed.pop(path, None)
        while len(self._images) > self._cache_capacity:
            self._images.popitem(last=False)
        self._emit_path_changed(path, [self.PixmapRole, self.FailedRole])

    def set_label(self, path: str, label: str) -> None:
        if not path or path not in self._paths:
            return
        if self._labels.get(path, "") == label:
            return
        self._labels[path] = str(label or "")
        self._emit_path_changed(path, [self.TitleRole])

    def set_failed(self, path: str, error: str) -> None:
        if not path:
            return
        self._failed[path] = str(error or "Could not load image")
        self._emit_path_changed(path, [self.FailedRole])

    def _emit_path_changed(self, path: str, roles: list[int]) -> None:
        location = self._path_locations.get(str(path))
        if location is None:
            return
        row_index, column = location
        index = self.index(row_index, column)
        self.dataChanged.emit(index, index, roles)


class _SectionedPhotoDelegate(QStyledItemDelegate):
    def __init__(self, tile_size: int, parent=None) -> None:
        super().__init__(parent)
        self.tile_size = max(96, int(tile_size))

    def sizeHint(self, option, index):  # type: ignore[override]
        if index.data(SectionedGalleryModel.HeaderRole) is not None:
            return QSize(1, 42)
        return QSize(self.tile_size + 18, self.tile_size + 58)

    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        section = index.data(SectionedGalleryModel.HeaderRole)
        if section is not None:
            self._paint_header(painter, option.rect, section, index.model())
            return
        path = str(index.data(SectionedGalleryModel.PathRole) or "")
        if not path:
            return
        self._paint_photo(painter, option, index, path)

    def _paint_header(self, painter: QPainter, rect: QRect, section: GallerySection, model) -> None:
        selected = section.section_id in model.checked_section_ids()
        collapsed = section.section_id in model._collapsed  # noqa: SLF001 - model owns the state
        paths = model.paths_for_section(section.section_id)
        depth = model.section_depth(section.section_id)
        painter.save()
        background = QColor(COLORS["surface_selected"] if selected else COLORS["surface_sunken"])
        border = QColor(COLORS["info"] if selected else COLORS["border"])
        painter.fillRect(rect.adjusted(0, 2, 0, -2), background)
        painter.setPen(QPen(border))
        painter.drawLine(rect.left(), rect.bottom() - 2, rect.right(), rect.bottom() - 2)
        painter.setPen(QColor(COLORS["text"]))
        preview_size = 26
        preview_gap = 4
        preview_right = rect.right() - 100
        preview_left = preview_right - (preview_size * 3) - (preview_gap * 2)
        show_previews = preview_left > rect.left() + 110
        title_right = preview_left - 8 if show_previews else rect.right() - 105
        title_left = rect.left() + 12 + depth * 18
        title_rect = QRect(title_left, rect.top(), max(1, title_right - title_left), rect.height())
        painter.drawText(
            title_rect,
            Qt.AlignmentFlag.AlignVCenter,
            f"{'▸' if collapsed else '▾'}  {model.header_text(section)}  ·  {len(paths)} photos",
        )
        for offset, path in enumerate(paths[:3] if show_previews else ()):
            preview_rect = QRect(preview_left + offset * (preview_size + preview_gap), rect.center().y() - preview_size // 2, preview_size, preview_size)
            painter.fillRect(preview_rect, QColor(COLORS["surface"]))
            image = model.image_for_path(path)
            if image is not None and not image.isNull():
                scaled = image.scaled(
                    preview_rect.size(),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
                source_x = max(0, (scaled.width() - preview_rect.width()) // 2)
                source_y = max(0, (scaled.height() - preview_rect.height()) // 2)
                painter.drawImage(preview_rect, scaled, QRect(source_x, source_y, preview_rect.width(), preview_rect.height()))
            painter.setPen(QPen(border))
            painter.drawRect(preview_rect.adjusted(0, 0, -1, -1))
        check = QRect(rect.right() - 72, rect.center().y() - 8, 16, 16)
        painter.setPen(QPen(border))
        painter.drawRect(check)
        if selected:
            painter.fillRect(check.adjusted(3, 3, -3, -3), QColor(COLORS["info"]))
        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawText(QRect(rect.right() - 40, rect.top(), 32, rect.height()), Qt.AlignmentFlag.AlignCenter, "⋯")
        painter.restore()

    def _paint_photo(self, painter: QPainter, option, index, path: str) -> None:
        rect = option.rect.adjusted(7, 5, -7, -5)
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        painter.fillRect(rect, QColor(COLORS["surface_selected"] if selected else COLORS["surface"]))
        painter.setPen(QPen(QColor(COLORS["info"] if selected else COLORS["border"])))
        painter.drawRect(rect.adjusted(0, 0, -1, -1))
        image_rect = QRect(rect.left() + 5, rect.top() + 5, max(1, rect.width() - 10), self.tile_size)
        image = index.data(SectionedGalleryModel.PixmapRole)
        if isinstance(image, QImage) and not image.isNull():
            scaled = image.scaled(image_rect.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawImage(image_rect.left() + max(0, (image_rect.width() - scaled.width()) // 2), image_rect.top() + max(0, (image_rect.height() - scaled.height()) // 2), scaled)
        else:
            painter.fillRect(image_rect, QColor(COLORS["surface_sunken"]))
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(image_rect, Qt.AlignmentFlag.AlignCenter, "Loading" if not index.data(SectionedGalleryModel.FailedRole) else "Unavailable")
        title = str(index.data(SectionedGalleryModel.TitleRole) or "").strip()
        filename = Path(path).name
        painter.setPen(QColor(COLORS["text"]))
        name_rect = QRect(rect.left() + 6, image_rect.bottom() + 4, max(1, rect.width() - 12), 18)
        if title:
            painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, painter.fontMetrics().elidedText(title, Qt.TextElideMode.ElideRight, name_rect.width()))
            painter.setPen(QColor(COLORS["text_muted"]))
            file_rect = QRect(rect.left() + 6, name_rect.bottom() + 1, max(1, rect.width() - 12), 16)
            painter.drawText(file_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, painter.fontMetrics().elidedText(filename, Qt.TextElideMode.ElideRight, file_rect.width()))
        else:
            painter.drawText(name_rect.adjusted(0, 4, 0, 4), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, painter.fontMetrics().elidedText(filename, Qt.TextElideMode.ElideRight, name_rect.width()))
        painter.restore()


class _SectionedTable(QTableView):
    header_action_requested = pyqtSignal(str, QPoint)
    header_clicked = pyqtSignal(str)
    header_checked = pyqtSignal(str)
    resized = pyqtSignal()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.resized.emit()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        index = self.indexAt(event.position().toPoint())
        section = self.model().section_at_header(index) if index.isValid() else None
        if section is not None:
            rect = self.visualRect(index)
            x = event.position().x()
            if x >= rect.right() - 46:
                self.header_action_requested.emit(section.section_id, self.viewport().mapToGlobal(event.position().toPoint()))
            elif x >= rect.right() - 88:
                self.header_checked.emit(section.section_id)
            else:
                self.header_clicked.emit(section.section_id)
            event.accept()
            return
        super().mousePressEvent(event)


class _GroupHoverPreviewPopup(QFrame):
    """A non-interactive contact-sheet preview for a photo group header."""

    IMAGE_SIZE = QSize(216, 216)

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setObjectName("groupHoverPreviewPopup")
        self.setStyleSheet(
            f"""
            QFrame#groupHoverPreviewPopup {{
                background: {COLORS["surface_raised"]};
                border: 1px solid {COLORS["border_strong"]};
                border-radius: 8px;
            }}
            QLabel {{ color: {COLORS["text"]}; }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        self.title = QLabel("")
        title_font = self.title.font()
        title_font.setBold(True)
        self.title.setFont(title_font)
        layout.addWidget(self.title)
        self.image = QLabel("Loading preview...")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setFixedSize(self.IMAGE_SIZE)
        self.image.setStyleSheet(f"background: {COLORS['surface_sunken']}; border: 1px solid {COLORS['border']};")
        layout.addWidget(self.image, alignment=Qt.AlignmentFlag.AlignCenter)
        self.note = QLabel("")
        self.note.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(self.note)
        self.setFixedWidth(240)
        manager = get_theme_manager()
        if manager is not None:
            manager.theme_changed.connect(self._apply_theme)

    def _apply_theme(self, *_args) -> None:
        self.setStyleSheet(
            f"QFrame#groupHoverPreviewPopup {{ background: {COLORS['surface_raised']}; "
            f"border: 1px solid {COLORS['border_strong']}; border-radius: 8px; }} "
            f"QLabel {{ color: {COLORS['text']}; }}"
        )
        self.image.setStyleSheet(
            f"background: {COLORS['surface_sunken']}; border: 1px solid {COLORS['border']};"
        )
        self.note.setStyleSheet(f"color: {COLORS['text_muted']};")

    def set_group(self, section: GallerySection, image: QImage, *, paths: tuple[str, ...], loading: bool) -> None:
        self.title.setText(f"{section.title or 'Photos'} · {len(paths)} photos")
        self.note.setText(f"+{max(0, len(paths) - 9)} more" if len(paths) > 9 else "")
        if image.isNull():
            self.image.setPixmap(QPixmap())
            self.image.setText("Loading preview..." if loading else "Preview unavailable")
            return
        self.image.setText("")
        self.image.setPixmap(QPixmap.fromImage(image))


class SectionedGallery(QWidget):
    ASYNC_PREPARE_PATH_THRESHOLD = 2_000
    organize_requested = pyqtSignal()
    analyze_requested = pyqtSignal(list)
    review_faces_requested = pyqtSignal(list)
    return_to_folder_requested = pyqtSignal()
    return_to_source_requested = pyqtSignal(str)
    paths_removed = pyqtSignal(list)
    paths_renamed = pyqtSignal(list)
    metadata_changed = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tile_size = 190
        self._thumbnail_service = ThumbnailService()
        # Keep decode work owned by this view so its signals cannot target a
        # deleted gallery during shutdown.
        self._job_pool = QThreadPool(self)
        self._job_pool.setMaxThreadCount(2)
        self._shutting_down = False
        self._metadata_service = PhotoMetadataService()
        self._loading_paths: set[str] = set()
        self._label_paths: set[str] = set()
        self._job_manager: JobManager | None = None
        self._work_coordinator: WorkCoordinator | None = None
        self._job_origin = "Photos"
        self._viewport_job_id: int | None = None
        self._viewport_job_generation = -1
        self._viewport_tasks: set[tuple[str, str]] = set()
        self._viewport_completed_tasks: set[tuple[str, str]] = set()
        self._generation = 0
        self._section_prepare_generation = 0
        self._section_prepare_job = None
        self._section_prepare_thread = None
        self._section_prepare_coordinated_job_ids: dict[AsyncJob, int] = {}
        self._pending_section_column_count: int | None = None
        self._retained_section_prepare_refs: list[tuple[object, object]] = []
        self._viewport_paths: set[str] = set()
        self._table_geometry: tuple[int, int, int, int] | None = None
        self._read_only = False
        self.face_service_provider = None
        # A routed gallery can retain the exact source context that opened a
        # photo without requiring the source workspace to stay visible.
        self.inspector_context_provider = None
        # Optional production-shell hook: (image_path, on_ready, on_failed).
        self.face_edit_request_handler = None
        self.face_edit_saved_callback = None
        self._model = SectionedGalleryModel(self)
        self._delegate = _SectionedPhotoDelegate(self._tile_size, self)
        self._hover_popup = _GroupHoverPreviewPopup(self)
        self._hover_section_id = ""
        self._hover_index = QModelIndex()
        self._actions = GalleryPane(self)
        self._actions.hide()
        self._actions.set_action_target_provider(self._current_action_target)
        self._actions.paths_removed.connect(self._on_paths_removed)
        self._actions.paths_renamed.connect(self._on_paths_renamed)
        self._actions.metadata_changed.connect(self.metadata_changed)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        toolbar = QHBoxLayout()
        self.route_label = QLabel("")
        self.route_label.setProperty("role", "section")
        self.route_label.setAccessibleName("Current photo set")
        self.route_label.hide()
        self.status_label = QLabel("Choose a folder to show its photos.")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setAccessibleName("Visible photo loading progress")
        self.progress_bar.hide()
        self.organize_button = QPushButton("Organize")
        self.organize_button.setToolTip("Arrange this folder using the current Organize preset.")
        self.organize_button.clicked.connect(self.organize_requested.emit)
        self.analyze_button = QPushButton("Organize this photo set")
        self.analyze_button.setToolTip("Use this explicit photo set as the input to Organize.")
        self.analyze_button.setAccessibleName("Organize current photo set")
        self.analyze_button.clicked.connect(lambda: self.analyze_requested.emit(self.selected_or_all_paths()))
        self.review_faces_button = QPushButton("Review people")
        self.review_faces_button.setToolTip("Open this explicit photo set in People. Detection starts only after you choose Detect faces.")
        self.review_faces_button.setAccessibleName("Review people in current photo set")
        self.review_faces_button.clicked.connect(lambda: self.review_faces_requested.emit(self.selected_or_all_paths()))
        self.back_to_folder_button = QPushButton("Back to Folder Photos")
        self.back_to_folder_button.setToolTip("Return to all photos in the selected folder.")
        self.back_to_folder_button.setAccessibleName("Return to folder photos")
        self.back_to_folder_button.clicked.connect(self.return_to_folder_requested.emit)
        self.return_to_source_button = QPushButton("Return to Source")
        self.return_to_source_button.setToolTip("Return to the workspace that opened this photo set.")
        self.return_to_source_button.setAccessibleName("Return to source workspace")
        self.return_to_source_button.clicked.connect(self._emit_return_to_source)
        self.expand_all_button = QPushButton("Expand all")
        self.expand_all_button.setToolTip("Show every photo group.")
        self.collapse_all_button = QPushButton("Collapse all")
        self.collapse_all_button.setToolTip("Hide the photos in every group.")
        self.expand_all_button.clicked.connect(self.expand_all_sections)
        self.collapse_all_button.clicked.connect(self.collapse_all_sections)
        toolbar.addWidget(self.route_label)
        toolbar.addWidget(self.status_label, stretch=1)
        toolbar.addWidget(self.progress_bar)
        toolbar.addWidget(self.analyze_button)
        toolbar.addWidget(self.review_faces_button)
        toolbar.addWidget(self.back_to_folder_button)
        toolbar.addWidget(self.return_to_source_button)
        toolbar.addWidget(self.expand_all_button)
        toolbar.addWidget(self.collapse_all_button)
        toolbar.addWidget(self.organize_button)
        layout.addLayout(toolbar)

        self.group_bar = QWidget(self)
        group_layout = QHBoxLayout(self.group_bar)
        group_layout.setContentsMargins(8, 4, 8, 4)
        self.group_summary = QLabel("")
        self.group_inspect = QPushButton("Inspect")
        self.group_reveal = QPushButton("Reveal")
        self.group_tag = QPushButton("Tag")
        self.group_actions = QPushButton("Actions")
        group_layout.addWidget(self.group_summary, stretch=1)
        for button in (self.group_inspect, self.group_reveal, self.group_tag, self.group_actions):
            group_layout.addWidget(button)
        self.group_inspect.clicked.connect(self._inspect_current_target)
        self.group_reveal.clicked.connect(self._actions.slotOpenSelectedFolders)
        self.group_tag.clicked.connect(self._actions.slotEditTags)
        menu = QMenu(self.group_actions)
        menu.addAction("Copy", self._actions.slotCopySelected)
        menu.addAction("Move", self._actions.slotMoveSelected)
        menu.addAction("Move to ClusterLens Trash", self._actions.slotDeleteSelect)
        self.group_actions.setMenu(menu)
        self.group_bar.hide()
        layout.addWidget(self.group_bar)

        self.set_photo_set_route()

        self.table = _SectionedTable(self)
        self.table.setModel(self._model)
        self.table.setItemDelegate(self._delegate)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        self.table.setShowGrid(False)
        self.table.setWordWrap(False)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().hide()
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.table.setAccessibleName("Photo-first clustered gallery")
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_photo_menu)
        self.table.doubleClicked.connect(self._open_photo)
        self.table.activated.connect(self._open_photo)
        self.table.header_clicked.connect(self._toggle_collapsed_section)
        self.table.header_checked.connect(self._toggle_checked_section)
        self.table.header_action_requested.connect(self._show_section_menu)
        self.table.resized.connect(self._apply_table_geometry)
        self.table.viewport().setMouseTracking(True)
        self.table.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.table.viewport().installEventFilter(self)
        self.table.verticalScrollBar().valueChanged.connect(self._on_table_scrolled)
        self.table.selectionModel().selectionChanged.connect(lambda *_args: self._refresh_group_bar())
        layout.addWidget(self.table, stretch=1)

    def set_job_manager(
        self,
        job_manager: JobManager | None,
        *,
        origin: str = "Photos",
        work_coordinator: WorkCoordinator | None = None,
    ) -> None:
        """Route visible gallery work to the shared Jobs surface."""
        self._job_manager = job_manager
        self._work_coordinator = work_coordinator
        self._job_origin = str(origin or "Photos")
        self._actions.configure_jobs(job_manager, work_coordinator, origin=self._job_origin)

    def set_photo_set_route(
        self,
        *,
        title: str = "",
        source: str = "",
        can_return: bool = False,
        can_review_faces: bool = False,
        can_analyze: bool = False,
    ) -> None:
        """Render the session-only route that owns the visible photo set."""

        normalized_title = str(title or "").strip()
        self.route_label.setText(normalized_title)
        self.route_label.setToolTip(str(source or normalized_title))
        self.route_label.setVisible(bool(normalized_title))
        self.back_to_folder_button.setVisible(bool(normalized_title))
        self.return_to_source_button.setVisible(bool(normalized_title) and bool(can_return))
        self.return_to_source_button.setProperty("route_source", str(source or ""))
        self.review_faces_button.setVisible(bool(can_review_faces))
        self.analyze_button.setVisible(bool(can_analyze))

    def selected_or_all_paths(self) -> list[str]:
        """Use an explicit selection when present, otherwise the visible route."""

        target = self._current_action_target()
        if target is not None and target.paths:
            return list(target.paths)
        return self._model.all_paths()

    def _emit_return_to_source(self) -> None:
        source = str(self.return_to_source_button.property("route_source") or "")
        if source:
            self.return_to_source_requested.emit(source)

    def set_loading_state(self, text: str) -> None:
        """Show folder discovery before thumbnail tasks can begin."""
        self.status_label.setText(str(text or "Loading photos…"))
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

    @staticmethod
    def _section_path_count(sections: list[GallerySection] | tuple[GallerySection, ...]) -> int:
        """Count declared paths without traversing or normalizing each path."""

        total = 0
        pending = list(sections)
        while pending:
            section = pending.pop()
            total += len(section.paths)
            pending.extend(section.children)
        return total

    def _selected_paths_and_anchor(self) -> tuple[set[str], str]:
        selection_model = self.table.selectionModel()
        selected_paths = {
            self._model.path_at(index)
            for index in (selection_model.selectedIndexes() if selection_model is not None else ())
            if self._model.path_at(index)
        }
        anchor_index = self.table.indexAt(QPoint(0, 0))
        anchor_path = self._model.path_at(anchor_index)
        return selected_paths, anchor_path

    def _restore_paths_and_anchor(self, selected_paths: set[str], anchor_path: str) -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.table):
            return
        selection_model = self.table.selectionModel()
        if selection_model is not None:
            selection_model.clearSelection()
            for path in selected_paths:
                location = self._model._path_locations.get(path)  # noqa: SLF001 - model owns stable row identity
                if location is None:
                    continue
                index = self._model.index(*location)
                selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
        location = self._model._path_locations.get(str(anchor_path))  # noqa: SLF001 - model owns stable row identity
        if location is not None:
            self.table.scrollTo(self._model.index(*location), QAbstractItemView.ScrollHint.PositionAtTop)

    def _cancel_section_preparation(self) -> None:
        job = self._section_prepare_job
        if job is not None:
            coordinated_job_id = self._section_prepare_coordinated_job_ids.get(job)
            if self._work_coordinator is not None and coordinated_job_id is not None:
                self._work_coordinator.cancel(coordinated_job_id)
            else:
                try:
                    job.cancel()
                except Exception:
                    pass
        self._section_prepare_job = None
        self._section_prepare_thread = None
        self._pending_section_column_count = None

    def _release_section_preparation_thread(self, thread: object) -> None:
        self._retained_section_prepare_refs = [
            (job, retained_thread)
            for job, retained_thread in self._retained_section_prepare_refs
            if retained_thread is not thread
        ]

    def _publish_prepared_sections(
        self,
        prepared: _PreparedSectionedGalleryState,
        *,
        status: str,
        selected_paths: set[str],
        anchor_path: str,
    ) -> None:
        self._model.apply_prepared_sections(prepared)
        self.progress_bar.hide()
        # Section resets can change which rows span the whole table even when
        # the viewport size and number of columns are unchanged.
        self._table_geometry = None
        self._apply_table_geometry()
        self.status_label.setText(status or f"{showing(len(self._model.all_paths()), 'photo')}.")
        self._refresh_group_bar()
        QTimer.singleShot(0, lambda: self._restore_paths_and_anchor(selected_paths, anchor_path))
        QTimer.singleShot(0, self._queue_visible_loads)

    def _prepare_sections_in_background(
        self,
        sections: tuple[GallerySection, ...],
        *,
        columns: int,
        collapsed_section_ids: set[str],
        status: str,
        selected_paths: set[str],
        anchor_path: str,
    ) -> None:
        self._cancel_section_preparation()
        self._section_prepare_generation += 1
        generation = self._section_prepare_generation
        columns = max(1, int(columns))
        self._pending_section_column_count = columns
        self.status_label.setText(f"Preparing {self._section_path_count(sections)} photo rows…")
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()

        def _run(_progress, cancel_check):
            return SectionedGalleryModel.prepare_sections(
                sections,
                columns=columns,
                collapsed_section_ids=collapsed_section_ids,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        self._section_prepare_job = job
        thread_holder: dict[str, object | None] = {"thread": None}

        def _clear() -> None:
            thread = thread_holder["thread"]
            self._section_prepare_coordinated_job_ids.pop(job, None)
            if self._section_prepare_job is job:
                self._section_prepare_job = None
                self._pending_section_column_count = None
            if self._section_prepare_thread is thread:
                self._section_prepare_thread = None

        def _completed(prepared: object) -> None:
            if self._shutting_down or generation != self._section_prepare_generation:
                _clear()
                return
            if not isinstance(prepared, _PreparedSectionedGalleryState):
                self.status_label.setText("Photo layout preparation returned an invalid result.")
                self.progress_bar.hide()
                _clear()
                return
            self.progress_bar.hide()
            self._publish_prepared_sections(
                prepared,
                status=status,
                selected_paths=selected_paths,
                anchor_path=anchor_path,
            )
            _clear()

        def _failed(message: str) -> None:
            if not self._shutting_down and generation == self._section_prepare_generation:
                self.status_label.setText(f"Photo layout preparation stopped: {str(message or 'unknown error')}")
                self.progress_bar.hide()
            _clear()

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_clear)

        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            thread_holder["thread"] = thread
            self._section_prepare_thread = thread
            self._retained_section_prepare_refs.append((job, thread))
            thread.finished.connect(lambda thread=thread: self._release_section_preparation_thread(thread))

        if self._work_coordinator is None:
            _launch()
            return
        coordinated_job_id = self._work_coordinator.submit_async_job(
            JobSpec(
                "Preparing photo layout",
                origin=self._job_origin,
                foreground=False,
                cpu_slots=1,
            ),
            job,
            _launch,
        )
        state = self._work_coordinator.job_manager.get(coordinated_job_id)
        if state is not None and state.status in {"queued", "running", "cancelling"}:
            self._section_prepare_coordinated_job_ids[job] = coordinated_job_id

    def set_sections(self, sections: list[GallerySection], *, status: str = "") -> None:
        if self._shutting_down:
            return
        self._hide_hover_preview()
        self._finish_viewport_job(status="cancelled")
        self._generation += 1
        self._loading_paths.clear()
        self._label_paths.clear()
        snapshot = tuple(sections)
        selected_paths, anchor_path = self._selected_paths_and_anchor()
        collapsed_section_ids = set(self._model._collapsed)  # noqa: SLF001 - model owns collapse identity
        if self._section_path_count(snapshot) >= self.ASYNC_PREPARE_PATH_THRESHOLD:
            self._prepare_sections_in_background(
                snapshot,
                columns=self._model.columnCount(),
                collapsed_section_ids=collapsed_section_ids,
                status=status,
                selected_paths=selected_paths,
                anchor_path=anchor_path,
            )
            return
        self._cancel_section_preparation()
        self._section_prepare_generation += 1
        prepared = SectionedGalleryModel.prepare_sections(
            snapshot,
            columns=self._model.columnCount(),
            collapsed_section_ids=collapsed_section_ids,
        )
        self._publish_prepared_sections(
            prepared,
            status=status,
            selected_paths=selected_paths,
            anchor_path=anchor_path,
        )

    def set_sections_with_collapsed(
        self,
        sections: list[GallerySection],
        *,
        collapsed_section_ids: set[str],
        status: str = "",
    ) -> None:
        """Publish sections and an explicit initial expansion state together."""

        if self._shutting_down:
            return
        self._hide_hover_preview()
        self._finish_viewport_job(status="cancelled")
        self._generation += 1
        self._loading_paths.clear()
        self._label_paths.clear()
        snapshot = tuple(sections)
        selected_paths, anchor_path = self._selected_paths_and_anchor()
        if self._section_path_count(snapshot) >= self.ASYNC_PREPARE_PATH_THRESHOLD:
            self._prepare_sections_in_background(
                snapshot,
                columns=self._model.columnCount(),
                collapsed_section_ids=set(collapsed_section_ids),
                status=status,
                selected_paths=selected_paths,
                anchor_path=anchor_path,
            )
            return
        self._cancel_section_preparation()
        self._section_prepare_generation += 1
        prepared = SectionedGalleryModel.prepare_sections(
            snapshot,
            columns=self._model.columnCount(),
            collapsed_section_ids=collapsed_section_ids,
        )
        self._publish_prepared_sections(
            prepared,
            status=status,
            selected_paths=selected_paths,
            anchor_path=anchor_path,
        )

    def set_empty_state(self, text: str, *, can_organize: bool = False) -> None:
        if self._shutting_down:
            return
        self.set_sections([])
        self.status_label.setText(text)
        self.progress_bar.hide()
        self.organize_button.setEnabled(bool(can_organize))

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only = bool(enabled)
        self._actions.set_read_only_mode(enabled)
        self.group_tag.setEnabled(not enabled)

    def set_view_preferences(self, *, thumbnail_size: int, cache_size: int, worker_count: int | None = None) -> None:
        self._tile_size = max(96, int(thumbnail_size))
        self._delegate = _SectionedPhotoDelegate(self._tile_size, self)
        self.table.setItemDelegate(self._delegate)
        self._thumbnail_service.set_qimage_cache_size(max(64, int(cache_size)))
        if worker_count is not None:
            self._job_pool.setMaxThreadCount(max(1, min(4, int(worker_count))))
        self._apply_table_geometry()

    def set_embedded_timeline_mode(self, enabled: bool) -> None:
        """Keep Library Timeline focused while retaining selected-photo actions."""

        if not enabled:
            return
        self.route_label.hide()
        self.organize_button.hide()
        self.analyze_button.hide()
        self.review_faces_button.hide()
        self.back_to_folder_button.hide()
        self.return_to_source_button.hide()
        self.table.setAccessibleName("Library timeline photos")

    def set_embedded_review_mode(self, enabled: bool) -> None:
        """Trim non-review controls when this is a small, virtual preview."""

        if not enabled:
            return
        self.route_label.hide()
        self.organize_button.hide()
        self.analyze_button.hide()
        self.review_faces_button.hide()
        self.back_to_folder_button.hide()
        self.return_to_source_button.hide()
        self.expand_all_button.hide()
        self.collapse_all_button.hide()
        self.group_bar.hide()
        self.table.setAccessibleName("Selected duplicate group photo preview")

    def configure_photo_tools(
        self,
        *,
        metadata_service=None,
        image_tag_service=None,
        face_service_provider=None,
        face_edit_request_handler=None,
        face_edit_saved_callback=None,
        inspector_context_provider=None,
    ) -> None:
        """Use the production Gallery services for timeline photo actions."""

        self._metadata_service = metadata_service or self._metadata_service
        self._actions.metadata_service = metadata_service
        self._actions.image_tag_service = image_tag_service
        self.face_service_provider = face_service_provider
        self.face_edit_request_handler = face_edit_request_handler
        self.face_edit_saved_callback = face_edit_saved_callback
        self.inspector_context_provider = inspector_context_provider

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._apply_table_geometry()

    def _apply_table_geometry(self) -> None:
        if not hasattr(self, "table"):
            return
        columns = max(1, self.table.viewport().width() // max(120, self._tile_size + 18))
        width = max(80, self.table.viewport().width() // columns)
        if columns != self._model.columnCount():
            if self._section_path_count(tuple(self._model.root_sections())) >= self.ASYNC_PREPARE_PATH_THRESHOLD:
                if self._pending_section_column_count != columns:
                    selected_paths, anchor_path = self._selected_paths_and_anchor()
                    self._prepare_sections_in_background(
                        tuple(self._model.root_sections()),
                        columns=columns,
                        collapsed_section_ids=set(self._model._collapsed),  # noqa: SLF001 - model owns collapse identity
                        status=self.status_label.text(),
                        selected_paths=selected_paths,
                        anchor_path=anchor_path,
                    )
                return
            self._model.set_column_count(columns)
        geometry = (columns, width, self._model.rowCount(), self._tile_size)
        if geometry == self._table_geometry:
            return
        self._table_geometry = geometry
        self.table.clearSpans()
        vertical_header = self.table.verticalHeader()
        vertical_header.setDefaultSectionSize(self._tile_size + 58)
        for column in range(columns):
            self.table.setColumnWidth(column, width)
        for row in self._model._header_rows:  # noqa: SLF001 - model precomputes sparse header geometry
            if columns > 1:
                self.table.setSpan(row, 0, 1, columns)
            self.table.setRowHeight(row, 42)

    def _visible_row_bounds(self) -> tuple[int, int]:
        count = self._model.rowCount()
        if count == 0:
            return 0, -1
        viewport = self.table.viewport().rect()
        first = self.table.rowAt(viewport.top())
        last = self.table.rowAt(max(viewport.top(), viewport.bottom() - 1))
        if first < 0:
            first = 0
        if last < 0:
            last = count - 1
        return max(0, first), min(count - 1, last)

    def _visible_indexes(self) -> list[QModelIndex]:
        indexes: list[QModelIndex] = []
        first, last = self._visible_row_bounds()
        for row in range(first, last + 1):
            index = self._model.index(row, 0)
            if self._model.section_at_header(index) is not None:
                continue
            for column in range(self._model.columnCount()):
                cell = self._model.index(row, column)
                if self._model.path_at(cell):
                    indexes.append(cell)
        return indexes

    def _queue_visible_loads(self) -> None:
        if self._shutting_down:
            return
        preview_paths: list[str] = []
        first, last = self._visible_row_bounds()
        for row in range(first, last + 1):
            index = self._model.index(row, 0)
            if self._model.section_at_header(index) is None:
                continue
            section = self._model.section_at_header(index)
            if section is not None:
                preview_paths.extend(self._model.paths_for_section(section.section_id)[:3])
        visible_paths = [self._model.path_at(index) for index in self._visible_indexes()]
        visible_paths = [str(path) for path in visible_paths if path]
        next_visible = set(visible_paths)
        if next_visible != self._viewport_paths:
            # A QThreadPool cannot interrupt an image decode safely, but it can
            # discard not-yet-started work.  A fresh epoch prevents an old
            # viewport result from repainting after a fast scroll.
            self._generation += 1
            self._job_pool.clear()
            self._loading_paths.clear()
            self._label_paths.clear()
            self._finish_viewport_job(status="cancelled")
            self._viewport_paths = next_visible
        # Current tiles always enter the high-priority queue first.  Header
        # contact-sheet previews are convenience work and must never delay the
        # images the user can actually see.
        self._queue_paths(visible_paths[:48], priority=10)
        self._queue_paths(preview_paths[:24], priority=-10)

    def _queue_visible_loads_after_layout(self) -> None:
        # Model resets can report stale visual rectangles for one event loop.
        # Retry after layout so collapsed thumbnails cannot remain on Loading.
        QTimer.singleShot(0, self._queue_visible_loads)
        QTimer.singleShot(60, self._queue_visible_loads)

    def _queue_paths(self, paths: list[str] | tuple[str, ...], *, priority: int = 0) -> None:
        if self._shutting_down:
            return
        generation = self._generation
        for path in dict.fromkeys(str(path) for path in paths if path):
            if path not in self._loading_paths and not self._model.has_image(path):
                self._loading_paths.add(path)
                self._begin_viewport_task(generation, path, "thumbnail")
                runnable = _ThumbnailRunnable(
                    path, self._tile_size, self._thumbnail_service, generation, self._job_pool
                )
                runnable.loaded.connect(self._on_thumbnail_loaded)
                runnable.failed.connect(self._on_thumbnail_failed)
                self._job_pool.start(runnable, int(priority))
            if path not in self._label_paths:
                self._label_paths.add(path)
                self._begin_viewport_task(generation, path, "label")
                label_runnable = _LabelRunnable(path, generation, self._job_pool)
                label_runnable.loaded.connect(self._on_label_loaded)
                self._job_pool.start(label_runnable, int(priority))

    def _on_thumbnail_loaded(self, generation: int, path: str, image: QImage) -> None:
        if generation == self._generation:
            self._loading_paths.discard(path)
        self._complete_viewport_task(generation, path, "thumbnail")
        if self._shutting_down or generation != self._generation or image.isNull():
            return
        self._model.set_image(path, image)
        self._refresh_hover_preview_for_path(path)

    def _on_thumbnail_failed(self, generation: int, path: str, error: str) -> None:
        if generation == self._generation:
            self._loading_paths.discard(path)
        self._complete_viewport_task(generation, path, "thumbnail")
        if not self._shutting_down and generation == self._generation:
            self._model.set_failed(path, error)
            self._refresh_hover_preview_for_path(path)

    def _on_label_loaded(self, generation: int, path: str, label: str) -> None:
        if generation == self._generation:
            self._label_paths.discard(path)
        self._complete_viewport_task(generation, path, "label")
        if not self._shutting_down and generation == self._generation:
            self._model.set_label(path, label)

    def _begin_viewport_task(self, generation: int, path: str, kind: str) -> None:
        # Label reads have their own cache and are not thumbnail work.  Keeping
        # them out of this job makes the displayed denominator a truthful count
        # of image decodes instead of a mixture of unrelated task units.
        if str(kind) != "thumbnail":
            return
        if generation != self._viewport_job_generation:
            self._finish_viewport_job(status="cancelled")
            self._viewport_job_generation = generation
            self._viewport_tasks.clear()
            self._viewport_completed_tasks.clear()
        task = (str(path), str(kind))
        if task in self._viewport_tasks:
            return
        self._viewport_tasks.add(task)
        if self._viewport_job_id is None and self._job_manager is not None:
            self._viewport_job_id = self._job_manager.register_job(
                "Loading visible thumbnails", origin=self._job_origin, foreground=False
            )
        self._update_viewport_progress()

    def _complete_viewport_task(self, generation: int, path: str, kind: str) -> None:
        if str(kind) != "thumbnail":
            return
        if generation != self._viewport_job_generation:
            return
        task = (str(path), str(kind))
        if task not in self._viewport_tasks:
            return
        self._viewport_completed_tasks.add(task)
        self._update_viewport_progress()
        if self._viewport_tasks.issubset(self._viewport_completed_tasks):
            self._finish_viewport_job(status="finished")

    def _update_viewport_progress(self) -> None:
        total = len(self._viewport_tasks)
        completed = len(self._viewport_completed_tasks)
        if total <= 0:
            self.progress_bar.hide()
            return
        value = int((completed * 100) / total)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(max(0, min(100, value)))
        self.progress_bar.show()
        text = thumbnail_progress(completed, total, qualifier="visible")
        self.status_label.setText(text)
        if self._job_manager is not None and self._viewport_job_id is not None:
            self._job_manager.update(self._viewport_job_id, progress=value, text=text)

    def _finish_viewport_job(self, *, status: str) -> None:
        total = len(self._viewport_tasks)
        completed = len(self._viewport_completed_tasks)
        if self._job_manager is not None and self._viewport_job_id is not None:
            self._job_manager.finish(self._viewport_job_id, status=status)
        self._viewport_job_id = None
        self._viewport_job_generation = -1
        self._viewport_tasks.clear()
        self._viewport_completed_tasks.clear()
        self.progress_bar.hide()
        if status == "finished" and total:
            self.status_label.setText(f"{thumbnail_progress(completed, total, done=True)}.")

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        """Drain view-owned thumbnail work before Qt tears down this view."""
        self._shutting_down = True
        self._section_prepare_generation += 1
        section_refs = [
            (getattr(self, "_section_prepare_job", None), getattr(self, "_section_prepare_thread", None)),
            *list(self._retained_section_prepare_refs),
        ]
        self._section_prepare_job = None
        self._section_prepare_thread = None
        self._retained_section_prepare_refs = []
        coordinated_job_ids = tuple(self._section_prepare_coordinated_job_ids.values())
        if coordinated_job_ids and self._work_coordinator is not None:
            for coordinated_job_id in coordinated_job_ids:
                self._work_coordinator.cancel(coordinated_job_id)
        else:
            for job, _thread in section_refs:
                if job is not None:
                    try:
                        job.cancel()
                    except Exception:
                        pass
        for _job, thread in section_refs:
            if thread is None:
                continue
            try:
                if thread.isRunning() and not wait_for_thread_shutdown(thread, timeout_ms=int(timeout_ms)):
                    return False
            except RuntimeError:
                continue
            except Exception:
                return False
        if self._work_coordinator is not None:
            for coordinated_job_id in coordinated_job_ids:
                self._work_coordinator.finish(coordinated_job_id, status="cancelled")
        self._section_prepare_coordinated_job_ids.clear()
        self._finish_viewport_job(status="cancelled")
        self._generation += 1
        self._loading_paths.clear()
        self._label_paths.clear()
        self._hide_hover_preview()
        try:
            self._job_pool.clear()
            return bool(self._job_pool.waitForDone(max(0, int(timeout_ms))))
        except RuntimeError:
            return True
        except Exception:
            return False

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self.shutdown_jobs():
            event.ignore()
            return
        super().closeEvent(event)

    def _toggle_checked_section(self, section_id: str) -> None:
        self._model.toggle_checked_section(section_id)
        self._refresh_group_bar()
        self.table.viewport().update()

    def _toggle_collapsed_section(self, section_id: str) -> None:
        self._hide_hover_preview()
        section = next((item for item in self._model.sections() if item.section_id == section_id), None)
        expanding = section_id in self._model._collapsed  # noqa: SLF001 - model owns collapsed state
        self._model.toggle_collapsed(section_id)
        self._apply_table_geometry()
        self._refresh_group_bar()
        if expanding and section is not None:
            self._queue_paths(self._model.paths_for_section(section.section_id)[:48])
        self._queue_visible_loads_after_layout()

    def expand_all_sections(self) -> None:
        if self._model.set_all_collapsed(False):
            self._apply_table_geometry()
            self._refresh_group_bar()
            self._queue_visible_loads_after_layout()

    def collapse_all_sections(self) -> None:
        if self._model.set_all_collapsed(True):
            self._apply_table_geometry()
            self._refresh_group_bar()

    def eventFilter(self, watched, event):  # type: ignore[override]
        if watched is self.table.viewport():
            if event.type() == QEvent.Type.ToolTip:
                return True
            if event.type() == QEvent.Type.MouseMove:
                self._update_hover_preview(self.table.indexAt(event.pos()))
            elif event.type() in {
                QEvent.Type.Leave,
                QEvent.Type.Hide,
                QEvent.Type.Wheel,
                QEvent.Type.MouseButtonPress,
            }:
                self._hide_hover_preview()
        return super().eventFilter(watched, event)

    def _on_table_scrolled(self, _value: int) -> None:
        self._hide_hover_preview()
        self._queue_visible_loads()

    def _update_hover_preview(self, index: QModelIndex) -> None:
        section = self._model.section_at_header(index)
        if section is None:
            self._hide_hover_preview()
            return
        if self._hover_section_id == section.section_id and self._hover_popup.isVisible():
            self._hover_index = index
            self._position_hover_preview()
            return
        self._hover_section_id = section.section_id
        self._hover_index = index
        self._queue_paths(self._model.paths_for_section(section.section_id)[:9])
        self._refresh_hover_preview(section)
        self._position_hover_preview()
        self._hover_popup.show()

    def _refresh_hover_preview_for_path(self, path: str) -> None:
        if not self._hover_section_id:
            return
        section = next((item for item in self._model.sections() if item.section_id == self._hover_section_id), None)
        if section is not None and path in self._model.paths_for_section(section.section_id)[:9]:
            self._refresh_hover_preview(section)

    def _refresh_hover_preview(self, section: GallerySection) -> None:
        canvas = QImage(_GroupHoverPreviewPopup.IMAGE_SIZE, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(QColor(COLORS["surface_sunken"]))
        painter = QPainter(canvas)
        cell = 72
        loading = False
        paths = self._model.paths_for_section(section.section_id)
        for offset, path in enumerate(paths[:9]):
            target = QRect((offset % 3) * cell, (offset // 3) * cell, cell, cell)
            image = self._model.image_for_path(path)
            if image is None or image.isNull():
                loading = True
                painter.fillRect(target, QColor(COLORS["surface"]))
            else:
                scaled = image.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
                source_x = max(0, (scaled.width() - target.width()) // 2)
                source_y = max(0, (scaled.height() - target.height()) // 2)
                painter.drawImage(target, scaled, QRect(source_x, source_y, target.width(), target.height()))
            painter.setPen(QPen(QColor(COLORS["border"])))
            painter.drawRect(target.adjusted(0, 0, -1, -1))
        painter.end()
        self._hover_popup.set_group(section, canvas, paths=paths, loading=loading)

    def _position_hover_preview(self) -> None:
        if not self._hover_index.isValid():
            return
        rect = self.table.visualRect(self._hover_index)
        if not rect.isValid():
            return
        anchor = self.table.viewport().mapToGlobal(rect.topRight())
        self._hover_popup.adjustSize()
        popup_rect = self._hover_popup.frameGeometry()
        screen = self.table.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(anchor, popup_rect.size())
        margin = 12
        target_x = anchor.x() + margin
        target_y = anchor.y() + margin
        if target_x + popup_rect.width() > available.right() - margin:
            left_anchor = self.table.viewport().mapToGlobal(rect.topLeft())
            target_x = left_anchor.x() - popup_rect.width() - margin
        if target_y + popup_rect.height() > available.bottom() - margin:
            target_y = max(available.top() + margin, available.bottom() - popup_rect.height() - margin)
        self._hover_popup.move(max(available.left() + margin, target_x), target_y)

    def _hide_hover_preview(self) -> None:
        self._hover_popup.hide()
        self._hover_section_id = ""
        self._hover_index = QModelIndex()

    def _current_action_target(self) -> SelectionTarget | None:
        selected_paths = [self._model.path_at(index) for index in self.table.selectionModel().selectedIndexes() if self._model.path_at(index)]
        if selected_paths:
            paths = tuple(dict.fromkeys(selected_paths))
            return SelectionTarget(paths=paths, kind="selected_gallery_images", label=f"Selected photos ({len(paths)})")
        checked_sections = self._model.checked_section_ids()
        if checked_sections:
            paths = tuple(self._model.paths_for_sections(checked_sections))
            return SelectionTarget(paths=paths, kind="selected_gallery_sections", label=f"Selected groups ({len(checked_sections)})")
        return None

    def _refresh_group_bar(self) -> None:
        target = self._current_action_target()
        if target is None:
            self.group_bar.hide()
            return
        self.group_summary.setText(f"{target.label} · {len(target.paths)} photos")
        self.group_bar.show()

    def _show_section_menu(self, section_id: str, point: QPoint) -> None:
        section = next((item for item in self._model.sections() if item.section_id == section_id), None)
        if section is None:
            return
        paths = self._model.paths_for_section(section.section_id)
        menu = QMenu(self)
        menu.addAction("Inspect photos", lambda: self._inspect_paths(list(paths)))
        menu.addAction("Reveal folders", lambda: self._run_for_paths(paths, self._actions.slotOpenSelectedFolders))
        menu.addAction("Tag photos", lambda: self._run_for_paths(paths, self._actions.slotEditTags))
        menu.addSeparator()
        menu.addAction("Copy photos", lambda: self._run_for_paths(paths, self._actions.slotCopySelected))
        menu.addAction("Move photos", lambda: self._run_for_paths(paths, self._actions.slotMoveSelected))
        menu.addAction("Move to ClusterLens Trash", lambda: self._run_for_paths(paths, self._actions.slotDeleteSelect))
        menu.exec(point)

    def _show_photo_menu(self, pos: QPoint) -> None:
        index = self.table.indexAt(pos)
        path = self._model.path_at(index)
        if not path:
            return
        menu = QMenu(self)
        menu.addAction("Open", lambda: self._open_photo(index))
        if not self._read_only:
            menu.addAction("Edit Face Regions", lambda: self._request_face_edit(path))
        menu.addAction("Reveal folder", lambda: self._run_for_paths((path,), self._actions.slotOpenSelectedFolders))
        menu.addAction("Tag", lambda: self._run_for_paths((path,), self._actions.slotEditTags))
        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _run_for_paths(self, paths: tuple[str, ...] | list[str], callback) -> None:
        original = self._actions.action_target_provider
        target = SelectionTarget(paths=tuple(paths), kind="gallery_section", label=f"Group ({len(paths)} photos)")
        self._actions.set_action_target_provider(lambda: target)
        try:
            callback()
        finally:
            self._actions.set_action_target_provider(original)

    def _inspect_current_target(self) -> None:
        target = self._current_action_target()
        if target is not None:
            self._inspect_paths(list(target.paths))

    def _inspect_paths(self, paths: list[str]) -> None:
        if not paths:
            return
        dialog = PhotoInspectorDialog(
            image_paths=list(paths),
            start_index=0,
            context_provider=self._inspector_context_for_path,
            metadata_service=self._metadata_service,
            display_mode="basic",
            face_service=self._active_face_service(),
            allow_face_edit=not self._read_only,
            face_edit_saved_callback=self.face_edit_saved_callback,
            allow_metadata_edit=not self._read_only,
            allow_file_rename=not self._read_only,
            rename_current_callback=lambda path: self._run_for_paths((path,), self._actions.slotPreviewBatchRename),
            job_manager=self._job_manager,
            work_coordinator=self._work_coordinator,
            parent=self,
        )
        dialog.exec()

    def _open_photo(self, index: QModelIndex, *, allow_face_edit: bool | None = None) -> None:
        path = self._model.path_at(index)
        if not path:
            return
        section = self._section_for_path(path)
        paths = list(self._model.paths_for_section(section.section_id)) if section is not None else [path]
        dialog = PhotoInspectorDialog(
            image_paths=paths,
            start_index=paths.index(path),
            context_provider=self._inspector_context_for_path,
            metadata_service=self._metadata_service,
            display_mode="basic",
            face_service=self._active_face_service(),
            allow_face_edit=bool(not self._read_only and (allow_face_edit is True or allow_face_edit is None)),
            face_edit_saved_callback=self.face_edit_saved_callback,
            allow_metadata_edit=not self._read_only,
            allow_file_rename=not self._read_only,
            rename_current_callback=lambda selected_path: self._run_for_paths((selected_path,), self._actions.slotPreviewBatchRename),
            job_manager=self._job_manager,
            work_coordinator=self._work_coordinator,
            parent=self,
        )
        dialog.exec()

    def _request_face_edit(self, image_path: str) -> None:
        path = str(image_path or "")
        if not path or self._read_only:
            return
        if self._active_face_service() is not None:
            self._open_photo_for_path(path, allow_face_edit=True)
            return
        handler = self.face_edit_request_handler
        if not callable(handler):
            self.status_label.setText("Face tools are unavailable. Open Faces or Names to configure a face model.")
            return

        self.status_label.setText("Preparing face tools for this photo…")

        def _ready() -> None:
            self._open_photo_for_path(path, allow_face_edit=True)

        def _failed(message: str) -> None:
            self.status_label.setText(f"Face tools could not be prepared: {str(message or 'unknown error')}")

        try:
            handler(path, _ready, _failed)
        except Exception as exc:
            _failed(str(exc))

    def _open_photo_for_path(self, path: str, *, allow_face_edit: bool) -> None:
        for row in range(self._model.rowCount()):
            for column in range(self._model.columnCount()):
                index = self._model.index(row, column)
                if self._model.path_at(index) == path:
                    self._open_photo(index, allow_face_edit=allow_face_edit)
                    return

    def _active_face_service(self):
        if not callable(self.face_service_provider):
            return None
        try:
            return self.face_service_provider()
        except Exception:
            return None

    def _inspector_context_for_path(self, path: str) -> dict[str, object]:
        if not callable(self.inspector_context_provider):
            return {}
        try:
            context = self.inspector_context_provider(str(path))
        except Exception:
            return {}
        return dict(context) if isinstance(context, dict) else {}

    def _section_for_path(self, path: str) -> GallerySection | None:
        return next((section for section in self._model.sections() if path in section.paths), None)

    def _on_paths_removed(self, paths: list[str]) -> None:
        removed = {str(path) for path in paths if path}
        if not removed:
            return

        def _remove(section: GallerySection) -> GallerySection | None:
            children = tuple(child for child in (_remove(item) for item in section.children) if child is not None)
            remaining = tuple(path for path in section.paths if path not in removed)
            if not remaining and not children:
                return None
            return GallerySection(
                section.section_id,
                remaining,
                section.kind,
                section.title,
                section.anchor_path,
                section.anchor_face_index,
                children,
            )

        sections = [section for section in (_remove(item) for item in self._model.root_sections()) if section is not None]
        self.set_sections(sections, status=f"Removed {len(removed)} photo(s) from the gallery.")
        self.paths_removed.emit(sorted(removed))

    def _on_paths_renamed(self, changed_paths: list[tuple[str, str]]) -> None:
        replacements = {
            str(source): str(target)
            for source, target in changed_paths
            if str(source).strip() and str(target).strip() and str(source) != str(target)
        }
        if not replacements:
            return

        def _replace(section: GallerySection) -> GallerySection:
            return GallerySection(
                section.section_id,
                tuple(replacements.get(path, path) for path in section.paths),
                section.kind,
                section.title,
                replacements.get(section.anchor_path, section.anchor_path),
                section.anchor_face_index,
                tuple(_replace(child) for child in section.children),
            )

        self.set_sections(
            [_replace(section) for section in self._model.root_sections()],
            status=f"Renamed {len(replacements)} photo(s).",
        )
        self.paths_renamed.emit([(source, target) for source, target in replacements.items()])


class _RunnableSignals(QObject):
    loaded = pyqtSignal(int, str, object)
    failed = pyqtSignal(int, str, str)


class _ThumbnailRunnable(QRunnable):
    def __init__(
        self,
        path: str,
        size: int,
        service: ThumbnailService,
        generation: int,
        signal_parent: QObject,
    ) -> None:
        super().__init__()
        self.path = str(path)
        self.size = int(size)
        self.service = service
        self.generation = int(generation)
        self.signals = _RunnableSignals(signal_parent)

    @property
    def loaded(self):
        return self.signals.loaded

    @property
    def failed(self):
        return self.signals.failed

    def run(self) -> None:
        try:
            image = self.service.load_qimage(self.path, self.size)
            if image.isNull():
                raise ValueError("Could not decode image")
            self.signals.loaded.emit(self.generation, self.path, image)
        except Exception as exc:
            self.signals.failed.emit(self.generation, self.path, str(exc))


class _LabelRunnable(QRunnable):
    def __init__(self, path: str, generation: int, signal_parent: QObject) -> None:
        super().__init__()
        self.path = str(path)
        self.generation = int(generation)
        self.signals = _RunnableSignals(signal_parent)

    @property
    def loaded(self):
        return self.signals.loaded

    def run(self) -> None:
        label = ""
        try:
            from app.services.gallery_actions import GalleryActionService

            label = GalleryActionService.read_exif_metadata_value(self.path, "ic_person_name")
            if not label:
                metadata = PhotoMetadataService().get_metadata(self.path, include_hashes=False)
                for key in ("XPTitle", "DocumentName", "ImageDescription"):
                    label = str(metadata.exif.get(key, "") or "").strip()
                    if label:
                        break
        except Exception:
            label = ""
        self.signals.loaded.emit(self.generation, self.path, label)
