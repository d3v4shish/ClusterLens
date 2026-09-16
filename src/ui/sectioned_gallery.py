from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, QPoint, QRect, QRunnable, QSize, Qt, QThreadPool, QTimer, pyqtSignal
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
from ui.gallery_model import GalleryImageModel
from ui.gallery_pane import GalleryPane
from ui.job_manager import JobManager
from ui.photo_inspector_dialog import PhotoInspectorDialog
from ui.theme import COLORS


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

    def set_sections(self, sections: list[GallerySection], *, reset_scroll_state: bool = True) -> None:
        unique: list[GallerySection] = []
        seen_paths: set[str] = set()
        seen_ids: set[str] = set()

        def _normalize(section: GallerySection) -> GallerySection | None:
            section_id = str(section.section_id or "").strip()
            if not section_id or section_id in seen_ids:
                return None
            seen_ids.add(section_id)
            paths = tuple(str(path) for path in section.paths if path and str(path) not in seen_paths)
            seen_paths.update(paths)
            children = tuple(child for child in (_normalize(item) for item in section.children) if child is not None)
            if not paths and not children:
                return None
            return GallerySection(
                section_id,
                paths,
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
        self.beginResetModel()
        self._sections = unique
        self._sections_by_id = {}
        self._paths_by_section = {}
        self._depth_by_section = {}

        def _index(section: GallerySection, depth: int) -> tuple[str, ...]:
            self._sections_by_id[section.section_id] = section
            self._depth_by_section[section.section_id] = depth
            paths = list(section.paths)
            for child in section.children:
                paths.extend(_index(child, depth + 1))
            unique_paths = tuple(dict.fromkeys(paths))
            self._paths_by_section[section.section_id] = unique_paths
            return unique_paths

        all_paths: list[str] = []
        for section in unique:
            all_paths.extend(_index(section, 0))
        self._all_paths = tuple(dict.fromkeys(all_paths))
        valid_paths = set(self._all_paths)
        self._paths = valid_paths
        self._collapsed.intersection_update(self._sections_by_id)
        self._checked_sections.intersection_update(self._sections_by_id)
        self._checked_paths.intersection_update(valid_paths)
        self._images = OrderedDict((path, image) for path, image in self._images.items() if path in valid_paths)
        self._labels = {path: label for path, label in self._labels.items() if path in valid_paths}
        self._failed = {path: error for path, error in self._failed.items() if path in valid_paths}
        self._rebuild_rows()
        self.endResetModel()

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
        for row_index, row in enumerate(self._rows):
            if row.kind != "photos" or path not in row.paths:
                continue
            column = row.paths.index(path)
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
    organize_requested = pyqtSignal()
    analyze_requested = pyqtSignal(list)
    review_faces_requested = pyqtSignal(list)
    return_to_folder_requested = pyqtSignal()
    return_to_source_requested = pyqtSignal(str)
    paths_removed = pyqtSignal(list)
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
        self._viewport_job_id: int | None = None
        self._viewport_job_generation = -1
        self._viewport_tasks: set[tuple[str, str]] = set()
        self._viewport_completed_tasks: set[tuple[str, str]] = set()
        self._generation = 0
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
        self.organize_button.setToolTip("Arrange this folder using the current Clustering settings.")
        self.organize_button.clicked.connect(self.organize_requested.emit)
        self.analyze_button = QPushButton("Analyze in Clustering")
        self.analyze_button.setToolTip("Use this explicit photo set as the input to Clustering.")
        self.analyze_button.setAccessibleName("Analyze current photo set in Clustering")
        self.analyze_button.clicked.connect(lambda: self.analyze_requested.emit(self.selected_or_all_paths()))
        self.review_faces_button = QPushButton("Review Faces")
        self.review_faces_button.setToolTip("Open this explicit photo set in Faces. Detection starts only after you choose Detect Faces.")
        self.review_faces_button.setAccessibleName("Review current photo set in Faces")
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

    def set_job_manager(self, job_manager: JobManager | None) -> None:
        """Route visible gallery work to the shared Jobs surface."""
        self._job_manager = job_manager
        self._actions.job_manager = job_manager

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

    def set_sections(self, sections: list[GallerySection], *, status: str = "") -> None:
        if self._shutting_down:
            return
        self._hide_hover_preview()
        self._finish_viewport_job(status="cancelled")
        self._generation += 1
        self._loading_paths.clear()
        self._label_paths.clear()
        self._model.set_sections(sections)
        self._apply_table_geometry()
        self.status_label.setText(status or f"Showing {len(self._model.all_paths())} photos.")
        self._refresh_group_bar()
        QTimer.singleShot(0, self._queue_visible_loads)

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
        self._model.set_sections(sections)
        self._model.set_collapsed_sections(collapsed_section_ids)
        self._apply_table_geometry()
        self.status_label.setText(status or f"Showing {len(self._model.all_paths())} photos.")
        self._refresh_group_bar()
        QTimer.singleShot(0, self._queue_visible_loads)

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
        self._model.set_column_count(columns)
        self.table.clearSpans()
        width = max(80, self.table.viewport().width() // columns)
        for column in range(columns):
            self.table.setColumnWidth(column, width)
        for row in range(self._model.rowCount()):
            index = self._model.index(row, 0)
            if self._model.section_at_header(index) is not None:
                if columns > 1:
                    self.table.setSpan(row, 0, 1, columns)
                self.table.setRowHeight(row, 42)
            else:
                self.table.setRowHeight(row, self._tile_size + 58)

    def _visible_indexes(self) -> list[QModelIndex]:
        viewport = self.table.viewport().rect()
        indexes: list[QModelIndex] = []
        for row in range(self._model.rowCount()):
            index = self._model.index(row, 0)
            rect = self.table.visualRect(index)
            if rect.bottom() < viewport.top() or rect.top() > viewport.bottom():
                continue
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
        viewport = self.table.viewport().rect()
        for row in range(self._model.rowCount()):
            index = self._model.index(row, 0)
            if self._model.section_at_header(index) is None:
                continue
            rect = self.table.visualRect(index)
            if rect.bottom() < viewport.top() or rect.top() > viewport.bottom():
                continue
            section = self._model.section_at_header(index)
            if section is not None:
                preview_paths.extend(self._model.paths_for_section(section.section_id)[:3])
        visible_paths = [self._model.path_at(index) for index in self._visible_indexes()]
        self._queue_paths((preview_paths + visible_paths)[:48])

    def _queue_visible_loads_after_layout(self) -> None:
        # Model resets can report stale visual rectangles for one event loop.
        # Retry after layout so collapsed thumbnails cannot remain on Loading.
        QTimer.singleShot(0, self._queue_visible_loads)
        QTimer.singleShot(60, self._queue_visible_loads)

    def _queue_paths(self, paths: list[str] | tuple[str, ...]) -> None:
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
                self._job_pool.start(runnable)
            if path not in self._label_paths:
                self._label_paths.add(path)
                self._begin_viewport_task(generation, path, "label")
                label_runnable = _LabelRunnable(path, generation, self._job_pool)
                label_runnable.loaded.connect(self._on_label_loaded)
                self._job_pool.start(label_runnable)

    def _on_thumbnail_loaded(self, generation: int, path: str, image: QImage) -> None:
        self._loading_paths.discard(path)
        self._complete_viewport_task(generation, path, "thumbnail")
        if self._shutting_down or generation != self._generation or image.isNull():
            return
        self._model.set_image(path, image)
        self._refresh_hover_preview_for_path(path)

    def _on_thumbnail_failed(self, generation: int, path: str, error: str) -> None:
        self._loading_paths.discard(path)
        self._complete_viewport_task(generation, path, "thumbnail")
        if not self._shutting_down and generation == self._generation:
            self._model.set_failed(path, error)
            self._refresh_hover_preview_for_path(path)

    def _on_label_loaded(self, generation: int, path: str, label: str) -> None:
        self._label_paths.discard(path)
        self._complete_viewport_task(generation, path, "label")
        if not self._shutting_down and generation == self._generation:
            self._model.set_label(path, label)

    def _begin_viewport_task(self, generation: int, path: str, kind: str) -> None:
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
                "Loading visible photos", origin="Photos", foreground=False
            )
        self._update_viewport_progress()

    def _complete_viewport_task(self, generation: int, path: str, kind: str) -> None:
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
        text = f"Loading visible photos: {completed}/{total}"
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
            self.status_label.setText(f"Loaded {completed}/{total} visible photo tasks.")

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        """Drain view-owned thumbnail work before Qt tears down this view."""
        self._shutting_down = True
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
            allow_face_edit=bool(not self._read_only and self._active_face_service() is not None),
            face_edit_saved_callback=self.face_edit_saved_callback,
            job_manager=self._job_manager,
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
            allow_face_edit=bool(
                not self._read_only
                and self._active_face_service() is not None
                and (allow_face_edit is True or allow_face_edit is None)
            ),
            face_edit_saved_callback=self.face_edit_saved_callback,
            job_manager=self._job_manager,
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
