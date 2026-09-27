from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QImageReader, QPainter, QPixmap
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QComboBox, QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListView, QProgressBar, QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from app.path_scope import PathScope
from app.services.thumbnails import ThumbnailService
from ui.async_job import AsyncJob, defer_async_job_dispose, start_job_in_thread, wait_for_thread_shutdown
from ui.common import HelpIconButton
from ui.error_mbox import confirmBox
from ui.entity_picker import EntityPickerDialog
from ui.gallery_pane import GalleryPane
from ui.icons import apply_icon
from ui.job_manager import JobManager
from ui.list_models import ListEntry, ListEntryModel, PagedListEntryModel, SidebarListEntryDelegate
from ui.search_pane import FaceTileItem, FaceTileItemDelegate, FaceTileListModel
from ui.theme import COLORS, apply_field_size, get_theme_manager
from ui.work_coordinator import JobSpec, WorkCoordinator

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService, FaceSearchResult, NamedPhotoSummary


NAMES_HELP = (
    "Names are durable face-to-person assignments stored in the global face database. "
    "Select a name to see every unique photo containing a face saved with that name. "
    "Every photo in this view already has the active saved name. Right-click one or more selected photos "
    "to rename or unlabel only those matching face rows. "
    "Use Find Similar Faces to search the selected saved identity's prototype and review face crops without leaving Names."
)

SELECTED_IMAGES_HELP = (
    "Select photos with Ctrl-click, Shift-click, or their checkboxes. Rename and Unlabel apply only to "
    "faces that currently have the active saved name, so other people in the same photo are never changed."
)

NAME_HOVER_PREVIEW_IMAGE_SIZE = QSize(216, 216)
NAME_HOVER_PREVIEW_MAX_ITEMS = 9
NAME_HOVER_PREVIEW_CACHE_SIZE = 24
NAME_SIMILAR_FACE_TILE_SIZE = QSize(112, 112)
NAME_SIMILAR_FACE_LIMIT = 60


@dataclass(frozen=True)
class _NameHoverPreviewLayout:
    image_size: QSize
    columns: int
    max_items: int
    popup_width: int


@dataclass(frozen=True)
class _NameHoverPreviewPayload:
    image: QImage
    shown_count: int
    cooccurring_people: tuple[tuple[str, int], ...]


class NameHoverPreviewPopup(QFrame):
    """Non-interactive bounded preview for a durable saved-name sidebar row."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("nameHoverPreviewPopup")
        self.setStyleSheet(
            f"""
            QFrame#nameHoverPreviewPopup {{
                background: {COLORS["surface_raised"]};
                border: 1px solid {COLORS["border_strong"]};
                border-radius: 8px;
            }}
            QLabel {{ color: {COLORS["text"]}; }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel("")
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize(), 10))
        self.title_label.setFont(title_font)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("Loading preview...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(NAME_HOVER_PREVIEW_IMAGE_SIZE)
        self.image_label.setStyleSheet(
            f'background: {COLORS["surface_sunken"]}; border: 1px solid {COLORS["border"]};'
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary_label)

        self.people_label = QLabel("")
        self.people_label.setObjectName("nameHoverPreviewPeople")
        self.people_label.setWordWrap(True)
        self.people_label.setTextFormat(Qt.TextFormat.PlainText)
        self.people_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        self.people_label.hide()
        layout.addWidget(self.people_label)

        self.note_label = QLabel("")
        self.note_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        layout.addWidget(self.note_label)
        self.setFixedWidth(300)
        manager = get_theme_manager()
        if manager is not None:
            manager.theme_changed.connect(self._apply_theme)

    def _apply_theme(self, *_args) -> None:
        self.setStyleSheet(
            f"QFrame#nameHoverPreviewPopup {{ background: {COLORS['surface_raised']}; "
            f"border: 1px solid {COLORS['border_strong']}; border-radius: 8px; }} "
            f"QLabel {{ color: {COLORS['text']}; }}"
        )
        self.image_label.setStyleSheet(
            f"background: {COLORS['surface_sunken']}; border: 1px solid {COLORS['border']};"
        )
        self.people_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.note_label.setStyleSheet(f"color: {COLORS['text_muted']};")

    def set_preview_layout(self, layout: _NameHoverPreviewLayout) -> None:
        self.image_label.setFixedSize(layout.image_size)
        self.setFixedWidth(layout.popup_width)

    def set_name_details(self, name: str, *, face_count: int, photo_count: int, max_items: int, loading: bool) -> None:
        self.title_label.setText(str(name))
        self.summary_label.setText(f"{int(photo_count)} photo(s) · {int(face_count)} saved face(s)")
        self.people_label.clear()
        self.people_label.hide()
        self.note_label.setText(f"Preparing up to {int(max_items)} photos")
        if loading:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Loading preview...")

    def set_preview_details(
        self,
        *,
        shown_count: int,
        photo_count: int,
        cooccurring_people: tuple[tuple[str, int], ...],
    ) -> None:
        self.note_label.setText(f"Showing {int(shown_count)} of {int(photo_count)} photo(s)")
        visible_people = tuple((str(name), int(count)) for name, count in cooccurring_people[:6] if str(name).strip())
        if not visible_people:
            self.people_label.clear()
            self.people_label.hide()
            return
        detail = ", ".join(f"{name} ({count})" for name, count in visible_people)
        self.people_label.setText(f"Also in these photos: {detail}")
        self.people_label.show()

    def set_preview_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Preview unavailable")
            return
        self.image_label.setText("")
        self.image_label.setPixmap(QPixmap.fromImage(image))


class NamesPane(QWidget):
    """Global browser and bounded editor for durable saved face labels."""

    face_labels_changed = pyqtSignal()
    open_in_gallery_requested = pyqtSignal(list, str)
    open_similar_faces_in_gallery_requested = pyqtSignal(list, str, dict)

    def __init__(
        self,
        face_service_provider: Callable[[], "FaceIndexService"],
        parent=None,
        *,
        job_manager: JobManager | None = None,
        work_coordinator: WorkCoordinator | None = None,
        active_scope_provider: Callable[[], object] | None = None,
    ) -> None:
        super().__init__(parent)
        self._face_service_provider = face_service_provider
        self._active_scope_provider = active_scope_provider
        self.job_manager = job_manager
        self.work_coordinator = work_coordinator
        self._entries: list[ListEntry] = []
        self._selected_name = ""
        self._refresh_token = 0
        self._photos_token = 0
        self._refresh_job = None
        self._refresh_thread = None
        self._photos_job = None
        self._photos_thread = None
        self._similar_job = None
        self._similar_thread = None
        self._deep_similar_job = None
        self._deep_similar_thread = None
        self._similar_token = 0
        self._similar_face_images: dict[tuple[str, int, tuple[int, int, int, int]], QImage] = {}
        self._similar_face_items: list[FaceTileItem] = []
        self._deep_similar_review: dict[str, object] = {}
        self._mutation_job = None
        self._mutation_thread = None
        self._read_only_mode = False
        self._recovered_database_paths: set[str] = set()
        self._name_summaries: dict[str, "NamedPhotoSummary"] = {}
        self._name_preview_epoch = 0
        self._hover_popup = NameHoverPreviewPopup(self)
        self._hover_name = ""
        self._hover_index = QModelIndex()
        self._hover_preview_generation = 0
        self._hover_preview_job = None
        self._hover_preview_thread = None
        self._retained_hover_preview_refs: list[tuple[object | None, object | None]] = []
        self._hover_preview_cache: OrderedDict[tuple[object, ...], _NameHoverPreviewPayload] = OrderedDict()
        self._operation_job_ids: dict[AsyncJob, int] = {}
        self._coordinated_job_ids: dict[AsyncJob, int] = {}
        self._operation_threads: list[tuple[AsyncJob, object]] = []
        self._visible_operation_slot = ""
        self._build_ui()
        self.gallery.configure_jobs(self.job_manager, self.work_coordinator, origin="People")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("People")
        title.setProperty("role", "section")
        title.setToolTip(NAMES_HELP)
        self.help_button = HelpIconButton(NAMES_HELP, self, help_key="names_workspace")
        self.refresh_button = QPushButton("")
        self.refresh_button.setProperty("iconOnly", True)
        self.refresh_button.setAccessibleName("Refresh people")
        self.refresh_button.setToolTip("Reload durable saved names and their photo counts from the global face database.")
        apply_icon(self.refresh_button, "retry")
        self.refresh_button.clicked.connect(self.refresh_names)
        self.open_in_gallery_button = QPushButton("Photos")
        self.open_in_gallery_button.setToolTip(
            "Open the active name's exact labeled photos as the current photo set. This does not search similar faces."
        )
        self.open_in_gallery_button.setAccessibleName("Open named photos")
        self.open_in_gallery_button.clicked.connect(self._open_named_photos_in_gallery)
        self.open_in_gallery_button.setEnabled(False)
        apply_icon(self.open_in_gallery_button, "reveal")
        self.find_similar_button = QPushButton("Similar")
        self.find_similar_button.setToolTip(
            "Search the selected saved name's identity prototype and show up to 60 matching face crops, including unlabeled similar faces."
        )
        self.find_similar_button.setAccessibleName("Find faces similar to selected name")
        self.find_similar_button.clicked.connect(self._find_similar_faces)
        self.find_similar_button.setEnabled(False)
        apply_icon(self.find_similar_button, "similar")
        self.deep_find_similar_button = QPushButton("Deep similar")
        self.deep_find_similar_button.setToolTip(
            "Search the complete saved face index through newly found unlabeled faces, then review the combined result before saving names."
        )
        self.deep_find_similar_button.clicked.connect(self._deep_find_similar_faces)
        self.deep_find_similar_button.setEnabled(False)
        apply_icon(self.deep_find_similar_button, "similar")
        title_row.addWidget(title)
        title_row.addWidget(self.help_button)
        self.scope_combo = QComboBox(self)
        self.scope_combo.addItem("Active roots", "active")
        self.scope_combo.addItem("All indexed faces", "global")
        self.scope_combo.setToolTip("Names and similarity searches use active roots by default. Choose All indexed faces only when you intentionally want the global face library.")
        apply_field_size(self.scope_combo, "short")
        self.scope_combo.currentIndexChanged.connect(lambda _index: self.active_scope_changed())
        title_row.addWidget(self.scope_combo)
        title_row.addStretch(1)
        title_row.addWidget(self.open_in_gallery_button)
        title_row.addWidget(self.find_similar_button)
        title_row.addWidget(self.deep_find_similar_button)
        title_row.addWidget(self.refresh_button)
        layout.addLayout(title_row)

        self.status_label = QLabel("")
        self.status_label.setProperty("role", "helper")
        self.status_label.setToolTip(NAMES_HELP)
        self.status_label.hide()
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setAccessibleName("People workspace operation progress")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        sidebar = QWidget(splitter)
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)
        sidebar_title_row = QHBoxLayout()
        sidebar_title_row.setContentsMargins(0, 0, 0, 0)
        sidebar_title = QLabel("People")
        sidebar_title.setToolTip(NAMES_HELP)
        sidebar_title_row.addWidget(sidebar_title)
        sidebar_title_row.addWidget(HelpIconButton(NAMES_HELP, sidebar, help_key="saved_names"))
        sidebar_title_row.addStretch(1)
        sidebar_layout.addLayout(sidebar_title_row)
        self.search_field = QLineEdit(sidebar)
        self.search_field.setPlaceholderText("Filter names")
        self.search_field.setToolTip("Filter saved names. This does not search visually similar people.")
        self.search_field.setAccessibleName("Filter saved names")
        self.search_field.textChanged.connect(self._apply_filter)
        sidebar_layout.addWidget(self.search_field)
        self.count_label = QLabel("0 names")
        self.count_label.setToolTip("Number of saved names matching the current filter.")
        sidebar_layout.addWidget(self.count_label)
        self.names_model = PagedListEntryModel(self, page_size=50)
        self.names_list = QListView(sidebar)
        self.names_list.setObjectName("namesWorkspaceList")
        self.names_list.setAccessibleName("Saved people")
        self.names_list.setProperty("sidebarList", True)
        self.names_list.setModel(self.names_model)
        self.names_list.setItemDelegate(SidebarListEntryDelegate(self.names_list))
        self.names_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.names_list.setUniformItemSizes(True)
        self.names_list.setToolTip(NAMES_HELP)
        self.names_list.setMouseTracking(True)
        self.names_list.viewport().setMouseTracking(True)
        self.names_list.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.names_list.viewport().installEventFilter(self)
        self.names_list.verticalScrollBar().valueChanged.connect(lambda _value: self._hide_name_hover_preview(cancel_preview=True))
        self.names_list.selectionModel().selectionChanged.connect(lambda *_args: self._on_name_selected())
        self.names_model.rowsInserted.connect(lambda *_args: self._update_count_label())
        self.names_model.modelReset.connect(lambda: self._hide_name_hover_preview(cancel_preview=True))
        sidebar_layout.addWidget(self.names_list, stretch=1)
        splitter.addWidget(sidebar)

        self.results_stack = QStackedWidget(splitter)
        self.gallery = GalleryPane(self.results_stack)
        self.gallery.setObjectName("namesWorkspaceGallery")
        self.gallery.set_action_bar_visible(False)
        self.gallery.set_empty_state(
            "Choose a saved name",
            "Select a name in the sidebar to show photos with a durable face label.",
            show_select_folder=False,
            show_run=False,
        )
        self.gallery.set_context_menu_action_provider(self._names_context_menu_actions)
        self.results_stack.addWidget(self.gallery)

        self.similar_faces_panel = QWidget(self.results_stack)
        self.similar_faces_panel.setObjectName("namesSimilarFacesPanel")
        similar_layout = QVBoxLayout(self.similar_faces_panel)
        similar_layout.setContentsMargins(0, 0, 0, 0)
        similar_layout.setSpacing(8)
        self.similar_faces_summary = QLabel("Select a saved name, then use Find Similar Faces.")
        self.similar_faces_summary.setWordWrap(True)
        self.similar_faces_summary.setToolTip(
            "Every tile is an individual matching face region. Scores compare the selected saved identity prototype to the indexed face."
        )
        similar_layout.addWidget(self.similar_faces_summary)
        similar_actions = QHBoxLayout()
        similar_actions.setContentsMargins(0, 0, 0, 0)
        self.show_named_photos_button = QPushButton("Show Named Photos")
        self.show_named_photos_button.setToolTip("Return to the selected name's exact durable labeled-photo list.")
        self.show_named_photos_button.clicked.connect(self._show_named_photos)
        self.open_selected_similar_face_button = QPushButton("Open Selected Photo")
        self.open_selected_similar_face_button.setToolTip("Open the selected matching face's photo with its face-region context.")
        self.open_selected_similar_face_button.setEnabled(False)
        self.open_selected_similar_face_button.clicked.connect(self._open_selected_similar_face_in_gallery)
        self.open_similar_faces_button = QPushButton("Open Match Photos")
        self.open_similar_faces_button.setToolTip("Open every photo with a matching face as the current photo set.")
        self.open_similar_faces_button.setEnabled(False)
        self.open_similar_faces_button.clicked.connect(self._open_similar_faces_in_gallery)
        self.select_all_deep_faces_button = QPushButton("Select All Unlabelled")
        self.select_all_deep_faces_button.setToolTip("Select every still-unlabelled face in the current deep-search review.")
        self.select_all_deep_faces_button.clicked.connect(self._select_all_deep_similar_faces)
        self.select_all_deep_faces_button.setEnabled(False)
        self.apply_deep_faces_button = QPushButton("Apply Selected Names")
        self.apply_deep_faces_button.setProperty("kind", "primary")
        self.apply_deep_faces_button.setToolTip("Write the selected deep-search face regions to XMP/EXIF and save their durable name.")
        self.apply_deep_faces_button.clicked.connect(self._apply_selected_deep_similar_faces)
        self.apply_deep_faces_button.setEnabled(False)
        similar_actions.addWidget(self.show_named_photos_button)
        similar_actions.addWidget(self.open_selected_similar_face_button)
        similar_actions.addWidget(self.open_similar_faces_button)
        similar_actions.addWidget(self.select_all_deep_faces_button)
        similar_actions.addWidget(self.apply_deep_faces_button)
        similar_actions.addStretch(1)
        similar_layout.addLayout(similar_actions)
        self.similar_faces_model = FaceTileListModel(
            self._similar_face_image_for_item,
            self._similar_face_cache_key_for_item,
            NAME_SIMILAR_FACE_TILE_SIZE,
            self,
        )
        self.similar_faces_list = QListView(self.similar_faces_panel)
        self.similar_faces_list.setAccessibleName("Faces similar to the selected person")
        self.similar_faces_list.setObjectName("namesSimilarFacesList")
        self.similar_faces_list.setViewMode(QListView.ViewMode.IconMode)
        self.similar_faces_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.similar_faces_list.setMovement(QListView.Movement.Static)
        self.similar_faces_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.similar_faces_list.setUniformItemSizes(True)
        self.similar_faces_list.setIconSize(NAME_SIMILAR_FACE_TILE_SIZE)
        self.similar_faces_list.setGridSize(QSize(150, 170))
        self.similar_faces_list.setSpacing(8)
        self.similar_faces_list.setModel(self.similar_faces_model)
        self.similar_faces_list.setItemDelegate(FaceTileItemDelegate(self.similar_faces_list))
        self.similar_faces_list.doubleClicked.connect(lambda _index: self._open_selected_similar_face_in_gallery())
        selection_model = self.similar_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_args: self._update_similar_face_actions())
        similar_layout.addWidget(self.similar_faces_list, stretch=1)
        self.results_stack.addWidget(self.similar_faces_panel)
        splitter.addWidget(self.results_stack)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 9)
        splitter.setSizes([280, 1060])
        layout.addWidget(splitter, stretch=1)

    def set_job_manager(self, job_manager: JobManager | None) -> None:
        """Attach the shared Jobs registry after lazy workspace construction."""

        self.job_manager = job_manager

    def _set_operation_progress(self, slot: str, value: int, text: str) -> None:
        self._visible_operation_slot = str(slot)
        if int(value) < 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(value))))
        self.progress_bar.show()
        if text:
            self.status_label.setText(str(text))
            self.status_label.show()

    def _finish_operation_progress(self, slot: str) -> None:
        if self._visible_operation_slot != str(slot):
            return
        self.progress_bar.hide()
        self._visible_operation_slot = ""

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        self.gallery.set_read_only_mode(self._read_only_mode)

    def _scope_roots(self) -> tuple[str, ...] | None:
        if str(self.scope_combo.currentData() or "active") == "global":
            return None
        try:
            value = self._active_scope_provider() if self._active_scope_provider is not None else PathScope()
        except Exception:
            return ()
        if isinstance(value, PathScope):
            return value.roots
        if isinstance(value, (list, tuple, set)):
            return PathScope.from_paths(value).roots
        return PathScope.from_paths([str(value or "")]).roots

    @staticmethod
    def _call_scoped(method, *args, scope_roots, **kwargs):
        """Keep lightweight legacy/test face-service adapters usable."""

        try:
            return method(*args, scope_roots=scope_roots, **kwargs)
        except TypeError as exc:
            if "scope_roots" not in str(exc):
                raise
            return method(*args, **kwargs)

    def active_scope_changed(self) -> None:
        """Discard stale root-scoped results after the shared scope changes."""

        self._refresh_token += 1
        self._photos_token += 1
        self._similar_token += 1
        for job in (self._refresh_job, self._photos_job, self._similar_job, self._deep_similar_job):
            self._cancel_operation_job(job)
        self._clear_similar_face_results()
        self.refresh_names(preserve_name=self._current_name() or self._selected_name)

    def refresh_names(self, *, preserve_name: str | None = None) -> None:
        """Load names asynchronously and recover legacy explicit manual labels once."""

        self._refresh_token += 1
        token = self._refresh_token
        selected_name = str(preserve_name or "").strip() or self._current_name() or self._selected_name
        service = self._face_service_provider()
        database_key = str(getattr(service, "db_path", "") or id(service))
        recover = database_key not in self._recovered_database_paths
        scope_roots = self._scope_roots()
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Loading saved names...")
        self.status_label.show()

        def _run(progress, cancel_check):
            progress(-1, "Loading saved names...")
            recovery = None
            if recover:
                recovery = service.recover_legacy_manual_face_labels()
            if cancel_check():
                return None
            return recovery, self._call_scoped(service.list_named_photo_summaries, scope_roots=scope_roots)

        def _done(result) -> None:
            if token != self._refresh_token or result is None:
                return
            recovery, summaries = result
            if recover:
                self._recovered_database_paths.add(database_key)
            self._set_summaries(list(summaries or []), preserve_name=selected_name)
            recovery_text = ""
            if recovery is not None and int(getattr(recovery, "promoted_count", 0) or 0):
                recovery_text = f" Recovered {int(recovery.promoted_count)} earlier manual name(s)."
            conflicts = int(getattr(recovery, "conflict_count", 0) or 0) if recovery is not None else 0
            if conflicts:
                recovery_text += f" {conflicts} conflicting older proposal(s) remain in review."
            if recovery_text:
                self.status_label.setText(f"{len(list(summaries or []))} saved name(s).{recovery_text}")
                self.status_label.show()
            else:
                self.status_label.setText(f"{len(list(summaries or []))} saved name(s).")
                self.status_label.hide()

        def _failed(_message: str) -> None:
            self.status_label.setText("Could not load saved people.")
            self.status_label.show()

        self._start_job("refresh", _run, _done, _failed)

    def _set_summaries(self, summaries: list["NamedPhotoSummary"], *, preserve_name: str) -> None:
        self._name_summaries = {
            str(summary.person_name): summary
            for summary in summaries
            if str(summary.person_name or "").strip()
        }
        self._name_preview_epoch += 1
        self._hover_preview_cache.clear()
        self._hide_name_hover_preview(cancel_preview=True)
        self._entries = [
            ListEntry(
                title=str(summary.person_name),
                subtitle=f"{summary.photo_count} photo(s) · {summary.face_count} face(s)",
                tooltip=(
                    f"{summary.person_name}: {summary.face_count} saved face(s) across "
                    f"{summary.photo_count} unique photo(s)."
                ),
                payload=summary.person_name,
            )
            for summary in summaries
            if str(summary.person_name or "").strip()
        ]
        self.names_model.set_source_items(self._entries)
        self._update_count_label()
        self._select_name(preserve_name)

    def _apply_filter(self) -> None:
        preserve_name = self._current_name() or self._selected_name
        self.names_model.set_filter_text(self.search_field.text())
        self._update_count_label()
        self._select_name(preserve_name)

    def _update_count_label(self) -> None:
        count = self.names_model.total_count
        self.count_label.setText(f"{count} name{'s' if count != 1 else ''}")

    def _select_name(self, name: str) -> None:
        row = self.names_model.row_for_payload(str(name or "")) if name else -1
        if row < 0 and self.names_model.rowCount():
            row = 0
        if row < 0:
            self._selected_name = ""
            self.find_similar_button.setEnabled(False)
            self._clear_similar_face_results()
            self.results_stack.setCurrentWidget(self.gallery)
            self.gallery.update_gallery_with_options(images=[], clear_pixmaps=False, reset_scroll=True)
            self.gallery.set_empty_state(
                "No saved names",
                "Name faces in Faces to add them here.",
                show_select_folder=False,
                show_run=False,
            )
            return
        index = self.names_model.index(row, 0)
        self.names_list.setCurrentIndex(index)
        self.names_list.selectionModel().select(
            index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
        )

    def _current_name(self) -> str:
        index = self.names_list.currentIndex()
        if not index.isValid():
            return ""
        return str(index.data(ListEntryModel.PayloadRole) or "").strip()

    def eventFilter(self, watched, event):  # type: ignore[override]
        if watched is self.names_list.viewport():
            if event.type() == QEvent.Type.ToolTip:
                return True
            if event.type() == QEvent.Type.MouseMove:
                self._update_name_hover_preview(self.names_list.indexAt(event.pos()))
            elif event.type() in {
                QEvent.Type.Leave,
                QEvent.Type.Hide,
                QEvent.Type.Wheel,
                QEvent.Type.MouseButtonPress,
            }:
                self._hide_name_hover_preview(cancel_preview=True)
        return super().eventFilter(watched, event)

    def _update_name_hover_preview(self, index: QModelIndex) -> None:
        if not index.isValid():
            self._hide_name_hover_preview(cancel_preview=True)
            return
        name = str(index.data(ListEntryModel.PayloadRole) or "").strip()
        summary = self._name_summaries.get(name)
        if not name or summary is None:
            self._hide_name_hover_preview(cancel_preview=True)
            return
        if self._hover_name == name and self._hover_popup.isVisible():
            self._hover_index = index
            self._position_name_hover_preview()
            return
        self._cancel_name_hover_preview_job()
        self._hover_name = name
        self._hover_index = index
        self._hover_preview_generation += 1
        generation = self._hover_preview_generation
        face_count = int(getattr(summary, "face_count", 0) or 0)
        photo_count = int(getattr(summary, "photo_count", 0) or 0)
        layout = self._name_hover_preview_layout()
        self._hover_popup.set_preview_layout(layout)
        self._hover_popup.set_name_details(
            name,
            face_count=face_count,
            photo_count=photo_count,
            max_items=layout.max_items,
            loading=True,
        )
        self._position_name_hover_preview()
        self._hover_popup.show()
        cache_key = (
            self._name_preview_epoch,
            name,
            face_count,
            photo_count,
            layout.image_size.width(),
            layout.image_size.height(),
            layout.columns,
            layout.max_items,
        )
        cached = self._hover_preview_cache.get(cache_key)
        if cached is not None:
            self._hover_preview_cache.move_to_end(cache_key)
            self._apply_name_hover_preview(cached, generation)
            return
        self._start_name_hover_preview_job(name, cache_key, generation, layout, photo_count)

    def _name_hover_preview_layout(self) -> _NameHoverPreviewLayout:
        screen = self.names_list.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 0, 0)
        if available.width() >= 1900 and available.height() >= 900:
            return _NameHoverPreviewLayout(QSize(480, 352), columns=5, max_items=20, popup_width=520)
        if available.width() >= 1320 and available.height() >= 760:
            return _NameHoverPreviewLayout(QSize(336, 252), columns=4, max_items=12, popup_width=376)
        return _NameHoverPreviewLayout(
            NAME_HOVER_PREVIEW_IMAGE_SIZE,
            columns=3,
            max_items=NAME_HOVER_PREVIEW_MAX_ITEMS,
            popup_width=300,
        )

    def _start_name_hover_preview_job(
        self,
        name: str,
        cache_key: tuple[object, ...],
        generation: int,
        layout: _NameHoverPreviewLayout,
        photo_count: int,
    ) -> None:
        service = self._face_service_provider()
        scope_roots = self._scope_roots()

        def _run(_progress, cancel_check):
            if cancel_check():
                return None
            paths = list(self._call_scoped(service.list_named_photo_paths, name, scope_roots=scope_roots) or [])
            if cancel_check():
                return None
            preview_paths = paths[:layout.max_items]
            image = ThumbnailService(qimage_cache_size=32).build_contact_sheet_qimage(
                preview_paths,
                layout.image_size,
                max_items=layout.max_items,
                columns=layout.columns,
                cancel_check=cancel_check,
            )
            if cancel_check():
                return None
            cooccurring_people: tuple[tuple[str, int], ...] = ()
            cooccurrence_method = getattr(service, "list_named_photo_cooccurrences", None)
            if callable(cooccurrence_method) and preview_paths:
                raw_people = self._call_scoped(
                    cooccurrence_method,
                    name,
                    preview_paths,
                    scope_roots=scope_roots,
                )
                normalized_people: list[tuple[str, int]] = []
                for person in raw_people or ():
                    if isinstance(person, tuple):
                        person_name = str(person[0] if person else "")
                        count = int(person[1] if len(person) > 1 else 0)
                    else:
                        person_name = str(getattr(person, "person_name", "") or "")
                        count = int(getattr(person, "photo_count", 0) or 0)
                    if person_name:
                        normalized_people.append((person_name, count))
                cooccurring_people = tuple(normalized_people)
            payload = _NameHoverPreviewPayload(image, len(preview_paths), cooccurring_people)
            return {"generation": generation, "cache_key": cache_key, "preview": payload, "photo_count": photo_count}

        job = AsyncJob(_run)

        def _completed(payload) -> None:
            if not isinstance(payload, dict) or int(payload.get("generation", -1)) != self._hover_preview_generation:
                return
            preview = payload.get("preview")
            if not isinstance(preview, _NameHoverPreviewPayload):
                preview = _NameHoverPreviewPayload(QImage(), 0, ())
            self._remember_name_hover_preview(cache_key, preview)
            self._apply_name_hover_preview(preview, generation)

        def _failed(_message: str) -> None:
            if generation == self._hover_preview_generation:
                self._hover_popup.set_preview_image(None)

        def _cancelled() -> None:
            if generation == self._hover_preview_generation:
                self._hover_popup.set_preview_image(None)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        for signal in (job.completed, job.failed, job.cancelled):
            signal.connect(lambda *_args, job=job: self._coordinated_job_ids.pop(job, None))
        self._hover_preview_job = job

        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            self._hover_preview_thread = thread
            thread.finished.connect(
                lambda thread=thread, job=job: self._on_name_hover_preview_thread_finished(job, thread)
            )

        if self.work_coordinator is None:
            _launch()
            return
        source_reads = tuple(scope_roots) if scope_roots is not None else ("/",)
        coordinated_job_id = self.work_coordinator.submit_async_job(
            JobSpec(
                "Loading saved-name preview",
                origin="People",
                foreground=False,
                io_bound=True,
                source_reads=source_reads,
                data_home_read=True,
            ),
            job,
            _launch,
        )
        manager = self.job_manager or self.work_coordinator.job_manager
        state = manager.get(coordinated_job_id)
        if state is not None and state.status in {"queued", "running", "cancelling"}:
            self._coordinated_job_ids[job] = coordinated_job_id

    def _apply_name_hover_preview(self, preview: _NameHoverPreviewPayload, generation: int) -> None:
        if generation != self._hover_preview_generation or not self._hover_name:
            return
        summary = self._name_summaries.get(self._hover_name)
        self._hover_popup.set_preview_image(preview.image if not preview.image.isNull() else None)
        self._hover_popup.set_preview_details(
            shown_count=preview.shown_count,
            photo_count=int(getattr(summary, "photo_count", 0) or 0),
            cooccurring_people=preview.cooccurring_people,
        )
        self._position_name_hover_preview()
        self._hover_popup.show()

    def _remember_name_hover_preview(self, cache_key: tuple[object, ...], preview: _NameHoverPreviewPayload) -> None:
        self._hover_preview_cache[cache_key] = preview
        self._hover_preview_cache.move_to_end(cache_key)
        while len(self._hover_preview_cache) > NAME_HOVER_PREVIEW_CACHE_SIZE:
            self._hover_preview_cache.popitem(last=False)

    def _position_name_hover_preview(self) -> None:
        if not self._hover_index.isValid():
            return
        rect = self.names_list.visualRect(self._hover_index)
        if not rect.isValid():
            return
        anchor = self.names_list.viewport().mapToGlobal(rect.topRight())
        self._hover_popup.adjustSize()
        popup_rect = self._hover_popup.frameGeometry()
        screen = self.names_list.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(anchor, popup_rect.size())
        margin = 12
        target_x = anchor.x() + margin
        target_y = anchor.y() + margin
        if target_x + popup_rect.width() > available.right() - margin:
            left_anchor = self.names_list.viewport().mapToGlobal(rect.topLeft())
            target_x = left_anchor.x() - popup_rect.width() - margin
        if target_y + popup_rect.height() > available.bottom() - margin:
            target_y = max(available.top() + margin, available.bottom() - popup_rect.height() - margin)
        self._hover_popup.move(max(available.left() + margin, target_x), target_y)

    def _hide_name_hover_preview(self, *, cancel_preview: bool) -> None:
        if cancel_preview:
            self._cancel_name_hover_preview_job()
            self._hover_preview_generation += 1
        self._hover_popup.hide()
        self._hover_name = ""
        self._hover_index = QModelIndex()

    def _cancel_name_hover_preview_job(self) -> None:
        job = self._hover_preview_job
        thread = self._hover_preview_thread
        if job is not None:
            if thread is not None:
                try:
                    if thread.isRunning() and not any(existing_thread is thread for _existing_job, existing_thread in self._retained_hover_preview_refs):
                        self._retained_hover_preview_refs.append((job, thread))
                except Exception:
                    pass
            self._cancel_operation_job(job)
        self._hover_preview_job = None
        self._hover_preview_thread = None

    def _on_name_hover_preview_thread_finished(self, job, thread) -> None:
        self._retained_hover_preview_refs = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._retained_hover_preview_refs
            if existing_thread is not thread
        ]
        if self._hover_preview_thread is thread:
            self._hover_preview_thread = None
            if self._hover_preview_job is job:
                self._hover_preview_job = None

    def _on_name_selected(self) -> None:
        name = self._current_name()
        if not name or name == self._selected_name and self._photos_thread is not None:
            return
        self._selected_name = name
        self._similar_token += 1
        for job in (self._similar_job, self._deep_similar_job):
            self._cancel_operation_job(job)
        self._clear_similar_face_results()
        self._show_named_photos()
        self._photos_token += 1
        token = self._photos_token
        self.open_in_gallery_button.setEnabled(False)
        self.find_similar_button.setEnabled(bool(name))
        self.deep_find_similar_button.setEnabled(bool(name))
        self._deep_similar_review = {}
        service = self._face_service_provider()
        scope_roots = self._scope_roots()
        self.status_label.setText(f"Loading photos for {name}...")

        def _run(progress, cancel_check):
            progress(-1, f"Loading photos for {name}...")
            paths = self._call_scoped(service.list_named_photo_paths, name, scope_roots=scope_roots)
            if cancel_check():
                return None
            return paths

        def _done(paths) -> None:
            if token != self._photos_token or name != self._selected_name or paths is None:
                return
            self.gallery.update_gallery_with_options(images=list(paths), clear_pixmaps=True, reset_scroll=True)
            self.gallery.set_empty_state(
                "No labeled photos",
                f"No visible saved face labels remain for {name}.",
                show_select_folder=False,
                show_run=False,
            )
            self.status_label.setText(f"{name} · {len(paths)} unique photo(s)")
            self.open_in_gallery_button.setEnabled(bool(paths))

        self._start_job("photos", _run, _done, lambda _message: self.status_label.setText(f"Could not load photos for {name}."))

    def _open_named_photos_in_gallery(self) -> None:
        paths = list(self.gallery.images)
        name = self._current_name() or self._selected_name
        if paths and name:
            self.open_in_gallery_requested.emit(paths, name)

    @staticmethod
    def _similar_face_key(item: FaceTileItem) -> tuple[str, int, tuple[int, int, int, int]]:
        return (
            str(item.image_path),
            int(item.face_index),
            tuple(int(value) for value in item.bbox),
        )

    def _similar_face_cache_key_for_item(
        self,
        item: FaceTileItem,
        requested_size: QSize,
    ) -> tuple[object, ...]:
        return (
            str(item.image_path),
            int(self._similar_token),
            0,
            tuple(int(value) for value in item.bbox),
            int(requested_size.width()),
            int(requested_size.height()),
            "names_similar_faces_v1",
        )

    def _similar_face_image_for_item(self, item: FaceTileItem, _requested_size: QSize) -> QImage:
        return self._similar_face_images.get(self._similar_face_key(item), QImage())

    @staticmethod
    def _load_similar_face_thumbnail(
        image_path: str,
        bbox: tuple[int, int, int, int],
    ) -> QImage:
        """Decode one bounded face crop in the search worker, never on Qt's UI thread."""

        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            return QImage()
        try:
            x1, y1, x2, y2 = (int(value) for value in bbox)
        except (TypeError, ValueError):
            return QImage()
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        width = max(2, right - left)
        height = max(2, bottom - top)
        pad_x = max(8, width // 6)
        pad_y = max(8, height // 6)
        crop_left = max(0, left - pad_x)
        crop_top = max(0, top - pad_y)
        crop_right = min(image.width(), right + pad_x)
        crop_bottom = min(image.height(), bottom + pad_y)
        if crop_right <= crop_left or crop_bottom <= crop_top:
            return QImage()
        crop = image.copy(crop_left, crop_top, crop_right - crop_left, crop_bottom - crop_top)
        scaled = crop.scaled(
            NAME_SIMILAR_FACE_TILE_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QImage(NAME_SIMILAR_FACE_TILE_SIZE, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(QColor(18, 18, 18))
        painter = QPainter(canvas)
        painter.drawImage(
            max(0, (canvas.width() - scaled.width()) // 2),
            max(0, (canvas.height() - scaled.height()) // 2),
            scaled,
        )
        painter.end()
        return canvas

    @staticmethod
    def _saved_name_for_similar_result(service, result) -> str:
        """Prefer the durable row over the query's fallback display name."""

        load_record = getattr(service, "load_face_record", None)
        if not callable(load_record):
            return ""
        try:
            record = load_record(str(result.image_path), int(result.face_index))
        except Exception:
            return ""
        return str(getattr(record, "person_name", "") or "").strip()

    def _find_similar_faces(self) -> None:
        name = self._current_name() or self._selected_name
        if not name:
            return
        self._similar_token += 1
        self._deep_similar_review = {}
        token = self._similar_token
        self.find_similar_button.setEnabled(False)
        self.status_label.setText(f"Finding faces similar to {name}...")
        service = self._face_service_provider()
        scope_roots = self._scope_roots()

        def _run(progress, cancel_check):
            progress(-1, f"Searching faces similar to {name}...")
            results = list(
                self._call_scoped(
                    service.search_by_person_name,
                    name,
                    top_k=NAME_SIMILAR_FACE_LIMIT,
                    scope_roots=scope_roots,
                    include_tiny_faces=True,
                )
                or []
            )
            thumbnails: list[tuple[object, str, QImage]] = []
            for index, result in enumerate(results, start=1):
                if cancel_check():
                    return None
                saved_name = self._saved_name_for_similar_result(service, result)
                thumbnail = self._load_similar_face_thumbnail(
                    str(result.image_path),
                    tuple(int(value) for value in result.face_bbox),
                )
                thumbnails.append((result, saved_name, thumbnail))
                progress(
                    int(index * 100 / max(1, len(results))),
                    f"Loading similar face crops {index}/{len(results)}",
                )
            return thumbnails

        def _done(matches) -> None:
            if token != self._similar_token or name != (self._current_name() or self._selected_name) or matches is None:
                return
            self._publish_similar_face_results(name, list(matches or []))
            self.find_similar_button.setEnabled(True)

        def _failed(_message: str) -> None:
            if token == self._similar_token:
                self.status_label.setText(f"Could not find faces similar to {name}.")
                self.find_similar_button.setEnabled(bool(self._current_name() or self._selected_name))

        self._start_job("similar", _run, _done, _failed)

    def _deep_find_similar_faces(self) -> None:
        name = self._current_name() or self._selected_name
        if not name:
            return
        self._similar_token += 1
        token = self._similar_token
        self.deep_find_similar_button.setEnabled(False)
        self.status_label.setText(f"Deep-searching faces for {name}...")
        service = self._face_service_provider()
        scope_roots = self._scope_roots()

        def _run(progress, cancel_check):
            expansion = self._call_scoped(
                service.deep_search_by_person_name,
                name,
                scope_roots=scope_roots,
                include_tiny_faces=True,
                progress=progress,
                cancel_check=cancel_check,
            )
            matches: list[tuple[object, str, QImage, int]] = []
            raw_matches = list(getattr(expansion, "matches", ()) or ())
            for index, match in enumerate(raw_matches, start=1):
                if cancel_check():
                    return None
                result = match.result
                thumbnail = self._load_similar_face_thumbnail(
                    str(result.image_path),
                    tuple(int(value) for value in result.face_bbox),
                )
                matches.append((result, str(match.durable_person_name or ""), thumbnail, int(match.discovery_round)))
                progress(int(index * 100 / max(1, len(raw_matches))), f"Loading deep-search face crops {index}/{len(raw_matches)}")
            return expansion, matches

        def _done(payload) -> None:
            if token != self._similar_token or name != (self._current_name() or self._selected_name) or payload is None:
                return
            expansion, matches = payload
            self._publish_deep_similar_face_results(name, expansion, list(matches or []))
            self.deep_find_similar_button.setEnabled(True)

        def _failed(_message: str) -> None:
            if token == self._similar_token:
                self.status_label.setText(f"Could not complete the deep search for {name}.")
                self.deep_find_similar_button.setEnabled(bool(self._current_name() or self._selected_name))

        self._start_job("deep_similar", _run, _done, _failed)

    def _publish_deep_similar_face_results(self, name: str, expansion, matches: list[tuple[object, str, QImage, int]]) -> None:
        self._publish_similar_face_results(name, [(result, saved_name, thumbnail) for result, saved_name, thumbnail, _round in matches])
        refs = set(getattr(expansion, "unlabeled_refs", ()) or ())
        rounds_by_ref = {
            (str(result.image_path), int(result.face_index)): int(round_number)
            for result, _saved_name, _thumbnail, round_number in matches
        }
        self._deep_similar_review = {
            "name": name,
            "refs": refs,
            "threshold": float(getattr(expansion, "threshold", 0.72) or 0.72),
        }
        self.similar_faces_summary.setText(
            f"{name} · {len(matches)} combined match(es) from {int(getattr(expansion, 'round_count', 0) or 0)} deep-search round(s). "
            f"{len(refs)} unlabeled face region(s) are preselected; deselect any you do not want to name."
        )
        for row, item in enumerate(self._similar_face_items):
            ref = self._face_ref_from_similar_item(item)
            round_number = rounds_by_ref.get(ref, 0)
            if round_number:
                item_payload = dict(item.payload) if isinstance(item.payload, dict) else {}
                item_payload["deep_round"] = round_number
                self._similar_face_items[row] = FaceTileItem(
                    image_path=item.image_path,
                    face_index=item.face_index,
                    bbox=item.bbox,
                    title=f"{item.title}\nRound {round_number}",
                    tooltip=f"{item.tooltip}\ndeep-search round={round_number}",
                    status=item.status,
                    draft_slot=item.draft_slot,
                    saved_face_index=item.saved_face_index,
                    payload=item_payload,
                )
        self.similar_faces_model.set_items(self._similar_face_items)
        self._select_all_deep_similar_faces()

    @staticmethod
    def _face_ref_from_similar_item(item: FaceTileItem | None) -> tuple[str, int] | None:
        if item is None or not str(item.image_path or "").strip():
            return None
        return (str(item.image_path), int(item.saved_face_index if item.saved_face_index >= 0 else item.face_index))

    def _select_all_deep_similar_faces(self) -> None:
        refs = set(self._deep_similar_review.get("refs", set()) or set())
        selection = self.similar_faces_list.selectionModel()
        if selection is None:
            return
        selection.clearSelection()
        for row, item in enumerate(self._similar_face_items):
            if self._face_ref_from_similar_item(item) not in refs:
                continue
            index = self.similar_faces_model.index(row, 0)
            selection.select(index, QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        self._update_similar_face_actions()

    def _apply_selected_deep_similar_faces(self) -> None:
        if self._read_only_mode:
            return
        name = str(self._deep_similar_review.get("name", "") or "").strip()
        eligible = set(self._deep_similar_review.get("refs", set()) or set())
        refs = [
            ref
            for ref in (
                self._face_ref_from_similar_item(self.similar_faces_model.item_at(index.row()))
                for index in self.similar_faces_list.selectionModel().selectedIndexes()
            )
            if ref is not None and ref in eligible
        ]
        refs = list(dict.fromkeys(refs))
        if not name or not refs:
            return
        if not confirmBox(
            "Apply deep face names",
            f"Save '{name}' on {len(refs)} selected face region(s) in {len({path for path, _index in refs})} photo(s)? "
            "Successful regions are written to XMP/EXIF before their durable labels are saved.",
            parent=self,
        ):
            return
        service = self._face_service_provider()
        threshold = float(self._deep_similar_review.get("threshold", 0.72) or 0.72)

        def _run(progress, cancel_check):
            return service.label_unlabeled_indexed_faces_with_metadata(
                name,
                refs,
                similarity_threshold=threshold,
                source="deep_name_similar",
                progress=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            affected_refs = set(getattr(result, "affected_refs", ()) or ())
            saved = len(affected_refs)
            skipped = len(getattr(result, "skipped_refs", ()) or ())
            failed = len(getattr(result, "failures", ()) or ())
            suffix = f"; skipped {skipped}" if skipped else ""
            suffix += f"; {failed} metadata failure(s)" if failed else ""
            suffix += "; cancelled after completed files" if bool(getattr(result, "cancelled", False)) else ""
            self.status_label.setText(f"Deep naming saved {saved} face region(s){suffix}. Refreshing saved names...")
            self._deep_similar_review["refs"] = eligible - affected_refs
            refreshed_items: list[FaceTileItem] = []
            for item in self._similar_face_items:
                if self._face_ref_from_similar_item(item) not in affected_refs:
                    refreshed_items.append(item)
                    continue
                refreshed_items.append(
                    FaceTileItem(
                        image_path=item.image_path,
                        face_index=item.face_index,
                        bbox=item.bbox,
                        title=item.title.replace("Unlabeled", name, 1),
                        tooltip=f"{item.tooltip}\nsaved deep name={name}",
                        status=item.status,
                        draft_slot=item.draft_slot,
                        saved_face_index=item.saved_face_index,
                        payload=item.payload,
                    )
                )
            self._similar_face_items = refreshed_items
            self.similar_faces_model.set_items(refreshed_items)
            self._update_similar_face_actions()
            self.face_labels_changed.emit()
            self.refresh_names(preserve_name=name)

        self._start_job(
            "mutation",
            _run,
            _done,
            lambda _message: self.status_label.setText("Could not save the selected deep face labels."),
            source_writes=tuple(dict.fromkeys(path for path, _face_index in refs)),
        )

    def _publish_similar_face_results(self, name: str, matches: list[tuple[object, str, QImage]]) -> None:
        items: list[FaceTileItem] = []
        images: dict[tuple[str, int, tuple[int, int, int, int]], QImage] = {}
        for result, saved_name, thumbnail in matches:
            image_path = str(getattr(result, "image_path", "") or "")
            face_index = int(getattr(result, "face_index", -1) or -1)
            bbox = tuple(int(value) for value in getattr(result, "face_bbox", ()))
            if not image_path or len(bbox) != 4:
                continue
            state = str(saved_name or "").strip() or "Unlabeled"
            score = float(getattr(result, "score", 0.0) or 0.0)
            quality = str(getattr(result, "quality_status", "") or "saved")
            item = FaceTileItem(
                image_path=image_path,
                face_index=face_index,
                bbox=bbox,
                title=f"{state}\n{score:.3f} · Face #{face_index + 1}",
                tooltip=(
                    f"{image_path}\nface #{face_index + 1}\n"
                    f"saved name={state}\nscore={score:.4f}\n"
                    f"bbox={bbox}\nquality={quality}"
                ),
                status=quality,
                saved_face_index=face_index,
                payload={"result": result, "saved_name": str(saved_name or "")},
            )
            items.append(item)
            if isinstance(thumbnail, QImage) and not thumbnail.isNull():
                images[self._similar_face_key(item)] = thumbnail
        self._similar_face_items = items
        self._similar_face_images = images
        self.similar_faces_model.set_items(items)
        self.results_stack.setCurrentWidget(self.similar_faces_panel)
        self.open_similar_faces_button.setEnabled(bool(items))
        self._update_similar_face_actions()
        self.similar_faces_summary.setText(
            f"{name} · {len(items)} matching face region(s). "
            "Each tile is a face crop; double-click a tile to open its source photo."
        )
        self.status_label.setText(f"Found {len(items)} face region(s) similar to {name}.")

    def _clear_similar_face_results(self) -> None:
        self._similar_face_items = []
        self._similar_face_images = {}
        self._deep_similar_review = {}
        if hasattr(self, "similar_faces_model"):
            self.similar_faces_model.set_items([])
        if hasattr(self, "similar_faces_summary"):
            self.similar_faces_summary.setText("Select a saved name, then use Find Similar Faces.")
        if hasattr(self, "open_similar_faces_button"):
            self.open_similar_faces_button.setEnabled(False)
        if hasattr(self, "open_selected_similar_face_button"):
            self.open_selected_similar_face_button.setEnabled(False)
        if hasattr(self, "select_all_deep_faces_button"):
            self.select_all_deep_faces_button.setEnabled(False)
        if hasattr(self, "apply_deep_faces_button"):
            self.apply_deep_faces_button.setEnabled(False)

    def _show_named_photos(self) -> None:
        self.results_stack.setCurrentWidget(self.gallery)

    def _selected_similar_face_item(self) -> FaceTileItem | None:
        index = self.similar_faces_list.currentIndex()
        return self.similar_faces_model.item_at(index.row()) if index.isValid() else None

    def _update_similar_face_actions(self) -> None:
        self.open_selected_similar_face_button.setEnabled(self._selected_similar_face_item() is not None)
        deep_refs = set(self._deep_similar_review.get("refs", set()) or set())
        selected_refs = {
            self._face_ref_from_similar_item(self.similar_faces_model.item_at(index.row()))
            for index in self.similar_faces_list.selectionModel().selectedIndexes()
        }
        self.select_all_deep_faces_button.setEnabled(bool(deep_refs) and not self._read_only_mode)
        self.apply_deep_faces_button.setEnabled(bool(deep_refs & selected_refs) and not self._read_only_mode)

    @staticmethod
    def _similar_face_context(item: FaceTileItem) -> dict[str, object]:
        payload = item.payload if isinstance(item.payload, dict) else {}
        result = payload.get("result") if isinstance(payload, dict) else None
        return {
            "face_search": {
                "face_index": int(item.face_index),
                "bbox": tuple(int(value) for value in item.bbox),
                "score": float(getattr(result, "score", 0.0) or 0.0),
                "confidence": float(getattr(result, "face_confidence", 0.0) or 0.0),
                "person_name": str(payload.get("saved_name", "") or ""),
            }
        }

    def _open_selected_similar_face_in_gallery(self) -> None:
        item = self._selected_similar_face_item()
        name = self._current_name() or self._selected_name
        if item is None or not name:
            return
        self.open_similar_faces_in_gallery_requested.emit(
            [str(item.image_path)],
            name,
            {str(item.image_path): self._similar_face_context(item)},
        )

    def _open_similar_faces_in_gallery(self) -> None:
        name = self._current_name() or self._selected_name
        if not name or not self._similar_face_items:
            return
        paths: list[str] = []
        context_by_path: dict[str, dict[str, object]] = {}
        for item in self._similar_face_items:
            path = str(item.image_path)
            if path not in context_by_path:
                paths.append(path)
                context_by_path[path] = self._similar_face_context(item)
        self.open_similar_faces_in_gallery_requested.emit(paths, name, context_by_path)

    def _selected_image_paths(self) -> list[str]:
        selected = self.gallery._selected_gallery_paths()
        if selected:
            return selected
        return list(self.gallery.model.checked_paths())

    def _names_context_menu_actions(self, _image_path: str) -> list[tuple[str, str, Callable[[], None], bool]]:
        """Provide face-safe label actions after Gallery prepares right-click selection."""

        selected_count = len(self._selected_image_paths())
        active_name = self._current_name() or self._selected_name
        busy = self._mutation_job is not None
        editable = selected_count > 0 and not busy and not self._read_only_mode
        return [
            (
                "Rename Selected…",
                "Move only the active name's face labels in the selected photos to a different saved name.",
                self._rename_selected_images,
                editable and bool(active_name),
            ),
            (
                "Unlabel Selected",
                "Remove only the active name's face labels in the selected photos. Other people are unchanged.",
                self._unlabel_selected_images,
                editable and bool(active_name),
            ),
        ]

    def _rename_selected_images(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            return
        source = self._choose_metadata_source_name(paths, preferred=self._current_name() or self._selected_name)
        if not source:
            return
        dialog = EntityPickerDialog(
            "Rename selected face labels",
            f"New name for {source} face label(s) in the selected photo(s):",
            choices=self._saved_name_choices(),
            initial=self._current_name() or self._selected_name,
            allow_create=True,
            entity_label="person name",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.selected_value()
        if not target or target.casefold() == source.casefold():
            return
        self._start_selected_image_mutation(
            operation="rename",
            source_name=source,
            target_name=target,
            image_paths=paths,
        )

    def _unlabel_selected_images(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            return
        source = self._choose_metadata_source_name(paths, preferred=self._current_name() or self._selected_name)
        if not source:
            return
        if not confirmBox(
            "Unlabel selected faces",
            (
                f"Remove {source} only from matching face(s) in {len(paths)} selected photo(s)? "
                "Other face labels in these photos will not change."
            ),
            parent=self,
        ):
            return
        self._start_selected_image_mutation(
            operation="unlabel",
            source_name=source,
            image_paths=paths,
        )

    def _choose_metadata_source_name(self, paths: list[str], *, preferred: str = "") -> str:
        """Choose a name physically present in the selected photos' regions."""

        service = self._face_service_provider()
        names_by_path: dict[str, tuple[str, ...]] = {}
        list_names = getattr(service, "face_region_names_for_paths", None)
        if not callable(list_names):
            return str(preferred or "").strip()
        try:
            names_by_path = dict(list_names(paths) or {})
        except Exception:
            names_by_path = {}
        counts: dict[str, tuple[int, int]] = {}
        for path, names in names_by_path.items():
            region_names = [str(value or "").strip() for value in names if str(value or "").strip()]
            for name in region_names:
                face_count, photo_count = counts.get(name, (0, 0))
                counts[name] = (face_count + 1, photo_count)
            for name in set(region_names):
                face_count, photo_count = counts.get(name, (0, 0))
                counts[name] = (face_count, photo_count + 1)
        preferred = str(preferred or "").strip()
        if preferred and preferred not in counts:
            # A legacy database label has no region yet. Keeping it available
            # lets the next mutation migrate it into face metadata.
            counts[preferred] = (0, 0)
        if not counts:
            return ""
        names = sorted(counts, key=lambda value: (value.casefold(), value))
        dialog = EntityPickerDialog(
            "Choose face-region name",
            "Name currently stored in selected photo regions:",
            choices=names,
            initial=preferred if preferred in names else names[0],
            allow_create=False,
            entity_label="saved name",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return ""
        selected = dialog.selected_value()
        return selected if selected.casefold() in {name.casefold() for name in names} else ""

    def _saved_name_choices(self) -> list[str]:
        return [str(entry.title) for entry in self._entries if str(entry.title or "").strip()]

    def _start_selected_image_mutation(
        self,
        *,
        operation: str,
        image_paths: list[str],
        source_name: str = "",
        target_name: str = "",
    ) -> None:
        if self._read_only_mode:
            return
        service = self._face_service_provider()
        paths = list(dict.fromkeys(str(path) for path in image_paths if str(path or "").strip()))
        if not paths:
            return
        self.status_label.setText("Saving selected face labels...")

        def _run(progress, cancel_check):
            progress(-1, "Saving selected face labels...")
            if operation == "name":
                return service.label_unlabeled_faces_in_images(target_name, paths)
            if operation == "rename":
                rename_regions = getattr(service, "rename_face_regions_in_images", None)
                return rename_regions(source_name, target_name, paths) if callable(rename_regions) else service.rename_labeled_faces_in_images(source_name, target_name, paths)
            unlabel_regions = getattr(service, "unlabel_face_regions_in_images", None)
            return unlabel_regions(source_name, paths) if callable(unlabel_regions) else service.unlabel_labeled_faces_in_images(source_name, paths)

        def _done(result) -> None:
            refs = list(getattr(result, "affected_refs", ()) or ())
            changed = len(refs) if refs else int(result or 0) if isinstance(result, int) else 0
            failures = len(getattr(result, "failures", ()) or ())
            label = {"name": "Named", "rename": "Renamed", "unlabel": "Unlabeled"}.get(operation, "Updated")
            suffix = f" ({failures} metadata failure(s))" if failures else ""
            self.status_label.setText(f"{label} {changed} face region(s){suffix}. Refreshing saved names...")
            self._selected_name = target_name if target_name else source_name
            self.face_labels_changed.emit()
            self.refresh_names(preserve_name=self._selected_name)

        self._start_job(
            "mutation",
            _run,
            _done,
            lambda _message: self.status_label.setText("Could not save the selected face labels."),
            source_writes=tuple(paths),
        )

    def _coordination_spec(
        self,
        slot: str,
        *,
        source_writes: tuple[str, ...] = (),
    ) -> JobSpec:
        labels = {
            "refresh": "Loading saved names",
            "photos": "Loading named photos",
            "similar": "Finding similar faces",
            "deep_similar": "Deep name and similar search",
            "mutation": "Saving face-region names",
        }
        scope_roots = self._scope_roots()
        source_reads = tuple(scope_roots) if scope_roots is not None else ("/",)
        mutation = str(slot) == "mutation"
        data_home_write = str(slot) in {"refresh", "mutation"}
        return JobSpec(
            labels.get(str(slot), "People workspace work"),
            origin="People",
            foreground=str(slot) in {"refresh", "similar", "deep_similar", "mutation"},
            io_bound=True,
            source_reads=source_reads if str(slot) in {"similar", "deep_similar"} else (),
            source_writes=tuple(source_writes) if mutation else (),
            data_home_read=not data_home_write,
            data_home_write=data_home_write,
            model_cache_read=str(slot) == "deep_similar",
        )

    def _cancel_operation_job(self, job: AsyncJob | None) -> None:
        if job is None:
            return
        coordinated_job_id = self._coordinated_job_ids.get(job)
        if coordinated_job_id is not None and self.work_coordinator is not None:
            self.work_coordinator.cancel(coordinated_job_id)
            return
        try:
            job.cancel()
        except Exception:
            pass

    def _start_job(
        self,
        slot: str,
        fn,
        on_completed,
        on_failed,
        *,
        source_writes: tuple[str, ...] = (),
    ) -> None:
        old_job = getattr(self, f"_{slot}_job", None)
        if old_job is not None:
            self._cancel_operation_job(old_job)
        job = AsyncJob(fn)
        setattr(self, f"_{slot}_job", job)
        job_id: int | None = None
        coordinated = self.work_coordinator is not None
        spec = self._coordination_spec(str(slot), source_writes=source_writes)
        if self.job_manager is not None and not coordinated:
            job_id = self.job_manager.register_job(
                spec.label,
                cancel_fn=job.cancel,
                origin=spec.origin,
                foreground=spec.foreground,
            )
            self._operation_job_ids[job] = job_id
            job.progress.connect(
                lambda value, text, job_id=job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=str(text),
                )
            )
        job.progress.connect(
            lambda value, text, slot=slot: self._set_operation_progress(str(slot), int(value), str(text))
        )

        def _finish_registered_job(status: str, error: str = "", *, update_ui: bool) -> None:
            current_job_id = self._operation_job_ids.pop(job, None)
            if current_job_id is not None and self.job_manager is not None:
                self.job_manager.finish(current_job_id, status=status, error=error)
            self._coordinated_job_ids.pop(job, None)
            if update_ui:
                self._finish_operation_progress(str(slot))

        if slot == "refresh":
            self.refresh_button.setEnabled(False)

        def _complete(result) -> None:
            is_current = getattr(self, f"_{slot}_job", None) is job
            _finish_registered_job("finished", update_ui=is_current)
            if not is_current:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            on_completed(result)

        def _fail(message: str) -> None:
            is_current = getattr(self, f"_{slot}_job", None) is job
            _finish_registered_job("failed", str(message), update_ui=is_current)
            if not is_current:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            on_failed(message)

        def _cancel() -> None:
            is_current = getattr(self, f"_{slot}_job", None) is job
            _finish_registered_job("cancelled", update_ui=is_current)
            if is_current:
                setattr(self, f"_{slot}_job", None)
                if slot == "refresh":
                    self.refresh_button.setEnabled(True)

        job.completed.connect(_complete)
        job.failed.connect(_fail)
        job.cancelled.connect(_cancel)
        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            setattr(self, f"_{slot}_thread", thread)
            self._operation_threads.append((job, thread))
            thread.finished.connect(
                lambda thread=thread, job=job, slot=slot: self._on_thread_finished(job, slot, thread)
            )

        if self.work_coordinator is not None:
            coordinated_job_id = self.work_coordinator.submit_async_job(spec, job, _launch)
            state = self.job_manager.get(coordinated_job_id) if self.job_manager is not None else None
            if state is not None and state.status in {"queued", "running", "cancelling"}:
                self._coordinated_job_ids[job] = coordinated_job_id
        else:
            _launch()

    def _on_thread_finished(self, job: AsyncJob, slot: str, thread) -> None:
        self._operation_threads[:] = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._operation_threads
            if existing_thread is not thread
        ]
        if getattr(self, f"_{slot}_thread", None) is thread:
            setattr(self, f"_{slot}_thread", None)
        defer_async_job_dispose(job)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready = True
        self._hide_name_hover_preview(cancel_preview=True)
        for slot in ("refresh", "photos", "similar", "deep_similar", "mutation"):
            job = getattr(self, f"_{slot}_job", None)
            self._cancel_operation_job(job)
        for _job, thread in list(self._operation_threads):
            ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
        if ready:
            if self.work_coordinator is not None:
                for coordinated_job_id in tuple(self._coordinated_job_ids.values()):
                    self.work_coordinator.finish(coordinated_job_id, status="cancelled")
            for job, _thread in list(self._operation_threads):
                defer_async_job_dispose(job)
            self._coordinated_job_ids.clear()
            self._operation_threads.clear()
        self._refresh_job = None
        self._refresh_thread = None
        self._photos_job = None
        self._photos_thread = None
        self._similar_job = None
        self._similar_thread = None
        self._deep_similar_job = None
        self._deep_similar_thread = None
        self._mutation_job = None
        self._mutation_thread = None
        for _job, thread in list(self._retained_hover_preview_refs):
            if thread is not None:
                ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
        self._retained_hover_preview_refs.clear()
        self._hover_preview_cache.clear()
        return self.gallery.shutdown_jobs(timeout_ms=timeout_ms) and ready
