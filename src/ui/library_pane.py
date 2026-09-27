from __future__ import annotations

from calendar import month_name
from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.path_scope import PathScope, normalized_path_is_within_scope, path_is_within_scope
from app.services.cluster_context import ClusterContextService, VisionLanguageSettings
from app.services.duplicate_review import DuplicateGroup, DuplicateReviewService
from app.services.library_catalog import CatalogQuery, CatalogTimeline, ClusterContextRecord, LibraryCatalogService, SmartAlbum
from ui.async_job import AsyncJob, defer_async_job_dispose, start_job_in_thread, wait_for_thread_shutdown
from ui.common import HelpIconButton, ResponsiveFlowLayout
from ui.count_copy import showing
from ui.error_mbox import confirmBox, errorBox
from ui.entity_picker import EntityPicker, EntityPickerDialog
from ui.gallery_pane import GalleryPane
from ui.icons import apply_icon
from ui.job_manager import JobManager
from ui.sectioned_gallery import GallerySection, SectionedGallery
from ui.theme import apply_field_size
from ui.work_coordinator import JobSpec, WorkCoordinator

if TYPE_CHECKING:
    from app.services.people_cleanup import PeopleCleanupGroup, PeopleCleanupService


@dataclass(frozen=True)
class TimelineDatePreferences:
    date_policy: str = "datetime_original_then_metadata"
    filename_patterns: tuple[str, ...] = ()
    raw_epoch_heuristic: bool = False
    grouping: str = "year_month"


class _LibraryStatusLabel(QLabel):
    """A status row that disappears once Library is simply ready."""

    _IDLE_TEXTS = {"Library timeline ready.", "Library catalog ready."}

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API override
        value = str(text or "")
        super().setText(value)
        self.setVisible(bool(value.strip()) and value.strip() not in self._IDLE_TEXTS)


class _DuplicateTrashPreviewDialog(QDialog):
    """Require an explicit candidate check before recoverable Trash work."""

    def __init__(self, kind_label: str, paths: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Preview {kind_label.lower()} candidates")
        self.setMinimumWidth(620)
        layout = QVBoxLayout(self)
        detail = QLabel(
            f"Review {len(paths)} proposed {kind_label.lower()} candidate(s). Proposed keepers are excluded. "
            "Nothing moves until you check files and confirm; completed moves go to recoverable ClusterLens Trash."
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.candidates = QListWidget(self)
        self.candidates.setObjectName("duplicateTrashPreviewCandidates")
        for path in paths:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            self.candidates.addItem(item)
        self.candidates.itemChanged.connect(self._update_accept_enabled)
        layout.addWidget(self.candidates, stretch=1)
        actions = QHBoxLayout()
        select_all = QPushButton("Select all")
        select_none = QPushButton("Select none")
        select_all.clicked.connect(lambda: self._set_all_checked(True))
        select_none.clicked.connect(lambda: self._set_all_checked(False))
        actions.addWidget(select_all)
        actions.addWidget(select_none)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok, self)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Move checked files to Trash")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._update_accept_enabled()

    def selected_paths(self) -> list[str]:
        return [
            str(self.candidates.item(row).data(Qt.ItemDataRole.UserRole) or "")
            for row in range(self.candidates.count())
            if self.candidates.item(row).checkState() == Qt.CheckState.Checked
        ]

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.candidates.blockSignals(True)
        try:
            for row in range(self.candidates.count()):
                self.candidates.item(row).setCheckState(state)
        finally:
            self.candidates.blockSignals(False)
        self._update_accept_enabled()

    def _update_accept_enabled(self, *_args) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(self.selected_paths()))


class LibraryPane(QWidget):
    """Local archive curation workspace backed entirely by managed runtime data."""

    open_in_gallery_requested = pyqtSignal(list, str)
    metadata_changed = pyqtSignal(list)
    paths_renamed = pyqtSignal(list)
    settings_requested = pyqtSignal(str)
    roots_changed = pyqtSignal(list)
    root_job_state_changed = pyqtSignal(str, str)
    catalog_batch_committed = pyqtSignal(object)

    PAGE_SIZE = 500
    _TIMELINE_GROUPINGS = ("year", "year_month", "year_month_week", "year_month_day")
    _TIMELINE_PRIMARY_KEYS = ("start", "end", "camera", "date_source", "grouping", "show")
    _TIMELINE_TWO_ROW_FILTER_KEYS = ("start", "end", "camera")
    _TIMELINE_TWO_ROW_POLICY_KEYS = ("date_source", "grouping", "show")
    _TIMELINE_ACTION_MIN_WIDTH = 160
    _TIMELINE_MAX_SCROLL_CONTENT_HEIGHT = 276
    _SEARCH_CONTEXT_TAB_HEIGHT = 236
    _SEARCH_ADVANCED_TAB_HEIGHT = 184
    _SEARCH_ADVANCED_CONTEXT_TAB_HEIGHT = 332
    _TAB_HEIGHTS = {
        "All Photos": 72,
        "Search": 72,
        "Albums": 72,
        "Cleanup": 322,
        "Batch Rename & Metadata": 270,
        "Recovery": 270,
        "People Cleanup": 300,
        "Cluster Context": 352,
    }

    def __init__(
        self,
        current_scope_provider: Callable[[], object],
        face_service_provider: Callable[[], object | None],
        parent=None,
        *,
        catalog: LibraryCatalogService | None = None,
        job_manager: JobManager | None = None,
        work_coordinator: WorkCoordinator | None = None,
        context_settings_provider: Callable[[], VisionLanguageSettings] | None = None,
        context_settings_changed: Callable[[VisionLanguageSettings, bool], None] | None = None,
        timeline_date_preferences_provider: Callable[[], TimelineDatePreferences] | None = None,
        timeline_date_preferences_changed: Callable[[TimelineDatePreferences], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog or LibraryCatalogService()
        self.current_scope_provider = current_scope_provider
        self.face_service_provider = face_service_provider
        self.job_manager = job_manager
        self.work_coordinator = work_coordinator
        self.context_settings_provider = context_settings_provider
        self.context_settings_changed = context_settings_changed
        self.timeline_date_preferences_provider = timeline_date_preferences_provider
        self.timeline_date_preferences_changed = timeline_date_preferences_changed
        self.context_service = ClusterContextService(self.catalog)
        self.duplicate_service = DuplicateReviewService(catalog=self.catalog)
        self._people_service: PeopleCleanupService | None = None
        self._jobs: list[tuple[AsyncJob, object]] = []
        self._coordinated_job_ids: dict[AsyncJob, int] = {}
        self._root_scan_jobs: dict[str, AsyncJob] = {}
        self._catalog_snapshot_job: AsyncJob | None = None
        self._catalog_snapshot_generation = 0
        self._catalog_scan_generation = 0
        self._catalog_committed_scan_stats: dict[str, dict[str, int]] = {}
        self._roots_cache: tuple[object, ...] = ()
        self._root_counts_cache: dict[str, int] = {}
        self._albums_cache: tuple[SmartAlbum, ...] = ()
        self._catalog_generation = 0
        self._asset_request_generation = 0
        self._asset_job: AsyncJob | None = None
        self._asset_loading = False
        self._asset_auto_loading = False
        self._timeline_request_generation = 0
        self._timeline_job: AsyncJob | None = None
        self._timeline_loading = False
        self._timeline_paths: tuple[str, ...] = ()
        self._timeline_total = 0
        self._timeline: CatalogTimeline | None = None
        self._duplicate_generation = 0
        self._duplicate_job: AsyncJob | None = None
        self._people_generation = 0
        self._people_job: AsyncJob | None = None
        self._people_name_generation = 0
        self._people_name_job: AsyncJob | None = None
        self._context_search_generation = 0
        self._context_search_job: AsyncJob | None = None
        self._timeline_loaded = False
        self._asset_offset = 0
        self._asset_total = 0
        self._active_query = CatalogQuery()
        self._duplicate_groups: list[DuplicateGroup] = []
        self._people_groups: list[PeopleCleanupGroup] = []
        self._selected_cluster_key = ""
        self._selected_cluster_members: tuple[str, ...] = ()
        self._read_only_mode = False
        self._force_catalog_date_refresh = False
        self._timeline_layout_mode = "narrow"
        self._timeline_advanced_layout_mode = "narrow"
        self._timeline_layout_signature: tuple[object, ...] = ()
        self._timeline_advanced_layout_signature: tuple[object, ...] = ()
        self._timeline_layout_timer = QTimer(self)
        self._timeline_layout_timer.setSingleShot(True)
        self._timeline_layout_timer.timeout.connect(self._update_timeline_layout)
        self.catalog_batch_committed.connect(self._on_catalog_batch_committed)
        self._suppress_surface_load = True
        self._build_ui()
        self._suppress_surface_load = False
        self._restore_context_settings()
        self._restore_timeline_date_preferences()
        QTimer.singleShot(0, self.refresh_roots)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # The production shell can show this workspace at its 720 px compact
        # width.  Give text and focus rings a protected edge rather than
        # letting controls touch the splitter/window boundary.
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        self.workspace_title = QLabel("Library")
        self.workspace_title.setProperty("role", "section")
        header.addWidget(self.workspace_title)
        self.library_help_button = HelpIconButton(
            "Library works only with explicitly registered roots. Register active roots or choose a folder, then refresh "
            "their local catalog before browsing Timeline or Search. Catalog refreshes run as visible, cancellable Jobs and "
            "read photo metadata without changing source photos. Naming faces and sending duplicate candidates to Trash are "
            "separate explicit actions.",
            self,
            help_key="library_workspace",
        )
        header.addWidget(self.library_help_button)
        header.addStretch(1)
        self.scan_button = QPushButton("Refresh registered roots")
        self.scan_button.setToolTip("Incrementally scan registered photo roots in a background Job. Source files are read only.")
        self.scan_button.clicked.connect(self.scan_roots)
        header.addWidget(self.scan_button)
        layout.addLayout(header)
        self.status_label = _LibraryStatusLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.library_splitter = splitter
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.splitterMoved.connect(lambda _position, _index: self._schedule_timeline_layout())
        sidebar = QWidget(splitter)
        self.library_sidebar = sidebar
        sidebar.setMinimumWidth(270)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(6)
        self.roots_heading = QLabel("Registered library roots")
        self.roots_heading.setToolTip("Only checked roots participate in Library-wide work.")
        side.addWidget(self.roots_heading)
        self.root_list = QListWidget(sidebar)
        self.root_list.setToolTip("Only enabled roots participate in Library, People Cleanup, duplicate review, and smart albums.")
        self.root_list.itemSelectionChanged.connect(self._root_selection_changed)
        self.root_list.itemChanged.connect(self._root_enabled_changed)
        side.addWidget(self.root_list, stretch=1)
        root_buttons = QGridLayout()
        root_buttons.setHorizontalSpacing(6)
        root_buttons.setVerticalSpacing(6)
        self.add_current_root_button = QPushButton("Register active roots")
        self.add_root_button = QPushButton("Choose folder…")
        self.remove_root_button = QPushButton("Remove selected")
        self.add_current_root_button.setToolTip("Register the currently active roots. This is explicit and starts a visible catalog refresh; selecting roots alone never catalogs photos.")
        self.add_root_button.setToolTip("Choose another folder to register. ClusterLens will only catalog this folder and its descendants.")
        self.remove_root_button.setToolTip("Remove the selected root's derived Library records. Source photos are never deleted.")
        self.add_current_root_button.setProperty("kind", "primary")
        self.add_current_root_button.clicked.connect(self.add_current_root)
        self.add_root_button.clicked.connect(self.add_root_folder)
        self.remove_root_button.clicked.connect(self.remove_selected_root)
        root_buttons.addWidget(self.add_current_root_button, 0, 0, 1, 2)
        root_buttons.addWidget(self.add_root_button, 1, 0)
        root_buttons.addWidget(self.remove_root_button, 1, 1)
        side.addLayout(root_buttons)
        # Root state and actions live in the shared Roots manager opened from
        # the shell's Edit roots button. Keep this list as an internal
        # selection model for compatibility, not as a second visible manager.
        self.roots_heading.hide()
        self.root_list.hide()
        self.add_current_root_button.hide()
        self.add_root_button.hide()
        self.remove_root_button.hide()
        self.scan_button.hide()
        albums_heading = QLabel("Smart albums")
        albums_heading.setToolTip("Saved Library filters. Albums contain no copied photos and update as the catalog changes.")
        side.addWidget(albums_heading)
        self.album_list = QListWidget(sidebar)
        self.album_list.setAccessibleName("Saved smart albums")
        self.album_list.itemSelectionChanged.connect(self._album_selection_changed)
        side.addWidget(self.album_list, stretch=1)
        album_buttons = QVBoxLayout()
        album_buttons.setSpacing(6)
        self.save_album_button = QPushButton("Save search")
        self.delete_album_button = QPushButton("Delete album")
        self.save_album_button.setAccessibleName("Save current search")
        self.delete_album_button.setAccessibleName("Delete selected album")
        self.save_album_button.setToolTip("Save the active Timeline or Search filters as a dynamic album.")
        self.delete_album_button.setToolTip("Delete only the selected saved filter; photos and catalog records stay intact.")
        self.save_album_button.clicked.connect(self.save_current_album)
        self.delete_album_button.clicked.connect(self.delete_selected_album)
        album_buttons.addWidget(self.save_album_button)
        album_buttons.addWidget(self.delete_album_button)
        side.addLayout(album_buttons)
        splitter.addWidget(sidebar)

        body = QWidget(splitter)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget(body)
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.tabs.currentChanged.connect(self._tab_changed)
        body_layout.addWidget(self.tabs)
        self._build_timeline_tab()
        self._build_all_photos_tab()
        self._build_search_tab()
        self._build_albums_tab()
        self._build_cleanup_tab()
        self._build_batch_tools_tab()
        self._build_recovery_tab()
        self._build_people_tab()
        self._build_context_tab()
        self.filter_chips_widget = QWidget(body)
        self.filter_chips_layout = QHBoxLayout(self.filter_chips_widget)
        self.filter_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.filter_chips_layout.setSpacing(6)
        self.filter_chips_label = QLabel("Active filters", self.filter_chips_widget)
        self.filter_chips_label.setProperty("role", "helper")
        self.filter_chips_layout.addWidget(self.filter_chips_label)
        self.filter_chip_buttons: dict[str, QToolButton] = {}
        for key, callback in (
            ("start", lambda: self._clear_timeline_filter(self.timeline_start)),
            ("end", lambda: self._clear_timeline_filter(self.timeline_end)),
            ("camera", lambda: self._clear_timeline_filter(self.timeline_camera)),
            ("search", self._clear_search_filter),
        ):
            button = QToolButton(self.filter_chips_widget)
            button.setAccessibleName(f"Remove {key} filter")
            button.clicked.connect(callback)
            button.hide()
            self.filter_chips_layout.addWidget(button)
            self.filter_chip_buttons[key] = button
        self.filter_chips_layout.addStretch(1)
        body_layout.addWidget(self.filter_chips_widget)
        self.gallery = GalleryPane(body)
        self.gallery.configure_jobs(self.job_manager, self.work_coordinator, origin="Library")
        self.gallery.set_action_visibility(show_actions=True, show_metadata_actions=True, show_file_actions=True)
        self.gallery.set_empty_state("Library timeline", "Register and refresh a library root to browse local photos.")
        self.gallery.metadata_changed.connect(self.metadata_changed.emit)
        self.gallery.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.timeline_gallery = SectionedGallery(body)
        self.timeline_gallery.set_job_manager(
            self.job_manager,
            origin="Library",
            work_coordinator=self.work_coordinator,
        )
        self.timeline_gallery.set_embedded_timeline_mode(True)
        self.timeline_gallery.set_empty_state("Register and refresh a library root to browse the full local timeline.")
        self.timeline_gallery.metadata_changed.connect(self.metadata_changed.emit)
        self.timeline_gallery.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.gallery_stack = QStackedWidget(body)
        self.gallery_stack.addWidget(self.timeline_gallery)
        self.gallery_stack.addWidget(self.gallery)
        body_layout.addWidget(self.gallery_stack, stretch=1)
        self.gallery_footer = QWidget(body)
        gallery_footer = QHBoxLayout(self.gallery_footer)
        gallery_footer.setContentsMargins(0, 0, 0, 0)
        self.timeline_count = QLabel("0 photos")
        self.load_more_button = QPushButton("Load more")
        self.open_gallery_button = QPushButton("Open visible photos")
        self.load_more_button.clicked.connect(self.load_more_assets)
        self.open_gallery_button.clicked.connect(self._open_visible_in_gallery)
        gallery_footer.addWidget(self.timeline_count)
        gallery_footer.addStretch(1)
        gallery_footer.addWidget(self.load_more_button)
        gallery_footer.addWidget(self.open_gallery_button)
        body_layout.addWidget(self.gallery_footer)
        splitter.addWidget(body)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter, stretch=1)
        self._surface_group = "library"
        self.set_surface_group("library", "timeline")
        self._apply_active_tab_height()
        self._update_controls()
        self._schedule_timeline_layout()

    def _build_timeline_tab(self) -> None:
        page = QWidget(self.tabs)
        self.timeline_page = page
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        self.timeline_scroll = QScrollArea(page)
        self.timeline_scroll.setObjectName("timelineControlsScrollArea")
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.timeline_form = QWidget(self.timeline_scroll)
        self.timeline_form_layout = QGridLayout(self.timeline_form)
        self.timeline_form_layout.setContentsMargins(0, 0, 0, 0)
        self.timeline_form_layout.setHorizontalSpacing(8)
        self.timeline_form_layout.setVerticalSpacing(6)
        self.timeline_scroll.setWidget(self.timeline_form)
        page_layout.addWidget(self.timeline_scroll)
        self.timeline_start = QLineEdit(page)
        self.timeline_start.setPlaceholderText("YYYY-MM-DD")
        self.timeline_start.setToolTip("Optional inclusive start date in YYYY-MM-DD format.")
        self.timeline_end = QLineEdit(page)
        self.timeline_end.setPlaceholderText("YYYY-MM-DD")
        self.timeline_end.setToolTip("Optional inclusive end date in YYYY-MM-DD format.")
        self.timeline_camera = QLineEdit(page)
        self.timeline_camera.setPlaceholderText("Camera contains")
        self.timeline_camera.setToolTip("Optional case-insensitive camera model filter.")
        self.timeline_date_source = QComboBox(page)
        self.timeline_date_source.addItem("Metadata → filename", "metadata_or_filename")
        self.timeline_date_source.addItem("DateTimeOriginal first", "datetime_original_then_metadata")
        self.timeline_date_source.addItem("Metadata only", "metadata_only")
        self.timeline_date_source.addItem("Filename first", "prefer_filename")
        self.timeline_date_source.addItem("Filename → DateTimeOriginal", "filename_only")
        current_policy = str(getattr(self.catalog, "filename_date_policy", "metadata_or_filename"))
        current_index = self.timeline_date_source.findData(current_policy)
        self.timeline_date_source.setCurrentIndex(max(0, current_index))
        self.timeline_date_source.setToolTip(
            "Choose the source used for Timeline capture time. Filename rules support safe dates, counters, and explicit Unix epochs. "
            "DateTimeOriginal first prefers the camera's original capture moment before other metadata, filename rules, and file modified time. "
            "The filename-first option uses camera DateTimeOriginal only when a filename cannot be parsed; it never substitutes file-modified time. "
            "Use Refresh dates after changing either control; it only rebuilds derived catalog values and never writes EXIF or XMP."
        )
        self.timeline_grouping = QComboBox(page)
        self.timeline_grouping.setAccessibleName("Timeline photo grouping")
        self.timeline_grouping.addItem("Year", "year")
        self.timeline_grouping.addItem("Year → Month", "year_month")
        self.timeline_grouping.addItem("Year → Month → Week", "year_month_week")
        self.timeline_grouping.addItem("Year → Month → Day", "year_month_day")
        self.timeline_grouping.setCurrentIndex(self.timeline_grouping.findData("year_month"))
        self.timeline_grouping.setToolTip(
            "Choose how virtual Timeline headers are nested. This only changes the displayed hierarchy; "
            "it does not rescan photos or change capture dates."
        )
        self.timeline_date_help_button = HelpIconButton(
            "Timeline can use EXIF capture time, an extracted filename time, or a safe fallback file-modified time. "
            "The rule list accepts one legacy strptime format or named rule per line and is evaluated in order before built-in formats. "
            "Examples: %Y%m%d_%H%M%S, {date:DDMMYYYY}{sequence:3}, IMG_Epoch_{epoch:s}, and IMG_Epoch_{epoch:ms}. "
            "A sequence orders same-day files without inventing a clock time. The raw ID/hash fallback is off by default and accepts one 10- or 13-digit Unix epoch from 1991 through the current UTC year; multiple candidates are ignored. "
            "Timeline accepts only 1991 through the current UTC year. For filename time, choose Filename first or Filename → DateTimeOriginal, then click Refresh dates. "
            "A capture time saved in the photo viewer is explicit user intent and takes precedence after Refresh dates. "
            "This action is cancellable and does not change source files, EXIF, or XMP.",
            page,
            help_key="timeline_filename_dates",
        )
        self.timeline_filename_patterns = QPlainTextEdit(page)
        self.timeline_filename_patterns.setObjectName("timelineFilenameDatePatterns")
        self.timeline_filename_patterns.setPlainText("\n".join(self.catalog.filename_date_patterns))
        self.timeline_filename_patterns.setPlaceholderText("One strptime format or named filename-time rule per line")
        self.timeline_filename_patterns.setTabChangesFocus(True)
        self.timeline_filename_patterns.setFixedHeight(62)
        self.timeline_filename_patterns.setMinimumWidth(260)
        self.timeline_filename_patterns.setToolTip(
            "One rule per line, tried from top to bottom. Legacy rules use %Y%m%d with optional %H%M or %H%M%S. "
            "Named examples: {date:DDMMYYYY}{sequence:3}, IMG_Epoch_{epoch:s}, IMG_Epoch_{epoch:ms}."
        )
        self.timeline_epoch_heuristic = QCheckBox("Recognize raw Unix epochs in IDs / hashes", page)
        self.timeline_epoch_heuristic.setToolTip(
            "Off by default. When enabled, a filename with exactly one valid 10-digit seconds or 13-digit milliseconds Unix epoch run is dated in UTC. "
            "The run may be surrounded by UUID/hash letters or punctuation. Multiple valid runs are ignored."
        )
        self.timeline_reload_button = QPushButton("Show timeline")
        self.timeline_reload_button.setToolTip("Apply the date and camera filters to the selected Library root.")
        self.timeline_reload_button.setProperty("kind", "primary")
        self.timeline_refresh_dates_button = QPushButton("Refresh dates")
        self.timeline_refresh_dates_button.setToolTip(
            "Rebuild derived Timeline dates for active registered roots using the selected source and filename patterns. "
            "Runs as a cancellable Job and never changes source files or metadata."
        )
        apply_icon(self.timeline_refresh_dates_button, "retry")
        self.timeline_reload_button.clicked.connect(self.load_timeline)
        self.timeline_refresh_dates_button.clicked.connect(self.refresh_timeline_dates)
        self.timeline_date_source.currentIndexChanged.connect(self._on_timeline_date_source_changed)
        self.timeline_grouping.currentIndexChanged.connect(self._on_timeline_grouping_changed)
        self.timeline_filename_patterns.textChanged.connect(self._on_timeline_filename_patterns_changed)
        self.timeline_epoch_heuristic.toggled.connect(self._on_timeline_epoch_heuristic_changed)
        apply_field_size(self.timeline_start, "short")
        apply_field_size(self.timeline_end, "short")
        apply_field_size(self.timeline_camera, "medium")
        self._timeline_cells = {
            "start": self._timeline_field_cell("From", self.timeline_start),
            "end": self._timeline_field_cell("To", self.timeline_end),
            "camera": self._timeline_field_cell("Camera", self.timeline_camera),
            "date_source": self._timeline_field_cell(
                "Timeline date", self.timeline_date_source, help_button=self.timeline_date_help_button
            ),
            "grouping": self._timeline_field_cell("Group photos", self.timeline_grouping),
            "patterns": self._timeline_field_cell("Filename time rules", self.timeline_filename_patterns),
            "epoch": self._timeline_field_cell("Optional recognition", self.timeline_epoch_heuristic),
            "refresh": self._timeline_field_cell("Derived dates", self.timeline_refresh_dates_button),
            "show": self._timeline_field_cell("Apply filters", self.timeline_reload_button),
        }
        self.timeline_advanced_cell = QWidget(self.timeline_form)
        advanced_layout = QVBoxLayout(self.timeline_advanced_cell)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)
        self.timeline_advanced_toggle = QToolButton(self.timeline_advanced_cell)
        self.timeline_advanced_toggle.setText("Advanced date parsing")
        self.timeline_advanced_toggle.setCheckable(True)
        self.timeline_advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.timeline_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.timeline_advanced_toggle.setAccessibleName("Show advanced date parsing controls")
        self.timeline_advanced_toggle.setToolTip(
            "Show filename rules, raw epoch recognition, and derived-date rebuilding. These controls never write source metadata."
        )
        advanced_layout.addWidget(self.timeline_advanced_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.timeline_advanced_body = QWidget(self.timeline_advanced_cell)
        self.timeline_advanced_layout = QGridLayout(self.timeline_advanced_body)
        self.timeline_advanced_layout.setContentsMargins(12, 0, 0, 0)
        self.timeline_advanced_layout.setHorizontalSpacing(8)
        self.timeline_advanced_layout.setVerticalSpacing(6)
        self.timeline_advanced_body.hide()
        advanced_layout.addWidget(self.timeline_advanced_body)
        self.timeline_advanced_toggle.toggled.connect(self._set_timeline_advanced_visible)
        self._timeline_cells["advanced"] = self.timeline_advanced_cell
        self._refresh_timeline_advanced_layout(force=True)
        self.setTabOrder(self.timeline_start, self.timeline_end)
        self.setTabOrder(self.timeline_end, self.timeline_camera)
        self.setTabOrder(self.timeline_camera, self.timeline_date_source)
        self.setTabOrder(self.timeline_date_source, self.timeline_date_help_button)
        self.setTabOrder(self.timeline_date_help_button, self.timeline_grouping)
        self.setTabOrder(self.timeline_grouping, self.timeline_filename_patterns)
        self.setTabOrder(self.timeline_filename_patterns, self.timeline_epoch_heuristic)
        self.setTabOrder(self.timeline_epoch_heuristic, self.timeline_refresh_dates_button)
        self.setTabOrder(self.timeline_refresh_dates_button, self.timeline_reload_button)
        self.tabs.addTab(page, "Timeline")
        self._refresh_timeline_layout()

    def _set_timeline_advanced_visible(self, visible: bool) -> None:
        self.timeline_advanced_body.setVisible(bool(visible))
        self.timeline_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self._refresh_timeline_layout(force=True)
        self._apply_active_tab_height()

    def _build_all_photos_tab(self) -> None:
        page = QWidget(self.tabs)
        self.all_photos_page = page
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        summary = QLabel("Browse every catalogued photo in the active registered roots.", page)
        summary.setWordWrap(True)
        self.show_all_photos_button = QPushButton("Show all photos", page)
        self.show_all_photos_button.setProperty("kind", "primary")
        self.show_all_photos_button.setToolTip(
            "Load catalogued photos in cancellable 500-photo pages without rescanning source folders."
        )
        self.show_all_photos_button.clicked.connect(self.load_all_photos)
        layout.addWidget(summary, stretch=1)
        layout.addWidget(self.show_all_photos_button)
        self.tabs.addTab(page, "All Photos")

    def load_all_photos(self) -> None:
        self.search_field.clear()
        self.run_search()

    def _build_albums_tab(self) -> None:
        page = QWidget(self.tabs)
        self.albums_page = page
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        summary = QLabel(
            "Choose a saved album from the left. Albums are saved filters; they never copy or move photos.",
            page,
        )
        summary.setWordWrap(True)
        self.save_album_from_tab_button = QPushButton("Save search", page)
        self.save_album_from_tab_button.setAccessibleName("Save current search")
        self.save_album_from_tab_button.setToolTip("Save the current Timeline or Search filters as a dynamic album.")
        self.save_album_from_tab_button.clicked.connect(self.save_current_album)
        layout.addWidget(summary, stretch=1)
        layout.addWidget(self.save_album_from_tab_button)
        self.tabs.addTab(page, "Albums")

    def _timeline_field_cell(self, label: str, field: QWidget, *, help_button: QWidget | None = None) -> QWidget:
        """Build one labeled Timeline control cell for the responsive grid."""

        cell = QWidget(self.timeline_form)
        cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label, cell)
        title.setProperty("role", "helper")
        title.setBuddy(field)
        label_row.addWidget(title)
        label_row.addStretch(1)
        layout.addLayout(label_row)
        field_row = QHBoxLayout()
        field_row.setContentsMargins(0, 0, 0, 0)
        field_row.setSpacing(4)
        field_row.addWidget(field, stretch=1)
        if help_button is not None:
            field_row.addWidget(help_button)
        layout.addLayout(field_row)
        return cell

    @staticmethod
    def _layout_required_width(widths: tuple[int, ...], spacing: int) -> int:
        return sum(widths) + max(0, len(widths) - 1) * max(0, int(spacing))

    def _timeline_usable_width(self) -> int:
        if not hasattr(self, "timeline_scroll"):
            return 0
        margins = self.timeline_form_layout.contentsMargins()
        return max(0, self.timeline_scroll.viewport().width() - margins.left() - margins.right())

    def _timeline_advanced_usable_width(self) -> int:
        """Return the actual body width available after its own indent."""

        margins = self.timeline_advanced_layout.contentsMargins()
        return max(0, self._timeline_usable_width() - margins.left() - margins.right())

    def _timeline_cell_minimum_width(self, key: str) -> int:
        cell = self._timeline_cells[key]
        width = max(1, int(cell.minimumSizeHint().width()))
        if key == "show":
            return max(width, self._TIMELINE_ACTION_MIN_WIDTH)
        return width

    def _timeline_primary_widths(self) -> dict[str, int]:
        return {key: self._timeline_cell_minimum_width(key) for key in self._TIMELINE_PRIMARY_KEYS}

    def _timeline_layout_for_width(self, width: int, widths: dict[str, int] | None = None) -> str:
        """Choose rows from actual control requirements, not screen constants."""

        measured = widths or self._timeline_primary_widths()
        spacing = max(0, self.timeline_form_layout.horizontalSpacing())
        one_row = self._layout_required_width(
            tuple(measured[key] for key in self._TIMELINE_PRIMARY_KEYS), spacing
        )
        two_rows = max(
            self._layout_required_width(tuple(measured[key] for key in self._TIMELINE_TWO_ROW_FILTER_KEYS), spacing),
            self._layout_required_width(tuple(measured[key] for key in self._TIMELINE_TWO_ROW_POLICY_KEYS), spacing),
        )
        if int(width) >= one_row:
            return "wide"
        if int(width) >= two_rows:
            return "medium"
        return "narrow"

    def _timeline_advanced_widths(self) -> dict[str, int]:
        return {
            key: self._timeline_cell_minimum_width(key)
            for key in ("patterns", "epoch", "refresh")
        }

    def _timeline_advanced_layout_for_width(self, width: int, widths: dict[str, int]) -> str:
        spacing = max(0, self.timeline_advanced_layout.horizontalSpacing())
        wide = self._layout_required_width((widths["patterns"], widths["epoch"], widths["refresh"]), spacing)
        medium = max(
            widths["patterns"],
            self._layout_required_width((widths["epoch"], widths["refresh"]), spacing),
        )
        if int(width) >= wide:
            return "wide"
        if int(width) >= medium:
            return "medium"
        return "narrow"

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            layout.takeAt(0)

    def _refresh_timeline_advanced_layout(self, *, force: bool = False) -> bool:
        if not hasattr(self, "timeline_advanced_layout"):
            return False
        widths = self._timeline_advanced_widths()
        mode = self._timeline_advanced_layout_for_width(self._timeline_advanced_usable_width(), widths)
        signature = (mode, tuple(sorted(widths.items())))
        if not force and signature == self._timeline_advanced_layout_signature:
            return False
        self._timeline_advanced_layout_signature = signature
        self._timeline_advanced_layout_mode = mode
        layout = self.timeline_advanced_layout
        self._clear_layout(layout)
        for column in range(3):
            layout.setColumnMinimumWidth(column, 0)
            layout.setColumnStretch(column, 0)
        if mode == "wide":
            for column, key in enumerate(("patterns", "epoch", "refresh")):
                layout.addWidget(self._timeline_cells[key], 0, column, alignment=Qt.AlignmentFlag.AlignTop)
                layout.setColumnMinimumWidth(column, widths[key])
            layout.setColumnStretch(0, 2)
            layout.setColumnStretch(1, 1)
            layout.setColumnStretch(2, 1)
        elif mode == "medium":
            layout.addWidget(self._timeline_cells["patterns"], 0, 0, 1, 2, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["epoch"], 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["refresh"], 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
            layout.setColumnMinimumWidth(0, widths["epoch"])
            layout.setColumnMinimumWidth(1, widths["refresh"])
            layout.setColumnStretch(0, 2)
            layout.setColumnStretch(1, 1)
        else:
            for row, key in enumerate(("patterns", "epoch", "refresh")):
                layout.addWidget(self._timeline_cells[key], row, 0, alignment=Qt.AlignmentFlag.AlignTop)
            layout.setColumnMinimumWidth(0, max(widths.values()))
            layout.setColumnStretch(0, 1)
        layout.invalidate()
        self.timeline_advanced_body.updateGeometry()
        return True

    def _refresh_timeline_layout(self, *, force: bool = False) -> bool:
        """Reflow Timeline controls from their current measured width contract."""

        if not hasattr(self, "timeline_scroll"):
            return False
        widths = self._timeline_primary_widths()
        mode = self._timeline_layout_for_width(self._timeline_usable_width(), widths)
        advanced_changed = self._refresh_timeline_advanced_layout(force=force)
        signature = (mode, tuple(sorted(widths.items())), self._timeline_advanced_layout_mode)
        if not force and signature == self._timeline_layout_signature and self.timeline_form_layout.count():
            return advanced_changed
        self._timeline_layout_signature = signature
        self._timeline_layout_mode = mode
        layout = self.timeline_form_layout
        self._clear_layout(layout)
        for column in range(6):
            layout.setColumnMinimumWidth(column, 0)
            layout.setColumnStretch(column, 0)
        if mode == "wide":
            for column, key in enumerate(self._TIMELINE_PRIMARY_KEYS):
                layout.addWidget(self._timeline_cells[key], 0, column, alignment=Qt.AlignmentFlag.AlignTop)
                layout.setColumnMinimumWidth(column, widths[key])
            for column, stretch in enumerate((0, 0, 2, 2, 0, 1)):
                layout.setColumnStretch(column, stretch)
            advanced_row = 1
            advanced_span = 6
        elif mode == "medium":
            layout.addWidget(self._timeline_cells["start"], 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["end"], 0, 1, alignment=Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["camera"], 0, 2, 1, 4, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["date_source"], 1, 0, 1, 3, Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["grouping"], 1, 3, alignment=Qt.AlignmentFlag.AlignTop)
            layout.addWidget(self._timeline_cells["show"], 1, 4, 1, 2, Qt.AlignmentFlag.AlignTop)
            layout.setColumnMinimumWidth(0, widths["start"])
            layout.setColumnMinimumWidth(1, widths["end"])
            layout.setColumnMinimumWidth(3, widths["grouping"])
            layout.setColumnMinimumWidth(4, widths["show"])
            for column, stretch in enumerate((0, 0, 2, 0, 1, 1)):
                layout.setColumnStretch(column, stretch)
            advanced_row = 2
            advanced_span = 6
        else:
            for row, key in enumerate(self._TIMELINE_PRIMARY_KEYS):
                layout.addWidget(self._timeline_cells[key], row, 0, alignment=Qt.AlignmentFlag.AlignTop)
            layout.setColumnMinimumWidth(0, max(widths.values()))
            layout.setColumnStretch(0, 1)
            advanced_row = len(self._TIMELINE_PRIMARY_KEYS)
            advanced_span = 1
        layout.addWidget(
            self._timeline_cells["advanced"],
            advanced_row,
            0,
            1,
            advanced_span,
            Qt.AlignmentFlag.AlignTop,
        )
        layout.invalidate()
        self.timeline_form.updateGeometry()
        return True

    def _schedule_timeline_layout(self) -> None:
        if not self._timeline_layout_timer.isActive():
            self._timeline_layout_timer.start(0)

    def _update_timeline_layout(self) -> None:
        changed = self._refresh_timeline_layout()
        if changed and self.tabs.currentWidget() is not None and self.tabs.tabText(self.tabs.currentIndex()) == "Timeline":
            self._apply_active_tab_height()

    def _build_search_tab(self) -> None:
        page = QWidget(self.tabs)
        self.search_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.search_field = QLineEdit(page)
        self.search_field.setPlaceholderText("Search filename, camera, EXIF, XMP, or generated cluster context")
        self.search_field.setToolTip("Search the selected registered root's filename, camera, EXIF, XMP, and saved cluster contexts.")
        self.search_field.returnPressed.connect(self.run_search)
        self.search_button = QPushButton("Search")
        self.search_button.setToolTip("Run the text search in a background Job.")
        self.search_button.setProperty("kind", "primary")
        self.search_button.clicked.connect(self.run_search)
        row.addWidget(self.search_field, stretch=1)
        row.addWidget(self.search_button)
        layout.addLayout(row)
        self.search_advanced_toggle = QToolButton(page)
        self.search_advanced_toggle.setText("Filters")
        self.search_advanced_toggle.setCheckable(True)
        self.search_advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.search_advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.search_advanced_toggle.setAccessibleName("Show search filters")
        self.search_advanced_toggle.setToolTip(
            "Filter the current search by capture date, camera, folder, or file extension. Filters query the local catalog only."
        )
        self.search_advanced_toggle.toggled.connect(self._set_search_filters_visible)
        layout.addWidget(self.search_advanced_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.search_advanced_filters = QWidget(page)
        filters = QFormLayout(self.search_advanced_filters)
        filters.setContentsMargins(0, 0, 0, 0)
        filters.setHorizontalSpacing(8)
        filters.setVerticalSpacing(5)
        self.search_start = QLineEdit(self.search_advanced_filters)
        self.search_start.setPlaceholderText("YYYY-MM-DD")
        self.search_start.setToolTip("Include photos captured on or after this local-catalog timestamp.")
        self.search_end = QLineEdit(self.search_advanced_filters)
        self.search_end.setPlaceholderText("YYYY-MM-DD")
        self.search_end.setToolTip("Include photos captured on or before this local-catalog timestamp.")
        self.search_camera = QLineEdit(self.search_advanced_filters)
        self.search_camera.setPlaceholderText("Make or model")
        self.search_camera.setToolTip("Match a camera make or model stored in the local catalog.")
        self.search_folder = QLineEdit(self.search_advanced_filters)
        self.search_folder.setPlaceholderText("Folder path")
        self.search_folder.setToolTip("Limit results to this folder and its subfolders within the active roots.")
        self.search_file_ext = QLineEdit(self.search_advanced_filters)
        self.search_file_ext.setPlaceholderText(".jpg")
        self.search_file_ext.setToolTip("Match one file extension, for example .jpg or .png.")
        for field, width in (
            (self.search_start, "short"),
            (self.search_end, "short"),
            (self.search_camera, "medium"),
            (self.search_file_ext, "short"),
        ):
            apply_field_size(field, width)
            field.returnPressed.connect(self.run_search)
        self.search_folder.returnPressed.connect(self.run_search)
        filters.addRow("From", self.search_start)
        filters.addRow("To", self.search_end)
        filters.addRow("Camera", self.search_camera)
        filters.addRow("Folder", self.search_folder)
        filters.addRow("File type", self.search_file_ext)
        self.search_advanced_filters.setVisible(False)
        layout.addWidget(self.search_advanced_filters)
        self.context_results_panel = QWidget(page)
        context_layout = QVBoxLayout(self.context_results_panel)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(6)
        self.context_results = QListWidget(self.context_results_panel)
        self.context_results.setAccessibleName("Cluster context results")
        self.context_results.setMaximumHeight(126)
        self.context_results.setToolTip("Generated cluster-context matches. Select one to open the current cluster members as a photo set.")
        self.context_results.itemDoubleClicked.connect(lambda _item: self.open_selected_context())
        self.context_results.itemSelectionChanged.connect(self._update_context_result_controls)
        context_layout.addWidget(self.context_results)
        self.open_context_button = QPushButton("Open selected context cluster", self.context_results_panel)
        self.open_context_button.setToolTip("Open the photos belonging to the selected generated context.")
        self.open_context_button.clicked.connect(self.open_selected_context)
        self.open_context_button.setEnabled(False)
        context_layout.addWidget(self.open_context_button)
        self.context_results_panel.setVisible(False)
        layout.addWidget(self.context_results_panel)
        self.tabs.addTab(page, "Search")

    def _set_context_results_panel_visible(self, visible: bool) -> None:
        """Show generated-context controls only when a search actually found them."""

        self.context_results_panel.setVisible(visible)
        self._update_context_result_controls()
        if self.tabs.count() and self.tabs.tabText(self.tabs.currentIndex()) == "Search":
            self._apply_active_tab_height()

    def _set_search_filters_visible(self, visible: bool) -> None:
        self.search_advanced_filters.setVisible(bool(visible))
        self.search_advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        if self.tabs.count() and self.tabs.tabText(self.tabs.currentIndex()) == "Search":
            self._apply_active_tab_height()

    def _update_context_result_controls(self) -> None:
        self.open_context_button.setEnabled(
            self.context_results_panel.isVisible() and self.context_results.currentItem() is not None
        )

    def _build_cleanup_tab(self) -> None:
        page = QWidget(self.tabs)
        self.cleanup_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.duplicate_scan_actions = QWidget(page)
        scan_actions = ResponsiveFlowLayout(self.duplicate_scan_actions, spacing=6)
        self.find_exact_duplicates_button = QPushButton("Find hash duplicates")
        self.find_exact_duplicates_button.setToolTip("Hash only same-size files with SHA-256. This does not load perceptual hashes, embeddings, or move files.")
        self.find_exact_duplicates_button.setProperty("kind", "primary")
        self.find_exact_duplicates_button.clicked.connect(lambda: self.find_duplicates(("exact",)))
        self.find_similar_duplicates_button = QPushButton("Find similar photos")
        self.find_similar_duplicates_button.setToolTip("Review near matches where at least two of pHash, dHash, and wHash agree. Nothing moves automatically.")
        self.find_similar_duplicates_button.clicked.connect(lambda: self.find_duplicates(("near",)))
        self.find_duplicates_button = QPushButton("Find both")
        self.find_duplicates_button.setToolTip("Run exact SHA-256 and conservative similar-photo review together in one cancellable Job.")
        self.find_duplicates_button.clicked.connect(lambda: self.find_duplicates(("exact", "near")))
        self.find_bursts_button = QPushButton("Find bursts")
        self.find_bursts_button.setToolTip("Advanced review: find short capture-time sequences with a visual-similarity check. Nothing moves automatically.")
        self.find_bursts_button.clicked.connect(lambda: self.find_duplicates(("burst",)))
        self.not_duplicate_button = QPushButton("Mark selected not duplicate")
        self.not_duplicate_button.setToolTip("Record that the selected proposed group is not a duplicate. It is reconsidered only after a source revision.")
        self.not_duplicate_button.clicked.connect(self.mark_selected_not_duplicate)
        self.open_duplicate_group_button = QPushButton("Open selected photos")
        self.open_duplicate_group_button.setToolTip(
            "Open every photo in the selected review group in the main gallery to inspect it at full size before acting."
        )
        self.open_duplicate_group_button.clicked.connect(self.open_selected_duplicate_group)
        self.trash_exact_duplicates_button = QPushButton("Review all hash candidates…")
        self.trash_exact_duplicates_button.setToolTip("Preview unchecked exact-byte candidates across every hash group in this review, then move only checked files to recoverable Trash.")
        self.trash_exact_duplicates_button.setProperty("kind", "danger")
        self.trash_exact_duplicates_button.clicked.connect(lambda: self.trash_duplicate_kind("exact"))
        self.trash_similar_duplicates_button = QPushButton("Review all similar candidates…")
        self.trash_similar_duplicates_button.setToolTip("Preview unchecked similar-photo candidates across every similar-photo group in this review, then move only checked files to recoverable Trash.")
        self.trash_similar_duplicates_button.setProperty("kind", "danger")
        self.trash_similar_duplicates_button.clicked.connect(lambda: self.trash_duplicate_kind("near"))
        # Compatibility alias for callers that still ask for the selected
        # group's action. The visible controls are deliberately batch-by-kind.
        self.trash_duplicates_button = self.trash_similar_duplicates_button
        for button in (
            self.find_exact_duplicates_button,
            self.find_similar_duplicates_button,
            self.find_duplicates_button,
            self.find_bursts_button,
        ):
            scan_actions.addWidget(button)
        layout.addWidget(self.duplicate_scan_actions)

        # Review actions are meaningful only after a result group is selected.
        # Keeping them out of the idle layout gives wide screens one concise
        # scan row, while the same flow layout wraps naturally on narrow ones.
        self.duplicate_selection_actions = QWidget(page)
        selection_actions = ResponsiveFlowLayout(self.duplicate_selection_actions, spacing=6)
        for button in (
            self.open_duplicate_group_button,
            self.not_duplicate_button,
            self.trash_exact_duplicates_button,
            self.trash_similar_duplicates_button,
        ):
            selection_actions.addWidget(button)
        self.duplicate_selection_actions.setVisible(False)
        layout.addWidget(self.duplicate_selection_actions)
        self.duplicate_list = QListWidget(page)
        self.duplicate_list.setAccessibleName("Duplicate review groups")
        self.duplicate_list.setToolTip("Exact hash, similar-photo, and burst groups. Proposed keepers are never moved automatically.")
        self.duplicate_list.itemSelectionChanged.connect(self._duplicate_selection_changed)
        self.duplicate_list.setMaximumHeight(126)
        layout.addWidget(self.duplicate_list)
        self.duplicate_detail = QLabel("Run duplicate review to prepare manual cleanup candidates.")
        self.duplicate_detail.setWordWrap(True)
        layout.addWidget(self.duplicate_detail)
        self.duplicate_preview_heading = QLabel("Selected-group preview", page)
        self.duplicate_preview_heading.setProperty("role", "helper")
        layout.addWidget(self.duplicate_preview_heading)
        self.duplicate_preview_gallery = SectionedGallery(page)
        self.duplicate_preview_gallery.setObjectName("duplicatePreviewGallery")
        self.duplicate_preview_gallery.set_job_manager(
            self.job_manager,
            origin="Tools",
            work_coordinator=self.work_coordinator,
        )
        self.duplicate_preview_gallery.set_embedded_review_mode(True)
        self.duplicate_preview_gallery.setMinimumHeight(232)
        self.duplicate_preview_gallery.set_empty_state("Select a duplicate group to load its virtual thumbnail preview.")
        layout.addWidget(self.duplicate_preview_gallery, stretch=1)
        self.tabs.addTab(page, "Cleanup")

    def _build_batch_tools_tab(self) -> None:
        page = QWidget(self.tabs)
        self.batch_tools_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        detail = QLabel(
            "Select photos below, then preview a rename or exchange editable metadata through XMP sidecars. "
            "Rename previews are reversible until confirmed; metadata imports show their scope before writing.",
            page,
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.batch_tool_actions = QWidget(page)
        actions = ResponsiveFlowLayout(self.batch_tool_actions, spacing=6)
        self.batch_rename_button = QPushButton("Preview batch rename…", page)
        self.batch_rename_button.setProperty("kind", "primary")
        self.batch_rename_button.setToolTip("Build and review new filenames for the selected photos before changing any file.")
        self.export_sidecars_button = QPushButton("Export XMP sidecars…", page)
        self.export_sidecars_button.setToolTip("Export editable metadata for the selected photos without changing originals.")
        self.import_sidecars_button = QPushButton("Import XMP sidecars…", page)
        self.import_sidecars_button.setToolTip("Preview and import sidecar metadata for the selected photos.")
        self.batch_rename_button.clicked.connect(lambda: self.gallery.slotPreviewBatchRename())
        self.export_sidecars_button.clicked.connect(lambda: self.gallery.slotExportMetadataSidecars())
        self.import_sidecars_button.clicked.connect(lambda: self.gallery.slotImportMetadataSidecars())
        actions.addWidget(self.batch_rename_button)
        actions.addWidget(self.export_sidecars_button)
        actions.addWidget(self.import_sidecars_button)
        layout.addWidget(self.batch_tool_actions)
        layout.addStretch(1)
        self.tabs.addTab(page, "Batch Rename & Metadata")

    def _build_recovery_tab(self) -> None:
        page = QWidget(self.tabs)
        self.recovery_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        detail = QLabel(
            "Review recoverable Trash, safety checkpoints, storage, and backup locations in Settings. "
            "Source-photo deletion is never performed from this page.",
            page,
        )
        detail.setWordWrap(True)
        layout.addWidget(detail)
        self.recovery_actions = QWidget(page)
        actions = ResponsiveFlowLayout(self.recovery_actions, spacing=6)
        self.open_recovery_settings_button = QPushButton("Open Safety && Recovery…", page)
        self.open_storage_settings_button = QPushButton("Open Data Home && Storage…", page)
        self.open_recovery_settings_button.setProperty("kind", "primary")
        self.open_recovery_settings_button.clicked.connect(lambda: self.settings_requested.emit("Safety & Recovery"))
        self.open_storage_settings_button.clicked.connect(lambda: self.settings_requested.emit("Storage"))
        actions.addWidget(self.open_recovery_settings_button)
        actions.addWidget(self.open_storage_settings_button)
        layout.addWidget(self.recovery_actions)
        layout.addStretch(1)
        self.tabs.addTab(page, "Recovery")

    def _build_people_tab(self) -> None:
        page = QWidget(self.tabs)
        self.people_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.load_people_button = QPushButton("Build People Cleanup Inbox")
        self.load_people_button.setToolTip("Group unlabelled faces from registered roots. This does not name faces until Apply name is used.")
        self.load_people_button.setProperty("kind", "primary")
        self.load_people_button.clicked.connect(self.load_people_inbox)
        layout.addWidget(self.load_people_button)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_label = QLabel("Name")
        self.people_name_field = EntityPicker(page, allow_create=True, entity_label="person name")
        self.people_name_field.setPlaceholderText("Name selected group")
        self.people_name_field.setToolTip("Enter the durable name to apply to every face in the selected review group.")
        self.people_apply_button = QPushButton("Apply name")
        self.people_apply_button.setToolTip("Write the selected group's face-region names and update durable face labels in a background Job.")
        self.people_reject_button = QPushButton("Reject suggestion")
        self.people_reject_button.setToolTip("Do not use the suggested existing name for this group. The photos are not changed.")
        self.people_split_button = QPushButton("Split group")
        self.people_split_button.setToolTip("Keep every face in this group separate in future People Cleanup suggestions until explicitly changed.")
        self.people_hide_button = QPushButton("Hide faces")
        self.people_hide_button.setToolTip("Hide selected false detections from the local face index. This does not delete source photos.")
        self.people_merge_button = QPushButton("Merge names…")
        self.people_merge_button.setToolTip("Merge one existing durable name into another across the registered face index.")
        self.people_apply_button.clicked.connect(self.apply_people_name)
        self.people_reject_button.clicked.connect(self.reject_people_suggestion)
        self.people_split_button.clicked.connect(self.split_people_group)
        self.people_hide_button.clicked.connect(self.hide_people_group)
        self.people_merge_button.clicked.connect(self.merge_people_names)
        self.people_name_field.textChanged.connect(lambda _text: self._update_controls())
        name_row.addWidget(name_label)
        name_row.addWidget(self.people_name_field, stretch=1)
        name_row.addWidget(self.people_apply_button)
        layout.addLayout(name_row)
        self.people_review_actions = QWidget(page)
        review_actions = ResponsiveFlowLayout(self.people_review_actions, spacing=6)
        review_actions.addWidget(self.people_reject_button)
        review_actions.addWidget(self.people_split_button)
        review_actions.addWidget(self.people_hide_button)
        review_actions.addWidget(self.people_merge_button)
        layout.addWidget(self.people_review_actions)
        self.people_summary = QLabel("Open Faces once, then build a registered-root People Cleanup Inbox.")
        self.people_summary.setWordWrap(True)
        layout.addWidget(self.people_summary)
        self.people_list = QListWidget(page)
        self.people_list.setAccessibleName("People cleanup groups")
        self.people_list.itemSelectionChanged.connect(self._people_selection_changed)
        layout.addWidget(self.people_list)
        self.tabs.addTab(page, "People Cleanup")

    def _build_context_tab(self) -> None:
        page = QWidget(self.tabs)
        self.context_page = page
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        self.context_provider = QComboBox(page)
        self.context_provider.addItem("Local Ollama", "ollama")
        self.context_provider.addItem("OpenAI-compatible endpoint", "openai-compatible")
        self.context_model = QLineEdit(page)
        self.context_model.setPlaceholderText("Vision model name")
        self.context_endpoint = QLineEdit(page)
        self.context_endpoint.setPlaceholderText("Ollama base URL or OpenAI chat-completions URL")
        self.context_key_environment = QLineEdit(page)
        self.context_key_environment.setPlaceholderText("CLUSTERLENS_LLM_API_KEY")
        self.context_auto = QCheckBox("Automatically describe newly completed clusters", page)
        self.context_remote_consent = QCheckBox("I consent to this remote request", page)
        self.context_remote_consent.setAccessibleName(
            "Consent to send the representative photo and displayed EXIF/XMP to the remote endpoint for this run"
        )
        form.addRow("Provider", self.context_provider)
        form.addRow("Model", self.context_model)
        form.addRow("Endpoint", self.context_endpoint)
        form.addRow("Remote API-key environment variable", self.context_key_environment)
        form.addRow(self.context_auto)
        form.addRow(self.context_remote_consent)
        layout.addLayout(form)
        row = QHBoxLayout()
        self.describe_cluster_button = QPushButton("Describe selected cluster")
        self.describe_cluster_button.setToolTip("Send the selected cluster's deterministic representative photo and scalar EXIF/XMP to the selected provider in a background Job.")
        self.describe_cluster_button.setProperty("kind", "primary")
        self.describe_cluster_button.clicked.connect(self.describe_selected_cluster)
        self.clear_context_button = QPushButton("Clear generated contexts")
        self.clear_context_button.setToolTip("Clear only generated descriptions and their search index. Source photos remain untouched.")
        self.clear_context_button.clicked.connect(self.clear_contexts)
        row.addWidget(self.describe_cluster_button)
        row.addWidget(self.clear_context_button)
        row.addStretch(1)
        layout.addLayout(row)
        self.context_status = QLabel("Select a cluster in Clustering to describe its first representative photo.")
        self.context_status.setWordWrap(True)
        layout.addWidget(self.context_status)
        self.tabs.addTab(page, "Cluster Context")
        self.context_provider.currentIndexChanged.connect(lambda _index: self._persist_context_settings())
        self.context_model.editingFinished.connect(self._persist_context_settings)
        self.context_endpoint.editingFinished.connect(self._persist_context_settings)
        self.context_key_environment.editingFinished.connect(self._persist_context_settings)
        self.context_auto.toggled.connect(lambda _checked: self._persist_context_settings())

    def set_selected_cluster(self, cluster_key: str, members: list[str] | tuple[str, ...]) -> None:
        self._selected_cluster_key = str(cluster_key or "")
        self._selected_cluster_members = tuple(str(path) for path in members if str(path or ""))
        if self._selected_cluster_members:
            self.context_status.setText(f"Selected cluster: {len(self._selected_cluster_members)} photos. Describe uses the deterministic first representative photo.")
        else:
            self.context_status.setText("Select a cluster in Clustering to describe its first representative photo.")
        self._update_controls()

    def automatic_context_enabled(self) -> bool:
        return bool(self.context_auto.isChecked())

    def describe_completed_clusters(self, clusters: dict[str, dict[int, list[str]]]) -> None:
        """Queue opt-in context generation sequentially after a completed run."""
        if not self.automatic_context_enabled():
            return
        settings = self._context_settings()
        if not settings.model:
            self.context_status.setText("Automatic cluster context is on, but no vision model is configured.")
            return
        if settings.is_remote and not settings.remote_consent:
            self.context_status.setText("Automatic remote context is waiting for this-run transfer consent in Library.")
            return
        items = [
            (f"{comparison_key}:{cluster_id}", tuple(paths))
            for comparison_key, by_cluster in sorted(clusters.items())
            for cluster_id, paths in sorted(by_cluster.items())
            if int(cluster_id) != -1 and paths
        ]
        if not items:
            return

        def _run(progress, cancel):
            records = []
            for index, (cluster_key, members) in enumerate(items, start=1):
                progress(int((index - 1) * 100 / len(items)), f"Describing cluster {index}/{len(items)}")
                records.append(
                    self.context_service.describe_cluster(
                        cluster_key=cluster_key,
                        members=members,
                        settings=settings,
                        progress_callback=lambda _value, text: progress(
                            int((index - 1) * 100 / len(items)), text
                        ),
                        cancel_check=cancel,
                    )
                )
            progress(100, f"Generated context for {len(records)} clusters")
            return records

        self._start_job(
            "Describing New Clusters",
            _run,
            lambda records: self.context_status.setText(f"Generated or reused context for {len(records)} newly completed clusters."),
        )

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        self.gallery.set_read_only_mode(self._read_only_mode)
        self.timeline_gallery.set_read_only_mode(self._read_only_mode)
        self._update_controls()

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
        """Apply the production Gallery integrations to both Library views."""

        self.gallery.metadata_service = metadata_service
        self.gallery.image_tag_service = image_tag_service
        self.gallery.face_edit_service_provider = face_service_provider
        self.gallery.face_edit_request_handler = face_edit_request_handler
        self.gallery.face_edit_saved_callback = face_edit_saved_callback
        self.gallery.inspector_context_provider = inspector_context_provider
        self.timeline_gallery.configure_photo_tools(
            metadata_service=metadata_service,
            image_tag_service=image_tag_service,
            face_service_provider=face_service_provider,
            face_edit_request_handler=face_edit_request_handler,
            face_edit_saved_callback=face_edit_saved_callback,
            inspector_context_provider=inspector_context_provider,
        )

    def apply_view_preferences(
        self,
        *,
        thumbnail_size: int,
        worker_count: int,
        prefetch_rows: int,
        pixmap_cache_size: int,
        qimage_cache_size: int,
    ) -> None:
        """Keep paged Search and virtual Timeline aligned with Gallery settings."""

        self.gallery.apply_view_preferences(
            thumbnail_size=thumbnail_size,
            worker_count=worker_count,
            prefetch_rows=prefetch_rows,
            pixmap_cache_size=pixmap_cache_size,
            qimage_cache_size=qimage_cache_size,
        )
        self.timeline_gallery.set_view_preferences(
            thumbnail_size=thumbnail_size,
            cache_size=qimage_cache_size,
            worker_count=worker_count,
        )

    def refresh_roots(self) -> None:
        self._catalog_snapshot_generation += 1
        generation = self._catalog_snapshot_generation
        if self._catalog_snapshot_job is not None:
            self._cancel_job(self._catalog_snapshot_job)

        def _run(_progress, cancel_check):
            roots = tuple(self.catalog.list_roots())
            if cancel_check():
                from infra.cancel import Cancelled

                raise Cancelled()
            counts = dict(self.catalog.root_asset_counts())
            albums = tuple(self.catalog.list_smart_albums())
            online = {root.root_id: Path(root.path).is_dir() for root in roots}
            return roots, counts, albums, online

        def _done(result) -> None:
            if generation != self._catalog_snapshot_generation or not self._qobject_alive():
                return
            roots, counts, albums, online = result
            self._roots_cache = tuple(roots)
            self._root_counts_cache = dict(counts)
            self._albums_cache = tuple(albums)
            self._apply_catalog_snapshot(dict(online))

        job = self._start_job("Refreshing Library navigation", _run, _done)
        self._catalog_snapshot_job = job
        job.completed.connect(lambda _result: self._clear_catalog_snapshot_job(job, generation))
        job.failed.connect(lambda _message: self._clear_catalog_snapshot_job(job, generation))
        job.cancelled.connect(lambda: self._clear_catalog_snapshot_job(job, generation))

    def _clear_catalog_snapshot_job(self, job: AsyncJob, generation: int) -> None:
        if generation == self._catalog_snapshot_generation and self._catalog_snapshot_job is job:
            self._catalog_snapshot_job = None

    def _apply_catalog_snapshot(self, online: dict[str, bool]) -> None:
        selected_id = self._selected_root_id()
        roots = self._roots_cache
        counts = self._root_counts_cache
        self.root_list.blockSignals(True)
        self.root_list.clear()
        for root in roots:
            label = f"{root.display_name}{'' if root.enabled else ' (disabled)'}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, root.root_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if root.enabled else Qt.CheckState.Unchecked)
            item.setToolTip(root.path if not root.last_error else f"{root.path}\nLast scan error: {root.last_error}")
            self.root_list.addItem(item)
            if root.root_id == selected_id:
                self.root_list.setCurrentItem(item)
        self.root_list.blockSignals(False)
        self.roots_changed.emit(
            [
                {
                    "root_id": root.root_id,
                    "path": root.path,
                    "enabled": root.enabled,
                    "online": bool(online.get(root.root_id, False)),
                    "photo_count": counts.get(root.root_id, 0),
                    "last_scan_at": root.last_scan_at,
                    "last_error": root.last_error,
                }
                for root in roots
            ]
        )
        self.refresh_albums()
        self._update_controls()

    def register_root_path(self, path: str) -> None:
        self._add_root(str(path or ""))

    def refresh_root_path(self, path: str) -> None:
        root = self._root_for_path(path)
        if root is not None:
            self.scan_roots(root_ids=[root.root_id])

    def cancel_root_scan(self, path: str) -> None:
        target = self._normalized_path_key(path)
        job = self._root_scan_jobs.get(target)
        if job is not None:
            self._cancel_job(job)

    def set_root_path_enabled(self, path: str, enabled: bool) -> None:
        root = self._root_for_path(path)
        if root is None:
            return

        def _done(_result) -> None:
            self.refresh_roots()
            self._reload_active_catalog_view()

        self._start_job(
            "Updating Library Root",
            lambda _progress, _cancel: self.catalog.set_root_enabled(root.root_id, bool(enabled)),
            _done,
        )

    def remove_root_path(self, path: str) -> None:
        root = self._root_for_path(path)
        if root is None:
            return
        if not confirmBox(
            "Remove Library root?",
            "This removes only derived Library records, albums' matches, and generated contexts for this root. Source photos are not touched.",
            parent=self,
        ):
            return

        def _done(_result) -> None:
            self.refresh_roots()
            self._reload_active_catalog_view()

        self._start_job(
            "Removing Library Root",
            lambda _progress, _cancel: self.catalog.remove_root(root.root_id),
            _done,
        )

    def refresh_albums(self) -> None:
        selected_id = self._selected_album_id()
        self.album_list.clear()
        for album in self._albums_cache:
            item = QListWidgetItem(album.name)
            item.setData(Qt.ItemDataRole.UserRole, album.album_id)
            item.setToolTip("Dynamic saved Library query")
            self.album_list.addItem(item)
            if album.album_id == selected_id:
                self.album_list.setCurrentItem(item)
        self._update_controls()

    @staticmethod
    def _normalized_path_key(path: str) -> str:
        return os.path.normcase(os.path.normpath(os.path.abspath(os.path.expanduser(str(path or "")))))

    def _root_for_path(self, path: str):
        target = self._normalized_path_key(path)
        return next(
            (root for root in self._roots_cache if self._normalized_path_key(root.path) == target),
            None,
        )

    def add_current_root(self) -> None:
        roots = self._active_scope().roots
        if not roots:
            errorBox("No active roots", "Choose one or more active roots first, or use Choose folder.", parent=self)
            return
        self._add_roots(list(roots))

    def add_root_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Add library root")
        if directory:
            self._add_root(directory)

    def _add_root(self, directory: str) -> None:
        self._add_roots([directory])

    def _add_roots(self, directories: list[str]) -> None:
        directories = [str(path) for path in directories if str(path or "").strip()]
        if not directories:
            return
        self.status_label.setText("Registering library root…")

        def _run(progress, cancel):
            roots = []
            normalized_directories = list(PathScope.from_paths(directories).roots)
            total = len(normalized_directories)
            for index, directory in enumerate(normalized_directories, start=1):
                if cancel():
                    return []
                progress(int((index - 1) * 100 / total), f"Registering library root {index}/{total}")
                roots.append(self.catalog.register_root(directory))
            progress(100, "Library roots registered")
            return roots

        def _done(roots) -> None:
            roots = list(roots or [])
            if not roots:
                return
            self.status_label.setText(f"Registered {len(roots)} library root(s). Refreshing their local catalog…")
            self.refresh_roots()
            self.scan_roots(root_ids=[root.root_id for root in roots])

        self._start_job("Registering Library Roots", _run, _done)

    def remove_selected_root(self) -> None:
        root_id = self._selected_root_id()
        if not root_id:
            return
        if not confirmBox("Remove library root?", "This removes only its local catalog records and generated contexts. Source photos are not touched.", parent=self):
            return
        def _run(progress, cancel):
            progress(0, "Removing library root records")
            self.catalog.remove_root(root_id)
            progress(100, "Library root removed")

        def _done(_result) -> None:
            self.refresh_roots()
            self.load_timeline()

        self._start_job("Removing Library Root", _run, _done)

    def scan_roots(self, *, root_ids: list[str] | None = None) -> None:
        if root_ids is None:
            root_ids = list(self._active_catalog_root_ids())
        if not root_ids:
            self.status_label.setText("No registered active roots to refresh. Register the active roots first.")
            return
        force = bool(self._force_catalog_date_refresh)
        self._force_catalog_date_refresh = False
        self.status_label.setText("Refreshing registered roots…")
        roots_by_id = {root.root_id: root.path for root in self._roots_cache}
        scanned_paths = [roots_by_id[root_id] for root_id in root_ids if root_id in roots_by_id]
        self._catalog_scan_generation += 1
        scan_generation = self._catalog_scan_generation
        self._catalog_committed_scan_stats = {}
        job = self._start_job(
            "Refreshing Library",
            lambda progress, cancel: self.catalog.scan_roots(
                root_ids,
                force=force,
                progress_callback=progress,
                cancel_check=cancel,
                on_committed_batch=lambda root_id, stats: self.catalog_batch_committed.emit(
                    (scan_generation, str(root_id), dict(stats))
                ),
            ),
            lambda result: self._catalog_scan_completed(result, scan_generation=scan_generation),
        )
        for path in scanned_paths:
            self._root_scan_jobs[self._normalized_path_key(path)] = job
            self.root_job_state_changed.emit(path, "refreshing")

        def _finished(*_args) -> None:
            for path in scanned_paths:
                key = self._normalized_path_key(path)
                if self._root_scan_jobs.get(key) is job:
                    self._root_scan_jobs.pop(key, None)
                    self.root_job_state_changed.emit(path, "")

        job.completed.connect(_finished)
        job.failed.connect(_finished)
        job.cancelled.connect(_finished)
        job.cancelled.connect(lambda: self._catalog_scan_cancelled(scan_generation))

    def _on_catalog_batch_committed(self, payload: object) -> None:
        try:
            generation, root_id, stats = payload
        except (TypeError, ValueError):
            return
        if int(generation) != self._catalog_scan_generation:
            return
        normalized = {str(key): int(value) for key, value in dict(stats).items()}
        self._catalog_committed_scan_stats[str(root_id)] = normalized
        committed = sum(int(item.get("updated", 0)) for item in self._catalog_committed_scan_stats.values())
        self.status_label.setText(f"Refreshing registered roots… {committed} photo(s) committed.")

    def _catalog_scan_cancelled(self, generation: int) -> None:
        if int(generation) != self._catalog_scan_generation:
            return
        committed = sum(int(item.get("updated", 0)) for item in self._catalog_committed_scan_stats.values())
        removed = sum(int(item.get("removed", 0)) for item in self._catalog_committed_scan_stats.values())
        self.status_label.setText(
            f"Library refresh cancelled. Showing {committed} committed update(s)"
            f"{f' and {removed} committed removal(s)' if removed else ''}."
        )
        self.refresh_roots()
        self._reload_active_catalog_view()

    def _on_timeline_date_source_changed(self, _index: int) -> None:
        policy = str(self.timeline_date_source.currentData() or "metadata_or_filename")
        setter = getattr(self.catalog, "set_filename_date_policy", None)
        if callable(setter):
            setter(policy)
        self._save_timeline_date_preferences()
        self._force_catalog_date_refresh = True
        self.status_label.setText("Timeline date source changed. Click Refresh dates to rebuild derived timeline dates.")

    def _on_timeline_grouping_changed(self, _index: int) -> None:
        """Regroup the in-memory Timeline result without launching a source scan."""

        self._save_timeline_date_preferences()
        if self._timeline is not None:
            self._publish_timeline(self._timeline)
            self._update_controls()

    def _on_timeline_filename_patterns_changed(self) -> None:
        self._force_catalog_date_refresh = True
        self.status_label.setText("Filename time rules changed. Click Refresh dates to rebuild derived timeline dates.")

    def _on_timeline_epoch_heuristic_changed(self, _enabled: bool) -> None:
        self.catalog.set_filename_epoch_heuristic(_enabled)
        self._save_timeline_date_preferences()
        self._force_catalog_date_refresh = True
        self.status_label.setText("Raw epoch recognition changed. Click Refresh dates to rebuild derived timeline dates.")

    def refresh_timeline_dates(self) -> None:
        """Persist and force a cancellable source-safe date recomputation."""

        try:
            self.catalog.set_filename_date_patterns(self.timeline_filename_patterns.toPlainText())
            self.catalog.set_filename_epoch_heuristic(self.timeline_epoch_heuristic.isChecked())
        except ValueError as exc:
            errorBox("Check filename time rules", str(exc), parent=self)
            return
        self._save_timeline_date_preferences()
        self._force_catalog_date_refresh = True
        self.scan_roots()

    def _catalog_scan_completed(self, result: dict[str, int], *, scan_generation: int | None = None) -> None:
        if scan_generation is not None and int(scan_generation) != self._catalog_scan_generation:
            return
        self._catalog_committed_scan_stats = {}
        details = []
        for key, label in (
            ("filtered", "excluded by source filters"),
            ("filename_dates", "filename dates"),
            ("filename_epochs", "explicit epochs"),
            ("raw_epochs", "raw ID/hash epochs"),
            ("ambiguous_epochs", "ambiguous raw IDs skipped"),
        ):
            count = int(result.get(key, 0))
            if count:
                details.append(f"{count} {label}")
        summary = (
            f"Catalog refreshed: {int(result.get('updated', 0))} updated, {int(result.get('unchanged', 0))} unchanged, "
            f"{int(result.get('removed', 0))} removed."
        )
        self.status_label.setText(f"{summary} {'; '.join(details)}." if details else summary)
        self.refresh_roots()
        self._reload_active_catalog_view()

    def load_timeline(self) -> None:
        self._timeline_loaded = True
        self._timeline = None
        self._catalog_generation += 1
        self._asset_offset = 0
        root_ids, scope_paths = self._catalog_scope()
        self._active_query = CatalogQuery(
            root_ids=root_ids, scope_paths=scope_paths, start_at=self.timeline_start.text().strip(), end_at=self.timeline_end.text().strip(),
            camera=self.timeline_camera.text().strip(), order="captured_desc", offset=0, limit=self.PAGE_SIZE,
        )
        self._load_timeline()

    def _load_timeline(self) -> None:
        """Load every matching path/date pair in a cancellable worker Job."""

        generation = self._catalog_generation
        self._timeline_request_generation += 1
        request_generation = self._timeline_request_generation
        if self._asset_job is not None:
            self._asset_request_generation += 1
            self._cancel_job(self._asset_job)
            self._asset_loading = False
        if self._timeline_job is not None:
            self._cancel_job(self._timeline_job)
        self._timeline_loading = True
        self.status_label.setText("Loading the full Library timeline…")
        if self._timeline_active():
            self.gallery_stack.setCurrentWidget(self.timeline_gallery)
            self.timeline_gallery.set_loading_state("Reading all catalogued capture dates…")

        query = self._active_query

        def _run(progress, cancel):
            return self.catalog.query_timeline(
                query,
                progress_callback=progress,
                cancel_check=cancel,
            )

        def _done(timeline: CatalogTimeline) -> None:
            if request_generation != self._timeline_request_generation:
                return
            self._timeline_loading = False
            self._timeline_job = None
            if generation != self._catalog_generation:
                self._update_controls()
                return
            self._publish_timeline(timeline)
            self._update_controls()

        job = self._start_job("Loading Library Timeline", _run, _done)
        self._timeline_job = job

        def _timeline_request_finished() -> None:
            if request_generation == self._timeline_request_generation:
                self._timeline_loading = False
                self._timeline_job = None
                self._update_controls()

        job.failed.connect(lambda _message: _timeline_request_finished())
        job.cancelled.connect(_timeline_request_finished)

    def _timeline_grouping(self) -> str:
        grouping = str(self.timeline_grouping.currentData() or "year_month")
        return grouping if grouping in self._TIMELINE_GROUPINGS else "year_month"

    def _publish_timeline(self, timeline: CatalogTimeline) -> None:
        """Publish an already-loaded scalar Timeline using the chosen hierarchy."""

        self._timeline = timeline
        self._timeline_paths = timeline.image_paths
        self._timeline_total = timeline.total_count
        sections, collapsed = self._timeline_sections(timeline)
        years = len(timeline.years)
        grouping = self._timeline_grouping()
        newest_label = {
            "year": "Newest year is expanded.",
            "year_month": "Newest month is expanded.",
            "year_month_week": "Newest week is expanded.",
            "year_month_day": "Newest day is expanded.",
        }[grouping]
        if sections:
            self.timeline_gallery.set_sections_with_collapsed(
                sections,
                collapsed_section_ids=collapsed,
                status=(
                    f"{timeline.total_count:,} catalogued photos in {years} year"
                    f"{'s' if years != 1 else ''}. {newest_label}"
                ),
            )
        else:
            self.timeline_gallery.set_empty_state("No catalog photos match this timeline filter.")
        self.timeline_count.setText(
            f"Showing {timeline.total_count:,} photo"
            f"{'s' if timeline.total_count != 1 else ''} in {years} year{'s' if years != 1 else ''}"
        )
        self.status_label.setText("Library timeline ready." if timeline.total_count else "No catalog photos match this view.")

    def _timeline_sections(self, timeline: CatalogTimeline) -> tuple[list[GallerySection], set[str]]:
        """Map scalar catalog dates to virtual headers without rereading source media."""

        sections: list[GallerySection] = []
        collapsed: set[str] = set()
        grouping = self._timeline_grouping()
        for year_index, year in enumerate(timeline.years):
            year_id = f"timeline:year:{year.year:04d}"
            if year_index:
                collapsed.add(year_id)
            if grouping == "year":
                sections.append(
                    GallerySection(
                        year_id,
                        tuple(path for month in year.months for path in month.image_paths),
                        kind="timeline_year",
                        title=str(year.year) if year.year > 0 else "Unparsed filename time",
                    )
                )
                continue
            months: list[GallerySection] = []
            for month_index, month in enumerate(year.months):
                month_id = f"{year_id}:month:{month.month:02d}"
                if year_index or month_index:
                    collapsed.add(month_id)
                if year.year > 0 and 1 <= month.month <= 12:
                    title = f"{month_name[month.month]} {year.year}"
                else:
                    title = "Unparsed filename time"
                if year.year <= 0 or grouping == "year_month" or not month.days:
                    months.append(
                        GallerySection(
                            month_id,
                            tuple(month.image_paths),
                            kind="timeline_month",
                            title=title,
                        )
                    )
                    continue
                children: list[GallerySection] = []
                if grouping == "year_month_week":
                    weeks: dict[int, list[str]] = {}
                    for day in month.days:
                        weeks.setdefault(day.iso_week, []).extend(day.image_paths)
                    for week_index, (week, image_paths) in enumerate(weeks.items()):
                        week_id = f"{month_id}:week:{week:02d}"
                        if year_index or month_index or week_index:
                            collapsed.add(week_id)
                        children.append(
                            GallerySection(
                                week_id,
                                tuple(image_paths),
                                kind="timeline_week",
                                title=f"Week {week:02d}",
                            )
                        )
                else:
                    for day_index, day in enumerate(month.days):
                        day_id = f"{month_id}:day:{day.day:02d}"
                        if year_index or month_index or day_index:
                            collapsed.add(day_id)
                        children.append(
                            GallerySection(
                                day_id,
                                tuple(day.image_paths),
                                kind="timeline_day",
                                title=f"{month_name[month.month]} {day.day}, {year.year}",
                            )
                        )
                months.append(
                    GallerySection(
                        month_id,
                        (),
                        kind="timeline_month",
                        title=title,
                        children=tuple(children),
                    )
                )
            sections.append(
                GallerySection(
                    year_id,
                    (),
                    kind="timeline_year",
                    title=str(year.year) if year.year > 0 else "Unparsed",
                    children=tuple(months),
                )
            )
        return sections, collapsed

    def ensure_timeline_loaded(self) -> None:
        if not self._timeline_loaded:
            self.load_timeline()

    def invalidate_catalog_view(self) -> None:
        """Drop presentation state after Settings clears derived catalog rows."""
        if self._asset_job is not None:
            self._cancel_job(self._asset_job)
        if self._timeline_job is not None:
            self._cancel_job(self._timeline_job)
        self._catalog_generation += 1
        self._asset_request_generation += 1
        self._timeline_request_generation += 1
        self._timeline_loaded = False
        self._asset_loading = False
        self._asset_auto_loading = False
        self._timeline_loading = False
        self._asset_offset = 0
        self._asset_total = 0
        self._timeline_paths = ()
        self._timeline_total = 0
        self._timeline = None
        self.gallery.update_gallery([])
        self.timeline_gallery.set_empty_state("Library cache was cleared. Refresh a registered root to rebuild it.")
        self.timeline_count.setText("0 photos")
        self.status_label.setText("Library cache was cleared. Refresh a registered root to rebuild it.")
        self._update_controls()

    def run_search(self) -> None:
        text = self.search_field.text().strip()
        self._catalog_generation += 1
        self._timeline_loaded = False
        self._timeline_request_generation += 1
        if self._timeline_job is not None:
            self._cancel_job(self._timeline_job)
        self._timeline_loading = False
        self._context_search_generation += 1
        context_generation = self._context_search_generation
        if self._context_search_job is not None:
            self._cancel_job(self._context_search_job)
            self._context_search_job = None
        self._asset_offset = 0
        # Search starts with a bounded first page for a quick, viewport-led
        # gallery paint, then continues through every match in cancellable
        # pages rather than silently stopping at one page.
        self._asset_auto_loading = True
        root_ids, scope_paths = self._catalog_scope()
        self._active_query = CatalogQuery(
            root_ids=root_ids,
            scope_paths=scope_paths,
            text=text,
            start_at=self.search_start.text().strip(),
            end_at=self.search_end.text().strip(),
            camera=self.search_camera.text().strip(),
            folder=self.search_folder.text().strip(),
            file_ext=self.search_file_ext.text().strip(),
            order="captured_desc",
            offset=0,
            limit=self.PAGE_SIZE,
        )
        self.context_results.clear()
        self._set_context_results_panel_visible(False)
        self.gallery_stack.setCurrentWidget(self.gallery)
        self._load_assets(reset=True)

        def _run(progress, cancel):
            progress(-1, "Searching generated cluster contexts…")
            return self.catalog.search_cluster_contexts(text)

        def _done(page) -> None:
            if context_generation != self._context_search_generation:
                return
            self._context_search_job = None
            for context in page.items:
                item = QListWidgetItem(f"{context.title} — {context.description}")
                item.setData(Qt.ItemDataRole.UserRole, context)
                item.setToolTip(f"{context.representative_path}\n{', '.join(context.keywords)}")
                self.context_results.addItem(item)
            self._set_context_results_panel_visible(self.context_results.count() > 0)

        if text:
            job = self._start_job("Searching Cluster Context", _run, _done)
            self._context_search_job = job
            job.failed.connect(lambda _message: self._clear_context_search_job(context_generation))
            job.cancelled.connect(lambda: self._clear_context_search_job(context_generation))

    def _load_assets(self, *, reset: bool) -> None:
        generation = self._catalog_generation
        self._asset_request_generation += 1
        request_generation = self._asset_request_generation
        if self._asset_job is not None:
            self._cancel_job(self._asset_job)
        if self._timeline_job is not None:
            self._timeline_request_generation += 1
            self._cancel_job(self._timeline_job)
            self._timeline_loading = False
        self.gallery_stack.setCurrentWidget(self.gallery)
        self._asset_loading = True
        query = CatalogQuery(**{**self.catalog.query_to_payload(self._active_query), "offset": self._asset_offset, "limit": self.PAGE_SIZE})
        self.status_label.setText("Loading Library photos…")

        def _run(progress, cancel):
            progress(-1, "Loading local catalog page…")
            if cancel():
                return None
            return self.catalog.query_assets(query)

        def _done(page) -> None:
            if request_generation != self._asset_request_generation:
                return
            self._asset_loading = False
            self._asset_job = None
            if page is None or generation != self._catalog_generation:
                self._update_controls()
                return
            paths = [asset.image_path for asset in page.items]
            if reset:
                self.gallery.update_gallery(paths)
            else:
                self.gallery.update_gallery([*self.gallery.images, *paths])
            self._asset_total = page.total_count
            self._asset_offset = page.next_offset or page.total_count
            self.timeline_count.setText(showing(len(self.gallery.images), "photo", total=page.total_count, matching=True))
            has_next_page = self._asset_offset < self._asset_total
            if self._asset_auto_loading and has_next_page:
                self.status_label.setText(
                    f"{showing(len(self.gallery.images), 'photo', total=page.total_count, matching=True)} · loading next page…"
                )
            else:
                self._asset_auto_loading = False
                self.status_label.setText("Library catalog ready." if page.total_count else "No catalog photos match this view.")
            self._update_controls()
            if self._asset_auto_loading and has_next_page:
                QTimer.singleShot(0, lambda: self._continue_asset_stream(request_generation, generation))

        job = self._start_job("Loading Library", _run, _done)
        self._asset_job = job

        def _asset_request_finished() -> None:
            if request_generation == self._asset_request_generation:
                self._asset_loading = False
                self._asset_job = None
                self._update_controls()

        job.failed.connect(lambda _message: _asset_request_finished())
        job.cancelled.connect(_asset_request_finished)

    def load_more_assets(self) -> None:
        if self._asset_auto_loading:
            self._asset_auto_loading = False
            self._asset_request_generation += 1
            if self._asset_job is not None:
                self._cancel_job(self._asset_job)
            self._asset_loading = False
            self.status_label.setText(f"Paused after {len(self.gallery.images)} of {self._asset_total} Library photos.")
            self._update_controls()
            return
        if self._timeline_active() or self._asset_loading or self._asset_offset >= self._asset_total:
            return
        self._load_assets(reset=False)

    def _continue_asset_stream(self, request_generation: int, catalog_generation: int) -> None:
        if (
            not self._asset_auto_loading
            or request_generation != self._asset_request_generation
            or catalog_generation != self._catalog_generation
            or self._asset_loading
            or self._asset_offset >= self._asset_total
            or self._timeline_active()
        ):
            return
        self._load_assets(reset=False)

    def save_current_album(self) -> None:
        name, accepted = _simple_text_prompt(self, "Save smart album", "Album name")
        if not accepted:
            return
        query = self._active_query

        def _done(_album) -> None:
            self.refresh_roots()
            self.status_label.setText("Saved Library smart album.")

        self._start_job(
            "Saving Library Smart Album",
            lambda progress, cancel: self.catalog.save_smart_album(name, query),
            _done,
        )

    def delete_selected_album(self) -> None:
        album_id = self._selected_album_id()
        if not album_id:
            return
        def _done(_result) -> None:
            self.refresh_roots()
            self.status_label.setText("Deleted Library smart album.")

        self._start_job(
            "Deleting Library Smart Album",
            lambda progress, cancel: self.catalog.delete_smart_album(album_id),
            _done,
        )

    def _album_selection_changed(self) -> None:
        album_id = self._selected_album_id()
        if not album_id:
            self._update_controls()
            return
        self._catalog_generation += 1
        self._asset_offset = 0
        self._asset_auto_loading = False
        album = next((item for item in self._albums_cache if item.album_id == album_id), None)
        if album is None:
            self._update_controls()
            return
        self._active_query = self.catalog.payload_to_query(dict(album.query))
        if self._timeline_active():
            self._timeline_loaded = True
            self._load_timeline()
        else:
            self._load_assets(reset=True)
        self._update_controls()

    def open_selected_context(self) -> None:
        item = self.context_results.currentItem()
        context = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        if isinstance(context, ClusterContextRecord):
            self.open_in_gallery_requested.emit(list(context.members), f"Cluster context: {context.title}")

    def find_duplicates(self, kinds: tuple[str, ...] = ("exact", "near", "burst")) -> None:
        requested_kinds = tuple(dict.fromkeys(str(kind) for kind in kinds if str(kind)))
        self._duplicate_generation += 1
        generation = self._duplicate_generation
        if self._duplicate_job is not None:
            self._cancel_job(self._duplicate_job)
        self.status_label.setText(f"Finding {' + '.join(self._duplicate_kind_label(kind).lower() for kind in requested_kinds)} candidates…")
        root_ids, scope_paths = self._catalog_scope()
        if not scope_paths:
            self.status_label.setText("Choose active roots before finding duplicates.")
            return

        def _run(progress, cancel):
            return self.duplicate_service.build_groups(
                root_ids=root_ids,
                scope_paths=scope_paths,
                kinds=requested_kinds,
                progress_callback=progress,
                cancel_check=cancel,
            )

        def _done(groups) -> None:
            if generation != self._duplicate_generation:
                return
            self._duplicate_job = None
            self._duplicate_groups = list(groups or [])
            self.duplicate_list.clear()
            for index, group in enumerate(self._duplicate_groups):
                item = QListWidgetItem(
                    f"{self._duplicate_kind_label(group.kind)} · {len(group.members)} photos · keeper: {Path(group.keeper_path).name}"
                )
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(group.summary)
                self.duplicate_list.addItem(item)
            self.status_label.setText(f"Prepared {len(self._duplicate_groups)} manual review groups.")
            self._duplicate_selection_changed()
            self._update_controls()

        job = self._start_job("Finding " + " + ".join(self._duplicate_kind_label(kind) for kind in requested_kinds), _run, _done)
        self._duplicate_job = job
        job.failed.connect(lambda _message: self._clear_duplicate_job(generation))
        job.cancelled.connect(lambda: self._clear_duplicate_job(generation))

    def mark_selected_not_duplicate(self) -> None:
        group = self._selected_duplicate_group()
        if group is None:
            return

        def _done(_result) -> None:
            self.status_label.setText("Saved the selected group's not-duplicate feedback. It will not reappear until source fingerprints change.")

        self._start_job(
            "Saving Duplicate Review Feedback",
            lambda progress, cancel: self.duplicate_service.mark_not_duplicate(
                group, progress_callback=progress, cancel_check=cancel,
            ),
            _done,
        )

    def trash_selected_duplicate_candidates(self) -> None:
        group = self._selected_duplicate_group()
        if group is None:
            return
        self.trash_duplicate_kind(group.kind)

    @staticmethod
    def _duplicate_kind_label(kind: str) -> str:
        return {"exact": "Hash duplicate", "near": "Similar photo", "similar": "Similar photo", "burst": "Burst"}.get(
            str(kind), str(kind).title()
        )

    def _duplicate_candidate_paths(self, kind: str) -> list[str]:
        normalized_kind = "near" if str(kind) == "similar" else str(kind)
        keeper_paths = {group.keeper_path for group in self._duplicate_groups if group.kind == normalized_kind}
        return list(
            dict.fromkeys(
                member.image_path
                for group in self._duplicate_groups
                if group.kind == normalized_kind
                for member in group.members
                if member.image_path != group.keeper_path and member.image_path not in keeper_paths
            )
        )

    def trash_duplicate_kind(self, kind: str) -> None:
        candidates = self._duplicate_candidate_paths(kind)
        if not candidates:
            self.status_label.setText(f"No {self._duplicate_kind_label(kind).lower()} candidates are available to preview.")
            return
        if self._read_only_mode:
            errorBox("Read-only safety mode", "Turn off read-only safety mode before moving files to Trash.", parent=self)
            return
        dialog = _DuplicateTrashPreviewDialog(self._duplicate_kind_label(kind), candidates, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        checked_paths = dialog.selected_paths()
        if not checked_paths:
            return

        def _run(progress, cancel):
            return self.gallery.action_service.move_to_trash(checked_paths, progress_callback=progress, cancel_check=cancel)

        def _done(result) -> None:
            moved = [source for source, _target in result.changed_paths]
            self.metadata_changed.emit(moved)
            self.status_label.setText(f"Moved {len(moved)} {self._duplicate_kind_label(kind).lower()} candidate(s) to ClusterLens Trash.")
            self.scan_roots()

        self._start_job(f"Trashing {self._duplicate_kind_label(kind)} Candidates", _run, _done)

    def load_people_inbox(self) -> None:
        # Face search is intentionally lazy: opening Library must not load
        # models or face-index dependencies until People Cleanup is requested.
        from app.services.people_cleanup import PeopleCleanupService

        face_service = self.face_service_provider()
        if face_service is None:
            errorBox("Faces are not ready", "Open Faces once to prepare the local face index, then return to People Cleanup.", parent=self)
            return
        self._request_people_name_choices(face_service)
        self._people_generation += 1
        generation = self._people_generation
        if self._people_job is not None:
            self._cancel_job(self._people_job)
        service = PeopleCleanupService(face_service, catalog=self.catalog)
        self._people_service = service
        root_ids, scope_paths = self._catalog_scope()
        if not scope_paths:
            self.people_summary.setText("Choose active roots before building People Cleanup.")
            return

        def _run(progress, cancel):
            return service.build_snapshot(
                root_ids=root_ids,
                scope_paths=scope_paths,
                progress_callback=progress,
                cancel_check=cancel,
            )

        def _done(snapshot) -> None:
            if generation != self._people_generation:
                return
            self._people_job = None
            self._people_groups = list(snapshot.groups)
            self.people_list.clear()
            for index, group in enumerate(self._people_groups):
                item = QListWidgetItem(f"{len(group.members)} faces · {group.summary}")
                item.setData(Qt.ItemDataRole.UserRole, index)
                self.people_list.addItem(item)
            self.people_summary.setText(f"{snapshot.completion_percent}% named · {snapshot.unlabelled_faces} unlabelled · {len(snapshot.groups)} review groups.")
            self._people_selection_changed()
            self._update_controls()

        job = self._start_job("Building People Cleanup Inbox", _run, _done)
        self._people_job = job
        job.failed.connect(lambda _message: self._clear_people_job(generation))
        job.cancelled.connect(lambda: self._clear_people_job(generation))

    def apply_people_name(self) -> None:
        group = self._selected_people_group()
        name = self.people_name_field.text().strip()
        if group is None or not name or self._people_service is None:
            return
        if self._read_only_mode:
            errorBox("Read-only safety mode", "Turn off read-only safety mode before naming face regions.", parent=self)
            return

        def _run(progress, cancel):
            return self._people_service.apply_name(name, group.members, progress=progress, cancel_check=cancel)

        def _done(result) -> None:
            self.status_label.setText(f"Named {len(result.affected_refs)} face regions; {len(result.failed_paths)} photos need attention.")
            self.metadata_changed.emit(sorted({str(ref.image_path) for ref in result.affected_refs}))
            self.load_people_inbox()

        self._start_job("Naming People Cleanup Faces", _run, _done)

    def reject_people_suggestion(self) -> None:
        group = self._selected_people_group()
        if group is None or not group.suggested_name or self._people_service is None:
            return
        def _done(_result) -> None:
            self.status_label.setText(f"Rejected {group.suggested_name} for this group.")
            self.load_people_inbox()

        self._start_job(
            "Rejecting People Suggestion",
            lambda progress, cancel: self._people_service.reject_suggestion(
                group.suggested_name, group.members, progress_callback=progress, cancel_check=cancel,
            ),
            _done,
        )

    def split_people_group(self) -> None:
        group = self._selected_people_group()
        if group is None or self._people_service is None:
            return
        if not confirmBox(
            "Split this people group?",
            "Each face in this group will remain separate in future People Cleanup reviews until you explicitly merge or relabel it.",
            parent=self,
        ):
            return

        def _done(_result) -> None:
            self.status_label.setText("Saved the split decision.")
            self.load_people_inbox()

        self._start_job(
            "Splitting People Group",
            lambda progress, cancel: self._people_service.split_group(group.members, progress_callback=progress, cancel_check=cancel),
            _done,
        )

    def hide_people_group(self) -> None:
        group = self._selected_people_group()
        if group is None or self._people_service is None or self._read_only_mode:
            return
        if not confirmBox("Hide selected faces?", "Hidden false detections stay in the local face index and can be restored from Faces.", parent=self):
            return
        def _done(_result) -> None:
            self.metadata_changed.emit([member.image_path for member in group.members])
            self.load_people_inbox()

        self._start_job(
            "Hiding People Faces",
            lambda progress, cancel: self._people_service.hide_members(group.members, progress_callback=progress, cancel_check=cancel),
            _done,
        )

    def merge_people_names(self) -> None:
        if self._people_service is None or self._read_only_mode:
            return
        choices = self.people_name_field.choices
        source_dialog = EntityPickerDialog(
            "Merge people",
            "Saved name to merge into another person:",
            choices=choices,
            allow_create=False,
            entity_label="person name",
            parent=self,
        )
        if source_dialog.exec() != source_dialog.DialogCode.Accepted:
            return
        source = source_dialog.selected_value()
        target_dialog = EntityPickerDialog(
            "Merge people",
            f"Merge {source or 'saved name'} into:",
            choices=choices,
            initial=self.people_name_field.text(),
            allow_create=True,
            entity_label="person name",
            parent=self,
        )
        if not source or target_dialog.exec() != target_dialog.DialogCode.Accepted:
            return
        target = target_dialog.selected_value()
        if not target or source.casefold() == target.casefold():
            return

        def _done(_result) -> None:
            self.status_label.setText(f"Merged {source.strip()} into {target.strip()}.")
            self.load_people_inbox()

        self._start_job(
            "Merging People Names",
            lambda progress, cancel: self._people_service.merge_people(source.strip(), target.strip()),
            _done,
        )

    def describe_selected_cluster(self) -> None:
        if not self._selected_cluster_members:
            errorBox("No cluster selected", "Select a cluster in Clustering first.", parent=self)
            return
        settings = self._context_settings()
        if settings.is_remote and not settings.remote_consent:
            errorBox("Remote consent required", "Confirm the representative-photo and EXIF/XMP transfer for this run before using a remote provider.", parent=self)
            return

        def _run(progress, cancel):
            return self.context_service.describe_cluster(
                cluster_key=self._selected_cluster_key, members=self._selected_cluster_members,
                settings=settings, progress_callback=progress, cancel_check=cancel,
            )

        def _done(record) -> None:
            self.context_status.setText(f"Saved context: {record.title}. Search will now return this cluster.")

        self._start_job("Describing Cluster Context", _run, _done)

    def clear_contexts(self) -> None:
        if not confirmBox("Clear generated cluster contexts?", "This clears only rebuildable generated descriptions and their search index. Source files are untouched.", parent=self):
            return
        self._start_job("Clearing Cluster Context", lambda progress, cancel: self.catalog.clear_cluster_contexts(), lambda count: self.context_status.setText(f"Cleared {count} generated cluster contexts."))

    def _context_settings(self) -> VisionLanguageSettings:
        provider = str(self.context_provider.currentData() or "ollama")
        endpoint = self.context_endpoint.text().strip()
        return VisionLanguageSettings(
            provider=provider,
            ollama_url=endpoint or "http://127.0.0.1:11434",
            ollama_model=self.context_model.text().strip() if provider == "ollama" else "",
            openai_url=endpoint if provider == "openai-compatible" else "",
            openai_model=self.context_model.text().strip() if provider == "openai-compatible" else "",
            api_key_environment=self.context_key_environment.text().strip() or "CLUSTERLENS_LLM_API_KEY",
            remote_consent=bool(self.context_remote_consent.isChecked()),
        )

    def _restore_context_settings(self) -> None:
        if self.context_settings_provider is None:
            return
        settings = self.context_settings_provider()
        widgets = (self.context_provider, self.context_model, self.context_endpoint, self.context_key_environment, self.context_auto)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            index = self.context_provider.findData(settings.provider)
            if index >= 0:
                self.context_provider.setCurrentIndex(index)
            self.context_model.setText(settings.model)
            self.context_endpoint.setText(settings.ollama_url if settings.provider == "ollama" else settings.openai_url)
            self.context_key_environment.setText(settings.api_key_environment)
            # Explicit consent is for the current application run only.
            self.context_remote_consent.setChecked(False)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

    def _restore_timeline_date_preferences(self) -> None:
        if self.timeline_date_preferences_provider is None:
            return
        try:
            preferences = self.timeline_date_preferences_provider()
            if preferences.filename_patterns:
                self.catalog.set_filename_date_patterns(preferences.filename_patterns)
            self.catalog.set_filename_date_policy(preferences.date_policy)
            self.catalog.set_filename_epoch_heuristic(preferences.raw_epoch_heuristic)
        except (TypeError, ValueError):
            return
        self.timeline_date_source.blockSignals(True)
        self.timeline_grouping.blockSignals(True)
        self.timeline_filename_patterns.blockSignals(True)
        self.timeline_epoch_heuristic.blockSignals(True)
        try:
            index = self.timeline_date_source.findData(self.catalog.filename_date_policy)
            self.timeline_date_source.setCurrentIndex(max(0, index))
            grouping = str(getattr(preferences, "grouping", "year_month") or "year_month")
            grouping_index = self.timeline_grouping.findData(grouping)
            self.timeline_grouping.setCurrentIndex(max(0, grouping_index))
            self.timeline_filename_patterns.setPlainText("\n".join(self.catalog.filename_date_patterns))
            self.timeline_epoch_heuristic.setChecked(self.catalog.filename_epoch_heuristic)
        finally:
            self.timeline_epoch_heuristic.blockSignals(False)
            self.timeline_filename_patterns.blockSignals(False)
            self.timeline_grouping.blockSignals(False)
            self.timeline_date_source.blockSignals(False)

    def _save_timeline_date_preferences(self) -> None:
        if self.timeline_date_preferences_changed is None:
            return
        self.timeline_date_preferences_changed(
            TimelineDatePreferences(
                str(self.catalog.filename_date_policy),
                tuple(self.catalog.filename_date_patterns),
                bool(self.catalog.filename_epoch_heuristic),
                self._timeline_grouping(),
            )
        )

    def _persist_context_settings(self) -> None:
        if self.context_settings_changed is not None:
            self.context_settings_changed(self._context_settings(), bool(self.context_auto.isChecked()))

    def _root_enabled_changed(self, item: QListWidgetItem) -> None:
        root_id = str(item.data(Qt.ItemDataRole.UserRole) or "")
        if not root_id:
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        self.root_list.setEnabled(False)

        def _done(_result) -> None:
            self.root_list.setEnabled(True)
            self.status_label.setText("Library root enabled." if enabled else "Library root disabled; its photos are excluded from curation.")
            self.refresh_roots()
            self._reload_active_catalog_view()

        job = self._start_job(
            "Updating Library Root",
            lambda progress, cancel: self.catalog.set_root_enabled(root_id, enabled),
            _done,
        )
        job.failed.connect(lambda _message: self.root_list.setEnabled(True))
        job.cancelled.connect(lambda: self.root_list.setEnabled(True))

    def _root_selection_changed(self) -> None:
        self._reload_active_catalog_view()
        self._update_controls()

    @staticmethod
    def _surface_key(value: str) -> str:
        return str(value or "").strip().lower().replace("&", "and").replace(" ", "_")

    def set_surface_group(self, group: str, section: str | None = None) -> None:
        """Expose one coherent shell surface without duplicating its workers.

        The application shell owns the visible contextual navigation.  This
        pane keeps the mature Library, cleanup, and context implementations,
        but only the pages belonging to the selected destination participate
        in layout or keyboard traversal.
        """

        surface_group = self._surface_key(group)
        visible_by_group = {
            "library": {"Timeline", "All Photos", "Search", "Albums"},
            "tools": {"Cleanup", "Batch Rename & Metadata", "Recovery"},
            "people": {"People Cleanup"},
            "context": {"Cluster Context"},
        }
        visible = visible_by_group.get(surface_group, visible_by_group["library"])
        self._surface_group = surface_group if surface_group in visible_by_group else "library"
        for index in range(self.tabs.count()):
            self.tabs.setTabVisible(index, self.tabs.tabText(index) in visible)
        self.tabs.tabBar().setVisible(False)
        titles = {
            "library": "Library",
            "tools": "Tools",
            "people": "People review",
            "context": "Cluster context",
        }
        self.workspace_title.setText(titles[self._surface_group])
        self.library_sidebar.setVisible(self._surface_group == "library")
        requested = self._surface_key(section or "")
        aliases = {
            "all": "all_photos",
            "photos": "all_photos",
            "duplicates": "cleanup",
            "rename": "batch_rename_and_metadata",
            "metadata": "batch_rename_and_metadata",
            "people": "people_cleanup",
            "review": "people_cleanup",
            "context": "cluster_context",
        }
        requested = aliases.get(requested, requested)
        target_index = -1
        for index in range(self.tabs.count()):
            if self.tabs.isTabVisible(index) and self._surface_key(self.tabs.tabText(index)) == requested:
                target_index = index
                break
        if target_index < 0:
            target_index = next(
                (index for index in range(self.tabs.count()) if self.tabs.isTabVisible(index)),
                0,
            )
        self.tabs.setCurrentIndex(target_index)
        self._tab_changed(target_index)

    def select_surface_section(self, section: str) -> None:
        self.set_surface_group(self._surface_group, section)

    def _reload_active_catalog_view(self) -> None:
        if self._timeline_active():
            self.load_timeline()
        elif self.tabs.tabText(self.tabs.currentIndex()) in {"All Photos", "Search", "Albums"}:
            self.run_search()

    def _tab_changed(self, _index: int) -> None:
        self._apply_active_tab_height()
        if not hasattr(self, "gallery_stack"):
            return
        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        browsing = tab_name in {"Timeline", "All Photos", "Search", "Albums"}
        shows_photo_picker = browsing or tab_name == "Batch Rename & Metadata"
        self.gallery_stack.setVisible(shows_photo_picker)
        self.gallery_footer.setVisible(shows_photo_picker)
        if tab_name not in {"All Photos", "Search", "Albums"}:
            self._asset_auto_loading = False
        if self._timeline_active():
            self.gallery_stack.setCurrentWidget(self.timeline_gallery)
        else:
            self.gallery_stack.setCurrentWidget(self.gallery)
        if self._timeline_active() and not self._timeline_loaded and not self._suppress_surface_load:
            self.load_timeline()
        elif (
            tab_name in {"All Photos", "Batch Rename & Metadata"}
            and not self.gallery.images
            and not self._asset_loading
            and not self._suppress_surface_load
        ):
            self.load_all_photos()
        self._update_controls()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if hasattr(self, "timeline_scroll"):
            self._schedule_timeline_layout()

    def changeEvent(self, event) -> None:  # type: ignore[override]
        super().changeEvent(event)
        if event.type() in {
            QEvent.Type.FontChange,
            QEvent.Type.StyleChange,
            QEvent.Type.LayoutDirectionChange,
        } and hasattr(self, "timeline_scroll"):
            self._schedule_timeline_layout()

    def _timeline_tab_height(self) -> int:
        """Fit the measured filter form, preserving scroll only when needed."""

        self.timeline_form_layout.activate()
        content_height = max(
            72,
            int(self.timeline_form_layout.minimumSize().height()),
            int(self.timeline_form_layout.sizeHint().height()),
        )
        content_height = min(content_height, self._TIMELINE_MAX_SCROLL_CONTENT_HEIGHT)
        frame_width = max(
            0,
            int(self.tabs.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth, None, self.tabs)),
        )
        # QAbstractScrollArea keeps a two-pixel viewport inset even with its
        # visible frame disabled. Reserve it so a one-row form does not grow a
        # needless vertical scrollbar (which would in turn steal width).
        return max(76, content_height + (frame_width * 2) + 2)

    def _apply_active_tab_height(self) -> None:
        """Let simple filter tabs return their space to the photo grid.

        A QTabWidget otherwise adopts the tallest page's size hint, leaving a
        large empty Timeline panel because People Cleanup has a review list.
        Each tab keeps a deliberate, scrollable working height instead.
        """
        tab_name = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        browsing = tab_name in {"Timeline", "All Photos", "Search", "Albums", "Batch Rename & Metadata"}
        if not browsing:
            self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.tabs.setMinimumHeight(280)
            self.tabs.setMaximumHeight(16777215)
            return
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        if tab_name == "Timeline":
            self._refresh_timeline_layout()
            height = self._timeline_tab_height()
        elif tab_name == "Search":
            if self.search_advanced_filters.isVisible() and self.context_results_panel.isVisible():
                height = self._SEARCH_ADVANCED_CONTEXT_TAB_HEIGHT
            elif self.search_advanced_filters.isVisible():
                height = self._SEARCH_ADVANCED_TAB_HEIGHT
            else:
                height = self._SEARCH_CONTEXT_TAB_HEIGHT if self.context_results_panel.isVisible() else self._TAB_HEIGHTS["Search"]
        else:
            height = self._TAB_HEIGHTS.get(tab_name, 270)
        self.tabs.setFixedHeight(height)

    def _duplicate_selection_changed(self) -> None:
        group = self._selected_duplicate_group()
        if group is None:
            self.duplicate_detail.setText("Select a duplicate or burst review group.")
            self.duplicate_preview_heading.setText("Selected-group preview")
            self.duplicate_preview_gallery.set_empty_state("Select a duplicate group to load its virtual thumbnail preview.")
        else:
            self.duplicate_detail.setText(
                f"{group.summary}\n{len(group.members)} photos. Proposed keeper: {Path(group.keeper_path).name}"
            )
            self.duplicate_detail.setToolTip(f"Proposed keeper: {group.keeper_path}")
            self.duplicate_preview_heading.setText(f"Selected-group preview · {len(group.members)} photos")
            self.duplicate_preview_gallery.set_sections(
                [
                    GallerySection(
                        section_id=f"duplicate-review-{self.duplicate_list.currentRow()}",
                        paths=tuple(candidate.image_path for candidate in group.members),
                        kind="duplicate_review",
                        title=f"Proposed keeper: {Path(group.keeper_path).name}",
                    )
                ],
                status=(
                    f"Virtual thumbnail preview · {len(group.members)} photos · "
                    f"keeper: {Path(group.keeper_path).name}"
                ),
            )
        self._update_controls()

    def open_selected_duplicate_group(self) -> None:
        group = self._selected_duplicate_group()
        if group is not None:
            self.open_in_gallery_requested.emit(
                [candidate.image_path for candidate in group.members],
                f"{self._duplicate_kind_label(group.kind)} review: {Path(group.keeper_path).name}",
            )

    def _clear_duplicate_job(self, generation: int) -> None:
        if generation == self._duplicate_generation:
            self._duplicate_job = None

    def _people_selection_changed(self) -> None:
        group = self._selected_people_group()
        if group and group.suggested_name:
            self.people_name_field.setText(group.suggested_name)
        self._update_controls()

    def _clear_people_job(self, generation: int) -> None:
        if generation == self._people_generation:
            self._people_job = None

    def _request_people_name_choices(self, face_service: object) -> None:
        loader = getattr(face_service, "list_known_person_names", None)
        if not callable(loader):
            return
        self._people_name_generation += 1
        generation = self._people_name_generation
        if self._people_name_job is not None:
            self._cancel_job(self._people_name_job)

        def _run(_progress, cancel):
            return list(loader(cancel_check=cancel) or ())

        def _done(names: object) -> None:
            if generation != self._people_name_generation:
                return
            self._people_name_job = None
            values = [str(name) for name in names or ()]
            self.people_name_field.set_choices(values)

        job = self._start_job("Loading saved people names", _run, _done)
        self._people_name_job = job
        job.failed.connect(lambda _message: self._clear_people_name_job(generation))
        job.cancelled.connect(lambda: self._clear_people_name_job(generation))

    def _clear_people_name_job(self, generation: int) -> None:
        if generation == self._people_name_generation:
            self._people_name_job = None

    def _clear_context_search_job(self, generation: int) -> None:
        if generation == self._context_search_generation:
            self._context_search_job = None

    def _on_gallery_paths_renamed(self, changed_paths: list[tuple[str, str]]) -> None:
        changed = [(str(source), str(target)) for source, target in changed_paths if str(source) and str(target)]
        if not changed:
            return
        self.paths_renamed.emit(changed)
        self.status_label.setText(
            f"Renamed {len(changed)} photo(s). Refresh registered roots to update the derived Library catalog; source metadata was not changed."
        )

    def _active_scope(self) -> PathScope:
        try:
            value = self.current_scope_provider()
        except Exception:
            return PathScope()
        if isinstance(value, PathScope):
            return value
        if isinstance(value, (list, tuple, set)):
            return PathScope.from_paths(value)
        return PathScope.from_paths([str(value or "")])

    def _active_catalog_root_ids(self) -> tuple[str, ...]:
        scope = self._active_scope()
        if scope.is_empty:
            return ()
        return tuple(
            root.root_id
            for root in self._roots_cache
            if root.enabled
            if any(
                normalized_path_is_within_scope(root.path, scope_root)
                or normalized_path_is_within_scope(scope_root, root.path)
                for scope_root in scope.roots
            )
        )

    def _catalog_scope(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        scope = self._active_scope()
        return self._active_catalog_root_ids(), scope.roots

    def active_scope_changed(self) -> None:
        """Invalidate only Library views; source cataloging remains explicit."""

        self._catalog_generation += 1
        self._asset_offset = 0
        if self._asset_job is not None:
            self._cancel_job(self._asset_job)
        if self._timeline_job is not None:
            self._cancel_job(self._timeline_job)
        if self._active_scope().is_empty:
            self.status_label.setText("Choose active roots to browse the registered Library catalog.")
            self.gallery.update_gallery([])
            self.timeline_gallery.set_empty_state("Choose active roots, then register and refresh them to browse Library.")
        else:
            self._reload_active_catalog_view()
        self._update_controls()

    def _selected_root_id(self) -> str:
        item = self.root_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    def _selected_album_id(self) -> str:
        item = self.album_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    def _selected_duplicate_group(self) -> DuplicateGroup | None:
        item = self.duplicate_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        return self._duplicate_groups[index] if 0 <= index < len(self._duplicate_groups) else None

    def _selected_people_group(self) -> PeopleCleanupGroup | None:
        item = self.people_list.currentItem()
        if item is None:
            return None
        value = item.data(Qt.ItemDataRole.UserRole)
        try:
            index = int(value)
        except (TypeError, ValueError):
            return None
        return self._people_groups[index] if 0 <= index < len(self._people_groups) else None

    def _timeline_active(self) -> bool:
        return bool(
            hasattr(self, "tabs")
            and self.tabs.count()
            and self.tabs.tabText(self.tabs.currentIndex()) == "Timeline"
        )

    def _open_visible_in_gallery(self) -> None:
        paths = self._timeline_paths if self._timeline_active() else tuple(self.gallery.images)
        if paths:
            self.open_in_gallery_requested.emit(
                list(paths),
                "Library timeline" if self._timeline_active() else "Library search",
            )

    def _update_controls(self) -> None:
        has_root = any(root.enabled for root in self._roots_cache)
        self.scan_button.setEnabled(has_root)
        self.timeline_refresh_dates_button.setEnabled(bool(self._active_catalog_root_ids()))
        self.remove_root_button.setEnabled(bool(self._selected_root_id()))
        self.delete_album_button.setEnabled(bool(self._selected_album_id()))
        browsing = self.tabs.tabText(self.tabs.currentIndex()) in {"Timeline", "All Photos", "Search", "Albums"}
        save_visible = browsing and self._current_query_differs_from_selected_album()
        self.save_album_button.setVisible(save_visible)
        self.save_album_from_tab_button.setVisible(save_visible)
        self._refresh_filter_chips()
        timeline = self._timeline_active()
        self.load_more_button.setVisible(not timeline)
        self.load_more_button.setText("Stop loading" if self._asset_auto_loading else "Load more")
        self.load_more_button.setEnabled(
            (not timeline)
            and (
                self._asset_auto_loading
                or ((not self._asset_loading) and self._asset_offset < self._asset_total)
            )
        )
        visible_paths = self._timeline_paths if timeline else tuple(self.gallery.images)
        self.open_gallery_button.setText("Open timeline photos" if timeline else "Open visible photos")
        self.open_gallery_button.setEnabled(bool(visible_paths))
        duplicate = self._selected_duplicate_group()
        self.duplicate_selection_actions.setVisible(duplicate is not None)
        self.open_duplicate_group_button.setEnabled(duplicate is not None)
        self.not_duplicate_button.setEnabled(duplicate is not None)
        duplicate_busy = self._duplicate_job is not None
        self.find_exact_duplicates_button.setEnabled(not duplicate_busy)
        self.find_similar_duplicates_button.setEnabled(not duplicate_busy)
        self.find_duplicates_button.setEnabled(not duplicate_busy)
        self.find_bursts_button.setEnabled(not duplicate_busy)
        self.trash_exact_duplicates_button.setEnabled(
            bool(self._duplicate_candidate_paths("exact")) and not self._read_only_mode and not duplicate_busy
        )
        self.trash_similar_duplicates_button.setEnabled(
            bool(self._duplicate_candidate_paths("near")) and not self._read_only_mode and not duplicate_busy
        )
        people = self._selected_people_group()
        self.people_apply_button.setEnabled(people is not None and bool(self.people_name_field.text().strip()) and not self._read_only_mode)
        self.people_reject_button.setEnabled(people is not None and bool(people.suggested_name))
        self.people_split_button.setEnabled(people is not None and self._people_service is not None)
        self.people_hide_button.setEnabled(people is not None and not self._read_only_mode)
        self.people_merge_button.setEnabled(self._people_service is not None and not self._read_only_mode)
        self.describe_cluster_button.setEnabled(bool(self._selected_cluster_members))

    def _current_query_differs_from_selected_album(self) -> bool:
        album_id = self._selected_album_id()
        if not album_id:
            return True
        album = next((item for item in self._albums_cache if item.album_id == album_id), None)
        if album is None:
            return True
        current = self.catalog.query_to_payload(self._active_query)
        saved = dict(album.query)
        for payload in (current, saved):
            payload.pop("offset", None)
            payload.pop("limit", None)
        return current != saved

    def _refresh_filter_chips(self) -> None:
        if not hasattr(self, "filter_chips_layout"):
            return
        tab_name = self.tabs.tabText(self.tabs.currentIndex())
        values = {"start": "", "end": "", "camera": "", "search": ""}
        if tab_name == "Timeline":
            values.update(
                {
                    "start": f"From: {self.timeline_start.text().strip()}" if self.timeline_start.text().strip() else "",
                    "end": f"To: {self.timeline_end.text().strip()}" if self.timeline_end.text().strip() else "",
                    "camera": f"Camera: {self.timeline_camera.text().strip()}" if self.timeline_camera.text().strip() else "",
                }
            )
        elif tab_name in {"Search", "Albums"} and self.search_field.text().strip():
            values["search"] = f"Search: {self.search_field.text().strip()}"
        any_visible = False
        for key, button in self.filter_chip_buttons.items():
            text = values[key]
            button.setText(f"{text}  ×")
            button.setToolTip(f"Remove {text} and refresh this view." if text else "")
            button.setVisible(bool(text))
            any_visible = any_visible or bool(text)
        self.filter_chips_widget.setVisible(any_visible)

    def _clear_timeline_filter(self, field: QLineEdit) -> None:
        field.clear()
        self.load_timeline()

    def _clear_search_filter(self) -> None:
        self.search_field.clear()
        for field in (
            self.search_start,
            self.search_end,
            self.search_camera,
            self.search_folder,
            self.search_file_ext,
        ):
            field.clear()
        self.run_search()

    def _coordination_spec(self, title: str) -> JobSpec:
        normalized = str(title).strip()
        lowered = normalized.casefold()
        roots = tuple(str(path) for path in self._active_scope().roots)
        writes_data_home = (
            normalized
            in {
                "Describing New Clusters",
                "Updating Library Root",
                "Removing Library Root",
                "Registering Library Roots",
                "Refreshing Library",
                "Saving Library Smart Album",
                "Deleting Library Smart Album",
                "Saving Duplicate Review Feedback",
                "Naming People Cleanup Faces",
                "Rejecting People Suggestion",
                "Splitting People Group",
                "Hiding People Faces",
                "Merging People Names",
                "Describing Cluster Context",
                "Clearing Cluster Context",
            }
            or lowered.startswith("trashing ")
        )
        reads_sources = (
            normalized in {"Describing New Clusters", "Refreshing Library", "Describing Cluster Context"}
            or lowered.startswith("finding ")
        )
        writes_sources = lowered.startswith("trashing ") or normalized == "Naming People Cleanup Faces"
        if lowered.startswith("trashing ") or "duplicate" in lowered:
            origin = "Tools"
        elif "people" in lowered or "face" in lowered:
            origin = "People"
        elif "cluster context" in lowered or normalized == "Describing New Clusters":
            origin = "Organize"
        else:
            origin = "Library"
        return JobSpec(
            normalized,
            origin=origin,
            io_bound=True,
            source_reads=roots if reads_sources else (),
            source_writes=roots if writes_sources else (),
            data_home_read=not writes_data_home,
            data_home_write=writes_data_home,
        )

    def _cancel_job(self, job: AsyncJob | None) -> None:
        if job is None:
            return
        coordinated_job_id = self._coordinated_job_ids.get(job)
        if coordinated_job_id is not None and self.work_coordinator is not None:
            self.work_coordinator.cancel(coordinated_job_id)
            return
        job.cancel()

    def _qobject_alive(self) -> bool:
        try:
            self.thread()
        except RuntimeError:
            return False
        return True

    def _set_status_safe(self, text: str) -> None:
        if not self._qobject_alive():
            return
        try:
            self.status_label.setText(str(text))
        except RuntimeError:
            return

    def _start_job(self, title: str, run, done) -> AsyncJob:
        job = AsyncJob(run)
        coordinated = self.work_coordinator is not None
        job_id = (
            self.job_manager.register_job(title, cancel_fn=job.cancel, origin="Library")
            if self.job_manager and not coordinated
            else None
        )
        holder: dict[str, object] = {}

        def _finish(status: str, error: str = "") -> None:
            self._jobs[:] = [pair for pair in self._jobs if pair[0] is not job]
            self._coordinated_job_ids.pop(job, None)
            if job_id is not None and self.job_manager:
                self.job_manager.finish(job_id, status=status, error=error)

        job.progress.connect(lambda value, text: self.job_manager.update(job_id, progress=value, text=text) if self.job_manager and job_id is not None else None)
        job.progress.connect(lambda _value, text: self._set_status_safe(str(text)))
        job.completed.connect(done)
        job.completed.connect(lambda _result: _finish("finished"))
        job.failed.connect(lambda message: self._set_status_safe(f"{title} failed: {message}"))
        job.failed.connect(lambda message: _finish("failed", str(message)))
        job.cancelled.connect(lambda: self._set_status_safe(f"{title} cancelled."))
        job.cancelled.connect(lambda: _finish("cancelled"))
        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            holder["thread"] = thread
            self._jobs.append((job, thread))

        if self.work_coordinator is not None:
            coordinated_job_id = self.work_coordinator.submit_async_job(
                self._coordination_spec(title),
                job,
                _launch,
            )
            state = self.job_manager.get(coordinated_job_id) if self.job_manager is not None else None
            if state is not None and state.status in {"queued", "running", "cancelling"}:
                self._coordinated_job_ids[job] = coordinated_job_id
        else:
            _launch()
        return job

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready = True
        for job in list(self._coordinated_job_ids):
            self._cancel_job(job)
        for job, _thread in list(self._jobs):
            if job not in self._coordinated_job_ids:
                job.cancel()
        for job, thread in list(self._jobs):
            ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
        if ready:
            if self.work_coordinator is not None:
                for coordinated_job_id in tuple(self._coordinated_job_ids.values()):
                    self.work_coordinator.finish(coordinated_job_id, status="cancelled")
            for job, _thread in list(self._jobs):
                for signal_name in ("started", "progress", "completed", "failed", "cancelled"):
                    try:
                        getattr(job, signal_name).disconnect()
                    except (TypeError, RuntimeError):
                        pass
                defer_async_job_dispose(job)
            self._coordinated_job_ids.clear()
            self._jobs.clear()
        gallery_ready = self.gallery.shutdown_jobs(timeout_ms=timeout_ms)
        timeline_ready = self.timeline_gallery.shutdown_jobs(timeout_ms=timeout_ms)
        return gallery_ready and timeline_ready and ready
