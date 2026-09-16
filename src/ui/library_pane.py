from __future__ import annotations

from calendar import month_name
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.path_scope import PathScope, path_is_within_scope
from app.services.cluster_context import ClusterContextService, VisionLanguageSettings
from app.services.duplicate_review import DuplicateGroup, DuplicateReviewService
from app.services.library_catalog import CatalogQuery, CatalogTimeline, ClusterContextRecord, LibraryCatalogService, SmartAlbum
from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.error_mbox import confirmBox, errorBox
from ui.entity_picker import EntityPicker, EntityPickerDialog
from ui.gallery_pane import GalleryPane
from ui.job_manager import JobManager
from ui.sectioned_gallery import GallerySection, SectionedGallery

if TYPE_CHECKING:
    from app.services.people_cleanup import PeopleCleanupGroup, PeopleCleanupService


class LibraryPane(QWidget):
    """Local archive curation workspace backed entirely by managed runtime data."""

    open_in_gallery_requested = pyqtSignal(list, str)
    metadata_changed = pyqtSignal(list)
    paths_renamed = pyqtSignal(list)

    PAGE_SIZE = 240
    _TAB_HEIGHTS = {
        "Timeline": 148,
        "Search": 236,
        "Cleanup": 270,
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
        context_settings_provider: Callable[[], VisionLanguageSettings] | None = None,
        context_settings_changed: Callable[[VisionLanguageSettings, bool], None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.catalog = catalog or LibraryCatalogService()
        self.current_scope_provider = current_scope_provider
        self.face_service_provider = face_service_provider
        self.job_manager = job_manager
        self.context_settings_provider = context_settings_provider
        self.context_settings_changed = context_settings_changed
        self.context_service = ClusterContextService(self.catalog)
        self.duplicate_service = DuplicateReviewService(catalog=self.catalog)
        self._people_service: PeopleCleanupService | None = None
        self._jobs: list[tuple[AsyncJob, object]] = []
        self._catalog_generation = 0
        self._asset_request_generation = 0
        self._asset_job: AsyncJob | None = None
        self._asset_loading = False
        self._timeline_request_generation = 0
        self._timeline_job: AsyncJob | None = None
        self._timeline_loading = False
        self._timeline_paths: tuple[str, ...] = ()
        self._timeline_total = 0
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
        self._build_ui()
        self._restore_context_settings()
        self.refresh_roots()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        # The production shell can show this workspace at its 720 px compact
        # width.  Give text and focus rings a protected edge rather than
        # letting controls touch the splitter/window boundary.
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Library")
        title.setProperty("role", "section")
        header.addWidget(title)
        header.addStretch(1)
        self.scan_button = QPushButton("Refresh registered roots")
        self.scan_button.setToolTip("Incrementally scan registered photo roots in a background Job. Source files are read only.")
        self.scan_button.clicked.connect(self.scan_roots)
        header.addWidget(self.scan_button)
        layout.addLayout(header)
        self.status_label = QLabel("Add a library root to build a local archive catalog.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.start_here_label = QLabel(
            "Start here: add a source folder, refresh its local catalog, then browse Timeline or Search. "
            "Library only reads photo metadata until you explicitly name faces or send selected duplicate candidates to Trash."
        )
        self.start_here_label.setProperty("role", "help")
        self.start_here_label.setWordWrap(True)
        self.start_here_label.setToolTip(
            "Registered roots are the only folders included in Library, duplicate review, and People Cleanup. "
            "Catalog refreshes run in Jobs and do not edit your photos."
        )
        layout.addWidget(self.start_here_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        sidebar = QWidget(splitter)
        sidebar.setMinimumWidth(270)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        side.setSpacing(6)
        roots_heading = QLabel("Registered library roots")
        roots_heading.setToolTip("Only checked roots participate in Library-wide work.")
        side.addWidget(roots_heading)
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
        albums_heading = QLabel("Smart albums")
        albums_heading.setToolTip("Saved Library filters. Albums contain no copied photos and update as the catalog changes.")
        side.addWidget(albums_heading)
        self.album_list = QListWidget(sidebar)
        self.album_list.itemSelectionChanged.connect(self._album_selection_changed)
        side.addWidget(self.album_list, stretch=1)
        album_buttons = QVBoxLayout()
        album_buttons.setSpacing(6)
        self.save_album_button = QPushButton("Save current search")
        self.delete_album_button = QPushButton("Delete selected album")
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
        self._build_search_tab()
        self._build_cleanup_tab()
        self._build_people_tab()
        self._build_context_tab()
        self.gallery = GalleryPane(body)
        self.gallery.job_manager = self.job_manager
        self.gallery.set_action_visibility(show_actions=True, show_metadata_actions=True, show_file_actions=True)
        self.gallery.set_empty_state("Library timeline", "Register and refresh a library root to browse local photos.")
        self.gallery.metadata_changed.connect(self.metadata_changed.emit)
        self.gallery.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.timeline_gallery = SectionedGallery(body)
        self.timeline_gallery.set_job_manager(self.job_manager)
        self.timeline_gallery.set_embedded_timeline_mode(True)
        self.timeline_gallery.set_empty_state("Register and refresh a library root to browse the full local timeline.")
        self.timeline_gallery.metadata_changed.connect(self.metadata_changed.emit)
        self.timeline_gallery.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.gallery_stack = QStackedWidget(body)
        self.gallery_stack.addWidget(self.timeline_gallery)
        self.gallery_stack.addWidget(self.gallery)
        body_layout.addWidget(self.gallery_stack, stretch=1)
        gallery_footer = QHBoxLayout()
        self.timeline_count = QLabel("0 photos")
        self.load_more_button = QPushButton("Load more")
        self.open_gallery_button = QPushButton("Open visible photos in Gallery")
        self.load_more_button.clicked.connect(self.load_more_assets)
        self.open_gallery_button.clicked.connect(self._open_visible_in_gallery)
        gallery_footer.addWidget(self.timeline_count)
        gallery_footer.addStretch(1)
        gallery_footer.addWidget(self.load_more_button)
        gallery_footer.addWidget(self.open_gallery_button)
        body_layout.addLayout(gallery_footer)
        splitter.addWidget(body)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([300, 900])
        layout.addWidget(splitter, stretch=1)
        self._apply_active_tab_height()
        self._update_controls()

    def _build_timeline_tab(self) -> None:
        page = QWidget(self.tabs)
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        start_label = QLabel("From")
        end_label = QLabel("To")
        camera_label = QLabel("Camera")
        date_source_label = QLabel("Timeline date")
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
        self.timeline_date_source.addItem("Metadata, then filename", "metadata_or_filename")
        self.timeline_date_source.addItem("Metadata only", "metadata_only")
        self.timeline_date_source.addItem("Prefer filename", "prefer_filename")
        self.timeline_date_source.addItem("Filename only", "filename_only")
        current_policy = str(getattr(self.catalog, "filename_date_policy", "metadata_or_filename"))
        current_index = self.timeline_date_source.findData(current_policy)
        self.timeline_date_source.setCurrentIndex(max(0, current_index))
        self.timeline_date_source.setToolTip(
            "Choose the source used for Timeline capture time. Filename recognition accepts only unambiguous year-first dates. "
            "Changing this re-reads catalog metadata on the next Refresh; it never writes EXIF or XMP."
        )
        self.timeline_reload_button = QPushButton("Show timeline")
        self.timeline_reload_button.setToolTip("Apply the date and camera filters to the selected Library root.")
        self.timeline_reload_button.setProperty("kind", "primary")
        self.timeline_reload_button.clicked.connect(self.load_timeline)
        self.timeline_date_source.currentIndexChanged.connect(self._on_timeline_date_source_changed)
        for field in (self.timeline_start, self.timeline_end, self.timeline_camera):
            field.setMinimumWidth(122)
        layout.addWidget(start_label, 0, 0)
        layout.addWidget(self.timeline_start, 0, 1)
        layout.addWidget(end_label, 0, 2)
        layout.addWidget(self.timeline_end, 0, 3)
        layout.addWidget(camera_label, 1, 0)
        layout.addWidget(self.timeline_camera, 1, 1, 1, 2)
        layout.addWidget(self.timeline_reload_button, 1, 3)
        layout.addWidget(date_source_label, 2, 0)
        layout.addWidget(self.timeline_date_source, 2, 1, 1, 3)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(3, 1)
        self.tabs.addTab(page, "Timeline")

    def _build_search_tab(self) -> None:
        page = QWidget(self.tabs)
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
        self.context_results = QListWidget(page)
        self.context_results.setMaximumHeight(126)
        self.context_results.setToolTip("Generated cluster-context matches. Select one to open the current cluster members in Gallery.")
        self.context_results.itemDoubleClicked.connect(lambda _item: self.open_selected_context())
        layout.addWidget(self.context_results)
        self.open_context_button = QPushButton("Open selected context cluster")
        self.open_context_button.setToolTip("Open the photos belonging to the selected generated context in the top Gallery.")
        self.open_context_button.clicked.connect(self.open_selected_context)
        layout.addWidget(self.open_context_button)
        self.tabs.addTab(page, "Search")

    def _build_cleanup_tab(self) -> None:
        page = QWidget(self.tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        actions = QGridLayout()
        actions.setHorizontalSpacing(8)
        actions.setVerticalSpacing(6)
        self.find_duplicates_button = QPushButton("Find duplicates and bursts")
        self.find_duplicates_button.setToolTip("Build review-only exact, near-duplicate, and burst groups in a background Job. Nothing is deleted automatically.")
        self.find_duplicates_button.setProperty("kind", "primary")
        self.find_duplicates_button.clicked.connect(self.find_duplicates)
        self.not_duplicate_button = QPushButton("Mark selected not duplicate")
        self.not_duplicate_button.setToolTip("Record that the selected proposed group is not a duplicate. It is reconsidered only after a source revision.")
        self.not_duplicate_button.clicked.connect(self.mark_selected_not_duplicate)
        self.trash_duplicates_button = QPushButton("Trash selected candidates")
        self.trash_duplicates_button.setToolTip("Move only the selected group's proposed non-keeper photos to recoverable ClusterLens Trash after confirmation.")
        self.trash_duplicates_button.setProperty("kind", "danger")
        self.trash_duplicates_button.clicked.connect(self.trash_selected_duplicate_candidates)
        actions.addWidget(self.find_duplicates_button, 0, 0, 1, 2)
        actions.addWidget(self.not_duplicate_button, 1, 0)
        actions.addWidget(self.trash_duplicates_button, 1, 1)
        actions.setColumnStretch(0, 1)
        actions.setColumnStretch(1, 1)
        layout.addLayout(actions)
        self.duplicate_list = QListWidget(page)
        self.duplicate_list.setToolTip("Exact, near, and burst groups. The proposed keeper is never moved automatically.")
        self.duplicate_list.itemSelectionChanged.connect(self._duplicate_selection_changed)
        layout.addWidget(self.duplicate_list)
        self.duplicate_detail = QLabel("Run duplicate review to prepare manual cleanup candidates.")
        self.duplicate_detail.setWordWrap(True)
        layout.addWidget(self.duplicate_detail)
        self.tabs.addTab(page, "Cleanup")

    def _build_people_tab(self) -> None:
        page = QWidget(self.tabs)
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
        review_actions = QGridLayout()
        review_actions.setHorizontalSpacing(8)
        review_actions.setVerticalSpacing(6)
        review_actions.addWidget(self.people_reject_button, 0, 0)
        review_actions.addWidget(self.people_split_button, 0, 1)
        review_actions.addWidget(self.people_hide_button, 1, 0)
        review_actions.addWidget(self.people_merge_button, 1, 1)
        review_actions.setColumnStretch(0, 1)
        review_actions.setColumnStretch(1, 1)
        layout.addLayout(review_actions)
        self.people_summary = QLabel("Open Faces once, then build a registered-root People Cleanup Inbox.")
        self.people_summary.setWordWrap(True)
        layout.addWidget(self.people_summary)
        self.people_list = QListWidget(page)
        self.people_list.itemSelectionChanged.connect(self._people_selection_changed)
        layout.addWidget(self.people_list)
        self.tabs.addTab(page, "People Cleanup")

    def _build_context_tab(self) -> None:
        page = QWidget(self.tabs)
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
        self.context_remote_consent = QCheckBox("I consent to send the representative photo and displayed EXIF/XMP to this remote endpoint for this run", page)
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
        selected_id = self._selected_root_id()
        self.root_list.blockSignals(True)
        self.root_list.clear()
        for root in self.catalog.list_roots():
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
        self.refresh_albums()
        self._update_controls()

    def refresh_albums(self) -> None:
        selected_id = self._selected_album_id()
        self.album_list.clear()
        for album in self.catalog.list_smart_albums():
            item = QListWidgetItem(album.name)
            item.setData(Qt.ItemDataRole.UserRole, album.album_id)
            item.setToolTip("Dynamic saved Library query")
            self.album_list.addItem(item)
            if album.album_id == selected_id:
                self.album_list.setCurrentItem(item)
        self._update_controls()

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
        directories = list(PathScope.from_paths(directories).roots)
        if not directories:
            return
        self.status_label.setText("Registering library root…")

        def _run(progress, cancel):
            roots = []
            total = len(directories)
            for index, directory in enumerate(directories, start=1):
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
        self._start_job(
            "Refreshing Library",
            lambda progress, cancel: self.catalog.scan_roots(
                root_ids,
                force=force,
                progress_callback=progress,
                cancel_check=cancel,
            ),
            lambda result: self._catalog_scan_completed(result),
        )

    def _on_timeline_date_source_changed(self, _index: int) -> None:
        policy = str(self.timeline_date_source.currentData() or "metadata_or_filename")
        setter = getattr(self.catalog, "set_filename_date_policy", None)
        if callable(setter):
            setter(policy)
        self._force_catalog_date_refresh = True
        self.status_label.setText("Timeline date source changed. Refresh registered roots to update derived timeline dates.")

    def _catalog_scan_completed(self, result: dict[str, int]) -> None:
        self.status_label.setText(
            f"Catalog refreshed: {int(result.get('updated', 0))} updated, {int(result.get('unchanged', 0))} unchanged, {int(result.get('removed', 0))} removed."
        )
        self.refresh_roots()
        self._reload_active_catalog_view()

    def load_timeline(self) -> None:
        self._timeline_loaded = True
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
            self._asset_job.cancel()
            self._asset_loading = False
        if self._timeline_job is not None:
            self._timeline_job.cancel()
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
            self._timeline_paths = timeline.image_paths
            self._timeline_total = timeline.total_count
            sections, collapsed = self._timeline_sections(timeline)
            years = len(timeline.years)
            if sections:
                self.timeline_gallery.set_sections_with_collapsed(
                    sections,
                    collapsed_section_ids=collapsed,
                    status=(
                        f"{timeline.total_count:,} catalogued photos in {years} year"
                        f"{'s' if years != 1 else ''}. Newest month is expanded."
                    ),
                )
            else:
                self.timeline_gallery.set_empty_state("No catalog photos match this timeline filter.")
            self.timeline_count.setText(
                f"Showing {timeline.total_count:,} catalogued photo"
                f"{'s' if timeline.total_count != 1 else ''} in {years} year{'s' if years != 1 else ''}"
            )
            self.status_label.setText("Library timeline ready." if timeline.total_count else "No catalog photos match this view.")
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

    @staticmethod
    def _timeline_sections(timeline: CatalogTimeline) -> tuple[list[GallerySection], set[str]]:
        """Map ordered catalog buckets to virtual nested Year → Month headers."""

        sections: list[GallerySection] = []
        collapsed: set[str] = set()
        for year_index, year in enumerate(timeline.years):
            year_id = f"timeline:year:{year.year:04d}"
            if year_index:
                collapsed.add(year_id)
            months: list[GallerySection] = []
            for month_index, month in enumerate(year.months):
                month_id = f"{year_id}:month:{month.month:02d}"
                if year_index or month_index:
                    collapsed.add(month_id)
                if year.year > 0 and 1 <= month.month <= 12:
                    title = f"{month_name[month.month]} {year.year}"
                else:
                    title = "Unknown capture date"
                months.append(
                    GallerySection(
                        month_id,
                        tuple(month.image_paths),
                        kind="timeline_month",
                        title=title,
                    )
                )
            sections.append(
                GallerySection(
                    year_id,
                    (),
                    kind="timeline_year",
                    title=str(year.year) if year.year > 0 else "Unknown date",
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
            self._asset_job.cancel()
        if self._timeline_job is not None:
            self._timeline_job.cancel()
        self._catalog_generation += 1
        self._asset_request_generation += 1
        self._timeline_request_generation += 1
        self._timeline_loaded = False
        self._asset_loading = False
        self._timeline_loading = False
        self._asset_offset = 0
        self._asset_total = 0
        self._timeline_paths = ()
        self._timeline_total = 0
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
            self._timeline_job.cancel()
        self._timeline_loading = False
        self._context_search_generation += 1
        context_generation = self._context_search_generation
        if self._context_search_job is not None:
            self._context_search_job.cancel()
            self._context_search_job = None
        self._asset_offset = 0
        root_ids, scope_paths = self._catalog_scope()
        self._active_query = CatalogQuery(root_ids=root_ids, scope_paths=scope_paths, text=text, order="captured_desc", offset=0, limit=self.PAGE_SIZE)
        self.context_results.clear()
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
            self._asset_job.cancel()
        if self._timeline_job is not None:
            self._timeline_request_generation += 1
            self._timeline_job.cancel()
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
            self.timeline_count.setText(f"Showing {len(self.gallery.images)} of {page.total_count} photos")
            self.status_label.setText("Library catalog ready." if page.total_count else "No catalog photos match this view.")
            self._update_controls()

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
        if self._timeline_active() or self._asset_loading or self._asset_offset >= self._asset_total:
            return
        self._load_assets(reset=False)

    def save_current_album(self) -> None:
        name, accepted = _simple_text_prompt(self, "Save smart album", "Album name")
        if not accepted:
            return
        query = self._active_query

        def _done(_album) -> None:
            self.refresh_albums()
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
            self.refresh_albums()
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
        album = self.catalog.get_smart_album(album_id)
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

    def find_duplicates(self) -> None:
        self._duplicate_generation += 1
        generation = self._duplicate_generation
        if self._duplicate_job is not None:
            self._duplicate_job.cancel()
        self.status_label.setText("Finding duplicate and burst candidates…")
        root_ids, scope_paths = self._catalog_scope()
        if not scope_paths:
            self.status_label.setText("Choose active roots before finding duplicates.")
            return

        def _run(progress, cancel):
            return self.duplicate_service.build_groups(
                root_ids=root_ids,
                scope_paths=scope_paths,
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
                item = QListWidgetItem(f"{group.kind.title()} · {len(group.members)} photos · keeper: {Path(group.keeper_path).name}")
                item.setData(Qt.ItemDataRole.UserRole, index)
                item.setToolTip(group.summary)
                self.duplicate_list.addItem(item)
            self.status_label.setText(f"Prepared {len(self._duplicate_groups)} manual duplicate/burst review groups.")
            self._duplicate_selection_changed()
            self._update_controls()

        job = self._start_job("Finding Duplicates", _run, _done)
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
        candidates = [member.image_path for member in group.members if member.image_path != group.keeper_path]
        if not candidates:
            return
        if self._read_only_mode:
            errorBox("Read-only safety mode", "Turn off read-only safety mode before moving files to Trash.", parent=self)
            return
        if not confirmBox("Move selected duplicate candidates to Trash?", f"The proposed keeper stays untouched. {len(candidates)} selected candidates will move to recoverable ClusterLens Trash.", parent=self):
            return

        def _run(progress, cancel):
            return self.gallery.action_service.move_to_trash(candidates, progress_callback=progress, cancel_check=cancel)

        def _done(result) -> None:
            moved = [source for source, _target in result.changed_paths]
            self.metadata_changed.emit(moved)
            self.status_label.setText(f"Moved {len(moved)} duplicate candidates to ClusterLens Trash.")
            self.scan_roots()

        self._start_job("Trashing Duplicate Candidates", _run, _done)

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
            self._people_job.cancel()
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

    def _reload_active_catalog_view(self) -> None:
        if self._timeline_active():
            self.load_timeline()
        elif self.tabs.tabText(self.tabs.currentIndex()) == "Search":
            self.run_search()

    def _tab_changed(self, _index: int) -> None:
        self._apply_active_tab_height()
        if not hasattr(self, "gallery_stack"):
            return
        if self._timeline_active():
            self.gallery_stack.setCurrentWidget(self.timeline_gallery)
        else:
            self.gallery_stack.setCurrentWidget(self.gallery)
        if self._timeline_active() and not self._timeline_loaded:
            self.load_timeline()
        self._update_controls()

    def _apply_active_tab_height(self) -> None:
        """Let simple filter tabs return their space to the photo grid.

        A QTabWidget otherwise adopts the tallest page's size hint, leaving a
        large empty Timeline panel because People Cleanup has a review list.
        Each tab keeps a deliberate, scrollable working height instead.
        """
        tab_name = self.tabs.tabText(self.tabs.currentIndex()) if self.tabs.count() else ""
        self.tabs.setFixedHeight(self._TAB_HEIGHTS.get(tab_name, 270))

    def _duplicate_selection_changed(self) -> None:
        group = self._selected_duplicate_group()
        self.duplicate_detail.setText(group.summary + f"\nProposed keeper: {group.keeper_path}" if group else "Select a duplicate or burst review group.")
        self._update_controls()

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
            self._people_name_job.cancel()

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
            for root in self.catalog.list_roots(include_disabled=False)
            if any(
                path_is_within_scope(root.path, scope_root) or path_is_within_scope(scope_root, root.path)
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
            self._asset_job.cancel()
        if self._timeline_job is not None:
            self._timeline_job.cancel()
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
        index = int(item.data(Qt.ItemDataRole.UserRole) or -1)
        return self._duplicate_groups[index] if 0 <= index < len(self._duplicate_groups) else None

    def _selected_people_group(self) -> PeopleCleanupGroup | None:
        item = self.people_list.currentItem()
        if item is None:
            return None
        index = int(item.data(Qt.ItemDataRole.UserRole) or -1)
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
        has_root = bool(self.catalog.list_roots(include_disabled=False))
        self.scan_button.setEnabled(has_root)
        self.remove_root_button.setEnabled(bool(self._selected_root_id()))
        self.delete_album_button.setEnabled(bool(self._selected_album_id()))
        timeline = self._timeline_active()
        self.load_more_button.setVisible(not timeline)
        self.load_more_button.setEnabled((not timeline) and (not self._asset_loading) and self._asset_offset < self._asset_total)
        visible_paths = self._timeline_paths if timeline else tuple(self.gallery.images)
        self.open_gallery_button.setText("Open timeline photos in Gallery" if timeline else "Open visible photos in Gallery")
        self.open_gallery_button.setEnabled(bool(visible_paths))
        duplicate = self._selected_duplicate_group()
        self.not_duplicate_button.setEnabled(duplicate is not None)
        self.trash_duplicates_button.setEnabled(duplicate is not None and not self._read_only_mode)
        people = self._selected_people_group()
        self.people_apply_button.setEnabled(people is not None and bool(self.people_name_field.text().strip()) and not self._read_only_mode)
        self.people_reject_button.setEnabled(people is not None and bool(people.suggested_name))
        self.people_split_button.setEnabled(people is not None and self._people_service is not None)
        self.people_hide_button.setEnabled(people is not None and not self._read_only_mode)
        self.people_merge_button.setEnabled(self._people_service is not None and not self._read_only_mode)
        self.describe_cluster_button.setEnabled(bool(self._selected_cluster_members))

    def _start_job(self, title: str, run, done) -> AsyncJob:
        job = AsyncJob(run)
        job_id = self.job_manager.register_job(title, cancel_fn=job.cancel, origin="Library") if self.job_manager else None
        holder: dict[str, object] = {}

        def _finish(status: str, error: str = "") -> None:
            self._jobs[:] = [pair for pair in self._jobs if pair[0] is not job]
            if job_id is not None and self.job_manager:
                self.job_manager.finish(job_id, status=status, error=error)

        job.progress.connect(lambda value, text: self.job_manager.update(job_id, progress=value, text=text) if self.job_manager and job_id is not None else None)
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))
        job.completed.connect(done)
        job.completed.connect(lambda _result: _finish("finished"))
        job.failed.connect(lambda message: self.status_label.setText(f"{title} failed: {message}"))
        job.failed.connect(lambda message: _finish("failed", str(message)))
        job.cancelled.connect(lambda: self.status_label.setText(f"{title} cancelled."))
        job.cancelled.connect(lambda: _finish("cancelled"))
        thread = start_job_in_thread(job)
        holder["thread"] = thread
        self._jobs.append((job, thread))
        return job

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready = True
        for job, _thread in list(self._jobs):
            job.cancel()
        for job, thread in list(self._jobs):
            ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
            if ready:
                job.dispose()
        if ready:
            self._jobs.clear()
        gallery_ready = self.gallery.shutdown_jobs(timeout_ms=timeout_ms)
        timeline_ready = self.timeline_gallery.shutdown_jobs(timeout_ms=timeout_ms)
        return gallery_ready and timeline_ready and ready
