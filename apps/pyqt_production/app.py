from __future__ import annotations

from dataclasses import dataclass, replace
import json
import logging
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING
import uuid

from PyQt6 import sip
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QMenu, QPushButton, QSplitter, QStackedWidget, QToolButton, QVBoxLayout, QWidget

from apps.pyqt_production.bootstrap import bootstrap_runtime
from apps.pyqt_production.first_run import FirstRunSetupDialog
from apps.pyqt_production.icon_assets import configure_linux_desktop_entry, production_app_icon, production_icon_path
from apps.pyqt_production.identity import PRODUCTION_DISPLAY_NAME, PRODUCTION_QSETTINGS_ORG, production_settings_store
from apps.pyqt_production.model_download_controller import ModelDownloadController
from apps.pyqt_production.session_controller import ClusteringSessionController
from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
from apps.pyqt_production.worker_protocol import ProductionClusterRequest
from apps.shared.runtime_migrations import RuntimeMigrationService
from apps.shared.runtime_support import RuntimeLayout, configure_rotating_logging, flush_logging_handlers, install_crash_handlers


RUNTIME_LAYOUT, _REPO_ROOT = bootstrap_runtime()
configure_rotating_logging(RUNTIME_LAYOUT)
install_crash_handlers(RUNTIME_LAYOUT)

from app.selection import SelectionTarget  # noqa: E402
from app.path_scope import PathScope  # noqa: E402
from app.services.cache_maintenance import CacheClearResult, CacheMaintenanceService, RuntimeStorageSummary  # noqa: E402
from app.services.data_home import DataHomeManager  # noqa: E402
from app.services.cluster_explanations import ClusterExplanation  # noqa: E402
from app.services.cluster_context import VisionLanguageSettings  # noqa: E402
from app.services.cluster_meanings import ClusterMeaning  # noqa: E402
from app.services.clustering_options import model_label, normalize_clustering_backends, normalize_embedding_models  # noqa: E402
from app.services.face_model_installer import FaceModelInstaller  # noqa: E402
from app.services.model_assets import MODEL_SOURCE_LABELS, TEXT_MODEL_ORDER, ModelAssetService, ModelDownloadPlan  # noqa: E402
from app.services.model_downloads import ModelDownloadItem, ModelDownloadService, normalize_model_download_items  # noqa: E402
from app.services.image_tags import ClusterTagSummary, ImageTagService  # noqa: E402
from app.services.saved_searches import SavedSearchService  # noqa: E402
from app.services.photo_metadata import PhotoMetadataService  # noqa: E402
from app.services.startup_readiness import StartupReadinessReport, inspect_startup_readiness  # noqa: E402
from infra.performance import apply_performance_overrides, detect_system_resources, select_performance_profile  # noqa: E402
from infra.qt_diagnostics import install_qt_message_handler  # noqa: E402
from infra.runtime import ExecutionPolicy, RuntimeCapabilityService  # noqa: E402
from infra.settings import get_production_settings_registry, get_settings  # noqa: E402
from ui.cluster_pane import ClusterPane  # noqa: E402
from ui.error_mbox import confirmBox, errorBox, infoBox  # noqa: E402
from ui.footer_bar import WorkspaceFooter  # noqa: E402
from ui.gallery_pane import GalleryPane  # noqa: E402
from ui.sectioned_gallery import GallerySection, SectionedGallery  # noqa: E402
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown  # noqa: E402
from ui.job_manager import JobManager  # noqa: E402
from ui.job_presentation import JobPresentationController  # noqa: E402
from ui.job_widgets import JobIndicatorWidget  # noqa: E402
from ui.work_coordinator import WorkCoordinator  # noqa: E402
from ui.library_pane import LibraryPane  # noqa: E402
from ui.mode_panes import ClusteringOptionsPane, SourcePane  # noqa: E402
from ui.recent_folders import RecentFolderHistory  # noqa: E402
from ui.runtime_widgets import RuntimeBadge  # noqa: E402
from ui.tags_pane import TagsPane  # noqa: E402
from ui.icons import apply_icon  # noqa: E402
from ui.theme import apply_ultra_dark  # noqa: E402


configure_rotating_logging(RUNTIME_LAYOUT, force=True)
LOGGER = logging.getLogger(__name__)
MIN_SUPPORTED_SCREEN_WIDTH = 1920
MIN_SUPPORTED_SCREEN_HEIGHT = 1080
COMPACT_LAYOUT_WIDTH = 1440
PACKAGED_LAUNCH_SMOKE_ENV = "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE"
PACKAGED_LAUNCH_REPORT_ENV = "CLUSTERLENS_PACKAGED_LAUNCH_REPORT"
PACKAGED_LAUNCH_SMOKE_SCENARIO = "settings-storage"
# A top-level Qt widget has no QObject parent.  Retain a close-pending window
# while its event-loop-driven cancellation drain owns live worker signals.
_CLOSE_PENDING_WINDOWS: list["ProductionClusterApp"] = []

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService


_FACE_SEARCH_MODULE = None


def _face_search_api():
    global _FACE_SEARCH_MODULE
    if _FACE_SEARCH_MODULE is None:
        from app.services import face_search as face_search_module

        _FACE_SEARCH_MODULE = face_search_module
    return _FACE_SEARCH_MODULE


@dataclass(frozen=True)
class ClusterTagContextPayload:
    image_tags_by_path: dict[str, tuple[str, ...]]
    cluster_tag_summaries: dict[str, dict[int, ClusterTagSummary]]
    gallery_extra_context_by_path: dict[str, dict[str, object]]


@dataclass(frozen=True)
class PreparedRunPayload:
    request: ProductionClusterRequest
    run_origin: str
    discovered_count: int
    matched_count: int


@dataclass(frozen=True)
class FacesWorkspacePayload:
    execution_policy: ExecutionPolicy
    capabilities: object
    service_cache: dict[tuple[str, str, str, str], object]
    services_global: dict[str, object]
    services_session: dict[str, object]


@dataclass(frozen=True)
class NamesWorkspacePayload:
    service: object
    service_key: tuple[str, str, str, str]


@dataclass(frozen=True)
class PhotoSetRoute:
    """A transient Gallery set opened by another workspace.

    Routes deliberately live only in memory.  Persisting result paths would
    make a restored session point at files the user may have moved or removed.
    """

    source: str
    title: str
    paths: tuple[str, ...]
    return_workspace: str
    context_by_path: dict[str, dict[str, object]]


class ProductionClusterApp(QMainWindow):
    def __init__(self, runtime_layout: RuntimeLayout):
        super().__init__()
        self._is_shutting_down = False
        self._close_pending = False
        self._close_retry_scheduled = False
        self._packaged_launch_smoke = bool(str(os.environ.get(PACKAGED_LAUNCH_SMOKE_ENV) or "").strip())
        self._unsupported_resolution_active = False
        self._screen_available_width = MIN_SUPPORTED_SCREEN_WIDTH
        self._screen_available_height = MIN_SUPPORTED_SCREEN_HEIGHT
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._mark_shutting_down)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        app_icon = QApplication.windowIcon()
        if app_icon.isNull():
            app_icon = production_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)
        self.runtime_layout = runtime_layout
        self.settings = get_settings()
        self.settings_registry = get_production_settings_registry()
        self.settings_store = production_settings_store()
        self.recent_folder_history = RecentFolderHistory(self.settings_store)
        self.runtime_service = RuntimeCapabilityService()
        self.job_manager = JobManager(self)
        self.work_coordinator = WorkCoordinator(
            self.job_manager,
            self,
            gpu_policy=self._gpu_conflict_policy(),
        )
        self.work_coordinator.gpu_conflict_requested.connect(self._show_gpu_conflict_choice)
        self._gpu_conflict_menus: list[QMenu] = []
        self.system_resources = detect_system_resources()
        self.performance_profile = self._effective_performance_profile()
        self.execution_policy = ExecutionPolicy(
            preferred_mode=self._preferred_execution_mode(),
            reason="Runtime verification starts when accelerated work is requested.",
        )
        self.photo_metadata_service = PhotoMetadataService()
        self.image_tag_service = ImageTagService()
        self.saved_search_service = SavedSearchService()
        self.cache_maintenance_service = CacheMaintenanceService(self.settings)
        self.data_home_manager = DataHomeManager(PRODUCTION_DISPLAY_NAME, runtime_layout.root)
        self.model_asset_service = ModelAssetService(runtime_model_assets_dir=runtime_layout.model_assets_dir)
        self.session_controller = ClusteringSessionController(runtime_layout, self)
        self.session_controller.set_keep_worker_warm(self._keep_worker_warm())
        self.model_download_controller = ModelDownloadController(runtime_layout, self)
        self.cluster_data: dict[str, dict[int, list[str]]] = {}
        self.membership_by_image: dict[str, dict[str, dict[str, object]]] = {}
        self.metrics_by_backend: dict[str, dict[str, object]] = {}
        self.cluster_explanations: dict[str, dict[int, ClusterExplanation]] = {}
        self.cluster_meanings: dict[str, dict[int, ClusterMeaning]] = {}
        self.cluster_tag_summaries: dict[str, dict[int, ClusterTagSummary]] = {}
        self.image_tags_by_path: dict[str, tuple[str, ...]] = {}
        self.gallery_extra_context_by_path: dict[str, dict[str, object]] = {}
        self.last_run_metrics: dict[str, object] = {}
        self.current_run_origin = "folder"
        self.current_tag_filter: tuple[str, ...] = ()
        self.current_tag_match = "Any"
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._startup_maintenance_job = None
        self._startup_maintenance_thread = None
        self._startup_maintenance_complete = False
        self._preflight_job = None
        self._preflight_thread = None
        self._preflight_generation = 0
        self._preflight_job_id: int | None = None
        self._tag_context_job = None
        self._tag_context_thread = None
        self._tag_context_job_id: int | None = None
        self._tag_context_generation = 0
        self._storage_usage_job = None
        self._storage_usage_thread = None
        self._storage_usage_generation = 0
        self._storage_clear_job = None
        self._storage_clear_thread = None
        self._post_install_model_job = None
        self._post_install_model_thread = None
        self._model_download_job_id: int | None = None
        self._pending_model_download_run: tuple[ProductionClusterRequest, str] | None = None
        self._active_post_install_model_name: str | None = None
        self._faces_init_job = None
        self._faces_init_thread = None
        self._faces_init_job_id: int | None = None
        self._faces_init_scheduled = False
        self._faces_init_generation = 0
        self._faces_focus_search_pending = False
        self._pending_photo_face_tool_requests: list[tuple[object, object]] = []
        self._pending_post_install_model_name: str | None = None
        self._last_storage_summary: RuntimeStorageSummary | None = None
        self._active_job_id: int | None = None
        self._clustering_mode = "basic"
        self._faces_mode = "basic"
        self._active_workspace = "clustering"
        self._gallery_paths: list[str] = []
        self._gallery_snapshot_key = ""
        self._gallery_discovery_generation = 0
        self._gallery_discovery_job = None
        self._gallery_discovery_thread = None
        self._gallery_primary_comparison_key = ""
        self._photo_set_route: PhotoSetRoute | None = None
        self._pending_face_review_paths: tuple[str, ...] = ()
        self._active_cluster_directory = ""
        self._active_cluster_scope_signature = ""
        self._scope_editor_pinned = False
        self._pane_visibility = {"source": True, "controls": True, "details": True}
        self._advanced_pane_visibility = {"source": True, "controls": True, "details": True}
        self._pane_restore_widths = {"source": 250, "controls": 420, "details": 440}
        self._advanced_cluster_splitter_sizes = [1]
        self._face_service_cache: dict[tuple[str, str, str, str], FaceIndexService] = {}
        self.face_services_global = {}
        self.face_services_session = {}
        self.face_service_global = None
        self.face_service_session = None
        self.faces_pane = None
        self.faces_placeholder = None
        self.names_pane = None
        self.names_placeholder = None
        self.tags_pane = None
        self.library_pane = None
        self._library_catalog_refresh_pending = False
        self._tag_suggestion_job = None
        self._tag_suggestion_thread = None
        self._tag_suggestion_job_id: int | None = None
        self._tag_suggestion_generation = 0
        self._faces_session_reset_pending = True
        self._names_init_job = None
        self._names_init_thread = None
        self._names_init_job_id: int | None = None
        self._names_init_scheduled = False
        self._startup_readiness_report: StartupReadinessReport | None = None
        self._startup_readiness_job = None
        self._startup_readiness_thread = None
        self._startup_readiness_job_id: int | None = None
        self._startup_readiness_generation = 0
        self._startup_readiness_error = ""
        self._main_gallery_context_overrides: dict[str, dict[str, object]] = {}
        self._startup_check_timer = QTimer(self)
        self._startup_check_timer.setSingleShot(True)
        self._startup_check_timer.timeout.connect(self._post_startup_checks)
        self._post_install_timer = QTimer(self)
        self._post_install_timer.setSingleShot(True)
        self._post_install_timer.timeout.connect(self._run_pending_post_install_model_download)
        self._build_ui()
        self._connect_signals()
        self._refresh_recent_folder_menus()
        restored_roots = self._restore_active_roots()
        self.source_pane.set_active_roots(restored_roots)
        restored_folder = str(self.settings_store.value("workspace/selected_folder", "") or "").strip()
        browse_root = restored_folder or self.source_pane.active_scope.primary_root
        if browse_root and Path(browse_root).is_dir():
            self.source_pane.set_selected_directory(browse_root, activate_scope=False)
        else:
            self._update_scope_summary(self.source_pane.active_scope)
        self._apply_workspace_preferences()
        self.runtime_badge.update_runtime(None, None)
        self._apply_safety_state()
        self._apply_startup_workflow_gate()
        self.set_faces_mode(self._preferred_faces_ui_mode())
        self.set_active_workspace(self._preferred_workspace())
        if not self._packaged_launch_smoke:
            QTimer.singleShot(0, self._start_startup_maintenance)
            self._startup_check_timer.start(0)

    def _cleanup_runtime_temp_on_startup(self) -> None:
        cleared, failures = self.cache_maintenance_service.clear_runtime_temp_files()
        if cleared:
            LOGGER.info("Cleared startup runtime temp targets: %s", ", ".join(cleared))
        if failures:
            LOGGER.warning("Startup runtime temp cleanup failed: %s", "; ".join(failures))

    def _recover_downloaded_model_storage(self) -> None:
        """Reconcile durable model state before any selector reads it.

        This path is local-only: it may promote an interrupted face install or
        repair a Hugging Face cache reference, but it never starts a download
        during application startup.
        """
        try:
            face_result = FaceModelInstaller(self.settings).recover_managed_models()
            if face_result.promoted_interrupted or face_result.restored:
                LOGGER.info(
                    "Recovered managed face models | promoted=%s restored=%s",
                    list(face_result.promoted_interrupted),
                    list(face_result.restored),
                )
            if face_result.unavailable:
                LOGGER.warning(
                    "Previously installed face models are missing and have no local recovery archive: %s",
                    ", ".join(face_result.unavailable),
                )
            if face_result.failures:
                LOGGER.warning("Face-model recovery completed with failures: %s", "; ".join(face_result.failures))
        except Exception:
            LOGGER.exception("Face-model startup recovery failed")
        try:
            clustering_result = ModelDownloadService(self.settings).recover_cached_models()
            if clustering_result.repaired_revisions:
                LOGGER.info(
                    "Recovered clustering model cache references: %s",
                    ", ".join(clustering_result.repaired_revisions),
                )
        except Exception:
            LOGGER.exception("Clustering-model startup recovery failed")

    def _run_runtime_migrations(self) -> None:
        try:
            result = RuntimeMigrationService(self.runtime_layout).run()
        except Exception:
            LOGGER.exception("Runtime migration failed")
            return
        if result.actions:
            LOGGER.info(
                "Runtime migrations applied | previous=%s current=%s actions=%s",
                result.previous_version,
                result.current_version,
                ", ".join(result.actions),
            )
        if result.failures:
            LOGGER.warning("Runtime migration completed with failures: %s", "; ".join(result.failures))

    def _start_startup_maintenance(self) -> None:
        """Recover local runtime state after the window can show its progress."""
        if self._is_shutting_down or self._close_pending or self._startup_maintenance_complete:
            return
        if self._thread_is_running(self._startup_maintenance_thread):
            return

        def _run(progress, cancel_check):
            progress(-1, "Migrating local workspace…")
            raise_if_cancelled(cancel_check)
            self._run_runtime_migrations()
            progress(-1, "Cleaning local temporary files…")
            raise_if_cancelled(cancel_check)
            self._cleanup_runtime_temp_on_startup()
            progress(-1, "Recovering downloaded model storage…")
            raise_if_cancelled(cancel_check)
            self._recover_downloaded_model_storage()
            progress(100, "Local workspace ready")
            return None

        job = AsyncJob(_run)
        self._startup_maintenance_job = job
        job_id = self.job_manager.register_job(
            "Preparing local workspace",
            cancel_fn=job.cancel,
            origin="Startup",
        )
        job.progress.connect(
            lambda value, text, job_id=job_id: self.job_manager.update(job_id, progress=value, text=text)
        )

        def _cleanup() -> None:
            if self._startup_maintenance_job is job:
                self._startup_maintenance_job = None
            if self._startup_maintenance_thread is thread:
                self._startup_maintenance_thread = None
            self._release_async_refs(job, thread)

        def _completed(_result) -> None:
            if (
                self._is_shutting_down
                or self._close_pending
                or not self._can_update_widget(getattr(self, "clustering_pane", None))
            ):
                self.job_manager.finish(job_id, status="cancelled")
                _cleanup()
                return
            self._startup_maintenance_complete = True
            self.job_manager.finish(job_id, status="finished")
            _cleanup()
            self._refresh_footer_storage_usage()
            self._begin_startup_readiness_check()
            if self._active_workspace == "names" and self.names_pane is None:
                self._start_names_workspace_load()
            elif self._active_workspace == "faces" and self.faces_pane is None:
                self._start_faces_workspace_load()

        def _failed(message: str) -> None:
            # Individual recovery helpers log their own recoverable failures;
            # an unexpected one must still leave the main app usable.
            if (
                self._is_shutting_down
                or self._close_pending
                or not self._can_update_widget(getattr(self, "clustering_pane", None))
            ):
                self.job_manager.finish(job_id, status="cancelled")
                _cleanup()
                return
            self._startup_maintenance_complete = True
            self.job_manager.finish(job_id, status="failed", error=str(message))
            _cleanup()
            self._refresh_footer_storage_usage()
            self._begin_startup_readiness_check()
            if self._can_update_widget(getattr(self, "footer_bar", None)):
                self.footer_bar.set_status(f"Startup maintenance needs attention: {message}")
            self._finish_pending_photo_face_tool_requests(
                error=f"Local workspace preparation failed: {message}"
            )

        def _cancelled() -> None:
            self.job_manager.finish(job_id, status="cancelled")
            _cleanup()
            if self._is_shutting_down or self._close_pending:
                return
            self._finish_pending_photo_face_tool_requests(
                error="Local workspace preparation was cancelled."
            )

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._startup_maintenance_thread = thread
        self._retain_async_refs(job, thread)

    def _build_ui(self) -> None:
        self.setWindowTitle(PRODUCTION_DISPLAY_NAME)
        self.resize(1920, 1080)
        self.setMinimumSize(1440, 900)
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._build_toolbar())

        self.scope_strip = QWidget(central)
        scope_layout = QHBoxLayout(self.scope_strip)
        scope_layout.setContentsMargins(8, 4, 8, 4)
        scope_layout.setSpacing(8)
        self.scope_summary_label = QLabel("No active roots", self.scope_strip)
        self.scope_summary_label.setObjectName("workspaceScopeSummary")
        self.scope_summary_label.setToolTip("Shared root scope for Gallery, Clustering, Faces, Names, Tags, and Library. Active roots include all descendants.")
        self.scope_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.edit_roots_button = QPushButton("Edit roots", self.scope_strip)
        self.edit_roots_button.setToolTip("Show the shared root picker. Selecting roots does not scan, catalog, or modify photos.")
        self.edit_roots_button.clicked.connect(self._edit_active_roots)
        scope_layout.addWidget(QLabel("Scope:", self.scope_strip))
        scope_layout.addWidget(self.scope_summary_label, stretch=1)
        scope_layout.addWidget(self.edit_roots_button)
        layout.addWidget(self.scope_strip)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        layout.addWidget(self.main_splitter, stretch=1)

        self.source_pane = SourcePane(self)
        self.source_pane.setMinimumWidth(220)
        self.source_pane.setMaximumWidth(280)
        self.source_pane.set_data_home_summary(
            str(self.runtime_layout.root),
            detail="Indexes, previews, models, recovery, backups, and reports",
        )

        self.workspace_stack = QStackedWidget(self)
        self.clustering_workspace = QWidget(self.workspace_stack)
        clustering_layout = QHBoxLayout(self.clustering_workspace)
        clustering_layout.setContentsMargins(0, 0, 0, 0)
        clustering_layout.setSpacing(0)

        self.clustering_splitter = QSplitter(Qt.Orientation.Horizontal, self.clustering_workspace)
        self.clustering_splitter.setChildrenCollapsible(False)
        self.clustering_splitter.setHandleWidth(6)
        clustering_layout.addWidget(self.clustering_splitter)

        self.clustering_pane = ClusteringOptionsPane(self, option_scope="production")
        self.gallery_pane = GalleryPane(self)
        self.cluster_pane = ClusterPane(self)

        self.gallery_pane.job_manager = self.job_manager
        self.gallery_pane.metadata_service = self.photo_metadata_service
        self.gallery_pane.image_tag_service = self.image_tag_service
        self.gallery_pane.set_inspector_display_mode("basic")
        self.gallery_pane.set_read_only_mode(self._read_only_mode())
        self.gallery_pane.set_action_target_provider(self.cluster_pane.current_selection_target)
        self.gallery_pane.inspector_context_provider = self._main_gallery_context_for_path
        self.gallery_pane.face_edit_service_provider = lambda: self.face_service_global
        self.gallery_pane.face_edit_request_handler = self._request_photo_face_tools
        self.gallery_pane.face_edit_saved_callback = self._on_main_gallery_face_labels_changed
        self.gallery_pane.set_empty_state(
            "Start clustering your photos",
            "Choose a folder, then run clustering to organize its photos into visual groups.",
            show_select_folder=True,
            show_run=True,
        )
        self.gallery_pane.empty_run_button.setEnabled(False)
        self.clustering_pane.setMinimumWidth(380)
        self.clustering_pane.setMaximumWidth(460)
        self.gallery_pane.setMinimumWidth(760)
        self.cluster_pane.setMinimumWidth(440)
        self.cluster_pane.setMaximumWidth(680)

        self.cluster_right_splitter = QSplitter(Qt.Orientation.Vertical, self.clustering_workspace)
        self.cluster_right_splitter.setChildrenCollapsible(False)
        self.cluster_right_splitter.setHandleWidth(4)
        self.cluster_right_splitter.addWidget(self.cluster_pane)
        self.cluster_right_splitter.setStretchFactor(0, 1)

        self.clustering_splitter.addWidget(self.clustering_pane)
        self.clustering_splitter.addWidget(self.gallery_pane)
        self.clustering_splitter.addWidget(self.cluster_right_splitter)
        self.clustering_splitter.setStretchFactor(0, 3)
        self.clustering_splitter.setStretchFactor(1, 8)
        self.clustering_splitter.setStretchFactor(2, 3)

        self.photo_gallery_workspace = QWidget(self.workspace_stack)
        photo_gallery_layout = QHBoxLayout(self.photo_gallery_workspace)
        photo_gallery_layout.setContentsMargins(0, 0, 0, 0)
        photo_gallery_layout.setSpacing(6)
        self.photo_gallery = SectionedGallery(self.photo_gallery_workspace)
        self.photo_gallery.set_job_manager(self.job_manager)
        self.photo_gallery._actions.metadata_service = self.photo_metadata_service
        self.photo_gallery._actions.image_tag_service = self.image_tag_service
        self.photo_gallery.face_service_provider = lambda: self.face_service_global
        self.photo_gallery.inspector_context_provider = self._photo_gallery_context_for_path
        self.photo_gallery.face_edit_request_handler = self._request_photo_face_tools
        self.photo_gallery.face_edit_saved_callback = self._on_main_gallery_face_labels_changed
        self.photo_gallery.set_read_only_mode(self._read_only_mode())
        photo_gallery_layout.addWidget(self.photo_gallery, stretch=1)

        self.faces_placeholder = self._build_faces_placeholder()
        self.names_placeholder = self._build_names_placeholder()
        self.tags_pane = TagsPane(
            self.image_tag_service,
            lambda: self.source_pane.active_scope,
            self.workspace_stack,
            job_manager=self.job_manager,
        )
        self.tags_pane.gallery.metadata_service = self.photo_metadata_service
        self.tags_pane.set_read_only_mode(self._read_only_mode())
        self.library_pane = LibraryPane(
            lambda: self.source_pane.active_scope,
            lambda: self.face_service_global,
            self.workspace_stack,
            job_manager=self.job_manager,
            context_settings_provider=self._library_context_settings,
            context_settings_changed=self._save_library_context_settings,
        )
        self.library_pane.context_auto.setChecked(self._library_auto_context_enabled())
        self.library_pane.configure_photo_tools(
            metadata_service=self.photo_metadata_service,
            image_tag_service=self.image_tag_service,
            face_service_provider=lambda: self.face_service_global,
            face_edit_request_handler=self._request_photo_face_tools,
            face_edit_saved_callback=self._on_main_gallery_face_labels_changed,
            inspector_context_provider=self._gallery_context_for_path,
        )
        self.library_pane.set_read_only_mode(self._read_only_mode())

        self.workspace_stack.addWidget(self.photo_gallery_workspace)
        self.workspace_stack.addWidget(self.clustering_workspace)
        self.workspace_stack.addWidget(self.faces_placeholder)
        self.workspace_stack.addWidget(self.names_placeholder)
        self.workspace_stack.addWidget(self.tags_pane)
        self.workspace_stack.addWidget(self.library_pane)
        self.workspace_stack.setMinimumWidth(1040)

        self.main_splitter.addWidget(self.source_pane)
        self.main_splitter.addWidget(self.workspace_stack)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 11)

        self.footer_bar = WorkspaceFooter(self)
        self.job_presentation = JobPresentationController(self.job_manager, self.footer_bar, self)
        layout.addWidget(self.footer_bar)
        self.statusBar().hide()
        self.set_clustering_mode("basic")

    def _build_faces_placeholder(self) -> QWidget:
        panel = QWidget(self.workspace_stack)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addStretch(1)
        self.faces_placeholder_title = QLabel("Faces workspace")
        self.faces_placeholder_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.faces_placeholder_title.setObjectName("facesLazyTitle")
        self.faces_placeholder_detail = QLabel("Face indexing, search, review, and identity tools load when opened.")
        self.faces_placeholder_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.faces_placeholder_detail.setWordWrap(True)
        self.faces_placeholder_button = QPushButton("Load Faces")
        self.faces_placeholder_button.setMinimumWidth(180)
        self.faces_placeholder_button.clicked.connect(self._start_faces_workspace_load)
        layout.addWidget(self.faces_placeholder_title)
        layout.addWidget(self.faces_placeholder_detail)
        layout.addWidget(self.faces_placeholder_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        return panel

    def _build_names_placeholder(self) -> QWidget:
        panel = QWidget(self.workspace_stack)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addStretch(1)
        self.names_placeholder_title = QLabel("Names workspace")
        self.names_placeholder_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.names_placeholder_title.setObjectName("namesLazyTitle")
        self.names_placeholder_detail = QLabel("Opening the saved global face-label database…")
        self.names_placeholder_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.names_placeholder_detail.setWordWrap(True)
        self.names_placeholder_button = QPushButton("Open Names")
        self.names_placeholder_button.setMinimumWidth(180)
        self.names_placeholder_button.clicked.connect(self._start_names_workspace_load)
        layout.addWidget(self.names_placeholder_title)
        layout.addWidget(self.names_placeholder_detail)
        layout.addWidget(self.names_placeholder_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)
        return panel

    def show_unsupported_resolution_view(self, screen_geometry: tuple[int, int]) -> None:
        # The placeholder is about to be deleted.  Invalidate and cancel its
        # worker first so a queued completion cannot touch replaced widgets.
        self._unsupported_resolution_active = True
        self._cancel_faces_workspace_load()
        width, height = screen_geometry
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(12)
        layout.addStretch(1)
        title = QLabel("Unsupported display resolution")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("unsupportedResolutionTitle")
        detail = QLabel(
            f"ClusterLens requires a logical display of at least {MIN_SUPPORTED_SCREEN_WIDTH}x{MIN_SUPPORTED_SCREEN_HEIGHT} "
            "in either landscape or portrait orientation. "
            f"This display reports {int(width)}x{int(height)}."
        )
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detail.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(detail)
        layout.addStretch(1)
        self.setCentralWidget(panel)
        self.setMinimumSize(640, 480)

    def _ensure_faces_workspace(self):
        if self._unsupported_resolution_active:
            return None
        if self.faces_pane is not None:
            return self.faces_pane

        # Keep the first window construction lightweight even when Faces is
        # the saved workspace.  The placeholder paints first; model/runtime
        # discovery begins on the next event-loop turn and remains cancellable.
        if not self._faces_init_scheduled and not self._thread_is_running(self._faces_init_thread):
            self._faces_init_scheduled = True
            generation = self._faces_init_generation
            QTimer.singleShot(0, lambda: self._start_faces_workspace_load(generation))
        return self.faces_placeholder

    def _cancel_faces_workspace_load(self) -> None:
        """Invalidate a lazy Faces load and notify pending Photos requests."""

        self._faces_init_generation += 1
        self._faces_init_scheduled = False
        job = self._faces_init_job
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        self._finish_pending_photo_face_tool_requests(error="Face tool preparation was cancelled.")

    def _request_photo_face_tools(self, image_path: str, on_ready, on_failed) -> None:
        """Prepare the global face service for a direct Photos edit request."""

        if self._is_shutting_down or self._unsupported_resolution_active:
            on_failed("The application is closing or the display is unsupported.")
            return
        if self._read_only_mode():
            on_failed("Face editing is disabled by read-only safety mode.")
            return
        report = self._startup_readiness_report
        if report is not None and not report.face_ready:
            on_failed(report.face_message)
            return
        if self._startup_readiness_error:
            on_failed(self._startup_readiness_error)
            return
        if self.face_service_global is not None:
            on_ready()
            return
        self._pending_photo_face_tool_requests.append((on_ready, on_failed))
        self.footer_bar.set_status(f"Preparing face tools for {Path(str(image_path)).name}…")
        self._set_activity("Preparing face tools…")
        if report is None and self._thread_is_running(self._startup_readiness_thread):
            # The local-only readiness job is already visible in Jobs. Wait
            # for it instead of racing a model/service construction.
            return
        if not self._startup_maintenance_complete:
            self._start_startup_maintenance()
            return
        # Names opens the same durable global face service. Do not construct
        # it twice when that workspace is already loading.
        if self._thread_is_running(self._names_init_thread):
            return
        if not self._thread_is_running(self._faces_init_thread):
            self._start_faces_workspace_load()

    def _finish_pending_photo_face_tool_requests(self, *, error: str = "") -> None:
        pending = list(self._pending_photo_face_tool_requests)
        self._pending_photo_face_tool_requests.clear()
        for on_ready, on_failed in pending:
            try:
                if error:
                    on_failed(error)
                else:
                    on_ready()
            except Exception:
                LOGGER.exception("Photo face-tools callback failed")

    def _ensure_names_workspace(self):
        if self.names_pane is not None:
            return self.names_pane
        if not self._names_init_scheduled and not self._thread_is_running(self._names_init_thread):
            self._names_init_scheduled = True
            QTimer.singleShot(0, self._start_names_workspace_load)
        return self.names_placeholder

    def _start_names_workspace_load(self) -> None:
        self._names_init_scheduled = False
        if self._is_shutting_down or self._close_pending:
            return
        if not self._startup_maintenance_complete:
            if self._can_update_widget(getattr(self, "names_placeholder_detail", None)):
                self.names_placeholder_detail.setText("Preparing local workspace before opening saved names…")
            self._start_startup_maintenance()
            return
        if self.names_pane is not None and self.face_service_global is not None:
            return
        if self._thread_is_running(self._names_init_thread):
            return

        model_root = str(self.settings_registry.get(self.settings_store, "faces/model_root", "") or "").strip()
        configured_detector = str(
            self.settings_registry.get(self.settings_store, "faces/default_detector/human") or ""
        ).strip()
        configured_embedder = str(
            self.settings_registry.get(self.settings_store, "faces/default_embedder/human") or ""
        ).strip()
        detector_score = float(
            self.settings_registry.get(self.settings_store, "faces/detector_score_threshold/human")
        )
        max_detections = max(
            1,
            int(self.settings_registry.get(self.settings_store, "faces/max_detections/human", 50)),
        )
        cache_dir = Path(self.settings.cache_dir)
        performance_profile = self.performance_profile
        if self._can_update_widget(getattr(self, "names_placeholder_detail", None)):
            self.names_placeholder_detail.setText("Opening the saved global face-label database…")
        if self._can_update_widget(getattr(self, "names_placeholder_button", None)):
            self.names_placeholder_button.setEnabled(False)

        def _run(progress, cancel_check):
            progress(-1, "Opening saved face labels…")
            face_search = _face_search_api()
            mode = "human"
            detector, embedder = face_search.resolve_ready_face_pipeline_ids(
                model_root,
                mode,
                configured_detector or face_search.DEFAULT_HUMAN_FACE_DETECTOR_ID,
                configured_embedder or face_search.DEFAULT_HUMAN_FACE_EMBEDDER_ID,
            )
            raise_if_cancelled(cancel_check)
            if detector == face_search.BUILTIN_HUMAN_DETECTOR_ID and embedder == face_search.BUILTIN_HUMAN_EMBEDDER_ID:
                db_path = cache_dir / "face_search_index.db"
            else:
                db_path = cache_dir / f"face_search_human_{detector}__{embedder}.db"
            service = face_search.FaceIndexService(
                mode=mode,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
                model_root=model_root,
                detector_id=detector,
                embedder_id=embedder,
                detector_score_threshold=detector_score,
                detector_max_detections=max_detections,
                performance_profile=performance_profile,
                db_path=db_path,
            )
            raise_if_cancelled(cancel_check)
            return NamesWorkspacePayload(
                service=service,
                service_key=("global", mode, detector, embedder),
            )

        job = AsyncJob(_run)
        self._names_init_job = job
        self._names_init_job_id = self.job_manager.register_job(
            "Opening Names", cancel_fn=job.cancel, origin="Names"
        )
        job.progress.connect(
            lambda value, text: self.job_manager.update(
                self._names_init_job_id or -1,
                progress=value,
                text=text,
            )
        )
        job.progress.connect(
            lambda _value, text: self.names_placeholder_detail.setText(str(text))
            if self._can_update_widget(getattr(self, "names_placeholder_detail", None))
            else None
        )

        def _finish(status: str, error: str = "") -> None:
            if self._names_init_job_id is not None:
                self.job_manager.finish(self._names_init_job_id, status=status, error=error)
            self._names_init_job_id = None

        def _completed(payload) -> None:
            if self._is_shutting_down or not isinstance(payload, NamesWorkspacePayload):
                _finish("cancelled")
                return
            self._face_service_cache[payload.service_key] = payload.service
            self.face_services_global = {"human": payload.service}
            self.face_service_global = payload.service
            if self.names_pane is None:
                self._build_names_workspace_widget()
            else:
                self.names_pane.refresh_names()
            _finish("finished")
            if self._pending_photo_face_tool_requests:
                report = self._startup_readiness_report
                if report is not None:
                    self._finish_pending_photo_face_tool_requests(
                        error="" if report.face_ready else report.face_message
                    )
                elif self._startup_readiness_error:
                    self._finish_pending_photo_face_tool_requests(error=self._startup_readiness_error)
                elif not self._thread_is_running(self._startup_readiness_thread):
                    self._begin_startup_readiness_check()

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            _finish("cancelled" if text.casefold() == "cancelled" else "failed", text)
            if self._pending_photo_face_tool_requests and not self._thread_is_running(self._faces_init_thread):
                self._finish_pending_photo_face_tool_requests(
                    error=text or "Face tools could not be prepared."
                )
            if self._can_update_widget(getattr(self, "names_placeholder_detail", None)):
                self.names_placeholder_detail.setText(f"Names could not open: {text}")
                self.names_placeholder_button.setText("Retry Names")
                self.names_placeholder_button.setEnabled(True)

        def _cancelled() -> None:
            _finish("cancelled")
            if self._pending_photo_face_tool_requests and not self._thread_is_running(self._faces_init_thread):
                self._finish_pending_photo_face_tool_requests(error="Face tools loading was cancelled.")
            if self._can_update_widget(getattr(self, "names_placeholder_detail", None)):
                self.names_placeholder_detail.setText("Names loading was cancelled.")
                self.names_placeholder_button.setEnabled(True)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._names_init_thread = thread
        self._retain_async_refs(job, thread)

        def _cleanup() -> None:
            self._release_async_refs(job, thread)
            if self._names_init_thread is thread:
                self._names_init_thread = None
            if self._names_init_job is job:
                self._names_init_job = None

        thread.finished.connect(_cleanup, Qt.ConnectionType.QueuedConnection)

    def _start_faces_workspace_load(self, expected_generation: int | None = None) -> None:
        self._faces_init_scheduled = False
        if expected_generation is not None and expected_generation != self._faces_init_generation:
            return
        if (
            self.faces_pane is not None
            or self._is_shutting_down
            or self._unsupported_resolution_active
            or (self._active_workspace != "faces" and not self._pending_photo_face_tool_requests)
        ):
            return
        if not self._startup_maintenance_complete:
            if self._can_update_widget(getattr(self, "faces_placeholder_detail", None)):
                self.faces_placeholder_detail.setText("Preparing local workspace before opening Faces…")
            self._start_startup_maintenance()
            return
        if self._thread_is_running(self._faces_init_thread):
            return

        load_generation = self._faces_init_generation

        model_root = str(self.settings_registry.get(self.settings_store, "faces/model_root", "") or "").strip()
        configured_detector = str(
            self.settings_registry.get(self.settings_store, "faces/default_detector/human") or ""
        ).strip()
        configured_embedder = str(
            self.settings_registry.get(self.settings_store, "faces/default_embedder/human") or ""
        ).strip()
        detector_score = float(
            self.settings_registry.get(self.settings_store, "faces/detector_score_threshold/human")
        )
        max_detections = max(
            1,
            int(self.settings_registry.get(self.settings_store, "faces/max_detections/human", 50)),
        )
        preferred_execution_mode = self._preferred_execution_mode()
        reset_face_session = bool(self._faces_session_reset_pending)
        cache_dir = Path(self.settings.cache_dir)
        performance_profile = self.performance_profile
        if getattr(self, "faces_placeholder_detail", None) is not None:
            self.faces_placeholder_detail.setText("Checking runtime and opening the saved face library…")
        if getattr(self, "faces_placeholder_button", None) is not None:
            self.faces_placeholder_button.setEnabled(False)

        def _run(progress, cancel_check):
            progress(-1, "Checking CPU and CUDA runtime…")
            policy = self.runtime_service.select_policy(preferred_execution_mode, refresh=True)
            capabilities = self.runtime_service.detect()
            if policy.cuda_required_unavailable:
                raise RuntimeError(policy.error or policy.reason)
            raise_if_cancelled(cancel_check)
            progress(-1, "Opening face databases…")
            face_search = _face_search_api()
            mode = "human"
            detector, embedder = face_search.resolve_ready_face_pipeline_ids(
                model_root,
                mode,
                configured_detector or face_search.DEFAULT_HUMAN_FACE_DETECTOR_ID,
                configured_embedder or face_search.DEFAULT_HUMAN_FACE_EMBEDDER_ID,
            )
            if reset_face_session:
                for path in cache_dir.glob("face_search*_session*.db*"):
                    raise_if_cancelled(cancel_check)
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        continue

            def _db_path(scope: str) -> Path:
                if detector == face_search.BUILTIN_HUMAN_DETECTOR_ID and embedder == face_search.BUILTIN_HUMAN_EMBEDDER_ID:
                    return cache_dir / ("face_search_session.db" if scope == "session" else "face_search_index.db")
                suffix = "_session" if scope == "session" else ""
                slug = f"{detector}__{embedder}"
                return cache_dir / f"face_search_human_{slug}{suffix}.db"

            service_cache: dict[tuple[str, str, str, str], object] = {}
            services: dict[str, object] = {}
            for scope in ("global", "session"):
                raise_if_cancelled(cancel_check)
                service = face_search.FaceIndexService(
                    mode=mode,
                    execution_policy=policy,
                    runtime_service=self.runtime_service,
                    model_root=model_root,
                    detector_id=detector,
                    embedder_id=embedder,
                    detector_score_threshold=detector_score,
                    detector_max_detections=max_detections,
                    performance_profile=performance_profile,
                    db_path=_db_path(scope),
                )
                service_cache[(scope, mode, detector, embedder)] = service
                services[scope] = service
            return FacesWorkspacePayload(
                execution_policy=policy,
                capabilities=capabilities,
                service_cache=service_cache,
                services_global={mode: services["global"]},
                services_session={mode: services["session"]},
            )

        job = AsyncJob(_run)
        self._faces_init_job = job
        self._faces_init_job_id = self.job_manager.register_job(
            "Opening Faces", cancel_fn=job.cancel, origin="Faces"
        )
        job.progress.connect(
            lambda value, text: self.job_manager.update(
                self._faces_init_job_id or -1,
                progress=value,
                text=text,
            )
        )
        job.progress.connect(
            lambda _value, text: self.faces_placeholder_detail.setText(str(text))
            if self._can_update_widget(getattr(self, "faces_placeholder_detail", None))
            else None
        )

        def _finish(status: str, error: str = "") -> None:
            if self._faces_init_job_id is not None:
                self.job_manager.finish(self._faces_init_job_id, status=status, error=error)
            self._faces_init_job_id = None

        def _completed(payload) -> None:
            if (
                self._is_shutting_down
                or self._unsupported_resolution_active
                or load_generation != self._faces_init_generation
                or (self._active_workspace != "faces" and not self._pending_photo_face_tool_requests)
                or not isinstance(payload, FacesWorkspacePayload)
            ):
                _finish("cancelled")
                self._finish_pending_photo_face_tool_requests(error="Face tools were cancelled before they were ready.")
                return
            self.execution_policy = payload.execution_policy
            self.runtime_badge.update_runtime(payload.capabilities, payload.execution_policy)
            self._face_service_cache = dict(payload.service_cache)
            self.face_services_global = dict(payload.services_global)
            self.face_services_session = dict(payload.services_session)
            self.face_service_global = self.face_services_global["human"]
            self.face_service_session = self.face_services_session["human"]
            self._faces_session_reset_pending = False
            if self._active_workspace == "faces":
                self._build_faces_workspace_widget()
            _finish("finished")
            self.footer_bar.set_status("Face tools are ready.")
            self._set_activity("Ready")
            self._finish_pending_photo_face_tool_requests()

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            _finish("cancelled" if text.casefold() == "cancelled" else "failed", text)
            self._finish_pending_photo_face_tool_requests(error=text or "Face tools could not be prepared.")
            if self._can_update_widget(getattr(self, "faces_placeholder_detail", None)):
                self.faces_placeholder_detail.setText(f"Faces could not open: {text}")
                self.faces_placeholder_button.setText("Retry Faces")
                self.faces_placeholder_button.setEnabled(True)

        def _cancelled() -> None:
            _finish("cancelled")
            self._finish_pending_photo_face_tool_requests(error="Face tools loading was cancelled.")
            if self._can_update_widget(getattr(self, "faces_placeholder_detail", None)):
                self.faces_placeholder_detail.setText("Faces loading was cancelled.")
                self.faces_placeholder_button.setEnabled(True)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._faces_init_thread = thread
        self._retain_async_refs(job, thread)

        def _cleanup() -> None:
            self._release_async_refs(job, thread)
            if self._faces_init_thread is thread:
                self._faces_init_thread = None
            if self._faces_init_job is job:
                self._faces_init_job = None
            if (
                self._active_workspace == "faces"
                and self.faces_pane is None
                and not self._is_shutting_down
                and not self._unsupported_resolution_active
            ):
                self._ensure_faces_workspace()

        thread.finished.connect(_cleanup, Qt.ConnectionType.QueuedConnection)

    def _build_faces_workspace_widget(self) -> None:
        if self.faces_pane is not None or self._is_shutting_down or self._unsupported_resolution_active:
            return

        from ui.search_pane import SearchPane
        pane = SearchPane(
            self.workspace_stack,
            face_service_global=self.face_service_global,
            face_service_session=self.face_service_session,
            face_services_global=self.face_services_global,
            face_services_session=self.face_services_session,
            enabled_tabs=["All Faces", "Folder Review", "Face Search", "Identities"],
            saved_search_service=self.saved_search_service,
            supported_face_modes=["human"],
        )
        pane.set_face_service_provider(self._face_service_for_pipeline)
        pane.current_directory_provider = lambda: self.source_pane.active_scope.primary_root
        pane.current_scope_roots_provider = lambda: self.source_pane.active_scope.roots
        pane.clustering_filter_state_provider = self._current_clustering_filter_state
        pane.use_onnx_provider = lambda: bool(self.clustering_pane.onnx_checkbox.isChecked())
        pane.job_manager = self.job_manager
        pane.results_gallery.job_manager = self.job_manager
        pane.results_gallery.metadata_service = self.photo_metadata_service
        pane.results_gallery.image_tag_service = self.image_tag_service
        pane.configure_face_pipeline_options(
            self._face_model_root(),
            self._preferred_face_pipeline_defaults(),
            refresh=False,
        )
        pane.set_active_face_mode("human", refresh=False)
        report = self._startup_readiness_report
        pane.set_startup_readiness(
            bool(report is not None and report.face_ready),
            report.face_message if report is not None else "Checking CUDA and downloaded face models…",
        )
        pane.set_read_only_mode(self._read_only_mode())
        if self.source_pane.active_scope.primary_root:
            pane.face_folder_path.setText(self.source_pane.active_scope.primary_root)
        pane.setMinimumWidth(self._layout_widths()["faces"])
        self.faces_pane = pane
        self._connect_faces_signals(pane)
        self._apply_workspace_preferences()
        self._apply_safety_state()

        placeholder = self.faces_placeholder
        placeholder_index = self.workspace_stack.indexOf(placeholder) if placeholder is not None else -1
        if placeholder_index >= 0:
            self.workspace_stack.insertWidget(placeholder_index, pane)
            self.workspace_stack.removeWidget(placeholder)
            placeholder.deleteLater()
            self.faces_placeholder = None
        else:
            self.workspace_stack.addWidget(pane)
        if self._active_workspace == "faces":
            self.workspace_stack.setCurrentWidget(pane)
            pane.ensure_current_faces_tab_loaded()
        if self._faces_focus_search_pending:
            self._faces_focus_search_pending = False
            target = getattr(pane, "face_identity_search", None) or getattr(pane, "face_query_path", None)
            if target is not None:
                target.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _build_names_workspace_widget(self) -> None:
        if self.names_pane is not None or self._is_shutting_down:
            return

        from ui.names_pane import NamesPane

        pane = NamesPane(
            lambda: self.face_service_global,
            self.workspace_stack,
            job_manager=self.job_manager,
            active_scope_provider=lambda: self.source_pane.active_scope,
        )
        pane.gallery.job_manager = self.job_manager
        pane.gallery.metadata_service = self.photo_metadata_service
        pane.gallery.image_tag_service = self.image_tag_service
        pane.set_read_only_mode(self._read_only_mode())
        pane.setMinimumWidth(self._layout_widths()["faces"])
        self.names_pane = pane
        pane.face_labels_changed.connect(self._on_names_face_labels_changed)
        pane.open_in_gallery_requested.connect(self._open_named_photos_in_gallery)
        pane.open_similar_faces_in_gallery_requested.connect(self._open_names_similar_faces_in_gallery)
        if self.faces_pane is not None:
            self.faces_pane.face_labels_changed.connect(pane.refresh_names)
        self._apply_workspace_preferences()

        placeholder = self.names_placeholder
        placeholder_index = self.workspace_stack.indexOf(placeholder) if placeholder is not None else -1
        if placeholder_index >= 0:
            self.workspace_stack.insertWidget(placeholder_index, pane)
            self.workspace_stack.removeWidget(placeholder)
            placeholder.deleteLater()
            self.names_placeholder = None
        else:
            self.workspace_stack.addWidget(pane)
        if self._active_workspace == "names":
            self.workspace_stack.setCurrentWidget(pane)
            pane.refresh_names()

    def _on_names_face_labels_changed(self) -> None:
        """Refresh already-created Faces views after an exact Names label edit."""

        if self.faces_pane is None:
            return
        self.faces_pane.refresh_face_library(reason="labels changed in Names")
        self.faces_pane.refresh_face_album(reason="labels changed in Names", force_refresh=True)

    def _on_main_gallery_face_labels_changed(self, image_path: str) -> None:
        """Refresh durable-name views after a Photos inspector face edit."""

        path = str(image_path or "")
        if path:
            self.photo_gallery.metadata_changed.emit([path])
            self.gallery_pane.metadata_changed.emit([path])
            self.footer_bar.set_status(f"Saved face edits for {Path(path).name}.")
        if self.names_pane is not None:
            self.names_pane.refresh_names()
        self._on_names_face_labels_changed()

    def _photo_gallery_context_for_path(self, path: str) -> dict[str, object]:
        """Combine normal photo context with the active route's face context."""

        context = self._gallery_context_for_path(str(path))
        route = self._photo_set_route
        if route is not None:
            route_context = route.context_by_path.get(str(path))
            if isinstance(route_context, dict):
                context.update(route_context)
            context["photo_set_route"] = {
                "source": route.source,
                "title": route.title,
            }
        return context

    def _connect_faces_signals(self, pane) -> None:
        pane.open_in_gallery_requested.connect(self._open_face_results_in_main_gallery)
        pane.append_to_gallery_requested.connect(self._append_face_results_to_main_gallery)
        pane.saved_clustering_filter_requested.connect(self._run_saved_clustering_filter)
        pane.open_face_model_settings_requested.connect(lambda: self.open_settings_dialog("Face Models"))
        pane.face_pipeline_controls_requested.connect(self._show_face_pipeline_controls)
        pane.face_pipeline_applied.connect(self._persist_face_pipeline_defaults)
        pane.install_face_model_requested.connect(self._request_face_model_download)
        pane.source_folder_changed.connect(self._set_shared_source_folder)
        if self.names_pane is not None:
            pane.face_labels_changed.connect(self.names_pane.refresh_names)
        pane.recent_folder_remove_requested.connect(self._remove_recent_folder)
        pane.recent_folders_clear_requested.connect(self._clear_recent_folders)
        pane.set_recent_directories(self.recent_folder_history.paths())
        self._apply_pending_photo_face_review(pane)

    def _show_face_pipeline_controls(self) -> None:
        self.set_active_workspace("faces")
        pane = self._ensure_faces_workspace()
        pane.open_face_pipeline_dialog()
        self.footer_bar.set_status("Face pipeline editor opened.")

    def _set_shared_source_folder(self, directory: str) -> None:
        directory = str(directory or "").strip()
        if directory and Path(directory).is_dir():
            self.source_pane.set_selected_directory(directory, activate_scope=False)
            self.source_pane.set_active_roots([directory])

    def _refresh_recent_folder_menus(self) -> None:
        paths = self.recent_folder_history.paths()
        self.source_pane.set_recent_directories(paths)
        if self.faces_pane is not None:
            self.faces_pane.set_recent_directories(paths)

    def _remove_recent_folder(self, directory: str) -> None:
        if self.recent_folder_history.remove(directory):
            self._refresh_recent_folder_menus()

    def _clear_recent_folders(self) -> None:
        if self.recent_folder_history.clear():
            self._refresh_recent_folder_menus()

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(self)
        bar.setObjectName("applicationHeader")
        bar.setAccessibleName("Application header")
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)

        self.feature_label = QLabel(PRODUCTION_DISPLAY_NAME)
        self.feature_label.setProperty("role", "section")
        self.feature_label.setAccessibleName("ClusterLens")
        self.gallery_workspace_button = QPushButton("Gallery")
        self.gallery_workspace_button.setCheckable(True)
        self.gallery_workspace_button.setProperty("nav", True)
        self.gallery_workspace_button.setAccessibleName("Open Gallery workspace")
        apply_icon(self.gallery_workspace_button, "workspace")
        self.gallery_workspace_button.clicked.connect(lambda: self.set_active_workspace("gallery"))
        self.clustering_workspace_button = QPushButton("Clustering")
        self.clustering_workspace_button.setCheckable(True)
        self.clustering_workspace_button.setProperty("nav", True)
        self.clustering_workspace_button.setAccessibleName("Open Clustering workspace")
        apply_icon(self.clustering_workspace_button, "workspace")
        self.clustering_workspace_button.clicked.connect(lambda: self.set_active_workspace("clustering"))
        self.faces_workspace_button = QPushButton("Faces")
        self.faces_workspace_button.setCheckable(True)
        self.faces_workspace_button.setProperty("nav", True)
        self.faces_workspace_button.setAccessibleName("Open Faces workspace")
        apply_icon(self.faces_workspace_button, "search")
        self.faces_workspace_button.clicked.connect(lambda: self.set_active_workspace("faces"))
        self.names_workspace_button = QPushButton("Names")
        self.names_workspace_button.setCheckable(True)
        self.names_workspace_button.setProperty("nav", True)
        self.names_workspace_button.setAccessibleName("Open Names workspace")
        self.names_workspace_button.setToolTip("Browse durable saved names and the photos containing their labeled faces.")
        apply_icon(self.names_workspace_button, "search")
        self.names_workspace_button.clicked.connect(lambda: self.set_active_workspace("names"))
        self.tags_workspace_button = QPushButton("Tags")
        self.tags_workspace_button.setCheckable(True)
        self.tags_workspace_button.setProperty("nav", True)
        self.tags_workspace_button.setAccessibleName("Open Tags workspace")
        self.tags_workspace_button.setToolTip("Browse, edit, filter, and cluster durable photo tags.")
        apply_icon(self.tags_workspace_button, "search")
        self.tags_workspace_button.clicked.connect(lambda: self.set_active_workspace("tags"))
        self.library_workspace_button = QPushButton("Library")
        self.library_workspace_button.setCheckable(True)
        self.library_workspace_button.setProperty("nav", True)
        self.library_workspace_button.setAccessibleName("Open Library workspace")
        self.library_workspace_button.setToolTip("Browse registered local photo roots, review duplicates, clean up people, and search generated cluster context.")
        apply_icon(self.library_workspace_button, "workspace")
        self.library_workspace_button.clicked.connect(lambda: self.set_active_workspace("library"))
        self.current_folder_label = QLabel("No folder selected")
        self.current_folder_label.setAccessibleName("Current folder")
        self.current_folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.activity_label = self.current_folder_label

        self.mode_selector = QComboBox(self)
        self.mode_selector.addItem("Basic", "basic")
        self.mode_selector.addItem("Advanced", "advanced")
        self.mode_selector.setAccessibleName("Workspace mode")
        self.mode_selector.currentIndexChanged.connect(self._on_mode_selector_changed)

        self.source_toggle = QPushButton("Folders")
        self.source_toggle.setCheckable(True)
        self.source_toggle.setChecked(True)
        self.source_toggle.setProperty("paneToggle", True)
        self.source_toggle.toggled.connect(lambda checked: self._set_pane_visible("source", checked))
        self.controls_toggle = QPushButton("Controls")
        self.controls_toggle.setCheckable(True)
        self.controls_toggle.setChecked(True)
        self.controls_toggle.setProperty("paneToggle", True)
        self.controls_toggle.toggled.connect(lambda checked: self._set_pane_visible("controls", checked))
        self.details_toggle = QPushButton("Compare")
        self.details_toggle.setCheckable(True)
        self.details_toggle.setChecked(True)
        self.details_toggle.setProperty("paneToggle", True)
        self.details_toggle.toggled.connect(lambda checked: self._set_pane_visible("details", checked))
        self.view_menu = QMenu("View", self)
        self.source_view_action = self.view_menu.addAction("Show folders")
        self.source_view_action.setCheckable(True)
        self.source_view_action.setChecked(True)
        self.source_view_action.toggled.connect(lambda checked: self._set_pane_visible("source", checked))
        self.controls_view_action = self.view_menu.addAction("Show clustering controls")
        self.controls_view_action.setCheckable(True)
        self.controls_view_action.setChecked(True)
        self.controls_view_action.toggled.connect(lambda checked: self._set_pane_visible("controls", checked))
        self.details_view_action = self.view_menu.addAction("Show comparison details")
        self.details_view_action.setCheckable(True)
        self.details_view_action.setChecked(True)
        self.details_view_action.toggled.connect(lambda checked: self._set_pane_visible("details", checked))
        self.view_button = QToolButton(self)
        self.view_button.setText("View")
        self.view_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.view_button.setMenu(self.view_menu)
        self.view_button.setAccessibleName("Open view options")
        self.view_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        apply_icon(self.view_button, "reveal")

        self.basic_mode_button = QPushButton("Basic")
        self.basic_mode_button.setCheckable(True)
        self.basic_mode_button.clicked.connect(lambda: self.set_active_workspace_mode("basic"))
        self.advanced_mode_button = QPushButton("Advanced")
        self.advanced_mode_button.setCheckable(True)
        self.advanced_mode_button.clicked.connect(lambda: self.set_active_workspace_mode("advanced"))
        self.runtime_badge = RuntimeBadge(self)
        self.health_badge = QLabel("Health: checking")
        self.health_badge.hide()
        self.jobs_widget = JobIndicatorWidget(self.job_manager, self)
        self.settings_button = QPushButton("Settings")
        self.settings_button.setAccessibleName("Open Settings")
        self.settings_button.setToolTip("Open Settings (Ctrl+,)")
        apply_icon(self.settings_button, "settings")
        self.settings_button.clicked.connect(self.open_settings_dialog)

        row.addWidget(self.feature_label)
        row.addWidget(self.gallery_workspace_button)
        row.addWidget(self.clustering_workspace_button)
        row.addWidget(self.faces_workspace_button)
        row.addWidget(self.names_workspace_button)
        row.addWidget(self.tags_workspace_button)
        row.addWidget(self.library_workspace_button)
        row.addWidget(self.current_folder_label, stretch=1)
        row.addWidget(self.mode_selector)
        row.addWidget(self.view_button)
        row.addWidget(self.jobs_widget)
        row.addWidget(self.runtime_badge)
        row.addWidget(self.settings_button)
        for legacy_widget in (
            self.source_toggle,
            self.controls_toggle,
            self.details_toggle,
            self.basic_mode_button,
            self.advanced_mode_button,
        ):
            legacy_widget.hide()
        QWidget.setTabOrder(self.gallery_workspace_button, self.clustering_workspace_button)
        QWidget.setTabOrder(self.clustering_workspace_button, self.faces_workspace_button)
        QWidget.setTabOrder(self.faces_workspace_button, self.names_workspace_button)
        QWidget.setTabOrder(self.names_workspace_button, self.tags_workspace_button)
        QWidget.setTabOrder(self.tags_workspace_button, self.library_workspace_button)
        QWidget.setTabOrder(self.library_workspace_button, self.mode_selector)
        QWidget.setTabOrder(self.mode_selector, self.view_button)
        QWidget.setTabOrder(self.view_button, self.jobs_widget)
        QWidget.setTabOrder(self.jobs_widget, self.runtime_badge)
        QWidget.setTabOrder(self.runtime_badge, self.settings_button)
        return bar

    def _on_mode_selector_changed(self, _index: int) -> None:
        mode = str(self.mode_selector.currentData() or "basic")
        self.set_active_workspace_mode(mode)

    def _connect_signals(self) -> None:
        self.source_pane.scope_changed.connect(self._on_active_scope_changed)
        self.source_pane.data_home_requested.connect(lambda: self.open_settings_dialog("Storage"))
        self.source_pane.recent_folder_remove_requested.connect(self._remove_recent_folder)
        self.source_pane.recent_folders_clear_requested.connect(self._clear_recent_folders)
        self.source_pane.run_requested.connect(self.run_clustering)
        self.source_pane.cancel_requested.connect(self.cancel_clustering)
        self.source_pane.hide_requested.connect(self._hide_source_pane)
        self.clustering_pane.run_requested.connect(self.run_clustering)
        self.clustering_pane.cancel_requested.connect(self.cancel_clustering)
        self.clustering_pane.hide_requested.connect(lambda: self._set_pane_visible("controls", False))
        self.cluster_pane.cluster_selected.connect(self.update_gallery)
        self.cluster_pane.recluster_requested.connect(self.recluster_selected_cluster)
        self.cluster_pane.open_in_gallery_requested.connect(self._open_selected_cluster_in_gallery)
        self.cluster_pane.hide_requested.connect(lambda: self._set_pane_visible("details", False))
        self.cluster_pane.selection_target_changed.connect(self._on_cluster_selection_target_changed)
        self.gallery_pane.first_paint_ready.connect(self._on_gallery_first_paint)
        self.gallery_pane.image_selected.connect(self._on_image_selected)
        self.gallery_pane.empty_select_folder_requested.connect(self._focus_folder_picker)
        self.gallery_pane.empty_run_requested.connect(self.run_clustering)
        self.runtime_badge.clicked.connect(self._open_runtime_status_details)
        self.gallery_pane.paths_removed.connect(self._on_gallery_paths_removed)
        self.gallery_pane.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.gallery_pane.metadata_changed.connect(self._on_gallery_metadata_changed)
        self.photo_gallery.organize_requested.connect(self.run_gallery_organize)
        self.photo_gallery.analyze_requested.connect(self._analyze_photo_set_in_clustering)
        self.photo_gallery.review_faces_requested.connect(self._review_photo_set_in_faces)
        self.photo_gallery.return_to_folder_requested.connect(self._return_to_folder_photo_gallery)
        self.photo_gallery.return_to_source_requested.connect(self._return_to_photo_set_source)
        self.photo_gallery.paths_removed.connect(self._on_gallery_paths_removed)
        self.photo_gallery.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.photo_gallery.metadata_changed.connect(self._on_gallery_metadata_changed)
        self.tags_pane.run_tag_filter_requested.connect(self._run_tag_filter_from_tags_workspace)
        self.tags_pane.generate_suggestions_requested.connect(self.generate_cluster_tag_suggestions)
        self.tags_pane.apply_suggestions_requested.connect(self.apply_cluster_tag_suggestions)
        self.tags_pane.metadata_changed.connect(self._on_gallery_metadata_changed)
        self.tags_pane.open_in_gallery_requested.connect(self._open_tagged_photos_in_gallery)
        self.library_pane.metadata_changed.connect(self._on_gallery_metadata_changed)
        self.library_pane.paths_renamed.connect(self._on_gallery_paths_renamed)
        self.library_pane.open_in_gallery_requested.connect(self._open_library_photos_in_gallery)
        self.footer_bar.clear_storage_requested.connect(self._request_runtime_storage_clear)
        self.session_controller.started.connect(self._on_clustering_started)
        self.session_controller.progress.connect(self._on_clustering_progress)
        self.session_controller.completed.connect(self._on_clustering_completed)
        self.session_controller.failed.connect(self._on_clustering_failed)
        self.session_controller.cancelled.connect(self._on_clustering_cancelled)
        self.session_controller.running_changed.connect(self._set_running_state)
        self.model_download_controller.started.connect(self._on_model_download_started)
        self.model_download_controller.progress.connect(self._on_model_download_progress)
        self.model_download_controller.completed.connect(self._on_model_download_completed)
        self.model_download_controller.failed.connect(self._on_model_download_failed)
        self.model_download_controller.cancelled.connect(self._on_model_download_cancelled)
        self._install_shell_shortcuts()

    def _install_shell_shortcuts(self) -> None:
        self._shell_shortcuts: list[QShortcut] = []
        shortcuts = (
            ("Ctrl+1", lambda: self.set_active_workspace("gallery")),
            ("Ctrl+2", lambda: self.set_active_workspace("clustering")),
            ("Ctrl+3", lambda: self.set_active_workspace("faces")),
            ("Ctrl+4", lambda: self.set_active_workspace("names")),
            ("Ctrl+5", lambda: self.set_active_workspace("tags")),
            ("Ctrl+6", lambda: self.set_active_workspace("library")),
            ("Ctrl+J", self._open_jobs_dialog),
            ("Ctrl+,", self.open_settings_dialog),
            ("Ctrl+O", self._focus_folder_picker),
            ("Ctrl+F", self._focus_workspace_search),
            ("Ctrl+R", self.run_clustering),
            ("Esc", self.cancel_clustering),
            ("F1", self._show_keyboard_help),
        )
        for sequence, handler in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
            shortcut.activated.connect(handler)
            self._shell_shortcuts.append(shortcut)

    def _focus_folder_picker(self) -> None:
        self._set_pane_visible("source", True)
        self.source_pane.file_tree.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _focus_workspace_search(self) -> None:
        if self._active_workspace != "faces":
            self.set_active_workspace("faces")
        pane = self._ensure_faces_workspace()
        if self.faces_pane is None:
            self._faces_focus_search_pending = True
            button = getattr(self, "faces_placeholder_button", None)
            if button is not None:
                button.setFocus(Qt.FocusReason.ShortcutFocusReason)
            return
        target = getattr(pane, "face_identity_search", None) or getattr(pane, "face_query_path", None)
        if target is not None:
            target.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def _open_jobs_dialog(self) -> None:
        """Keep the Jobs surface reachable without relying on footer focus."""

        opener = getattr(self.jobs_widget, "_open_jobs_dialog", None)
        if callable(opener):
            opener()

    def _show_keyboard_help(self) -> None:
        infoBox(
            "Keyboard shortcuts",
            "Ctrl+1 Gallery\nCtrl+2 Clustering\nCtrl+3 Faces\nCtrl+4 Names\nCtrl+5 Tags\nCtrl+6 Library\nCtrl+J Jobs\nCtrl+O Choose folder\nCtrl+F Search\nCtrl+R Run clustering\nEsc Cancel active work\nCtrl+, Settings\nF1 Help",
        )

    def _open_runtime_status_details(self) -> None:
        dialog = ProductionSettingsDialog(
            self.settings_store,
            self.runtime_service,
            describe_rebuildable_caches=self.cache_maintenance_service.describe_rebuildable_caches,
            describe_generated_storage=self._describe_generated_storage,
            clear_rebuildable_caches=self._clear_rebuildable_caches,
            clear_library_catalog=self._clear_library_catalog,
            clear_runtime_temp_files=self._clear_runtime_temp_files,
            clear_face_storage=self._clear_face_storage,
            clear_model_caches=self._clear_model_caches,
            clear_logs=self._clear_logs,
            clear_runtime_reports=self._clear_runtime_reports,
            clear_model_assets=self._clear_model_assets,
            can_clear_rebuildable_caches=lambda: not self._background_runtime_work_active(),
            data_home_manager=self.data_home_manager,
            active_source_roots_provider=lambda: tuple(self.source_pane.active_scope.roots),
            runtime_layout=self.runtime_layout,
            support_metadata_provider=self._support_metadata,
            model_download_controller=self.model_download_controller,
            job_manager=self.job_manager,
            parent=self,
        )
        dialog.runtime_rescanned.connect(self._begin_startup_readiness_check)
        support_index = next(
            (index for index in range(dialog.tabs.count()) if dialog.tabs.tabText(index) in {"Support", "Diagnostics"}),
            0,
        )
        dialog.tabs.setCurrentIndex(support_index)
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._apply_settings_values(dialog.values())
        self._refresh_workspace_ui()

    def _preferred_execution_mode(self) -> str:
        return str(self.settings_registry.get(self.settings_store, "runtime/preferred_mode", self.settings.preferred_execution_mode))

    def _gpu_conflict_policy(self) -> str:
        return str(self.settings_registry.get(self.settings_store, "runtime/gpu_conflict_policy", "ask") or "ask")

    def _show_gpu_conflict_choice(self, job_id: int, label: str) -> None:
        """Offer an asynchronous resource choice without nesting the event loop."""
        menu = QMenu(self)
        menu.setTitle("GPU is busy")
        queue_action = menu.addAction("Queue for GPU")
        cpu_action = menu.addAction("Run on CPU for now")
        cancel_action = menu.addAction("Cancel this job")
        queue_action.setToolTip("Keep this job queued until the active GPU job reaches a safe checkpoint.")
        cpu_action.setToolTip("Run this independent job with the CPU fallback. Jobs will record the fallback.")
        cancel_action.setToolTip("Cancel before this job starts; no work has been performed.")
        queue_action.triggered.connect(lambda: self.work_coordinator.resolve_gpu_conflict(job_id, "queue"))
        cpu_action.triggered.connect(lambda: self.work_coordinator.resolve_gpu_conflict(job_id, "cpu"))
        cancel_action.triggered.connect(lambda: self.work_coordinator.resolve_gpu_conflict(job_id, "cancel"))

        def _discard() -> None:
            if menu in self._gpu_conflict_menus:
                self._gpu_conflict_menus.remove(menu)
            menu.deleteLater()

        menu.aboutToHide.connect(_discard)
        self._gpu_conflict_menus.append(menu)
        self.footer_bar.set_status(f"GPU is busy for {label}. Choose Queue, CPU fallback, or Cancel in Jobs.")
        menu.popup(self.mapToGlobal(self.rect().center()))

    def _preferred_performance_profile(self) -> str:
        return str(self.settings_registry.get(self.settings_store, "performance/profile", self.settings.default_performance_profile))

    def _effective_performance_profile(self):
        base = select_performance_profile(self._preferred_performance_profile(), self.system_resources)
        return apply_performance_overrides(
            base,
            cpu_batch_size=int(self.settings_registry.get(self.settings_store, "performance/batch_size_cpu", base.cpu_batch_size)),
            gpu_batch_size=int(self.settings_registry.get(self.settings_store, "performance/batch_size_gpu", base.gpu_batch_size)),
            embedding_preprocess_workers=int(
                self.settings_registry.get(
                    self.settings_store,
                    "performance/decode_workers",
                    base.embedding_preprocess_workers,
                )
            ),
            thumbnail_workers=int(self.settings_registry.get(self.settings_store, "gallery/thumbnail_workers", base.thumbnail_workers)),
            thumbnail_prefetch_rows=int(
                self.settings_registry.get(self.settings_store, "gallery/prefetch_rows", base.thumbnail_prefetch_rows)
            ),
            vram_headroom_mb=int(
                self.settings_registry.get(self.settings_store, "performance/vram_headroom_mb", base.vram_headroom_mb)
            ),
        )

    def _preferred_workspace(self) -> str:
        return self._normalize_workspace_id(str(self.settings_registry.get(self.settings_store, "workspace/default_view", "gallery")))

    def _preferred_faces_ui_mode(self) -> str:
        return self._normalize_ui_mode(str(self.settings_registry.get(self.settings_store, "workspace/faces_mode", "basic")))

    def _preferred_face_mode(self) -> str:
        return "human"

    def _face_model_root(self) -> str:
        return str(self.settings_registry.get(self.settings_store, "faces/model_root", "") or "").strip()

    def _preferred_face_detector(self, mode: str) -> str:
        return self._preferred_face_pipeline_ids(mode)[0]

    def _preferred_face_embedder(self, mode: str) -> str:
        return self._preferred_face_pipeline_ids(mode)[1]

    def _preferred_face_pipeline_request_ids(self, mode: str) -> tuple[str, str]:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        configured_detector = str(
            self.settings_registry.get(self.settings_store, f"faces/default_detector/{mode_id}") or ""
        ).strip()
        configured_embedder = str(
            self.settings_registry.get(self.settings_store, f"faces/default_embedder/{mode_id}") or ""
        ).strip()
        if mode_id == "human":
            configured_detector = configured_detector or face_search.DEFAULT_HUMAN_FACE_DETECTOR_ID
            configured_embedder = configured_embedder or face_search.DEFAULT_HUMAN_FACE_EMBEDDER_ID
        return (
            face_search.normalize_face_component_id(configured_detector),
            face_search.normalize_face_component_id(configured_embedder),
        )

    def _preferred_face_pipeline_ids(self, mode: str) -> tuple[str, str]:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        configured_detector, configured_embedder = self._preferred_face_pipeline_request_ids(mode_id)
        return face_search.resolve_ready_face_pipeline_ids(
            self._face_model_root(),
            mode_id,
            configured_detector,
            configured_embedder,
        )

    def _preferred_face_detector_score_threshold(self, mode: str) -> float:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        return float(
            self.settings_registry.get(
                self.settings_store,
                f"faces/detector_score_threshold/{mode_id}",
                None,
            )
        )

    def _preferred_face_max_detections(self, mode: str) -> int:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        return max(
            1,
            int(
                self.settings_registry.get(
                    self.settings_store,
                    f"faces/max_detections/{mode_id}",
                    face_search.DEFAULT_FACE_MAX_DETECTIONS,
                )
            ),
        )

    def _preferred_face_pipeline_defaults(self) -> dict[str, dict[str, object]]:
        defaults: dict[str, dict[str, object]] = {}
        for mode in ("human",):
            preferred_detector_id, preferred_embedder_id = self._preferred_face_pipeline_request_ids(mode)
            detector_id, embedder_id = self._preferred_face_pipeline_ids(mode)
            defaults[mode] = {
                "detector_id": detector_id,
                "embedder_id": embedder_id,
                "preferred_detector_id": preferred_detector_id,
                "preferred_embedder_id": preferred_embedder_id,
                "score_threshold": self._preferred_face_detector_score_threshold(mode),
                "max_detections": self._preferred_face_max_detections(mode),
            }
            raw_preferences = str(
                self.settings_registry.get(self.settings_store, f"faces/pipeline_preferences/{mode}", "") or ""
            ).strip()
            if raw_preferences:
                try:
                    saved_preferences = json.loads(raw_preferences)
                except (TypeError, ValueError):
                    saved_preferences = None
                if isinstance(saved_preferences, dict):
                    defaults[mode].update({str(key): value for key, value in saved_preferences.items()})
        return defaults

    def _persist_face_pipeline_defaults(self, preferences_by_mode: dict[str, object]) -> None:
        for raw_mode, raw_preferences in dict(preferences_by_mode or {}).items():
            mode = _face_search_api().normalize_face_mode(str(raw_mode))
            if mode != "human" or not isinstance(raw_preferences, dict):
                continue
            preferences = {str(key): value for key, value in raw_preferences.items()}
            detector_id = str(preferences.get("detector_id") or "").strip()
            embedder_id = str(preferences.get("embedder_id") or "").strip()
            if detector_id:
                self.settings_registry.set(self.settings_store, f"faces/default_detector/{mode}", detector_id)
            if embedder_id:
                self.settings_registry.set(self.settings_store, f"faces/default_embedder/{mode}", embedder_id)
            self.settings_registry.set(
                self.settings_store,
                f"faces/detector_score_threshold/{mode}",
                float(preferences.get("score_threshold", 0.35) or 0.0),
            )
            self.settings_registry.set(
                self.settings_store,
                f"faces/max_detections/{mode}",
                max(1, int(preferences.get("max_detections", 50) or 50)),
            )
            self.settings_registry.set(
                self.settings_store,
                f"faces/pipeline_preferences/{mode}",
                json.dumps(preferences, ensure_ascii=False, sort_keys=True),
            )
        self.settings_store.sync()

    def _keep_worker_warm(self) -> bool:
        return bool(self.settings_registry.get(self.settings_store, "performance/keep_worker_warm", False))

    def _offline_model_downloads(self) -> bool:
        return bool(self.settings_registry.get(self.settings_store, "models/offline_mode", bool(getattr(sys, "frozen", False))))

    def _read_only_mode(self) -> bool:
        return bool(self.settings_registry.get(self.settings_store, "safety/read_only_mode", False))

    def _library_context_settings(self) -> VisionLanguageSettings:
        provider = str(self.settings_registry.get(self.settings_store, "library/vision_provider", "ollama") or "ollama")
        model = str(self.settings_registry.get(self.settings_store, "library/vision_model", "") or "")
        endpoint = str(self.settings_registry.get(self.settings_store, "library/vision_endpoint", "http://127.0.0.1:11434") or "")
        key_environment = str(
            self.settings_registry.get(
                self.settings_store,
                "library/vision_api_key_environment",
                "CLUSTERLENS_LLM_API_KEY",
            )
            or "CLUSTERLENS_LLM_API_KEY"
        )
        return VisionLanguageSettings(
            provider=provider,
            ollama_url=endpoint if provider == "ollama" else "http://127.0.0.1:11434",
            ollama_model=model if provider == "ollama" else "",
            openai_url=endpoint if provider == "openai-compatible" else "",
            openai_model=model if provider == "openai-compatible" else "",
            api_key_environment=key_environment,
            remote_consent=False,
        )

    def _library_auto_context_enabled(self) -> bool:
        return bool(self.settings_registry.get(self.settings_store, "library/auto_cluster_context", False))

    def _save_library_context_settings(self, settings: VisionLanguageSettings, auto_enabled: bool) -> None:
        self.settings_registry.set(self.settings_store, "library/vision_provider", settings.provider)
        self.settings_registry.set(self.settings_store, "library/vision_model", settings.model)
        endpoint = settings.ollama_url if settings.provider == "ollama" else settings.openai_url
        self.settings_registry.set(self.settings_store, "library/vision_endpoint", endpoint)
        self.settings_registry.set(self.settings_store, "library/vision_api_key_environment", settings.api_key_environment)
        self.settings_registry.set(self.settings_store, "library/auto_cluster_context", bool(auto_enabled))
        self.settings_store.sync()

    def _clear_face_session_dbs(self) -> None:
        for path in self.settings.cache_dir.glob("face_search*_session*.db*"):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _normalize_workspace_id(value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"gallery", "photos", "photo gallery"}:
            return "gallery"
        if text in {"faces", "face_search", "face search"}:
            return "faces"
        if text in {"names", "people", "saved names"}:
            return "names"
        if text in {"tags", "tag", "tag manager"}:
            return "tags"
        if text in {"library", "archive", "photo library"}:
            return "library"
        return "clustering"

    @staticmethod
    def _normalize_ui_mode(mode: str) -> str:
        return "advanced" if str(mode or "").strip().lower() == "advanced" else "basic"

    @staticmethod
    def _face_pipeline_slug(detector_id: str, embedder_id: str) -> str:
        face_search = _face_search_api()
        detector = face_search.normalize_face_component_id(detector_id, "detector")
        embedder = face_search.normalize_face_component_id(embedder_id, "embedder")
        return f"{detector}__{embedder}"

    def _face_db_path_for_pipeline(self, scope: str, mode: str, detector_id: str, embedder_id: str) -> Path:
        face_search = _face_search_api()
        scope_id = "session" if str(scope or "").strip().lower() == "session" else "global"
        mode_id = "human"
        detector = face_search.normalize_face_component_id(detector_id, face_search.default_face_detector_id(self._face_model_root(), mode_id))
        embedder = face_search.normalize_face_component_id(embedder_id, face_search.default_face_embedder_id(self._face_model_root(), mode_id))
        if mode_id == "human" and detector == face_search.BUILTIN_HUMAN_DETECTOR_ID and embedder == face_search.BUILTIN_HUMAN_EMBEDDER_ID:
            return self.settings.cache_dir / ("face_search_session.db" if scope_id == "session" else "face_search_index.db")
        scope_suffix = "_session" if scope_id == "session" else ""
        pipeline_slug = self._face_pipeline_slug(detector, embedder)
        return self.settings.cache_dir / f"face_search_{mode_id}_{pipeline_slug}{scope_suffix}.db"

    def _face_service_for_pipeline(
        self,
        scope: str,
        mode: str,
        detector_id: str,
        embedder_id: str,
    ) -> FaceIndexService:
        face_search = _face_search_api()
        scope_id = "session" if str(scope or "").strip().lower() == "session" else "global"
        mode_id = "human"
        requested_detector = face_search.normalize_face_component_id(
            detector_id,
            face_search.default_face_detector_id(self._face_model_root(), mode_id),
        )
        requested_embedder = face_search.normalize_face_component_id(
            embedder_id,
            face_search.default_face_embedder_id(self._face_model_root(), mode_id),
        )
        detector, embedder = face_search.resolve_ready_face_pipeline_ids(
            self._face_model_root(),
            mode_id,
            requested_detector,
            requested_embedder,
        )
        if (detector, embedder) != (requested_detector, requested_embedder):
            LOGGER.info(
                "Face pipeline %s/%s is not ready; using ready pipeline %s/%s.",
                requested_detector,
                requested_embedder,
                detector,
                embedder,
            )
        key = (scope_id, mode_id, detector, embedder)
        service = self._face_service_cache.get(key)
        if service is None:
            service = face_search.FaceIndexService(
                mode=mode_id,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
                model_root=self._face_model_root(),
                detector_id=detector,
                embedder_id=embedder,
                detector_score_threshold=self._preferred_face_detector_score_threshold(mode_id),
                detector_max_detections=self._preferred_face_max_detections(mode_id),
                performance_profile=self.performance_profile,
                db_path=self._face_db_path_for_pipeline(scope_id, mode_id, detector, embedder),
            )
            self._face_service_cache[key] = service
        return service

    def _build_face_services(self, *, reset_face_session: bool) -> None:
        if reset_face_session:
            self._clear_face_session_dbs()
        self._face_service_cache = {}
        self.face_services_global = {
            mode: self._face_service_for_pipeline(
                "global",
                mode,
                self._preferred_face_detector(mode),
                self._preferred_face_embedder(mode),
            )
            for mode in ("human",)
        }
        self.face_services_session = {
            mode: self._face_service_for_pipeline(
                "session",
                mode,
                self._preferred_face_detector(mode),
                self._preferred_face_embedder(mode),
            )
            for mode in ("human",)
        }
        self.face_service_global = self.face_services_global["human"]
        self.face_service_session = self.face_services_session["human"]

    def _apply_workspace_preferences(self) -> None:
        thumbnail_size = int(self.settings_registry.get(self.settings_store, "gallery/thumbnail_size", self.settings.thumbnail_size))
        power_user_mode = bool(self.settings_registry.get(self.settings_store, "workspace/power_user_mode", False))
        self.setProperty("powerUserMode", power_user_mode)
        if power_user_mode:
            self.set_clustering_mode("advanced")
            self.set_faces_mode("advanced")
        self.performance_profile = self._effective_performance_profile()
        seen_face_services: set[int] = set()
        for service in tuple(getattr(self, "_face_service_cache", {}).values()):
            if id(service) in seen_face_services:
                continue
            seen_face_services.add(id(service))
            configure = getattr(service, "configure_index_performance", None)
            if callable(configure):
                configure(self.performance_profile)
        self.gallery_pane.apply_view_preferences(
            thumbnail_size=thumbnail_size,
            worker_count=self.performance_profile.thumbnail_workers,
            prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
            pixmap_cache_size=self.performance_profile.pixmap_cache_size,
            qimage_cache_size=self.performance_profile.qimage_cache_size,
        )
        self.photo_gallery.set_view_preferences(
            thumbnail_size=thumbnail_size,
            cache_size=self.performance_profile.qimage_cache_size,
        )
        if self.faces_pane is not None:
            self.faces_pane.results_gallery.apply_view_preferences(
                thumbnail_size=thumbnail_size,
                worker_count=self.performance_profile.thumbnail_workers,
                prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
                pixmap_cache_size=self.performance_profile.pixmap_cache_size,
                qimage_cache_size=self.performance_profile.qimage_cache_size,
            )
            self.faces_pane.apply_face_tile_preferences(
                worker_count=self.performance_profile.thumbnail_workers,
                cache_size=self.performance_profile.pixmap_cache_size,
            )
        if self.names_pane is not None:
            self.names_pane.gallery.apply_view_preferences(
                thumbnail_size=thumbnail_size,
                worker_count=self.performance_profile.thumbnail_workers,
                prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
                pixmap_cache_size=self.performance_profile.pixmap_cache_size,
                qimage_cache_size=self.performance_profile.qimage_cache_size,
            )
        if self.tags_pane is not None:
            self.tags_pane.gallery.apply_view_preferences(
                thumbnail_size=thumbnail_size,
                worker_count=self.performance_profile.thumbnail_workers,
                prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
                pixmap_cache_size=self.performance_profile.pixmap_cache_size,
                qimage_cache_size=self.performance_profile.qimage_cache_size,
            )
        if self.library_pane is not None:
            self.library_pane.apply_view_preferences(
                thumbnail_size=thumbnail_size,
                worker_count=self.performance_profile.thumbnail_workers,
                prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
                pixmap_cache_size=self.performance_profile.pixmap_cache_size,
                qimage_cache_size=self.performance_profile.qimage_cache_size,
            )
        self.runtime_badge.setVisible(bool(self.settings_registry.get(self.settings_store, "runtime/show_badge", self.settings.show_runtime_badge)))
        self._reflow_main_splitter(force=False)

    def _sync_mode_buttons(self) -> None:
        active_mode = self._clustering_mode if self._active_workspace == "clustering" else self._faces_mode
        for key, button in {"basic": self.basic_mode_button, "advanced": self.advanced_mode_button}.items():
            button.blockSignals(True)
            button.setChecked(active_mode == key)
            button.blockSignals(False)
        if hasattr(self, "mode_selector"):
            target_index = self.mode_selector.findData(active_mode)
            if target_index >= 0:
                self.mode_selector.blockSignals(True)
                self.mode_selector.setCurrentIndex(target_index)
                self.mode_selector.blockSignals(False)

    def _sync_workspace_buttons(self) -> None:
        for key, button in {
            "gallery": self.gallery_workspace_button,
            "clustering": self.clustering_workspace_button,
            "faces": self.faces_workspace_button,
            "names": self.names_workspace_button,
            "tags": self.tags_workspace_button,
            "library": self.library_workspace_button,
        }.items():
            button.blockSignals(True)
            button.setChecked(self._active_workspace == key)
            button.blockSignals(False)

    def _source_pane_visible(self) -> bool:
        if self._scope_editor_pinned:
            return True
        if self._active_workspace == "gallery":
            return True
        if self._active_workspace in {"names", "tags", "library"}:
            return False
        if self._active_workspace == "faces":
            return bool(self._advanced_pane_visibility["source"])
        if self._clustering_mode == "basic":
            return True
        return bool(self._pane_visibility["source"])

    def _edit_active_roots(self) -> None:
        self._scope_editor_pinned = True
        self._reflow_main_splitter(force=True)
        self.source_pane.file_tree.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.footer_bar.set_status("Edit the checked active roots. This does not start a scan.")

    def _hide_source_pane(self) -> None:
        self._scope_editor_pinned = False
        self._set_pane_visible("source", False)
        self._reflow_main_splitter(force=True)

    def _refresh_workspace_ui(self) -> None:
        is_gallery = self._active_workspace == "gallery"
        is_clustering = self._active_workspace == "clustering"
        is_faces = self._active_workspace == "faces"
        is_names = self._active_workspace == "names"
        is_tags = self._active_workspace == "tags"
        is_library = self._active_workspace == "library"
        faces_pane = self._ensure_faces_workspace() if is_faces else None
        names_pane = self._ensure_names_workspace() if is_names else None
        if is_gallery:
            self.workspace_stack.setCurrentWidget(self.photo_gallery_workspace)
        elif is_names:
            self.workspace_stack.setCurrentWidget(names_pane)
        elif is_tags:
            self.workspace_stack.setCurrentWidget(self.tags_pane)
            if self.tags_pane.inventory_model.rowCount() == 0:
                self.tags_pane.refresh()
        elif is_library:
            self.workspace_stack.setCurrentWidget(self.library_pane)
            if self._library_catalog_refresh_pending:
                self._library_catalog_refresh_pending = False
                self.library_pane.invalidate_catalog_view()
            self.library_pane.ensure_timeline_loaded()
        else:
            self.workspace_stack.setCurrentWidget(self.clustering_workspace if is_clustering else faces_pane)
        if is_faces and self.faces_pane is not None:
            self.faces_pane.ensure_current_faces_tab_loaded()
        if self.faces_pane is not None:
            self.faces_pane.set_ui_mode(self._faces_mode)
        self.source_pane.set_basic_mode(
            is_clustering and self._clustering_mode == "basic",
            running=self.session_controller.is_running(),
        )
        self.source_toggle.setVisible(False)
        self.controls_toggle.setVisible(False)
        self.details_toggle.setVisible(False)
        self.feature_label.setText(PRODUCTION_DISPLAY_NAME)
        self.controls_view_action.setVisible(is_clustering and self._clustering_mode == "advanced")
        self.details_view_action.setVisible(is_clustering and self._clustering_mode == "advanced")
        self.source_view_action.setVisible((is_faces) or (is_clustering and self._clustering_mode == "advanced"))
        self.mode_selector.setVisible(not is_gallery and not is_names and not is_tags and not is_library)
        self._sync_mode_buttons()
        self._sync_workspace_buttons()
        self._sync_pane_toggle_buttons()
        self._reflow_main_splitter(force=True)

    def set_active_workspace(self, workspace_id: str) -> None:
        target = self._normalize_workspace_id(workspace_id)
        if self._active_workspace == "faces" and target != "faces":
            self._cancel_faces_workspace_load()
        self._active_workspace = target
        self.settings_store.setValue("workspace/default_view", self._active_workspace)
        self._refresh_workspace_ui()

    def set_active_workspace_mode(self, mode: str) -> None:
        if self._active_workspace in {"gallery", "names", "tags", "library"}:
            return
        if self._active_workspace == "faces":
            self.set_faces_mode(mode)
            return
        self.set_clustering_mode(mode)

    def set_faces_mode(self, mode: str) -> None:
        self._faces_mode = self._normalize_ui_mode(mode)
        self.settings_store.setValue("workspace/faces_mode", self._faces_mode)
        if self.faces_pane is not None:
            self.faces_pane.set_ui_mode(self._faces_mode)
        if self._active_workspace == "faces":
            self._refresh_workspace_ui()
        else:
            self._sync_mode_buttons()

    def _selected_startup_clustering_models(self) -> tuple[str, ...]:
        return tuple(
            normalize_embedding_models(
                self.clustering_pane.selected_embedding_models(),
                default_model=self.settings.default_model,
                scope="production",
            )
        )

    def _begin_startup_readiness_check(self) -> None:
        """Check local runtime/model readiness without downloading or loading inference models."""

        if self._is_shutting_down or self._close_pending:
            return
        if not self._startup_maintenance_complete:
            self._start_startup_maintenance()
            return
        previous_job = self._startup_readiness_job
        if previous_job is not None:
            try:
                previous_job.cancel()
            except Exception:
                pass
        self._startup_readiness_generation += 1
        generation = self._startup_readiness_generation
        self._startup_readiness_report = None
        self._startup_readiness_error = ""
        self._apply_startup_workflow_gate()
        clustering_models = self._selected_startup_clustering_models()
        face_detector, face_embedder = self._preferred_face_pipeline_request_ids("human")
        preferred_mode = self._preferred_execution_mode()
        face_model_root = self._face_model_root()

        def _run(progress, cancel_check):
            progress(-1, "Checking CUDA and downloaded models…")
            raise_if_cancelled(cancel_check)
            report = inspect_startup_readiness(
                self.runtime_service,
                self.model_asset_service,
                preferred_execution_mode=preferred_mode,
                clustering_models=clustering_models,
                face_model_root=face_model_root,
                face_detector_id=face_detector,
                face_embedder_id=face_embedder,
            )
            raise_if_cancelled(cancel_check)
            return report

        job = AsyncJob(_run)
        self._startup_readiness_job = job
        job_id = self.job_manager.register_job(
            "Checking runtime readiness",
            cancel_fn=job.cancel,
            origin="Startup",
        )
        self._startup_readiness_job_id = job_id
        thread_holder: dict[str, object | None] = {"thread": None}

        def _finish(status: str, error: str = "") -> None:
            self.job_manager.finish(job_id, status=status, error=error)
            if self._startup_readiness_job_id == job_id:
                self._startup_readiness_job_id = None

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            if self._startup_readiness_job is job:
                self._startup_readiness_job = None
            if self._startup_readiness_thread is thread:
                self._startup_readiness_thread = None
            self._release_async_refs(job, thread)

        def _completed(report: object) -> None:
            if (
                self._is_shutting_down
                or generation != self._startup_readiness_generation
                or not self._can_update_widget(getattr(self, "runtime_badge", None))
                or not self._can_update_widget(getattr(self, "footer_bar", None))
            ):
                _finish("cancelled")
                _cleanup()
                return
            if not isinstance(report, StartupReadinessReport):
                _failed("The readiness check returned an invalid result.")
                return
            self._startup_readiness_report = report
            self._startup_readiness_error = ""
            self.execution_policy = report.execution_policy
            self.runtime_badge.update_runtime(report.capabilities, report.execution_policy)
            self._apply_startup_workflow_gate()
            if self._pending_photo_face_tool_requests:
                if report.face_ready:
                    if not self._thread_is_running(self._names_init_thread):
                        self._start_faces_workspace_load()
                else:
                    self._finish_pending_photo_face_tool_requests(error=report.face_message)
            if report.clustering_ready and report.face_ready:
                self.footer_bar.set_status("Runtime, clustering models, and face models are ready.")
                self._set_activity("Ready")
            else:
                missing = " | ".join(
                    message
                    for ready, message in (
                        (report.clustering_ready, report.clustering_message),
                        (report.face_ready, report.face_message),
                    )
                    if not ready
                )
                self.footer_bar.set_status(missing)
                self._set_activity("Model setup required")
            _finish("finished")
            _cleanup()

        def _failed(message: str) -> None:
            if (
                self._is_shutting_down
                or generation != self._startup_readiness_generation
                or not self._can_update_widget(getattr(self, "footer_bar", None))
            ):
                _finish("cancelled")
                _cleanup()
                return
            text = str(message or "Readiness check failed.").strip()
            self._startup_readiness_error = text
            self.footer_bar.set_status(f"Runtime readiness could not be confirmed: {text}")
            self._set_activity("Readiness check failed")
            self._finish_pending_photo_face_tool_requests(error=text)
            _finish("failed", text)
            _cleanup()

        def _cancelled() -> None:
            _finish("cancelled")
            _cleanup()

        job.progress.connect(
            lambda value, text: self.job_manager.update(job_id, progress=value, text=str(text))
        )
        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._startup_readiness_thread = thread
        self._retain_async_refs(job, thread)

    def _apply_startup_workflow_gate(self) -> None:
        """Keep model-dependent workflows disabled until local readiness is known."""

        if self._is_shutting_down or self._close_pending:
            return

        report = self._startup_readiness_report
        clustering_ready = bool(report is not None and report.clustering_ready)
        face_ready = bool(report is not None and report.face_ready)
        pending = report is None
        run_in_progress = self._run_request_active()
        clustering_tip = (
            self._startup_readiness_error
            if self._startup_readiness_error
            else (
                "Checking CUDA and downloaded clustering models…"
                if pending
                else (report.clustering_message if report is not None else "Runtime readiness is unavailable.")
            )
        )
        for button in (
            getattr(self.source_pane, "basic_run_button", None),
            getattr(self.clustering_pane, "cluster_button", None),
            getattr(self.gallery_pane, "empty_run_button", None),
            getattr(getattr(self, "photo_gallery", None), "organize_button", None),
        ):
            if not self._can_update_widget(button):
                continue
            can_run = clustering_ready and not run_in_progress
            if button is getattr(self.gallery_pane, "empty_run_button", None):
                can_run = can_run and not self.source_pane.active_scope.is_empty
            elif button is getattr(getattr(self, "photo_gallery", None), "organize_button", None):
                can_run = can_run and len(self._gallery_paths) >= 2
            button.setEnabled(can_run)
            button.setToolTip(clustering_tip if not clustering_ready else "Run clustering with the selected local model.")
        if self._can_update_widget(self.faces_pane):
            self.faces_pane.set_startup_readiness(
                face_ready,
                report.face_message
                if report is not None
                else (self._startup_readiness_error or "Checking CUDA and downloaded face models…"),
            )
        elif self._can_update_widget(getattr(self, "faces_placeholder_detail", None)):
            if pending:
                self.faces_placeholder_detail.setText(
                    self._startup_readiness_error or "Checking CUDA and downloaded face models…"
                )
            elif not face_ready:
                self.faces_placeholder_detail.setText(
                    f"Face model setup is required before scanning or searching. {report.face_message}"
                )
            else:
                self.faces_placeholder_detail.setText("Face indexing, search, review, and identity tools are ready to load.")

    def _clustering_ready_for_action(self) -> bool:
        report = self._startup_readiness_report
        if report is not None and report.clustering_ready:
            return True
        message = (
            "ClusterLens is still checking CUDA and downloaded clustering models."
            if report is None
            else report.clustering_message
        )
        errorBox("Clustering is not ready", message)
        return False

    def _build_request(self, *, source_paths: list[str] | None = None) -> ProductionClusterRequest:
        similarity_modes = self.clustering_pane.selected_similarity_modes()
        embedding_models = normalize_embedding_models(
            self.clustering_pane.selected_embedding_models(),
            default_model=self.settings.default_model,
            scope="production",
        )
        clustering_backends = normalize_clustering_backends(
            self.clustering_pane.selected_clustering_backends(),
            scope="production",
        )
        return ProductionClusterRequest(
            directory=self.source_pane.active_scope.primary_root,
            embedding_models=embedding_models,
            num_clusters=int(self.clustering_pane.cluster_spinbox.value()),
            clustering_backends=clustering_backends,
            recursive=bool(self.clustering_pane.recursive_checkbox.isChecked()),
            similarity_mode=similarity_modes[0],
            similarity_modes=similarity_modes,
            outlier_policy=self.clustering_pane.outlier_combobox.currentText(),
            use_onnx=bool(self.clustering_pane.onnx_checkbox.isChecked()),
            reuse_result_cache=bool(self.clustering_pane.result_cache_checkbox.isChecked()),
            use_embedding_cache_lookup=bool(self.clustering_pane.embedding_cache_lookup_checkbox.isChecked()),
            source_roots=list(self.source_pane.active_scope.roots),
            source_paths=source_paths,
            performance_profile=self._preferred_performance_profile(),
            batch_size_cpu=int(self.performance_profile.cpu_batch_size),
            batch_size_gpu=int(self.performance_profile.gpu_batch_size),
            preprocess_workers=int(self.performance_profile.embedding_preprocess_workers),
            vram_headroom_mb=int(self.performance_profile.vram_headroom_mb),
            preferred_execution_mode=self._preferred_execution_mode(),
            tag_filter=self.tags_pane.selected_filter_tags() if self.tags_pane is not None else [],
            tag_match=self.tags_pane.selected_match_mode() if self.tags_pane is not None else "Any",
            generate_cluster_meanings=False,
            generate_cluster_explanations=True,
            cluster_meaning_model="auto",
            backend_options_by_backend=self.clustering_pane.backend_options_by_backend(),
        )

    def _confirm_large_folder_if_needed(self, request: ProductionClusterRequest) -> bool:
        # Discovery is deliberately completed by the cancellable preflight job.
        # Never enumerate the source folder from this UI-thread confirmation.
        estimated = len(request.source_paths or ())
        if estimated < 50_000:
            return True
        severity = "very large" if estimated >= 100_000 else "large"
        count_text = str(estimated)
        message = (
            f"This looks like a {severity} run: about {count_text} image(s).\n\n"
            "Large runs can use substantial RAM, VRAM, disk cache, and time. "
            "Use low_memory profile if this machine has limited memory, or run a smaller/tag-filtered folder first.\n\n"
            f"Performance profile: {self._preferred_performance_profile()}\n"
            f"Runtime cache: {self.settings.cache_dir}\n\n"
            "Continue?"
        )
        return bool(confirmBox("Large folder guardrail", message, parent=self))

    def run_clustering(self) -> None:
        if not self._clustering_ready_for_action():
            return
        if self._run_request_active():
            errorBox("Busy", "A production clustering run is already active.")
            return
        if self._thread_is_running(self._storage_clear_thread):
            errorBox("Busy", "Wait for storage cleanup to finish before starting a new clustering run.")
            return
        try:
            request = self._build_request()
        except Exception as exc:
            LOGGER.exception("Failed to build clustering request")
            errorBox("Clustering could not start", str(exc))
            return
        self._start_request_preflight(request)

    def recluster_selected_cluster(self) -> None:
        if not self._clustering_ready_for_action():
            return
        if self._run_request_active():
            errorBox("Busy", "A production clustering run is already active.")
            return
        if self._thread_is_running(self._storage_clear_thread):
            errorBox("Busy", "Wait for storage cleanup to finish before starting a new clustering run.")
            return
        target = self.cluster_pane.current_selection_target()
        if target is None or len(target.paths) < 2:
            errorBox("Select a cluster", "Choose a cluster with at least two images first.")
            return
        request = self._build_request(source_paths=list(target.paths))
        request = self._prepare_request_model_downloads(request)
        if request is None:
            return
        if not self._start_request_with_assets(request, run_origin="recluster"):
            errorBox("Busy", "A production clustering run is already active.")

    def _prepare_request_model_downloads(self, request: ProductionClusterRequest) -> ProductionClusterRequest | None:
        plan = self.model_asset_service.build_download_plan(
            request.embedding_models,
            generate_cluster_meanings=bool(request.generate_cluster_meanings),
            requested_meaning_model=request.cluster_meaning_model,
        )
        if not plan.requires_download:
            use_bundled_onnx = any(self.model_asset_service.find_bundle(model_name) is not None for model_name in request.embedding_models)
            return replace(request, allow_model_downloads=False, use_onnx=bool(request.use_onnx or use_bundled_onnx))

        if self._offline_model_downloads():
            LOGGER.info("Offline model mode is enabled; using bundled/local fallback for missing models: %s", plan.unavailable_items)
            self.footer_bar.set_status("Offline model mode enabled. Using bundled/local fallback.")
            self._set_activity("Offline model fallback")
        elif confirmBox("Model download required", self._model_download_prompt_text(plan), parent=self):
            self.footer_bar.set_status("Model download approved. It will appear in Jobs and can be cancelled safely.")
            self._set_activity("Model download approved")
            return replace(request, allow_model_downloads=True)

        safe_models = list(plan.runnable_without_download)
        fallback_model = plan.bundled_fallback_model
        if not safe_models and fallback_model:
            safe_models = [fallback_model]
        if not safe_models:
            errorBox(
                "No bundled fallback model",
                (
                    "The selected run needs model files that are not bundled and no bundled fallback model was found.\n\n"
                    "Allow the download, or package at least the Fast Preview ONNX bundle with the final binary."
                ),
            )
            return None

        disabled_meanings = bool(request.generate_cluster_meanings and plan.meaning_requires_download)
        fallback_request = replace(
            request,
            embedding_models=safe_models,
            use_onnx=True,
            allow_model_downloads=False,
            generate_cluster_meanings=bool(request.generate_cluster_meanings and not disabled_meanings),
        )
        status = f"Download declined. Running with bundled/local model(s): {', '.join(safe_models)}."
        if disabled_meanings:
            status += " Advanced Cluster Meaning naming is disabled for this run."
        self.footer_bar.set_status(status)
        self._set_activity("Using bundled/local fallback")
        LOGGER.info(
            "Model downloads declined; requested=%s fallback=%s disabled_meanings=%s missing=%s",
            request.embedding_models,
            safe_models,
            disabled_meanings,
            plan.unavailable_items,
        )
        return fallback_request

    def _model_download_prompt_text(self, plan: ModelDownloadPlan) -> str:
        lines = [
            "The selected run needs model files that are not bundled with this build.",
            f"Requested models: {', '.join(plan.requested_models)}",
            "",
            "Required downloads:",
        ]
        for item in self._model_download_items_for_plan(plan):
            source = MODEL_SOURCE_LABELS.get(item.model_name, "external model weights")
            asset_kind = "image weights + text tokenizer" if item.require_text else "image weights"
            lines.append(f"- {item.model_name} ({asset_kind}): {source}")
        lines.extend(
            [
                "",
                f"Download location: {self.settings.cache_dir}",
                "If you choose No, the app will run only with already local or bundled model assets.",
            ]
        )
        if plan.runnable_without_download:
            lines.append(f"Available without download: {', '.join(plan.runnable_without_download)}")
        if plan.bundled_fallback_model:
            lines.append(f"Bundled fallback: {model_label(plan.bundled_fallback_model)}")
        else:
            lines.append("Bundled fallback: none found")
        lines.append("Downloads are deduplicated, resumable in the shared cache, visible in Jobs, and cancellable.")
        lines.append("Fallback policy: declining uses available local/bundled models and disables unavailable advanced naming.")
        lines.append("")
        lines.append("Allow the download now?")
        return "\n".join(lines)

    @staticmethod
    def _model_download_items_for_plan(plan: ModelDownloadPlan) -> tuple[ModelDownloadItem, ...]:
        items = [ModelDownloadItem(model_name, False) for model_name in plan.models_requiring_download]
        if plan.meaning_requires_download and plan.meaning_model:
            items.append(ModelDownloadItem(plan.meaning_model, True))
        return normalize_model_download_items(items)

    def cancel_clustering(self) -> None:
        if self._preflight_job is not None:
            self._cancel_request_preflight()
            return
        if self._pending_model_download_run is not None and self.model_download_controller.is_running():
            self.model_download_controller.cancel()
            return
        self.session_controller.cancel()

    def _start_request_with_assets(self, request: ProductionClusterRequest, *, run_origin: str) -> bool:
        if not request.allow_model_downloads:
            return self._start_request(request, run_origin=run_origin)

        plan = self.model_asset_service.build_download_plan(
            request.embedding_models,
            generate_cluster_meanings=bool(request.generate_cluster_meanings),
            requested_meaning_model=request.cluster_meaning_model,
        )
        items = self._model_download_items_for_plan(plan)
        if not items:
            return self._start_request(replace(request, allow_model_downloads=False), run_origin=run_origin)
        if self.model_download_controller.is_running():
            return False

        # Network access and model initialization are deliberately completed in
        # a separate killable process. The clustering worker itself is offline.
        self._pending_model_download_run = (replace(request, allow_model_downloads=False), str(run_origin))
        self._set_running_state(True)
        if self.model_download_controller.start(items):
            return True
        self._pending_model_download_run = None
        self._set_running_state(False)
        return False

    def _start_request(self, request: ProductionClusterRequest, *, run_origin: str) -> bool:
        policy = self.runtime_service.select_policy(request.preferred_execution_mode, refresh=True)
        self.execution_policy = policy
        self.runtime_badge.update_runtime(self.runtime_service.detect(), policy)
        if policy.cuda_required_unavailable:
            message = policy.error or policy.reason or "CUDA was requested, but no compatible CUDA runtime is available."
            self.footer_bar.set_status(message)
            self._set_activity("CUDA unavailable")
            errorBox("CUDA unavailable", message)
            return False
        self.current_run_origin = str(run_origin or "folder")
        self._active_cluster_directory = str(request.directory or "")
        self._active_cluster_scope_signature = PathScope.from_paths(request.source_roots).signature if request.source_roots else ""
        self.current_tag_filter = tuple(request.tag_filter)
        self.current_tag_match = request.tag_match or "Any"
        self._cancel_cluster_tag_context_refresh()
        self._main_gallery_context_overrides = {}
        self.cluster_data = {}
        self.membership_by_image = {}
        self.metrics_by_backend = {}
        self.cluster_explanations = {}
        self.cluster_meanings = {}
        self.cluster_tag_summaries = {}
        self.image_tags_by_path = {}
        self.gallery_extra_context_by_path = {}
        self.last_run_metrics = {}
        self.cluster_pane.update_clusters(
            {},
            membership_by_image={},
            metrics_by_backend={},
            cluster_summaries={},
            cluster_explanations={},
            cluster_meanings={},
        )
        self.gallery_pane.set_membership_context({}, {})
        self.gallery_pane.update_gallery([])
        self.gallery_pane.inspector_context_provider = self._main_gallery_context_for_path
        self.footer_bar.set_metrics("")
        self.footer_bar.set_status("Preparing clustering run...")
        self._set_activity("Preparing clustering run...")
        return self.session_controller.start(request)

    def _run_request_active(self) -> bool:
        return (
            self.session_controller.is_running()
            or self._thread_is_running(self._preflight_thread)
            or self.model_download_controller.is_running()
        )

    def _background_runtime_work_active(self) -> bool:
        return any(
            (
                self._run_request_active(),
                self._thread_is_running(self._tag_context_thread),
                self._thread_is_running(self._storage_usage_thread),
                self._thread_is_running(self._storage_clear_thread),
                self.model_download_controller.is_running(),
            )
        )

    def _update_footer_storage_state(self) -> None:
        if not self._can_update_widget(getattr(self, "footer_bar", None)):
            return
        reason = (
            "Unavailable while clustering, tag refresh, or another background runtime task is active."
            if self._background_runtime_work_active()
            else ""
        )
        self.footer_bar.set_clear_storage_enabled(not self._background_runtime_work_active(), reason=reason)

    def _start_request_preflight(self, request: ProductionClusterRequest) -> None:
        self._cancel_request_preflight()
        self._preflight_generation += 1
        generation = self._preflight_generation
        self._set_running_state(True)
        self.footer_bar.set_status("Scanning active roots...")
        self._set_activity("Preparing clustering run...")
        self._update_footer_storage_state()
        job_id = self.job_manager.register_job(
            "Preparing clustering run", cancel_fn=self.cancel_clustering, origin="Clustering"
        )
        self._preflight_job_id = job_id
        request_copy = replace(request)

        def _run(progress, cancel_check):
            from app.services.discovery import ImageDiscoveryService
            from app.services.embedding_index import EmbeddingIndexService

            progress(-1, "Scanning active roots...")
            discovery_service = ImageDiscoveryService()
            if request_copy.source_roots:
                discovered = discovery_service.discover_roots_result(
                    list(request_copy.source_roots),
                    recursive=bool(request_copy.recursive),
                    progress_callback=progress,
                    cancel_check=cancel_check,
                )
            else:
                discovered = discovery_service.discover_result(
                    request_copy.directory,
                    recursive=bool(request_copy.recursive),
                    progress_callback=progress,
                    cancel_check=cancel_check,
                )
            raise_if_cancelled(cancel_check)
            discovered_count = int(discovered.image_count)
            if discovered_count == 0:
                raise ValueError("No images were discovered in the active roots.")
            progress(10, f"Active-root scan complete: {discovered_count} image(s) discovered.")

            if not request_copy.tag_filter:
                requested_paths = tuple(request_copy.source_paths or ())
                source_paths = list(discovered.paths)
                source_fingerprints = list(discovered.fingerprints)
                if requested_paths:
                    requested_set = set(requested_paths)
                    source_paths = [path for path in discovered.paths if path in requested_set]
                    source_fingerprints = [fingerprint for fingerprint in discovered.fingerprints if fingerprint[0] in requested_set]
                    if len(source_paths) < 2:
                        raise ValueError("The displayed gallery changed before organizing could start. Refresh the folder and try again.")
                snapshot_key = (
                    EmbeddingIndexService.build_snapshot_key_from_fingerprints(source_fingerprints)
                    if requested_paths
                    else discovered.snapshot_key
                )
                return PreparedRunPayload(
                    request=replace(
                        request_copy,
                        source_paths=source_paths,
                        source_fingerprints=source_fingerprints,
                        source_snapshot_key=snapshot_key,
                    ),
                    run_origin="gallery" if requested_paths else "folder",
                    discovered_count=discovered_count,
                    matched_count=len(source_paths),
                )

            def _tag_progress(value: int, status: str) -> None:
                mapped = 10 + int(max(0, min(100, value)) * 0.65)
                progress(mapped, status)

            tags_by_path = self.image_tag_service.load_tags_for_paths(
                list(discovered.paths),
                progress_callback=_tag_progress,
                cancel_check=cancel_check,
            )

            def _filter_progress(value: int, status: str) -> None:
                mapped = 75 + int(max(0, min(100, value)) * 0.25)
                progress(mapped, status)

            source_paths = self.image_tag_service.select_paths_by_tags(
                list(discovered.paths),
                request_copy.tag_filter,
                request_copy.tag_match,
                tags_by_path=tags_by_path,
                progress_callback=_filter_progress,
                cancel_check=cancel_check,
            )
            matched_count = len(source_paths)
            if matched_count < 2:
                raise ValueError("Not enough tagged images match the current tag filter.")
            progress(100, f"Tag filter ready: {matched_count} of {discovered_count} image(s) matched.")
            source_set = set(source_paths)
            source_fingerprints = [
                fingerprint for fingerprint in discovered.fingerprints if fingerprint[0] in source_set
            ]
            snapshot_key = EmbeddingIndexService.build_snapshot_key_from_fingerprints(source_fingerprints)
            return PreparedRunPayload(
                request=replace(
                    request_copy,
                    source_paths=source_paths,
                    source_fingerprints=source_fingerprints,
                    source_snapshot_key=snapshot_key,
                ),
                run_origin="tag_filter",
                discovered_count=discovered_count,
                matched_count=matched_count,
            )

        job = AsyncJob(_run)
        self._preflight_job = job
        thread_holder: dict[str, object | None] = {"thread": None}

        def _current_generation() -> bool:
            return generation == self._preflight_generation

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            if thread is not None:
                self._release_async_refs(job, thread)
            if self._preflight_thread is thread:
                self._preflight_thread = None
            if self._preflight_job is job:
                self._preflight_job = None
            self._update_footer_storage_state()

        def _done(payload: object) -> None:
            if not _current_generation() or not isinstance(payload, PreparedRunPayload):
                _cleanup()
                return
            preflight_job_id = self._preflight_job_id
            self._preflight_job_id = None
            if preflight_job_id is not None:
                self.job_manager.finish(preflight_job_id, status="finished")
            if not self._confirm_large_folder_if_needed(payload.request):
                self.footer_bar.set_status("Clustering cancelled before inference.")
                self._set_activity("Cancelled")
                self._set_running_state(False)
                _cleanup()
                return
            prepared_request = self._prepare_request_model_downloads(payload.request)
            if prepared_request is None:
                self._set_running_state(False)
                _cleanup()
                return
            if payload.run_origin == "tag_filter":
                self.footer_bar.set_status(
                    f"Tag filter ready: {payload.matched_count} of {payload.discovered_count} image(s) matched. Starting clustering..."
                )
            else:
                self.footer_bar.set_status(
                    f"Folder ready: {payload.discovered_count} image(s). Starting clustering..."
                )
            self._set_activity("Starting worker...")
            started = self._start_request_with_assets(prepared_request, run_origin=payload.run_origin)
            if not started:
                self.footer_bar.set_status("Clustering could not start because another run is active.")
                self._set_activity("Idle")
                self._set_running_state(False)
            _cleanup()

        def _failed(message: str) -> None:
            if not _current_generation():
                _cleanup()
                return
            preflight_job_id = self._preflight_job_id
            self._preflight_job_id = None
            if preflight_job_id is not None:
                self.job_manager.finish(preflight_job_id, status="failed", error=message)
            self.footer_bar.set_status(f"Clustering could not start: {message}")
            self._set_activity("Preflight failed")
            self._set_running_state(False)
            errorBox("Clustering could not start", message)
            _cleanup()

        def _cancelled() -> None:
            if not _current_generation():
                _cleanup()
                return
            preflight_job_id = self._preflight_job_id
            self._preflight_job_id = None
            if preflight_job_id is not None:
                self.job_manager.finish(preflight_job_id, status="cancelled")
            self.footer_bar.set_status("Clustering preparation cancelled.")
            self._set_activity("Cancelled")
            self._set_running_state(False)
            _cleanup()

        job.progress.connect(lambda value, status: self._on_preflight_progress(job_id, value, status))
        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._preflight_thread = thread
        self._update_footer_storage_state()

    def _cancel_request_preflight(self) -> None:
        job = self._preflight_job
        thread = self._preflight_thread
        self._retain_async_refs(job, thread)
        self._preflight_job = None
        self._preflight_thread = None
        if job is None:
            return
        try:
            job.cancel()
        except Exception:
            pass
        self._update_footer_storage_state()

    def _on_preflight_progress(self, job_id: int | None, value: int, status: str) -> None:
        self._set_activity(status)
        if job_id is not None:
            self.job_manager.update(job_id, progress=value, text=status)

    def _restore_active_roots(self) -> tuple[str, ...]:
        raw_value = self.settings_store.value("workspace/active_roots", None)
        roots: list[str] = []
        if isinstance(raw_value, (list, tuple)):
            roots = [str(value) for value in raw_value if str(value or "").strip()]
        elif str(raw_value or "").strip():
            try:
                payload = json.loads(str(raw_value))
                if isinstance(payload, list):
                    roots = [str(value) for value in payload if str(value or "").strip()]
            except (TypeError, ValueError):
                roots = [str(raw_value)]
        if not roots:
            legacy = str(self.settings_store.value("workspace/selected_folder", "") or "").strip()
            if legacy:
                roots = [legacy]
        return PathScope.from_paths(roots).roots

    def _update_scope_summary(self, scope: PathScope) -> None:
        if scope.is_empty:
            text = "No active roots"
            tooltip = "Choose one or more roots. No workspace will scan or root-scope search until then."
        elif len(scope.roots) == 1:
            text = f"1 active root · {Path(scope.primary_root).name or scope.primary_root}"
            tooltip = scope.primary_root
        else:
            text = f"{len(scope.roots)} active roots · includes subfolders"
            tooltip = "\n".join(scope.roots)
        self.scope_summary_label.setText(text)
        self.scope_summary_label.setToolTip(tooltip)
        self.current_folder_label.setText(text)
        self.current_folder_label.setToolTip(tooltip)
        self.footer_bar.set_selected_folder(text)

    def _on_directory_changed(self, directory: str) -> None:
        """Compatibility entry point for older callers that set one folder."""

        self.source_pane.set_active_roots([directory] if str(directory or "").strip() else ())

    def _on_active_scope_changed(self, value: object) -> None:
        scope = value if isinstance(value, PathScope) else PathScope.from_paths(value if isinstance(value, (list, tuple, set)) else ())
        self.settings_store.setValue("workspace/active_roots", json.dumps(list(scope.roots)))
        self.settings_store.setValue("workspace/selected_folder", scope.primary_root)
        changed_history = False
        for root in scope.roots:
            changed_history = self.recent_folder_history.record(root) or changed_history
        if changed_history:
            self._refresh_recent_folder_menus()
        self._update_scope_summary(scope)
        self._photo_set_route = None
        self.photo_gallery.set_photo_set_route()
        if self.faces_pane is not None:
            self.faces_pane.current_scope_roots_provider = lambda: self.source_pane.active_scope.roots
            if scope.primary_root and self.faces_pane.face_folder_path.text().strip() != scope.primary_root:
                self.faces_pane.face_folder_path.blockSignals(True)
                self.faces_pane.face_folder_path.setText(scope.primary_root)
                self.faces_pane.face_folder_path.blockSignals(False)
            self.faces_pane.active_scope_changed()
        if self.names_pane is not None:
            self.names_pane.active_scope_changed()
        if self.tags_pane is not None and self._active_workspace == "tags":
            self.tags_pane.refresh()
        if self.library_pane is not None:
            self.library_pane.active_scope_changed()
        self._load_gallery_scope(scope)
        self._apply_startup_workflow_gate()

    def _load_gallery_folder(self, directory: str) -> None:
        """Compatibility wrapper for focused legacy callers/tests."""

        self._load_gallery_scope(PathScope.from_paths([directory]))

    def _load_gallery_scope(self, scope: PathScope) -> None:
        """Discover the active-root union in a worker without blocking Qt."""
        self._gallery_discovery_generation += 1
        generation = self._gallery_discovery_generation
        previous = self._gallery_discovery_job
        if previous is not None:
            try:
                previous.cancel()
            except Exception:
                pass
        self._gallery_paths = []
        self._gallery_snapshot_key = ""
        if scope.is_empty:
            self.photo_gallery.set_empty_state("Choose active roots to show their photos.")
            return
        self.photo_gallery.set_loading_state(f"Finding photos in {len(scope.roots)} active root(s)…")

        def _run(progress, cancel_check):
            from app.services.discovery import ImageDiscoveryService

            return ImageDiscoveryService().discover_roots_result(
                list(scope.roots),
                recursive=bool(self.clustering_pane.recursive_checkbox.isChecked()),
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        self._gallery_discovery_job = job
        self.job_manager.bind_async_job(job, "Finding active-root photos", origin="Gallery")

        def _finished(result) -> None:
            if generation != self._gallery_discovery_generation or self._is_shutting_down:
                return
            self._gallery_paths = list(getattr(result, "paths", ()) or ())
            self._gallery_snapshot_key = str(getattr(result, "snapshot_key", "") or "")
            if not self._gallery_paths:
                self.photo_gallery.set_empty_state("No supported photos were found in the active roots.", can_organize=False)
                self._apply_startup_workflow_gate()
                return
            sections = []
            for root in scope.roots:
                paths = tuple(path for path in self._gallery_paths if PathScope.from_paths([root]).contains(path))
                if paths:
                    sections.append(GallerySection(f"root:{root}", paths, kind="all", title=Path(root).name or root))
            self.photo_gallery.set_sections(
                sections or [GallerySection("all", tuple(self._gallery_paths), kind="all", title="All photos")],
                status=f"Showing {len(self._gallery_paths)} photos across {len(scope.roots)} active root(s). Organize when you are ready.",
            )
            self._apply_startup_workflow_gate()

        def _failed(message: str) -> None:
            if generation == self._gallery_discovery_generation and not self._is_shutting_down:
                self.photo_gallery.set_empty_state(f"Could not read the active roots: {message}")

        job.completed.connect(_finished)
        job.failed.connect(_failed)
        thread = start_job_in_thread(job)
        self._gallery_discovery_thread = thread
        self._retain_async_refs(job, thread)
        thread.finished.connect(lambda: self._release_async_refs(job, thread), Qt.ConnectionType.QueuedConnection)

    @staticmethod
    def _unique_photo_paths(paths: list[str] | tuple[str, ...]) -> list[str]:
        return list(dict.fromkeys(str(path) for path in paths if str(path or "").strip()))

    def _open_photo_set_route(
        self,
        paths: list[str] | tuple[str, ...],
        *,
        source: str,
        title: str,
        return_workspace: str,
        context_by_path: dict[str, dict[str, object]] | None = None,
        append: bool = False,
    ) -> None:
        """Publish a bounded, transient Gallery route from another workspace."""

        route_paths = self._unique_photo_paths(paths)
        if append and self._photo_set_route is not None:
            route_paths = self._unique_photo_paths([*self._photo_set_route.paths, *route_paths])
        if not route_paths:
            return
        route_context = {
            str(path): dict(context)
            for path, context in dict(context_by_path or {}).items()
            if str(path or "").strip() and isinstance(context, dict)
        }
        if append and self._photo_set_route is not None:
            route_context = {**self._photo_set_route.context_by_path, **route_context}
        route = PhotoSetRoute(
            source=str(source or "Gallery"),
            title=str(title or "Photo set"),
            paths=tuple(route_paths),
            return_workspace=str(return_workspace or "gallery"),
            context_by_path=route_context,
        )
        self._photo_set_route = route
        self.photo_gallery.inspector_context_provider = self._photo_gallery_context_for_path
        self.photo_gallery.set_photo_set_route(
            title=f"{route.title} · {len(route.paths)} photos",
            source=route.return_workspace,
            can_return=route.return_workspace != "gallery",
            can_review_faces=True,
            can_analyze=True,
        )
        self.photo_gallery.set_sections(
            [GallerySection("route", route.paths, kind="photos", title=route.title)],
            status=f"Showing {len(route.paths)} photo(s) from {route.source}. Select photos, then use the visible route actions.",
        )
        self.set_active_workspace("gallery")
        self.footer_bar.set_status(f"Opened {len(route.paths)} photo(s) from {route.source} in Gallery.")
        self._set_activity(f"Gallery route: {route.source}")

    def _return_to_folder_photo_gallery(self) -> None:
        """Drop only session routing; the selected-folder gallery remains durable."""

        self._photo_set_route = None
        self.photo_gallery.set_photo_set_route()
        if not self._gallery_paths:
            self.photo_gallery.set_empty_state("Choose a folder to show its photos.")
        else:
            self.photo_gallery.set_sections(
                [GallerySection("all", tuple(self._gallery_paths), kind="all", title="All photos")],
                status=f"Showing {len(self._gallery_paths)} folder photo(s). Organize when you are ready.",
            )
        self.footer_bar.set_status("Returned to selected-folder photos.")

    def _return_to_photo_set_source(self, workspace_id: str) -> None:
        target = self._normalize_workspace_id(workspace_id)
        if target == "gallery":
            return
        self.set_active_workspace(target)
        self.footer_bar.set_status("Returned to the workspace that opened this photo set.")

    def _analyze_photo_set_in_clustering(self, paths: list[str]) -> None:
        selected_paths = self._unique_photo_paths(paths)
        if len(selected_paths) < 2:
            errorBox("More photos needed", "Select at least two photos before analyzing this photo set.")
            return
        if not self._clustering_ready_for_action() or self._run_request_active():
            return
        request = self._build_request()
        request = replace(
            request,
            source_paths=selected_paths,
            tag_filter=[],
            generate_cluster_meanings=False,
            generate_cluster_explanations=False,
        )
        self.set_active_workspace("clustering")
        self.footer_bar.set_status(f"Preparing {len(selected_paths)} Gallery photo(s) for clustering…")
        self._start_request_preflight(request)

    def _review_photo_set_in_faces(self, paths: list[str]) -> None:
        selected_paths = tuple(self._unique_photo_paths(paths))
        if not selected_paths:
            return
        # This is only a review hand-off.  SearchPane displays the explicit
        # scope and waits for its Detect Faces action before indexing anything.
        self._pending_face_review_paths = selected_paths
        self.set_active_workspace("faces")
        if self.faces_pane is not None:
            self._apply_pending_photo_face_review(self.faces_pane)
        self.footer_bar.set_status(f"{len(selected_paths)} photo(s) are ready in Faces. Choose Detect Faces to start indexing.")

    def _apply_pending_photo_face_review(self, pane) -> None:
        paths = tuple(self._pending_face_review_paths)
        if not paths or pane is None:
            return
        setter = getattr(pane, "set_explicit_review_scope_paths", None)
        if not callable(setter):
            return
        setter(list(paths))
        self._pending_face_review_paths = ()

    def run_gallery_organize(self) -> None:
        if not self._clustering_ready_for_action():
            return
        if self._run_request_active():
            errorBox("Busy", "Wait for the active clustering run to finish before organizing this gallery.")
            return
        if len(self._gallery_paths) < 2:
            self.photo_gallery.set_empty_state("Choose a folder with at least two photos before organizing.")
            return
        request = self._build_request()
        models = list(request.embedding_models or [])
        backends = list(request.clustering_backends or [])
        modes = list(request.similarity_modes or [request.similarity_mode])
        request = replace(
            request,
            embedding_models=models[:1],
            clustering_backends=backends[:1],
            similarity_modes=modes[:1],
            similarity_mode=modes[0] if modes else request.similarity_mode,
            tag_filter=[],
            source_paths=list(self._gallery_paths),
            generate_cluster_meanings=False,
            generate_cluster_explanations=False,
        )
        self._gallery_primary_comparison_key = "::".join(
            (request.embedding_models[0], request.similarity_mode, request.clustering_backends[0])
        )
        self.footer_bar.set_status("Organizing photos with the current primary clustering settings...")
        self._start_request_preflight(request)

    def _gallery_sections_from_clusters(self) -> list[GallerySection]:
        if not self.cluster_data:
            return [GallerySection("all", tuple(self._gallery_paths), kind="all", title="All photos")]
        comparison_key = self._gallery_primary_comparison_key
        if comparison_key not in self.cluster_data:
            comparison_key = next(iter(self.cluster_data))
        clusters = self.cluster_data.get(comparison_key, {})
        gallery_paths = set(self._gallery_paths)
        ordered = sorted(
            ((int(cluster_id), [path for path in paths if path in gallery_paths]) for cluster_id, paths in clusters.items()),
            key=lambda item: (item[0] == -1, -len(item[1]), item[0]),
        )
        sections: list[GallerySection] = []
        used: set[str] = set()
        for cluster_id, paths in ordered:
            if not paths:
                continue
            used.update(paths)
            sections.append(
                GallerySection(
                    section_id=f"cluster:{comparison_key}:{cluster_id}",
                    paths=tuple(paths),
                    kind="other" if cluster_id == -1 else "photos",
                    title="Other photos" if cluster_id == -1 else "Photo group",
                )
            )
        remaining = [path for path in self._gallery_paths if path not in used]
        if remaining:
            sections.append(GallerySection("other:unassigned", tuple(remaining), kind="other", title="Other photos"))
        return sections or [GallerySection("all", tuple(self._gallery_paths), kind="all", title="All photos")]

    def _on_model_download_started(self, item_keys: tuple) -> None:
        label = "Downloading model assets"
        job_id = self.job_manager.register_job(
            label, cancel_fn=self.model_download_controller.cancel, origin="Models"
        )
        self._model_download_job_id = job_id
        item_text = ", ".join(str(item).split(":text=", 1)[0] for item in item_keys)
        status = f"Downloading model assets: {item_text}" if item_text else "Downloading model assets"
        self.job_manager.update(job_id, progress=-1, text="Cache lookup and download starting")
        self.footer_bar.set_status(status)
        self._set_activity("Downloading model assets...")
        self._update_footer_storage_state()

    def _on_model_download_progress(self, value: int, status: str) -> None:
        self.footer_bar.set_status(status)
        self._set_activity(status)
        if self._model_download_job_id is not None:
            self.job_manager.update(self._model_download_job_id, progress=value, text=status)

    def _finish_model_download_job(self, status: str, *, error: str = "", cache_text: str = "") -> None:
        job_id = self._model_download_job_id
        self._model_download_job_id = None
        if job_id is None:
            return
        if cache_text:
            self.job_manager.update(
                job_id,
                progress=100 if status == "finished" else -1,
                text=cache_text,
                cache_status=cache_text,
            )
        self.job_manager.finish(job_id, status=status, error=error, cache_status=cache_text)

    def _on_model_download_completed(self, result: dict) -> None:
        requested = tuple(str(item) for item in result.get("requested") or ())
        downloaded = tuple(str(item) for item in result.get("downloaded") or ())
        reused = tuple(str(item) for item in result.get("reused") or ())
        cache_parts = []
        if downloaded:
            cache_parts.append(f"downloaded {len(downloaded)}")
        if reused:
            cache_parts.append(f"reused {len(reused)} cached")
        cache_text = "Cache ready" + (f" ({', '.join(cache_parts)})" if cache_parts else "")
        self._finish_model_download_job("finished", cache_text=cache_text)
        self.footer_bar.set_status(f"{cache_text}.")
        self._set_activity("Model cache ready")

        pending_run = self._pending_model_download_run
        self._pending_model_download_run = None
        post_install_name = self._active_post_install_model_name
        self._active_post_install_model_name = None
        self._refresh_footer_storage_usage()
        if self._is_shutting_down:
            return
        if any(item.split(":text=", 1)[0] == "facenet" for item in requested):
            self._reload_face_services_from_settings()
        self._begin_startup_readiness_check()
        if pending_run is not None:
            request, run_origin = pending_run
            verification = self.model_asset_service.build_download_plan(
                request.embedding_models,
                generate_cluster_meanings=bool(request.generate_cluster_meanings),
                requested_meaning_model=request.cluster_meaning_model,
            )
            if verification.requires_download:
                missing = ", ".join(verification.unavailable_items)
                message = f"Model download finished, but the local readiness check still reports missing assets: {missing}."
                self._set_running_state(False)
                self.footer_bar.set_status(message)
                self._set_activity("Model verification failed")
                errorBox("Model verification failed", message)
                return
            self.footer_bar.set_status("Model cache verified. Starting clustering...")
            self._set_activity("Starting clustering worker...")
            if not self._start_request(replace(request, allow_model_downloads=False), run_origin=run_origin):
                self._set_running_state(False)
                errorBox("Clustering could not start", "The model cache is ready, but the clustering worker could not start.")
            return
        if post_install_name:
            infoBox("Model downloaded", f"{model_label(post_install_name)} is ready in the shared runtime cache.")

    def _on_model_download_failed(self, message: str) -> None:
        pending_run = self._pending_model_download_run
        post_install_name = self._active_post_install_model_name
        self._pending_model_download_run = None
        self._active_post_install_model_name = None
        self._finish_model_download_job("failed", error=message)
        self.footer_bar.set_status(f"Model download failed: {message}")
        self._set_activity("Model download failed")
        if pending_run is not None:
            self._set_running_state(False)
        self._refresh_footer_storage_usage()
        if not self._is_shutting_down and (pending_run is not None or post_install_name):
            errorBox("Model download failed", message)

    def _on_model_download_cancelled(self) -> None:
        pending_run = self._pending_model_download_run
        self._pending_model_download_run = None
        self._active_post_install_model_name = None
        self._finish_model_download_job("cancelled", cache_text="Cancelled; completed and partial cache files are retained")
        self.footer_bar.set_status("Model download cancelled. Cached files will be reused on retry.")
        self._set_activity("Model download cancelled")
        if pending_run is not None:
            self._set_running_state(False)
        self._refresh_footer_storage_usage()

    def _on_clustering_started(self) -> None:
        job_id = self.job_manager.register_job(
            "Production clustering", cancel_fn=self.session_controller.cancel, origin="Clustering"
        )
        self._active_job_id = job_id
        self.footer_bar.set_status("Clustering started.")
        self._set_activity("Clustering started...")
        self._update_footer_storage_state()

    def _on_clustering_progress(self, value: int, status: str) -> None:
        self._set_activity(status)
        job_id = getattr(self, "_active_job_id", None)
        if job_id is not None:
            self.job_manager.update(job_id, progress=value, text=status)

    def _on_clustering_completed(self, payload: dict) -> None:
        self.footer_bar.set_status("Clustering finished.")
        job_id = getattr(self, "_active_job_id", None)
        self.cluster_data = _restore_cluster_dict(payload.get("clusters_by_key") or {})
        self.membership_by_image = _restore_membership(payload.get("membership_by_image") or {})
        self.metrics_by_backend = {str(key): dict(value) for key, value in dict(payload.get("metrics_by_key") or {}).items()}
        fallback_reasons = list(
            dict.fromkeys(
                str(metrics.get("compute_fallback_reason") or "").strip()
                for metrics in self.metrics_by_backend.values()
                if str(metrics.get("compute_fallback_reason") or "").strip()
            )
        )
        fallback_notice = f" CPU fallback: {'; '.join(fallback_reasons)}" if fallback_reasons else ""
        if job_id is not None:
            self.job_manager.finish(
                job_id,
                status="finished",
                cache_status=f"CPU fallback: {'; '.join(fallback_reasons)}" if fallback_reasons else "",
            )
            self._active_job_id = None
        self.cluster_explanations = _restore_cluster_explanations(payload.get("cluster_explanations_by_key") or {})
        self.cluster_meanings = _restore_cluster_meanings(payload.get("cluster_meanings_by_key") or {})
        self.last_run_metrics = dict(payload.get("metrics") or {})
        self.last_run_metrics["run_origin"] = self.current_run_origin
        self.last_run_metrics["tag_match"] = self.current_tag_match
        if fallback_reasons:
            self.last_run_metrics["compute_fallback_reason"] = "; ".join(fallback_reasons)
        if self.current_tag_filter:
            self.last_run_metrics["tag_filter"] = ", ".join(self.current_tag_filter)
        self.cluster_tag_summaries = _restore_cluster_summaries(payload.get("cluster_summaries") or {})
        self.cluster_pane.update_clusters(
            self.cluster_data,
            membership_by_image=self.membership_by_image,
            metrics_by_backend=self.metrics_by_backend,
            cluster_summaries=self.cluster_tag_summaries,
            cluster_explanations=self.cluster_explanations,
            cluster_meanings=self.cluster_meanings,
        )
        self.gallery_pane.set_membership_context(self.membership_by_image, self.metrics_by_backend)
        if self._gallery_paths and self._active_cluster_scope_signature == self.source_pane.active_scope.signature:
            self.photo_gallery.set_sections(
                self._gallery_sections_from_clusters(),
                status="Photos organized by the current primary clustering result.",
            )
        self.gallery_pane.inspector_context_provider = self._main_gallery_context_for_path
        self.clustering_pane.update_metrics(self.last_run_metrics)
        self.footer_bar.set_metrics(self.clustering_pane._last_metrics_text)
        default_selection = self.cluster_pane.select_default_cluster()
        if default_selection is not None:
            comparison_key, cluster_id = default_selection
            warm_note = " Worker kept warm for the next run." if self.session_controller.is_worker_warm() else ""
            self.footer_bar.set_status(f"Clustering finished. Showing {comparison_key} / cluster {cluster_id}.{fallback_notice}{warm_note}")
            self.update_gallery(comparison_key, cluster_id)
        else:
            warm_note = " Worker kept warm for the next run." if self.session_controller.is_worker_warm() else ""
            self.footer_bar.set_status(f"Clustering finished with no clusters to preview.{fallback_notice}{warm_note}")
        self._set_activity("Refreshing tag summaries...")
        self._start_cluster_tag_context_refresh()
        if self.library_pane is not None:
            self.library_pane.describe_completed_clusters(self.cluster_data)
        self._refresh_footer_storage_usage()

    def _on_clustering_failed(self, message: str) -> None:
        self.footer_bar.set_status(f"Clustering failed: {message}")
        self._set_activity("Failed")
        job_id = getattr(self, "_active_job_id", None)
        if job_id is not None:
            self.job_manager.finish(job_id, status="failed", error=message)
            self._active_job_id = None
        self._refresh_footer_storage_usage()
        errorBox("Clustering failed", message)

    def _on_clustering_cancelled(self) -> None:
        self.footer_bar.set_status("Clustering cancelled.")
        job_id = getattr(self, "_active_job_id", None)
        if job_id is not None:
            self.job_manager.finish(job_id, status="cancelled")
            self._active_job_id = None
        self.last_run_metrics = {
            "status": "cancelled",
            "run_origin": self.current_run_origin,
            "tag_match": self.current_tag_match,
        }
        self._set_activity("Cancelled")
        self._refresh_footer_storage_usage()

    def _set_running_state(self, running: bool) -> None:
        self.clustering_pane.set_running(running)
        self.source_pane.set_running(running)
        if hasattr(self, "photo_gallery"):
            report = self._startup_readiness_report
            self.photo_gallery.organize_button.setEnabled(
                (not running)
                and len(self._gallery_paths) >= 2
                and bool(report is not None and report.clustering_ready)
            )
        if not running:
            self._apply_startup_workflow_gate()
        self._update_footer_storage_state()

    def update_gallery(self, comparison_key: str, cluster_id: int) -> None:
        images = list(self.cluster_data.get(comparison_key, {}).get(int(cluster_id), []))
        self._set_activity(f"Showing {comparison_key} / cluster {cluster_id}")
        self.gallery_pane.update_gallery(images)

    def _on_image_selected(self, image_path: str) -> None:
        self.cluster_pane.update_membership(image_path, self.membership_by_image, self.metrics_by_backend)
        self.cluster_pane.highlight_membership(image_path)

    def _on_gallery_first_paint(self, latency_ms: int) -> None:
        self.last_run_metrics["gallery_first_paint_ms"] = int(latency_ms)
        self.footer_bar.set_metrics(self.clustering_pane.update_metrics(self.last_run_metrics))

    def _on_gallery_paths_removed(self, paths: list[str]) -> None:
        removed = {str(path) for path in paths if path}
        if not removed:
            return
        self._gallery_paths = [path for path in self._gallery_paths if path not in removed]
        route = self._photo_set_route
        if route is not None:
            remaining_route_paths = tuple(path for path in route.paths if path not in removed)
            remaining_context = {
                path: context
                for path, context in route.context_by_path.items()
                if path not in removed
            }
            self._photo_set_route = replace(
                route,
                paths=remaining_route_paths,
                context_by_path=remaining_context,
            ) if remaining_route_paths else None
            if not remaining_route_paths:
                self.photo_gallery.set_photo_set_route()
        self._main_gallery_context_overrides = {
            image_path: context
            for image_path, context in self._main_gallery_context_overrides.items()
            if image_path not in removed
        }
        self.footer_bar.set_status(f"Refreshing cluster context after removing {len(removed)} image(s)...")
        self._set_activity("Refreshing after file operation...")
        self.cluster_data = {
            comparison_key: {
                cluster_id: [path for path in image_paths if path not in removed]
                for cluster_id, image_paths in clusters.items()
                if [path for path in image_paths if path not in removed]
            }
            for comparison_key, clusters in self.cluster_data.items()
        }
        self.membership_by_image = {
            image_path: payload
            for image_path, payload in self.membership_by_image.items()
            if image_path not in removed
        }
        self.cluster_explanations = {}
        self.cluster_meanings = {}
        self.cluster_pane.update_clusters(
            self.cluster_data,
            membership_by_image=self.membership_by_image,
            metrics_by_backend=self.metrics_by_backend,
            cluster_summaries=self.cluster_tag_summaries,
            cluster_explanations=self.cluster_explanations,
            cluster_meanings=self.cluster_meanings,
            preserve_selection=True,
        )
        self._start_cluster_tag_context_refresh()
        current_target = self.cluster_pane.current_selection_target()
        if current_target is not None:
            self.gallery_pane.update_gallery_with_options(
                images=current_target.as_list(),
                clear_pixmaps=False,
                reset_scroll=False,
            )

    def _on_gallery_paths_renamed(self, changed_paths: list[tuple[str, str]]) -> None:
        """Keep every in-memory workspace reference coherent after a journaled rename."""
        replacements = {
            str(source): str(target)
            for source, target in changed_paths
            if str(source).strip() and str(target).strip() and str(source) != str(target)
        }
        if not replacements:
            return
        self._gallery_paths = [replacements.get(path, path) for path in self._gallery_paths]
        route = self._photo_set_route
        if route is not None:
            self._photo_set_route = replace(
                route,
                paths=tuple(replacements.get(path, path) for path in route.paths),
                context_by_path={
                    replacements.get(path, path): context
                    for path, context in route.context_by_path.items()
                },
            )
        self._main_gallery_context_overrides = {
            replacements.get(path, path): context
            for path, context in self._main_gallery_context_overrides.items()
        }
        self.cluster_data = {
            comparison_key: {
                cluster_id: [replacements.get(path, path) for path in image_paths]
                for cluster_id, image_paths in clusters.items()
            }
            for comparison_key, clusters in self.cluster_data.items()
        }
        self.membership_by_image = {
            replacements.get(path, path): payload
            for path, payload in self.membership_by_image.items()
        }
        self.image_tags_by_path = {
            replacements.get(path, path): payload
            for path, payload in self.image_tags_by_path.items()
        }
        self.cluster_explanations = {}
        self.cluster_meanings = {}
        self.cluster_pane.update_clusters(
            self.cluster_data,
            membership_by_image=self.membership_by_image,
            metrics_by_backend=self.metrics_by_backend,
            cluster_summaries=self.cluster_tag_summaries,
            cluster_explanations=self.cluster_explanations,
            cluster_meanings=self.cluster_meanings,
            preserve_selection=True,
        )
        current_target = self.cluster_pane.current_selection_target()
        if current_target is not None:
            self.gallery_pane.update_gallery_with_options(
                images=current_target.as_list(),
                clear_pixmaps=True,
                reset_scroll=False,
            )
        self.footer_bar.set_status(f"Renamed {len(replacements)} photo(s); in-memory workspace references were updated.")
        self._set_activity("Refreshing after batch rename...")
        self._start_cluster_tag_context_refresh()

    def _on_gallery_metadata_changed(self, paths: list[str]) -> None:
        changed = {str(path) for path in paths if path}
        route = self._photo_set_route
        if route is not None and changed.intersection(route.paths):
            self.photo_gallery.status_label.setText(
                f"Metadata changed for {len(changed.intersection(route.paths))} routed photo(s). Source views are refreshing."
            )
        self.footer_bar.set_status("Refreshing cluster context after metadata changes...")
        self._set_activity("Refreshing after metadata change...")
        self.cluster_pane.update_clusters(
            self.cluster_data,
            membership_by_image=self.membership_by_image,
            metrics_by_backend=self.metrics_by_backend,
            cluster_summaries=self.cluster_tag_summaries,
            cluster_explanations=self.cluster_explanations,
            cluster_meanings=self.cluster_meanings,
            preserve_selection=True,
        )
        self._start_cluster_tag_context_refresh()

    def set_clustering_mode(self, mode: str) -> None:
        normalized = "advanced" if str(mode).strip().lower() == "advanced" else "basic"
        previous_mode = self._clustering_mode
        if previous_mode == "advanced" and normalized != previous_mode:
            self._advanced_pane_visibility = dict(self._pane_visibility)
            self._advanced_cluster_splitter_sizes = [int(value) for value in self.cluster_right_splitter.sizes()]
        self._clustering_mode = normalized
        if normalized == "advanced":
            self._pane_visibility = dict(self._advanced_pane_visibility)
            if self._advanced_cluster_splitter_sizes:
                self.cluster_right_splitter.setSizes(self._advanced_cluster_splitter_sizes)
        else:
            self._pane_visibility = {"source": True, "controls": False, "details": True}
            self.cluster_right_splitter.setSizes([1])
        self.cluster_pane.set_basic_mode(normalized == "basic")
        self.gallery_pane.set_inspector_display_mode(normalized)
        self._apply_gallery_action_visibility()
        self._apply_safety_state()
        self.source_pane.set_basic_mode(normalized == "basic", running=self.session_controller.is_running())
        if self._active_workspace == "clustering":
            self._refresh_workspace_ui()
            self._set_activity(f"{'Advanced' if normalized == 'advanced' else 'Basic'} mode ready")
        else:
            self._sync_mode_buttons()
            self._sync_pane_toggle_buttons()

    def _remember_visible_pane_widths(self) -> None:
        widths = self._layout_widths()
        main_sizes = self.main_splitter.sizes()
        if len(main_sizes) >= 2 and self.source_pane.isVisible() and int(main_sizes[0]) > 0:
            self._pane_restore_widths["source"] = max(widths["source"], int(main_sizes[0]))
        cluster_sizes = self.clustering_splitter.sizes()
        if len(cluster_sizes) != 3:
            return
        if self.clustering_pane.isVisible() and int(cluster_sizes[0]) > 0:
            self._pane_restore_widths["controls"] = max(widths["controls"], int(cluster_sizes[0]))
        if self.cluster_right_splitter.isVisible() and int(cluster_sizes[2]) > 0:
            self._pane_restore_widths["details"] = max(widths["details"], int(cluster_sizes[2]))

    def _sync_pane_toggle_buttons(self) -> None:
        mapping = {
            "source": (self.source_toggle, self._source_pane_visible()),
            "controls": (self.controls_toggle, bool(self._advanced_pane_visibility["controls"])),
            "details": (self.details_toggle, bool(self._advanced_pane_visibility["details"])),
        }
        for _key, (button, checked) in mapping.items():
            button.blockSignals(True)
            button.setChecked(bool(checked))
            button.blockSignals(False)
        action_mapping = {
            "source": getattr(self, "source_view_action", None),
            "controls": getattr(self, "controls_view_action", None),
            "details": getattr(self, "details_view_action", None),
        }
        for key, action in action_mapping.items():
            if action is None:
                continue
            checked = mapping[key][1]
            action.blockSignals(True)
            action.setChecked(bool(checked))
            action.blockSignals(False)

    def _set_pane_visible(self, pane_key: str, visible: bool) -> None:
        visible = bool(visible)
        if pane_key == "source":
            if self._active_workspace == "clustering" and self._clustering_mode != "advanced":
                return
            if self._advanced_pane_visibility.get("source") == visible and self._pane_visibility.get("source") == visible:
                self._sync_pane_toggle_buttons()
                return
            self._advanced_pane_visibility["source"] = visible
            self._pane_visibility["source"] = visible
            self._sync_pane_toggle_buttons()
            self._reflow_main_splitter(force=True)
            return
        if self._active_workspace != "clustering" or self._clustering_mode != "advanced":
            return
        if self._pane_visibility.get(pane_key) == visible:
            self._sync_pane_toggle_buttons()
            return
        self._pane_visibility[pane_key] = visible
        self._advanced_pane_visibility[pane_key] = visible
        self._sync_pane_toggle_buttons()
        self._reflow_main_splitter(force=True)

    def _reflow_main_splitter(self, *, force: bool = False) -> None:
        _ = force
        self._remember_visible_pane_widths()
        widths = self._layout_widths()
        total_width = max(widths["window"], int(self.width() or self._screen_available_width or widths["window"]))
        source_visible = self._source_pane_visible()
        source = min(widths["source_max"], max(widths["source"], int(self._pane_restore_widths["source"]))) if source_visible else 0
        workspace = max(widths["workspace"], total_width - source - 24)
        if source_visible and source + workspace + 24 > total_width:
            source = max(widths["source"], total_width - workspace - 24)
        self.source_pane.setVisible(source_visible)
        self.main_splitter.setSizes([source, workspace])
        if self._clustering_mode == "basic":
            controls = 0
            details = min(widths["details_max"], max(widths["details"], int(self._pane_restore_widths["details"])))
            gallery = max(widths["gallery"], workspace - details - 18)
            if gallery + details > workspace:
                details = max(widths["details"], workspace - widths["gallery"])
                gallery = max(widths["gallery"], workspace - details)
            self.clustering_pane.setVisible(False)
            self.cluster_right_splitter.setVisible(True)
            self.clustering_splitter.setSizes([controls, gallery, details])
            return

        controls = min(widths["controls_max"], max(widths["controls"], int(self._pane_restore_widths["controls"]))) if self._pane_visibility["controls"] else 0
        details = min(widths["details_max"], max(widths["details"], int(self._pane_restore_widths["details"]))) if self._pane_visibility["details"] else 0
        gallery = max(widths["gallery"], workspace - controls - details - 18)
        visible_side_total = controls + details
        if gallery + visible_side_total > workspace:
            remaining_for_sides = max(0, workspace - widths["gallery"])
            if controls and details:
                controls = max(widths["controls"], min(controls, int(remaining_for_sides * 0.55)))
                details = max(widths["details"], remaining_for_sides - controls)
            elif controls:
                controls = max(widths["controls"], remaining_for_sides)
            elif details:
                details = max(widths["details"], remaining_for_sides)
            gallery = max(widths["gallery"], workspace - controls - details)
        self.clustering_pane.setVisible(self._pane_visibility["controls"])
        self.cluster_right_splitter.setVisible(self._pane_visibility["details"])
        self.clustering_splitter.setSizes([controls, gallery, details])

    def apply_screen_geometry_constraints(self, available_size: tuple[int, int]) -> None:
        width, height = available_size
        self._screen_available_width = max(1, int(width))
        self._screen_available_height = max(1, int(height))
        widths = self._layout_widths()
        min_height = min(900, max(720, self._screen_available_height))
        self.setMinimumSize(widths["window"], min_height)
        self.source_pane.setMinimumWidth(widths["source"])
        self.source_pane.setMaximumWidth(widths["source_max"])
        self.clustering_pane.setMinimumWidth(widths["controls"])
        self.clustering_pane.setMaximumWidth(widths["controls_max"])
        self.gallery_pane.setMinimumWidth(widths["gallery"])
        self.cluster_pane.setMinimumWidth(widths["details"])
        self.cluster_pane.setMaximumWidth(widths["details_max"])
        self.workspace_stack.setMinimumWidth(widths["workspace"])
        self.current_folder_label.setMinimumWidth(widths["activity"])
        if self.faces_pane is not None:
            self.faces_pane.setMinimumWidth(widths["faces"])
        if self.names_pane is not None:
            self.names_pane.setMinimumWidth(widths["faces"])
        if self.library_pane is not None:
            self.library_pane.setMinimumWidth(widths["faces"])
        self._reflow_main_splitter(force=True)

    def _layout_widths(self) -> dict[str, int]:
        compact = int(self._screen_available_width or MIN_SUPPORTED_SCREEN_WIDTH) < COMPACT_LAYOUT_WIDTH
        if compact:
            return {
                "window": min(COMPACT_LAYOUT_WIDTH, max(800, int(self._screen_available_width or 1080))),
                "source": 168,
                "source_max": 220,
                "workspace": 720,
                "controls": 240,
                "controls_max": 320,
                "gallery": 340,
                "details": 200,
                "details_max": 300,
                "faces": 720,
                "activity": 120,
                "health": 100,
            }
        return {
            "window": 1440,
            "source": 220,
            "source_max": 280,
            "workspace": 1040,
            "controls": 380,
            "controls_max": 460,
            "gallery": 760,
            "details": 440,
            "details_max": 680,
            "faces": 1040,
            "activity": 280,
            "health": 130,
        }

    def _apply_gallery_action_visibility(self) -> None:
        if self._clustering_mode == "advanced":
            self.gallery_pane.set_action_visibility(
                show_actions=True,
                show_metadata_actions=True,
                show_file_actions=True,
            )
            return
        self.gallery_pane.set_action_visibility(
            show_actions=True,
            show_metadata_actions=False,
            show_file_actions=True,
        )

    def _apply_safety_state(self) -> None:
        read_only = self._read_only_mode()
        self.gallery_pane.set_read_only_mode(read_only)
        if hasattr(self, "photo_gallery"):
            self.photo_gallery.set_read_only_mode(read_only)
        if self.tags_pane is not None:
            self.tags_pane.set_read_only_mode(read_only)
        if self.faces_pane is not None:
            self.faces_pane.set_read_only_mode(read_only)
        if self.names_pane is not None:
            self.names_pane.set_read_only_mode(read_only)
        if self.library_pane is not None:
            self.library_pane.set_read_only_mode(read_only)
        self._refresh_health_badge()

    def _apply_runtime_status(self) -> None:
        self.performance_profile = self._effective_performance_profile()
        capabilities = self.runtime_service.detect()
        self.execution_policy = self.runtime_service.select_policy(self._preferred_execution_mode())
        self.runtime_badge.update_runtime(capabilities, self.execution_policy)
        if self.faces_pane is not None:
            self.faces_pane.reset_similarity_execution_services()
        self._refresh_health_badge()

    def open_tag_manager(self) -> None:
        self.set_active_workspace("tags")
        if self.tags_pane is not None:
            self.tags_pane.refresh()

    def _on_cluster_selection_target_changed(self, _target: SelectionTarget | None) -> None:
        self.gallery_pane.refresh_selection_target_hint()
        self._refresh_tags_suggestion_state()
        if self.library_pane is not None:
            target = _target
            if target is None:
                self.library_pane.set_selected_cluster("", ())
            else:
                comparison_key = str(target.source_context.get("comparison_key") or target.source_context.get("backend") or "cluster")
                try:
                    cluster_id = int(target.source_context.get("cluster_id", -1) or -1)
                except (TypeError, ValueError):
                    cluster_id = -1
                self.library_pane.set_selected_cluster(f"{comparison_key}:{cluster_id}", target.as_list())

    @staticmethod
    def _cluster_selection_identity(target: SelectionTarget | None) -> tuple[str, int, tuple[str, ...]] | None:
        if target is None:
            return None
        comparison_key = str(target.source_context.get("comparison_key") or target.source_context.get("backend") or "")
        try:
            cluster_id = int(target.source_context.get("cluster_id", -1))
        except (TypeError, ValueError):
            cluster_id = -1
        return comparison_key, cluster_id, tuple(sorted(str(path) for path in target.as_list()))

    def _refresh_tags_suggestion_state(self) -> None:
        if self.tags_pane is None:
            return
        tags, target = self._selected_cluster_meaning_tags()
        running = self._tag_suggestion_job is not None
        if target is None:
            self.tags_pane.set_cluster_suggestion_state(
                "Cluster suggestions: select a cluster in Clustering.", can_generate=False, can_apply=False
            )
            return
        if running:
            self.tags_pane.set_cluster_suggestion_state(
                f"Generating suggestions for {target.label}…", can_generate=False, can_apply=False
            )
            return
        if tags:
            self.tags_pane.set_cluster_suggestion_state(
                f"Suggestions for {target.label}: {', '.join(tags)}", can_generate=True, can_apply=True
            )
            return
        self.tags_pane.set_cluster_suggestion_state(
            f"No suggestions for {target.label}. Generate them on demand.", can_generate=True, can_apply=False
        )

    def _run_tag_filter_from_tags_workspace(self, tags: list[str], match: str) -> None:
        if self.tags_pane is not None:
            self.tags_pane.set_filter(tags, match)
        self.set_active_workspace("clustering")
        self.run_clustering()

    def generate_cluster_tag_suggestions(self) -> None:
        if self._thread_is_running(self._tag_suggestion_thread):
            self.footer_bar.set_status("Cluster tag suggestions are already being generated.")
            return
        target = self.cluster_pane.current_selection_target()
        selection_identity = self._cluster_selection_identity(target)
        if target is None or selection_identity is None:
            errorBox("Select a cluster", "Select a cluster before generating tag suggestions.")
            return
        comparison_key, cluster_id, source_paths = selection_identity
        if not source_paths:
            errorBox("Empty cluster", "The selected cluster does not contain any photos to inspect.")
            return
        self._tag_suggestion_generation += 1
        generation = self._tag_suggestion_generation
        selected_models = list(self.clustering_pane.selected_embedding_models())
        use_onnx = bool(self.clustering_pane.onnx_checkbox.isChecked())
        use_embedding_cache_lookup = bool(self.clustering_pane.embedding_cache_lookup_checkbox.isChecked())
        profile = self.performance_profile
        preferred_mode = self._preferred_execution_mode()
        snapshot_key = self._gallery_snapshot_key or f"selection:{comparison_key}:{cluster_id}:{len(source_paths)}"

        def _run(progress, cancel_check):
            from app.services.cluster_meanings import ClusterMeaningService
            from ml.embeddings import EmbeddingService, ModelManager

            runtime_service = RuntimeCapabilityService()
            execution_policy = runtime_service.select_policy(preferred_mode)
            if execution_policy.cuda_required_unavailable:
                raise RuntimeError(execution_policy.error or execution_policy.reason)
            progress(2, "Preparing the selected cluster for tag suggestions…")
            fingerprints: dict[str, tuple[int, int]] = {}
            for index, image_path in enumerate(source_paths, start=1):
                raise_if_cancelled(cancel_check)
                try:
                    stat = Path(image_path).stat()
                    fingerprints[image_path] = (int(stat.st_mtime_ns), int(stat.st_size))
                except OSError:
                    continue
                if index == len(source_paths) or index % 20 == 0:
                    progress(min(12, int(index * 12 / len(source_paths))), "Checking selected photo versions…")
            model_manager = ModelManager(
                use_onnx=use_onnx,
                execution_policy=execution_policy,
                runtime_service=runtime_service,
                performance_profile=profile,
                allow_model_downloads=False,
            )
            embedding_service = EmbeddingService(model_manager=model_manager, performance_profile=profile)
            return ClusterMeaningService().generate(
                clusters_by_key={comparison_key: {cluster_id: list(source_paths)}},
                all_image_paths=list(source_paths),
                snapshot_key=snapshot_key,
                path_fingerprints=fingerprints,
                embedding_service=embedding_service,
                selected_models=selected_models,
                requested_model="auto",
                use_embedding_cache_lookup=use_embedding_cache_lookup,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        job_id = self.job_manager.register_job("Generating cluster tag suggestions", cancel_fn=job.cancel, origin="Tags")
        self._tag_suggestion_job = job
        self._tag_suggestion_job_id = job_id
        thread_holder: dict[str, object] = {}
        self._refresh_tags_suggestion_state()

        def _finish(status: str, error: str = "") -> None:
            self.job_manager.finish(job_id, status=status, error=error)
            if self._tag_suggestion_job is job:
                self._tag_suggestion_job = None
                self._tag_suggestion_job_id = None

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            self._release_async_refs(job, thread)
            if self._tag_suggestion_thread is thread:
                self._tag_suggestion_thread = None
            self._refresh_tags_suggestion_state()

        def _done(result) -> None:
            _finish("finished")
            meanings, _metrics = result
            meaning = meanings.get(comparison_key, {}).get(cluster_id)
            if generation != self._tag_suggestion_generation or self._cluster_selection_identity(self.cluster_pane.current_selection_target()) != selection_identity:
                self.footer_bar.set_status("Generated tag suggestions were discarded because the selected cluster changed.")
                return
            if meaning is not None:
                self.cluster_meanings.setdefault(comparison_key, {})[cluster_id] = meaning
                self.cluster_pane.update_clusters(
                    self.cluster_data,
                    membership_by_image=self.membership_by_image,
                    metrics_by_backend=self.metrics_by_backend,
                    cluster_summaries=self.cluster_tag_summaries,
                    cluster_explanations=self.cluster_explanations,
                    cluster_meanings=self.cluster_meanings,
                    preserve_selection=True,
                )
            self.footer_bar.set_status("Cluster tag suggestions are ready." if meaning and meaning.status == "ok" else "No usable tag suggestions were available for this cluster.")

        def _failed(message: str) -> None:
            _finish("failed", str(message))
            self.footer_bar.set_status(f"Cluster tag suggestion generation failed: {message}")
            errorBox("Tag suggestions failed", str(message))

        def _cancelled() -> None:
            _finish("cancelled")
            self.footer_bar.set_status("Cluster tag suggestion generation cancelled.")

        job.progress.connect(lambda value, text: self.job_manager.update(job_id, progress=value, text=text))
        job.progress.connect(lambda _value, text: self._set_activity(text))
        job.completed.connect(_done)
        job.completed.connect(lambda _result: _cleanup())
        job.failed.connect(_failed)
        job.failed.connect(lambda _message: _cleanup())
        job.cancelled.connect(_cancelled)
        job.cancelled.connect(_cleanup)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._tag_suggestion_thread = thread
        self._retain_async_refs(job, thread)

    def _selected_cluster_meaning_tags(self) -> tuple[list[str], SelectionTarget | None]:
        target = self.cluster_pane.current_selection_target()
        if target is None:
            return [], None
        comparison_key = str(target.source_context.get("comparison_key") or target.source_context.get("backend") or "")
        try:
            cluster_id = int(target.source_context.get("cluster_id", -1))
        except Exception:
            cluster_id = -1
        meaning = self.cluster_meanings.get(comparison_key, {}).get(cluster_id)
        if meaning is None or meaning.status != "ok" or not meaning.labels:
            return [], target
        tags: list[str] = []
        seen: set[str] = set()
        for label in meaning.labels[:3]:
            display_tag = self.image_tag_service.clean_display_tag(label.label)
            normalized = self.image_tag_service.normalize_tag(display_tag)
            if not display_tag or not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tags.append(display_tag)
        return tags, target

    def apply_cluster_tag_suggestions(self) -> None:
        if self._read_only_mode():
            self.footer_bar.set_status("Read-only safety mode is enabled. Suggested tag writes are disabled.")
            self._set_activity("Read-only mode")
            return
        tags, target = self._selected_cluster_meaning_tags()
        if target is None:
            errorBox("Select a cluster", "Select a cluster before applying suggested tags.")
            return
        if not tags:
            errorBox("No tag suggestions", "Cluster Meaning has no model labels available for the selected cluster.")
            return
        message = (
            f"Apply suggested tag(s) {', '.join(tags)} to {target.count} image(s) from {target.label}?\n\n"
            "These are model-derived guesses and will be written to the app tag database only, not EXIF."
        )
        if not confirmBox("Apply suggested tags", message, parent=self):
            return
        source_paths = target.as_list()

        def _run(progress, cancel_check):
            return self.image_tag_service.apply_tag_edit(
                source_paths,
                add_tags=tags,
                mirror_to_exif=False,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            failures = list(getattr(result, "failures", []))
            affected_paths = list(getattr(result, "affected_paths", []))
            self._set_activity("Ready")
            if affected_paths:
                self._on_gallery_metadata_changed(affected_paths)
                if self.tags_pane is not None:
                    self.tags_pane.refresh()
            if failures:
                errorBox("Suggested tags partly failed", "\n".join(failures[:8]))
                return
            infoBox("Suggested tags applied", f"Added {', '.join(tags)} to {len(affected_paths)} image(s).")

        job = AsyncJob(_run)
        suggested_tags_job_id = self.job_manager.register_job(
            "Applying suggested tags", cancel_fn=job.cancel, origin="Tags"
        )
        thread_holder: dict[str, object] = {}

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            self._release_async_refs(job, thread)

        job.progress.connect(
            lambda value, status: self.job_manager.update(
                suggested_tags_job_id,
                progress=value,
                text=status,
            )
        )
        job.progress.connect(lambda _value, status: self._set_activity(status))

        def _completed(result) -> None:
            failures = list(getattr(result, "failures", []))
            if failures:
                self.job_manager.finish(
                    suggested_tags_job_id,
                    status="failed",
                    error="; ".join(str(item) for item in failures[:8]),
                )
            else:
                self.job_manager.finish(suggested_tags_job_id, status="finished")
            _done(result)

        def _failed(message: str) -> None:
            self.job_manager.finish(suggested_tags_job_id, status="failed", error=message)
            errorBox("Suggested tags failed", message)

        def _cancelled() -> None:
            self.job_manager.finish(suggested_tags_job_id, status="cancelled")
            self.footer_bar.set_status("Suggested tag update cancelled.")

        job.completed.connect(_completed)
        job.completed.connect(lambda _result: _cleanup())
        job.failed.connect(_failed)
        job.failed.connect(lambda _message: _cleanup())
        job.cancelled.connect(_cancelled)
        job.cancelled.connect(_cleanup)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._retain_async_refs(job, thread)

    def _create_settings_dialog(self, *, parent=None) -> ProductionSettingsDialog:
        return ProductionSettingsDialog(
            self.settings_store,
            self.runtime_service,
            describe_rebuildable_caches=self.cache_maintenance_service.describe_rebuildable_caches,
            describe_generated_storage=self._describe_generated_storage,
            clear_rebuildable_caches=self._clear_rebuildable_caches,
            clear_library_catalog=self._clear_library_catalog,
            clear_runtime_temp_files=self._clear_runtime_temp_files,
            clear_face_storage=self._clear_face_storage,
            clear_model_caches=self._clear_model_caches,
            clear_logs=self._clear_logs,
            clear_runtime_reports=self._clear_runtime_reports,
            clear_model_assets=self._clear_model_assets,
            can_clear_rebuildable_caches=lambda: not self._background_runtime_work_active(),
            data_home_manager=self.data_home_manager,
            active_source_roots_provider=lambda: tuple(self.source_pane.active_scope.roots),
            runtime_layout=self.runtime_layout,
            support_metadata_provider=self._support_metadata,
            model_download_controller=self.model_download_controller,
            job_manager=self.job_manager,
            auto_refresh=not self._packaged_launch_smoke,
            parent=self if parent is None else parent,
        )

    def open_settings_dialog(self, initial_tab: str | None = None) -> None:
        dialog = self._create_settings_dialog(parent=self)
        dialog.runtime_rescanned.connect(self._begin_startup_readiness_check)
        if initial_tab:
            aliases = {
                "models": "Clustering Models",
                "face models": "Face Models",
            }
            target_tab = aliases.get(str(initial_tab).casefold(), str(initial_tab))
            for index in range(dialog.tabs.count()):
                if dialog.tabs.tabText(index).casefold() == target_tab.casefold():
                    dialog.tabs.setCurrentIndex(index)
                    break
        dialog_result = dialog.exec()
        face_models_changed = bool(dialog.face_model_inventory_changed())
        if dialog_result != dialog.DialogCode.Accepted:
            if face_models_changed:
                self._reload_face_services_from_settings()
                self._begin_startup_readiness_check()
            self._refresh_workspace_ui()
            return
        self._apply_settings_values(dialog.values())
        self.footer_bar.set_status("Production settings updated.")
        self._set_activity("Settings updated")
        self._refresh_footer_storage_usage()
        self._refresh_workspace_ui()

    def _apply_settings_values(self, values: dict[str, object]) -> None:
        for key, value in self.settings_registry.validate_values(values).items():
            self.settings_registry.set(self.settings_store, key, value)
        self.settings_store.sync()
        self._apply_runtime_status()
        self.work_coordinator.gpu_policy = self._gpu_conflict_policy()
        self._apply_workspace_preferences()
        self._apply_safety_state()
        self.session_controller.set_keep_worker_warm(self._keep_worker_warm())
        self._reload_face_services_from_settings()
        self._begin_startup_readiness_check()

    def _reload_face_services_from_settings(self) -> None:
        if self.faces_pane is None:
            self._face_service_cache = {}
            self.face_services_global = {}
            self.face_services_session = {}
            self.face_service_global = None
            self.face_service_session = None
            if self.names_pane is not None:
                self.names_pane.status_label.setText("Face settings changed. Reload Names to use the selected pipeline.")
                self.names_pane.refresh_button.setEnabled(False)
                self._start_names_workspace_load()
            return
        self._build_face_services(reset_face_session=False)
        self.faces_pane.face_service_global = self.face_service_global
        self.faces_pane.face_service_session = self.face_service_session
        self.faces_pane.set_face_services(
            face_services_global=self.face_services_global,
            face_services_session=self.face_services_session,
            refresh=False,
        )
        self.faces_pane.set_face_service_provider(self._face_service_for_pipeline)
        self.faces_pane.configure_face_pipeline_options(
            self._face_model_root(),
            self._preferred_face_pipeline_defaults(),
            refresh=False,
        )
        self.faces_pane.set_active_face_mode("human", refresh=True)
        if self.names_pane is not None:
            self.names_pane.refresh_names()

    def _post_startup_checks(self) -> None:
        if self._is_shutting_down or self._close_pending or not self.isVisible():
            return
        self._maybe_show_first_run_setup()
        if self._is_shutting_down:
            return
        self._maybe_show_last_crash_notice()
        self._refresh_health_badge()
        self._begin_startup_readiness_check()

    def _maybe_show_first_run_setup(self) -> None:
        if self.settings_store.value("setup/completed", False, bool):
            return
        dialog = FirstRunSetupDialog(self.settings_store, self.runtime_layout, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            values = dialog.values()
            download_default_model = bool(values.get("models/download_default_after_setup"))
            self._apply_settings_values(values)
            self.footer_bar.set_status("First-run setup saved.")
            self._set_activity("First-run setup saved")
            if download_default_model:
                self._pending_post_install_model_name = str(self.settings.default_model or "").strip()
                self._post_install_timer.start(0)
        else:
            self.footer_bar.set_status("First-run setup skipped. Open Settings to change production defaults.")
        self.settings_store.setValue("setup/completed", True)
        self.settings_store.sync()

    def _run_pending_post_install_model_download(self) -> None:
        model_name = str(self._pending_post_install_model_name or "").strip()
        self._pending_post_install_model_name = None
        if not model_name:
            return
        self._start_post_install_model_download(model_name)

    def _start_post_install_model_download(self, model_name: str) -> None:
        model_name = str(model_name or self.settings.default_model).strip().lower()
        if not model_name:
            return
        if self.model_download_controller.is_running():
            self.footer_bar.set_status("A model download is already active. Open Jobs to monitor or cancel it.")
            return
        self.settings_store.setValue("models/download_default_after_setup", False)
        self.settings_store.sync()
        require_text = model_name in TEXT_MODEL_ORDER
        if self.model_asset_service.model_available_without_download(model_name, require_text=require_text):
            self.footer_bar.set_status(f"{model_label(model_name)} is already available locally.")
            self._set_activity("Model ready")
            return
        self._active_post_install_model_name = model_name
        if not self.model_download_controller.start([ModelDownloadItem(model_name, require_text=require_text)]):
            self._active_post_install_model_name = None
            self.footer_bar.set_status("The model download could not start.")

    def _request_face_model_download(self, model_name: str) -> None:
        model_name = str(model_name or "facenet").strip().lower()
        require_text = model_name in TEXT_MODEL_ORDER
        if self.model_asset_service.model_available_without_download(model_name, require_text=require_text):
            self._reload_face_services_from_settings()
            self.footer_bar.set_status(f"{model_label(model_name)} is already available locally. Scan again to continue.")
            return
        if self.model_download_controller.is_running():
            infoBox(
                "Model download already running",
                "Another model download is active. Open Jobs to monitor or cancel it, then scan again.",
            )
            return
        if not confirmBox(
            "Install face model?",
            (
                f"Face scanning needs {model_label(model_name)} model files.\n\n"
                f"Download location: {self.settings.cache_dir}\n\n"
                "The download is checksum-verified, shared across runs, resumable from the cache, "
                "visible in Jobs, and cancellable.\n\n"
                "Install it now?"
            ),
            parent=self,
        ):
            self.footer_bar.set_status("Face-model installation cancelled. No download was started.")
            return
        self._active_post_install_model_name = model_name
        if not self.model_download_controller.start([ModelDownloadItem(model_name, require_text=require_text)]):
            self._active_post_install_model_name = None
            errorBox("Model download could not start", "Open Settings > Face Models and try the installation again.")

    def _maybe_show_last_crash_notice(self) -> None:
        app = QApplication.instance()
        if app is not None and str(app.platformName() or "").strip().lower() == "offscreen":
            return
        crash_path = self.runtime_layout.last_crash_json
        if not crash_path.exists():
            return
        try:
            crash_mtime = str(int(crash_path.stat().st_mtime))
        except OSError:
            crash_mtime = "unknown"
        last_seen = self.settings_store.value("diagnostics/last_seen_crash_mtime", "", str)
        if last_seen == crash_mtime:
            return
        self.settings_store.setValue("diagnostics/last_seen_crash_mtime", crash_mtime)
        self.settings_store.sync()
        infoBox(
            "Previous crash report found",
            (
                "The previous production run wrote a crash record. "
                "Open Settings > Support to view it or export a support bundle.\n\n"
                f"Crash record: {crash_path}"
            ),
        )

    def _clear_rebuildable_caches(self, *, progress_callback=None, cancel_check=None):
        before = self.cache_maintenance_service.describe_rebuildable_caches(cancel_check=cancel_check)
        raise_if_cancelled(cancel_check)
        cleared_targets, failures = self._clear_embedding_cache(cancel_check=cancel_check)
        disk_cleared, disk_failures = self.cache_maintenance_service.clear_rebuildable_disk_targets(
            exclude={"embeddings.sqlite3"},
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        cleared_targets.extend(disk_cleared)
        failures.extend(disk_failures)
        raise_if_cancelled(cancel_check)
        after = self.cache_maintenance_service.describe_rebuildable_caches(cancel_check=cancel_check)
        return CacheClearResult(
            cleared_targets=tuple(cleared_targets),
            freed_bytes=max(0, int(before.total_bytes - after.total_bytes)),
            failures=tuple(failures),
        )

    def _describe_generated_storage(self, *, progress_callback=None, cancel_check=None):
        return self.cache_maintenance_service.describe_generated_storage(
            config_location=str(self.settings_store.fileName() or "platform defaults"),
            runtime_layout=self.runtime_layout,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _clear_generated_storage_categories(self, categories: tuple[str, ...], *, progress_callback=None, cancel_check=None):
        before = self._describe_generated_storage(cancel_check=cancel_check)
        cleared_targets: list[str] = []
        failures: list[str] = []
        total = max(1, len(categories))
        for index, category in enumerate(categories):
            raise_if_cancelled(cancel_check)

            def _progress(value: int, text: str, *, _index: int = index) -> None:
                if progress_callback:
                    progress_callback(int((_index * 100 + max(0, min(100, int(value)))) / total), text)

            cleared, category_failures = self.cache_maintenance_service.clear_generated_storage_category(
                category,
                runtime_layout=self.runtime_layout,
                progress_callback=_progress,
                cancel_check=cancel_check,
            )
            cleared_targets.extend(cleared)
            failures.extend(category_failures)
        raise_if_cancelled(cancel_check)
        after = self._describe_generated_storage(cancel_check=cancel_check)
        freed_bytes = sum(
            max(0, int(before.target_bytes.get(category, 0)) - int(after.target_bytes.get(category, 0)))
            for category in categories
        )
        return CacheClearResult(
            cleared_targets=tuple(cleared_targets),
            freed_bytes=freed_bytes,
            failures=tuple(failures),
        )

    def _clear_runtime_temp_files(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("temp_files",), progress_callback=progress_callback, cancel_check=cancel_check
        )

    def _clear_library_catalog(self, *, progress_callback=None, cancel_check=None):
        if self.library_pane is None:
            return CacheClearResult((), 0, ("Library workspace is unavailable.",))
        catalog = self.library_pane.catalog
        before = catalog.catalog_storage_bytes()
        catalog.clear_library_cache(
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )
        after = catalog.catalog_storage_bytes()
        # The dialog owns the worker. Refresh the Qt view on the next normal
        # workspace update rather than touching widgets from that worker.
        self._library_catalog_refresh_pending = True
        return CacheClearResult(
            cleared_targets=("library_catalog",),
            freed_bytes=max(0, int(before - after)),
            failures=(),
        )

    def _clear_face_storage(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("face_databases", "ann_files"), progress_callback=progress_callback, cancel_check=cancel_check
        )

    def _clear_model_caches(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("model_caches",), progress_callback=progress_callback, cancel_check=cancel_check
        )

    def _clear_logs(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("logs",), progress_callback=progress_callback, cancel_check=cancel_check
        )

    def _clear_runtime_reports(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("crash_reports", "support_bundles", "benchmarks"),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def _clear_model_assets(self, *, progress_callback=None, cancel_check=None):
        return self._clear_generated_storage_categories(
            ("model_assets",), progress_callback=progress_callback, cancel_check=cancel_check
        )

    def _clear_embedding_cache(self, *, cancel_check=None) -> tuple[list[str], list[str]]:
        failures: list[str] = []
        db_path = Path(self.settings.embedding_cache_db)
        removed = False
        for suffix in ("", "-wal", "-shm"):
            raise_if_cancelled(cancel_check)
            target = db_path.with_name(db_path.name + suffix)
            try:
                if target.exists():
                    target.unlink(missing_ok=True)
                    removed = True
            except OSError as exc:
                failures.append(f"{target.name}: {exc}")
        try:
            self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            failures.append(f"cache_dir: {exc}")
        if failures:
            return (["embeddings.sqlite3"] if removed else []), failures
        return ["embeddings.sqlite3"], []

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        size = float(max(0, int(size_bytes)))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{int(size_bytes)} B"

    def _set_footer_storage_summary(self, summary: RuntimeStorageSummary) -> None:
        if not self._can_update_widget(getattr(self, "footer_bar", None)):
            return
        self._last_storage_summary = summary
        text = f"Storage: {self._format_bytes(summary.total_bytes)}"
        lines = [f"Runtime root: {summary.runtime_root}", ""]
        labels = {
            "rebuildable_caches": "Rebuildable caches",
            "runtime_temp_files": "Runtime temp files",
            "tag_database": "Tag database",
            "other_cache_data": "Other cache data",
            "logs": "Logs",
            "crash_reports": "Crash reports",
            "benchmarks": "Benchmarks",
            "support_bundles": "Support bundles",
            "model_assets": "Model assets",
        }
        for key, label in labels.items():
            lines.append(f"{label}: {self._format_bytes(summary.target_bytes.get(key, 0))}")
        lines.append("")
        lines.append("Clear rebuildable data removes caches, temporary files, support bundles, and benchmark artifacts.")
        lines.append("Tags, logs, crash records, and model assets are preserved.")
        self.footer_bar.set_storage_usage(text, tooltip="\n".join(lines))

    def _refresh_footer_storage_usage(self) -> None:
        if self._is_shutting_down or not self._can_update_widget(getattr(self, "footer_bar", None)):
            return
        if self._thread_is_running(self._storage_clear_thread):
            return
        if self._thread_is_running(self._storage_usage_thread):
            return
        self._storage_usage_generation += 1
        generation = self._storage_usage_generation
        self.footer_bar.set_storage_usage("Storage: scanning...", tooltip="Scanning runtime storage usage...")
        self._update_footer_storage_state()

        def _run(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            return self.cache_maintenance_service.describe_runtime_storage(
                self.runtime_layout,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        self._storage_usage_job = job
        thread_holder: dict[str, object | None] = {"thread": None}

        def _current_generation() -> bool:
            return generation == self._storage_usage_generation

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            if thread is not None:
                self._release_async_refs(job, thread)
            if self._storage_usage_thread is thread:
                self._storage_usage_thread = None
            if self._storage_usage_job is job:
                self._storage_usage_job = None
            self._update_footer_storage_state()

        def _done(payload: object) -> None:
            if self._is_shutting_down:
                _cleanup()
                return
            if _current_generation() and isinstance(payload, RuntimeStorageSummary):
                self._set_footer_storage_summary(payload)
            _cleanup()

        def _failed(message: str) -> None:
            if self._is_shutting_down:
                _cleanup()
                return
            if _current_generation() and self._can_update_widget(getattr(self, "footer_bar", None)):
                self.footer_bar.set_storage_usage("Storage: unavailable", tooltip=f"Storage refresh failed: {message}")
                LOGGER.warning("Footer storage refresh failed: %s", message)
            _cleanup()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cleanup)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._storage_usage_thread = thread
        self._update_footer_storage_state()

    def _request_runtime_storage_clear(self) -> None:
        if self._background_runtime_work_active():
            self.footer_bar.set_status("Wait for background work to finish before clearing caches.")
            self._set_activity("Busy")
            self._update_footer_storage_state()
            return
        summary = self._last_storage_summary
        current_usage_text = self._format_bytes(summary.total_bytes) if summary is not None else "still scanning"
        if not confirmBox(
            "Clear caches and generated runtime data",
            (
                "This clears rebuildable caches, runtime temp files, support bundles, benchmark artifacts, and in-memory gallery caches.\n\n"
                f"Current runtime storage: {current_usage_text}.\n"
                "Tag data, logs, crash records, and model assets are preserved."
            ),
            parent=self,
        ):
            return
        self.gallery_pane.clear_memory_caches(reload_visible=False)
        self.footer_bar.set_status("Clearing caches and generated runtime data...")
        self._set_activity("Clearing caches...")
        self._update_footer_storage_state()

        def _run(progress, cancel_check):
            progress(-1, "Clearing caches and generated runtime data...")
            before = self.cache_maintenance_service.describe_runtime_storage(
                self.runtime_layout,
                cancel_check=cancel_check,
            )
            raise_if_cancelled(cancel_check)
            cleared_targets, failures = self._clear_embedding_cache(cancel_check=cancel_check)
            disk_cleared, disk_failures = self.cache_maintenance_service.clear_runtime_cleanup_targets(
                self.runtime_layout,
                exclude={"embeddings.sqlite3"},
                progress_callback=progress,
                cancel_check=cancel_check,
            )
            cleared_targets.extend(disk_cleared)
            failures.extend(disk_failures)
            raise_if_cancelled(cancel_check)
            after = self.cache_maintenance_service.describe_runtime_storage(
                self.runtime_layout,
                cancel_check=cancel_check,
            )
            return CacheClearResult(
                cleared_targets=tuple(cleared_targets),
                freed_bytes=max(0, int(before.total_bytes - after.total_bytes)),
                failures=tuple(failures),
            )

        job = AsyncJob(_run)
        self._storage_clear_job = job
        storage_clear_job_id = self.job_manager.register_job(
            "Clearing runtime caches", cancel_fn=job.cancel, origin="Storage"
        )
        thread_holder: dict[str, object | None] = {"thread": None}

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            if thread is not None:
                self._release_async_refs(job, thread)
            if self._storage_clear_thread is thread:
                self._storage_clear_thread = None
            if self._storage_clear_job is job:
                self._storage_clear_job = None
            self._update_footer_storage_state()
            self._refresh_footer_storage_usage()

        def _done(result: object) -> None:
            if isinstance(result, CacheClearResult):
                if result.failures:
                    self.job_manager.finish(
                        storage_clear_job_id,
                        status="failed",
                        error="; ".join(result.failures[:8]),
                    )
                else:
                    self.job_manager.finish(storage_clear_job_id, status="finished")
                self.footer_bar.set_status(
                    f"Cleared {len(result.cleared_targets)} targets and freed {self._format_bytes(result.freed_bytes)}."
                )
                self._set_activity("Caches cleared")
                if result.failures:
                    LOGGER.warning("Runtime cache clear completed with errors: %s", "; ".join(result.failures))
            else:
                self.job_manager.finish(storage_clear_job_id, status="finished")
                self.footer_bar.set_status("Caches cleared.")
                self._set_activity("Caches cleared")
            if self.gallery_pane.images:
                self.gallery_pane.schedule_visible_refresh()
            _cleanup()

        def _failed(message: str) -> None:
            self.job_manager.finish(storage_clear_job_id, status="failed", error=message)
            self.footer_bar.set_status(f"Cache clear failed: {message}")
            self._set_activity("Cache clear failed")
            errorBox("Cache clear failed", message)
            _cleanup()

        def _cancelled() -> None:
            self.job_manager.finish(storage_clear_job_id, status="cancelled")
            self.footer_bar.set_status("Cache clear cancelled. Already removed rebuildable files remain removed.")
            self._set_activity("Cache clear cancelled")
            _cleanup()

        job.progress.connect(
            lambda value, status: self.job_manager.update(
                storage_clear_job_id,
                progress=value,
                text=status,
            )
        )
        job.progress.connect(lambda _value, status: self._set_activity(status))
        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._storage_clear_thread = thread
        self._update_footer_storage_state()

    def _support_metadata(self) -> dict[str, object]:
        return {
            "workspace": self._active_workspace,
            "mode": self._clustering_mode,
            "faces_mode": self._faces_mode,
            "active_roots": list(self.source_pane.active_scope.roots),
            "last_run_metrics": self.last_run_metrics,
            "runtime": self.runtime_service.diagnostics(self._preferred_execution_mode()),
            "runtime_root": str(self.runtime_layout.root),
            "logs_dir": str(self.runtime_layout.logs_dir),
            "cache_dir": str(self.runtime_layout.cache_dir),
            "runtime_storage_bytes": getattr(self._last_storage_summary, "total_bytes", 0),
            "offline_model_downloads": self._offline_model_downloads(),
            "read_only_mode": self._read_only_mode(),
            "model_inventory": [item.__dict__ for item in self.model_asset_service.model_inventory(self.clustering_pane.selected_embedding_models())],
            "current_run_origin": self.current_run_origin,
            "current_tag_filter": list(self.current_tag_filter),
            "current_tag_match": self.current_tag_match,
        }

    def _capture_face_context(self, paths: list[str]) -> dict[str, dict[str, object]]:
        if self.faces_pane is None:
            return {}
        return {
            str(path): dict(context)
            for path in paths
            if str(path or "").strip()
            for context in [self.faces_pane.context_for_path(str(path))]
            if isinstance(context, dict) and context
        }

    def _current_clustering_filter_state(self) -> dict[str, object]:
        return {
            "tags": self.tags_pane.selected_filter_tags() if self.tags_pane is not None else [],
            "tag_match": self.tags_pane.selected_match_mode() if self.tags_pane is not None else "Any",
        }

    def _run_saved_clustering_filter(self, payload: dict) -> None:
        tags = payload.get("tags", payload.get("tag_filter", [])) if isinstance(payload, dict) else []
        if isinstance(tags, str):
            tag_values = [tag.strip() for tag in tags.split(",") if tag.strip()]
        else:
            tag_values = [str(tag).strip() for tag in list(tags or []) if str(tag).strip()]
        if not tag_values:
            errorBox("Saved clustering filter is empty", "The saved clustering filter does not contain any tags.")
            return
        tag_match = str((payload or {}).get("tag_match") or (payload or {}).get("match") or "Any").strip().lower()
        if self.tags_pane is not None:
            self.tags_pane.set_filter(tag_values, "All" if tag_match == "all" else "Any")
        self.set_active_workspace("clustering")
        self.run_clustering()

    def _open_face_results_in_main_gallery(self, paths: list[str]) -> None:
        ordered_paths = list(dict.fromkeys(str(path) for path in paths if str(path or "").strip()))
        if not ordered_paths:
            return
        self._open_photo_set_route(
            ordered_paths,
            source="Faces results",
            title="Face results",
            return_workspace="faces",
            context_by_path=self._capture_face_context(ordered_paths),
        )

    def _append_face_results_to_main_gallery(self, paths: list[str]) -> None:
        ordered_paths = list(dict.fromkeys(str(path) for path in paths if str(path or "").strip()))
        if not ordered_paths:
            return
        self._open_photo_set_route(
            ordered_paths,
            source="Faces results",
            title="Face results",
            return_workspace="faces",
            context_by_path=self._capture_face_context(ordered_paths),
            append=True,
        )

    def _open_named_photos_in_gallery(self, paths: list[str], name: str) -> None:
        person_name = str(name or "").strip()
        self._open_photo_set_route(
            paths,
            source=f"Names · {person_name}",
            title=f"Photos named {person_name}",
            return_workspace="names",
            context_by_path={
                str(path): {"named_face": {"person_name": person_name}}
                for path in self._unique_photo_paths(paths)
            },
        )

    def _open_names_similar_faces_in_gallery(
        self,
        paths: list[str],
        name: str,
        context_by_path: dict[str, dict[str, object]],
    ) -> None:
        person_name = str(name or "").strip()
        ordered_paths = self._unique_photo_paths(paths)
        if not ordered_paths or not person_name:
            return
        self._open_photo_set_route(
            ordered_paths,
            source=f"Names · similar to {person_name}",
            title=f"Faces similar to {person_name}",
            return_workspace="names",
            context_by_path={
                str(path): dict(context_by_path.get(str(path), {}) or {})
                for path in ordered_paths
            },
        )

    def _open_tagged_photos_in_gallery(self, paths: list[str], tag: str) -> None:
        tag_name = str(tag or "").strip()
        self._open_photo_set_route(
            paths,
            source=f"Tags · {tag_name}",
            title=f"Photos tagged {tag_name}",
            return_workspace="tags",
            context_by_path={
                str(path): {"tag_route": {"tag": tag_name}}
                for path in self._unique_photo_paths(paths)
            },
        )

    def _open_library_photos_in_gallery(self, paths: list[str], source: str) -> None:
        route_source = str(source or "Library").strip() or "Library"
        self._open_photo_set_route(
            paths,
            source=f"Library · {route_source}",
            title=route_source,
            return_workspace="library",
            context_by_path={
                str(path): {"library_route": {"source": route_source}}
                for path in self._unique_photo_paths(paths)
            },
        )

    def _open_selected_cluster_in_gallery(self, target: SelectionTarget | None) -> None:
        if target is None:
            return
        context = dict(target.source_context or {})
        self._open_photo_set_route(
            list(target.paths),
            source=f"Clustering · {target.label}",
            title=target.label,
            return_workspace="clustering",
            context_by_path={str(path): {"cluster_route": context} for path in target.paths},
        )

    def _main_gallery_context_for_path(self, path: str) -> dict[str, object]:
        context = self._gallery_context_for_path(path)
        override = self._main_gallery_context_overrides.get(path)
        if isinstance(override, dict):
            context.update(override)
        return context

    def _gallery_context_for_path(self, path: str) -> dict[str, object]:
        context: dict[str, object] = {
            "membership": self.membership_by_image.get(path, {}),
            "tags": list(self.image_tags_by_path.get(path, ())),
        }
        extra = self.gallery_extra_context_by_path.get(path)
        if isinstance(extra, dict):
            context.update(extra)
        return context

    def _set_activity(self, text: str) -> None:
        if self._close_pending or not self._can_update_widget(getattr(self, "footer_bar", None)):
            return
        normalized = str(text or "").strip() or "Idle"
        self.footer_bar.set_status(normalized)

    def _refresh_health_badge(self) -> None:
        if self._is_shutting_down or not self._can_update_widget(getattr(self, "runtime_badge", None)):
            return
        issues: list[str] = []
        notes: list[str] = [
            f"Runtime root: {self.runtime_layout.root}",
            f"App log: {self.runtime_layout.app_log}",
            f"Audit log: {self.gallery_pane.action_service.audit_log_path if hasattr(self, 'gallery_pane') else ''}",
        ]
        if self._offline_model_downloads():
            notes.append("Model policy: offline/fallback mode")
        else:
            notes.append("Model policy: prompt before downloads")
        if self._read_only_mode():
            notes.append("Data safety: read-only mode enabled")
        try:
            bundled = self.model_asset_service.bundled_model_names(valid_only=True)
            if bundled:
                notes.append(f"Valid bundled models: {', '.join(bundled)}")
            else:
                issues.append("No valid bundled fallback model asset found")
        except Exception as exc:
            issues.append(f"Model asset check failed: {exc}")
        if self.runtime_layout.last_crash_json.exists():
            issues.append("Crash report present")
            notes.append(f"Last crash: {self.runtime_layout.last_crash_json}")
        if issues:
            notes.extend(["", "Application health notes:"])
            notes.extend(f"- {issue}" for issue in issues)
            self.runtime_badge.set_health("warning", notes)
        else:
            self.runtime_badge.set_health("ok", notes)

    @staticmethod
    def _thread_is_running(thread) -> bool:
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:
            return False

    @staticmethod
    def _qt_object_alive(widget: object | None) -> bool:
        if widget is None:
            return False
        try:
            return not bool(sip.isdeleted(widget))
        except Exception:
            return True

    def _can_update_widget(self, widget: object | None) -> bool:
        return not self._is_shutting_down and not self._close_pending and self._qt_object_alive(widget)

    @staticmethod
    def _wait_for_thread(thread, timeout_ms: int = 2500) -> bool:
        return wait_for_thread_shutdown(thread, timeout_ms=timeout_ms)

    def _retain_async_refs(self, job: object | None, thread: object | None) -> None:
        if not self._thread_is_running(thread):
            return
        if any(existing_thread is thread for _existing_job, existing_thread in self._retained_async_refs):
            return
        self._retained_async_refs.append((job, thread))

    def _release_async_refs(self, job: object | None, thread: object | None) -> None:
        self._retained_async_refs = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._retained_async_refs
            if existing_thread is not thread
        ]
        if self._tag_context_thread is thread:
            self._tag_context_thread = None
            if self._tag_context_job is job:
                self._tag_context_job = None

    def _cancel_cluster_tag_context_refresh(self) -> None:
        job = self._tag_context_job
        thread = self._tag_context_thread
        job_id = self._tag_context_job_id
        self._retain_async_refs(job, thread)
        self._tag_context_job = None
        self._tag_context_thread = None
        self._tag_context_job_id = None
        if job_id is not None:
            self.job_manager.finish(job_id, status="cancelled")
        if job is None:
            return
        self._tag_context_generation += 1
        try:
            job.cancel()
        except Exception:
            pass
        self._update_footer_storage_state()

    def _start_cluster_tag_context_refresh(self) -> None:
        snapshot = {
            str(comparison_key): {
                int(cluster_id): tuple(str(path) for path in image_paths if str(path or "").strip())
                for cluster_id, image_paths in clusters.items()
            }
            for comparison_key, clusters in self.cluster_data.items()
        }
        if not snapshot:
            self.image_tags_by_path = {}
            self.cluster_tag_summaries = {}
            self.gallery_extra_context_by_path = {}
            self._set_activity("Ready")
            self._update_footer_storage_state()
            return

        self._cancel_cluster_tag_context_refresh()
        self._tag_context_generation += 1
        generation = self._tag_context_generation
        run_origin = self.current_run_origin
        tag_filter = tuple(self.current_tag_filter)
        tag_match = self.current_tag_match or "Any"

        def _run(progress, cancel_check):
            all_paths: list[str] = []
            seen: set[str] = set()
            for clusters in snapshot.values():
                for image_paths in clusters.values():
                    for image_path in image_paths:
                        normalized = str(image_path or "").strip()
                        if not normalized or normalized in seen:
                            continue
                        seen.add(normalized)
                        all_paths.append(normalized)

            def _tag_progress(value: int, status: str) -> None:
                progress(min(80, max(0, int(value * 0.8))), status)

            tags_by_path = self.image_tag_service.load_tags_for_paths(
                all_paths,
                progress_callback=_tag_progress,
                cancel_check=cancel_check,
            )
            progress(80, "Refreshing cluster summaries...")
            total_clusters = max(1, sum(len(clusters) for clusters in snapshot.values()))
            processed = 0
            cluster_tag_summaries: dict[str, dict[int, ClusterTagSummary]] = {}
            for comparison_key, clusters in snapshot.items():
                cluster_tag_summaries[comparison_key] = {}
                for cluster_id, image_paths in clusters.items():
                    raise_if_cancelled(cancel_check)
                    cluster_tag_summaries[comparison_key][int(cluster_id)] = self.image_tag_service.summarize_paths(
                        list(image_paths),
                        tags_by_path=tags_by_path,
                    )
                    processed += 1
                    progress(80 + int(processed * 20 / total_clusters), f"Refreshing cluster summaries {processed}/{total_clusters}")

            gallery_extra_context_by_path = {
                image_path: _build_gallery_context(
                    image_tags=tags_by_path.get(image_path, ()),
                    run_origin=run_origin,
                    tag_filter=tag_filter,
                    tag_match=tag_match,
                )
                for image_path in all_paths
            }
            return ClusterTagContextPayload(
                image_tags_by_path=tags_by_path,
                cluster_tag_summaries=cluster_tag_summaries,
                gallery_extra_context_by_path=gallery_extra_context_by_path,
            )

        job = AsyncJob(_run)
        self._tag_context_job = job
        job_id = self.job_manager.register_job(
            "Refreshing cluster tag context",
            cancel_fn=job.cancel,
            origin="Clustering",
            foreground=False,
        )
        self._tag_context_job_id = job_id
        job.progress.connect(
            lambda value, text, job_id=job_id: self.job_manager.update(
                job_id, progress=value, text=str(text)
            )
        )
        self._set_activity("Refreshing cluster tag context...")
        thread_holder: dict[str, object | None] = {"thread": None}

        def _current_generation() -> bool:
            return generation == self._tag_context_generation

        def _finish_job(status: str, error: str = "") -> None:
            self.job_manager.finish(job_id, status=status, error=error)
            if self._tag_context_job_id == job_id:
                self._tag_context_job_id = None

        def _done(payload: object) -> None:
            _finish_job("finished")
            if not _current_generation() or not isinstance(payload, ClusterTagContextPayload):
                return
            self.image_tags_by_path = dict(payload.image_tags_by_path)
            self.cluster_tag_summaries = {
                comparison_key: dict(summaries)
                for comparison_key, summaries in payload.cluster_tag_summaries.items()
            }
            self.gallery_extra_context_by_path = {
                image_path: dict(context)
                for image_path, context in payload.gallery_extra_context_by_path.items()
            }
            self.cluster_pane.update_clusters(
                self.cluster_data,
                self.membership_by_image,
                self.metrics_by_backend,
                self.cluster_tag_summaries,
                self.cluster_explanations,
                self.cluster_meanings,
                preserve_selection=True,
            )
            self.footer_bar.set_status("Cluster context ready.")
            self._set_activity("Ready")
            self._refresh_footer_storage_usage()

        def _failed(message: str) -> None:
            _finish_job("failed", message)
            if not _current_generation():
                return
            self.footer_bar.set_status(f"Cluster context refresh failed: {message}")
            self._set_activity("Cluster context refresh failed")
            LOGGER.warning("Production tag-context refresh failed: %s", message)

        def _cancelled() -> None:
            _finish_job("cancelled")
            if not _current_generation():
                return
            self.footer_bar.set_status("Cluster context refresh cancelled.")
            self._set_activity("Ready")

        def _cleanup() -> None:
            thread = thread_holder.get("thread")
            if thread is not None:
                self._release_async_refs(job, thread)
            if self._tag_context_thread is thread:
                self._tag_context_thread = None
            if self._tag_context_job is job:
                self._tag_context_job = None
            self._update_footer_storage_state()

        job.progress.connect(lambda _value, status: self._set_activity(status))
        job.completed.connect(_done)
        job.completed.connect(lambda _payload: _cleanup())
        job.failed.connect(_failed)
        job.failed.connect(lambda _message: _cleanup())
        job.cancelled.connect(_cancelled)
        job.cancelled.connect(lambda: _cleanup())
        thread = start_job_in_thread(job)
        thread_holder["thread"] = thread
        self._tag_context_thread = thread
        self._update_footer_storage_state()

    def closeEvent(self, event) -> None:
        # Mark shutdown before cancelling or waiting. Worker completion signals
        # may already be queued on the GUI thread; their handlers must see the
        # guard before this window (and its footer) can be deleted.
        self._close_pending = True
        self._is_shutting_down = True
        self.job_presentation.shutdown()
        self._startup_check_timer.stop()
        self._post_install_timer.stop()
        self._pending_post_install_model_name = None
        self._pending_model_download_run = None
        self._active_post_install_model_name = None
        self._cancel_faces_workspace_load()
        self._cancel_request_preflight()
        self._cancel_cluster_tag_context_refresh()
        runtime_jobs = (
            (self._startup_maintenance_job, self._startup_maintenance_thread),
            (self._storage_usage_job, self._storage_usage_thread),
            (self._storage_clear_job, self._storage_clear_thread),
            (self._post_install_model_job, self._post_install_model_thread),
            (self._startup_readiness_job, self._startup_readiness_thread),
            (self._faces_init_job, self._faces_init_thread),
            (self._names_init_job, self._names_init_thread),
            (self._tag_suggestion_job, self._tag_suggestion_thread),
            (self._gallery_discovery_job, self._gallery_discovery_thread),
        )
        for job, thread in runtime_jobs:
            if job is None:
                continue
            self._retain_async_refs(job, thread)
            try:
                job.cancel()
            except Exception:
                pass
        self._storage_usage_job = None
        self._storage_usage_thread = None
        self._startup_maintenance_job = None
        self._startup_maintenance_thread = None
        self._storage_clear_job = None
        self._storage_clear_thread = None
        self._post_install_model_job = None
        self._post_install_model_thread = None
        self._startup_readiness_job = None
        self._startup_readiness_thread = None
        self._startup_readiness_job_id = None
        self._faces_init_job = None
        self._faces_init_thread = None
        self._names_init_job = None
        self._names_init_thread = None
        self._tag_suggestion_job = None
        self._tag_suggestion_thread = None
        self._tag_suggestion_job_id = None
        self._preflight_job = None
        self._preflight_thread = None
        # Tests create a disposable runtime root per case.  Their teardown
        # releases that root immediately after close(), so it must drain the
        # same event-driven workers deterministically.  Interactive closes
        # remain non-blocking and retry from the event loop below.
        drain_timeout_ms = 2500 if bool(os.environ.get("PYTEST_CURRENT_TEST")) else 0
        # Closing is a cancellation/drain state machine.  Never wait for a
        # worker on Qt's event loop: a queued completion needs that same loop
        # to run before the following close attempt can succeed.
        ready_to_close = self._wait_for_thread(self._tag_context_thread, drain_timeout_ms)
        retained_jobs = list(self._retained_async_refs)
        for job, thread in retained_jobs:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            ready_to_close = self._wait_for_thread(thread, drain_timeout_ms) and ready_to_close
        if ready_to_close:
            for job, thread in (*runtime_jobs, *retained_jobs):
                dispose = getattr(job, "dispose", None)
                if callable(dispose):
                    try:
                        dispose()
                    except Exception:
                        pass
                try:
                    if getattr(thread, "_job", None) is job:
                        thread._job = None
                except (AttributeError, RuntimeError):
                    pass
        self._retained_async_refs = []
        ready_to_close = self.model_download_controller.shutdown(drain_timeout_ms) and ready_to_close
        ready_to_close = self.session_controller.shutdown(drain_timeout_ms) and ready_to_close
        try:
            ready_to_close = self.gallery_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            ready_to_close = self.photo_gallery.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        if self.faces_pane is not None:
            try:
                ready_to_close = self.faces_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        if self.names_pane is not None:
            try:
                ready_to_close = self.names_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        if self.tags_pane is not None:
            try:
                ready_to_close = self.tags_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        if self.library_pane is not None:
            try:
                ready_to_close = self.library_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        try:
            ready_to_close = self.cluster_pane.shutdown_jobs(timeout_ms=drain_timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        if not ready_to_close:
            if self._can_update_widget(getattr(self, "footer_bar", None)):
                self.footer_bar.set_status("Waiting for background work to stop before closing.")
            self._set_activity("Stopping background work...")
            if not self._close_retry_scheduled:
                self._close_retry_scheduled = True

                def _retry_close() -> None:
                    self._close_retry_scheduled = False
                    if self._close_pending:
                        self.close()

                QTimer.singleShot(50, _retry_close)
            if self not in _CLOSE_PENDING_WINDOWS:
                _CLOSE_PENDING_WINDOWS.append(self)
            self.hide()
            event.ignore()
            return
        try:
            _CLOSE_PENDING_WINDOWS.remove(self)
        except ValueError:
            pass
        self.settings_store.setValue("workspace/default_view", self._active_workspace)
        self.settings_store.setValue("workspace/faces_mode", self._faces_mode)
        self.settings_store.sync()
        super().closeEvent(event)

    def _mark_shutting_down(self) -> None:
        self._is_shutting_down = True
        self._pending_model_download_run = None
        self._active_post_install_model_name = None
        try:
            self.model_download_controller.cancel()
        except Exception:
            pass
        for timer_name in ("_startup_check_timer", "_post_install_timer"):
            timer = getattr(self, timer_name, None)
            if timer is None:
                continue
            try:
                timer.stop()
            except Exception:
                pass
        for generation_name in (
            "_preflight_generation",
            "_tag_context_generation",
            "_storage_usage_generation",
            "_faces_init_generation",
            "_startup_readiness_generation",
        ):
            if hasattr(self, generation_name):
                try:
                    setattr(self, generation_name, int(getattr(self, generation_name)) + 1)
                except Exception:
                    pass
        for job_name in (
            "_preflight_job",
            "_tag_context_job",
            "_storage_usage_job",
            "_storage_clear_job",
            "_post_install_model_job",
            "_startup_readiness_job",
            "_faces_init_job",
            "_names_init_job",
            "_gallery_discovery_job",
        ):
            job = getattr(self, job_name, None)
            if job is None:
                continue
            try:
                job.cancel()
            except Exception:
                pass
        self._finish_pending_photo_face_tool_requests(error="The application is closing.")


def _restore_cluster_dict(raw: dict[str, object]) -> dict[str, dict[int, list[str]]]:
    restored: dict[str, dict[int, list[str]]] = {}
    for comparison_key, clusters in raw.items():
        restored[str(comparison_key)] = {
            int(cluster_id): [str(path) for path in image_paths]
            for cluster_id, image_paths in dict(clusters).items()
        }
    return restored


def _restore_membership(raw: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    return {
        str(image_path): {str(key): dict(value) for key, value in dict(membership).items()}
        for image_path, membership in dict(raw).items()
    }


def _restore_cluster_summaries(raw: dict[str, object]) -> dict[str, dict[int, ClusterTagSummary]]:
    restored: dict[str, dict[int, ClusterTagSummary]] = {}
    for comparison_key, summaries in dict(raw).items():
        restored[str(comparison_key)] = {
            int(cluster_id): ClusterTagSummary(
                tag_counts=dict(context.get("tag_counts") or {}),
                top_tags=tuple((str(tag), int(count)) for tag, count in list(context.get("top_tags") or [])),
                tagged_image_count=int(context.get("tagged_image_count", 0) or 0),
                unique_tag_count=int(context.get("unique_tag_count", 0) or 0),
            )
            for cluster_id, context in dict(summaries).items()
        }
    return restored


def _restore_cluster_explanations(raw: dict[str, object]) -> dict[str, dict[int, ClusterExplanation]]:
    restored: dict[str, dict[int, ClusterExplanation]] = {}
    for comparison_key, explanations in dict(raw).items():
        restored[str(comparison_key)] = {
            int(cluster_id): ClusterExplanation.from_context(dict(context))
            for cluster_id, context in dict(explanations).items()
        }
    return restored


def _restore_cluster_meanings(raw: dict[str, object]) -> dict[str, dict[int, ClusterMeaning]]:
    restored: dict[str, dict[int, ClusterMeaning]] = {}
    for comparison_key, meanings in dict(raw).items():
        restored[str(comparison_key)] = {
            int(cluster_id): ClusterMeaning.from_context(dict(context))
            for cluster_id, context in dict(meanings).items()
        }
    return restored


def _build_gallery_context(
    *,
    image_tags: tuple[str, ...],
    run_origin: str,
    tag_filter: tuple[str, ...],
    tag_match: str,
) -> dict[str, object]:
    context: dict[str, object] = {"tags": list(image_tags)}
    if run_origin:
        context["run_origin"] = run_origin
    if tag_filter:
        context["tag_filter"] = list(tag_filter)
        context["tag_match"] = tag_match or "Any"
    return context


def run() -> int:
    configure_rotating_logging(RUNTIME_LAYOUT, force=True)
    startup_id = uuid.uuid4().hex[:8]
    LOGGER.info(
        "startup begin | id=%s pid=%s executable=%s cwd=%s runtime_root=%s app_log=%s",
        startup_id,
        os.getpid(),
        sys.executable,
        Path.cwd(),
        RUNTIME_LAYOUT.root,
        RUNTIME_LAYOUT.app_log,
    )
    flush_logging_handlers()
    try:
        app = QApplication.instance() or QApplication([])
        app.setApplicationName(PRODUCTION_DISPLAY_NAME)
        app.setApplicationDisplayName(PRODUCTION_DISPLAY_NAME)
        app.setOrganizationName(PRODUCTION_QSETTINGS_ORG)
        desktop_entry_path = configure_linux_desktop_entry(app, executable_path=Path(sys.executable))
        icon_path = production_icon_path()
        app_icon = production_app_icon()
        if app_icon.isNull():
            LOGGER.warning("app icon unavailable | id=%s expected=%s", startup_id, icon_path or "apps/pyqt_production/assets/app_icon.png")
        else:
            app.setWindowIcon(app_icon)
            LOGGER.info("app icon applied | id=%s path=%s", startup_id, icon_path)
        if desktop_entry_path is not None:
            LOGGER.info("linux desktop entry ready | id=%s path=%s", startup_id, desktop_entry_path)
        LOGGER.info(
            "qt application ready | id=%s app_instance=%s primary_screen=%s",
            startup_id,
            id(app),
            _screen_summary(QApplication.primaryScreen()),
        )
        install_qt_message_handler()
        LOGGER.info("qt diagnostics installed | id=%s log=%s", startup_id, RUNTIME_LAYOUT.qt_diagnostics_log)
        apply_ultra_dark(app)
        LOGGER.info("theme applied | id=%s", startup_id)
        window = ProductionClusterApp(RUNTIME_LAYOUT)
        smoke_scenario = str(os.environ.get(PACKAGED_LAUNCH_SMOKE_ENV) or "").strip()
        if smoke_scenario:
            # The launch verifier always supplies an isolated runtime root. Do
            # not let the first-run modal race its noninteractive smoke path.
            window.settings_store.setValue("setup/completed", True)
            window.settings_store.sync()
        LOGGER.info("window constructed | id=%s title=%s", startup_id, window.windowTitle())
        _show_main_window_on_screen(window)
        LOGGER.info(
            "window shown | id=%s visible=%s maximized=%s geometry=%s frame=%s screen=%s",
            startup_id,
            window.isVisible(),
            window.isMaximized(),
            window.geometry().getRect(),
            window.frameGeometry().getRect(),
            _screen_summary(window.screen()),
        )
        if smoke_scenario:
            QTimer.singleShot(
                0,
                lambda: _complete_packaged_launch_smoke(
                    app,
                    window,
                    scenario=smoke_scenario,
                    report_path=str(os.environ.get(PACKAGED_LAUNCH_REPORT_ENV) or ""),
                ),
            )
        LOGGER.info("event loop entered | id=%s", startup_id)
        flush_logging_handlers()
        exit_code = int(app.exec())
        LOGGER.info("event loop exited | id=%s exit_code=%s", startup_id, exit_code)
        return exit_code
    except Exception:
        LOGGER.exception("startup failed | id=%s", startup_id)
        raise
    finally:
        flush_logging_handlers()


def _complete_packaged_launch_smoke(
    app: QApplication,
    window: ProductionClusterApp,
    *,
    scenario: str,
    report_path: str,
) -> None:
    """Write isolated packaged-launch evidence then end the verifier process."""
    destination = Path(report_path).expanduser().resolve() if report_path else None
    dialog = None
    checks: dict[str, bool] = {}
    error = ""
    try:
        if scenario != PACKAGED_LAUNCH_SMOKE_SCENARIO:
            raise ValueError(f"unsupported packaged-launch smoke scenario: {scenario}")
        if destination is None:
            raise ValueError(f"{PACKAGED_LAUNCH_REPORT_ENV} is required for packaged launch smoke")
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite packaged-launch report: {destination}")
        dialog = window._create_settings_dialog(parent=window)
        dialog.show()
        app.processEvents()
        storage_tab_names = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
        expected_categories = set(
            window.cache_maintenance_service.generated_storage_targets(runtime_layout=window.runtime_layout)
        )
        generated_categories = set()
        try:
            generated_categories = set(
                window.cache_maintenance_service.describe_generated_storage(runtime_layout=window.runtime_layout).target_bytes
            )
        except OSError:
            generated_categories = set()
        checks = {
            "frozen_executable": bool(getattr(sys, "frozen", False)),
            "window_icon": not app.windowIcon().isNull(),
            "settings_dialog_visible": bool(dialog.isVisible()),
            "storage_tab": "Storage" in storage_tab_names,
            "generated_storage_controls": all(
                getattr(dialog, attribute, None) is not None
                for attribute in (
                    "refresh_cache_usage_button",
                    "clear_cache_button",
                    "clear_runtime_temp_button",
                    "clear_face_storage_button",
                    "clear_model_caches_button",
                    "clear_logs_button",
                    "clear_runtime_reports_button",
                    "clear_model_assets_button",
                )
            ),
            "generated_storage_categories_match_service": generated_categories == expected_categories,
        }
    except Exception as exc:
        error = str(exc)
        LOGGER.exception("packaged launch smoke failed")
    finally:
        if destination is not None and not destination.exists():
            payload = {
                "report_version": "1",
                "scenario": f"packaged-launch-{scenario}",
                "validation": "PASS" if checks and all(checks.values()) and not error else "FAIL",
                "checks": checks,
                "error": error,
            }
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            except OSError:
                LOGGER.exception("could not write packaged launch smoke report")
        if dialog is not None:
            dialog.close()
        # This verifier process owns an isolated runtime and exits immediately.
        # Do not route it through the interactive close retry path, whose
        # contract is to wait for user-visible background work to finish.
        window.hide()
        app.quit()


def _show_main_window_on_screen(window: ProductionClusterApp) -> None:
    """Open the production shell on a visible desktop area.

    Windows can restore Qt top-level windows with negative coordinates after
    monitor changes. For a production app, hidden/off-screen startup is a
    launch failure, so prefer a maximized on-screen window and re-check once
    the event loop has processed native placement.
    """
    screen = QApplication.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        logical = screen.geometry()
        if not _screen_geometry_supported(logical):
            window.show_unsupported_resolution_view((int(logical.width()), int(logical.height())))
        else:
            window.apply_screen_geometry_constraints((int(available.width()), int(available.height())))
        target_width = min(max(1440, int(available.width() * 0.96)), available.width())
        target_height = min(max(900, int(available.height() * 0.96)), available.height())
        window.resize(target_width, target_height)
        window.move(available.left(), available.top())

    window.showMaximized()
    window.raise_()
    window.activateWindow()
    _ensure_window_visible(window)
    _schedule_window_visibility_check(window, 0)
    _schedule_window_visibility_check(window, 250)


def _schedule_window_visibility_check(window: ProductionClusterApp, delay_ms: int) -> None:
    timers = getattr(window, "_visibility_recheck_timers", None)
    if not isinstance(timers, list):
        timers = []
        setattr(window, "_visibility_recheck_timers", timers)
    timer = QTimer(window)
    timer.setSingleShot(True)

    def _run(timer_ref=timer) -> None:
        try:
            _ensure_window_visible(window)
        finally:
            try:
                timers.remove(timer_ref)
            except ValueError:
                pass
            timer_ref.deleteLater()

    timer.timeout.connect(_run)
    timers.append(timer)
    timer.start(max(int(delay_ms), 0))


def _ensure_window_visible(window: ProductionClusterApp) -> None:
    if window.isMaximized() or window.isFullScreen():
        return
    screen = window.screen() or QApplication.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    frame = window.frameGeometry()
    if available.contains(frame.center()) and frame.top() >= available.top():
        return
    window.move(available.left(), available.top())
    LOGGER.warning(
        "Repositioned production window onto visible desktop: screen=%s frame=%s",
        available.getRect(),
        window.frameGeometry().getRect(),
    )


def _screen_geometry_supported(geometry) -> bool:
    if geometry is None:
        return True
    try:
        width = int(geometry.width())
        height = int(geometry.height())
        long_edge = max(width, height)
        short_edge = min(width, height)
        return long_edge >= max(MIN_SUPPORTED_SCREEN_WIDTH, MIN_SUPPORTED_SCREEN_HEIGHT) and short_edge >= min(
            MIN_SUPPORTED_SCREEN_WIDTH,
            MIN_SUPPORTED_SCREEN_HEIGHT,
        )
    except Exception:
        return True


def _screen_summary(screen: object | None) -> str:
    if screen is None:
        return "none"
    name = getattr(screen, "name", lambda: "")()
    available = getattr(screen, "availableGeometry", lambda: None)()
    if available is None:
        return str(name or "unknown")
    return f"{name or 'unknown'} available={available.getRect()}"


if __name__ == "__main__":
    raise SystemExit(run())
