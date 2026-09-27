from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QElapsedTimer, Qt, QThread, QSettings, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMainWindow, QPushButton, QSplitter, QStackedWidget, QVBoxLayout, QWidget

from app.feature_registry import FeatureModule
from app.services.cache_maintenance import CacheClearResult, CacheMaintenanceService, CacheUsageSummary
from app.services.cluster_explanations import ClusterExplanation
from app.services.cluster_meanings import ClusterMeaning
from app.services.clustering_pipeline import ClusteringPipelineService, ClusteringRequest, MultiBackendClusteringResult, RunCancelled
from app.services.image_tags import ClusterTagSummary, ImageTagService
from app.services.performance_dashboard import PerformanceDashboardService
from app.services.photo_metadata import PhotoMetadataService
from app.services.saved_searches import SavedSearchService
from infra.logging_config import configure_logging, get_logger
from infra.performance import detect_system_resources, select_performance_profile
from infra.qt_diagnostics import append_qt_diagnostic, install_qt_message_handler
from infra.runtime import RuntimeCapabilityService
from infra.settings import get_settings
from ml.embeddings import EmbeddingService, ModelManager
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown
from ui.cluster_pane import ClusterPane
from ui.error_mbox import errorBox, infoBox
from ui.footer_bar import WorkspaceFooter
from ui.job_manager import JobManager
from ui.job_presentation import JobPresentationController
from ui.job_widgets import JobIndicatorWidget
from ui.recent_folders import RecentFolderHistory
from ui.mode_panes import ClusteringOptionsPane, SourcePane
from ui.runtime_widgets import RuntimeBadge
from ui.theme import apply_ultra_dark

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService
    from ui.names_pane import NamesPane

LOGGER = get_logger(__name__)

_FACE_SEARCH_MODULE = None


def _face_search_api():
    global _FACE_SEARCH_MODULE
    if _FACE_SEARCH_MODULE is None:
        from app.services import face_search as face_search_module

        _FACE_SEARCH_MODULE = face_search_module
    return _FACE_SEARCH_MODULE


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in {"1", "true", "yes", "on"}


def _configure_linux_dialog_fallback() -> None:
    if not sys.platform.startswith("linux"):
        return
    if _env_flag("CLUSTERLENS_USE_NATIVE_FILE_DIALOGS"):
        LOGGER.info("Native Qt file dialogs explicitly enabled by CLUSTERLENS_USE_NATIVE_FILE_DIALOGS.")
        return
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    LOGGER.info(
        "Using Qt non-native file dialogs on Linux. "
        "Set CLUSTERLENS_USE_NATIVE_FILE_DIALOGS=1 to restore native dialogs."
    )


class ClusteringWorker(QThread):
    progress_changed = pyqtSignal(int, str)
    completed = pyqtSignal(object, dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, pipeline: ClusteringPipelineService, request: ClusteringRequest):
        super().__init__()
        self.pipeline = pipeline
        self.request = request
        self._cancel_requested = False
        self.setObjectName(f"ClusteringWorker:{id(self):x}")

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self):
        append_qt_diagnostic(f"[ThreadStart] {self.objectName()} request_dir={self.request.directory}")
        try:
            clusters, metrics = self.pipeline.run(
                self.request,
                progress_callback=self._emit_progress,
                cancel_check=lambda: self._cancel_requested,
            )
        except RunCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(clusters, metrics)
        finally:
            append_qt_diagnostic(f"[ThreadFinish] {self.objectName()} cancel_requested={self._cancel_requested}")

    def _emit_progress(self, value: int, status: str) -> None:
        self.progress_changed.emit(value, status)


@dataclass(frozen=True)
class ClusterTagContextPayload:
    image_tags_by_path: dict[str, tuple[str, ...]]
    cluster_tag_summaries: dict[str, dict[int, ClusterTagSummary]]
    gallery_extra_context_by_path: dict[str, dict[str, object]]


class ClusterGalleryApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.settings_store = QSettings(self.settings.app_name, self.settings.app_name)
        self.recent_folder_history = RecentFolderHistory(self.settings_store)
        self.runtime_service = RuntimeCapabilityService()
        self.job_manager = JobManager(self)
        self.performance_dashboard_service = PerformanceDashboardService()
        self.system_resources = detect_system_resources()
        self.performance_profile = select_performance_profile(self._preferred_performance_profile(), self.system_resources)
        self.cluster_data: dict[str, dict[int, list[str]]] = {}
        self.membership_by_image: dict[str, dict[str, dict[str, object]]] = {}
        self.metrics_by_backend: dict[str, dict[str, object]] = {}
        self.cluster_explanations: dict[str, dict[int, ClusterExplanation]] = {}
        self.cluster_meanings: dict[str, dict[int, ClusterMeaning]] = {}
        self.image_tags_by_path: dict[str, tuple[str, ...]] = {}
        self.cluster_tag_summaries: dict[str, dict[int, ClusterTagSummary]] = {}
        self.gallery_extra_context_by_path: dict[str, dict[str, object]] = {}
        self.current_run_origin = "folder"
        self.current_tag_filter: tuple[str, ...] = ()
        self.current_tag_match = ""
        self.worker: ClusteringWorker | None = None
        self._clustering_job_id: int | None = None
        self._tag_context_thread = None
        self._tag_context_job = None
        self._tag_context_job_id: int | None = None
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._async_thread_roles: dict[object, tuple[str, object | None]] = {}
        self._tag_context_generation = 0
        self._warmup_thread = None
        self._warmup_job = None
        self._warmup_job_id: int | None = None
        self._warmup_signature: tuple[str, bool, str, str] | None = None
        self._warmup_cache: set[tuple[str, bool, str, str]] = set()
        self._warmup_state = "idle"
        self.gallery_timer = QElapsedTimer()
        self.last_run_metrics: dict[str, object] = {}
        self._pane_restore_widths = {"source": 250, "controls": 430, "details": 520}
        self._pane_visibility = {"source": True, "controls": True, "details": True}
        self._advanced_pane_visibility = {"source": True, "controls": True, "details": True}
        self._advanced_cluster_splitter_sizes = [620, 280]
        self._clustering_mode = "basic"
        self._faces_mode = "basic"
        self._active_workspace = "clustering"
        self._face_service_cache: dict[tuple[str, str, str, str], FaceIndexService] = {}
        self._main_gallery_context_overrides: dict[str, dict[str, object]] = {}
        self.execution_policy = self.runtime_service.select_policy(self._preferred_execution_mode())
        self.feature_modules = self._build_feature_modules()
        self._build_services(reset_face_session=True)
        self.setWindowTitle("Image Clustering Workspace")
        self.resize(1920, 1080)
        self.init_ui()
        self.restore_ui_state()
        self.update_runtime_status()

    def _build_feature_modules(self) -> dict[str, FeatureModule]:
        return {
            "clustering": FeatureModule(
                feature_id="clustering",
                label="Clustering",
                enabled=True,
                build_panel=lambda parent: ClusteringOptionsPane(parent),
                attach=lambda app: app._attach_clustering_feature(),
                detach=lambda app: None,
            ),
            "face_search": FeatureModule(feature_id="face_search", label="Face Search", enabled=False),
            "duplicate_search": FeatureModule(feature_id="duplicate_search", label="Duplicate Search", enabled=False),
            "image_similarity": FeatureModule(feature_id="image_similarity", label="Image Similarity", enabled=False),
            "text_search": FeatureModule(feature_id="text_search", label="Text Search", enabled=False),
            "review": FeatureModule(feature_id="review", label="Review", enabled=False),
        }

    def _build_services(self, *, reset_face_session: bool = False) -> None:
        self.performance_profile = select_performance_profile(self._preferred_performance_profile(), self.system_resources)
        self.model_manager = ModelManager(
            execution_policy=self.execution_policy,
            runtime_service=self.runtime_service,
            performance_profile=self.performance_profile,
        )
        self.embedding_service = EmbeddingService(
            model_manager=self.model_manager,
            performance_profile=self.performance_profile,
        )
        self.pipeline = ClusteringPipelineService(embedding_service=self.embedding_service)
        self.photo_metadata_service = PhotoMetadataService()
        self.image_tag_service = ImageTagService()
        self.saved_search_service = SavedSearchService()
        self.cache_maintenance_service = CacheMaintenanceService(self.settings)
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
            for mode in ("human", "dog", "cat")
        }
        self.face_services_session = {
            mode: self._face_service_for_pipeline(
                "session",
                mode,
                self._preferred_face_detector(mode),
                self._preferred_face_embedder(mode),
            )
            for mode in ("human", "dog", "cat")
        }
        self.face_service_global = self.face_services_global["human"]
        self.face_service_session = self.face_services_session["human"]

    def init_ui(self) -> None:
        from ui.gallery_pane import GalleryPane
        from ui.names_pane import NamesPane
        from ui.search_pane import SearchPane

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        root_layout = QVBoxLayout(self.central_widget)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        root_layout.addWidget(self._build_toolbar())
        root_layout.addWidget(self._build_startup_tutorial_panel())

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(6)
        root_layout.addWidget(self.main_splitter, stretch=1)

        self.source_pane = SourcePane(self)
        self.source_pane.directory_changed.connect(self.on_directory_changed)
        self.source_pane.recent_folder_remove_requested.connect(self._remove_recent_folder)
        self.source_pane.recent_folders_clear_requested.connect(self._clear_recent_folders)
        self.source_pane.hide_requested.connect(lambda: self._set_pane_visible("source", False))

        clustering_feature = self.feature_modules["clustering"]
        self.clustering_pane = clustering_feature.build_panel(self)
        clustering_feature.attach(self)
        self.clustering_pane.hide_requested.connect(lambda: self._set_pane_visible("controls", False))

        self.gallery_pane = GalleryPane(self)
        self.gallery_pane.job_manager = self.job_manager
        self.gallery_pane.metadata_service = self.photo_metadata_service
        self.gallery_pane.image_tag_service = self.image_tag_service
        self.gallery_pane.review_action_mode = "disabled"
        self.gallery_pane.inspector_context_provider = self._main_gallery_context_for_path
        self.gallery_pane.face_edit_service_provider = lambda: self.face_service_global
        self.gallery_pane.face_edit_saved_callback = self._on_main_gallery_face_labels_changed
        self.gallery_pane.first_paint_ready.connect(self.on_gallery_first_paint)
        self.gallery_pane.image_selected.connect(self.on_center_gallery_image_selected)
        self.gallery_pane.paths_removed.connect(self.on_gallery_paths_removed)
        self.gallery_pane.paths_renamed.connect(self.on_gallery_paths_renamed)
        self.gallery_pane.metadata_changed.connect(self.on_gallery_metadata_changed)

        self.cluster_pane = ClusterPane(self)
        self.cluster_pane.cluster_selected.connect(self.update_gallery)
        self.cluster_pane.recluster_requested.connect(self.recluster_selected_cluster)
        self.cluster_pane.selection_target_changed.connect(lambda _target: self.gallery_pane.refresh_selection_target_hint())
        self.cluster_pane.hide_requested.connect(lambda: self._set_pane_visible("details", False))
        self.gallery_pane.set_action_target_provider(self.cluster_pane.current_selection_target)

        self.cluster_right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.cluster_right_splitter.setChildrenCollapsible(False)
        self.cluster_right_splitter.setHandleWidth(4)
        self.cluster_right_splitter.addWidget(self.cluster_pane)
        self.cluster_right_splitter.setStretchFactor(0, 1)

        self.workspace_stack = QStackedWidget(self)
        self.clustering_workspace = QWidget(self.workspace_stack)
        clustering_layout = QHBoxLayout(self.clustering_workspace)
        clustering_layout.setContentsMargins(0, 0, 0, 0)
        clustering_layout.setSpacing(0)
        self.clustering_splitter = QSplitter(Qt.Orientation.Horizontal, self.clustering_workspace)
        self.clustering_splitter.setChildrenCollapsible(False)
        self.clustering_splitter.setHandleWidth(6)
        self.clustering_splitter.addWidget(self.clustering_pane)
        self.clustering_splitter.addWidget(self.gallery_pane)
        self.clustering_splitter.addWidget(self.cluster_right_splitter)
        self.clustering_splitter.setStretchFactor(0, 3)
        self.clustering_splitter.setStretchFactor(1, 8)
        self.clustering_splitter.setStretchFactor(2, 3)
        clustering_layout.addWidget(self.clustering_splitter)

        self.faces_pane = SearchPane(
            self.workspace_stack,
            face_service_global=self.face_service_global,
            face_service_session=self.face_service_session,
            face_services_global=self.face_services_global,
            face_services_session=self.face_services_session,
            enabled_tabs=["All Faces", "Faces In Folder", "Face Search", "Identities"],
            saved_search_service=self.saved_search_service,
        )
        self.faces_pane.set_face_service_provider(self._face_service_for_pipeline)
        self.faces_pane.current_directory_provider = lambda: self.source_pane.selected_directory
        self.faces_pane.clustering_filter_state_provider = self._current_clustering_filter_state
        self.faces_pane.use_onnx_provider = lambda: bool(self.clustering_pane.onnx_checkbox.isChecked())
        self.faces_pane.job_manager = self.job_manager
        self.faces_pane.results_gallery.job_manager = self.job_manager
        self.faces_pane.results_gallery.metadata_service = self.photo_metadata_service
        self.faces_pane.results_gallery.image_tag_service = self.image_tag_service
        self.faces_pane.open_in_gallery_requested.connect(self._open_face_results_in_main_gallery)
        self.faces_pane.append_to_gallery_requested.connect(self._append_face_results_to_main_gallery)
        self.faces_pane.saved_clustering_filter_requested.connect(self._run_saved_clustering_filter)
        self.faces_pane.open_face_model_settings_requested.connect(self.open_settings_dialog)
        self.faces_pane.source_folder_changed.connect(self._set_shared_source_folder)
        self.faces_pane.recent_folder_remove_requested.connect(self._remove_recent_folder)
        self.faces_pane.recent_folders_clear_requested.connect(self._clear_recent_folders)
        self.faces_pane.configure_face_pipeline_options(
            self._face_model_root(),
            self._preferred_face_pipeline_defaults(),
            refresh=False,
        )
        self.faces_pane.set_active_face_mode(self._preferred_face_mode(), refresh=False)
        self._refresh_recent_folder_menus()

        self.names_pane = NamesPane(
            lambda: self.face_service_global,
            self.workspace_stack,
            job_manager=self.job_manager,
        )
        self.names_pane.gallery.job_manager = self.job_manager
        self.names_pane.gallery.metadata_service = self.photo_metadata_service
        self.names_pane.gallery.image_tag_service = self.image_tag_service
        self.faces_pane.face_labels_changed.connect(self.names_pane.refresh_names)
        self.names_pane.face_labels_changed.connect(self._on_names_face_labels_changed)

        self.workspace_stack.addWidget(self.clustering_workspace)
        self.workspace_stack.addWidget(self.faces_pane)
        self.workspace_stack.addWidget(self.names_pane)

        self.source_pane.setMinimumWidth(220)
        self.source_pane.setMaximumWidth(280)
        self.clustering_pane.setMinimumWidth(380)
        self.clustering_pane.setMaximumWidth(460)
        self.gallery_pane.setMinimumWidth(760)
        self.cluster_right_splitter.setMinimumWidth(440)
        self.cluster_right_splitter.setMaximumWidth(680)
        self.workspace_stack.setMinimumWidth(1040)
        self.faces_pane.setMinimumWidth(1040)
        self.names_pane.setMinimumWidth(1040)

        self.main_splitter.addWidget(self.source_pane)
        self.main_splitter.addWidget(self.workspace_stack)
        self.main_splitter.setStretchFactor(0, 2)
        self.main_splitter.setStretchFactor(1, 11)

        self.footer_bar = WorkspaceFooter(self)
        self.job_presentation = JobPresentationController(self.job_manager, self.footer_bar, self)
        self.job_manager.job_added.connect(lambda _job_id: self._update_metrics_overlay())
        self.job_manager.job_updated.connect(lambda _job_id: self._update_metrics_overlay())
        self.job_manager.job_finished.connect(lambda _job_id: self._update_metrics_overlay())
        root_layout.addWidget(self.footer_bar)
        self.statusBar().hide()
        self.apply_workspace_preferences()

    def _attach_clustering_feature(self) -> None:
        self.clustering_pane.run_requested.connect(self.run_clustering)
        self.clustering_pane.cancel_requested.connect(self.cancel_clustering)
        self.source_pane.run_requested.connect(self.run_clustering)
        self.source_pane.cancel_requested.connect(self.cancel_clustering)

    def _build_startup_tutorial_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        self.startup_tutorial_label = QLabel(
            "Start with a source folder, run clustering, then review results in the gallery. "
            "Use Faces for local identity review, Settings for CPU/GPU and storage controls, "
            "and User Flows for step-by-step workflows. Generated app data is local and can be "
            "cleared from storage controls; source images are changed only by explicit file actions."
        )
        self.startup_tutorial_label.setWordWrap(True)
        self.startup_tutorial_dismiss_button = QPushButton("Dismiss Tutorial")
        self.startup_tutorial_dismiss_button.clicked.connect(self._dismiss_startup_tutorial)
        layout.addWidget(self.startup_tutorial_label)
        layout.addWidget(self.startup_tutorial_dismiss_button)
        dismissed = self.settings_store.value("tutorial/app_first_run_dismissed", False, bool)
        panel.setVisible(not bool(dismissed))
        self.startup_tutorial_panel = panel
        return panel

    def _show_startup_tutorial(self) -> None:
        if hasattr(self, "startup_tutorial_panel"):
            self.startup_tutorial_panel.setVisible(True)

    def _dismiss_startup_tutorial(self) -> None:
        self.settings_store.setValue("tutorial/app_first_run_dismissed", True)
        if hasattr(self, "startup_tutorial_panel"):
            self.startup_tutorial_panel.setVisible(False)

    def _build_toolbar(self) -> QWidget:
        bar = QWidget(self)
        row = QHBoxLayout(bar)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.feature_label = QLabel("Feature: Clustering")
        self.clustering_workspace_button = QPushButton("Clustering")
        self.clustering_workspace_button.setCheckable(True)
        self.clustering_workspace_button.setProperty("nav", True)
        self.clustering_workspace_button.clicked.connect(lambda: self.set_active_workspace("clustering"))
        self.faces_workspace_button = QPushButton("Faces")
        self.faces_workspace_button.setCheckable(True)
        self.faces_workspace_button.setProperty("nav", True)
        self.faces_workspace_button.clicked.connect(lambda: self.set_active_workspace("faces"))
        self.names_workspace_button = QPushButton("Names")
        self.names_workspace_button.setCheckable(True)
        self.names_workspace_button.setProperty("nav", True)
        self.names_workspace_button.setToolTip("Browse durable saved names and the photos containing their labeled faces.")
        self.names_workspace_button.clicked.connect(lambda: self.set_active_workspace("names"))
        self.current_folder_label = QLabel("No folder selected")
        self.basic_mode_button = QPushButton("Basic")
        self.basic_mode_button.setCheckable(True)
        self.basic_mode_button.clicked.connect(lambda: self.set_active_workspace_mode("basic"))
        self.advanced_mode_button = QPushButton("Advanced")
        self.advanced_mode_button.setCheckable(True)
        self.advanced_mode_button.clicked.connect(lambda: self.set_active_workspace_mode("advanced"))
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
        self.runtime_badge = RuntimeBadge(self)
        self.tutorial_button = QPushButton("Tutorial")
        self.tutorial_button.setToolTip("Show the first-run ClusterLens tutorial.")
        self.user_flows_button = QPushButton("User Flows")
        self.user_flows_button.setToolTip("Open the bundled ClusterLens user-flow guide.")
        self.settings_button = QPushButton("Settings")
        self.job_indicator = JobIndicatorWidget(self.job_manager, self)
        self.tutorial_button.clicked.connect(self._show_startup_tutorial)
        self.user_flows_button.clicked.connect(self.open_user_flows)
        self.settings_button.clicked.connect(self.open_settings_dialog)

        row.addWidget(self.feature_label)
        row.addWidget(self.clustering_workspace_button)
        row.addWidget(self.faces_workspace_button)
        row.addWidget(self.names_workspace_button)
        row.addWidget(self.current_folder_label, stretch=1)
        row.addWidget(self.basic_mode_button)
        row.addWidget(self.advanced_mode_button)
        row.addWidget(self.source_toggle)
        row.addWidget(self.controls_toggle)
        row.addWidget(self.details_toggle)
        row.addWidget(self.runtime_badge)
        row.addWidget(self.tutorial_button)
        row.addWidget(self.user_flows_button)
        row.addWidget(self.settings_button)
        row.addWidget(self.job_indicator)
        return bar

    def _preferred_execution_mode(self) -> str:
        return self.settings_store.value("runtime/preferred_mode", self.settings.preferred_execution_mode, str)

    def _preferred_performance_profile(self) -> str:
        return self.settings_store.value("performance/profile", self.settings.default_performance_profile, str)

    def _preferred_workspace(self) -> str:
        return self._normalize_workspace_id(self.settings_store.value("workspace/default_view", "clustering", str))

    def _preferred_faces_ui_mode(self) -> str:
        return self._normalize_ui_mode(self.settings_store.value("workspace/faces_mode", "basic", str))

    def _preferred_face_mode(self) -> str:
        face_search = _face_search_api()
        return face_search.normalize_face_mode(self.settings_store.value("faces/default_mode", "human", str))

    def _face_model_root(self) -> str:
        root = str(self.settings_store.value("faces/model_root", "", str) or "").strip()
        if root:
            return root
        return str(self.settings_store.value("faces/animal_model_root", "", str) or "").strip()

    def _animal_model_root(self) -> str:
        return self._face_model_root()

    def _preferred_face_detector(self, mode: str) -> str:
        return self._preferred_face_pipeline_ids(mode)[0]

    def _preferred_face_embedder(self, mode: str) -> str:
        return self._preferred_face_pipeline_ids(mode)[1]

    def _preferred_face_pipeline_ids(self, mode: str) -> tuple[str, str]:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        configured_detector = str(
            self.settings_store.value(f"faces/default_detector/{mode_id}", "", str) or ""
        ).strip()
        configured_embedder = str(
            self.settings_store.value(f"faces/default_embedder/{mode_id}", "", str) or ""
        ).strip()
        if mode_id == "human":
            configured_detector = configured_detector or face_search.DEFAULT_HUMAN_FACE_DETECTOR_ID
            configured_embedder = configured_embedder or face_search.DEFAULT_HUMAN_FACE_EMBEDDER_ID
        return face_search.resolve_ready_face_pipeline_ids(
            self._face_model_root(),
            mode_id,
            configured_detector,
            configured_embedder,
        )

    def _preferred_face_detector_score_threshold(self, mode: str) -> float:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        if mode_id == "human" and not self.settings_store.contains(f"faces/detector_score_threshold/{mode_id}"):
            default_profile = face_search.face_model_profile_config(mode_id, face_search.DEFAULT_HUMAN_FACE_PROFILE_ID)
            return float((default_profile or {}).get("score_threshold", 0.35))
        return float(self.settings_store.value(f"faces/detector_score_threshold/{mode_id}", face_search.DEFAULT_FACE_SCORE_THRESHOLD, float))

    def _preferred_face_max_detections(self, mode: str) -> int:
        face_search = _face_search_api()
        mode_id = face_search.normalize_face_mode(mode)
        return max(1, int(self.settings_store.value(f"faces/max_detections/{mode_id}", face_search.DEFAULT_FACE_MAX_DETECTIONS, int)))

    def _preferred_face_pipeline_defaults(self) -> dict[str, dict[str, object]]:
        defaults: dict[str, dict[str, object]] = {}
        for mode in ("human", "dog", "cat"):
            detector_id, embedder_id = self._preferred_face_pipeline_ids(mode)
            defaults[mode] = {
                "detector_id": detector_id,
                "embedder_id": embedder_id,
                "score_threshold": self._preferred_face_detector_score_threshold(mode),
                "max_detections": self._preferred_face_max_detections(mode),
            }
        return defaults

    def _face_pipeline_settings_snapshot(self) -> dict[str, object]:
        snapshot: dict[str, object] = {
            "faces/model_root": self._face_model_root(),
            "faces/default_mode": self._preferred_face_mode(),
        }
        for mode in ("human", "dog", "cat"):
            snapshot[f"faces/default_detector/{mode}"] = self._preferred_face_detector(mode)
            snapshot[f"faces/default_embedder/{mode}"] = self._preferred_face_embedder(mode)
            snapshot[f"faces/detector_score_threshold/{mode}"] = self._preferred_face_detector_score_threshold(mode)
            snapshot[f"faces/max_detections/{mode}"] = self._preferred_face_max_detections(mode)
        return snapshot

    def _clear_face_session_dbs(self) -> None:
        for path in self.settings.cache_dir.glob("face_search*_session*.db*"):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _face_pipeline_slug(detector_id: str, embedder_id: str) -> str:
        face_search = _face_search_api()
        detector = face_search.normalize_face_component_id(detector_id, "detector")
        embedder = face_search.normalize_face_component_id(embedder_id, "embedder")
        return f"{detector}__{embedder}"

    def _face_db_path_for_pipeline(self, scope: str, mode: str, detector_id: str, embedder_id: str) -> Path:
        face_search = _face_search_api()
        scope_id = "session" if str(scope or "").strip().lower() == "session" else "global"
        mode_id = face_search.normalize_face_mode(mode)
        detector = face_search.normalize_face_component_id(detector_id, face_search.default_face_detector_id(self._face_model_root(), mode_id))
        embedder = face_search.normalize_face_component_id(embedder_id, face_search.default_face_embedder_id(self._face_model_root(), mode_id))
        if mode_id == "human" and detector == face_search.BUILTIN_HUMAN_DETECTOR_ID and embedder == face_search.BUILTIN_HUMAN_EMBEDDER_ID:
            return self.settings.cache_dir / ("face_search_session.db" if scope_id == "session" else "face_search_index.db")
        if mode_id == "dog" and detector == face_search.LEGACY_DEFAULT_BUNDLE_ID and embedder == face_search.LEGACY_DEFAULT_BUNDLE_ID:
            return self.settings.cache_dir / ("face_search_dog_session.db" if scope_id == "session" else "face_search_dog.db")
        if mode_id == "cat" and detector == face_search.LEGACY_DEFAULT_BUNDLE_ID and embedder == face_search.LEGACY_DEFAULT_BUNDLE_ID:
            return self.settings.cache_dir / ("face_search_cat_session.db" if scope_id == "session" else "face_search_cat.db")
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
        mode_id = face_search.normalize_face_mode(mode)
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
                db_path=self._face_db_path_for_pipeline(scope_id, mode_id, detector, embedder),
            )
            self._face_service_cache[key] = service
        return service

    @staticmethod
    def _normalize_workspace_id(value: str) -> str:
        text = str(value or "").strip().lower()
        if text in {"faces", "face_search", "face search"}:
            return "faces"
        if text in {"names", "people", "saved names"}:
            return "names"
        return "clustering"

    def _load_json(self, key: str) -> dict[str, object]:
        raw = self.settings_store.value(key, "", str)
        if not raw:
            return {}
        try:
            return dict(json.loads(raw))
        except Exception:
            return {}

    def apply_workspace_preferences(self) -> None:
        self.performance_profile = select_performance_profile(self._preferred_performance_profile(), self.system_resources)
        thumbnail_size = int(self.settings_store.value("gallery/thumbnail_size", self.settings.thumbnail_size, int))
        self.gallery_pane.apply_view_preferences(
            thumbnail_size=thumbnail_size,
            worker_count=self.performance_profile.thumbnail_workers,
            prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
            pixmap_cache_size=self.performance_profile.pixmap_cache_size,
            qimage_cache_size=self.performance_profile.qimage_cache_size,
        )
        self.faces_pane.results_gallery.apply_view_preferences(
            thumbnail_size=thumbnail_size,
            worker_count=self.performance_profile.thumbnail_workers,
            prefetch_rows=self.performance_profile.thumbnail_prefetch_rows,
            pixmap_cache_size=self.performance_profile.pixmap_cache_size,
            qimage_cache_size=self.performance_profile.qimage_cache_size,
        )
        self.names_pane.gallery.apply_view_preferences(
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
        self.runtime_badge.setVisible(self.settings_store.value("runtime/show_badge", self.settings.show_runtime_badge, bool))
        self._reflow_main_splitter(force=False)

    def _sync_mode_buttons(self) -> None:
        active_mode = self._clustering_mode if self._active_workspace == "clustering" else self._faces_mode
        mapping = {
            "basic": self.basic_mode_button,
            "advanced": self.advanced_mode_button,
        }
        for key, button in mapping.items():
            button.blockSignals(True)
            button.setChecked(active_mode == key)
            button.blockSignals(False)

    def _sync_workspace_buttons(self) -> None:
        mapping = {
            "clustering": self.clustering_workspace_button,
            "faces": self.faces_workspace_button,
            "names": self.names_workspace_button,
        }
        for key, button in mapping.items():
            button.blockSignals(True)
            button.setChecked(self._active_workspace == key)
            button.blockSignals(False)

    def _source_pane_visible(self) -> bool:
        if self._active_workspace == "names":
            return False
        if self._active_workspace == "faces":
            return bool(self._advanced_pane_visibility["source"])
        if self._clustering_mode == "basic":
            return True
        return bool(self._pane_visibility["source"])

    def _refresh_workspace_ui(self) -> None:
        is_clustering = self._active_workspace == "clustering"
        is_faces = self._active_workspace == "faces"
        is_names = self._active_workspace == "names"
        workspace = self.clustering_workspace if is_clustering else (self.faces_pane if is_faces else self.names_pane)
        self.workspace_stack.setCurrentWidget(workspace)
        if is_faces:
            self.faces_pane.ensure_faces_workspace_loaded()
        if is_names:
            self.names_pane.refresh_names()
        self.faces_pane.set_ui_mode(self._faces_mode)
        self.source_pane.set_basic_mode(
            is_clustering and self._clustering_mode == "basic",
            running=bool(self.worker and self.worker.isRunning()),
        )
        self.basic_mode_button.setVisible(not is_names)
        self.advanced_mode_button.setVisible(not is_names)
        self.source_toggle.setVisible(not is_names and ((not is_clustering) or self._clustering_mode == "advanced"))
        self.controls_toggle.setVisible(is_clustering and self._clustering_mode == "advanced")
        self.details_toggle.setVisible(is_clustering and self._clustering_mode == "advanced")
        if is_clustering:
            mode_label = "Basic" if self._clustering_mode == "basic" else "Advanced"
            self.feature_label.setText(f"Feature: Clustering | {mode_label}")
        elif is_faces:
            mode_label = "Basic" if self._faces_mode == "basic" else "Advanced"
            self.feature_label.setText(f"Feature: Faces | {mode_label}")
        else:
            self.feature_label.setText("Feature: Names")
        self._sync_mode_buttons()
        self._sync_workspace_buttons()
        self._sync_pane_toggle_buttons()
        self._reflow_main_splitter(force=True)

    def set_active_workspace(self, workspace_id: str) -> None:
        self._active_workspace = self._normalize_workspace_id(workspace_id)
        self._refresh_workspace_ui()
        if self._active_workspace == "clustering":
            self.maybe_warm_runtime()

    def _on_names_face_labels_changed(self) -> None:
        """Refresh the existing Faces views after a bounded Names label edit."""

        self.faces_pane.refresh_face_library(reason="labels changed in Names")
        self.faces_pane.refresh_face_album(reason="labels changed in Names", force_refresh=True)

    def _on_main_gallery_face_labels_changed(self, _image_path: str) -> None:
        """Synchronize Photos inspector region edits with Names and Faces."""

        self.names_pane.refresh_names()
        self._on_names_face_labels_changed()

    @staticmethod
    def _normalize_ui_mode(mode: str) -> str:
        return "advanced" if str(mode or "").strip().lower() == "advanced" else "basic"

    def set_active_workspace_mode(self, mode: str) -> None:
        if self._active_workspace == "faces":
            self.set_faces_mode(mode)
            return
        self.set_clustering_mode(mode)

    def set_faces_mode(self, mode: str) -> None:
        self._faces_mode = self._normalize_ui_mode(mode)
        if hasattr(self, "faces_pane"):
            self.faces_pane.set_ui_mode(self._faces_mode)
        if self._active_workspace == "faces":
            self._refresh_workspace_ui()
        else:
            self._sync_mode_buttons()

    def set_clustering_mode(self, mode: str) -> None:
        mode = str(mode or "basic").strip().lower()
        if mode not in {"basic", "advanced"}:
            mode = "basic"
        previous_mode = self._clustering_mode
        if previous_mode == "advanced" and mode != previous_mode:
            self._advanced_pane_visibility = dict(self._pane_visibility)
            self._advanced_cluster_splitter_sizes = [int(value) for value in self.cluster_right_splitter.sizes()]
        self._clustering_mode = mode
        if mode == "advanced":
            self._pane_visibility = dict(self._advanced_pane_visibility)
            self.cluster_right_splitter.setSizes(self._advanced_cluster_splitter_sizes)
        else:
            self._pane_visibility = {"source": True, "controls": False, "details": True}
            self.cluster_right_splitter.setSizes([1])
        self.gallery_pane.set_action_bar_visible(mode == "advanced")
        self.cluster_pane.set_basic_mode(mode == "basic")
        self._refresh_workspace_ui()
        self._update_metrics_overlay()

    def restore_ui_state(self) -> None:
        geometry = self.settings_store.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        self._advanced_pane_visibility = {"source": True, "controls": True, "details": True}
        self._pane_visibility = dict(self._advanced_pane_visibility)
        splitter_sizes = self.settings_store.value("window/main_splitter")
        if splitter_sizes:
            try:
                values = [int(value) for value in splitter_sizes]
                if len(values) == 4:
                    self._pane_restore_widths["source"] = max(220, values[0])
                    self._pane_restore_widths["controls"] = max(380, values[1])
                    self._pane_restore_widths["details"] = max(440, values[3])
            except Exception:
                pass
        cluster_sizes = self.settings_store.value("window/cluster_right_splitter")
        if cluster_sizes:
            self._advanced_cluster_splitter_sizes = [int(value) for value in cluster_sizes]
        else:
            self._advanced_cluster_splitter_sizes = [1]
        self.cluster_right_splitter.setSizes(self._advanced_cluster_splitter_sizes)

        saved_directory = self.settings_store.value("sourcepane/selected_directory", "", str)
        if saved_directory:
            self.source_pane.set_selected_directory(saved_directory)
        self.clustering_pane.apply_state(self._load_json("clusteringpane/state"))
        self.faces_pane.apply_state(self._load_json("facespane/state"))
        self.clustering_pane.set_running(False)
        self.source_pane.set_running(False)
        self.set_clustering_mode("basic")
        self.set_faces_mode(self._preferred_faces_ui_mode())
        self.set_active_workspace(self._preferred_workspace())
        self.gallery_pane.refresh_selection_target_hint()
        self.footer_bar.set_status("Idle")
        self.on_directory_changed(self.source_pane.selected_directory)

    @staticmethod
    def _thread_is_running(thread) -> bool:
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:
            return False

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

    def _track_async_thread(self, thread, job: object | None, *, role: str) -> None:
        if thread is None:
            return
        self._async_thread_roles[thread] = (str(role), job)
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _handle_async_thread_finished(self, thread) -> None:
        role, job = self._async_thread_roles.pop(thread, (None, None))
        if role == "warmup":
            if self._warmup_thread is thread:
                self._warmup_thread = None
                if self._warmup_job is job:
                    self._warmup_job = None
            return
        if role == "tag_context":
            self._release_async_refs(job, thread)

    def _on_async_thread_finished(self, thread=None) -> None:
        self._handle_async_thread_finished(thread)

    def _shutdown_background_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True

        worker = self.worker
        if worker is not None and self._thread_is_running(worker):
            try:
                worker.cancel()
            except Exception:
                pass
            ready_to_close = self._wait_for_thread(worker, timeout_ms) and ready_to_close
        if worker is not None and not self._thread_is_running(worker):
            self.worker = None

        if self._warmup_job is not None:
            try:
                self._warmup_job.cancel()
            except Exception:
                pass
        ready_to_close = self._wait_for_thread(self._warmup_thread, timeout_ms) and ready_to_close
        if not self._thread_is_running(self._warmup_thread):
            self._warmup_job = None
            self._warmup_thread = None

        current_tag_job = self._tag_context_job
        current_tag_thread = self._tag_context_thread
        tag_pairs = [(current_tag_job, current_tag_thread), *list(self._retained_async_refs)]
        for job, thread in tag_pairs:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            ready_to_close = self._wait_for_thread(thread, timeout_ms) and ready_to_close
        if ready_to_close:
            self._tag_context_job = None
            self._tag_context_thread = None
            self._retained_async_refs = []
            self._async_thread_roles = {}

        try:
            ready_to_close = self.gallery_pane.shutdown_jobs(timeout_ms=timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            ready_to_close = self.cluster_pane.shutdown_jobs(timeout_ms=timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            ready_to_close = self.faces_pane.shutdown_jobs(timeout_ms=timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            ready_to_close = self.names_pane.shutdown_jobs(timeout_ms=timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False

        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self._shutdown_background_jobs():
            self.footer_bar.set_status("Waiting for background jobs to stop before closing.")
            event.ignore()
            return
        if self._clustering_mode == "advanced":
            self._advanced_pane_visibility = dict(self._pane_visibility)
            self._advanced_cluster_splitter_sizes = [int(value) for value in self.cluster_right_splitter.sizes()]
        self.settings_store.setValue("window/geometry", self.saveGeometry())
        self.settings_store.setValue(
            "window/main_splitter",
            [
                int(self._pane_restore_widths["source"]),
                int(self._pane_restore_widths["controls"]),
                max(820, int(self.gallery_pane.width())),
                int(self._pane_restore_widths["details"]),
            ],
        )
        self.settings_store.setValue("window/cluster_right_splitter", self._advanced_cluster_splitter_sizes)
        self.settings_store.setValue("layout/source_visible", self._advanced_pane_visibility["source"])
        self.settings_store.setValue("layout/controls_visible", self._advanced_pane_visibility["controls"])
        self.settings_store.setValue("layout/details_visible", self._advanced_pane_visibility["details"])
        self.settings_store.setValue("sourcepane/selected_directory", self.source_pane.selected_directory)
        self.settings_store.setValue("workspace/faces_mode", self._faces_mode)
        self.settings_store.setValue("clusteringpane/state", json.dumps(self.clustering_pane.export_state()))
        self.settings_store.setValue("facespane/state", json.dumps(self.faces_pane.export_state()))
        return super().closeEvent(event)

    def update_runtime_status(self) -> None:
        diagnostics = self.runtime_service.diagnostics(self._preferred_execution_mode())
        self.execution_policy = diagnostics["policy"]
        self.runtime_badge.update_runtime(diagnostics["capabilities"], diagnostics["policy"])
        self.footer_bar.set_status(diagnostics["policy"].reason)
        self._update_metrics_overlay()

    def reconfigure_runtime(self) -> None:
        self.execution_policy = self.runtime_service.select_policy(self._preferred_execution_mode(), refresh=True)
        self._warmup_cache.clear()
        self._warmup_signature = None
        self._warmup_state = "idle"
        self._build_services(reset_face_session=False)
        self.gallery_pane.metadata_service = self.photo_metadata_service
        self.gallery_pane.image_tag_service = self.image_tag_service
        self.faces_pane.face_service_global = self.face_service_global
        self.faces_pane.face_service_session = self.face_service_session
        self.faces_pane.set_face_services(
            face_services_global=self.face_services_global,
            face_services_session=self.face_services_session,
            refresh=False,
        )
        self.faces_pane.results_gallery.metadata_service = self.photo_metadata_service
        self.faces_pane.results_gallery.image_tag_service = self.image_tag_service
        self.faces_pane.set_face_service_provider(self._face_service_for_pipeline)
        self.faces_pane.configure_face_pipeline_options(
            self._face_model_root(),
            self._preferred_face_pipeline_defaults(),
            refresh=False,
        )
        self.faces_pane.set_active_face_mode(self.faces_pane.current_face_mode(), refresh=True)
        if self._active_workspace == "names":
            self.names_pane.refresh_names()
        self.update_runtime_status()

    def _update_metrics_overlay(self) -> None:
        metrics = dict(self.last_run_metrics or {})
        metrics["warmup_state"] = self._warmup_state
        metrics["runtime_reason"] = self.execution_policy.reason
        metrics["performance_profile"] = self.performance_profile.name
        dashboard = self.performance_dashboard_service.snapshot(
            metrics,
            jobs=self.job_manager.history(limit=20),
        )
        if self.last_run_metrics:
            self.clustering_pane.update_metrics(metrics)
        self.footer_bar.set_performance_dashboard(dashboard.to_text())

    def _reflow_main_splitter(self, *, force: bool) -> None:
        _ = force
        main_sizes = self.main_splitter.sizes()
        if main_sizes and len(main_sizes) >= 2 and self.source_pane.isVisible() and int(main_sizes[0]) > 0:
            self._pane_restore_widths["source"] = max(220, int(main_sizes[0]))
        cluster_sizes = self.clustering_splitter.sizes()
        if cluster_sizes and len(cluster_sizes) == 3:
            if self._clustering_mode == "advanced" and self._pane_visibility["controls"] and int(cluster_sizes[0]) > 0:
                self._pane_restore_widths["controls"] = max(380, int(cluster_sizes[0]))
            if self._clustering_mode == "advanced" and self._pane_visibility["details"] and int(cluster_sizes[2]) > 0:
                self._pane_restore_widths["details"] = max(440, int(cluster_sizes[2]))
        total_width = max(1920, self.width() or 1920)
        source_visible = self._source_pane_visible()
        source = self._pane_restore_widths["source"] if source_visible else 0
        workspace = max(1040, total_width - source - 24)
        self.source_pane.setVisible(source_visible)
        self.main_splitter.setSizes([source, workspace])
        if self._clustering_mode == "basic":
            controls = 0
            right = max(440, self._pane_restore_widths["details"])
            gallery = max(820, workspace - right - 18)
            self.clustering_pane.setMinimumWidth(0)
            self.clustering_pane.setMaximumWidth(460)
            self.cluster_right_splitter.setMinimumWidth(440)
            self.cluster_right_splitter.setMaximumWidth(680)
            self.clustering_pane.setVisible(False)
            self.cluster_right_splitter.setVisible(True)
            self.clustering_splitter.setSizes([controls, gallery, right])
            return
        controls = self._pane_restore_widths["controls"] if self._pane_visibility["controls"] else 0
        right = self._pane_restore_widths["details"] if self._pane_visibility["details"] else 0
        gallery = max(820, workspace - controls - right - 18)
        self.clustering_pane.setMinimumWidth(380)
        self.clustering_pane.setMaximumWidth(460)
        self.cluster_right_splitter.setMinimumWidth(440)
        self.cluster_right_splitter.setMaximumWidth(680)
        self.clustering_pane.setVisible(self._pane_visibility["controls"])
        self.cluster_right_splitter.setVisible(self._pane_visibility["details"])
        self.clustering_splitter.setSizes([controls, gallery, right])

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

    def _set_pane_visible(self, pane_key: str, visible: bool) -> None:
        visible = bool(visible)
        if pane_key == "source":
            if self._active_workspace == "clustering" and self._clustering_mode != "advanced":
                return
            if self._advanced_pane_visibility.get("source") == visible and self._pane_visibility.get("source") == visible:
                return
            self._advanced_pane_visibility["source"] = visible
            self._pane_visibility["source"] = visible
            self._sync_pane_toggle_buttons()
            self._reflow_main_splitter(force=True)
            return
        if self._active_workspace != "clustering" or self._clustering_mode != "advanced":
            return
        if self._pane_visibility.get(pane_key) == visible:
            return
        self._pane_visibility[pane_key] = visible
        self._advanced_pane_visibility[pane_key] = visible
        self._sync_pane_toggle_buttons()
        self._reflow_main_splitter(force=True)

    def maybe_warm_runtime(self) -> None:
        if self._active_workspace != "clustering":
            self._warmup_state = "deferred"
            self._update_metrics_overlay()
            return
        if not self.settings_store.value("runtime/allow_gpu_warmup", self.settings.allow_gpu_warmup, bool):
            self._warmup_state = "disabled"
            self.footer_bar.set_status("Warm-up disabled.")
            self._update_metrics_overlay()
            return
        if self.execution_policy.effective_mode == "cpu":
            self._warmup_state = "cpu-only"
            self.footer_bar.set_status("CPU runtime active.")
            self._update_metrics_overlay()
            return

        model_name = self.clustering_pane.selected_embedding_models()[0]
        use_onnx = bool(self.clustering_pane.onnx_checkbox.isChecked())
        signature = (model_name, use_onnx, self.execution_policy.effective_mode, self.execution_policy.onnx_provider)
        if signature in self._warmup_cache:
            self._warmup_state = f"ready ({model_name})"
            self.footer_bar.set_status(f"Runtime warm and ready for {model_name}.")
            self._update_metrics_overlay()
            return
        try:
            asset_service = self.embedding_service.model_manager.model_asset_service
        except Exception:
            asset_service = None
        local_model_ready = True
        if asset_service is not None:
            try:
                local_model_ready = bool(asset_service.model_available_without_download(model_name))
            except Exception:
                local_model_ready = True
        if not local_model_ready:
            self._warmup_state = f"skipped ({model_name} requires download)"
            self.footer_bar.set_status(f"Warm-up skipped for {model_name}; model is not cached locally.")
            self._update_metrics_overlay()
            return
        if self._warmup_thread is not None:
            try:
                if self._warmup_thread.isRunning():
                    return
            except RuntimeError:
                self._warmup_thread = None
                self._warmup_job = None

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, f"Warming {model_name}...")
            self.embedding_service.model_manager.get_bundle(model_name, use_onnx=use_onnx)
            return signature

        self._warmup_state = f"warming ({model_name})"
        self.footer_bar.set_status(f"Warming {model_name}...")
        self._update_metrics_overlay()
        job = AsyncJob(_run)
        self._warmup_job_id = self.job_manager.register_job(
            "Runtime warm-up",
            cancel_fn=job.cancel,
            origin="Clustering",
        )
        job.progress.connect(lambda value, text: self.job_manager.update(self._warmup_job_id or -1, progress=value, text=text))

        def _finish(status: str, error: str = "") -> None:
            if self._warmup_job_id is not None:
                self.job_manager.finish(self._warmup_job_id, status=status, error=error)
                self._warmup_job_id = None

        def _done(result: object) -> None:
            if isinstance(result, tuple):
                self._warmup_cache.add(result)
                self._warmup_signature = result
            self._warmup_state = f"ready ({model_name})"
            self.footer_bar.set_status(f"Runtime warm and ready for {model_name}.")
            self._update_metrics_overlay()
            _finish("finished")

        def _failed(message: str) -> None:
            self._warmup_state = f"failed ({message})"
            self.footer_bar.set_status(f"Warm-up failed: {message}")
            self._update_metrics_overlay()
            _finish("failed", message)

        def _cancelled() -> None:
            self._warmup_state = "cancelled"
            self.footer_bar.set_status("Warm-up cancelled.")
            self._update_metrics_overlay()
            _finish("cancelled")

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        self._warmup_job = job
        thread = start_job_in_thread(job)
        self._track_async_thread(thread, job, role="warmup")
        self._warmup_thread = thread

    def user_flows_guide_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "docs" / "USER_FLOWS.md"

    def open_user_flows(self) -> None:
        guide_path = self.user_flows_guide_path()
        if not guide_path.exists():
            errorBox("User flows unavailable", f"Guide not found: {guide_path}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide_path))):
            errorBox("Open user flows failed", f"Could not open: {guide_path}")
            return
        self.footer_bar.set_status(f"Opened user flows: {guide_path}")

    def open_settings_dialog(self) -> None:
        from ui.settings_dialog import SettingsDialog

        dialog = SettingsDialog(
            self.settings_store,
            self.runtime_service,
            describe_rebuildable_caches=self.describe_rebuildable_caches,
            describe_generated_storage=self.describe_generated_storage,
            prepare_rebuildable_cache_clear=self.prepare_rebuildable_cache_clear,
            clear_rebuildable_caches=self.clear_rebuildable_caches,
            clear_runtime_temp_files=self.clear_runtime_temp_files,
            clear_face_storage=self.clear_face_storage,
            clear_model_caches=self.clear_model_caches,
            clear_logs=self.clear_logs,
            can_clear_rebuildable_caches=self.can_clear_rebuildable_caches,
            parent=self,
        )
        if dialog.exec() != dialog.DialogCode.Accepted:
            if dialog.face_model_inventory_changed():
                self.reconfigure_runtime()
                self.footer_bar.set_status("Face model inventory updated.")
            return
        previous_mode = self._preferred_execution_mode()
        previous_profile = self._preferred_performance_profile()
        previous_face_settings = self._face_pipeline_settings_snapshot()
        for key, value in dialog.values().items():
            self.settings_store.setValue(key, value)
        self.apply_workspace_preferences()
        if (
            previous_mode != self._preferred_execution_mode()
            or previous_profile != self._preferred_performance_profile()
            or previous_face_settings != self._face_pipeline_settings_snapshot()
            or dialog.face_model_inventory_changed()
        ):
            self.reconfigure_runtime()
        else:
            self.faces_pane.configure_face_pipeline_options(
                self._face_model_root(),
                self._preferred_face_pipeline_defaults(),
                refresh=False,
            )
            self.faces_pane.set_active_face_mode(self.faces_pane.current_face_mode(), refresh=True)
            self.update_runtime_status()
        self.footer_bar.set_status("Workspace preferences updated.")
        infoBox("Settings saved", "Workspace preferences updated.")

    def can_clear_rebuildable_caches(self) -> bool:
        return not bool(self.worker and self.worker.isRunning())

    def describe_rebuildable_caches(self) -> CacheUsageSummary:
        return self.cache_maintenance_service.describe_rebuildable_caches()

    def describe_generated_storage(self):
        return self.cache_maintenance_service.describe_generated_storage(config_location=self.settings_store.fileName())

    def prepare_rebuildable_cache_clear(self) -> None:
        self.gallery_pane.clear_memory_caches()
        self.embedding_service.cache_service.clear_memory_cache()

    def clear_rebuildable_caches(self) -> CacheClearResult:
        before = self.cache_maintenance_service.describe_rebuildable_caches()
        failures: list[str] = []
        cleared_targets: list[str] = []
        try:
            self.embedding_service.cache_service.clear_disk_cache()
            cleared_targets.append("embeddings.sqlite3")
        except Exception as exc:
            failures.append(f"embeddings.sqlite3: {exc}")
        cleared, disk_failures = self.cache_maintenance_service.clear_rebuildable_disk_targets(
            exclude={"embeddings.sqlite3", "thumbnails/"}
        )
        cleared_targets.extend(cleared)
        failures.extend(disk_failures)
        try:
            self.gallery_pane.thumbnail_service.clear_disk_cache()
            cleared_targets.append("thumbnails/")
        except Exception as exc:
            failures.append(f"thumbnails/: {exc}")
        after = self.cache_maintenance_service.describe_rebuildable_caches()
        unique_targets = tuple(dict.fromkeys(cleared_targets))
        freed_bytes = max(0, int(before.total_bytes) - int(after.total_bytes))
        return CacheClearResult(cleared_targets=unique_targets, freed_bytes=freed_bytes, failures=tuple(failures))

    def _clear_generated_storage_targets(self, clear_fn) -> CacheClearResult:
        before = self.cache_maintenance_service.describe_generated_storage(config_location=self.settings_store.fileName())
        cleared, failures = clear_fn()
        after = self.cache_maintenance_service.describe_generated_storage(config_location=self.settings_store.fileName())
        freed_bytes = max(0, int(before.total_bytes) - int(after.total_bytes))
        return CacheClearResult(cleared_targets=tuple(cleared), freed_bytes=freed_bytes, failures=tuple(failures))

    def clear_runtime_temp_files(self) -> CacheClearResult:
        return self._clear_generated_storage_targets(self.cache_maintenance_service.clear_runtime_temp_files)

    def clear_face_storage(self) -> CacheClearResult:
        return self._clear_generated_storage_targets(self.cache_maintenance_service.clear_face_storage_targets)

    def clear_model_caches(self) -> CacheClearResult:
        return self._clear_generated_storage_targets(self.cache_maintenance_service.clear_model_cache_targets)

    def clear_logs(self) -> CacheClearResult:
        return self._clear_generated_storage_targets(self.cache_maintenance_service.clear_log_files)

    def on_directory_changed(self, directory: str) -> None:
        directory = str(directory or "").strip()
        if directory and self.recent_folder_history.record(directory):
            self._refresh_recent_folder_menus()
        append_qt_diagnostic(f"[Main] on_directory_changed pid={os.getpid()} directory={directory or '<empty>'}")
        if not directory:
            self.current_folder_label.setText("No folder selected")
            self.current_folder_label.setToolTip("")
            self.footer_bar.set_status("Select a folder to start clustering.")
        else:
            name = Path(directory).name or directory
            self.current_folder_label.setText(f"Folder: {name}")
            self.current_folder_label.setToolTip(directory)
            self.footer_bar.set_status(f"Folder selected: {directory}")
        if (
            hasattr(self, "faces_pane")
            and self._active_workspace == "faces"
            and not self.faces_pane.face_folder_path.text().strip()
            and self.faces_pane.is_face_folder_tab_active()
        ):
            self.faces_pane.refresh_face_library(refresh_people=True)
        self.maybe_warm_runtime()

    def _set_shared_source_folder(self, directory: str) -> None:
        path = str(directory or "").strip()
        if path and Path(path).is_dir() and path != self.source_pane.selected_directory:
            self.source_pane.set_selected_directory(path)

    def _refresh_recent_folder_menus(self) -> None:
        paths = self.recent_folder_history.paths()
        if hasattr(self, "source_pane"):
            self.source_pane.set_recent_directories(paths)
        if hasattr(self, "faces_pane"):
            self.faces_pane.set_recent_directories(paths)

    def _remove_recent_folder(self, directory: str) -> None:
        if self.recent_folder_history.remove(directory):
            self._refresh_recent_folder_menus()

    def _clear_recent_folders(self) -> None:
        if self.recent_folder_history.clear():
            self._refresh_recent_folder_menus()

    def _current_scope_paths(self) -> list[str] | None:
        target = self.cluster_pane.current_selection_target()
        if target is None:
            return None
        paths = [str(path) for path in target.as_list() if str(path or "").strip()]
        return paths or None

    def _current_clustering_filter_state(self) -> dict[str, object]:
        tags = []
        match = "Any"
        if hasattr(self, "clustering_pane"):
            tags = self.clustering_pane.selected_tag_filters()
            match = self.clustering_pane.selected_tag_match_mode()
        return {"tags": tags, "tag_match": match}

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
        self.clustering_pane.tag_filter_field.setText(", ".join(tag_values))
        self.clustering_pane.tag_match_combobox.setCurrentText("All" if tag_match == "all" else "Any")
        self.set_active_workspace("clustering")
        self.run_clustering()

    def _build_clustering_request(self, *, source_paths: list[str] | None = None) -> ClusteringRequest:
        similarity_modes = self.clustering_pane.selected_similarity_modes()
        return ClusteringRequest(
            directory=self.source_pane.selected_directory,
            embedding_models=self.clustering_pane.selected_embedding_models(),
            num_clusters=self.clustering_pane.cluster_spinbox.value(),
            clustering_backends=self.clustering_pane.selected_clustering_backends(),
            recursive=self.clustering_pane.recursive_checkbox.isChecked(),
            similarity_mode=similarity_modes[0],
            similarity_modes=similarity_modes,
            outlier_policy=self.clustering_pane.outlier_combobox.currentText(),
            use_onnx=self.clustering_pane.onnx_checkbox.isChecked(),
            reuse_result_cache=self.clustering_pane.result_cache_checkbox.isChecked(),
            use_embedding_cache_lookup=self.clustering_pane.embedding_cache_lookup_checkbox.isChecked(),
            source_paths=source_paths,
            performance_profile=self.performance_profile.name,
            generate_cluster_meanings=self._clustering_mode == "advanced",
            cluster_meaning_model="auto",
        )

    def _start_clustering_request(
        self,
        request: ClusteringRequest,
        *,
        run_origin: str,
        tag_filter: list[str] | tuple[str, ...] = (),
        tag_match: str = "",
    ) -> None:
        append_qt_diagnostic(
            "[Main] start_clustering "
            f"pid={os.getpid()} dir={request.directory} images={len(request.source_paths or []) if request.source_paths else 0} "
            f"models={','.join(request.embedding_models)} backends={','.join(request.clustering_backends)} "
            f"recursive={request.recursive} onnx={request.use_onnx}"
        )
        self.current_run_origin = str(run_origin or "folder")
        self.current_tag_filter = tuple(str(tag).strip() for tag in (tag_filter or ()) if str(tag).strip())
        self.current_tag_match = str(tag_match or "").strip() if self.current_tag_filter else ""
        self._cancel_cluster_tag_context_refresh()
        self.clustering_pane.set_running(True)
        self.source_pane.set_running(True)
        self._main_gallery_context_overrides = {}
        self.clustering_pane.update_metrics({})
        self.footer_bar.set_metrics("")
        self.cluster_data = {}
        self.membership_by_image = {}
        self.metrics_by_backend = {}
        self.cluster_explanations = {}
        self.cluster_meanings = {}
        self.image_tags_by_path = {}
        self.cluster_tag_summaries = {}
        self.gallery_extra_context_by_path = {}
        self.last_run_metrics = {}
        self.cluster_pane.update_clusters(
            self.cluster_data,
            self.membership_by_image,
            self.metrics_by_backend,
            self.cluster_tag_summaries,
            self.cluster_explanations,
            self.cluster_meanings,
        )
        self.gallery_pane.set_membership_context(self.membership_by_image, self.metrics_by_backend)
        self.gallery_pane.update_gallery([])
        self.worker = ClusteringWorker(self.pipeline, request)
        self.worker.progress_changed.connect(
            self.clustering_pane.update_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self.worker.progress_changed.connect(
            self.source_pane.update_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self.worker.progress_changed.connect(
            self._on_clustering_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self.worker.completed.connect(
            self.on_clustering_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self.worker.failed.connect(
            self.on_clustering_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self.worker.cancelled.connect(
            self.on_clustering_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )
        self._clustering_job_id = self.job_manager.register_job(
            "Clustering",
            cancel_fn=self.worker.cancel,
            origin="Clustering",
        )
        self.job_manager.update(self._clustering_job_id, progress=0, text="Preparing clustering run")
        self.worker.start()

    def run_clustering(self) -> None:
        if self.worker and self.worker.isRunning():
            errorBox("Clustering already running")
            return
        try:
            tag_filter = self.clustering_pane.selected_tag_filters()
            tag_match = self.clustering_pane.selected_tag_match_mode()
            source_paths: list[str] | None = None
            run_origin = "folder"
            if tag_filter:
                discovery = self.pipeline.discovery_service.discover_result(
                    self.source_pane.selected_directory,
                    recursive=self.clustering_pane.recursive_checkbox.isChecked(),
                )
                source_paths = self.image_tag_service.select_paths_by_tags(list(discovery.paths), tag_filter, tag_match)
                if len(source_paths) < 2:
                    message = f"Found {len(source_paths)} images matching tags {', '.join(tag_filter)}."
                    self.footer_bar.set_status(message)
                    errorBox("Not enough tagged images", message)
                    return
                run_origin = "tag_filter"
            request = self._build_clustering_request(source_paths=source_paths)
            self._start_clustering_request(
                request,
                run_origin=run_origin,
                tag_filter=tag_filter,
                tag_match=tag_match,
            )
        except Exception as exc:
            LOGGER.exception("Failed to start clustering for %s", self.source_pane.selected_directory)
            self.footer_bar.set_status(f"Clustering could not start: {exc}")
            errorBox("Clustering could not start", str(exc))

    def recluster_selected_cluster(self) -> None:
        if self.worker and self.worker.isRunning():
            errorBox("Clustering already running")
            return
        target = self.cluster_pane.current_selection_target()
        if target is None or len(target.paths) < 2:
            errorBox("Select a cluster", "Choose a cluster with at least two images first.")
            return
        request = self._build_clustering_request(source_paths=list(target.paths))
        self._start_clustering_request(request, run_origin="recluster")

    def _on_clustering_progress(self, value: int, status: str) -> None:
        if self._clustering_job_id is None:
            return
        self.job_manager.update(self._clustering_job_id, progress=int(value), text=str(status))

    def cancel_clustering(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.footer_bar.set_status("Cancelling clustering...")
            if self._clustering_job_id is not None:
                self.job_manager.update(self._clustering_job_id, text="Cancelling")

    def on_clustering_finished(self, result: MultiBackendClusteringResult, metrics: dict) -> None:
        self.cluster_data = result.clusters_by_key
        self.membership_by_image = result.membership_by_image
        self.metrics_by_backend = result.metrics_by_key
        self.cluster_explanations = result.cluster_explanations_by_key
        self.cluster_meanings = result.cluster_meanings_by_key
        self.last_run_metrics = dict(metrics)
        self.last_run_metrics["run_origin"] = self.current_run_origin
        if self.current_tag_filter:
            self.last_run_metrics["tag_filter"] = ", ".join(self.current_tag_filter)
        self.last_run_metrics["tag_match"] = self.current_tag_match or "Any"
        self.last_run_metrics["warmup_state"] = self._warmup_state
        self.last_run_metrics["runtime_reason"] = self.execution_policy.reason
        self.clustering_pane.set_running(False)
        self.source_pane.set_running(False)
        self.image_tags_by_path = {}
        self.cluster_tag_summaries = {}
        self.gallery_extra_context_by_path = {}
        self.cluster_pane.update_clusters(
            self.cluster_data,
            self.membership_by_image,
            self.metrics_by_backend,
            self.cluster_tag_summaries,
            self.cluster_explanations,
            self.cluster_meanings,
        )
        self.gallery_pane.set_membership_context(self.membership_by_image, self.metrics_by_backend)
        self.gallery_pane.inspector_context_provider = self._gallery_context_for_path
        if self._clustering_job_id is not None:
            self.job_manager.finish(self._clustering_job_id, status="finished")
            self._clustering_job_id = None
        self.footer_bar.set_status("Clustering finished.")
        self._update_metrics_overlay()
        self._start_cluster_tag_context_refresh()
        LOGGER.info("Clustering finished")

    def on_clustering_failed(self, message: str) -> None:
        self.clustering_pane.set_running(False)
        self.source_pane.set_running(False)
        self.footer_bar.set_status(f"Clustering failed: {message}")
        errorBox("Clustering failed", message)
        if self._clustering_job_id is not None:
            self.job_manager.finish(self._clustering_job_id, status="failed", error=message)
            self._clustering_job_id = None

    def on_clustering_cancelled(self) -> None:
        self.clustering_pane.set_running(False)
        self.source_pane.set_running(False)
        self.footer_bar.set_status("Clustering cancelled.")
        self.last_run_metrics = {
            "status": "cancelled",
            "run_origin": self.current_run_origin,
            "warmup_state": self._warmup_state,
            "runtime_reason": self.execution_policy.reason,
        }
        self._update_metrics_overlay()
        if self._clustering_job_id is not None:
            self.job_manager.finish(self._clustering_job_id, status="cancelled")
            self._clustering_job_id = None

    def update_gallery(self, backend: str, cluster_id: int) -> None:
        self.gallery_timer.restart()
        self._main_gallery_context_overrides = {}
        self.gallery_pane.update_gallery_with_options(
            images=self.cluster_data.get(backend, {}).get(cluster_id, []),
            clear_pixmaps=False,
            reset_scroll=True,
        )

    def on_gallery_first_paint(self, latency_ms: int) -> None:
        if not self.last_run_metrics:
            return
        self.last_run_metrics["gallery_first_paint_ms"] = latency_ms
        self._update_metrics_overlay()

    def on_center_gallery_image_selected(self, image_path: str) -> None:
        images = list(self.gallery_pane.images)
        if image_path not in images:
            return
        self.cluster_pane.update_membership(image_path, self.membership_by_image, self.metrics_by_backend)
        self.cluster_pane.highlight_membership(image_path)

    def on_gallery_paths_removed(self, paths: list[str]) -> None:
        removed = {str(path) for path in paths if path}
        if not removed:
            return
        self._main_gallery_context_overrides = {
            image_path: context
            for image_path, context in self._main_gallery_context_overrides.items()
            if image_path not in removed
        }
        self.cluster_data = {
            backend: {
                cluster_id: [path for path in image_paths if path not in removed]
                for cluster_id, image_paths in clusters.items()
                if [path for path in image_paths if path not in removed]
            }
            for backend, clusters in self.cluster_data.items()
        }
        self.membership_by_image = {
            image_path: payload
            for image_path, payload in self.membership_by_image.items()
            if image_path not in removed
        }
        self.cluster_explanations = {}
        self.cluster_meanings = {}
        self.image_tags_by_path = {
            image_path: payload
            for image_path, payload in self.image_tags_by_path.items()
            if image_path not in removed
        }
        self._refresh_cluster_tag_context()
        self.cluster_pane.update_clusters(
            self.cluster_data,
            self.membership_by_image,
            self.metrics_by_backend,
            self.cluster_tag_summaries,
            self.cluster_explanations,
            self.cluster_meanings,
            preserve_selection=True,
        )
        current_target = self.cluster_pane.current_selection_target()
        if current_target is not None:
            images = current_target.as_list()
            self.gallery_pane.update_gallery_with_options(images=images, clear_pixmaps=False, reset_scroll=False)
            return

    def on_gallery_paths_renamed(self, changed_paths: list[tuple[str, str]]) -> None:
        replacements = {
            str(source): str(target)
            for source, target in changed_paths
            if str(source).strip() and str(target).strip() and str(source) != str(target)
        }
        if not replacements:
            return
        self._main_gallery_context_overrides = {
            replacements.get(path, path): context
            for path, context in self._main_gallery_context_overrides.items()
        }
        self.cluster_data = {
            backend: {
                cluster_id: [replacements.get(path, path) for path in image_paths]
                for cluster_id, image_paths in clusters.items()
            }
            for backend, clusters in self.cluster_data.items()
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
        self._refresh_cluster_tag_context()
        self.cluster_pane.update_clusters(
            self.cluster_data,
            self.membership_by_image,
            self.metrics_by_backend,
            self.cluster_tag_summaries,
            self.cluster_explanations,
            self.cluster_meanings,
            preserve_selection=True,
        )
        current_target = self.cluster_pane.current_selection_target()
        if current_target is not None:
            self.gallery_pane.update_gallery_with_options(
                images=current_target.as_list(),
                clear_pixmaps=True,
                reset_scroll=False,
            )

    def on_gallery_metadata_changed(self, paths: list[str]) -> None:
        self._refresh_cluster_tag_context()
        self.cluster_pane.update_clusters(
            self.cluster_data,
            self.membership_by_image,
            self.metrics_by_backend,
            self.cluster_tag_summaries,
            self.cluster_explanations,
            self.cluster_meanings,
            preserve_selection=True,
        )

    def _all_cluster_paths(self) -> list[str]:
        ordered_paths: list[str] = []
        seen: set[str] = set()
        for clusters in self.cluster_data.values():
            for image_paths in clusters.values():
                for image_path in image_paths:
                    normalized = str(image_path or "").strip()
                    if not normalized or normalized in seen:
                        continue
                    seen.add(normalized)
                    ordered_paths.append(normalized)
        return ordered_paths

    def _cluster_data_snapshot(self) -> dict[str, dict[int, tuple[str, ...]]]:
        return {
            str(comparison_key): {
                int(cluster_id): tuple(str(path) for path in image_paths if str(path or "").strip())
                for cluster_id, image_paths in clusters.items()
            }
            for comparison_key, clusters in self.cluster_data.items()
        }

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

    def _start_cluster_tag_context_refresh(self) -> None:
        snapshot = self._cluster_data_snapshot()
        if not snapshot:
            self.image_tags_by_path = {}
            self.cluster_tag_summaries = {}
            self.gallery_extra_context_by_path = {}
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
            progress(80, "Summarizing cluster tags...")
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
                    progress(
                        80 + int(processed * 20 / total_clusters),
                        f"Summarizing cluster tags {processed}/{total_clusters}",
                    )

            gallery_extra_context_by_path: dict[str, dict[str, object]] = {}
            for image_path in all_paths:
                context: dict[str, object] = {"tags": list(tags_by_path.get(image_path, ()))}
                if run_origin:
                    context["run_origin"] = run_origin
                if tag_filter:
                    context["tag_filter"] = list(tag_filter)
                    context["tag_match"] = tag_match
                gallery_extra_context_by_path[image_path] = context
            return ClusterTagContextPayload(
                image_tags_by_path=tags_by_path,
                cluster_tag_summaries=cluster_tag_summaries,
                gallery_extra_context_by_path=gallery_extra_context_by_path,
            )

        job = AsyncJob(_run)
        self._tag_context_job = job
        job_id = self.job_manager.register_job(
            "Loading cluster tag summaries",
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

        def _finish_job(status: str, error: str = "") -> None:
            self.job_manager.finish(job_id, status=status, error=error)
            if self._tag_context_job_id == job_id:
                self._tag_context_job_id = None

        def _current_generation() -> bool:
            return generation == self._tag_context_generation

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
            self.footer_bar.set_status("Clustering finished.")

        def _failed(message: str) -> None:
            _finish_job("failed", message)
            if not _current_generation():
                return
            self.footer_bar.set_status(f"Clustering finished. Tag summary refresh failed: {message}")
            LOGGER.warning("Cluster tag context refresh failed: %s", message)

        def _cancelled() -> None:
            _finish_job("cancelled")
            if not _current_generation():
                return
            self.footer_bar.set_status("Clustering finished.")

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._track_async_thread(thread, job, role="tag_context")
        self._tag_context_thread = thread

    def _refresh_cluster_tag_context(self) -> None:
        """Refresh tag context through the visible background-work path."""
        self._start_cluster_tag_context_refresh()

    def _capture_face_context(self, paths: list[str]) -> dict[str, dict[str, object]]:
        return {
            str(path): dict(context)
            for path in paths
            if str(path or "").strip()
            for context in [self.faces_pane.context_for_path(str(path))]
            if isinstance(context, dict) and context
        }

    def _open_face_results_in_main_gallery(self, paths: list[str]) -> None:
        ordered_paths = list(dict.fromkeys(str(path) for path in paths if str(path or "").strip()))
        if not ordered_paths:
            return
        self._main_gallery_context_overrides = self._capture_face_context(ordered_paths)
        self.gallery_pane.set_membership_context({}, {})
        self.set_active_workspace("clustering")
        self.gallery_pane.update_gallery_with_options(
            images=ordered_paths,
            clear_pixmaps=True,
            reset_scroll=True,
        )
        self.footer_bar.set_status(f"Opened {len(ordered_paths)} face result(s) in main gallery.")

    def _append_face_results_to_main_gallery(self, paths: list[str]) -> None:
        ordered_paths = list(dict.fromkeys(str(path) for path in paths if str(path or "").strip()))
        if not ordered_paths:
            return
        self._main_gallery_context_overrides.update(self._capture_face_context(ordered_paths))
        combined = list(dict.fromkeys([*list(self.gallery_pane.images), *ordered_paths]))
        self.set_active_workspace("clustering")
        self.gallery_pane.update_gallery_with_options(
            images=combined,
            clear_pixmaps=True,
            reset_scroll=False,
        )
        self.footer_bar.set_status(f"Appended {len(ordered_paths)} face result(s) to main gallery.")

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
        if self.current_run_origin:
            context["run_origin"] = self.current_run_origin
        extra = self.gallery_extra_context_by_path.get(path)
        if isinstance(extra, dict):
            context.update(extra)
        return context

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from apps.pyqt_production.__main__ import main as production_main

    raise SystemExit(production_main(sys.argv[1:]))
