from __future__ import annotations

from collections.abc import Callable
from collections import deque
import json
from pathlib import Path
import sys

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, QSettings, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QHeaderView,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary, GeneratedStorageSummary
from app.services.clustering_options import clustering_model_names, model_label
from app.services.gallery_actions import CLUSTERLENS_TRASH_DIR_NAME, GalleryActionService
from app.services.face_model_installer import FaceModelInstaller, face_model_runtime_root_dir
from app.services.model_assets import TEXT_MODEL_ORDER, ModelAssetService
from app.services.model_downloads import ModelDownloadItem
from apps.pyqt_production.model_download_controller import ModelDownloadController
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown
from ui.error_mbox import confirmBox, errorBox, infoBox
from ui.icons import apply_icon
from ui.job_manager import JobManager
from apps.pyqt_production.identity import PRODUCTION_DISPLAY_NAME
from apps.shared.runtime_support import RuntimeLayout, open_path_in_shell
from apps.shared.support_bundle import export_support_bundle
from infra.performance import detect_system_resources, select_performance_profile
from infra.runtime import RuntimeCapabilityService, available_execution_modes
from infra.settings import get_production_settings_registry, get_settings


class OperationJournalTableModel(QAbstractTableModel):
    HEADERS = ("Date", "Action", "Files", "Results", "Recovery", "Destination")

    def __init__(self, parent=None, *, page_size: int = 50) -> None:
        super().__init__(parent)
        self._entries: list[dict[str, object]] = []
        self.page_size = max(1, int(page_size))
        self._visible_count = 0

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else self._visible_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.HEADERS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> object:
        if not index.isValid():
            return None
        entry = self.entry_at(index.row())
        if entry is None:
            return None
        values = (
            str(entry.get("timestamp_utc") or ""),
            self._operation_label(str(entry.get("operation") or "")),
            str(entry.get("requested_count") or 0),
            f"{max(0, int(entry.get('requested_count') or 0) - int(entry.get('failure_count') or 0))} succeeded, {int(entry.get('failure_count') or 0)} failed",
            str(entry.get("recovery_status") or ("cancelled" if entry.get("cancelled") else "complete")),
            str(entry.get("destination") or "—"),
        )
        if self._role_is(role, Qt.ItemDataRole.DisplayRole) or self._role_is(role, Qt.ItemDataRole.EditRole):
            return values[index.column()]
        if self._role_is(role, Qt.ItemDataRole.ToolTipRole):
            if index.column() in {3, 4, 5}:
                return json.dumps(entry, ensure_ascii=False, sort_keys=True)
            return values[index.column()]
        if self._role_is(role, Qt.ItemDataRole.UserRole):
            return entry
        return None

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> object:
        if orientation == Qt.Orientation.Horizontal and self._role_is(role, Qt.ItemDataRole.DisplayRole):
            if 0 <= section < len(self.HEADERS):
                return self.HEADERS[section]
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

    def set_entries(self, entries: list[dict[str, object]]) -> None:
        self.beginResetModel()
        self._entries = list(entries)
        self._visible_count = min(self.page_size, len(self._entries))
        self.endResetModel()

    def canFetchMore(self, parent: QModelIndex = QModelIndex()) -> bool:
        return not parent.isValid() and self._visible_count < len(self._entries)

    def fetchMore(self, parent: QModelIndex = QModelIndex()) -> None:
        if parent.isValid() or not self.canFetchMore(parent):
            return
        start = self._visible_count
        end = min(start + self.page_size, len(self._entries))
        self.beginInsertRows(QModelIndex(), start, end - 1)
        self._visible_count = end
        self.endInsertRows()

    def entry_at(self, row: int) -> dict[str, object] | None:
        if 0 <= row < self._visible_count:
            return self._entries[row]
        return None

    @staticmethod
    def _operation_label(operation: str) -> str:
        return {
            "delete_to_trash": "Moved to ClusterLens Trash",
            "move": "Moved photos",
            "rename": "Renamed photos",
            "copy": "Copied photos",
            "restore": "Restored photos",
            "write_exif_comment": "Updated photo metadata",
            "write_exif_metadata": "Updated photo metadata",
        }.get(str(operation), str(operation).replace("_", " ").strip().capitalize() or "File operation")

    @staticmethod
    def _role_is(role: object, target: object) -> bool:
        if role == target:
            return True
        target_value = getattr(target, "value", target)
        return role == target_value


class ProductionSettingsDialog(QDialog):
    runtime_rescanned = pyqtSignal()

    def __init__(
        self,
        settings_store: QSettings,
        runtime_service: RuntimeCapabilityService,
        *,
        describe_rebuildable_caches: Callable[[], CacheUsageSummary] | None = None,
        describe_generated_storage: Callable[[], GeneratedStorageSummary] | None = None,
        clear_rebuildable_caches: Callable[[], CacheClearResult] | None = None,
        clear_library_catalog: Callable[[], CacheClearResult] | None = None,
        clear_runtime_temp_files: Callable[[], CacheClearResult] | None = None,
        clear_face_storage: Callable[[], CacheClearResult] | None = None,
        clear_model_caches: Callable[[], CacheClearResult] | None = None,
        clear_logs: Callable[[], CacheClearResult] | None = None,
        clear_runtime_reports: Callable[[], CacheClearResult] | None = None,
        clear_model_assets: Callable[[], CacheClearResult] | None = None,
        can_clear_rebuildable_caches: Callable[[], bool] | None = None,
        runtime_layout: RuntimeLayout,
        support_metadata_provider,
        model_download_controller: ModelDownloadController | None = None,
        job_manager: JobManager | None = None,
        auto_refresh: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.app_settings = get_settings()
        self.settings_registry = get_production_settings_registry()
        self.settings_store = settings_store
        self.runtime_service = runtime_service
        self.describe_rebuildable_caches = describe_rebuildable_caches
        self.describe_generated_storage = describe_generated_storage
        self.clear_rebuildable_caches = clear_rebuildable_caches
        self.clear_library_catalog = clear_library_catalog
        self.clear_runtime_temp_files = clear_runtime_temp_files
        self.clear_face_storage = clear_face_storage
        self.clear_model_caches = clear_model_caches
        self.clear_logs = clear_logs
        self.clear_runtime_reports = clear_runtime_reports
        self.clear_model_assets = clear_model_assets
        self.can_clear_rebuildable_caches = can_clear_rebuildable_caches or (lambda: True)
        self.runtime_layout = runtime_layout
        self.support_metadata_provider = support_metadata_provider
        self._owns_model_download_controller = model_download_controller is None
        self.model_download_controller = model_download_controller or ModelDownloadController(runtime_layout, self)
        self.job_manager = job_manager
        self.model_asset_service = ModelAssetService(runtime_model_assets_dir=runtime_layout.model_assets_dir)
        self.face_model_installer = FaceModelInstaller(self.app_settings)
        self.system_resources = detect_system_resources()
        self._verify_job = None
        self._verify_thread = None
        self._cache_usage_job = None
        self._cache_usage_thread = None
        self._cache_clear_job = None
        self._cache_clear_thread = None
        self._model_job = None
        self._model_thread = None
        self._active_model_download_name: str | None = None
        self._model_inventory_job = None
        self._model_inventory_thread = None
        self._face_model_job = None
        self._face_model_thread = None
        self._face_model_inventory_changed = False
        self._external_face_model_root = ""
        self._journal_restore_job = None
        self._journal_restore_thread = None
        self._journal_refresh_job = None
        self._journal_refresh_thread = None
        self._thread_roles: dict[object, tuple[str, object | None]] = {}
        self._settings_job_ids: dict[object, int] = {}
        self._last_verify: dict[str, object] | None = None
        self.gallery_action_service = GalleryActionService()

        self.setWindowTitle(f"{PRODUCTION_DISPLAY_NAME} Settings")
        target_width, target_height = 900, 720
        screen = self.screen()
        if screen is not None:
            available = screen.availableGeometry()
            target_width = min(target_width, max(640, available.width() - 48))
            target_height = min(target_height, max(480, available.height() - 48))
        self.resize(target_width, target_height)
        self._build_ui()
        self.model_download_controller.progress.connect(self._on_model_download_progress)
        self.model_download_controller.completed.connect(self._on_model_download_completed)
        self.model_download_controller.failed.connect(self._on_model_download_failed)
        self.model_download_controller.cancelled.connect(self._on_model_download_cancelled)
        self.model_download_controller.running_changed.connect(self._on_model_download_running_changed)
        self._load_values()
        if auto_refresh:
            self.refresh_runtime_diagnostics()
            self.refresh_cache_usage()
            self.refresh_model_inventory()
            self._refresh_face_model_inventory()
            self.refresh_operation_journal()
            self.refresh_log_viewer()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)
        self._build_general_tab()
        self._build_runtime_tab()
        self._build_models_tab()
        self._build_storage_tab()
        self._build_safety_tab()
        self._build_updates_tab()
        self._build_diagnostics_tab()

        self.operation_progress_bar = QProgressBar(self)
        self.operation_progress_bar.setTextVisible(False)
        self.operation_progress_bar.setAccessibleName("Settings operation progress")
        self.operation_progress_bar.hide()
        layout.addWidget(self.operation_progress_bar)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        self.cancel_settings_tasks_button = buttons.addButton(
            "Cancel Active Tasks",
            QDialogButtonBox.ButtonRole.ActionRole,
        )
        self.cancel_settings_tasks_button.setToolTip(
            "Cancel active scans, verification, cache maintenance, model maintenance, and model downloads."
        )
        self.cancel_settings_tasks_button.clicked.connect(self._cancel_active_settings_tasks)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._update_settings_cancel_state()

    def _build_general_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.thumbnail_size = QSpinBox()
        self.thumbnail_size.setRange(96, 512)
        self.runtime_badge = QCheckBox("Show runtime status in the application header")
        self.dense_ui = QCheckBox("Use compact spacing")
        form.addRow("Photo thumbnail size", self.thumbnail_size)
        form.addRow(self.runtime_badge)
        form.addRow(self.dense_ui)
        note = QLabel("Common appearance and workspace preferences apply after you choose OK.")
        note.setWordWrap(True)
        form.addRow(note)
        self.tabs.addTab(tab, "General")

    def _build_runtime_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.execution_mode = QComboBox()
        for mode in available_execution_modes():
            self.execution_mode.addItem(str(mode).upper() if mode != "auto" else "Auto (prefer CUDA)", mode)
        self.precision_mode = QComboBox()
        self.precision_mode.addItem("Automatic", "auto")
        self.precision_mode.addItem("Full precision (FP32)", "fp32")
        self.precision_mode.addItem("Mixed precision (FP16 with fallback)", "fp16")
        self.performance_profile = QComboBox()
        self.performance_profile.addItem("Low memory", "low_memory")
        self.performance_profile.addItem("Balanced", "balanced")
        self.performance_profile.addItem("Maximum speed", "max_speed")
        self.performance_profile.setToolTip(
            "Choose a validated balance of throughput and memory use. Advanced values remain available below."
        )
        self.thumbnail_workers = QSpinBox()
        self.thumbnail_workers.setRange(1, max(8, self.system_resources.logical_cpu_count))
        self.thumbnail_workers.setEnabled(False)
        self.prefetch_rows = QSpinBox()
        self.prefetch_rows.setRange(0, max(16, self.system_resources.logical_cpu_count * 2))
        self.prefetch_rows.setEnabled(False)
        self.batch_size_cpu = QSpinBox()
        self.batch_size_cpu.setRange(1, 512)
        self.batch_size_gpu = QSpinBox()
        self.batch_size_gpu.setRange(1, 1024)
        self.decode_workers = QSpinBox()
        self.decode_workers.setRange(1, max(8, self.system_resources.logical_cpu_count * 2))
        self.vram_headroom_mb = QSpinBox()
        self.vram_headroom_mb.setRange(256, 65536)
        self.vram_headroom_mb.setSingleStep(256)
        self.vram_headroom_mb.setSuffix(" MB")
        self.gpu_warmup = QCheckBox("Warm selected embedding model after folder change")
        self.keep_worker_warm = QCheckBox("Keep clustering worker warm between runs")
        self.keep_worker_warm.setToolTip(
            "Keeps Python, Torch, and loaded models alive after a run. Faster repeated runs, but uses more RAM/VRAM while idle."
        )
        form.addRow("Compute device", self.execution_mode)
        form.addRow("Performance preset", self.performance_profile)
        note = QLabel(
            "Auto prefers a compatible NVIDIA CUDA device and visibly falls back to CPU. "
            "Choosing CUDA explicitly never falls back silently."
        )
        note.setWordWrap(True)
        form.addRow(note)

        self.advanced_performance_group = QGroupBox("Advanced performance", tab)
        self.advanced_performance_group.setCheckable(True)
        self.advanced_performance_group.setChecked(False)
        advanced_form = QFormLayout(self.advanced_performance_group)
        advanced_form.addRow("Precision", self.precision_mode)
        advanced_form.addRow("CPU batch size", self.batch_size_cpu)
        advanced_form.addRow("CUDA batch size", self.batch_size_gpu)
        advanced_form.addRow("Decode workers", self.decode_workers)
        advanced_form.addRow("CUDA memory reserve", self.vram_headroom_mb)
        advanced_form.addRow("Thumbnail workers", self.thumbnail_workers)
        advanced_form.addRow("Thumbnail prefetch rows", self.prefetch_rows)
        advanced_form.addRow(self.gpu_warmup)
        advanced_form.addRow(self.keep_worker_warm)
        form.addRow(self.advanced_performance_group)
        self.tabs.addTab(tab, "Performance")

        self.performance_profile.currentIndexChanged.connect(
            lambda _index: self._apply_profile_preview(update_tuning=True)
        )

    def _build_models_tab(self) -> None:
        clustering_tab = QWidget(self)
        layout = QVBoxLayout(clustering_tab)
        self.offline_model_downloads = QCheckBox("Offline mode: never download missing model files automatically")
        layout.addWidget(self.offline_model_downloads)

        self.model_inventory_table = QTableWidget(0, 5, self)
        self.model_inventory_table.setHorizontalHeaderLabels(["Model", "State", "Size", "Checksum", "Source / License"])
        self.model_inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.model_inventory_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.model_inventory_table.verticalHeader().setVisible(False)
        self.model_inventory_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.model_inventory_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.model_inventory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.model_inventory_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.model_inventory_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.model_inventory_table, stretch=1)

        actions = QGridLayout()
        self.refresh_models_button = QPushButton("Refresh Models")
        self.install_model_button = QPushButton("Download / Install Selected")
        self.cancel_model_download_button = QPushButton("Cancel Download")
        self.delete_model_cache_button = QPushButton("Delete Cached Download")
        self.open_model_assets_button = QPushButton("Open Model Assets")
        self.open_download_cache_button = QPushButton("Open Download Cache")
        for index, button in enumerate((
            self.refresh_models_button,
            self.install_model_button,
            self.cancel_model_download_button,
            self.delete_model_cache_button,
            self.open_model_assets_button,
            self.open_download_cache_button,
        )):
            actions.addWidget(button, index // 3, index % 3)
        layout.addLayout(actions)

        self.model_status_label = QLabel()
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)
        layout.addStretch(1)

        models_scroll = QScrollArea(self)
        models_scroll.setObjectName("models_scroll_area")
        models_scroll.setWidgetResizable(True)
        models_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        models_scroll.setWidget(clustering_tab)
        self.tabs.addTab(models_scroll, "Clustering Models")

        face_tab = QWidget(self)
        face_layout = QGridLayout(face_tab)
        face_layout.setColumnStretch(0, 1)
        face_layout.setColumnStretch(1, 1)
        self.face_model_pack_combo = QComboBox(face_tab)
        for label, profile_id in (
            ("Default GPU — SCRFD 10G + ArcFace R100", "latest_gpu"),
            ("Balanced — SCRFD 2.5G + ArcFace R50", "recommended"),
            ("Edge — SCRFD 500M + MobileFaceNet", "edge"),
            ("Accuracy — SCRFD 10G + AdaFace R100", "accuracy"),
            ("Maximum Accuracy — SCRFD 34GF + AdaFace R100", "max_accuracy"),
            ("OpenCV CPU — YuNet + SFace", "opencv_cpu"),
            ("SFace embedders only", "sface"),
            ("YuNet detectors only", "yunet"),
            ("YOLO5Face detector only", "yolo"),
        ):
            self.face_model_pack_combo.addItem(label, profile_id)
        self.install_face_model_pack_button = QPushButton("Install Selected Face Pack")
        self.open_face_model_cache_button = QPushButton("Open Face Model Cache")
        self.clear_face_download_cache_button = QPushButton("Clear Face Download Cache")
        face_layout.addWidget(self.face_model_pack_combo, 0, 0)
        face_layout.addWidget(self.install_face_model_pack_button, 0, 1)
        self.external_face_model_root_label = QLabel("External model folder: not selected", face_tab)
        self.external_face_model_root_label.setWordWrap(True)
        self.choose_external_face_model_root_button = QPushButton("Choose Face Model Folder")
        self.open_external_face_model_root_button = QPushButton("Open Face Model Folder")
        face_layout.addWidget(self.external_face_model_root_label, 1, 0, 1, 2)
        face_layout.addWidget(self.choose_external_face_model_root_button, 2, 0)
        face_layout.addWidget(self.open_external_face_model_root_button, 2, 1)

        self.face_model_inventory_table = QTableWidget(0, 5, face_tab)
        self.face_model_inventory_table.setHorizontalHeaderLabels(["Component", "Type", "Hardware", "State", "Location"])
        self.face_model_inventory_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.face_model_inventory_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.face_model_inventory_table.verticalHeader().setVisible(False)
        for column in range(4):
            self.face_model_inventory_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.face_model_inventory_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        face_layout.addWidget(self.face_model_inventory_table, 3, 0, 1, 2)

        self.installed_face_model_combo = QComboBox(face_tab)
        self.delete_face_model_button = QPushButton("Delete Installed Face Component")
        face_layout.addWidget(self.installed_face_model_combo, 4, 0)
        face_layout.addWidget(self.delete_face_model_button, 4, 1)
        face_layout.addWidget(self.open_face_model_cache_button, 5, 0)
        face_layout.addWidget(self.clear_face_download_cache_button, 5, 1)
        self.face_model_status_label = QLabel()
        self.face_model_status_label.setWordWrap(True)
        face_layout.addWidget(self.face_model_status_label, 6, 0, 1, 2)

        face_scroll = QScrollArea(self)
        face_scroll.setObjectName("face_models_scroll_area")
        face_scroll.setWidgetResizable(True)
        face_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        face_scroll.setWidget(face_tab)
        self.tabs.addTab(face_scroll, "Face Models")

        self.refresh_models_button.clicked.connect(self.refresh_model_inventory)
        self.install_model_button.clicked.connect(self._download_selected_model)
        self.cancel_model_download_button.clicked.connect(self.model_download_controller.cancel)
        self.delete_model_cache_button.clicked.connect(self._delete_selected_model_cache)
        self.open_model_assets_button.clicked.connect(lambda: self._open_path(self.runtime_layout.model_assets_dir))
        self.open_download_cache_button.clicked.connect(lambda: self._open_path(self.app_settings.cache_dir))
        self.model_inventory_table.itemSelectionChanged.connect(self._update_model_action_state)
        self.install_face_model_pack_button.clicked.connect(self._install_selected_face_model_pack)
        self.delete_face_model_button.clicked.connect(self._delete_selected_face_model)
        self.open_face_model_cache_button.clicked.connect(
            lambda: self._open_path(self.face_model_installer.runtime_root())
        )
        self.clear_face_download_cache_button.clicked.connect(self._clear_face_download_cache)
        self.choose_external_face_model_root_button.clicked.connect(self._choose_external_face_model_root)
        self.open_external_face_model_root_button.clicked.connect(self._open_external_face_model_root)

    def _build_storage_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.runtime_root_label = QLabel(str(self.runtime_layout.root))
        self.runtime_root_label.setWordWrap(True)
        self.config_location_label = QLabel(str(self.settings_store.fileName() or "platform defaults"))
        self.config_location_label.setWordWrap(True)
        self.cache_root_label = QLabel(str(self.app_settings.cache_dir))
        self.cache_root_label.setWordWrap(True)
        self.generated_storage_text = QTextEdit()
        self.generated_storage_text.setReadOnly(True)
        self.generated_storage_text.setPlaceholderText("Scanning generated storage usage...")
        self.cache_usage_text = QTextEdit()
        self.cache_usage_text.setReadOnly(True)
        self.cache_usage_text.setPlaceholderText("Scanning rebuildable cache usage...")
        form.addRow("Runtime root", self.runtime_root_label)
        form.addRow("Config location", self.config_location_label)
        form.addRow("Runtime cache location", self.cache_root_label)
        form.addRow("Generated storage usage", self.generated_storage_text)
        form.addRow("Rebuildable cache usage", self.cache_usage_text)
        layout.addLayout(form)

        actions = QGridLayout()
        self.refresh_cache_usage_button = QPushButton("Refresh Cache Usage")
        self.clear_cache_button = QPushButton("Clear Rebuildable Caches")
        self.clear_library_catalog_button = QPushButton("Clear Library Cache")
        self.clear_runtime_temp_button = QPushButton("Clear Temp Files")
        self.clear_face_storage_button = QPushButton("Clear Face DBs / ANN")
        self.clear_model_caches_button = QPushButton("Clear Model Caches")
        self.clear_logs_button = QPushButton("Clear Logs")
        self.clear_runtime_reports_button = QPushButton("Clear Reports")
        self.clear_model_assets_button = QPushButton("Clear Installed Model Assets")
        actions.addWidget(self.refresh_cache_usage_button, 0, 0)
        actions.addWidget(self.clear_cache_button, 0, 1)
        actions.addWidget(self.clear_runtime_temp_button, 0, 2)
        actions.addWidget(self.clear_library_catalog_button, 1, 0)
        actions.addWidget(self.clear_face_storage_button, 1, 1)
        actions.addWidget(self.clear_model_caches_button, 1, 2)
        actions.addWidget(self.clear_logs_button, 2, 0)
        actions.addWidget(self.clear_runtime_reports_button, 2, 1)
        actions.addWidget(self.clear_model_assets_button, 2, 2)
        layout.addLayout(actions)
        self.cache_status_label = QLabel(
            "Each clear action affects only the named generated-data category. Source photos, tags, durable face labels, "
            "recovery history, and settings are preserved."
        )
        self.cache_status_label.setWordWrap(True)
        layout.addWidget(self.cache_status_label)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Storage")

        self.refresh_cache_usage_button.clicked.connect(self.refresh_cache_usage)
        self.clear_cache_button.clicked.connect(self._clear_rebuildable_caches)
        self.clear_library_catalog_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Library Cache?",
                "This removes derived Library photo metadata and generated cluster descriptions. Registered roots, saved albums, review decisions, source images, and face labels are preserved. Refresh a root to rebuild the catalog.",
                self.clear_library_catalog,
            )
        )
        self.clear_runtime_temp_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Temp Files?",
                "This removes runtime temp files. Resumable model-download files are preserved. Source images are not changed.",
                self.clear_runtime_temp_files,
            )
        )
        self.clear_face_storage_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Face DBs And ANN Files?",
                "This removes generated face databases and ANN sidecar files. Durable face labels and source images are not changed.",
                self.clear_face_storage,
            )
        )
        self.clear_model_caches_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Model Caches?",
                "This removes generated model caches. Models may need to be reinstalled or downloaded later.",
                self.clear_model_caches,
            )
        )
        self.clear_logs_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Logs?",
                "This removes generated log files while preserving the file-operation recovery journal. Source images are not changed.",
                self.clear_logs,
            )
        )
        self.clear_runtime_reports_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Reports?",
                "This removes crash reports, support bundles, and benchmark reports. Recovery history and source images are not changed.",
                self.clear_runtime_reports,
            )
        )
        self.clear_model_assets_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Installed Model Assets?",
                "This removes installed model assets. Clustering and face actions may need a later model install or download. Source images are not changed.",
                self.clear_model_assets,
            )
        )

    def _build_updates_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.update_checks_enabled = QCheckBox("Check for signed updates")
        self.update_checks_enabled.setToolTip(
            "Default is off. When enabled, update manifests and artifacts must pass signature, checksum, OS, architecture, and variant checks."
        )
        self.update_channel = QComboBox()
        self.update_channel.addItems(["stable", "beta", "nightly"])
        form.addRow(self.update_checks_enabled)
        form.addRow("Update channel", self.update_channel)
        note = QLabel("Updates never switch between CPU and CUDA variants silently.")
        note.setWordWrap(True)
        form.addRow(note)
        self.tabs.addTab(tab, "Updates")

    def _build_safety_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        self.read_only_mode = QCheckBox("Read-only safety mode")
        self.read_only_mode.setToolTip(
            "Disables file copy/move/delete, tag edits, suggested-tag writes, and EXIF writes. "
            "Use this when reviewing clusters before committing file operations."
        )
        layout.addWidget(self.read_only_mode)

        journal_note = QLabel(
            "File operations are written to an append-only audit journal. "
            "Trash and move operations can be restored when the destination files still exist and original paths are free."
        )
        journal_note.setWordWrap(True)
        layout.addWidget(journal_note)

        self.operation_journal_model = OperationJournalTableModel(self, page_size=50)
        self.operation_journal_table = QTableView(self)
        self.operation_journal_table.setModel(self.operation_journal_model)
        self.operation_journal_model.rowsInserted.connect(
            lambda *_args: self.journal_status_label.setText(
                f"Showing {self.operation_journal_model.rowCount()} of {len(getattr(self, '_journal_entries', []))} recovery operations."
            )
        )
        self.operation_journal_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.operation_journal_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.operation_journal_table.verticalHeader().setVisible(False)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.operation_journal_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.operation_journal_table, stretch=1)

        actions = QGridLayout()
        self.refresh_journal_button = QPushButton("Refresh history")
        self.reveal_journal_target_button = QPushButton("Reveal target")
        self.retry_journal_button = QPushButton("Retry failed files")
        self.restore_journal_button = QPushButton("Restore (skip conflicts)")
        self.restore_unique_journal_button = QPushButton("Restore with unique names")
        apply_icon(self.reveal_journal_target_button, "reveal")
        apply_icon(self.retry_journal_button, "retry")
        apply_icon(self.restore_journal_button, "restore")
        apply_icon(self.restore_unique_journal_button, "recovery")
        actions.addWidget(self.refresh_journal_button, 0, 0)
        actions.addWidget(self.reveal_journal_target_button, 0, 1)
        actions.addWidget(self.retry_journal_button, 0, 2)
        actions.addWidget(self.restore_journal_button, 1, 0)
        actions.addWidget(self.restore_unique_journal_button, 1, 1, 1, 2)
        layout.addLayout(actions)

        self.journal_status_label = QLabel("")
        self.journal_status_label.setWordWrap(True)
        layout.addWidget(self.journal_status_label)
        self.journal_file_results = QTextEdit(self)
        self.journal_file_results.setReadOnly(True)
        self.journal_file_results.setMaximumHeight(120)
        self.journal_file_results.setPlaceholderText("Select an operation to see per-file results.")
        layout.addWidget(self.journal_file_results)
        self.tabs.addTab(tab, "Safety & Recovery")

        self.refresh_journal_button.clicked.connect(self.refresh_operation_journal)
        self.reveal_journal_target_button.clicked.connect(self._reveal_selected_journal_target)
        self.retry_journal_button.clicked.connect(self._retry_selected_journal_operation)
        self.restore_journal_button.clicked.connect(lambda: self._restore_selected_journal_operation(conflict_policy="skip"))
        self.restore_unique_journal_button.clicked.connect(lambda: self._restore_selected_journal_operation(conflict_policy="unique_name"))
        self.operation_journal_table.selectionModel().selectionChanged.connect(lambda *_args: self._on_journal_selection_changed())
        self.read_only_mode.toggled.connect(self._update_journal_action_state)

    def _build_diagnostics_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        form.addRow("Runtime root", QLabel(str(self.runtime_layout.root)))
        form.addRow("Logs", QLabel(str(self.runtime_layout.logs_dir)))
        form.addRow("Cache", QLabel(str(self.runtime_layout.cache_dir)))
        form.addRow("Crash", QLabel(str(self.runtime_layout.crash_dir)))
        form.addRow("Benchmarks", QLabel(str(self.runtime_layout.benchmarks_dir)))
        form.addRow("Support bundles", QLabel(str(self.runtime_layout.support_dir)))
        layout.addLayout(form)

        note = QLabel(
            "Runtime details, verification output, logs, and paths are provided here for troubleshooting and support. "
            "Resource checks only inspect installed local providers; they never download, move, or open model files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        open_logs = QPushButton("Open Logs")
        open_cache = QPushButton("Open Cache")
        open_journal = QPushButton("Open operation journal")
        export_bundle = QPushButton("Export Support Bundle")
        view_crash = QPushButton("View Last Crash")
        self.refresh_runtime_button = QPushButton("Rescan GPU Resources")
        self.refresh_runtime_button.setToolTip(
            "In a background worker, check NVIDIA/Torch CUDA, CUDA ONNX for SCRFD and ArcFace, and cuML HDBSCAN. "
            "This does not install packages, download models, or change the saved compute preference."
        )
        self.verify_runtime_button = QPushButton("Verify Runtime")
        buttons.addWidget(open_logs)
        buttons.addWidget(open_cache)
        buttons.addWidget(open_journal)
        buttons.addWidget(export_bundle)
        buttons.addWidget(view_crash)
        buttons.addWidget(self.refresh_runtime_button)
        buttons.addWidget(self.verify_runtime_button)
        layout.addLayout(buttons)

        self.runtime_text = QTextEdit(self)
        self.runtime_text.setReadOnly(True)
        self.runtime_text.setMaximumHeight(280)
        self.runtime_text.setPlaceholderText("Runtime diagnostics")
        layout.addWidget(self.runtime_text)

        log_actions = QHBoxLayout()
        self.refresh_log_button = QPushButton("Refresh App Log")
        self.open_app_log_button = QPushButton("Open app.log")
        log_actions.addWidget(self.refresh_log_button)
        log_actions.addWidget(self.open_app_log_button)
        layout.addLayout(log_actions)
        self.log_viewer = QTextEdit(self)
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setMaximumHeight(180)
        self.log_viewer.setPlaceholderText("Latest app.log lines will appear here.")
        layout.addWidget(self.log_viewer)
        self.tabs.addTab(tab, "Support")

        open_logs.clicked.connect(lambda: self._open_path(self.runtime_layout.logs_dir))
        open_cache.clicked.connect(lambda: self._open_path(self.runtime_layout.cache_dir))
        open_journal.clicked.connect(lambda: self._open_path(self.gallery_action_service.journal_path))
        export_bundle.clicked.connect(self._export_support_bundle)
        view_crash.clicked.connect(self._view_last_crash)
        self.refresh_runtime_button.clicked.connect(self._rescan_runtime_resources)
        self.verify_runtime_button.clicked.connect(self._verify_gpu)
        self.refresh_log_button.clicked.connect(self.refresh_log_viewer)
        self.open_app_log_button.clicked.connect(lambda: self._open_path(self.runtime_layout.app_log))

    def _build_about_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        self.about_text = QTextEdit(self)
        self.about_text.setReadOnly(True)
        self.about_text.setPlainText(self._about_text())
        layout.addWidget(self.about_text, stretch=1)
        actions = QHBoxLayout()
        open_runtime = QPushButton("Open Runtime Root")
        open_settings = QPushButton("Open Settings Store Location")
        actions.addWidget(open_runtime)
        actions.addWidget(open_settings)
        layout.addLayout(actions)
        self.tabs.addTab(tab, "About")

        open_runtime.clicked.connect(lambda: self._open_path(self.runtime_layout.root))
        open_settings.clicked.connect(lambda: self._open_path(Path(self.settings_store.fileName()).parent))

    def _load_values(self) -> None:
        registry = self.settings_registry
        self._set_combo_data(self.execution_mode, registry.get(self.settings_store, "runtime/preferred_mode", self.app_settings.preferred_execution_mode))
        self._set_combo_data(self.precision_mode, registry.get(self.settings_store, "runtime/precision", "auto"))
        self._set_combo_data(self.performance_profile, registry.get(self.settings_store, "performance/profile", self.app_settings.default_performance_profile))
        self.batch_size_cpu.setValue(int(registry.get(self.settings_store, "performance/batch_size_cpu", self.app_settings.batch_size_cpu)))
        self.batch_size_gpu.setValue(int(registry.get(self.settings_store, "performance/batch_size_gpu", self.app_settings.batch_size_gpu)))
        self.decode_workers.setValue(int(registry.get(self.settings_store, "performance/decode_workers", self.system_resources.logical_cpu_count)))
        self.vram_headroom_mb.setValue(int(registry.get(self.settings_store, "performance/vram_headroom_mb", 1024)))
        self.thumbnail_size.setValue(int(registry.get(self.settings_store, "gallery/thumbnail_size", self.app_settings.thumbnail_size)))
        self.gpu_warmup.setChecked(bool(registry.get(self.settings_store, "runtime/allow_gpu_warmup", self.app_settings.allow_gpu_warmup)))
        self.keep_worker_warm.setChecked(bool(registry.get(self.settings_store, "performance/keep_worker_warm", False)))
        self.runtime_badge.setChecked(bool(registry.get(self.settings_store, "runtime/show_badge", self.app_settings.show_runtime_badge)))
        self.dense_ui.setChecked(bool(registry.get(self.settings_store, "workspace/dense_ui", self.app_settings.default_dense_ui)))
        self.offline_model_downloads.setChecked(
            bool(registry.get(self.settings_store, "models/offline_mode", bool(getattr(sys, "frozen", False))))
        )
        self.read_only_mode.setChecked(bool(registry.get(self.settings_store, "safety/read_only_mode", False)))
        self.update_checks_enabled.setChecked(bool(registry.get(self.settings_store, "updates/checks_enabled", False)))
        self.update_channel.setCurrentText(str(registry.get(self.settings_store, "updates/channel", "stable")))
        self._external_face_model_root = str(registry.get(self.settings_store, "faces/model_root", "") or "").strip()
        self._update_external_face_model_root_label()
        self._apply_profile_preview(update_tuning=False)

    def values(self) -> dict[str, object]:
        return self.settings_registry.validate_values({
            "runtime/preferred_mode": str(self.execution_mode.currentData() or "auto"),
            "runtime/precision": str(self.precision_mode.currentData() or "auto"),
            "performance/profile": str(self.performance_profile.currentData() or "balanced"),
            "runtime/allow_gpu_warmup": bool(self.gpu_warmup.isChecked()),
            "performance/keep_worker_warm": bool(self.keep_worker_warm.isChecked()),
            "performance/batch_size_cpu": int(self.batch_size_cpu.value()),
            "performance/batch_size_gpu": int(self.batch_size_gpu.value()),
            "performance/decode_workers": int(self.decode_workers.value()),
            "performance/vram_headroom_mb": int(self.vram_headroom_mb.value()),
            "runtime/show_badge": bool(self.runtime_badge.isChecked()),
            "workspace/default_view": str(self.settings_registry.get(self.settings_store, "workspace/default_view", "gallery")),
            "gallery/thumbnail_size": int(self.thumbnail_size.value()),
            "gallery/thumbnail_workers": int(self.thumbnail_workers.value()),
            "gallery/prefetch_rows": int(self.prefetch_rows.value()),
            "workspace/dense_ui": bool(self.dense_ui.isChecked()),
            "models/offline_mode": bool(self.offline_model_downloads.isChecked()),
            "faces/model_root": self._external_face_model_root,
            "safety/read_only_mode": bool(self.read_only_mode.isChecked()),
            "updates/checks_enabled": bool(self.update_checks_enabled.isChecked()),
            "updates/channel": self.update_channel.currentText(),
        })

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(str(value))
        if index >= 0:
            combo.setCurrentIndex(index)

    def refresh_model_inventory(self) -> None:
        if self._thread_is_running(self._model_inventory_thread):
            return
        self.model_status_label.setText("Loading model inventory...")
        inventory_models = tuple(clustering_model_names(scope="production")) + ("facenet",)
        job = AsyncJob(
            lambda progress, cancel_check: self.model_asset_service.model_inventory(
                inventory_models,
                progress_callback=progress,
                cancel_check=cancel_check,
            )
        )
        self._model_inventory_job = job

        def _done(result: object) -> None:
            self._populate_model_inventory(list(result) if isinstance(result, (list, tuple)) else [])
            self.model_status_label.setText("Model inventory is up to date.")

        def _failed(message: str) -> None:
            self.model_status_label.setText(f"Model inventory could not be loaded: {message}")

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: self.model_status_label.setText("Model inventory refresh cancelled."))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="model_inventory")
        self._model_inventory_thread = thread
        self._update_model_action_state()

    def _populate_model_inventory(self, items: list[object]) -> None:
        self.model_inventory_table.setRowCount(len(items))
        for row, item in enumerate(items):
            state_parts = []
            if item.packaged:
                state_parts.append("packaged")
            if item.cached:
                state_parts.append("cached")
            if not state_parts:
                state_parts.append("missing")
            if item.validation_status == "invalid":
                state_parts.append("invalid")
            state_text = ", ".join(state_parts)
            values = [
                model_label(item.model_name),
                state_text,
                self._format_bytes(item.size_bytes),
                item.validation_status,
                f"{item.source_label}\n{item.license}\n{item.source_url}",
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(str(value))
                table_item.setToolTip(item.validation_message if column == 3 else str(value))
                table_item.setData(Qt.ItemDataRole.UserRole, item.model_name)
                if item.validation_status == "invalid":
                    table_item.setForeground(Qt.GlobalColor.red)
                self.model_inventory_table.setItem(row, column, table_item)
        self.model_inventory_table.resizeRowsToContents()
        self._update_model_action_state()

    def _selected_model_name(self) -> str | None:
        row = self.model_inventory_table.currentRow()
        if row < 0:
            return None
        item = self.model_inventory_table.item(row, 0)
        if item is None:
            return None
        model_name = item.data(Qt.ItemDataRole.UserRole)
        return str(model_name or "").strip().lower() or None

    def _model_actions_busy(self) -> bool:
        return (
            self.model_download_controller.is_running()
            or self._thread_is_running(self._model_inventory_thread)
            or self._thread_is_running(self._model_thread)
            or self._thread_is_running(self._face_model_thread)
        )

    def _update_model_action_state(self) -> None:
        busy = self._model_actions_busy()
        has_selection = self._selected_model_name() is not None
        mutation_allowed = bool(self.can_clear_rebuildable_caches())
        self.refresh_models_button.setEnabled(not busy)
        self.install_model_button.setEnabled((not busy) and mutation_allowed and has_selection)
        self.cancel_model_download_button.setEnabled(self.model_download_controller.is_running())
        self.delete_model_cache_button.setEnabled((not busy) and mutation_allowed and has_selection)
        self.open_model_assets_button.setEnabled(not busy)
        self.open_download_cache_button.setEnabled(not busy)
        self._update_face_model_action_state()
        self._update_settings_cancel_state()

    def _download_selected_model(self) -> None:
        model_name = self._selected_model_name()
        if not model_name:
            return
        if self.model_download_controller.is_running():
            self.model_status_label.setText("A model download is already active. Monitor or cancel it from Jobs.")
            return
        if not self.can_clear_rebuildable_caches():
            self.model_status_label.setText("Wait for the active production task before installing another model.")
            return
        if not confirmBox(
            "Download selected model?",
            (
                f"This will download and verify model files for {model_label(model_name)} into:\n"
                f"{self.app_settings.cache_dir}\n\n"
                "The download is shared, cached, resumable, visible in Jobs, and cancellable.\n\n"
                "Continue?"
            ),
            parent=self,
        ):
            return
        self.model_status_label.setText(f"Downloading or initializing {model_label(model_name)}...")
        self._active_model_download_name = model_name
        require_text = model_name in TEXT_MODEL_ORDER
        if not self.model_download_controller.start([ModelDownloadItem(model_name, require_text=require_text)]):
            self._active_model_download_name = None
            self.model_status_label.setText("The model download could not start.")
        self._update_model_action_state()

    def _on_model_download_progress(self, value: int, status: str) -> None:
        progress_text = "" if int(value) < 0 else f" ({int(value)}%)"
        self.model_status_label.setText(f"{status}{progress_text}")

    def _on_model_download_running_changed(self, _running: bool) -> None:
        self._update_model_action_state()
        self._update_settings_cancel_state()

    def _on_model_download_completed(self, result: dict) -> None:
        model_name = self._active_model_download_name
        if model_name is None:
            self.model_status_label.setText("Shared model download completed and verified.")
            self.refresh_model_inventory()
            return
        self._active_model_download_name = None
        downloaded = len(result.get("downloaded") or ())
        reused = len(result.get("reused") or ())
        self.model_status_label.setText(
            f"{model_label(model_name)} is ready. Downloaded: {downloaded}; reused from cache: {reused}."
        )
        self._update_model_action_state()
        infoBox("Model installed", f"{model_label(model_name)} is now available in the runtime cache.")
        self.refresh_model_inventory()

    def _on_model_download_failed(self, message: str) -> None:
        if self._active_model_download_name is None:
            self.model_status_label.setText(f"Shared model download failed: {message}")
            self.refresh_model_inventory()
            return
        self._active_model_download_name = None
        self.model_status_label.setText(f"Model install failed: {message}")
        self._update_model_action_state()
        errorBox("Model install failed", message)
        self.refresh_model_inventory()

    def _on_model_download_cancelled(self) -> None:
        if self._active_model_download_name is None:
            self.model_status_label.setText("Shared model download cancelled. Partial cache files were retained.")
            self.refresh_model_inventory()
            return
        self._active_model_download_name = None
        self.model_status_label.setText("Model download cancelled. Cached partial files will be reused on retry.")
        self._update_model_action_state()
        self.refresh_model_inventory()

    def _delete_selected_model_cache(self) -> None:
        model_name = self._selected_model_name()
        if not model_name:
            return
        if self._model_actions_busy() or not self.can_clear_rebuildable_caches():
            self.model_status_label.setText("Wait for active production and model tasks before deleting cached files.")
            return
        if not confirmBox(
            "Delete cached model download?",
            (
                f"This removes cached download files for {model_label(model_name)}.\n\n"
                "Packaged model_assets are preserved. The next run may ask to download again."
            ),
            parent=self,
        ):
            return
        self.model_status_label.setText(f"Deleting cached files for {model_label(model_name)}...")

        def _run(progress, cancel_check):
            return self.model_asset_service.delete_cached_model(
                model_name,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        self._model_job = job

        def _done(result: object) -> None:
            removed, failures = result if isinstance(result, tuple) and len(result) == 2 else ((), ("Invalid result",))
            if failures:
                errorBox("Model cache delete completed with errors", "\n".join(list(failures)[:8]))
            else:
                infoBox("Model cache deleted", f"Removed {len(removed)} cached item(s) for {model_label(model_name)}.")
            self.refresh_model_inventory()

        def _failed(message: str) -> None:
            self.model_status_label.setText(f"Model cache delete failed: {message}")
            errorBox("Model cache delete failed", message)

        job.progress.connect(lambda _value, text: self.model_status_label.setText(str(text)))
        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(
            lambda: self.model_status_label.setText("Model cache deletion cancelled. Already removed files remain removed.")
        )
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="model_delete")
        self._model_thread = thread
        self._update_model_action_state()

    def _refresh_face_model_inventory(self) -> None:
        items = list(self.face_model_installer.inventory())
        current_id = str(self.installed_face_model_combo.currentData() or "")
        self.installed_face_model_combo.blockSignals(True)
        self.installed_face_model_combo.clear()
        installed = [item for item in items if bool(item.installed)]
        for item in installed:
            size_text = self._format_bytes(int(item.size_bytes))
            self.installed_face_model_combo.addItem(
                f"{item.display_name} — {item.status}, {size_text}",
                item.bundle_id,
            )
        if current_id:
            index = self.installed_face_model_combo.findData(current_id)
            if index >= 0:
                self.installed_face_model_combo.setCurrentIndex(index)
        self.installed_face_model_combo.blockSignals(False)
        self._populate_face_model_inventory_table()
        if installed:
            verified = sum("verified" in str(item.status) for item in installed)
            self.face_model_status_label.setText(
                f"{len(installed)} face component(s) installed; {verified} managed component(s) verified. "
                f"Cache: {self.face_model_installer.runtime_root()}"
            )
        else:
            self.face_model_status_label.setText(
                "No optional face pack is installed. The built-in FaceNet row above must also be installed "
                "before using the built-in face embedder."
            )
        self._update_face_model_action_state()

    def _update_external_face_model_root_label(self) -> None:
        root = str(self._external_face_model_root or "").strip()
        self.external_face_model_root_label.setText(
            f"External model folder: {root}" if root else "External model folder: not selected"
        )
        self.external_face_model_root_label.setToolTip(root or "Choose a folder containing downloaded face-model ONNX files.")

    def _choose_external_face_model_root(self) -> None:
        current = str(self._external_face_model_root or self.face_model_installer.runtime_root())
        selected = QFileDialog.getExistingDirectory(self, "Choose Face Model Folder", current)
        if not selected:
            return
        self._external_face_model_root = str(Path(selected).expanduser())
        self._update_external_face_model_root_label()
        self._refresh_face_model_inventory()

    def _open_external_face_model_root(self) -> None:
        root = str(self._external_face_model_root or "").strip()
        if root:
            self._open_path(Path(root))

    def _populate_face_model_inventory_table(self) -> None:
        from app.services.face_search import list_face_detector_bundles, list_face_embedder_bundles

        rows: list[tuple[str, str, str, str, str]] = []
        root = self._external_face_model_root
        for bundle in list_face_detector_bundles(root, "human"):
            state = "Built-in" if bundle.source_kind == "builtin" and bundle.available else (
                "Downloaded" if bundle.available else f"Install for {self._hardware_label(bundle.hardware_class)}"
            )
            location = str(bundle.detector_path or bundle.bundle_dir or "")
            rows.append((bundle.display_name, "Detector", bundle.hardware_class or "CPU/GPU", state, location))
        for bundle in list_face_embedder_bundles(root, "human"):
            state = "Built-in" if bundle.source_kind == "builtin" and bundle.available else (
                "Downloaded" if bundle.available else f"Install for {self._hardware_label(bundle.hardware_class)}"
            )
            location = str(bundle.embedder_path or bundle.bundle_dir or "")
            rows.append((bundle.display_name, "Embedder", bundle.hardware_class or "CPU/GPU", state, location))
        rows.sort(key=lambda row: (row[1], row[0].casefold()))
        table = self.face_model_inventory_table
        table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, value in enumerate(row):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                table.setItem(row_index, column, item)
        table.resizeRowsToContents()

    @staticmethod
    def _hardware_label(value: str) -> str:
        text = str(value or "").casefold()
        has_cpu = "cpu" in text
        has_gpu = "gpu" in text or "cuda" in text
        if has_cpu and has_gpu:
            return "CPU/GPU"
        if has_gpu:
            return "GPU"
        if has_cpu:
            return "CPU"
        return "CPU/GPU"

    def _update_face_model_action_state(self) -> None:
        if not hasattr(self, "install_face_model_pack_button"):
            return
        busy = (
            self._thread_is_running(self._face_model_thread)
            or self._thread_is_running(self._model_thread)
            or self.model_download_controller.is_running()
        )
        mutation_allowed = bool(self.can_clear_rebuildable_caches())
        self.face_model_pack_combo.setEnabled(not busy)
        self.install_face_model_pack_button.setEnabled((not busy) and mutation_allowed)
        has_installed = self.installed_face_model_combo.count() > 0
        self.installed_face_model_combo.setEnabled(not busy and has_installed)
        self.delete_face_model_button.setEnabled((not busy) and mutation_allowed and has_installed)
        self.open_face_model_cache_button.setEnabled(not busy)
        self.clear_face_download_cache_button.setEnabled((not busy) and mutation_allowed)
        self.choose_external_face_model_root_button.setEnabled(not busy)
        self.open_external_face_model_root_button.setEnabled(not busy and bool(self._external_face_model_root))

    def _install_selected_face_model_pack(self) -> None:
        profile_id = str(self.face_model_pack_combo.currentData() or "latest_gpu")
        label = self.face_model_pack_combo.currentText()
        installers = {
            "recommended": self.face_model_installer.install_recommended,
            "edge": self.face_model_installer.install_edge,
            "accuracy": self.face_model_installer.install_accuracy,
            "latest_gpu": self.face_model_installer.install_latest_gpu,
            "max_accuracy": self.face_model_installer.install_max_accuracy,
            "opencv_cpu": self.face_model_installer.install_opencv_cpu,
            "sface": self.face_model_installer.install_sface,
            "yunet": self.face_model_installer.install_yunet,
            "yolo": self.face_model_installer.install_yolo,
        }
        installer = installers.get(profile_id)
        if installer is None:
            self.face_model_status_label.setText("Unknown face model pack.")
            return
        if self._model_actions_busy() or not self.can_clear_rebuildable_caches():
            self.face_model_status_label.setText("Wait for active production and model tasks before installing a face pack.")
            return
        if not confirmBox(
            "Install face inference pack?",
            (
                f"Pack: {label}\n"
                f"Managed model directory: {self.face_model_installer.runtime_root()}\n"
                f"Reusable download cache: {Path(self.app_settings.cache_dir) / 'face_model_downloads'}\n\n"
                "Downloads are checksum-verified, resumable, deduplicated across ClusterLens processes, "
                "visible in Jobs, and cancellable.\n\nContinue?"
            ),
            parent=self,
        ):
            return
        success_message = None
        if profile_id == "latest_gpu":
            success_message = lambda result: (
                "Default GPU face pipeline ready: SCRFD 10G + ArcFace R100. "
                f"Verified components: {', '.join(str(item) for item in (result or ())) or 'reused from cache'}."
            )
        self._start_face_model_job(label, installer, success_message=success_message)

    def _start_face_model_job(
        self,
        label: str,
        installer,
        *,
        success_message: Callable[[object], str] | None = None,
        mark_inventory_changed: bool = True,
        job_role: str = "face_model",
    ) -> None:
        if self._thread_is_running(self._face_model_thread):
            return
        job = AsyncJob(lambda progress, cancel_check: installer(progress, cancel_check))
        self._face_model_job = job
        self.face_model_status_label.setText(f"{label}: preparing shared cache...")

        def _done(result: object) -> None:
            bundle_ids = tuple(str(item) for item in result) if isinstance(result, (list, tuple)) else ()
            if mark_inventory_changed:
                self._face_model_inventory_changed = bool(bundle_ids) or self._face_model_inventory_changed
            self._refresh_face_model_inventory()
            if success_message is not None:
                self.face_model_status_label.setText(str(success_message(result)))
            else:
                self.face_model_status_label.setText(
                    f"{label} ready: {', '.join(bundle_ids) if bundle_ids else 'all files already present'}."
                )

        def _failed(message: str) -> None:
            self.face_model_status_label.setText(f"{label} failed: {message}")
            errorBox("Face model task failed", message)

        job.progress.connect(lambda value, text: self.face_model_status_label.setText(
            f"{text}{'' if int(value) < 0 else f' ({int(value)}%)'}"
        ))
        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(
            lambda: self.face_model_status_label.setText(
                f"{label} cancelled. Completed files and resumable partial downloads were retained."
            )
        )
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role=job_role)
        self._face_model_thread = thread
        self._update_model_action_state()

    def _delete_selected_face_model(self) -> None:
        bundle_id = str(self.installed_face_model_combo.currentData() or "").strip()
        label = self.installed_face_model_combo.currentText()
        if not bundle_id or self._model_actions_busy() or not self.can_clear_rebuildable_caches():
            return
        if not confirmBox(
            "Delete installed face component?",
            (
                f"Delete {label} from the managed face-model directory?\n\n"
                "The verified download cache is retained, so reinstalling does not download the same archive twice."
            ),
            parent=self,
        ):
            return

        def _delete(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            removed, failures = self.face_model_installer.delete_installed_model(bundle_id)
            if failures:
                raise RuntimeError("\n".join(failures))
            return (bundle_id,) if removed else ()

        self._start_face_model_job("Delete face component", _delete, job_role="face_model_delete")

    def _clear_face_download_cache(self) -> None:
        if self._model_actions_busy() or not self.can_clear_rebuildable_caches():
            self.face_model_status_label.setText("Wait for active production and model tasks before clearing downloads.")
            return
        cache_dir = self.face_model_installer.download_cache_dir()
        if not confirmBox(
            "Clear reusable face downloads?",
            (
                f"Delete verified archives and resumable partial downloads from:\n{cache_dir}\n\n"
                "Installed face components remain available. A future reinstall may need to download its source archive again. "
                "The action is shown in Jobs and can be cancelled between files."
            ),
            parent=self,
        ):
            return

        def _clear(progress, cancel_check):
            removed, freed_bytes, failures = self.face_model_installer.clear_download_cache(progress, cancel_check)
            if failures:
                raise RuntimeError("\n".join(failures[:8]))
            return {"removed": len(removed), "freed_bytes": int(freed_bytes)}

        self._start_face_model_job(
            "Clear face download cache",
            _clear,
            success_message=lambda result: (
                f"Face download cache cleared: {int(result.get('removed', 0))} file(s), "
                f"{self._format_bytes(int(result.get('freed_bytes', 0)))} freed."
                if isinstance(result, dict)
                else "Face download cache cleared."
            ),
            mark_inventory_changed=False,
            job_role="face_cache_clear",
        )

    def face_model_inventory_changed(self) -> bool:
        return bool(self._face_model_inventory_changed)

    def refresh_operation_journal(self) -> None:
        if self._thread_is_running(self._journal_refresh_thread):
            return
        self.journal_status_label.setText("Loading recovery history...")
        job = AsyncJob(lambda _progress, _cancel_check: list(reversed(self.gallery_action_service.read_audit_entries(limit=1000))))
        self._journal_refresh_job = job

        def _done(result: object) -> None:
            entries = list(result) if isinstance(result, list) else []
            self._journal_entries = entries
            self.operation_journal_model.set_entries(entries)
            self.operation_journal_table.resizeRowsToContents()
            if entries:
                self.journal_status_label.setText(
                    f"Showing {self.operation_journal_model.rowCount()} of {len(entries)} recovery operations. Scroll to load more."
                )
            else:
                self.journal_status_label.setText("No recoverable file operations have been recorded yet.")
            self._update_journal_action_state()

        def _failed(message: str) -> None:
            self.journal_status_label.setText(f"Recovery history could not be loaded: {message}")

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: self.journal_status_label.setText("Recovery history refresh cancelled."))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="journal_refresh")
        self._journal_refresh_thread = thread

    def _selected_journal_entry(self) -> dict[str, object] | None:
        current = self.operation_journal_table.selectionModel().currentIndex()
        row = current.row() if current.isValid() else -1
        if row < 0:
            return None
        return self.operation_journal_model.entry_at(row)

    def _selected_restorable_paths(self) -> list[tuple[str, str]]:
        entry = self._selected_journal_entry()
        if not entry:
            return []
        if str(entry.get("operation") or "") not in {"delete_to_trash", "move", "rename"}:
            return []
        changed = entry.get("changed_paths") or []
        pairs: list[tuple[str, str]] = []
        for pair in changed:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pairs.append((str(pair[0]), str(pair[1])))
        return pairs

    def _selected_retryable_paths(self) -> list[str]:
        entry = self._selected_journal_entry()
        if not entry or str(entry.get("operation") or "") not in {"copy", "move", "delete_to_trash"}:
            return []
        requested = [str(path) for path in list(entry.get("requested_paths") or []) if str(path)]
        completed = {str(path) for path in list(entry.get("affected_paths") or []) if str(path)}
        for pair in list(entry.get("changed_paths") or []):
            if isinstance(pair, (list, tuple)) and pair:
                completed.add(str(pair[0]))
        return [path for path in requested if path not in completed]

    def _update_journal_action_state(self) -> None:
        restorable = bool(self._selected_restorable_paths())
        busy = self._thread_is_running(self._journal_restore_thread)
        self.refresh_journal_button.setEnabled(not busy)
        self.reveal_journal_target_button.setEnabled((not busy) and restorable)
        self.retry_journal_button.setEnabled((not busy) and bool(self._selected_retryable_paths()) and not bool(self.read_only_mode.isChecked()))
        self.restore_journal_button.setEnabled((not busy) and restorable and not bool(self.read_only_mode.isChecked()))
        self.restore_unique_journal_button.setEnabled((not busy) and restorable and not bool(self.read_only_mode.isChecked()))

    def _on_journal_selection_changed(self) -> None:
        self._update_journal_action_state()
        entry = self._selected_journal_entry() or {}
        results = list(entry.get("file_results") or [])
        if not results:
            self.journal_file_results.clear()
            return
        lines = []
        for result in results:
            if not isinstance(result, dict):
                continue
            source = str(result.get("source") or "")
            target = str(result.get("target") or "")
            status = str(result.get("status") or "unknown")
            error = str(result.get("error") or "")
            line = f"{status}: {source}"
            if target and target != source:
                line += f" → {target}"
            if error:
                line += f" — {error}"
            lines.append(line)
        self.journal_file_results.setPlainText("\n".join(lines))

    def _retry_selected_journal_operation(self) -> None:
        if self.read_only_mode.isChecked():
            infoBox("Read-only mode", "Turn off read-only mode before retrying a file operation.")
            return
        entry = self._selected_journal_entry() or {}
        paths = self._selected_retryable_paths()
        operation = str(entry.get("operation") or "")
        destination = str(entry.get("destination") or "")
        if not paths:
            self.journal_status_label.setText("The selected operation has no failed files that can be retried.")
            return
        if not confirmBox(
            "Retry failed files?",
            (
                f"Action: {OperationJournalTableModel._operation_label(operation)}\n"
                f"Files to retry: {len(paths)}\n"
                f"Destination: {destination or f'{CLUSTERLENS_TRASH_DIR_NAME} beside each source folder'}\n"
                "Recovery: moved files remain available from this Recovery view."
            ),
            parent=self,
        ):
            self.journal_status_label.setText("Retry cancelled. No files were changed.")
            return

        def _run(progress, cancel_check):
            if operation == "copy":
                return self.gallery_action_service.copy_to_directory(paths, destination, progress, cancel_check)
            if operation == "move":
                return self.gallery_action_service.move_to_directory(paths, destination, progress, cancel_check)
            return self.gallery_action_service.move_to_trash(paths, progress, cancel_check)

        self._start_recovery_job("Retrying failed files", _run)

    def _start_recovery_job(self, label: str, fn) -> None:
        self.journal_status_label.setText(f"{label}...")
        job = AsyncJob(fn)
        self._journal_restore_job = job

        def _done(result: object) -> None:
            failures = list(getattr(result, "failures", []))
            changed = len(list(getattr(result, "changed_paths", []))) + len(list(getattr(result, "affected_paths", [])))
            self.journal_status_label.setText(f"{label} complete: {changed} succeeded, {len(failures)} failed.")
            self.refresh_operation_journal()
            self._update_journal_action_state()

        def _failed(message: str) -> None:
            self.journal_status_label.setText(f"{label} failed: {message}")
            errorBox(f"{label} failed", message)
            self._update_journal_action_state()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: self.journal_status_label.setText(f"{label} cancelled."))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="journal_restore")
        self._journal_restore_thread = thread
        self._update_journal_action_state()

    def _reveal_selected_journal_target(self) -> None:
        pairs = self._selected_restorable_paths()
        if not pairs:
            self.journal_status_label.setText("Select a move, rename, or ClusterLens Trash operation with changed paths to reveal.")
            return
        _original, current = pairs[0]
        target = Path(str(current))
        self._open_path(target if target.exists() else target.parent)

    def _restore_selected_journal_operation(self, *, conflict_policy: str = "skip") -> None:
        if self.read_only_mode.isChecked():
            infoBox("Read-only mode", "Turn off read-only safety mode before restoring files from the operation journal.")
            self._update_journal_action_state()
            return
        pairs = self._selected_restorable_paths()
        if not pairs:
            self.journal_status_label.setText("Select a move, rename, or ClusterLens Trash operation with changed paths to restore.")
            return
        conflict_preview = ""
        if str(conflict_policy).lower() == "unique_name":
            original = Path(pairs[0][0])
            preview = self.gallery_action_service._unique_target(original.parent, original.name)
            conflict_preview = f"\nExample restored path: {preview}"
        if not confirmBox(
            "Restore selected operation?",
            (
                "Action: Restore files\n"
                f"Target: original folders from the selected operation\n"
                f"Files affected: {len(pairs)}\n"
                "Result: available moved files return to their original folders.\n"
                "Recovery: this restore is journaled, but is not automatically undone.\n\n"
                "Missing moved files are reported as failures. "
                + (
                    f"Existing original paths get unique restored names.{conflict_preview}"
                    if str(conflict_policy).lower() == "unique_name"
                    else "Existing original paths are skipped, not overwritten."
                )
            ),
            parent=self,
        ):
            return
        self.journal_status_label.setText("Restoring selected operation...")
        self._update_journal_action_state()

        job = AsyncJob(
            lambda progress, cancel_check: self.gallery_action_service.restore_changed_paths(
                pairs,
                progress,
                cancel_check,
                conflict_policy=conflict_policy,
            )
        )
        self._journal_restore_job = job

        def _done(result: object) -> None:
            failures = list(getattr(result, "failures", []))
            success_count = len(getattr(result, "changed_paths", []))
            self.journal_status_label.setText(f"Restore complete: {success_count} restored, {len(failures)} failure(s).")
            if failures:
                errorBox("Restore completed with errors", "\n".join(failures[:8]))
            else:
                infoBox("Restore complete", f"Restored {success_count} file(s).")
            self.refresh_operation_journal()
            self._update_journal_action_state()

        def _failed(message: str) -> None:
            self.journal_status_label.setText(f"Restore failed: {message}")
            errorBox("Restore failed", message)
            self._update_journal_action_state()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: self.journal_status_label.setText("Restore cancelled."))
        job.cancelled.connect(self._update_journal_action_state)
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="journal_restore")
        self._journal_restore_thread = thread
        self._update_journal_action_state()

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        size = float(max(0, int(size_bytes)))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{size_bytes} B"

    def _format_generated_storage(self, summary: GeneratedStorageSummary) -> str:
        lines = [
            f"Runtime root: {summary.runtime_root}",
            f"Config location: {summary.config_location or 'platform defaults'}",
            f"Cache root: {summary.cache_root}",
            "",
        ]
        labels = {
            "logs": "Logs (recovery journal preserved)",
            "thumbnails": "Thumbnails",
            "rebuildable_caches": "Rebuildable caches",
            "library_catalog": "Library catalog cache (roots and review choices preserved when cleared)",
            "face_databases": "Face databases",
            "ann_files": "Face ANN files",
            "model_caches": "Model caches",
            "temp_files": "Temp files",
            "crash_reports": "Crash reports",
            "support_bundles": "Support bundles",
            "benchmarks": "Benchmark reports",
            "model_assets": "Installed model assets",
        }
        for key, label in labels.items():
            paths = tuple(summary.target_paths.get(key, ()))
            lines.append(f"{label}: {self._format_bytes(summary.target_bytes.get(key, 0))}")
            lines.append(f"  {', '.join(paths) if paths else '(not configured)'}")
        lines.extend(["", f"Total generated storage size: {self._format_bytes(summary.total_bytes)}"])
        return "\n".join(lines)

    def refresh_runtime_diagnostics(self, details: dict[str, object] | None = None) -> None:
        if details is None:
            details = self.runtime_service.diagnostics(str(self.execution_mode.currentData() or "auto"))
        capabilities = details["capabilities"]
        policy = details["policy"]
        packages = details.get("packages") or {}
        hdbscan_status = details.get("cuml_hdbscan") or {}
        optional_details = details.get("optional_details") or {}
        cpu_details = details.get("cpu_details") or {}
        gpu_probe_policy = details.get("gpu_probe_policy")
        remediation = list(details.get("remediation") or [])
        profile = select_performance_profile(str(self.performance_profile.currentData() or "balanced"), self.system_resources)
        vector_device = "GPU (Torch CUDA)" if policy.torch_device == "cuda" else "CPU (NumPy/BLAS)"
        hdbscan_available = bool(hdbscan_status.get("available"))
        hdbscan_detail = str(hdbscan_status.get("detail") or "cuML HDBSCAN was not verified.")
        hdbscan_device = "GPU (cuML)" if policy.torch_device == "cuda" and hdbscan_available else "CPU (native HDBSCAN)"
        simd_features = ", ".join(cpu_details.get("numpy_simd") or ()) or "runtime dispatch unavailable"
        preferred_cpu = policy.preferred_mode == "cpu"
        torch_status = (
            f"READY — {capabilities.cuda_device_name or 'CUDA device'} ({capabilities.cuda_total_memory_mb} MB)"
            if capabilities.torch_cuda_available
            else "UNAVAILABLE — this process cannot use a CUDA Torch device"
        )
        checked_onnx_provider = str(getattr(gpu_probe_policy, "onnx_provider", policy.onnx_provider))
        if checked_onnx_provider == "CUDAExecutionProvider" and preferred_cpu:
            onnx_status = "AVAILABLE — CUDAExecutionProvider was verified; CPU mode is currently active"
        elif checked_onnx_provider == "CUDAExecutionProvider":
            onnx_status = "READY — SCRFD and ArcFace ONNX inference will use CUDAExecutionProvider"
        elif preferred_cpu and gpu_probe_policy is None:
            onnx_status = "NOT CHECKED — CPU mode was selected; use Rescan GPU Resources to check it explicitly"
        elif capabilities.has_onnx_cuda:
            onnx_status = "UNAVAILABLE — CUDA ONNX is listed but failed provider verification"
        else:
            onnx_status = "UNAVAILABLE — this process has no verified CUDAExecutionProvider"
        if capabilities.torch_cuda_available and hdbscan_available and preferred_cpu:
            hdbscan_status_text = "AVAILABLE — cuML HDBSCAN is installed; CPU mode is currently active"
        elif capabilities.torch_cuda_available and hdbscan_available:
            hdbscan_status_text = "READY — cuML HDBSCAN is installed and importable"
        elif not capabilities.torch_cuda_available:
            hdbscan_status_text = "UNAVAILABLE — no usable CUDA Torch device"
        else:
            hdbscan_status_text = f"UNAVAILABLE — {hdbscan_detail}"

        text = [
            "GPU acceleration checklist (local inspection; no model downloads):",
            f"1. NVIDIA / Torch CUDA: {torch_status}",
            f"2. CUDA ONNX for SCRFD / ArcFace: {onnx_status}",
            f"3. cuML HDBSCAN: {hdbscan_status_text}",
            "",
            "Current process and model artifacts:",
            f"- Python: {sys.executable}",
            f"- Managed face-model directory: {face_model_runtime_root_dir(self.app_settings)}",
            f"- Clustering-model cache: {self.app_settings.cache_dir}",
            "- SCRFD and ArcFace ONNX files are provider-neutral: CUDA uses the existing files through ONNX Runtime, not a separate GPU download.",
            "- This rescan only probes installed packages/providers. It never downloads, moves, or opens model files or user photos.",
            "",
            f"Preferred mode: {policy.preferred_mode}",
            f"Effective mode: {policy.effective_mode}",
            f"Precision: {self.precision_mode.currentData() or 'auto'}",
            f"Performance profile: {profile.name}",
            f"Warm worker between runs: {'enabled' if self.keep_worker_warm.isChecked() else 'disabled'}",
            f"CPU batch size: {self.batch_size_cpu.value()}",
            f"CUDA batch size: {self.batch_size_gpu.value()}",
            f"Decode workers: {self.decode_workers.value()}",
            f"CUDA VRAM headroom MB: {self.vram_headroom_mb.value()}",
            "",
            "Packages:",
            f"- torch: {packages.get('torch') or '-'}",
            f"- onnx: {packages.get('onnx') or '-'}",
            f"- onnxruntime: {packages.get('onnxruntime') or '-'}",
            f"- onnxruntime-gpu: {packages.get('onnxruntime-gpu') or '-'}",
            f"- cuml: {packages.get('cuml') or '-'}",
            f"- cupy: {packages.get('cupy') or '-'}",
            f"- hf_xet: {packages.get('hf_xet') or '-'}",
            f"- flash-attn: {packages.get('flash-attn') or '-'}",
            "",
            f"Torch device: {policy.torch_device}",
            f"ONNX provider: {policy.onnx_provider}",
            f"ONNX face indexing: {'GPU (CUDA)' if policy.onnx_provider == 'CUDAExecutionProvider' else 'CPU'}",
            f"Built-in Torch face indexing: {'GPU (CUDA)' if policy.torch_device == 'cuda' else 'CPU'}",
            f"Semantic PCA: {vector_device}",
            f"Cosine K-means: {vector_device}",
            f"HDBSCAN: {hdbscan_device}",
            f"HDBSCAN provider: {hdbscan_detail}",
            f"Cluster quality scoring: {vector_device}",
            f"Graph neighbor search: {vector_device}",
            f"Dense similarity scoring: {vector_device}",
            f"Face similarity and identity matrices: {vector_device}",
            f"CUDA build: {capabilities.torch_cuda_build}",
            f"CUDA available: {capabilities.torch_cuda_available}",
            f"CUDA device: {capabilities.cuda_device_name or '-'}",
            f"CUDA memory MB: {capabilities.cuda_total_memory_mb}",
            f"ONNX version: {capabilities.onnx_version or '-'}",
            f"ONNX providers: {', '.join(capabilities.onnx_providers) if capabilities.onnx_providers else '-'}",
            f"Logical CPU cores: {profile.logical_cpu_count}",
            f"Thumbnail workers: {profile.thumbnail_workers}",
            f"Thumbnail prefetch rows: {profile.thumbnail_prefetch_rows}",
            f"Embedding preprocess workers: {profile.embedding_preprocess_workers}",
            f"Embedding cache entries: {profile.embedding_memory_cache_size}",
            f"Pixmap cache entries: {profile.pixmap_cache_size}",
            f"CPU SIMD dispatch: {simd_features}",
            f"CPU BLAS: {cpu_details.get('blas') or '-'} ({int(cpu_details.get('blas_threads') or 0)} threads)",
            f"CPU JPEG decode: {'libjpeg-turbo' if cpu_details.get('libjpeg_turbo') else 'standard Pillow codec'}",
            "",
            "Optional accelerators:",
            f"- hf_xet download accelerator: {'installed' if optional_details.get('hf_xet_installed') else 'missing'}",
            f"- Flash SDP enabled: {bool(optional_details.get('flash_sdp_enabled'))}",
            "",
            f"Reason: {policy.reason}",
        ]
        if self._last_verify:
            remediation = list(self._last_verify.get("remediation") or remediation)
            text.extend(["", "Runtime Verify:"])
            for label, key in (
                ("Torch smoke", "torch_smoke"),
                ("ONNX smoke", "onnx_smoke"),
                ("Flash attention verify", "flash_attention_smoke"),
            ):
                smoke = self._last_verify.get(key)
                if isinstance(smoke, dict) and smoke.get("ok"):
                    text.append(f"- {label}: OK")
                elif isinstance(smoke, dict):
                    text.append(f"- {label}: FAIL ({smoke.get('error')})")
                else:
                    text.append(f"- {label}: skipped")
        if remediation:
            text.extend(["", "Suggested remediation:"])
            text.extend(f"- {item}" for item in remediation)
        text.extend(
            [
                "",
                "Production notes:",
                "- CUDA acceleration requires a CUDA-enabled Torch build plus NVIDIA drivers.",
                "- HDBSCAN uses cuML on the pinned Linux CUDA runtime and reports a native-CPU fallback if cuML is missing or fails. Perceptual hashes, ORB, image decode, SQLite, filesystem I/O, and Qt event handling remain CPU work.",
                "- CPU vector fallbacks use contiguous batches through NumPy/BLAS; image, embedding, thumbnail, pixmap, and clustering-result caches remain bounded by the selected performance profile.",
                "- Keep worker warm is fastest for repeated clustering, but leaves the worker process and loaded model memory resident until disabled or app exit.",
                "- low_memory disables aggressive parallelism/caching; max_speed increases caches, allows all selected backends to run concurrently, and uses a faster, slightly less exact KMeans path on larger folders.",
                "- hf_xet is optional but improves Hugging Face download speed; include it in the packaged runtime if you ship an exe.",
                "- Flash attention is not bundled automatically. If that optimization matters, ship a Torch/CUDA stack that provides it and verify it on the target machine.",
                "- Restart the app after changing GPU/runtime packages.",
            ]
        )
        self.runtime_text.setPlainText("\n".join(text))

    def _update_runtime_action_state(self) -> None:
        busy = self._thread_is_running(self._verify_thread)
        self.refresh_runtime_button.setEnabled(not busy)
        self.verify_runtime_button.setEnabled(not busy)

    def _rescan_runtime_resources(self) -> None:
        if self._thread_is_running(self._verify_thread):
            return

        preferred_mode = str(self.execution_mode.currentData() or "auto")
        probe_mode = "cuda" if preferred_mode == "cpu" else preferred_mode

        def _run(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            progress(-1, "Checking the current process and NVIDIA / Torch CUDA (no models or photos are opened)...")
            self.runtime_service.detect(refresh=True)
            raise_if_cancelled(cancel_check)
            progress(-1, "Checking CUDA ONNX provider for SCRFD and ArcFace (no model download)...")
            gpu_probe_policy = self.runtime_service.select_policy(preferred_mode=probe_mode, refresh=True)
            raise_if_cancelled(cancel_check)
            progress(-1, "Checking installed cuML HDBSCAN and CPU fallback resources...")
            result = self.runtime_service.diagnostics(preferred_mode, refresh=False)
            result["gpu_probe_policy"] = gpu_probe_policy
            raise_if_cancelled(cancel_check)
            return result

        job = AsyncJob(_run)
        self._verify_job = job

        def _done(result: object) -> None:
            self._last_verify = None
            details = result if isinstance(result, dict) else None
            self.refresh_runtime_diagnostics(details)
            self.runtime_rescanned.emit()

        job.progress.connect(lambda _value, text: self.runtime_text.setPlainText(str(text)))
        job.completed.connect(_done)
        job.failed.connect(lambda message: errorBox("Runtime rescan failed", str(message)))
        job.cancelled.connect(
            lambda: self.runtime_text.setPlainText(
                "Runtime rescan cancelled. No execution settings were changed."
            )
        )
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="runtime_rescan")
        self._verify_thread = thread
        self._update_runtime_action_state()

    def _apply_profile_preview(self, *, update_tuning: bool = True) -> None:
        profile = select_performance_profile(str(self.performance_profile.currentData() or "balanced"), self.system_resources)
        if update_tuning:
            self.batch_size_cpu.setValue(profile.cpu_batch_size)
            self.batch_size_gpu.setValue(profile.gpu_batch_size)
            self.decode_workers.setValue(profile.embedding_preprocess_workers)
            self.vram_headroom_mb.setValue(profile.vram_headroom_mb)
        self.thumbnail_workers.setValue(profile.thumbnail_workers)
        self.prefetch_rows.setValue(profile.thumbnail_prefetch_rows)
        if hasattr(self, "runtime_text"):
            self.refresh_runtime_diagnostics()

    def _cache_actions_busy(self) -> bool:
        return any(job is not None for job in (self._cache_usage_job, self._cache_clear_job))

    def _update_cache_action_state(self) -> None:
        busy = self._cache_actions_busy()
        self.refresh_cache_usage_button.setEnabled(not busy)
        allow_clear = bool(self.clear_rebuildable_caches) and bool(self.can_clear_rebuildable_caches())
        self.clear_cache_button.setEnabled((not busy) and allow_clear)
        for button, callback in (
            (self.clear_runtime_temp_button, self.clear_runtime_temp_files),
            (self.clear_library_catalog_button, self.clear_library_catalog),
            (self.clear_face_storage_button, self.clear_face_storage),
            (self.clear_model_caches_button, self.clear_model_caches),
            (self.clear_logs_button, self.clear_logs),
            (self.clear_runtime_reports_button, self.clear_runtime_reports),
            (self.clear_model_assets_button, self.clear_model_assets),
        ):
            button.setEnabled((not busy) and bool(callback) and bool(self.can_clear_rebuildable_caches()))
        if not allow_clear and not busy:
            self.cache_status_label.setText("Cache clearing is unavailable while clustering or another production background task is running.")

    def refresh_cache_usage(self) -> None:
        if self.describe_rebuildable_caches is None and self.describe_generated_storage is None:
            self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            self.generated_storage_text.setPlainText("Generated storage usage is unavailable.")
            self._update_cache_action_state()
            return
        if self._thread_is_running(self._cache_usage_thread):
            return
        self.cache_status_label.setText("Scanning rebuildable cache usage...")
        self._update_cache_action_state()

        def _run_usage(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            result = None
            generated = None
            if self.describe_rebuildable_caches is not None:
                try:
                    result = self.describe_rebuildable_caches(
                        progress_callback=progress,
                        cancel_check=cancel_check,
                    )
                except TypeError as exc:
                    if "unexpected keyword" not in str(exc):
                        raise
                    result = self.describe_rebuildable_caches()
            if self.describe_generated_storage is not None:
                try:
                    generated = self.describe_generated_storage(
                        progress_callback=progress,
                        cancel_check=cancel_check,
                    )
                except TypeError as exc:
                    if "unexpected keyword" not in str(exc):
                        raise
                    generated = self.describe_generated_storage()
            raise_if_cancelled(cancel_check)
            return result, generated

        job = AsyncJob(_run_usage)
        self._cache_usage_job = job

        def _done(result: object) -> None:
            cache_result = result[0] if isinstance(result, tuple) and result else result
            generated_result = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            if isinstance(cache_result, CacheUsageSummary):
                lines = [f"Runtime cache root: {cache_result.cache_root}", ""]
                for name, size in cache_result.target_bytes.items():
                    lines.append(f"{name} {self._format_bytes(size)}")
                lines.extend(
                    [
                        "",
                        f"Total rebuildable cache size: {self._format_bytes(cache_result.total_bytes)}",
                        "",
                        "Preserved: image tags, logs, crash records, support bundles, model downloads, and model assets.",
                    ]
                )
                self.cache_usage_text.setPlainText("\n".join(lines))
                self.cache_status_label.setText("Rebuildable cache usage refreshed.")
            else:
                self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            if isinstance(generated_result, GeneratedStorageSummary):
                self.generated_storage_text.setPlainText(self._format_generated_storage(generated_result))
            elif self.describe_generated_storage is not None:
                self.generated_storage_text.setPlainText("Generated storage usage is unavailable.")
            self._update_cache_action_state()

        def _failed(message: str) -> None:
            self.cache_usage_text.setPlainText("Failed to scan cache usage.")
            self.generated_storage_text.setPlainText("Failed to scan generated storage usage.")
            self.cache_status_label.setText(f"Cache usage refresh failed: {message}")
            self._update_cache_action_state()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        def _cancelled() -> None:
            self.cache_usage_text.setPlainText("Cache usage scan cancelled.")
            self.generated_storage_text.setPlainText("Generated storage scan cancelled.")
            self.cache_status_label.setText("Cache usage scan cancelled.")
            self._update_cache_action_state()

        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_usage")
        self._cache_usage_thread = thread

    def _clear_rebuildable_caches(self) -> None:
        if self.clear_rebuildable_caches is None:
            return
        if not self.can_clear_rebuildable_caches():
            self.cache_status_label.setText("Wait for the current production background task to finish before clearing caches.")
            self._update_cache_action_state()
            return
        if not confirmBox(
            "Clear Rebuildable Production Caches?",
            "This removes embeddings, clustering results, indexes, thumbnails, ONNX exports, meanings, and temp files.\n\n"
            "Image tags, logs, crash records, support bundles, model downloads, and model assets are preserved.",
            parent=self,
        ):
            return
        self.cache_status_label.setText("Clearing rebuildable caches...")
        self._update_cache_action_state()

        def _run_clear(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            try:
                result = self.clear_rebuildable_caches(
                    progress_callback=progress,
                    cancel_check=cancel_check,
                )
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                result = self.clear_rebuildable_caches()
            raise_if_cancelled(cancel_check)
            return result

        job = AsyncJob(_run_clear)
        self._cache_clear_job = job

        def _done(result: object) -> None:
            if isinstance(result, CacheClearResult):
                self.cache_status_label.setText(
                    f"Cleared {len(result.cleared_targets)} targets and freed {self._format_bytes(result.freed_bytes)}."
                )
                if result.failures:
                    errorBox("Cache clear completed with errors", "\n".join(result.failures[:8]))
                else:
                    infoBox(
                        "Caches cleared",
                        f"Cleared {len(result.cleared_targets)} rebuildable targets and freed {self._format_bytes(result.freed_bytes)}.",
                    )
            else:
                self.cache_status_label.setText("Rebuildable caches cleared.")
            self._update_cache_action_state()
            self.refresh_cache_usage()

        def _failed(message: str) -> None:
            self.cache_status_label.setText(f"Cache clear failed: {message}")
            self._update_cache_action_state()
            errorBox("Cache clear failed", message)
            self.refresh_cache_usage()

        def _cancelled() -> None:
            self.cache_status_label.setText("Cache clear cancelled. Already removed rebuildable files remain removed.")
            self._update_cache_action_state()
            self.refresh_cache_usage()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_clear")
        self._cache_clear_thread = thread

    def _clear_generated_storage(
        self,
        title: str,
        message: str,
        callback: Callable[[], CacheClearResult] | None,
    ) -> None:
        if callback is None:
            return
        if not self.can_clear_rebuildable_caches():
            self.cache_status_label.setText(
                "Wait for the current production background task to finish before clearing generated data."
            )
            self._update_cache_action_state()
            return
        if not confirmBox(title, message, parent=self):
            return
        self.cache_status_label.setText(title.rstrip("?") + "...")
        self._update_cache_action_state()

        def _run_clear(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            try:
                result = callback(progress_callback=progress, cancel_check=cancel_check)
            except TypeError as exc:
                if "unexpected keyword" not in str(exc):
                    raise
                result = callback()
            raise_if_cancelled(cancel_check)
            return result

        job = AsyncJob(_run_clear)
        self._cache_clear_job = job

        def _done(result: object) -> None:
            if isinstance(result, CacheClearResult):
                self.cache_status_label.setText(
                    f"Cleared {len(result.cleared_targets)} target(s) and freed {self._format_bytes(result.freed_bytes)}."
                )
                if result.failures:
                    errorBox("Storage clear completed with errors", "\n".join(result.failures[:8]))
                else:
                    infoBox(
                        "Storage cleared",
                        f"Cleared {len(result.cleared_targets)} target(s) and freed {self._format_bytes(result.freed_bytes)}.",
                    )
            else:
                self.cache_status_label.setText("Generated storage cleared.")
            self._update_cache_action_state()
            self.refresh_cache_usage()

        def _failed(message_text: str) -> None:
            self.cache_status_label.setText(f"Storage clear failed: {message_text}")
            self._update_cache_action_state()
            errorBox("Storage clear failed", message_text)
            self.refresh_cache_usage()

        def _cancelled() -> None:
            self.cache_status_label.setText("Storage clear cancelled. Already removed generated files remain removed.")
            self._update_cache_action_state()
            self.refresh_cache_usage()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_clear")
        self._cache_clear_thread = thread

    def _verify_gpu(self) -> None:
        if self._thread_is_running(self._verify_thread):
            return

        def _run(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            progress(-1, "Probing runtimes...")
            result = self.runtime_service.verify(str(self.execution_mode.currentData() or "auto"))
            raise_if_cancelled(cancel_check)
            return result

        job = AsyncJob(_run)
        self._verify_job = job

        def _done(result: object) -> None:
            self._last_verify = result if isinstance(result, dict) else None
            self.refresh_runtime_diagnostics()

        job.completed.connect(_done)
        job.failed.connect(lambda message: errorBox("Runtime verify failed", str(message)))
        job.cancelled.connect(
            lambda: self.runtime_text.setPlainText(
                "Runtime verification cancelled. No execution settings were changed."
            )
        )
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="verify")
        self._verify_thread = thread
        self._update_runtime_action_state()

    def _open_path(self, path: Path) -> None:
        try:
            open_path_in_shell(path)
        except Exception as exc:
            errorBox("Open failed", str(exc))

    def _view_last_crash(self) -> None:
        if not self.runtime_layout.last_crash_json.exists():
            infoBox("No crash recorded", "No crash record has been written for this production runtime yet.")
            return
        self._open_path(self.runtime_layout.last_crash_json)

    def _export_support_bundle(self) -> None:
        try:
            metadata = dict(self.support_metadata_provider() or {})
            bundle_path = export_support_bundle(
                self.runtime_layout,
                bundle_name="support_bundle",
                metadata=metadata,
            )
        except Exception as exc:
            errorBox("Bundle export failed", str(exc))
            return
        infoBox("Support bundle exported", str(bundle_path))

    def refresh_log_viewer(self) -> None:
        self.log_viewer.setPlainText("\n".join(self._read_tail(self.runtime_layout.app_log, limit=220)))

    def _about_text(self) -> str:
        diagnostics = self.runtime_service.diagnostics(
            str(self.settings_registry.get(self.settings_store, "runtime/preferred_mode", self.app_settings.preferred_execution_mode))
        )
        packages = diagnostics.get("packages") or {}
        return "\n".join(
            [
                f"App: Image Clustering PyQt Production {self._app_version()}",
                f"Python: {sys.version.split()[0]}",
                f"Executable: {sys.executable}",
                f"Frozen executable: {bool(getattr(sys, 'frozen', False))}",
                "",
                f"Runtime root: {self.runtime_layout.root}",
                f"Settings file: {self.settings_store.fileName()}",
                f"Logs: {self.runtime_layout.logs_dir}",
                f"Cache: {self.runtime_layout.cache_dir}",
                f"Model assets: {self.runtime_layout.model_assets_dir}",
                f"Crash reports: {self.runtime_layout.crash_dir}",
                "",
                "Core runtime packages:",
                f"- torch: {packages.get('torch') or '-'}",
                f"- onnxruntime: {packages.get('onnxruntime') or '-'}",
                f"- onnx: {packages.get('onnx') or '-'}",
                f"- cuml: {packages.get('cuml') or '-'}",
                f"- cupy: {packages.get('cupy') or '-'}",
                f"- hf_xet: {packages.get('hf_xet') or '-'}",
                f"- flash-attn: {packages.get('flash-attn') or '-'}",
                "",
                "Support policy:",
                "- Logs, crash records, operation journals, and support bundles are kept under the runtime root.",
                "- Rebuildable caches and temp files can be cleared from Storage or the footer.",
                "- Model downloads are never required without an explicit prompt unless offline mode is disabled and the user approves.",
            ]
        )

    @staticmethod
    def _app_version() -> str:
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        try:
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("version"):
                    return line.split("=", 1)[1].strip().strip('"')
        except OSError:
            pass
        return "unknown"

    @staticmethod
    def _read_tail(path: Path, *, limit: int) -> list[str]:
        if not path.exists():
            return [f"No log file found at {path}."]
        rows: deque[str] = deque(maxlen=max(1, int(limit)))
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    rows.append(line.rstrip())
        except OSError as exc:
            return [f"Could not read {path}: {exc}"]
        return list(rows)

    @staticmethod
    def _thread_is_running(thread) -> bool:
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except RuntimeError:
            return False

    def _track_thread(self, thread, job: object | None, *, role: str) -> None:
        if thread is None:
            return
        self._thread_roles[thread] = (str(role), job)
        if self.job_manager is not None and job is not None:
            label = {
                "cache_usage": "Scanning cache usage",
                "cache_clear": "Clearing rebuildable caches",
                "verify": "Verifying runtime",
                "runtime_rescan": "Rescanning runtime resources",
                "model": "Installing model",
                "model_delete": "Deleting cached model",
                "model_inventory": "Refreshing model inventory",
                "face_model": "Installing face inference pack",
                "face_model_delete": "Deleting installed face component",
                "face_cache_clear": "Clearing reusable face downloads",
                "journal_restore": "Recovering files",
                "journal_refresh": "Refreshing recovery history",
            }.get(str(role), "Settings task")
            job_id = self.job_manager.register_job(
                label,
                cancel_fn=getattr(job, "cancel", None),
                origin="Settings",
            )
            self._settings_job_ids[thread] = job_id
            progress_signal = getattr(job, "progress", None)
            if progress_signal is not None:
                progress_signal.connect(
                    lambda value, text, job_id=job_id: self.job_manager.update(
                        job_id,
                        progress=value,
                        text=text,
                    )
                )
                progress_signal.connect(self._on_settings_job_progress)
            getattr(job, "completed").connect(
                lambda _result, job_id=job_id: self.job_manager.finish(job_id, status="finished")
            )
            getattr(job, "failed").connect(
                lambda message, job_id=job_id: self.job_manager.finish(job_id, status="failed", error=message)
            )
            getattr(job, "cancelled").connect(
                lambda job_id=job_id: self.job_manager.finish(job_id, status="cancelled")
            )
            self.operation_progress_bar.setRange(0, 0)
            self.operation_progress_bar.show()
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._update_settings_cancel_state()

    def _on_settings_job_progress(self, value: int, _text: str) -> None:
        if int(value) < 0:
            self.operation_progress_bar.setRange(0, 0)
        else:
            self.operation_progress_bar.setRange(0, 100)
            self.operation_progress_bar.setValue(max(0, min(100, int(value))))
        self.operation_progress_bar.show()

    def _release_finished_thread(self, thread) -> None:
        if thread is None:
            return
        self._settings_job_ids.pop(thread, None)
        role, job = self._thread_roles.pop(thread, (None, None))
        if role == "cache_usage" and self._cache_usage_thread is thread:
            self._cache_usage_thread = None
            if self._cache_usage_job is job:
                self._cache_usage_job = None
            self._update_cache_action_state()
            return
        if role == "cache_clear" and self._cache_clear_thread is thread:
            self._cache_clear_thread = None
            if self._cache_clear_job is job:
                self._cache_clear_job = None
            self._update_cache_action_state()
            return
        if role in {"verify", "runtime_rescan"} and self._verify_thread is thread:
            self._verify_thread = None
            if self._verify_job is job:
                self._verify_job = None
            self._update_runtime_action_state()
            return
        if role in {"model", "model_delete"} and self._model_thread is thread:
            self._model_thread = None
            if self._model_job is job:
                self._model_job = None
            self._update_model_action_state()
            return
        if role == "model_inventory" and self._model_inventory_thread is thread:
            self._model_inventory_thread = None
            if self._model_inventory_job is job:
                self._model_inventory_job = None
            self._update_model_action_state()
            return
        if role in {"face_model", "face_model_delete", "face_cache_clear"} and self._face_model_thread is thread:
            self._face_model_thread = None
            if self._face_model_job is job:
                self._face_model_job = None
            self._update_model_action_state()
            return
        if role == "journal_restore" and self._journal_restore_thread is thread:
            self._journal_restore_thread = None
            if self._journal_restore_job is job:
                self._journal_restore_job = None
            self._update_journal_action_state()
            return
        if role == "journal_refresh" and self._journal_refresh_thread is thread:
            self._journal_refresh_thread = None
            if self._journal_refresh_job is job:
                self._journal_refresh_job = None
        if not self._settings_job_ids:
            self.operation_progress_bar.hide()

    def _on_async_thread_finished(self, thread=None) -> None:
        self._release_finished_thread(thread)
        if not self._settings_job_ids:
            self.operation_progress_bar.hide()
        self._update_settings_cancel_state()

    def _active_settings_jobs(self) -> tuple[object, ...]:
        jobs = (
            self._cache_usage_job,
            self._cache_clear_job,
            self._verify_job,
            self._model_job,
            self._model_inventory_job,
            self._face_model_job,
            self._journal_restore_job,
            self._journal_refresh_job,
        )
        return tuple(job for job in jobs if job is not None)

    def _update_settings_cancel_state(self) -> None:
        button = getattr(self, "cancel_settings_tasks_button", None)
        if button is None:
            return
        button.setEnabled(bool(self._active_settings_jobs()) or self.model_download_controller.is_running())

    def _cancel_active_settings_tasks(self) -> None:
        cancelled_any = False
        if self.model_download_controller.is_running():
            self.model_download_controller.cancel()
            cancelled_any = True
        for job in self._active_settings_jobs():
            try:
                job.cancel()
                cancelled_any = True
            except (AttributeError, RuntimeError):
                continue
        if cancelled_any:
            self.model_status_label.setText("Cancellation requested. Completed cache files remain reusable.")
            if hasattr(self, "face_model_status_label"):
                self.face_model_status_label.setText(
                    "Cancellation requested. Verified face models and resumable partial downloads remain reusable."
                )
        self._update_settings_cancel_state()

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        if self._owns_model_download_controller:
            ready_to_close = self.model_download_controller.shutdown(timeout_ms) and ready_to_close
        jobs = (
            (self._cache_usage_job, self._cache_usage_thread),
            (self._cache_clear_job, self._cache_clear_thread),
            (self._verify_job, self._verify_thread),
            (self._model_job, self._model_thread),
            (self._model_inventory_job, self._model_inventory_thread),
            (self._face_model_job, self._face_model_thread),
            (self._journal_restore_job, self._journal_restore_thread),
            (self._journal_refresh_job, self._journal_refresh_thread),
        )
        for job, thread in jobs:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            if not wait_for_thread_shutdown(thread, timeout_ms=timeout_ms):
                ready_to_close = False
        if ready_to_close:
            for job, thread in jobs:
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
            self._cache_usage_job = None
            self._cache_usage_thread = None
            self._cache_clear_job = None
            self._cache_clear_thread = None
            self._verify_job = None
            self._verify_thread = None
            self._model_job = None
            self._model_thread = None
            self._model_inventory_job = None
            self._model_inventory_thread = None
            self._face_model_job = None
            self._face_model_thread = None
            self._journal_restore_job = None
            self._journal_restore_thread = None
            self._journal_refresh_job = None
            self._journal_refresh_thread = None
            self._thread_roles = {}
            self._settings_job_ids = {}
        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        super().closeEvent(event)

    def done(self, result: int) -> None:
        if not self.shutdown_jobs():
            return
        super().done(result)
