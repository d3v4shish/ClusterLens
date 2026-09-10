from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QUrl, pyqtSlot
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary, GeneratedStorageSummary
from app.services.face_model_installer import FaceModelInstaller, face_model_runtime_root_dir
from app.services.face_search import (
    DEFAULT_FACE_MAX_DETECTIONS,
    DEFAULT_FACE_SCORE_THRESHOLD,
    default_face_detector_id,
    default_face_embedder_id,
    face_detector_choices,
    face_embedder_choices,
    inspect_animal_face_bundle,
    normalize_face_component_id,
    normalize_face_mode,
    resolve_face_detector_bundle,
    resolve_face_embedder_bundle,
)
from infra.performance import detect_system_resources, select_performance_profile
from infra.runtime import RuntimeCapabilityService, available_execution_modes
from infra.settings import get_settings
from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.error_mbox import confirmBox, errorBox, infoBox


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings_store: QSettings,
        runtime_service: RuntimeCapabilityService,
        *,
        describe_rebuildable_caches: Callable[[], CacheUsageSummary] | None = None,
        describe_generated_storage: Callable[[], GeneratedStorageSummary] | None = None,
        prepare_rebuildable_cache_clear: Callable[[], None] | None = None,
        clear_rebuildable_caches: Callable[[], CacheClearResult] | None = None,
        clear_runtime_temp_files: Callable[[], CacheClearResult] | None = None,
        clear_face_storage: Callable[[], CacheClearResult] | None = None,
        clear_model_caches: Callable[[], CacheClearResult] | None = None,
        clear_logs: Callable[[], CacheClearResult] | None = None,
        can_clear_rebuildable_caches: Callable[[], bool] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.app_settings = get_settings()
        self.settings_store = settings_store
        self.runtime_service = runtime_service
        self.describe_rebuildable_caches = describe_rebuildable_caches
        self.describe_generated_storage = describe_generated_storage
        self.prepare_rebuildable_cache_clear = prepare_rebuildable_cache_clear
        self.clear_rebuildable_caches = clear_rebuildable_caches
        self.clear_runtime_temp_files = clear_runtime_temp_files
        self.clear_face_storage = clear_face_storage
        self.clear_model_caches = clear_model_caches
        self.clear_logs = clear_logs
        self.can_clear_rebuildable_caches = can_clear_rebuildable_caches or (lambda: True)
        self.face_model_installer = FaceModelInstaller(self.app_settings)
        self.system_resources = detect_system_resources()
        self.setWindowTitle("Settings")
        self.resize(860, 600)
        self._verify_job = None
        self._verify_thread = None
        self._cache_usage_job = None
        self._cache_usage_thread = None
        self._cache_clear_job = None
        self._cache_clear_thread = None
        self._face_model_job = None
        self._face_model_thread = None
        self._thread_roles: dict[object, tuple[str, object | None]] = {}
        self._last_verify: dict[str, object] | None = None
        self._face_model_inventory_changed = False
        self._build_ui()
        self._load_values()
        self.refresh_runtime_diagnostics()
        self.refresh_cache_usage()

    def _build_ui(self) -> None:
        def _scroll_tab(widget: QWidget) -> QScrollArea:
            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll.setWidget(widget)
            return scroll

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)

        runtime_tab = QWidget(self)
        runtime_form = QFormLayout(runtime_tab)
        self.execution_mode = QComboBox()
        self.execution_mode.addItems(list(available_execution_modes()))
        self.gpu_warmup = QCheckBox("Warm selected embedding model after folder change")
        self.runtime_badge = QCheckBox("Show runtime badge in toolbar")
        runtime_form.addRow("Preferred execution mode", self.execution_mode)
        runtime_form.addRow(self.gpu_warmup)
        runtime_form.addRow(self.runtime_badge)

        runtime_actions = QHBoxLayout()
        self.refresh_button = QPushButton("Refresh Diagnostics")
        self.install_cuda_button = QPushButton("Install CUDA (NVIDIA)")
        self.verify_gpu_button = QPushButton("Verify GPU")
        if sys.platform != "win32":
            self.install_cuda_button.setVisible(False)
        runtime_actions.addWidget(self.refresh_button)
        runtime_actions.addWidget(self.install_cuda_button)
        runtime_actions.addWidget(self.verify_gpu_button)
        runtime_form.addRow(runtime_actions)

        self.runtime_text = QTextEdit()
        self.runtime_text.setReadOnly(True)
        runtime_form.addRow(QLabel("Runtime diagnostics"), self.runtime_text)
        self.tabs.addTab(_scroll_tab(runtime_tab), "Runtime")

        ui_tab = QWidget(self)
        ui_form = QFormLayout(ui_tab)
        self.default_workspace = QComboBox()
        self.default_workspace.addItems(["clustering", "faces"])
        self.default_face_mode = QComboBox()
        self.default_face_mode.addItems(["human", "dog", "cat"])
        self._face_editor_mode = "human"
        self._face_pipeline_defaults_by_mode: dict[str, dict[str, object]] = {
            "human": {},
            "dog": {},
            "cat": {},
        }
        self.performance_profile = QComboBox()
        self.performance_profile.addItems(["balanced", "max_speed"])
        self.thumbnail_size = QSpinBox()
        self.thumbnail_size.setRange(96, 512)
        self.thumbnail_workers = QSpinBox()
        self.thumbnail_workers.setRange(1, max(8, self.system_resources.logical_cpu_count))
        self.thumbnail_workers.setEnabled(False)
        self.prefetch_rows = QSpinBox()
        self.prefetch_rows.setRange(0, max(16, self.system_resources.logical_cpu_count * 2))
        self.prefetch_rows.setEnabled(False)
        self.dense_ui = QCheckBox("Use dense desktop spacing")
        self.face_model_root = QLineEdit()
        self.face_model_root.setPlaceholderText("Optional external root for additional human/dog/cat detector and embedder bundles")
        model_root_row = QWidget(self)
        model_root_layout = QHBoxLayout(model_root_row)
        model_root_layout.setContentsMargins(0, 0, 0, 0)
        self.browse_face_model_root_button = QPushButton("Browse")
        self.browse_face_model_root_button.clicked.connect(self._browse_face_model_root)
        model_root_layout.addWidget(self.face_model_root)
        model_root_layout.addWidget(self.browse_face_model_root_button)
        self.face_model_cache_label = QLabel(str(face_model_runtime_root_dir(self.app_settings)))
        self.face_model_cache_label.setWordWrap(True)
        face_model_actions_row = QWidget(self)
        face_model_actions_layout = QHBoxLayout(face_model_actions_row)
        face_model_actions_layout.setContentsMargins(0, 0, 0, 0)
        self.refresh_face_models_button = QPushButton("Refresh Face Models")
        self.install_recommended_face_models_button = QPushButton("Install Recommended")
        self.install_edge_face_models_button = QPushButton("Install Edge")
        self.install_accuracy_face_models_button = QPushButton("Install Accuracy")
        self.install_latest_gpu_face_models_button = QPushButton("Install Latest GPU")
        self.install_max_accuracy_face_models_button = QPushButton("Install Max Accuracy")
        self.install_sface_face_models_button = QPushButton("Install SFace")
        self.install_yolo_face_model_button = QPushButton("Install YOLO")
        self.install_yunet_face_model_button = QPushButton("Install YuNet")
        self.open_face_model_cache_button = QPushButton("Open Face Model Cache")
        face_model_actions_layout.addWidget(self.refresh_face_models_button)
        face_model_actions_layout.addWidget(self.install_recommended_face_models_button)
        face_model_actions_layout.addWidget(self.install_edge_face_models_button)
        face_model_actions_layout.addWidget(self.install_accuracy_face_models_button)
        face_model_actions_layout.addWidget(self.install_latest_gpu_face_models_button)
        face_model_actions_layout.addWidget(self.install_max_accuracy_face_models_button)
        face_model_actions_layout.addWidget(self.install_sface_face_models_button)
        face_model_actions_layout.addWidget(self.install_yolo_face_model_button)
        face_model_actions_layout.addWidget(self.install_yunet_face_model_button)
        face_model_actions_layout.addWidget(self.open_face_model_cache_button)
        face_model_delete_row = QWidget(self)
        face_model_delete_layout = QHBoxLayout(face_model_delete_row)
        face_model_delete_layout.setContentsMargins(0, 0, 0, 0)
        self.delete_face_model_combo = QComboBox()
        self.delete_face_model_button = QPushButton("Delete Installed Face Model")
        face_model_delete_layout.addWidget(self.delete_face_model_combo)
        face_model_delete_layout.addWidget(self.delete_face_model_button)
        self.face_model_status = QLabel()
        self.face_model_status.setWordWrap(True)
        self.default_face_detector = QComboBox()
        self.default_face_embedder = QComboBox()
        self.default_face_score_threshold = QSpinBox()
        self.default_face_score_threshold.setRange(0, 100)
        self.default_face_score_threshold.setSuffix("%")
        self.default_face_max_detections = QSpinBox()
        self.default_face_max_detections.setRange(1, 500)
        ui_form.addRow("Default feature", self.default_workspace)
        ui_form.addRow("Default face mode", self.default_face_mode)
        ui_form.addRow("Face model root", model_root_row)
        ui_form.addRow("Managed face-model cache", self.face_model_cache_label)
        ui_form.addRow("Face model actions", face_model_actions_row)
        ui_form.addRow("Installed face model", face_model_delete_row)
        ui_form.addRow("Default detector", self.default_face_detector)
        ui_form.addRow("Default embedder", self.default_face_embedder)
        ui_form.addRow("Detector score threshold", self.default_face_score_threshold)
        ui_form.addRow("Max detections per image", self.default_face_max_detections)
        ui_form.addRow("Face model status", self.face_model_status)
        ui_form.addRow("Performance profile", self.performance_profile)
        ui_form.addRow("Thumbnail size", self.thumbnail_size)
        ui_form.addRow("Thumbnail workers", self.thumbnail_workers)
        ui_form.addRow("Thumbnail prefetch rows", self.prefetch_rows)
        ui_form.addRow(self.dense_ui)
        self.animal_model_root = self.face_model_root
        self.browse_animal_model_root_button = self.browse_face_model_root_button
        self.animal_model_status = self.face_model_status
        self.tabs.addTab(_scroll_tab(ui_tab), "Workspace")

        storage_tab = QWidget(self)
        storage_layout = QVBoxLayout(storage_tab)
        storage_form = QFormLayout()
        self.runtime_root_label = QLabel(str(self.app_settings.base_dir))
        self.runtime_root_label.setWordWrap(True)
        self.config_location_label = QLabel(str(self.settings_store.fileName() or "platform defaults"))
        self.config_location_label.setWordWrap(True)
        self.cache_root_label = QLabel(str(self.app_settings.cache_dir))
        self.cache_root_label.setWordWrap(True)
        self.cache_usage_text = QTextEdit()
        self.cache_usage_text.setReadOnly(True)
        self.cache_usage_text.setPlaceholderText("Scanning rebuildable cache usage...")
        self.generated_storage_text = QTextEdit()
        self.generated_storage_text.setReadOnly(True)
        self.generated_storage_text.setPlaceholderText("Scanning generated storage usage...")
        storage_form.addRow("Runtime root", self.runtime_root_label)
        storage_form.addRow("Config location", self.config_location_label)
        storage_form.addRow("Runtime cache location", self.cache_root_label)
        storage_form.addRow("Generated storage usage", self.generated_storage_text)
        storage_form.addRow("Rebuildable cache usage", self.cache_usage_text)
        storage_layout.addLayout(storage_form)
        storage_actions = QHBoxLayout()
        self.refresh_cache_usage_button = QPushButton("Refresh Cache Usage")
        self.clear_cache_button = QPushButton("Clear Rebuildable Caches")
        self.clear_runtime_temp_button = QPushButton("Clear Temp Files")
        self.clear_face_storage_button = QPushButton("Clear Face DBs / ANN")
        self.clear_model_caches_button = QPushButton("Clear Model Caches")
        self.clear_logs_button = QPushButton("Clear Logs")
        storage_actions.addWidget(self.refresh_cache_usage_button)
        storage_actions.addWidget(self.clear_cache_button)
        storage_actions.addWidget(self.clear_runtime_temp_button)
        storage_actions.addWidget(self.clear_face_storage_button)
        storage_actions.addWidget(self.clear_model_caches_button)
        storage_actions.addWidget(self.clear_logs_button)
        storage_layout.addLayout(storage_actions)
        self.cache_status_label = QLabel(
            "Rebuildable caches include embeddings, clustering results, indexes, installed face-model bundles, thumbnails, ONNX exports, and temp files."
        )
        self.cache_status_label.setWordWrap(True)
        storage_layout.addWidget(self.cache_status_label)
        storage_layout.addStretch(1)
        self.tabs.addTab(_scroll_tab(storage_tab), "Storage")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.refresh_button.clicked.connect(self.refresh_runtime_diagnostics)
        self.install_cuda_button.clicked.connect(lambda: self._launch_installer("enable_gpu_cuda.ps1"))
        self.verify_gpu_button.clicked.connect(self._verify_gpu)
        self.install_yunet_face_model_button.clicked.connect(self._install_yunet_face_model)
        self.install_edge_face_models_button.clicked.connect(self._install_edge_face_models)
        self.install_accuracy_face_models_button.clicked.connect(self._install_accuracy_face_models)
        self.install_latest_gpu_face_models_button.clicked.connect(self._install_latest_gpu_face_models)
        self.install_max_accuracy_face_models_button.clicked.connect(self._install_max_accuracy_face_models)
        self.install_sface_face_models_button.clicked.connect(self._install_sface_face_models)
        self.performance_profile.currentIndexChanged.connect(self._apply_profile_preview)
        self.refresh_cache_usage_button.clicked.connect(self.refresh_cache_usage)
        self.clear_cache_button.clicked.connect(self._clear_rebuildable_caches)
        self.clear_runtime_temp_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Temp Files?",
                "This removes runtime temp files and partial model-download files. Source images are not changed.",
                self.clear_runtime_temp_files,
            )
        )
        self.clear_face_storage_button.clicked.connect(
            lambda: self._clear_generated_storage(
                "Clear Face DBs And ANN Files?",
                "This removes generated face databases and ANN sidecar files. Source images are not changed.",
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
                "This removes generated log files. Source images are not changed.",
                self.clear_logs,
            )
        )
        self.face_model_root.textChanged.connect(lambda _text: self._on_face_model_root_changed())
        self.default_face_mode.currentTextChanged.connect(self._on_default_face_mode_changed)
        self.default_face_detector.currentIndexChanged.connect(lambda _index: self._save_face_editor_state())
        self.default_face_embedder.currentIndexChanged.connect(lambda _index: self._save_face_editor_state())
        self.default_face_score_threshold.valueChanged.connect(lambda _value: self._save_face_editor_state())
        self.default_face_max_detections.valueChanged.connect(lambda _value: self._save_face_editor_state())
        self.refresh_face_models_button.clicked.connect(self._refresh_face_model_install_state)
        self.install_recommended_face_models_button.clicked.connect(self._install_recommended_face_models)
        self.install_yolo_face_model_button.clicked.connect(self._install_yolo_face_model)
        self.delete_face_model_button.clicked.connect(self._delete_selected_face_model)
        self.open_face_model_cache_button.clicked.connect(self._open_face_model_cache)

    def _load_values(self) -> None:
        self.execution_mode.setCurrentText(self.settings_store.value("runtime/preferred_mode", self.app_settings.preferred_execution_mode, str))
        self.gpu_warmup.setChecked(self.settings_store.value("runtime/allow_gpu_warmup", self.app_settings.allow_gpu_warmup, bool))
        self.runtime_badge.setChecked(self.settings_store.value("runtime/show_badge", self.app_settings.show_runtime_badge, bool))
        default_feature = self.settings_store.value("workspace/default_view", "clustering", str)
        default_feature = str(default_feature or "clustering").strip().lower()
        if default_feature not in {"clustering", "faces"}:
            default_feature = "clustering"
        self.default_workspace.setCurrentText(default_feature)
        default_face_mode = str(self.settings_store.value("faces/default_mode", "human", str) or "human").strip().lower()
        if default_face_mode not in {"human", "dog", "cat"}:
            default_face_mode = "human"
        self._face_editor_mode = default_face_mode
        self.default_face_mode.blockSignals(True)
        self.default_face_mode.setCurrentText(default_face_mode)
        self.default_face_mode.blockSignals(False)
        model_root = str(self.settings_store.value("faces/model_root", "", str) or "").strip()
        if not model_root:
            model_root = str(self.settings_store.value("faces/animal_model_root", "", str) or "")
        self.face_model_root.blockSignals(True)
        self.face_model_root.setText(model_root)
        self.face_model_root.blockSignals(False)
        for mode in ("human", "dog", "cat"):
            self._face_pipeline_defaults_by_mode[mode] = {
                "detector_id": str(
                    self.settings_store.value(
                        f"faces/default_detector/{mode}",
                        default_face_detector_id(model_root, mode),
                        str,
                    )
                    or default_face_detector_id(model_root, mode)
                ),
                "embedder_id": str(
                    self.settings_store.value(
                        f"faces/default_embedder/{mode}",
                        default_face_embedder_id(model_root, mode),
                        str,
                    )
                    or default_face_embedder_id(model_root, mode)
                ),
                "score_threshold": float(
                    self.settings_store.value(
                        f"faces/detector_score_threshold/{mode}",
                        DEFAULT_FACE_SCORE_THRESHOLD,
                        float,
                    )
                ),
                "max_detections": max(
                    1,
                    int(
                        self.settings_store.value(
                            f"faces/max_detections/{mode}",
                            DEFAULT_FACE_MAX_DETECTIONS,
                            int,
                        )
                    ),
                ),
            }
        self.performance_profile.setCurrentText(self.settings_store.value("performance/profile", self.app_settings.default_performance_profile, str))
        self.thumbnail_size.setValue(int(self.settings_store.value("gallery/thumbnail_size", self.app_settings.thumbnail_size, int)))
        self.dense_ui.setChecked(self.settings_store.value("workspace/dense_ui", self.app_settings.default_dense_ui, bool))
        self._apply_profile_preview()
        self._refresh_face_pipeline_controls()
        self._refresh_face_model_status()
        self._refresh_face_model_install_state()

    def values(self) -> dict[str, object]:
        self._save_face_editor_state()
        return {
            "runtime/preferred_mode": self.execution_mode.currentText(),
            "performance/profile": self.performance_profile.currentText(),
            "runtime/allow_gpu_warmup": bool(self.gpu_warmup.isChecked()),
            "runtime/show_badge": bool(self.runtime_badge.isChecked()),
            "workspace/default_view": self.default_workspace.currentText(),
            "faces/default_mode": self.default_face_mode.currentText(),
            "faces/model_root": self.face_model_root.text().strip(),
            "faces/animal_model_root": self.face_model_root.text().strip(),
            "gallery/thumbnail_size": int(self.thumbnail_size.value()),
            "gallery/thumbnail_workers": int(self.thumbnail_workers.value()),
            "gallery/prefetch_rows": int(self.prefetch_rows.value()),
            "workspace/dense_ui": bool(self.dense_ui.isChecked()),
            **{
                f"faces/default_detector/{mode}": str(prefs.get("detector_id") or "")
                for mode, prefs in self._face_pipeline_defaults_by_mode.items()
            },
            **{
                f"faces/default_embedder/{mode}": str(prefs.get("embedder_id") or "")
                for mode, prefs in self._face_pipeline_defaults_by_mode.items()
            },
            **{
                f"faces/detector_score_threshold/{mode}": float(prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0)
                for mode, prefs in self._face_pipeline_defaults_by_mode.items()
            },
            **{
                f"faces/max_detections/{mode}": int(prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS)
                for mode, prefs in self._face_pipeline_defaults_by_mode.items()
            },
        }

    def _browse_face_model_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Face Model Root")
        if directory:
            self.face_model_root.setText(str(directory))

    def _browse_animal_model_root(self) -> None:
        self._browse_face_model_root()

    def _populate_choice_combo(self, combo: QComboBox, choices: list[tuple[str, str]], target_id: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        for item_id, label in choices:
            combo.addItem(str(label), str(item_id))
        normalized_target = normalize_face_component_id(target_id, "")
        if normalized_target and not any(str(item_id) == normalized_target for item_id, _label in choices):
            combo.addItem(f"{normalized_target} (missing)", normalized_target)
        if normalized_target:
            for index in range(combo.count()):
                if str(combo.itemData(index) or "") == normalized_target:
                    combo.setCurrentIndex(index)
                    break
        combo.blockSignals(False)

    def _save_face_editor_state(self) -> None:
        mode = normalize_face_mode(self._face_editor_mode)
        prefs = self._face_pipeline_defaults_by_mode.setdefault(mode, {})
        detector_id = str(self.default_face_detector.currentData() or prefs.get("detector_id") or "")
        embedder_id = str(self.default_face_embedder.currentData() or prefs.get("embedder_id") or "")
        prefs["detector_id"] = normalize_face_component_id(detector_id, default_face_detector_id(self.face_model_root.text().strip(), mode))
        prefs["embedder_id"] = normalize_face_component_id(embedder_id, default_face_embedder_id(self.face_model_root.text().strip(), mode))
        prefs["score_threshold"] = float(self.default_face_score_threshold.value()) / 100.0
        prefs["max_detections"] = max(1, int(self.default_face_max_detections.value()))

    def _refresh_face_pipeline_controls(self) -> None:
        mode = normalize_face_mode(self.default_face_mode.currentText())
        self._face_editor_mode = mode
        root = self.face_model_root.text().strip()
        prefs = self._face_pipeline_defaults_by_mode.setdefault(mode, {})
        detector_id = str(prefs.get("detector_id") or default_face_detector_id(root, mode))
        embedder_id = str(prefs.get("embedder_id") or default_face_embedder_id(root, mode))
        self._populate_choice_combo(self.default_face_detector, face_detector_choices(root, mode), detector_id)
        self._populate_choice_combo(self.default_face_embedder, face_embedder_choices(root, mode), embedder_id)
        self.default_face_score_threshold.blockSignals(True)
        self.default_face_score_threshold.setValue(int(round(float(prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0) * 100.0)))
        self.default_face_score_threshold.blockSignals(False)
        self.default_face_max_detections.blockSignals(True)
        self.default_face_max_detections.setValue(max(1, int(prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS)))
        self.default_face_max_detections.blockSignals(False)

    def _face_model_actions_busy(self) -> bool:
        return self._face_model_job is not None

    def _populate_face_model_delete_combo(self) -> None:
        current_id = str(self.delete_face_model_combo.currentData() or "")
        items = list(self.face_model_installer.inventory())
        self.delete_face_model_combo.blockSignals(True)
        self.delete_face_model_combo.clear()
        for item in items:
            suffix = "" if item.installed else " (not installed)"
            self.delete_face_model_combo.addItem(f"{item.display_name}{suffix}", item.bundle_id)
        if current_id:
            for index in range(self.delete_face_model_combo.count()):
                if str(self.delete_face_model_combo.itemData(index) or "") == current_id:
                    self.delete_face_model_combo.setCurrentIndex(index)
                    break
        self.delete_face_model_combo.blockSignals(False)

    def _update_face_model_action_state(self) -> None:
        busy = self._face_model_actions_busy()
        self.refresh_face_models_button.setEnabled(not busy)
        self.install_recommended_face_models_button.setEnabled(not busy)
        self.install_edge_face_models_button.setEnabled(not busy)
        self.install_accuracy_face_models_button.setEnabled(not busy)
        self.install_latest_gpu_face_models_button.setEnabled(not busy)
        self.install_max_accuracy_face_models_button.setEnabled(not busy)
        self.install_sface_face_models_button.setEnabled(not busy)
        self.install_yolo_face_model_button.setEnabled(not busy)
        self.install_yunet_face_model_button.setEnabled(not busy)
        has_installed = any(item.installed for item in self.face_model_installer.inventory())
        self.delete_face_model_combo.setEnabled((not busy) and self.delete_face_model_combo.count() > 0)
        self.delete_face_model_button.setEnabled((not busy) and has_installed and self.delete_face_model_combo.count() > 0)
        self.open_face_model_cache_button.setEnabled(True)

    def _refresh_face_model_install_state(self) -> None:
        self.face_model_cache_label.setText(str(self.face_model_installer.runtime_root()))
        self._populate_face_model_delete_combo()
        self._refresh_face_model_status()
        self._update_face_model_action_state()

    def _refresh_face_model_status(self) -> None:
        root = self.face_model_root.text().strip()
        managed_items = self.face_model_installer.inventory()
        def _bundle_label(bundle, fallback_id: str) -> str:
            if bundle is None:
                return fallback_id
            label = str(getattr(bundle, "display_name", "") or fallback_id)
            hardware = str(getattr(bundle, "hardware_class", "") or "").strip()
            if hardware:
                label = f"{label} [{hardware}]"
            return label
        lines = [
            "Human catalog: built-in Torch fallback plus bundled curated detector/embedder entries.",
            f"Managed cache: {self.face_model_installer.runtime_root()}",
        ]
        installed_items = [item for item in managed_items if item.installed]
        if installed_items:
            lines.append(
                "Installed bundles: "
                + ", ".join(f"{item.display_name} ({self._format_bytes(item.size_bytes)})" for item in installed_items)
                + "."
            )
        else:
            lines.append("Installed bundles: none.")
        for mode in ("human", "dog", "cat"):
            detector_count = len(face_detector_choices(root, mode))
            embedder_count = len(face_embedder_choices(root, mode))
            prefs = self._face_pipeline_defaults_by_mode.get(mode, {})
            detector_id = str(prefs.get("detector_id") or default_face_detector_id(root, mode))
            embedder_id = str(prefs.get("embedder_id") or default_face_embedder_id(root, mode))
            try:
                detector_bundle = resolve_face_detector_bundle(root, mode, detector_id)
                detector_state = "ready" if detector_bundle.available else "install required"
            except Exception as exc:
                detector_bundle = None
                detector_state = str(exc)
            try:
                embedder_bundle = resolve_face_embedder_bundle(root, mode, embedder_id)
                embedder_state = "ready" if embedder_bundle.available else "install required"
            except Exception as exc:
                embedder_bundle = None
                embedder_state = str(exc)
            if mode == "human":
                lines.append(
                    f"Human choices: detectors={detector_count}, embedders={embedder_count}. "
                    f"Defaults: {_bundle_label(detector_bundle, detector_id)} ({detector_state}) + "
                    f"{_bundle_label(embedder_bundle, embedder_id)} ({embedder_state})."
                )
                continue
            ready, message = inspect_animal_face_bundle(root, mode)
            prefix = mode.capitalize()
            status = "ready" if ready else "unavailable"
            lines.append(
                f"{prefix}: {status}. Defaults: {_bundle_label(detector_bundle, detector_id)} ({detector_state}) + "
                f"{_bundle_label(embedder_bundle, embedder_id)} ({embedder_state}). "
                f"Choices: detectors={detector_count}, embedders={embedder_count}. {message}"
            )
        self.face_model_status.setText("\n".join(lines))

    def _run_face_model_job(self, label: str, fn) -> None:
        if self._face_model_thread is not None:
            try:
                if self._face_model_thread.isRunning():
                    return
            except RuntimeError:
                self._face_model_thread = None
                self._face_model_job = None
        self._update_face_model_action_state()
        self.face_model_status.setText(f"{label}...")

        def _run(progress, cancel_check):
            return fn(progress, cancel_check)

        job = AsyncJob(_run)

        def _done(result: object) -> None:
            changed = False
            bundle_ids = tuple(str(item) for item in result) if isinstance(result, (list, tuple)) else ()
            if bundle_ids:
                changed = True
                self._face_model_inventory_changed = True
            self._face_model_job = None
            self._face_model_thread = None
            self._refresh_face_model_install_state()
            if changed:
                infoBox("Face models updated", f"{label} completed for: {', '.join(bundle_ids)}.")
            self._update_face_model_action_state()

        def _failed(message: str) -> None:
            self._face_model_job = None
            self._face_model_thread = None
            self._refresh_face_model_install_state()
            self.face_model_status.setText(f"{label} failed: {message}")
            self._update_face_model_action_state()
            errorBox(f"{label} failed", message)

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        self._face_model_job = job
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="face_model")
        self._face_model_thread = thread

    def _install_recommended_face_models(self) -> None:
        self._run_face_model_job(
            "Install Recommended",
            lambda progress, cancel_check: self.face_model_installer.install_recommended(progress, cancel_check),
        )

    def _install_yolo_face_model(self) -> None:
        self._run_face_model_job(
            "Install YOLO",
            lambda progress, cancel_check: self.face_model_installer.install_yolo(progress, cancel_check),
        )

    def _install_accuracy_face_models(self) -> None:
        self._run_face_model_job(
            "Install Accuracy",
            lambda progress, cancel_check: self.face_model_installer.install_accuracy(progress, cancel_check),
        )

    def _install_edge_face_models(self) -> None:
        self._run_face_model_job(
            "Install Edge",
            lambda progress, cancel_check: self.face_model_installer.install_edge(progress, cancel_check),
        )

    def _install_latest_gpu_face_models(self) -> None:
        self._run_face_model_job(
            "Install Latest GPU",
            lambda progress, cancel_check: self.face_model_installer.install_latest_gpu(progress, cancel_check),
        )

    def _install_max_accuracy_face_models(self) -> None:
        self._run_face_model_job(
            "Install Max Accuracy",
            lambda progress, cancel_check: self.face_model_installer.install_max_accuracy(progress, cancel_check),
        )

    def _install_sface_face_models(self) -> None:
        self._run_face_model_job(
            "Install SFace",
            lambda progress, cancel_check: self.face_model_installer.install_sface(progress, cancel_check),
        )

    def _install_yunet_face_model(self) -> None:
        self._run_face_model_job(
            "Install YuNet",
            lambda progress, cancel_check: self.face_model_installer.install_yunet(progress, cancel_check),
        )

    def _delete_selected_face_model(self) -> None:
        bundle_id = str(self.delete_face_model_combo.currentData() or "").strip()
        if not bundle_id:
            return
        item_text = self.delete_face_model_combo.currentText()
        if not confirmBox("Delete installed face model", f"Delete {item_text} from the managed face-model cache?"):
            return
        def _run(_progress, _cancel_check):
            removed, failures = self.face_model_installer.delete_installed_model(bundle_id)
            if failures:
                raise RuntimeError("\n".join(failures))
            return (bundle_id,) if removed else ()
        self._run_face_model_job(
            "Delete Installed Face Model",
            _run,
        )

    def _open_face_model_cache(self) -> None:
        cache_dir = self.face_model_installer.runtime_root()
        cache_dir.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(cache_dir)))

    def _on_default_face_mode_changed(self, mode: str) -> None:
        self._save_face_editor_state()
        self._face_editor_mode = normalize_face_mode(mode)
        self._refresh_face_pipeline_controls()
        self._refresh_face_model_status()

    def _on_face_model_root_changed(self) -> None:
        self._save_face_editor_state()
        root = self.face_model_root.text().strip()
        for mode in ("human", "dog", "cat"):
            prefs = self._face_pipeline_defaults_by_mode.setdefault(mode, {})
            detector_id = str(prefs.get("detector_id") or "")
            embedder_id = str(prefs.get("embedder_id") or "")
            if not detector_id:
                prefs["detector_id"] = default_face_detector_id(root, mode)
            if not embedder_id:
                prefs["embedder_id"] = default_face_embedder_id(root, mode)
        self._refresh_face_pipeline_controls()
        self._refresh_face_model_install_state()

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
            "logs": "Logs",
            "thumbnails": "Thumbnails",
            "rebuildable_caches": "Rebuildable caches",
            "face_databases": "Face DBs",
            "ann_files": "ANN files",
            "model_caches": "Model caches",
            "temp_files": "Temp files",
        }
        for key, label in labels.items():
            paths = tuple(summary.target_paths.get(key, ()))
            path_text = ", ".join(paths) if paths else "(none found)"
            lines.append(f"{label}: {self._format_bytes(summary.target_bytes.get(key, 0))}")
            lines.append(f"  {path_text}")
        lines.extend(["", f"Total generated storage size: {self._format_bytes(summary.total_bytes)}"])
        return "\n".join(lines)

    def _cache_actions_busy(self) -> bool:
        return any(job is not None for job in (self._cache_usage_job, self._cache_clear_job))

    def _update_cache_action_state(self) -> None:
        busy = self._cache_actions_busy()
        self.refresh_cache_usage_button.setEnabled(not busy)
        allow_clear = bool(self.clear_rebuildable_caches) and bool(self.can_clear_rebuildable_caches())
        self.clear_cache_button.setEnabled((not busy) and allow_clear)
        for button, callback in (
            (self.clear_runtime_temp_button, self.clear_runtime_temp_files),
            (self.clear_face_storage_button, self.clear_face_storage),
            (self.clear_model_caches_button, self.clear_model_caches),
            (self.clear_logs_button, self.clear_logs),
        ):
            button.setEnabled((not busy) and bool(callback) and bool(self.can_clear_rebuildable_caches()))
        if not allow_clear and not busy:
            self.cache_status_label.setText("Cache clearing is unavailable while clustering is running.")

    def refresh_cache_usage(self) -> None:
        if self.describe_rebuildable_caches is None:
            self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            self._update_cache_action_state()
            return
        if self._cache_usage_thread is not None:
            try:
                if self._cache_usage_thread.isRunning():
                    return
            except RuntimeError:
                self._cache_usage_thread = None
                self._cache_usage_job = None
        self.cache_status_label.setText("Scanning rebuildable cache usage...")
        self._update_cache_action_state()

        def _run(progress, cancel_check):
            _ = progress
            _ = cancel_check
            rebuildable = self.describe_rebuildable_caches()
            generated = self.describe_generated_storage() if callable(self.describe_generated_storage) else None
            return rebuildable, generated

        job = AsyncJob(_run)

        def _done(result: object) -> None:
            generated = None
            if isinstance(result, tuple):
                cache_result = result[0] if result else None
                generated = result[1] if len(result) > 1 else None
            else:
                cache_result = result
            if isinstance(cache_result, CacheUsageSummary):
                lines = [f"Runtime cache root: {cache_result.cache_root}", ""]
                for name, size in cache_result.target_bytes.items():
                    lines.append(f"{name} {self._format_bytes(size)}")
                lines.extend(
                    [
                        "",
                        f"Total rebuildable cache size: {self._format_bytes(cache_result.total_bytes)}",
                        "",
                        "Preserved: image tags, face/search indexes, perceptual hashes, downloaded face models, Hugging Face downloads, and Torch downloads.",
                    ]
                )
                self.cache_usage_text.setPlainText("\n".join(lines))
                self.cache_status_label.setText("Rebuildable cache usage refreshed.")
            else:
                self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            if isinstance(generated, GeneratedStorageSummary):
                self.generated_storage_text.setPlainText(self._format_generated_storage(generated))
            elif callable(self.describe_generated_storage):
                self.generated_storage_text.setPlainText("Generated storage usage is unavailable.")
            self._update_cache_action_state()

        def _failed(message: str) -> None:
            self.cache_usage_text.setPlainText("Failed to scan cache usage.")
            self.cache_status_label.setText(f"Cache usage refresh failed: {message}")
            self._update_cache_action_state()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        self._cache_usage_job = job
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_usage")
        self._cache_usage_thread = thread

    def _clear_rebuildable_caches(self) -> None:
        if self.clear_rebuildable_caches is None:
            return
        if not self.can_clear_rebuildable_caches():
            self.cache_status_label.setText("Wait for the current clustering run to finish before clearing caches.")
            self._update_cache_action_state()
            return
        if not confirmBox(
            "Clear Rebuildable Caches?",
            "This removes embeddings, clustering results, indexes, thumbnails, ONNX exports, and temp files.\n\n"
            "Tags, face/search indexes, perceptual hashes, downloaded face models, Hugging Face downloads, and Torch downloads are preserved.",
            parent=self,
        ):
            return
        if callable(self.prepare_rebuildable_cache_clear):
            self.prepare_rebuildable_cache_clear()
        self.cache_status_label.setText("Clearing rebuildable caches...")
        self._update_cache_action_state()

        def _run(progress, cancel_check):
            _ = progress
            _ = cancel_check
            return self.clear_rebuildable_caches()

        job = AsyncJob(_run)

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

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        self._cache_clear_job = job
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_clear")
        self._cache_clear_thread = thread

    def _clear_generated_storage(self, title: str, message: str, callback: Callable[[], CacheClearResult] | None) -> None:
        if callback is None:
            return
        if not self.can_clear_rebuildable_caches():
            self.cache_status_label.setText("Wait for the current clustering run to finish before clearing generated data.")
            self._update_cache_action_state()
            return
        if not confirmBox(title, message, parent=self):
            return
        self.cache_status_label.setText(title.rstrip("?") + "...")
        self._update_cache_action_state()

        def _run(progress, cancel_check):
            _ = progress
            _ = cancel_check
            return callback()

        job = AsyncJob(_run)

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

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        self._cache_clear_job = job
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_clear")
        self._cache_clear_thread = thread

    def refresh_runtime_diagnostics(self) -> None:
        details = self.runtime_service.diagnostics(self.execution_mode.currentText())
        capabilities = details["capabilities"]
        policy = details["policy"]
        packages = details.get("packages") or {}
        optional_details = details.get("optional_details") or {}
        remediation = details["remediation"]
        profile = select_performance_profile(self.performance_profile.currentText(), self.system_resources)

        text = [
            f"Preferred mode: {policy.preferred_mode}",
            f"Effective mode: {policy.effective_mode}",
            f"Performance profile: {profile.name}",
            "",
            "Packages:",
            f"- torch: {packages.get('torch') or '-'}",
            f"- onnx: {packages.get('onnx') or '-'}",
            f"- onnxruntime: {packages.get('onnxruntime') or '-'}",
            f"- onnxruntime-gpu: {packages.get('onnxruntime-gpu') or '-'}",
            f"- hf_xet: {packages.get('hf_xet') or '-'}",
            f"- flash-attn: {packages.get('flash-attn') or '-'}",
            "",
            f"Torch device: {policy.torch_device}",
            f"ONNX provider: {policy.onnx_provider}",
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
            f"Thumbnail cache entries: {profile.pixmap_cache_size}",
            "",
            "Optional accelerators:",
            f"- hf_xet download accelerator: {'installed' if optional_details.get('hf_xet_installed') else 'missing'}",
            f"- Flash SDP enabled: {bool(optional_details.get('flash_sdp_enabled'))}",
            "",
            f"Reason: {policy.reason}",
        ]

        if self._last_verify:
            verify = self._last_verify
            torch_smoke = verify.get("torch_smoke")
            onnx_smoke = verify.get("onnx_smoke")
            flash_attention_smoke = verify.get("flash_attention_smoke")
            text.extend(["", "Verify GPU:"])
            if isinstance(torch_smoke, dict):
                if torch_smoke.get("ok"):
                    text.append(f"- Torch smoke: OK ({torch_smoke.get('ms')} ms)")
                else:
                    text.append(f"- Torch smoke: FAIL ({torch_smoke.get('error')})")
            else:
                text.append("- Torch smoke: (skipped)")
            if isinstance(onnx_smoke, dict):
                if onnx_smoke.get("ok"):
                    text.append(f"- ONNX smoke: OK ({onnx_smoke.get('provider')}, {onnx_smoke.get('ms')} ms)")
                else:
                    text.append(f"- ONNX smoke: FAIL ({onnx_smoke.get('error')})")
            else:
                text.append("- ONNX smoke: (skipped)")
            if isinstance(flash_attention_smoke, dict):
                if flash_attention_smoke.get("ok"):
                    text.append("- Flash attention verify: OK")
                else:
                    text.append(f"- Flash attention verify: FAIL ({flash_attention_smoke.get('error')})")
            else:
                text.append("- Flash attention verify: (skipped)")
            remediation = list(verify.get("remediation") or remediation)

        if remediation:
            text.extend(["", "Suggested remediation:"])
            text.extend(f"- {item}" for item in remediation)

        text.extend(
            [
                "",
                "Notes:",
                "- CUDA acceleration requires a CUDA-enabled Torch build plus NVIDIA drivers.",
                "- hf_xet is optional but improves Hugging Face download speed; include it in the packaged Python runtime if you ship a one-file exe.",
                "- Flash attention is not bundled automatically. If you want that optimization in the final exe, ship a Torch/CUDA stack that already provides it and verify it on the target machine.",
                "- After installing runtimes, restart the app.",
            ]
        )

        self.runtime_text.setPlainText("\n".join(text))

    def _apply_profile_preview(self) -> None:
        profile = select_performance_profile(self.performance_profile.currentText(), self.system_resources)
        self.thumbnail_workers.setValue(profile.thumbnail_workers)
        self.prefetch_rows.setValue(profile.thumbnail_prefetch_rows)
        if hasattr(self, "runtime_text"):
            self.refresh_runtime_diagnostics()

    def _launch_installer(self, script_name: str) -> None:
        if sys.platform != "win32":
            errorBox("Unsupported installer", "Runtime installer shortcuts are currently Windows-only. Install platform-specific Torch/ONNX packages with your package manager.")
            return
        script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
        if not script_path.exists():
            errorBox("Missing script", f"Installer script not found: {script_path}")
            return
        try:
            subprocess.Popen(
                [
                    "powershell.exe",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script_path),
                    str(Path(sys.executable)),
                ]
            )
        except Exception as exc:
            errorBox("Launch failed", str(exc))
            return
        infoBox("Installer started", "The installer has been launched. Restart the app after it completes.")

    def _verify_gpu(self) -> None:
        if self._verify_thread is not None:
            try:
                if self._verify_thread.isRunning():
                    return
            except RuntimeError:
                self._verify_thread = None
                self._verify_job = None

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Probing runtimes...")
            return self.runtime_service.verify(self.execution_mode.currentText())

        job = AsyncJob(_run)

        def _done(result: object) -> None:
            if isinstance(result, dict):
                self._last_verify = result
            else:
                self._last_verify = None
            self.refresh_runtime_diagnostics()

        job.completed.connect(_done)
        job.failed.connect(lambda msg: errorBox("Verify failed", str(msg)))
        self._verify_job = job
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="verify")
        self._verify_thread = thread

    def _track_thread(self, thread, job: object | None, *, role: str) -> None:
        if thread is None:
            return
        self._thread_roles[thread] = (str(role), job)
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _release_finished_thread(self, thread) -> None:
        if thread is None:
            return
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
        if role == "verify" and self._verify_thread is thread:
            self._verify_thread = None
            if self._verify_job is job:
                self._verify_job = None
            return
        if role == "face_model" and self._face_model_thread is thread:
            self._face_model_thread = None
            if self._face_model_job is job:
                self._face_model_job = None
            self._update_face_model_action_state()

    def _on_async_thread_finished(self, thread=None) -> None:
        self._release_finished_thread(thread)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        for job, thread in [
            (self._cache_usage_job, self._cache_usage_thread),
            (self._cache_clear_job, self._cache_clear_thread),
            (self._verify_job, self._verify_thread),
            (self._face_model_job, self._face_model_thread),
        ]:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    ready_to_close = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        if ready_to_close:
            self._cache_usage_job = None
            self._cache_usage_thread = None
            self._cache_clear_job = None
            self._cache_clear_thread = None
            self._verify_job = None
            self._verify_thread = None
            self._face_model_job = None
            self._face_model_thread = None
            self._thread_roles = {}
        return ready_to_close

    def face_model_inventory_changed(self) -> bool:
        return bool(self._face_model_inventory_changed)

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)


