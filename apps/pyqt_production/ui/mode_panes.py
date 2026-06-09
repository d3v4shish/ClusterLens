from __future__ import annotations

import os

from PyQt6.QtCore import QDir, pyqtSignal
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.services.clustering_options import (
    backend_tooltip,
    clustering_backend_names,
    clustering_model_names,
    default_backend_names,
    model_label,
    model_tooltip,
    normalize_clustering_backends,
    normalize_embedding_models,
)
from app.services.similarity_modes import SUPPORTED_SIMILARITY_MODES, normalize_similarity_modes
from infra.settings import get_settings
from .common import build_help_inline, build_help_label


CLUSTERING_HELP = {
    "embedding_models": (
        "Choose which embedding models turn images into feature vectors.\n"
        "The selected folder is clustered once per enabled model.\n"
        "Different models can separate the same images in different ways.\n"
        "More enabled models add more comparison columns and more runtime."
    ),
    "cluster_backends": (
        "Choose which clustering backends build groups from the embeddings.\n"
        "Each enabled backend produces its own comparison column.\n"
        "Backends trade off speed, shape sensitivity, and outlier handling.\n"
        "Running more backends gives more comparisons but costs more time."
    ),
    "similarity": (
        "Choose which similarity modes should become comparison columns.\n"
        "Semantic uses a PCA-compressed normalized projection of the embeddings.\n"
        "Cosine uses the full normalized embedding vectors.\n"
        "Each selected mode runs across every enabled model and backend."
    ),
    "outlier": (
        "Choose how uncertain images are handled after clustering.\n"
        "Assign keeps every image inside a cluster.\n"
        "Isolate pulls weak matches away, while Keep preserves backend output.\n"
        "Different policies can change final cluster sizes and previews."
    ),
    "clusters": (
        "Set the requested number of clusters for fixed-count backends.\n"
        "Use this as the target grouping granularity for the current run.\n"
        "Small subsets from Tag Filter or Recluster can fail if this number is too high.\n"
        "If matched images drop below this count, the run is rejected."
    ),
    "recursive_scan": (
        "Include images from subfolders under the selected folder.\n"
        "This expands the source scope before Tag Filter is applied.\n"
        "Turn it off to cluster only the files directly inside the chosen folder."
    ),
    "use_onnx_runtime": (
        "Prefer ONNX execution for models that support it.\n"
        "This can improve startup or hardware compatibility on some systems.\n"
        "Models without ONNX support still use their normal runtime."
    ),
    "reuse_cluster_result_cache": (
        "Reuse a saved clustering result when the folder snapshot and options still match.\n"
        "The cache is keyed by discovered files and clustering parameters, not just folder path.\n"
        "Changing tags does not invalidate the cluster result, but summaries are recomputed."
    ),
    "use_embedding_cache_lookup": (
        "Reuse saved embedding vectors for matching images before running the model again.\n"
        "This only controls embedding-cache reads for clustering runs.\n"
        "Cluster Result Cache is separate and can still skip whole clustering results.\n"
        "Turn this off to force fresh embeddings while still writing new vectors for later runs."
    ),
    "tag_filter": (
        "Filter images by tags inside the selected folder scope.\n"
        "The current folder and Recursive Scan decide which files are considered first.\n"
        "Enter comma-separated image tags from the local tag database.\n"
        "Tag Match = Any means at least one listed tag; All means every listed tag.\n"
        "Small matches can still fail later if Clusters is higher than the matched image count."
    ),
    "tag_match": (
        "Choose how multiple tag names are matched for Tag Filter.\n"
        "Any keeps images with at least one listed tag.\n"
        "All keeps only images that contain every listed tag.\n"
        "This match happens after folder discovery and before clustering starts."
    ),
    "run_clustering": (
        "Start clustering for the selected folder and current options.\n"
        "If Tag Filter is filled, the folder scope is narrowed before the run begins.\n"
        "Each enabled model and backend contributes to the comparison view."
    ),
    "cancel": (
        "Stop the active clustering request.\n"
        "This cancels the in-flight run and leaves the last completed result visible.\n"
        "Use it when the current folder, model mix, or subset is too expensive."
    ),
}


class SourcePane(QWidget):
    directory_changed = pyqtSignal(str)
    state_changed = pyqtSignal()
    hide_requested = pyqtSignal()
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.selected_directory = "D:/" if os.name == "nt" else "/home"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        title = QLabel("Folders")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.hide_button = QPushButton("Hide")
        self.hide_button.setProperty("paneToggle", True)
        self.hide_button.setFixedHeight(24)
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.hide_button)
        layout.addLayout(header)

        self.directory_combobox = QComboBox()
        self.populate_drives()
        self.directory_combobox.currentIndexChanged.connect(self.on_directory_changed)
        layout.addWidget(self.directory_combobox)

        self.file_model = QFileSystemModel()
        self.file_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.file_model.setRootPath(self.selected_directory)

        self.file_tree = QTreeView()
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(self.selected_directory))
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setColumnHidden(1, True)
        self.file_tree.setColumnHidden(2, True)
        self.file_tree.setColumnHidden(3, True)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setStyleSheet(
            "QTreeView { font-size: 12px; }"
            "QTreeView::item { height: 24px; padding: 3px; }"
        )
        self.file_tree.clicked.connect(self.on_directory_selected)
        layout.addWidget(self.file_tree, stretch=1)

        self.basic_action_section = QWidget(self)
        basic_action_layout = QVBoxLayout(self.basic_action_section)
        basic_action_layout.setContentsMargins(0, 0, 0, 0)
        basic_action_layout.setSpacing(6)

        self.basic_run_button = QPushButton("Run Clustering")
        self.basic_cancel_button = QPushButton("Cancel")
        self.basic_run_button.setToolTip(CLUSTERING_HELP["run_clustering"])
        self.basic_cancel_button.setToolTip(CLUSTERING_HELP["cancel"])
        self.basic_run_button.clicked.connect(self.run_requested.emit)
        self.basic_cancel_button.clicked.connect(self.cancel_requested.emit)
        basic_action_layout.addWidget(self.basic_run_button)
        basic_action_layout.addWidget(self.basic_cancel_button)
        layout.addWidget(self.basic_action_section)
        self.set_running(False)
        self.basic_action_section.hide()

    def populate_drives(self) -> None:
        self.directory_combobox.clear()
        if os.name == "nt":
            from string import ascii_uppercase

            drives = [f"{letter}:/" for letter in ascii_uppercase if os.path.exists(f"{letter}:/")]
            self.selected_directory = drives[0] if drives else "C:/"
        else:
            drives = ["/", "/home"]
            self.selected_directory = "/home"
        self.directory_combobox.addItems(drives)

    def on_directory_selected(self, index) -> None:
        self.selected_directory = self.file_model.filePath(index)
        self.directory_changed.emit(self.selected_directory)
        self.state_changed.emit()

    def on_directory_changed(self, _index) -> None:
        root_directory = self.directory_combobox.currentText()
        self.selected_directory = root_directory
        self.file_model.setRootPath(root_directory)
        self.file_tree.setRootIndex(self.file_model.index(root_directory))
        self.directory_changed.emit(self.selected_directory)
        self.state_changed.emit()

    def set_selected_directory(self, directory: str) -> None:
        directory = str(directory or "").strip()
        if not directory or not os.path.exists(directory):
            return
        drive = os.path.splitdrive(directory)[0]
        if drive:
            combo_value = f"{drive}/".replace("\\", "/")
            idx = self.directory_combobox.findText(combo_value)
            if idx >= 0:
                self.directory_combobox.setCurrentIndex(idx)
        self.selected_directory = directory
        self.file_model.setRootPath(directory)
        self.file_tree.setRootIndex(self.file_model.index(directory))
        self.directory_changed.emit(self.selected_directory)

    def set_running(self, running: bool) -> None:
        self.basic_run_button.setVisible(not running)
        self.basic_run_button.setEnabled(not running)
        self.basic_cancel_button.setVisible(running)
        self.basic_cancel_button.setEnabled(running)

    def update_progress(self, value: int, status: str) -> None:
        _ = value
        _ = status

    def set_basic_mode(self, enabled: bool, *, running: bool = False) -> None:
        enabled = bool(enabled)
        self.hide_button.setVisible(not enabled)
        self.basic_action_section.setVisible(enabled)
        self.set_running(running)


class ClusteringOptionsPane(QWidget):
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    state_changed = pyqtSignal()
    hide_requested = pyqtSignal()

    def __init__(self, parent=None, *, option_scope: str = "legacy"):
        super().__init__(parent)
        self.settings = get_settings()
        self.option_scope = str(option_scope or "legacy").strip().lower()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        title = QLabel("Clustering")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.hide_button = QPushButton("Hide")
        self.hide_button.setProperty("paneToggle", True)
        self.hide_button.setFixedHeight(24)
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.hide_button)
        layout.addLayout(header)

        embedding_group = QGroupBox("Embedding Models")
        embedding_group.setToolTip(CLUSTERING_HELP["embedding_models"])
        embedding_layout = QGridLayout(embedding_group)
        embedding_layout.setHorizontalSpacing(10)
        embedding_layout.setVerticalSpacing(4)
        self.embedding_checkboxes = {}
        model_names = clustering_model_names(self.option_scope)
        for index, model_name in enumerate(model_names):
            label = model_label(model_name)
            checkbox = QCheckBox(label)
            checkbox.setToolTip(model_tooltip(model_name))
            checkbox.setChecked(model_name == self.settings.default_model)
            checkbox.toggled.connect(self.state_changed.emit)
            self.embedding_checkboxes[model_name] = checkbox
            embedding_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(embedding_group)

        backend_group = QGroupBox("Cluster Backends")
        backend_group.setToolTip(CLUSTERING_HELP["cluster_backends"])
        backend_layout = QGridLayout(backend_group)
        backend_layout.setHorizontalSpacing(10)
        backend_layout.setVerticalSpacing(4)
        self.backend_checkboxes = {}
        backend_names = clustering_backend_names(self.option_scope)
        default_backends = set(default_backend_names(self.option_scope))
        for index, backend in enumerate(backend_names):
            checkbox = QCheckBox(backend)
            checkbox.setToolTip(backend_tooltip(backend))
            checkbox.setChecked(backend in default_backends)
            checkbox.toggled.connect(self.state_changed.emit)
            self.backend_checkboxes[backend] = checkbox
            backend_layout.addWidget(checkbox, index // 2, index % 2)
        layout.addWidget(backend_group)

        options_group = QGroupBox("Options")
        options_layout = QFormLayout(options_group)
        options_layout.setHorizontalSpacing(12)
        options_layout.setVerticalSpacing(8)

        similarity_widget = QWidget(options_group)
        similarity_layout = QHBoxLayout(similarity_widget)
        similarity_layout.setContentsMargins(0, 0, 0, 0)
        similarity_layout.setSpacing(10)
        self.similarity_checkboxes = {}
        default_modes = normalize_similarity_modes(None, self.settings.default_similarity_mode)
        for mode in SUPPORTED_SIMILARITY_MODES:
            checkbox = QCheckBox(mode)
            checkbox.setToolTip(CLUSTERING_HELP["similarity"])
            checkbox.setChecked(mode in default_modes)
            checkbox.toggled.connect(self._on_similarity_toggled)
            self.similarity_checkboxes[mode] = checkbox
            similarity_layout.addWidget(checkbox)
        similarity_layout.addStretch(1)
        options_layout.addRow(build_help_label("Similarity", CLUSTERING_HELP["similarity"], help_key="similarity"), similarity_widget)

        self.outlier_combobox = QComboBox()
        self.outlier_combobox.addItems(["assign", "isolate", "keep"])
        self.outlier_combobox.setCurrentText(self.settings.default_outlier_policy)
        self.outlier_combobox.setToolTip(CLUSTERING_HELP["outlier"])
        self.outlier_combobox.currentIndexChanged.connect(self.state_changed.emit)
        options_layout.addRow(build_help_label("Outlier", CLUSTERING_HELP["outlier"], help_key="outlier"), self.outlier_combobox)

        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setMinimum(self.settings.min_cluster_count)
        self.cluster_spinbox.setMaximum(500)
        self.cluster_spinbox.setValue(self.settings.default_cluster_count)
        self.cluster_spinbox.setToolTip(CLUSTERING_HELP["clusters"])
        self.cluster_spinbox.valueChanged.connect(self.state_changed.emit)
        options_layout.addRow(build_help_label("Clusters", CLUSTERING_HELP["clusters"], help_key="clusters"), self.cluster_spinbox)

        self.recursive_checkbox = QCheckBox("Recursive Scan")
        self.recursive_checkbox.setChecked(self.settings.recursive_scan)
        self.recursive_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addRow(build_help_inline(self.recursive_checkbox, CLUSTERING_HELP["recursive_scan"], help_key="recursive_scan"))

        self.onnx_checkbox = QCheckBox("Use ONNX Runtime")
        self.onnx_checkbox.setChecked(self.settings.default_use_onnx)
        self.onnx_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addRow(build_help_inline(self.onnx_checkbox, CLUSTERING_HELP["use_onnx_runtime"], help_key="use_onnx_runtime"))

        self.result_cache_checkbox = QCheckBox("Reuse Cluster Result Cache")
        self.result_cache_checkbox.setChecked(self.settings.default_reuse_result_cache)
        self.result_cache_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addRow(
            build_help_inline(
                self.result_cache_checkbox,
                CLUSTERING_HELP["reuse_cluster_result_cache"],
                help_key="reuse_cluster_result_cache",
            )
        )

        self.embedding_cache_lookup_checkbox = QCheckBox("Use Embedding Cache Lookup")
        self.embedding_cache_lookup_checkbox.setChecked(True)
        self.embedding_cache_lookup_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addRow(
            build_help_inline(
                self.embedding_cache_lookup_checkbox,
                CLUSTERING_HELP["use_embedding_cache_lookup"],
                help_key="use_embedding_cache_lookup",
            )
        )

        self.tag_filter_field = QLineEdit()
        self.tag_filter_field.setPlaceholderText("Comma-separated tags")
        self.tag_filter_field.setToolTip(CLUSTERING_HELP["tag_filter"])
        self.tag_filter_field.textChanged.connect(self.state_changed.emit)
        options_layout.addRow(build_help_label("Tag Filter", CLUSTERING_HELP["tag_filter"], help_key="tag_filter"), self.tag_filter_field)

        self.tag_match_combobox = QComboBox()
        self.tag_match_combobox.addItems(["Any", "All"])
        self.tag_match_combobox.setToolTip(CLUSTERING_HELP["tag_match"])
        self.tag_match_combobox.currentIndexChanged.connect(self.state_changed.emit)
        options_layout.addRow(build_help_label("Tag Match", CLUSTERING_HELP["tag_match"], help_key="tag_match"), self.tag_match_combobox)
        layout.addWidget(options_group)

        button_row = QHBoxLayout()
        self.cluster_button = QPushButton("Run Clustering")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        button_row.addWidget(build_help_inline(self.cluster_button, CLUSTERING_HELP["run_clustering"], help_key="run_clustering"), 1)
        button_row.addWidget(build_help_inline(self.cancel_button, CLUSTERING_HELP["cancel"], help_key="cancel"), 1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.cluster_button.clicked.connect(self.run_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._last_metrics_text = ""

    def set_running(self, running: bool) -> None:
        self.cluster_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def update_progress(self, value: int, status: str) -> None:
        _ = value
        _ = status

    def selected_clustering_backends(self) -> list[str]:
        selected = [backend for backend, checkbox in self.backend_checkboxes.items() if checkbox.isChecked()]
        return normalize_clustering_backends(selected, scope=self.option_scope)

    def selected_embedding_models(self) -> list[str]:
        selected = [name for name, checkbox in self.embedding_checkboxes.items() if checkbox.isChecked()]
        return normalize_embedding_models(
            selected,
            default_model=self.settings.default_model,
            scope=self.option_scope,
        )

    def selected_tag_filters(self) -> list[str]:
        return [tag.strip() for tag in self.tag_filter_field.text().split(",") if tag.strip()]

    def selected_similarity_modes(self) -> list[str]:
        selected = [mode for mode, checkbox in self.similarity_checkboxes.items() if checkbox.isChecked()]
        if not selected:
            self.similarity_checkboxes["semantic"].setChecked(True)
            selected = ["semantic"]
        return selected

    def _on_similarity_toggled(self, _checked: bool) -> None:
        if not any(checkbox.isChecked() for checkbox in self.similarity_checkboxes.values()):
            self.similarity_checkboxes["semantic"].setChecked(True)
            return
        self.state_changed.emit()

    def selected_tag_match_mode(self) -> str:
        return self.tag_match_combobox.currentText().strip() or "Any"

    def export_state(self) -> dict[str, object]:
        similarity_modes = self.selected_similarity_modes()
        return {
            "embedding_models": self.selected_embedding_models(),
            "clustering_backends": self.selected_clustering_backends(),
            "similarity_modes": similarity_modes,
            "similarity_mode": similarity_modes[0],
            "outlier_policy": self.outlier_combobox.currentText(),
            "cluster_count": int(self.cluster_spinbox.value()),
            "recursive": bool(self.recursive_checkbox.isChecked()),
            "use_onnx": bool(self.onnx_checkbox.isChecked()),
            "reuse_result_cache": bool(self.result_cache_checkbox.isChecked()),
            "use_embedding_cache_lookup": bool(self.embedding_cache_lookup_checkbox.isChecked()),
            "tag_filter": self.tag_filter_field.text().strip(),
            "tag_match": self.selected_tag_match_mode(),
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        embedding_models = set(
            normalize_embedding_models(
                state.get("embedding_models"),
                default_model=self.settings.default_model,
                scope=self.option_scope,
            )
        )
        for name, checkbox in self.embedding_checkboxes.items():
            checkbox.setChecked(name in embedding_models)
        clustering_backends = set(
            normalize_clustering_backends(
                state.get("clustering_backends"),
                scope=self.option_scope,
            )
        )
        for name, checkbox in self.backend_checkboxes.items():
            checkbox.setChecked(name in clustering_backends)
        modes = normalize_similarity_modes(
            state.get("similarity_modes") if isinstance(state.get("similarity_modes"), list) else None,
            str(state.get("similarity_mode") or "") if state.get("similarity_mode") else None,
        )
        for name, checkbox in self.similarity_checkboxes.items():
            checkbox.blockSignals(True)
            checkbox.setChecked(name in modes)
            checkbox.blockSignals(False)
        if not any(checkbox.isChecked() for checkbox in self.similarity_checkboxes.values()):
            self.similarity_checkboxes["semantic"].setChecked(True)
        if state.get("outlier_policy"):
            self.outlier_combobox.setCurrentText(str(state.get("outlier_policy")))
        if state.get("cluster_count"):
            self.cluster_spinbox.setValue(int(state.get("cluster_count") or self.settings.default_cluster_count))
        self.recursive_checkbox.setChecked(bool(state.get("recursive", self.settings.recursive_scan)))
        self.onnx_checkbox.setChecked(bool(state.get("use_onnx", self.settings.default_use_onnx)))
        self.result_cache_checkbox.setChecked(bool(state.get("reuse_result_cache", self.settings.default_reuse_result_cache)))
        self.embedding_cache_lookup_checkbox.setChecked(bool(state.get("use_embedding_cache_lookup", True)))
        self.tag_filter_field.setText(str(state.get("tag_filter") or ""))
        self.tag_match_combobox.setCurrentText(str(state.get("tag_match") or "Any"))

    def update_metrics(self, metrics: dict) -> None:
        ordered_keys = [
            "performance_profile",
            "run_origin",
            "tag_filter",
            "tag_match",
            "source_scope",
            "image_count",
            "scan_time_s",
            "discovery_time_s",
            "embedding_models",
            "similarity_modes",
            "warmup_state",
            "embedding_cache_lookup",
            "cache_lookup_time_s",
            "preprocess_time_s",
            "inference_time_s",
            "embedding_time_s",
            "prepared_matrix_time_s",
            "clustering_time_s",
            "index_persist_time_s",
            "index_reuse",
            "model_load_time_s",
            "cache_hits",
            "cache_misses",
            "result_cache_hit",
            "batch_size",
            "oom_backoff",
            "model_device",
            "effective_mode",
            "onnx_provider",
            "runtime_reason",
            "embedding_backend",
            "amp_enabled",
            "clustering_backends",
            "backend_cache_hits",
            "cluster_quality_score",
            "outlier_count",
            "silhouette_score",
            "silhouette_sample_size",
            "gallery_first_paint_ms",
            "snapshot_key",
            "faiss_index_path",
            "status",
        ]
        parts = [f"{key}: {metrics[key]}" for key in ordered_keys if key in metrics and metrics[key] not in {"", None}]
        self._last_metrics_text = " | ".join(parts)
        return self._last_metrics_text


