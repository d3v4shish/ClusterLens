from __future__ import annotations

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QToolButton,
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
from app.path_scope import PathScope, path_is_within_scope
from infra.settings import get_settings
from .common import ResponsiveFlowLayout, build_help_inline, build_help_label
from .icons import apply_icon
from .theme import apply_field_size


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


class ActiveRootFileSystemModel(QFileSystemModel):
    """Filesystem tree model with lightweight active-root check state."""

    root_toggle_requested = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._active_scope = PathScope()

    def set_active_scope(self, scope: PathScope) -> None:
        self._active_scope = scope
        # QFileSystemModel materializes rows lazily; a layout notification is
        # cheaper and safer than recursively walking every visible descendant.
        self.layoutChanged.emit()

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):  # type: ignore[override]
        if role == Qt.ItemDataRole.CheckStateRole and index.isValid() and index.column() == 0:
            path = self.filePath(index)
            if path in self._active_scope.roots:
                return Qt.CheckState.Checked
            if any(path_is_within_scope(root, path) for root in self._active_scope.roots):
                return Qt.CheckState.PartiallyChecked
            return Qt.CheckState.Unchecked
        return super().data(index, role)

    def flags(self, index):  # type: ignore[override]
        flags = super().flags(index)
        if index.isValid() and index.column() == 0 and self.isDir(index):
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):  # type: ignore[override]
        if role == Qt.ItemDataRole.CheckStateRole and index.isValid() and self.isDir(index):
            path = self.filePath(index)
            checked = value == Qt.CheckState.Checked or int(value) == int(Qt.CheckState.Checked)
            self.root_toggle_requested.emit(path, bool(checked))
            return True
        return super().setData(index, value, role)


# Kept in its own module so the global Roots editor has a small, focused
# state machine instead of sharing the large organize-options module.
from .source_pane import SourcePane


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

    def _build_legacy_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        title = QLabel("Organize photos")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.hide_button = QPushButton("Hide")
        self.hide_button.setProperty("paneToggle", True)
        self.hide_button.setFixedHeight(24)
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.hide_button)
        layout.addLayout(header)

        self._applying_preset = False
        self.preset_group = QGroupBox("Organization preset", self)
        preset_layout = QFormLayout(self.preset_group)
        self.preset_combo = QComboBox(self.preset_group)
        self.preset_combo.addItem("Quick groups", "fast_preview")
        self.preset_combo.addItem("Similar scenes", "balanced")
        self.preset_combo.addItem("Events", "events")
        self.preset_combo.addItem("Documents", "documents")
        self.preset_combo.addItem("People-heavy", "people_heavy")
        self.preset_combo.addItem("Detailed groups", "high_quality")
        self.preset_combo.addItem("Custom", "custom")
        self.preset_combo.setCurrentIndex(self.preset_combo.findData("balanced"))
        self.preset_summary = QLabel("SigLIP with CUDA-first cosine K-means and deterministic CPU fallback.")
        self.preset_summary.setWordWrap(True)
        preset_layout.addRow("Preset", self.preset_combo)
        preset_layout.addRow(self.preset_summary)
        self.preset_group.setVisible(self.option_scope == "production")
        layout.addWidget(self.preset_group)

        self.technical_panel = QWidget(self)
        technical_layout = QVBoxLayout(self.technical_panel)
        technical_layout.setContentsMargins(0, 0, 0, 0)
        technical_layout.setSpacing(6)

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
        technical_layout.addWidget(embedding_group)

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
        technical_layout.addWidget(backend_group)

        self.hdbscan_options_group = QGroupBox("HDBSCAN tuning")
        self.hdbscan_options_group.setToolTip(
            "Tune density clustering. These values are used by CUDA cuML when available and by native CPU HDBSCAN otherwise."
        )
        hdbscan_layout = QFormLayout(self.hdbscan_options_group)
        self.hdbscan_min_cluster_size_spin = QSpinBox()
        self.hdbscan_min_cluster_size_spin.setRange(2, 10_000)
        self.hdbscan_min_cluster_size_spin.setValue(2)
        self.hdbscan_min_cluster_size_spin.setToolTip("Smallest dense group that HDBSCAN may keep as a cluster.")
        hdbscan_layout.addRow("Minimum cluster size", self.hdbscan_min_cluster_size_spin)

        self.hdbscan_min_samples_spin = QSpinBox()
        self.hdbscan_min_samples_spin.setRange(0, 10_000)
        self.hdbscan_min_samples_spin.setSpecialValueText("Auto")
        self.hdbscan_min_samples_spin.setValue(0)
        self.hdbscan_min_samples_spin.setToolTip("Use Auto to let HDBSCAN derive the density threshold from minimum cluster size.")
        hdbscan_layout.addRow("Minimum samples", self.hdbscan_min_samples_spin)

        self.hdbscan_cluster_selection_epsilon_spin = QDoubleSpinBox()
        self.hdbscan_cluster_selection_epsilon_spin.setRange(0.0, 2.0)
        self.hdbscan_cluster_selection_epsilon_spin.setDecimals(3)
        self.hdbscan_cluster_selection_epsilon_spin.setSingleStep(0.01)
        self.hdbscan_cluster_selection_epsilon_spin.setValue(0.0)
        self.hdbscan_cluster_selection_epsilon_spin.setToolTip("Merge clusters separated by less than this distance; zero disables extra merging.")
        hdbscan_layout.addRow("Merge epsilon", self.hdbscan_cluster_selection_epsilon_spin)

        self.hdbscan_allow_single_cluster_checkbox = QCheckBox("Allow one cluster")
        self.hdbscan_allow_single_cluster_checkbox.setToolTip("Permit HDBSCAN to return one overall cluster when the density structure supports it.")
        hdbscan_layout.addRow(self.hdbscan_allow_single_cluster_checkbox)
        technical_layout.addWidget(self.hdbscan_options_group)
        hdbscan_checkbox = self.backend_checkboxes.get("hdbscan")
        if hdbscan_checkbox is not None:
            hdbscan_checkbox.toggled.connect(self._refresh_hdbscan_controls)

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

        # The production shell has a dedicated Tags workspace.  Keep this
        # legacy surface for the standalone compatibility shell until it gains
        # that workspace too.
        if self.option_scope != "production":
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

        technical_layout.addWidget(options_group)
        layout.addWidget(self.technical_panel)

        button_row = QHBoxLayout()
        self.cluster_button = QPushButton("Organize photos")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        button_row.addWidget(build_help_inline(self.cluster_button, CLUSTERING_HELP["run_clustering"], help_key="run_clustering"), 1)
        button_row.addWidget(build_help_inline(self.cancel_button, CLUSTERING_HELP["cancel"], help_key="cancel"), 1)
        layout.addLayout(button_row)
        layout.addStretch(1)

        self.cluster_button.clicked.connect(self.run_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._last_metrics_text = ""
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for checkbox in (*self.embedding_checkboxes.values(), *self.backend_checkboxes.values(), *self.similarity_checkboxes.values()):
            checkbox.toggled.connect(self._mark_custom_preset)
        for widget in (
            self.outlier_combobox,
            self.cluster_spinbox,
            self.recursive_checkbox,
            self.onnx_checkbox,
            self.result_cache_checkbox,
            self.embedding_cache_lookup_checkbox,
            *([self.tag_filter_field, self.tag_match_combobox] if self.option_scope != "production" else []),
            self.hdbscan_min_cluster_size_spin,
            self.hdbscan_min_samples_spin,
            self.hdbscan_cluster_selection_epsilon_spin,
            self.hdbscan_allow_single_cluster_checkbox,
        ):
            signal = getattr(widget, "textChanged", None) or getattr(widget, "valueChanged", None) or getattr(widget, "toggled", None) or getattr(widget, "currentIndexChanged", None)
            if signal is not None:
                signal.connect(self._mark_custom_preset)
        if self.option_scope == "production":
            self._apply_preset("balanced")
        else:
            self.technical_panel.show()
        self._refresh_hdbscan_controls()

    def _build_ui(self) -> None:
        """Build the production controls as a compact, tabbed advanced pane."""

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(6)
        self.controls_scroll = QScrollArea(self)
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setFrameStyle(0)
        controls_body = QWidget(self.controls_scroll)
        layout = QVBoxLayout(controls_body)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        self.controls_scroll.setWidget(controls_body)
        outer_layout.addWidget(self.controls_scroll, 1)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self.hide_button = QToolButton(self)
        self.hide_button.setProperty("iconOnly", True)
        self.hide_button.setToolTip("Hide advanced organize controls.")
        self.hide_button.setAccessibleName("Hide advanced organize controls")
        apply_icon(self.hide_button, "collapse")
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header.addStretch(1)
        header.addWidget(self.hide_button)
        layout.addLayout(header)

        self._applying_preset = False
        self.preset_group = QGroupBox("Preset", self)
        preset_layout = QVBoxLayout(self.preset_group)
        preset_layout.setContentsMargins(8, 8, 8, 8)
        preset_layout.setSpacing(4)
        self.preset_combo = QComboBox(self.preset_group)
        self.preset_combo.addItem("Quick groups", "fast_preview")
        self.preset_combo.addItem("Similar scenes", "balanced")
        self.preset_combo.addItem("Events", "events")
        self.preset_combo.addItem("Documents", "documents")
        self.preset_combo.addItem("People-heavy", "people_heavy")
        self.preset_combo.addItem("Detailed groups", "high_quality")
        self.preset_combo.addItem("Custom", "custom")
        self.preset_combo.setCurrentIndex(self.preset_combo.findData("balanced"))
        self.preset_combo.setToolTip("Choose a grouping preset. Changing an option changes this to Custom.")
        self.preset_summary = QLabel("Balanced visual grouping with GPU-first execution and deterministic CPU fallback.")
        self.preset_summary.setProperty("role", "helper")
        self.preset_summary.setWordWrap(True)
        self.preset_summary.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._sync_preset_summary_height()
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.preset_summary)
        self.preset_group.setVisible(self.option_scope == "production")
        layout.addWidget(self.preset_group)

        def field_cell(label: str, field: QWidget, parent: QWidget) -> QWidget:
            cell = QWidget(parent)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(3)
            title = QLabel(label, cell)
            title.setProperty("role", "helper")
            title.setBuddy(field)
            cell_layout.addWidget(title)
            cell_layout.addWidget(field)
            return cell

        self.technical_panel = QTabWidget(self)
        self.technical_panel.setObjectName("organizeAdvancedTabs")
        self.technical_panel.setDocumentMode(True)

        models_page = QWidget(self.technical_panel)
        models_layout = QVBoxLayout(models_page)
        models_layout.setContentsMargins(0, 0, 0, 0)
        models_layout.setSpacing(6)
        embedding_group = QGroupBox("Embedding models", models_page)
        embedding_group.setToolTip(CLUSTERING_HELP["embedding_models"])
        embedding_layout = ResponsiveFlowLayout(embedding_group, spacing=8)
        self.embedding_options_layout = embedding_layout
        self.embedding_checkboxes = {}
        for model_name in clustering_model_names(self.option_scope):
            checkbox = QCheckBox(model_label(model_name), embedding_group)
            checkbox.setToolTip(model_tooltip(model_name))
            checkbox.setChecked(model_name == self.settings.default_model)
            checkbox.toggled.connect(self.state_changed.emit)
            self.embedding_checkboxes[model_name] = checkbox
            embedding_layout.addWidget(checkbox)
        models_layout.addWidget(embedding_group)
        models_layout.addStretch(1)
        models_scroll = QScrollArea(self.technical_panel)
        models_scroll.setWidgetResizable(True)
        models_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        models_scroll.setWidget(models_page)
        self.technical_panel.addTab(models_scroll, "Models")

        grouping_page = QWidget(self.technical_panel)
        grouping_layout = QVBoxLayout(grouping_page)
        grouping_layout.setContentsMargins(0, 0, 0, 0)
        grouping_layout.setSpacing(6)
        backend_group = QGroupBox("Backends", grouping_page)
        backend_group.setToolTip(CLUSTERING_HELP["cluster_backends"])
        backend_layout = ResponsiveFlowLayout(backend_group, spacing=8)
        self.backend_options_layout = backend_layout
        self.backend_checkboxes = {}
        default_backends = set(default_backend_names(self.option_scope))
        for backend in clustering_backend_names(self.option_scope):
            checkbox = QCheckBox(backend, backend_group)
            checkbox.setToolTip(backend_tooltip(backend))
            checkbox.setChecked(backend in default_backends)
            checkbox.toggled.connect(self.state_changed.emit)
            self.backend_checkboxes[backend] = checkbox
            backend_layout.addWidget(checkbox)
        grouping_layout.addWidget(backend_group)

        comparison_group = QGroupBox("Comparison", grouping_page)
        comparison_layout = QGridLayout(comparison_group)
        comparison_layout.setContentsMargins(8, 8, 8, 8)
        comparison_layout.setHorizontalSpacing(8)
        comparison_layout.setVerticalSpacing(6)
        similarity_widget = QWidget(comparison_group)
        similarity_layout = ResponsiveFlowLayout(similarity_widget, spacing=8)
        self.similarity_options_layout = similarity_layout
        self.similarity_checkboxes = {}
        default_modes = normalize_similarity_modes(None, self.settings.default_similarity_mode)
        for mode in SUPPORTED_SIMILARITY_MODES:
            checkbox = QCheckBox(mode.capitalize(), similarity_widget)
            checkbox.setToolTip(CLUSTERING_HELP["similarity"])
            checkbox.setChecked(mode in default_modes)
            checkbox.toggled.connect(self._on_similarity_toggled)
            self.similarity_checkboxes[mode] = checkbox
            similarity_layout.addWidget(checkbox)
        comparison_layout.addWidget(field_cell("Similarity", similarity_widget, comparison_group), 0, 0, 1, 2)

        self.outlier_combobox = QComboBox(comparison_group)
        self.outlier_combobox.addItems(["assign", "isolate", "keep"])
        self.outlier_combobox.setCurrentText(self.settings.default_outlier_policy)
        self.outlier_combobox.setToolTip(CLUSTERING_HELP["outlier"])
        self.outlier_combobox.currentIndexChanged.connect(self.state_changed.emit)
        apply_field_size(self.outlier_combobox, "short")
        self._comparison_outlier_cell = field_cell("Outliers", self.outlier_combobox, comparison_group)
        comparison_layout.addWidget(self._comparison_outlier_cell, 1, 0)

        self.cluster_spinbox = QSpinBox(comparison_group)
        self.cluster_spinbox.setMinimum(self.settings.min_cluster_count)
        self.cluster_spinbox.setMaximum(500)
        self.cluster_spinbox.setValue(self.settings.default_cluster_count)
        self.cluster_spinbox.setToolTip(CLUSTERING_HELP["clusters"])
        self.cluster_spinbox.valueChanged.connect(self.state_changed.emit)
        apply_field_size(self.cluster_spinbox, "numeric")
        self._comparison_cluster_cell = field_cell("Clusters", self.cluster_spinbox, comparison_group)
        comparison_layout.addWidget(self._comparison_cluster_cell, 1, 1)
        self._comparison_layout = comparison_layout
        grouping_layout.addWidget(comparison_group)

        self.hdbscan_options_group = QGroupBox("HDBSCAN", grouping_page)
        self.hdbscan_options_group.setToolTip(
            "Tune density clustering. Values apply to CUDA cuML when available and native CPU HDBSCAN otherwise."
        )
        hdbscan_layout = ResponsiveFlowLayout(self.hdbscan_options_group, spacing=8)
        self.hdbscan_options_layout = hdbscan_layout
        self.hdbscan_min_cluster_size_spin = QSpinBox(self.hdbscan_options_group)
        self.hdbscan_min_cluster_size_spin.setRange(2, 10_000)
        self.hdbscan_min_cluster_size_spin.setValue(2)
        self.hdbscan_min_cluster_size_spin.setToolTip("Smallest dense group that HDBSCAN may keep as a cluster.")
        apply_field_size(self.hdbscan_min_cluster_size_spin, "numeric")
        hdbscan_layout.addWidget(field_cell("Min cluster", self.hdbscan_min_cluster_size_spin, self.hdbscan_options_group))
        self.hdbscan_min_samples_spin = QSpinBox(self.hdbscan_options_group)
        self.hdbscan_min_samples_spin.setRange(0, 10_000)
        self.hdbscan_min_samples_spin.setSpecialValueText("Auto")
        self.hdbscan_min_samples_spin.setValue(0)
        self.hdbscan_min_samples_spin.setToolTip("Auto derives the density threshold from minimum cluster size.")
        apply_field_size(self.hdbscan_min_samples_spin, "numeric")
        hdbscan_layout.addWidget(field_cell("Min samples", self.hdbscan_min_samples_spin, self.hdbscan_options_group))
        self.hdbscan_cluster_selection_epsilon_spin = QDoubleSpinBox(self.hdbscan_options_group)
        self.hdbscan_cluster_selection_epsilon_spin.setRange(0.0, 2.0)
        self.hdbscan_cluster_selection_epsilon_spin.setDecimals(3)
        self.hdbscan_cluster_selection_epsilon_spin.setSingleStep(0.01)
        self.hdbscan_cluster_selection_epsilon_spin.setValue(0.0)
        self.hdbscan_cluster_selection_epsilon_spin.setToolTip("Merge clusters closer than this distance; zero disables extra merging.")
        apply_field_size(self.hdbscan_cluster_selection_epsilon_spin, "numeric")
        hdbscan_layout.addWidget(field_cell("Merge epsilon", self.hdbscan_cluster_selection_epsilon_spin, self.hdbscan_options_group))
        self.hdbscan_allow_single_cluster_checkbox = QCheckBox("Allow one cluster", self.hdbscan_options_group)
        self.hdbscan_allow_single_cluster_checkbox.setToolTip("Permit one overall cluster when density structure supports it.")
        hdbscan_layout.addWidget(self.hdbscan_allow_single_cluster_checkbox)
        grouping_layout.addWidget(self.hdbscan_options_group)
        grouping_layout.addStretch(1)
        hdbscan_checkbox = self.backend_checkboxes.get("hdbscan")
        if hdbscan_checkbox is not None:
            hdbscan_checkbox.toggled.connect(self._refresh_hdbscan_controls)
        grouping_scroll = QScrollArea(self.technical_panel)
        grouping_scroll.setWidgetResizable(True)
        grouping_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        grouping_scroll.setWidget(grouping_page)
        self.technical_panel.addTab(grouping_scroll, "Grouping")

        runtime_page = QWidget(self.technical_panel)
        runtime_layout = QVBoxLayout(runtime_page)
        runtime_layout.setContentsMargins(0, 0, 0, 0)
        runtime_layout.setSpacing(6)
        runtime_group = QGroupBox("Runtime", runtime_page)
        runtime_group.setToolTip("Runtime choices are local, cancellable, and do not alter source photos.")
        runtime_controls = QVBoxLayout(runtime_group)
        runtime_controls.setContentsMargins(8, 8, 8, 8)
        runtime_controls.setSpacing(5)
        self.recursive_checkbox = QCheckBox("Include subfolders", runtime_group)
        self.recursive_checkbox.setChecked(self.settings.recursive_scan)
        self.recursive_checkbox.setToolTip(CLUSTERING_HELP["recursive_scan"])
        self.recursive_checkbox.toggled.connect(self.state_changed.emit)
        self.onnx_checkbox = QCheckBox("Use ONNX Runtime", runtime_group)
        self.onnx_checkbox.setChecked(self.settings.default_use_onnx)
        self.onnx_checkbox.setToolTip(CLUSTERING_HELP["use_onnx_runtime"])
        self.onnx_checkbox.toggled.connect(self.state_changed.emit)
        self.result_cache_checkbox = QCheckBox("Reuse result cache", runtime_group)
        self.result_cache_checkbox.setChecked(self.settings.default_reuse_result_cache)
        self.result_cache_checkbox.setToolTip(CLUSTERING_HELP["reuse_cluster_result_cache"])
        self.result_cache_checkbox.toggled.connect(self.state_changed.emit)
        self.embedding_cache_lookup_checkbox = QCheckBox("Reuse embeddings", runtime_group)
        self.embedding_cache_lookup_checkbox.setChecked(True)
        self.embedding_cache_lookup_checkbox.setToolTip(CLUSTERING_HELP["use_embedding_cache_lookup"])
        self.embedding_cache_lookup_checkbox.toggled.connect(self.state_changed.emit)
        for checkbox in (self.recursive_checkbox, self.onnx_checkbox, self.result_cache_checkbox, self.embedding_cache_lookup_checkbox):
            runtime_controls.addWidget(checkbox)
        if self.option_scope != "production":
            self.tag_filter_field = QLineEdit(runtime_group)
            self.tag_filter_field.setPlaceholderText("Comma-separated tags")
            self.tag_filter_field.setToolTip(CLUSTERING_HELP["tag_filter"])
            self.tag_filter_field.textChanged.connect(self.state_changed.emit)
            self.tag_match_combobox = QComboBox(runtime_group)
            self.tag_match_combobox.addItems(["Any", "All"])
            self.tag_match_combobox.setToolTip(CLUSTERING_HELP["tag_match"])
            self.tag_match_combobox.currentIndexChanged.connect(self.state_changed.emit)
            apply_field_size(self.tag_match_combobox, "short")
            tag_grid = QGridLayout()
            tag_grid.addWidget(field_cell("Tags", self.tag_filter_field, runtime_group), 0, 0, 1, 2)
            tag_grid.addWidget(field_cell("Match", self.tag_match_combobox, runtime_group), 1, 0)
            runtime_controls.addLayout(tag_grid)
        runtime_layout.addWidget(runtime_group)
        runtime_layout.addStretch(1)
        runtime_scroll = QScrollArea(self.technical_panel)
        runtime_scroll.setWidgetResizable(True)
        runtime_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        runtime_scroll.setWidget(runtime_page)
        self.technical_panel.addTab(runtime_scroll, "Runtime")
        self.technical_panel.setTabToolTip(0, "Embedding models")
        self.technical_panel.setTabToolTip(1, "Grouping and comparison options")
        self.technical_panel.setTabToolTip(2, "Runtime and cache options")
        self._sync_technical_tab_labels()
        self.technical_panel.setCurrentIndex(1)
        layout.addWidget(self.technical_panel, 1)

        self.run_actions_row = QWidget(self)
        button_row = ResponsiveFlowLayout(self.run_actions_row, spacing=6)
        self.cluster_button = QPushButton("Organize", self)
        self.cluster_button.setProperty("kind", "primary")
        self.cluster_button.setToolTip(CLUSTERING_HELP["run_clustering"])
        apply_icon(self.cluster_button, "organize")
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.setToolTip(CLUSTERING_HELP["cancel"])
        self.cancel_button.setEnabled(False)
        apply_icon(self.cancel_button, "cancel")
        button_row.addWidget(self.cluster_button)
        button_row.addWidget(self.cancel_button)
        outer_layout.addWidget(self.run_actions_row)

        self.cluster_button.clicked.connect(self.run_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self._last_metrics_text = ""
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        for checkbox in (*self.embedding_checkboxes.values(), *self.backend_checkboxes.values(), *self.similarity_checkboxes.values()):
            checkbox.toggled.connect(self._mark_custom_preset)
        for widget in (
            self.outlier_combobox, self.cluster_spinbox, self.recursive_checkbox, self.onnx_checkbox,
            self.result_cache_checkbox, self.embedding_cache_lookup_checkbox,
            *([self.tag_filter_field, self.tag_match_combobox] if self.option_scope != "production" else []),
            self.hdbscan_min_cluster_size_spin, self.hdbscan_min_samples_spin,
            self.hdbscan_cluster_selection_epsilon_spin, self.hdbscan_allow_single_cluster_checkbox,
        ):
            signal = getattr(widget, "textChanged", None) or getattr(widget, "valueChanged", None) or getattr(widget, "toggled", None) or getattr(widget, "currentIndexChanged", None)
            if signal is not None:
                signal.connect(self._mark_custom_preset)
        if self.option_scope == "production":
            self._apply_preset("balanced")
        self._refresh_hdbscan_controls()

    def _sync_preset_summary_height(self) -> None:
        if hasattr(self, "preset_summary"):
            self.preset_summary.setMinimumHeight(self.preset_summary.fontMetrics().lineSpacing() * 3)

    def _sync_technical_tab_labels(self) -> None:
        if not hasattr(self, "technical_panel") or self.technical_panel.count() < 3:
            return
        app = QApplication.instance()
        try:
            scale = int(app.property("clusterlens_text_scale") or 100) if app is not None else 100
        except (TypeError, ValueError):
            scale = 100
        labels = ("Model", "Group", "Run") if scale >= 150 else ("Models", "Grouping", "Runtime")
        for index, label in enumerate(labels):
            self.technical_panel.setTabText(index, label)

    def changeEvent(self, event) -> None:
        result = super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.StyleChange}:
            self._sync_preset_summary_height()
            self._sync_technical_tab_labels()
        return result

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._refresh_compact_grid()

    def _refresh_compact_grid(self) -> None:
        """Avoid forcing two short controls beside one another in a narrow rail."""

        layout = getattr(self, "_comparison_layout", None)
        if layout is None:
            return
        narrow = self.width() < 350
        layout.removeWidget(self._comparison_outlier_cell)
        layout.removeWidget(self._comparison_cluster_cell)
        if narrow:
            layout.addWidget(self._comparison_outlier_cell, 1, 0, 1, 2)
            layout.addWidget(self._comparison_cluster_cell, 2, 0, 1, 2)
        else:
            layout.addWidget(self._comparison_outlier_cell, 1, 0)
            layout.addWidget(self._comparison_cluster_cell, 1, 1)

    def _on_preset_changed(self, _index: int) -> None:
        preset = str(self.preset_combo.currentData() or "custom")
        self._apply_preset(preset)
        self.state_changed.emit()

    def _mark_custom_preset(self, *_args) -> None:
        if self._applying_preset or self.option_scope != "production":
            return
        custom_index = self.preset_combo.findData("custom")
        if self.preset_combo.currentIndex() != custom_index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(custom_index)
            self.preset_combo.blockSignals(False)
        self.technical_panel.show()
        self.preset_summary.setText("Custom configuration.")

    def _refresh_hdbscan_controls(self, *_args) -> None:
        checkbox = self.backend_checkboxes.get("hdbscan")
        self.hdbscan_options_group.setVisible(bool(checkbox and checkbox.isChecked()))

    def _apply_preset(self, preset: str) -> None:
        preset = str(preset or "custom")
        summaries = {
            "fast_preview": "Fast, low-memory preview.",
            "balanced": "Balanced visual grouping with GPU-first execution.",
            "events": "Event-aware groups with isolated outliers.",
            "documents": "Broad structure for scans and documents.",
            "people_heavy": "Stronger comparisons for people-heavy sets.",
            "high_quality": "Higher-quality groups; uses more time and memory.",
            "custom": "Custom configuration.",
        }
        self.preset_summary.setText(summaries.get(preset, summaries["custom"]))
        # The production controls pane is only shown in Advanced workspace
        # mode, so presets should populate its controls instead of concealing
        # them. Hiding this panel made HDBSCAN, graph clustering, model,
        # similarity, outlier, and runtime options appear to have been removed.
        self.technical_panel.setVisible(True)
        if preset == "custom":
            return
        configurations = {
            "fast_preview": {
                "models": {"fast_preview"}, "backends": {"cosine-kmeans"}, "modes": {"semantic"},
                "outlier": "assign", "clusters": 8, "onnx": True,
            },
            "balanced": {
                "models": {"siglip"}, "backends": {"cosine-kmeans"}, "modes": {"semantic"},
                "outlier": "isolate", "clusters": 12, "onnx": False,
            },
            "events": {
                "models": {"siglip"}, "backends": {"hdbscan", "graph"}, "modes": {"semantic"},
                "outlier": "isolate", "clusters": 18, "onnx": False,
            },
            "documents": {
                "models": {"clip"}, "backends": {"cosine-kmeans", "graph"}, "modes": {"cosine"},
                "outlier": "isolate", "clusters": 12, "onnx": False,
            },
            "people_heavy": {
                "models": {"siglip", "clip"}, "backends": {"cosine-kmeans", "hdbscan"}, "modes": {"semantic", "cosine"},
                "outlier": "isolate", "clusters": 16, "onnx": False,
            },
            "high_quality": {
                "models": {"dinov2_base", "clip"}, "backends": {"cosine-kmeans", "graph"}, "modes": {"semantic", "cosine"},
                "outlier": "isolate", "clusters": 16, "onnx": False,
            },
        }
        config = configurations.get(preset)
        if config is None:
            return
        self._applying_preset = True
        try:
            for name, checkbox in self.embedding_checkboxes.items():
                checkbox.setChecked(name in config["models"])
            for name, checkbox in self.backend_checkboxes.items():
                checkbox.setChecked(name in config["backends"])
            for name, checkbox in self.similarity_checkboxes.items():
                checkbox.setChecked(name in config["modes"])
            self.outlier_combobox.setCurrentText(str(config["outlier"]))
            self.cluster_spinbox.setValue(int(config["clusters"]))
            self.onnx_checkbox.setChecked(bool(config["onnx"]))
            self.result_cache_checkbox.setChecked(True)
            self.embedding_cache_lookup_checkbox.setChecked(True)
            self.hdbscan_min_cluster_size_spin.setValue(2)
            self.hdbscan_min_samples_spin.setValue(0)
            self.hdbscan_cluster_selection_epsilon_spin.setValue(0.0)
            self.hdbscan_allow_single_cluster_checkbox.setChecked(False)
        finally:
            self._applying_preset = False
        self._refresh_hdbscan_controls()

    def set_running(self, running: bool) -> None:
        self.cluster_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)

    def update_progress(self, value: int, status: str) -> None:
        _ = value
        _ = status

    def selected_clustering_backends(self) -> list[str]:
        selected = [backend for backend, checkbox in self.backend_checkboxes.items() if checkbox.isChecked()]
        return normalize_clustering_backends(selected, scope=self.option_scope)

    def backend_options_by_backend(self) -> dict[str, dict[str, object]]:
        hdbscan_checkbox = self.backend_checkboxes.get("hdbscan")
        if hdbscan_checkbox is None or not hdbscan_checkbox.isChecked():
            return {}
        return {
            "hdbscan": {
                "min_cluster_size": int(self.hdbscan_min_cluster_size_spin.value()),
                "min_samples": int(self.hdbscan_min_samples_spin.value()),
                "cluster_selection_epsilon": float(self.hdbscan_cluster_selection_epsilon_spin.value()),
                "allow_single_cluster": bool(self.hdbscan_allow_single_cluster_checkbox.isChecked()),
            }
        }

    def selected_embedding_models(self) -> list[str]:
        selected = [name for name, checkbox in self.embedding_checkboxes.items() if checkbox.isChecked()]
        return normalize_embedding_models(
            selected,
            default_model=self.settings.default_model,
            scope=self.option_scope,
        )

    def selected_tag_filters(self) -> list[str]:
        field = getattr(self, "tag_filter_field", None)
        return [tag.strip() for tag in field.text().split(",") if tag.strip()] if field is not None else []

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
        combobox = getattr(self, "tag_match_combobox", None)
        return combobox.currentText().strip() or "Any" if combobox is not None else "Any"

    def export_state(self) -> dict[str, object]:
        similarity_modes = self.selected_similarity_modes()
        return {
            "preset": str(self.preset_combo.currentData() or "custom"),
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
            "tag_filter": getattr(self, "tag_filter_field", None).text().strip() if hasattr(self, "tag_filter_field") else "",
            "tag_match": self.selected_tag_match_mode(),
            "backend_options_by_backend": self.backend_options_by_backend(),
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self._applying_preset = True
        preset = str(state.get("preset") or "custom")
        preset_index = self.preset_combo.findData(preset)
        if preset_index >= 0:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(preset_index)
            self.preset_combo.blockSignals(False)
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
        raw_backend_options = state.get("backend_options_by_backend", {})
        backend_options = dict(raw_backend_options) if isinstance(raw_backend_options, dict) else {}
        raw_hdbscan_options = backend_options.get("hdbscan", {})
        hdbscan_options = dict(raw_hdbscan_options) if isinstance(raw_hdbscan_options, dict) else {}
        self.hdbscan_min_cluster_size_spin.setValue(max(2, int(hdbscan_options.get("min_cluster_size", 2) or 2)))
        self.hdbscan_min_samples_spin.setValue(max(0, int(hdbscan_options.get("min_samples", 0) or 0)))
        self.hdbscan_cluster_selection_epsilon_spin.setValue(
            max(0.0, float(hdbscan_options.get("cluster_selection_epsilon", 0.0) or 0.0))
        )
        self.hdbscan_allow_single_cluster_checkbox.setChecked(bool(hdbscan_options.get("allow_single_cluster", False)))
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
        if hasattr(self, "tag_filter_field"):
            self.tag_filter_field.setText(str(state.get("tag_filter") or ""))
            self.tag_match_combobox.setCurrentText(str(state.get("tag_match") or "Any"))
        self._applying_preset = False
        self.technical_panel.setVisible(True)
        self._refresh_hdbscan_controls()
        self.preset_summary.setText({
            "fast_preview": "Fast, low-memory preview.",
            "balanced": "Balanced visual grouping with GPU-first execution.",
            "events": "Event-aware groups with isolated outliers.",
            "documents": "Broad structure for scans and documents.",
            "people_heavy": "Stronger comparisons for people-heavy sets.",
            "high_quality": "Higher-quality groups; uses more time and memory.",
            "custom": "Custom configuration.",
        }.get(preset, "Custom settings."))

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
            "compute_fallback_reason",
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


class ReviewScopePane(QWidget):
    clear_requested = pyqtSignal()
    export_requested = pyqtSignal()
    use_as_scope_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title = QLabel("Review")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        self.summary_label = QLabel("Review collection is empty.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.use_scope_button = QPushButton("Use Review As Scope")
        self.export_button = QPushButton("Export Paths")
        self.clear_button = QPushButton("Clear Review")
        layout.addWidget(self.use_scope_button)
        layout.addWidget(self.export_button)
        layout.addWidget(self.clear_button)
        layout.addStretch(1)

        self.clear_button.clicked.connect(self.clear_requested.emit)
        self.export_button.clicked.connect(self.export_requested.emit)
        self.use_scope_button.clicked.connect(self.use_as_scope_requested.emit)

    def set_summary(self, text: str) -> None:
        self.summary_label.setText(str(text or "Review collection is empty."))
