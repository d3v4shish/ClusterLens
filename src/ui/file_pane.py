from __future__ import annotations

import os

from PyQt6.QtCore import QDir, pyqtSignal
from PyQt6.QtGui import QFileSystemModel
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.services.similarity_modes import SUPPORTED_SIMILARITY_MODES, normalize_similarity_modes
from infra.settings import get_settings

from .common import Log


class FilePane(QWidget):
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    directory_changed = pyqtSignal(str)
    state_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.selected_directory = "D:/" if os.name == "nt" else "/home"
        self.init_ui()
        Log.debug("FilePane Init.")

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        directory_group = QGroupBox("Source")
        directory_layout = QVBoxLayout(directory_group)
        self.directory_combobox = QComboBox()
        self.directory_combobox.setStyleSheet("QComboBox { font-size: 13px; padding: 6px; }")
        self.populate_drives()
        self.directory_combobox.currentIndexChanged.connect(self.on_directory_changed)
        directory_layout.addWidget(self.directory_combobox)

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
        # Larger folder selection targets.
        self.file_tree.setStyleSheet(
            "QTreeView { font-size: 13px; }"
            "QTreeView::item { height: 26px; padding: 4px; }"
        )
        self.file_tree.clicked.connect(self.on_directory_selected)
        directory_layout.addWidget(self.file_tree)
        layout.addWidget(directory_group)

        embedding_group = QGroupBox("Embedding Models")
        embedding_layout = QVBoxLayout(embedding_group)
        self.embedding_checkboxes = {}
        onnx_ready = {"fast_preview", "convnext", "resnet"}
        for model_name in [
            "fast_preview",
            "clip",
            "siglip",
            "convnext",
            "phash_embedding",
            "resnet",
            "vgg",
            "vit",
            "facenet",
            "dino",
            "dino_large",
        ]:
            label = model_name
            if model_name in onnx_ready:
                label = f"{model_name} (ONNX)"
            else:
                label = f"{model_name} (Torch)"
            checkbox = QCheckBox(label)
            checkbox.setToolTip("ONNX-ready model (CUDA/CPU compatible)" if model_name in onnx_ready else "Torch model")
            checkbox.setChecked(model_name == self.settings.default_model)
            checkbox.toggled.connect(self.state_changed.emit)
            self.embedding_checkboxes[model_name] = checkbox
            embedding_layout.addWidget(checkbox)
        layout.addWidget(embedding_group)

        backend_group = QGroupBox("Cluster Backends")
        backend_layout = QVBoxLayout(backend_group)
        self.backend_checkboxes = {}
        for backend in ["cosine-kmeans", "hdbscan", "graph", "faiss", "sklearn"]:
            checkbox = QCheckBox(backend)
            checkbox.setChecked(backend in {"cosine-kmeans", "hdbscan", "graph"})
            checkbox.toggled.connect(self.state_changed.emit)
            self.backend_checkboxes[backend] = checkbox
            backend_layout.addWidget(checkbox)
        layout.addWidget(backend_group)

        options_group = QGroupBox("Clustering")
        options_layout = QVBoxLayout(options_group)

        similarity_row = QHBoxLayout()
        similarity_row.addWidget(QLabel("Similarity"))
        self.similarity_checkboxes = {}
        default_modes = normalize_similarity_modes(None, self.settings.default_similarity_mode)
        for mode in SUPPORTED_SIMILARITY_MODES:
            checkbox = QCheckBox(mode)
            checkbox.setChecked(mode in default_modes)
            checkbox.toggled.connect(self._on_similarity_toggled)
            self.similarity_checkboxes[mode] = checkbox
            similarity_row.addWidget(checkbox)
        similarity_row.addStretch(1)
        options_layout.addLayout(similarity_row)

        outlier_row = QHBoxLayout()
        outlier_row.addWidget(QLabel("Outlier"))
        self.outlier_combobox = QComboBox()
        self.outlier_combobox.addItems(["assign", "isolate", "keep"])
        self.outlier_combobox.setCurrentText(self.settings.default_outlier_policy)
        self.outlier_combobox.currentIndexChanged.connect(self.state_changed.emit)
        outlier_row.addWidget(self.outlier_combobox)
        options_layout.addLayout(outlier_row)

        cluster_row = QHBoxLayout()
        cluster_row.addWidget(QLabel("Clusters"))
        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setMinimum(self.settings.min_cluster_count)
        self.cluster_spinbox.setMaximum(500)
        self.cluster_spinbox.setValue(self.settings.default_cluster_count)
        self.cluster_spinbox.valueChanged.connect(self.state_changed.emit)
        cluster_row.addWidget(self.cluster_spinbox)
        options_layout.addLayout(cluster_row)

        self.recursive_checkbox = QCheckBox("Recursive Scan")
        self.recursive_checkbox.setChecked(self.settings.recursive_scan)
        self.recursive_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addWidget(self.recursive_checkbox)

        self.onnx_checkbox = QCheckBox("Use ONNX Runtime")
        self.onnx_checkbox.setChecked(self.settings.default_use_onnx)
        self.onnx_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addWidget(self.onnx_checkbox)

        self.result_cache_checkbox = QCheckBox("Reuse Cluster Result Cache")
        self.result_cache_checkbox.setChecked(self.settings.default_reuse_result_cache)
        self.result_cache_checkbox.toggled.connect(self.state_changed.emit)
        options_layout.addWidget(self.result_cache_checkbox)
        layout.addWidget(options_group)

        button_row = QHBoxLayout()
        self.cluster_button = QPushButton("Run Clustering")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        button_row.addWidget(self.cluster_button)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)

        self.file_pane_progress_bar = QProgressBar()
        self.file_pane_progress_bar.setVisible(False)
        self.file_pane_progress_bar.setValue(0)
        layout.addWidget(self.file_pane_progress_bar)

        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.metrics_label)
        layout.addStretch(1)

        self.cluster_button.clicked.connect(self.run_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

    def populate_drives(self):
        self.directory_combobox.clear()
        if os.name == "nt":
            from string import ascii_uppercase

            drives = [f"{letter}:/" for letter in ascii_uppercase if os.path.exists(f"{letter}:/")]
            self.selected_directory = drives[0] if drives else "C:/"
        else:
            drives = ["/", "/home"]
            self.selected_directory = "/home"
        self.directory_combobox.addItems(drives)

    def on_directory_selected(self, index):
        self.selected_directory = self.file_model.filePath(index)
        self.directory_changed.emit(self.selected_directory)
        self.state_changed.emit()

    def on_directory_changed(self, index):
        root_directory = self.directory_combobox.currentText()
        self.selected_directory = root_directory
        self.file_model.setRootPath(root_directory)
        self.file_tree.setRootIndex(self.file_model.index(root_directory))
        self.directory_changed.emit(self.selected_directory)
        self.state_changed.emit()
        Log.debug("Selected Directory: %s", self.selected_directory)

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
        self.cluster_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.file_pane_progress_bar.setVisible(running)
        if not running:
            self.file_pane_progress_bar.reset()

    def update_progress(self, value: int, status: str) -> None:
        self.file_pane_progress_bar.setVisible(True)
        self.file_pane_progress_bar.setValue(value)
        self.file_pane_progress_bar.setFormat(f"{status} ... %p%")

    def selected_clustering_backends(self) -> list[str]:
        return [backend for backend, checkbox in self.backend_checkboxes.items() if checkbox.isChecked()]

    def selected_embedding_models(self) -> list[str]:
        selected = [name for name, checkbox in self.embedding_checkboxes.items() if checkbox.isChecked()]
        if not selected:
            selected = [self.settings.default_model]
        return selected

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

    def export_state(self) -> dict[str, object]:
        similarity_modes = self.selected_similarity_modes()
        return {
            "selected_directory": self.selected_directory,
            "embedding_models": self.selected_embedding_models(),
            "clustering_backends": self.selected_clustering_backends(),
            "similarity_modes": similarity_modes,
            "similarity_mode": similarity_modes[0],
            "outlier_policy": self.outlier_combobox.currentText(),
            "cluster_count": int(self.cluster_spinbox.value()),
            "recursive": bool(self.recursive_checkbox.isChecked()),
            "use_onnx": bool(self.onnx_checkbox.isChecked()),
            "reuse_result_cache": bool(self.result_cache_checkbox.isChecked()),
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        self.set_selected_directory(str(state.get("selected_directory", self.selected_directory)))
        for name, checkbox in self.embedding_checkboxes.items():
            checkbox.setChecked(name in set(state.get("embedding_models") or []))
        for name, checkbox in self.backend_checkboxes.items():
            checkbox.setChecked(name in set(state.get("clustering_backends") or []))
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

    def update_metrics(self, metrics: dict) -> None:
        ordered_keys = [
            "image_count",
            "scan_time_s",
            "embedding_models",
            "similarity_modes",
            "embedding_time_s",
            "clustering_time_s",
            "model_load_time_s",
            "cache_hits",
            "cache_misses",
            "result_cache_hit",
            "batch_size",
            "model_device",
            "effective_mode",
            "onnx_provider",
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
        self.metrics_label.setText("\n".join(parts))
