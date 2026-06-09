from __future__ import annotations

import sys

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from apps.shared.runtime_support import RuntimeLayout
from apps.pyqt_production.identity import PRODUCTION_DISPLAY_NAME
from app.services.clustering_options import model_label
from infra.performance import detect_system_resources, select_performance_profile
from infra.runtime import available_execution_modes
from infra.settings import get_settings


class FirstRunSetupDialog(QDialog):
    """Small first-run policy dialog for production installs.

    The goal is not to block the user with every tuning knob. It exposes only
    decisions that materially affect data safety, downloads, and runtime memory.
    Everything can be changed later from Settings.
    """

    def __init__(self, settings_store: QSettings, runtime_layout: RuntimeLayout, parent=None) -> None:
        super().__init__(parent)
        self.settings_store = settings_store
        self.runtime_layout = runtime_layout
        self.app_settings = get_settings()
        self.setWindowTitle(f"{PRODUCTION_DISPLAY_NAME} First Run Setup")
        self.resize(720, 480)
        self._build_ui()
        self._load_defaults()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose the production safety defaults for this workstation. "
            "These settings can be changed later from Settings."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.offline_mode = QCheckBox("Never download missing model files without asking")
        self.offline_mode.setToolTip("Recommended for packaged/offline installs. The app falls back to bundled assets when possible.")
        self.download_default_model = QCheckBox(
            f"Download default model after setup ({model_label(self.app_settings.default_model)})"
        )
        self.download_default_model.setToolTip(
            "Downloads model weights into the app runtime cache after install. No model weights are stored in the source tree."
        )
        self.read_only_mode = QCheckBox("Start in read-only safety mode")
        self.read_only_mode.setToolTip("Disables file moves, deletes, tag writes, and EXIF writes until you turn it off.")
        self.execution_mode = QComboBox()
        self.execution_mode.addItems(list(available_execution_modes()))
        self.performance_profile = QComboBox()
        self.performance_profile.addItems(["low_memory", "balanced", "max_speed"])
        self.keep_worker_warm = QCheckBox("Keep ML worker warm between runs")
        self.keep_worker_warm.setToolTip("Faster repeated clustering, but keeps RAM/VRAM allocated while idle.")

        form.addRow("Model downloads", self.offline_mode)
        form.addRow("Post-install model", self.download_default_model)
        form.addRow("Data safety", self.read_only_mode)
        form.addRow("Runtime", self.execution_mode)
        form.addRow("Performance", self.performance_profile)
        form.addRow("Repeated runs", self.keep_worker_warm)
        layout.addLayout(form)

        resources = detect_system_resources()
        profile = select_performance_profile("balanced", resources)
        paths = QLabel(
            "Runtime folders:\n"
            f"Logs: {self.runtime_layout.logs_dir}\n"
            f"Cache: {self.runtime_layout.cache_dir}\n"
            f"Crash reports: {self.runtime_layout.crash_dir}\n"
            f"Model assets: {self.runtime_layout.model_assets_dir}\n\n"
            f"Detected CPU cores: {profile.logical_cpu_count}"
        )
        paths.setWordWrap(True)
        layout.addWidget(paths)

        buttons = QDialogButtonBox(parent=self)
        self.save_button = buttons.addButton("Save Setup", QDialogButtonBox.ButtonRole.AcceptRole)
        self.skip_button = buttons.addButton("Skip", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_defaults(self) -> None:
        self.offline_mode.setChecked(self.settings_store.value("models/offline_mode", bool(getattr(sys, "frozen", False)), bool))
        self.download_default_model.setChecked(self.settings_store.value("models/download_default_after_setup", False, bool))
        self.read_only_mode.setChecked(self.settings_store.value("safety/read_only_mode", False, bool))
        self.execution_mode.setCurrentText(
            self.settings_store.value("runtime/preferred_mode", self.app_settings.preferred_execution_mode, str)
        )
        self.performance_profile.setCurrentText(
            self.settings_store.value("performance/profile", self.app_settings.default_performance_profile, str)
        )
        self.keep_worker_warm.setChecked(self.settings_store.value("performance/keep_worker_warm", False, bool))

    def values(self) -> dict[str, object]:
        return {
            "models/offline_mode": bool(self.offline_mode.isChecked()),
            "models/download_default_after_setup": bool(self.download_default_model.isChecked()),
            "safety/read_only_mode": bool(self.read_only_mode.isChecked()),
            "runtime/preferred_mode": self.execution_mode.currentText(),
            "performance/profile": self.performance_profile.currentText(),
            "performance/keep_worker_warm": bool(self.keep_worker_warm.isChecked()),
        }
