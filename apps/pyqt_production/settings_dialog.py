from __future__ import annotations

from collections.abc import Callable
from collections import deque
import json
from pathlib import Path
import subprocess
import sys

from PyQt6.QtCore import QSettings, Qt, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QHeaderView,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary
from app.services.clustering_options import clustering_model_names, model_label
from app.services.gallery_actions import GalleryActionService
from app.services.model_assets import BUNDLED_ONNX_INPUT_SIZES, ModelAssetService
from apps.pyqt_production.ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from apps.pyqt_production.ui.error_mbox import confirmBox, errorBox, infoBox
from apps.pyqt_production.identity import PRODUCTION_DISPLAY_NAME
from apps.shared.runtime_support import RuntimeLayout, open_path_in_shell
from apps.shared.support_bundle import export_support_bundle
from infra.performance import detect_system_resources, select_performance_profile
from infra.runtime import RuntimeCapabilityService, available_execution_modes
from infra.settings import get_settings


class ProductionSettingsDialog(QDialog):
    def __init__(
        self,
        settings_store: QSettings,
        runtime_service: RuntimeCapabilityService,
        *,
        describe_rebuildable_caches: Callable[[], CacheUsageSummary] | None = None,
        clear_rebuildable_caches: Callable[[], CacheClearResult] | None = None,
        can_clear_rebuildable_caches: Callable[[], bool] | None = None,
        runtime_layout: RuntimeLayout,
        support_metadata_provider,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.app_settings = get_settings()
        self.settings_store = settings_store
        self.runtime_service = runtime_service
        self.describe_rebuildable_caches = describe_rebuildable_caches
        self.clear_rebuildable_caches = clear_rebuildable_caches
        self.can_clear_rebuildable_caches = can_clear_rebuildable_caches or (lambda: True)
        self.runtime_layout = runtime_layout
        self.support_metadata_provider = support_metadata_provider
        self.model_asset_service = ModelAssetService(runtime_model_assets_dir=runtime_layout.model_assets_dir)
        self.system_resources = detect_system_resources()
        self._verify_job = None
        self._verify_thread = None
        self._cache_usage_job = None
        self._cache_usage_thread = None
        self._cache_clear_job = None
        self._cache_clear_thread = None
        self._model_job = None
        self._model_thread = None
        self._journal_restore_job = None
        self._journal_restore_thread = None
        self._release_gate_job = None
        self._release_gate_thread = None
        self._thread_roles: dict[object, tuple[str, object | None]] = {}
        self._last_verify: dict[str, object] | None = None
        self.gallery_action_service = GalleryActionService()

        self.setWindowTitle(f"{PRODUCTION_DISPLAY_NAME} Settings")
        self.resize(900, 640)
        self._build_ui()
        self._load_values()
        self.refresh_runtime_diagnostics()
        self.refresh_cache_usage()
        self.refresh_model_inventory()
        self.refresh_operation_journal()
        self.refresh_log_viewer()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs)
        self._build_runtime_tab()
        self._build_models_tab()
        self._build_storage_tab()
        self._build_safety_tab()
        self._build_diagnostics_tab()
        self._build_about_tab()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_runtime_tab(self) -> None:
        tab = QWidget(self)
        form = QFormLayout(tab)
        self.execution_mode = QComboBox()
        self.execution_mode.addItems(list(available_execution_modes()))
        self.performance_profile = QComboBox()
        self.performance_profile.addItems(["low_memory", "balanced", "max_speed"])
        self.performance_profile.setToolTip(
            "low_memory limits worker counts and caches; balanced is the default; max_speed uses larger caches, unbounded backend parallelism, and faster KMeans on larger folders."
        )
        self.thumbnail_size = QSpinBox()
        self.thumbnail_size.setRange(96, 512)
        self.thumbnail_workers = QSpinBox()
        self.thumbnail_workers.setRange(1, max(8, self.system_resources.logical_cpu_count))
        self.thumbnail_workers.setEnabled(False)
        self.prefetch_rows = QSpinBox()
        self.prefetch_rows.setRange(0, max(16, self.system_resources.logical_cpu_count * 2))
        self.prefetch_rows.setEnabled(False)
        self.gpu_warmup = QCheckBox("Warm selected embedding model after folder change")
        self.keep_worker_warm = QCheckBox("Keep clustering worker warm between runs")
        self.keep_worker_warm.setToolTip(
            "Keeps Python, Torch, and loaded models alive after a run. Faster repeated runs, but uses more RAM/VRAM while idle."
        )
        self.runtime_badge = QCheckBox("Show runtime badge in toolbar")
        self.dense_ui = QCheckBox("Use dense desktop spacing")

        form.addRow("Preferred execution mode", self.execution_mode)
        form.addRow("Performance profile", self.performance_profile)
        form.addRow("Thumbnail size", self.thumbnail_size)
        form.addRow("Thumbnail workers", self.thumbnail_workers)
        form.addRow("Thumbnail prefetch rows", self.prefetch_rows)
        form.addRow(self.gpu_warmup)
        form.addRow(self.keep_worker_warm)
        form.addRow(self.runtime_badge)
        form.addRow(self.dense_ui)

        actions = QHBoxLayout()
        self.refresh_runtime_button = QPushButton("Refresh Diagnostics")
        self.verify_runtime_button = QPushButton("Run Runtime Verify")
        actions.addWidget(self.refresh_runtime_button)
        actions.addWidget(self.verify_runtime_button)
        form.addRow(actions)

        self.runtime_text = QTextEdit()
        self.runtime_text.setReadOnly(True)
        form.addRow(QLabel("Runtime diagnostics"), self.runtime_text)
        self.tabs.addTab(tab, "Runtime")

        self.refresh_runtime_button.clicked.connect(self.refresh_runtime_diagnostics)
        self.verify_runtime_button.clicked.connect(self._verify_gpu)
        self.performance_profile.currentIndexChanged.connect(self._apply_profile_preview)
        self.execution_mode.currentIndexChanged.connect(self.refresh_runtime_diagnostics)
        self.keep_worker_warm.toggled.connect(self.refresh_runtime_diagnostics)

    def _build_models_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        self.offline_model_downloads = QCheckBox("Offline mode: never download missing model files automatically")
        self.offline_model_downloads.setToolTip(
            "When enabled, clustering uses only bundled model assets or already cached model files. Missing models fall back or fail visibly."
        )
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

        actions = QHBoxLayout()
        self.refresh_models_button = QPushButton("Refresh Models")
        self.install_model_button = QPushButton("Download / Install Selected")
        self.delete_model_cache_button = QPushButton("Delete Cached Download")
        self.open_model_assets_button = QPushButton("Open Model Assets")
        self.open_download_cache_button = QPushButton("Open Download Cache")
        for button in (
            self.refresh_models_button,
            self.install_model_button,
            self.delete_model_cache_button,
            self.open_model_assets_button,
            self.open_download_cache_button,
        ):
            actions.addWidget(button)
        layout.addLayout(actions)

        self.model_status_label = QLabel(
            "Packaged ONNX assets are verified by checksum when metadata contains sha256. "
            "Cached Torch/Hugging Face downloads are shown separately and can be deleted here."
        )
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        self.license_text = QTextEdit(self)
        self.license_text.setReadOnly(True)
        self.license_text.setMaximumHeight(140)
        layout.addWidget(self.license_text)
        self.tabs.addTab(tab, "Models")

        self.refresh_models_button.clicked.connect(self.refresh_model_inventory)
        self.install_model_button.clicked.connect(self._download_selected_model)
        self.delete_model_cache_button.clicked.connect(self._delete_selected_model_cache)
        self.open_model_assets_button.clicked.connect(lambda: self._open_path(self.runtime_layout.model_assets_dir))
        self.open_download_cache_button.clicked.connect(lambda: self._open_path(self.app_settings.cache_dir))
        self.model_inventory_table.itemSelectionChanged.connect(self._update_model_action_state)

    def _build_storage_tab(self) -> None:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        form = QFormLayout()
        self.cache_root_label = QLabel(str(self.app_settings.cache_dir))
        self.cache_root_label.setWordWrap(True)
        self.cache_usage_text = QTextEdit()
        self.cache_usage_text.setReadOnly(True)
        self.cache_usage_text.setPlaceholderText("Scanning rebuildable cache usage...")
        form.addRow("Runtime cache location", self.cache_root_label)
        form.addRow("Rebuildable cache usage", self.cache_usage_text)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.refresh_cache_usage_button = QPushButton("Refresh Cache Usage")
        self.clear_cache_button = QPushButton("Clear Rebuildable Caches")
        actions.addWidget(self.refresh_cache_usage_button)
        actions.addWidget(self.clear_cache_button)
        layout.addLayout(actions)
        self.cache_status_label = QLabel(
            "Rebuildable production caches include embeddings, cluster results, indexes, thumbnails, ONNX exports, meanings, and temp files."
        )
        self.cache_status_label.setWordWrap(True)
        layout.addWidget(self.cache_status_label)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Storage")

        self.refresh_cache_usage_button.clicked.connect(self.refresh_cache_usage)
        self.clear_cache_button.clicked.connect(self._clear_rebuildable_caches)

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

        self.operation_journal_table = QTableWidget(0, 6, self)
        self.operation_journal_table.setHorizontalHeaderLabels(["Time", "Operation", "Requested", "Failures", "Cancelled", "Operation Id"])
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

        actions = QHBoxLayout()
        self.refresh_journal_button = QPushButton("Refresh Journal")
        self.open_journal_button = QPushButton("Open Audit Log")
        self.restore_journal_button = QPushButton("Restore Selected Move/Trash")
        actions.addWidget(self.refresh_journal_button)
        actions.addWidget(self.open_journal_button)
        actions.addWidget(self.restore_journal_button)
        layout.addLayout(actions)

        self.journal_status_label = QLabel("")
        self.journal_status_label.setWordWrap(True)
        layout.addWidget(self.journal_status_label)
        self.tabs.addTab(tab, "Safety")

        self.refresh_journal_button.clicked.connect(self.refresh_operation_journal)
        self.open_journal_button.clicked.connect(lambda: self._open_path(self.gallery_action_service.audit_log_path))
        self.restore_journal_button.clicked.connect(self._restore_selected_journal_operation)
        self.operation_journal_table.itemSelectionChanged.connect(self._update_journal_action_state)
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
            "Use Run Runtime Verify, then review the Runtime tab for install and final-exe bundling guidance. "
            "Optional accelerators such as hf_xet and Flash Attention are reported there."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        buttons = QHBoxLayout()
        open_logs = QPushButton("Open Logs")
        open_cache = QPushButton("Open Cache")
        export_bundle = QPushButton("Export Support Bundle")
        view_crash = QPushButton("View Last Crash")
        run_verify = QPushButton("Run Runtime Verify")
        self.run_release_gates_button = QPushButton("Run Release Gates")
        buttons.addWidget(open_logs)
        buttons.addWidget(open_cache)
        buttons.addWidget(export_bundle)
        buttons.addWidget(view_crash)
        buttons.addWidget(run_verify)
        buttons.addWidget(self.run_release_gates_button)
        layout.addLayout(buttons)

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
        self.release_gate_status_label = QLabel("")
        self.release_gate_status_label.setWordWrap(True)
        layout.addWidget(self.release_gate_status_label)
        self.tabs.addTab(tab, "Diagnostics")

        open_logs.clicked.connect(lambda: self._open_path(self.runtime_layout.logs_dir))
        open_cache.clicked.connect(lambda: self._open_path(self.runtime_layout.cache_dir))
        export_bundle.clicked.connect(self._export_support_bundle)
        view_crash.clicked.connect(self._view_last_crash)
        run_verify.clicked.connect(self._verify_gpu)
        self.run_release_gates_button.clicked.connect(self._run_release_gates)
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
        self.execution_mode.setCurrentText(self.settings_store.value("runtime/preferred_mode", self.app_settings.preferred_execution_mode, str))
        self.performance_profile.setCurrentText(self.settings_store.value("performance/profile", self.app_settings.default_performance_profile, str))
        self.thumbnail_size.setValue(int(self.settings_store.value("gallery/thumbnail_size", self.app_settings.thumbnail_size, int)))
        self.gpu_warmup.setChecked(self.settings_store.value("runtime/allow_gpu_warmup", self.app_settings.allow_gpu_warmup, bool))
        self.keep_worker_warm.setChecked(self.settings_store.value("performance/keep_worker_warm", False, bool))
        self.runtime_badge.setChecked(self.settings_store.value("runtime/show_badge", self.app_settings.show_runtime_badge, bool))
        self.dense_ui.setChecked(self.settings_store.value("workspace/dense_ui", self.app_settings.default_dense_ui, bool))
        self.offline_model_downloads.setChecked(
            self.settings_store.value("models/offline_mode", bool(getattr(sys, "frozen", False)), bool)
        )
        self.read_only_mode.setChecked(self.settings_store.value("safety/read_only_mode", False, bool))
        self._apply_profile_preview()

    def values(self) -> dict[str, object]:
        return {
            "runtime/preferred_mode": self.execution_mode.currentText(),
            "performance/profile": self.performance_profile.currentText(),
            "runtime/allow_gpu_warmup": bool(self.gpu_warmup.isChecked()),
            "performance/keep_worker_warm": bool(self.keep_worker_warm.isChecked()),
            "runtime/show_badge": bool(self.runtime_badge.isChecked()),
            "workspace/default_view": "clustering",
            "gallery/thumbnail_size": int(self.thumbnail_size.value()),
            "gallery/thumbnail_workers": int(self.thumbnail_workers.value()),
            "gallery/prefetch_rows": int(self.prefetch_rows.value()),
            "workspace/dense_ui": bool(self.dense_ui.isChecked()),
            "models/offline_mode": bool(self.offline_model_downloads.isChecked()),
            "safety/read_only_mode": bool(self.read_only_mode.isChecked()),
        }

    def refresh_model_inventory(self) -> None:
        items = self.model_asset_service.model_inventory(clustering_model_names(scope="production"))
        self.model_inventory_table.setRowCount(len(items))
        license_lines: list[str] = []
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
            license_lines.append(
                f"{model_label(item.model_name)}\nSource: {item.source_url or item.source_label}\nLicense: {item.license}\n"
            )
        self.model_inventory_table.resizeRowsToContents()
        self.license_text.setPlainText("\n".join(license_lines))
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
        return self._thread_is_running(self._model_thread)

    def _update_model_action_state(self) -> None:
        busy = self._model_actions_busy()
        has_selection = self._selected_model_name() is not None
        self.refresh_models_button.setEnabled(not busy)
        self.install_model_button.setEnabled((not busy) and has_selection)
        self.delete_model_cache_button.setEnabled((not busy) and has_selection)
        self.open_model_assets_button.setEnabled(not busy)
        self.open_download_cache_button.setEnabled(not busy)

    def _download_selected_model(self) -> None:
        model_name = self._selected_model_name()
        if not model_name:
            return
        if not confirmBox(
            "Download selected model?",
            (
                f"This will download or initialize model files for {model_label(model_name)} into:\n"
                f"{self.app_settings.cache_dir}\n\n"
                "Continue?"
            ),
            parent=self,
        ):
            return
        self.model_status_label.setText(f"Downloading or initializing {model_label(model_name)}...")
        self._update_model_action_state()

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, f"Downloading or initializing {model_name}...")
            from infra.performance import select_performance_profile
            from infra.runtime import RuntimeCapabilityService
            from ml.embeddings import ModelManager

            runtime_service = RuntimeCapabilityService()
            execution_policy = runtime_service.select_policy(self.execution_mode.currentText())
            profile = select_performance_profile(self.performance_profile.currentText(), self.system_resources)
            manager = ModelManager(
                use_onnx=model_name in BUNDLED_ONNX_INPUT_SIZES,
                execution_policy=execution_policy,
                runtime_service=runtime_service,
                performance_profile=profile,
                allow_model_downloads=True,
            )
            bundle = manager.get_bundle(model_name, use_onnx=model_name in BUNDLED_ONNX_INPUT_SIZES)
            return {"model": model_name, "signature": bundle.signature}

        job = AsyncJob(_run)
        self._model_job = job

        def _done(result: object) -> None:
            self.model_status_label.setText(f"Model install complete: {result}")
            infoBox("Model installed", f"{model_label(model_name)} is now available in the runtime cache.")
            self._update_model_action_state()
            self.refresh_model_inventory()

        def _failed(message: str) -> None:
            self.model_status_label.setText(f"Model install failed: {message}")
            errorBox("Model install failed", message)
            self._update_model_action_state()
            self.refresh_model_inventory()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="model")
        self._model_thread = thread

    def _delete_selected_model_cache(self) -> None:
        model_name = self._selected_model_name()
        if not model_name:
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
        removed, failures = self.model_asset_service.delete_cached_model(model_name)
        if failures:
            errorBox("Model cache delete completed with errors", "\n".join(failures[:8]))
        else:
            infoBox("Model cache deleted", f"Removed {len(removed)} cached item(s) for {model_label(model_name)}.")
        self.refresh_model_inventory()

    def refresh_operation_journal(self) -> None:
        entries = list(reversed(self.gallery_action_service.read_audit_entries(limit=300)))
        self._journal_entries = entries
        self.operation_journal_table.setRowCount(len(entries))
        for row, entry in enumerate(entries):
            values = [
                str(entry.get("timestamp_utc") or ""),
                str(entry.get("operation") or ""),
                str(entry.get("requested_count") or 0),
                str(entry.get("failure_count") or 0),
                "yes" if entry.get("cancelled") else "no",
                str(entry.get("operation_id") or ""),
            ]
            payload = json.dumps(entry, ensure_ascii=False, sort_keys=True)
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, payload)
                item.setToolTip(payload if column == 5 else value)
                self.operation_journal_table.setItem(row, column, item)
        self.operation_journal_table.resizeRowsToContents()
        if entries:
            self.journal_status_label.setText(
                f"Showing latest {len(entries)} operation journal entries from {self.gallery_action_service.audit_log_path}."
            )
        else:
            self.journal_status_label.setText(f"No file-operation journal entries found at {self.gallery_action_service.audit_log_path}.")
        self._update_journal_action_state()

    def _selected_journal_entry(self) -> dict[str, object] | None:
        row = self.operation_journal_table.currentRow()
        if row < 0:
            return None
        item = self.operation_journal_table.item(row, 0)
        if item is None:
            return None
        raw = item.data(Qt.ItemDataRole.UserRole)
        try:
            payload = json.loads(str(raw or "{}"))
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def _selected_restorable_paths(self) -> list[tuple[str, str]]:
        entry = self._selected_journal_entry()
        if not entry:
            return []
        if str(entry.get("operation") or "") not in {"delete_to_trash", "move"}:
            return []
        changed = entry.get("changed_paths") or []
        pairs: list[tuple[str, str]] = []
        for pair in changed:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                pairs.append((str(pair[0]), str(pair[1])))
        return pairs

    def _update_journal_action_state(self) -> None:
        restorable = bool(self._selected_restorable_paths())
        busy = self._thread_is_running(self._journal_restore_thread)
        self.refresh_journal_button.setEnabled(not busy)
        self.open_journal_button.setEnabled(not busy)
        self.restore_journal_button.setEnabled((not busy) and restorable and not bool(self.read_only_mode.isChecked()))

    def _restore_selected_journal_operation(self) -> None:
        if self.read_only_mode.isChecked():
            infoBox("Read-only mode", "Turn off read-only safety mode before restoring files from the operation journal.")
            self._update_journal_action_state()
            return
        pairs = self._selected_restorable_paths()
        if not pairs:
            self.journal_status_label.setText("Select a move or trash operation with changed paths to restore.")
            return
        if not confirmBox(
            "Restore selected operation?",
            (
                f"This will move {len(pairs)} file(s) back to their original paths when possible.\n\n"
                "Restore is conservative: existing original paths are not overwritten, and missing moved files are reported as failures."
            ),
            parent=self,
        ):
            return
        self.journal_status_label.setText("Restoring selected operation...")
        self._update_journal_action_state()

        job = AsyncJob(lambda progress, cancel_check: self.gallery_action_service.restore_changed_paths(pairs, progress, cancel_check))
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

    def refresh_runtime_diagnostics(self) -> None:
        details = self.runtime_service.diagnostics(self.execution_mode.currentText())
        capabilities = details["capabilities"]
        policy = details["policy"]
        packages = details.get("packages") or {}
        optional_details = details.get("optional_details") or {}
        remediation = list(details.get("remediation") or [])
        profile = select_performance_profile(self.performance_profile.currentText(), self.system_resources)

        text = [
            f"Preferred mode: {policy.preferred_mode}",
            f"Effective mode: {policy.effective_mode}",
            f"Performance profile: {profile.name}",
            f"Warm worker between runs: {'enabled' if self.keep_worker_warm.isChecked() else 'disabled'}",
            "",
            "Packages:",
            f"- torch: {packages.get('torch') or '-'}",
            f"- onnx: {packages.get('onnx') or '-'}",
            f"- onnxruntime: {packages.get('onnxruntime') or '-'}",
            f"- onnxruntime-directml: {packages.get('onnxruntime-directml') or '-'}",
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
            f"Pixmap cache entries: {profile.pixmap_cache_size}",
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
                "- DirectML GPU acceleration applies only on Windows and only to ONNX-enabled production models such as fast_preview and resnet.",
                "- CUDA acceleration requires a CUDA-enabled Torch build plus NVIDIA drivers.",
                "- Keep worker warm is fastest for repeated clustering, but leaves the worker process and loaded model memory resident until disabled or app exit.",
                "- low_memory disables aggressive parallelism/caching; max_speed increases caches, allows all selected backends to run concurrently, and uses a faster, slightly less exact KMeans path on larger folders.",
                "- hf_xet is optional but improves Hugging Face download speed; include it in the packaged runtime if you ship an exe.",
                "- Flash attention is not bundled automatically. If that optimization matters, ship a Torch/CUDA stack that provides it and verify it on the target machine.",
                "- Restart the app after changing GPU/runtime packages.",
            ]
        )
        self.runtime_text.setPlainText("\n".join(text))

    def _apply_profile_preview(self) -> None:
        profile = select_performance_profile(self.performance_profile.currentText(), self.system_resources)
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
        if not allow_clear and not busy:
            self.cache_status_label.setText("Cache clearing is unavailable while clustering or another production background task is running.")

    def refresh_cache_usage(self) -> None:
        if self.describe_rebuildable_caches is None:
            self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            self._update_cache_action_state()
            return
        if self._thread_is_running(self._cache_usage_thread):
            return
        self.cache_status_label.setText("Scanning rebuildable cache usage...")
        self._update_cache_action_state()

        job = AsyncJob(lambda _progress, _cancel_check: self.describe_rebuildable_caches())
        self._cache_usage_job = job

        def _done(result: object) -> None:
            if isinstance(result, CacheUsageSummary):
                lines = [f"Runtime cache root: {result.cache_root}", ""]
                for name, size in result.target_bytes.items():
                    lines.append(f"{name} {self._format_bytes(size)}")
                lines.extend(
                    [
                        "",
                        f"Total rebuildable cache size: {self._format_bytes(result.total_bytes)}",
                        "",
                        "Preserved: image tags, logs, crash records, support bundles, model downloads, and model assets.",
                    ]
                )
                self.cache_usage_text.setPlainText("\n".join(lines))
                self.cache_status_label.setText("Rebuildable cache usage refreshed.")
            else:
                self.cache_usage_text.setPlainText("Cache usage is unavailable.")
            self._update_cache_action_state()

        def _failed(message: str) -> None:
            self.cache_usage_text.setPlainText("Failed to scan cache usage.")
            self.cache_status_label.setText(f"Cache usage refresh failed: {message}")
            self._update_cache_action_state()

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
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

        job = AsyncJob(lambda _progress, _cancel_check: self.clear_rebuildable_caches())
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

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _failed("cancelled"))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="cache_clear")
        self._cache_clear_thread = thread

    def _verify_gpu(self) -> None:
        if self._thread_is_running(self._verify_thread):
            return

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Probing runtimes...")
            return self.runtime_service.verify(self.execution_mode.currentText())

        job = AsyncJob(_run)
        self._verify_job = job

        def _done(result: object) -> None:
            self._last_verify = result if isinstance(result, dict) else None
            self.refresh_runtime_diagnostics()

        job.completed.connect(_done)
        job.failed.connect(lambda message: errorBox("Runtime verify failed", str(message)))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="verify")
        self._verify_thread = thread

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

    def _run_release_gates(self) -> None:
        if self._thread_is_running(self._release_gate_thread):
            return
        report_dir = self.runtime_layout.benchmarks_dir / "release_gates"
        self.release_gate_status_label.setText(f"Running release gates. Report directory: {report_dir}")
        self.run_release_gates_button.setEnabled(False)

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Running release gates...")
            repo_root = Path(__file__).resolve().parents[2]
            report_dir.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "apps.pyqt_production.release_gates",
                    "--report-dir",
                    str(report_dir),
                ],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=600,
            )
            return {
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "report_dir": str(report_dir),
            }

        job = AsyncJob(_run)
        self._release_gate_job = job

        def _done(result: object) -> None:
            payload = dict(result) if isinstance(result, dict) else {}
            code = int(payload.get("returncode", 1))
            report_text = f"Release gates {'passed' if code == 0 else 'failed'} with exit code {code}.\nReport: {payload.get('report_dir')}"
            if payload.get("stdout"):
                report_text += f"\n\nOutput:\n{str(payload.get('stdout'))[-4000:]}"
            if payload.get("stderr"):
                report_text += f"\n\nErrors:\n{str(payload.get('stderr'))[-2000:]}"
            self.release_gate_status_label.setText(report_text)
            self.run_release_gates_button.setEnabled(True)

        def _failed(message: str) -> None:
            self.release_gate_status_label.setText(f"Release gates failed to run: {message}")
            self.run_release_gates_button.setEnabled(True)
            errorBox("Release gates failed", message)

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(lambda: self.release_gate_status_label.setText("Release gates cancelled."))
        job.cancelled.connect(lambda: self.run_release_gates_button.setEnabled(True))
        thread = start_job_in_thread(job)
        self._track_thread(thread, job, role="release_gates")
        self._release_gate_thread = thread

    def _about_text(self) -> str:
        diagnostics = self.runtime_service.diagnostics(self.settings_store.value("runtime/preferred_mode", self.app_settings.preferred_execution_mode, str))
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
        if role == "model" and self._model_thread is thread:
            self._model_thread = None
            if self._model_job is job:
                self._model_job = None
            self._update_model_action_state()
            return
        if role == "journal_restore" and self._journal_restore_thread is thread:
            self._journal_restore_thread = None
            if self._journal_restore_job is job:
                self._journal_restore_job = None
            self._update_journal_action_state()
            return
        if role == "release_gates" and self._release_gate_thread is thread:
            self._release_gate_thread = None
            if self._release_gate_job is job:
                self._release_gate_job = None
            self.run_release_gates_button.setEnabled(True)

    def _on_async_thread_finished(self, thread=None) -> None:
        self._release_finished_thread(thread)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        for job, thread in (
            (self._cache_usage_job, self._cache_usage_thread),
            (self._cache_clear_job, self._cache_clear_thread),
            (self._verify_job, self._verify_thread),
            (self._model_job, self._model_thread),
            (self._journal_restore_job, self._journal_restore_thread),
            (self._release_gate_job, self._release_gate_thread),
        ):
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            if not wait_for_thread_shutdown(thread, timeout_ms=timeout_ms):
                ready_to_close = False
        if ready_to_close:
            self._cache_usage_job = None
            self._cache_usage_thread = None
            self._cache_clear_job = None
            self._cache_clear_thread = None
            self._verify_job = None
            self._verify_thread = None
            self._model_job = None
            self._model_thread = None
            self._journal_restore_job = None
            self._journal_restore_thread = None
            self._release_gate_job = None
            self._release_gate_thread = None
            self._thread_roles = {}
        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        super().closeEvent(event)
