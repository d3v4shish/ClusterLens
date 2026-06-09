from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

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
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary
from infra.performance import detect_system_resources, select_performance_profile
from infra.runtime import RuntimeCapabilityService, available_execution_modes
from infra.settings import get_settings
from ui.async_job import AsyncJob, start_job_in_thread
from ui.error_mbox import confirmBox, errorBox, infoBox


class SettingsDialog(QDialog):
    def __init__(
        self,
        settings_store: QSettings,
        runtime_service: RuntimeCapabilityService,
        *,
        describe_rebuildable_caches: Callable[[], CacheUsageSummary] | None = None,
        prepare_rebuildable_cache_clear: Callable[[], None] | None = None,
        clear_rebuildable_caches: Callable[[], CacheClearResult] | None = None,
        can_clear_rebuildable_caches: Callable[[], bool] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.app_settings = get_settings()
        self.settings_store = settings_store
        self.runtime_service = runtime_service
        self.describe_rebuildable_caches = describe_rebuildable_caches
        self.prepare_rebuildable_cache_clear = prepare_rebuildable_cache_clear
        self.clear_rebuildable_caches = clear_rebuildable_caches
        self.can_clear_rebuildable_caches = can_clear_rebuildable_caches or (lambda: True)
        self.system_resources = detect_system_resources()
        self.setWindowTitle("Settings")
        self.resize(860, 600)
        self._verify_job = None
        self._verify_thread = None
        self._cache_usage_job = None
        self._cache_usage_thread = None
        self._cache_clear_job = None
        self._cache_clear_thread = None
        self._thread_roles: dict[object, tuple[str, object | None]] = {}
        self._last_verify: dict[str, object] | None = None
        self._build_ui()
        self._load_values()
        self.refresh_runtime_diagnostics()
        self.refresh_cache_usage()

    def _build_ui(self) -> None:
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
        self.install_directml_button = QPushButton("Install DirectML (Windows GPU)")
        self.install_cuda_button = QPushButton("Install CUDA (NVIDIA)")
        self.verify_gpu_button = QPushButton("Verify GPU")
        if sys.platform != "win32":
            self.install_directml_button.setVisible(False)
            self.install_cuda_button.setVisible(False)
        runtime_actions.addWidget(self.refresh_button)
        runtime_actions.addWidget(self.install_directml_button)
        runtime_actions.addWidget(self.install_cuda_button)
        runtime_actions.addWidget(self.verify_gpu_button)
        runtime_form.addRow(runtime_actions)

        self.runtime_text = QTextEdit()
        self.runtime_text.setReadOnly(True)
        runtime_form.addRow(QLabel("Runtime diagnostics"), self.runtime_text)
        self.tabs.addTab(runtime_tab, "Runtime")

        ui_tab = QWidget(self)
        ui_form = QFormLayout(ui_tab)
        self.default_workspace = QComboBox()
        self.default_workspace.addItems(["clustering"])
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
        ui_form.addRow("Default feature", self.default_workspace)
        ui_form.addRow("Performance profile", self.performance_profile)
        ui_form.addRow("Thumbnail size", self.thumbnail_size)
        ui_form.addRow("Thumbnail workers", self.thumbnail_workers)
        ui_form.addRow("Thumbnail prefetch rows", self.prefetch_rows)
        ui_form.addRow(self.dense_ui)
        self.tabs.addTab(ui_tab, "Workspace")

        storage_tab = QWidget(self)
        storage_layout = QVBoxLayout(storage_tab)
        storage_form = QFormLayout()
        self.cache_root_label = QLabel(str(self.app_settings.cache_dir))
        self.cache_root_label.setWordWrap(True)
        self.cache_usage_text = QTextEdit()
        self.cache_usage_text.setReadOnly(True)
        self.cache_usage_text.setPlaceholderText("Scanning rebuildable cache usage...")
        storage_form.addRow("Runtime cache location", self.cache_root_label)
        storage_form.addRow("Rebuildable cache usage", self.cache_usage_text)
        storage_layout.addLayout(storage_form)
        storage_actions = QHBoxLayout()
        self.refresh_cache_usage_button = QPushButton("Refresh Cache Usage")
        self.clear_cache_button = QPushButton("Clear Rebuildable Caches")
        storage_actions.addWidget(self.refresh_cache_usage_button)
        storage_actions.addWidget(self.clear_cache_button)
        storage_layout.addLayout(storage_actions)
        self.cache_status_label = QLabel(
            "Rebuildable caches include embeddings, clustering results, indexes, thumbnails, ONNX exports, and temp files."
        )
        self.cache_status_label.setWordWrap(True)
        storage_layout.addWidget(self.cache_status_label)
        storage_layout.addStretch(1)
        self.tabs.addTab(storage_tab, "Storage")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.refresh_button.clicked.connect(self.refresh_runtime_diagnostics)
        self.install_directml_button.clicked.connect(lambda: self._launch_installer("enable_gpu_directml.ps1"))
        self.install_cuda_button.clicked.connect(lambda: self._launch_installer("enable_gpu_cuda.ps1"))
        self.verify_gpu_button.clicked.connect(self._verify_gpu)
        self.performance_profile.currentIndexChanged.connect(self._apply_profile_preview)
        self.refresh_cache_usage_button.clicked.connect(self.refresh_cache_usage)
        self.clear_cache_button.clicked.connect(self._clear_rebuildable_caches)

    def _load_values(self) -> None:
        self.execution_mode.setCurrentText(self.settings_store.value("runtime/preferred_mode", self.app_settings.preferred_execution_mode, str))
        self.gpu_warmup.setChecked(self.settings_store.value("runtime/allow_gpu_warmup", self.app_settings.allow_gpu_warmup, bool))
        self.runtime_badge.setChecked(self.settings_store.value("runtime/show_badge", self.app_settings.show_runtime_badge, bool))
        default_feature = self.settings_store.value("workspace/default_view", "clustering", str)
        self.default_workspace.setCurrentText("clustering" if default_feature != "clustering" else default_feature)
        self.performance_profile.setCurrentText(self.settings_store.value("performance/profile", self.app_settings.default_performance_profile, str))
        self.thumbnail_size.setValue(int(self.settings_store.value("gallery/thumbnail_size", self.app_settings.thumbnail_size, int)))
        self.dense_ui.setChecked(self.settings_store.value("workspace/dense_ui", self.app_settings.default_dense_ui, bool))
        self._apply_profile_preview()

    def values(self) -> dict[str, object]:
        return {
            "runtime/preferred_mode": self.execution_mode.currentText(),
            "performance/profile": self.performance_profile.currentText(),
            "runtime/allow_gpu_warmup": bool(self.gpu_warmup.isChecked()),
            "runtime/show_badge": bool(self.runtime_badge.isChecked()),
            "workspace/default_view": self.default_workspace.currentText(),
            "gallery/thumbnail_size": int(self.thumbnail_size.value()),
            "gallery/thumbnail_workers": int(self.thumbnail_workers.value()),
            "gallery/prefetch_rows": int(self.prefetch_rows.value()),
            "workspace/dense_ui": bool(self.dense_ui.isChecked()),
        }

    @staticmethod
    def _format_bytes(size_bytes: int) -> str:
        size = float(max(0, int(size_bytes)))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
            size /= 1024.0
        return f"{size_bytes} B"

    def _cache_actions_busy(self) -> bool:
        return any(job is not None for job in (self._cache_usage_job, self._cache_clear_job))

    def _update_cache_action_state(self) -> None:
        busy = self._cache_actions_busy()
        self.refresh_cache_usage_button.setEnabled(not busy)
        allow_clear = bool(self.clear_rebuildable_caches) and bool(self.can_clear_rebuildable_caches())
        self.clear_cache_button.setEnabled((not busy) and allow_clear)
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
            return self.describe_rebuildable_caches()

        job = AsyncJob(_run)

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
                        "Preserved: image tags, face/search indexes, perceptual hashes, huggingface downloads, and torch downloads.",
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
            "Tags, face/search indexes, perceptual hashes, huggingface downloads, and torch downloads are preserved.",
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
                "- DirectML GPU acceleration applies only to Windows ONNX-enabled torchvision models (fast_preview, convnext, resnet).",
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
            self._on_async_thread_finished,
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

    @pyqtSlot()
    def _on_async_thread_finished(self) -> None:
        self._release_finished_thread(self.sender())

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        for job, thread in [
            (self._cache_usage_job, self._cache_usage_thread),
            (self._cache_clear_job, self._cache_clear_thread),
            (self._verify_job, self._verify_thread),
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
                    thread.quit()
                    ready_to_close = bool(thread.wait(timeout_ms)) and ready_to_close
            except Exception:
                ready_to_close = False
        if ready_to_close:
            self._cache_usage_job = None
            self._cache_usage_thread = None
            self._cache_clear_job = None
            self._cache_clear_thread = None
            self._verify_job = None
            self._verify_thread = None
            self._thread_roles = {}
        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)


