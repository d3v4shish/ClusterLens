from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton

from infra.runtime import ExecutionPolicy, RuntimeCapabilities


class RuntimeBadge(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAccessibleName("Runtime and application health")
        self.setProperty("kind", "quiet")
        self._runtime_label = "Runtime"
        self._runtime_tooltip: list[str] = []
        self._runtime_state = "checking"
        self._cuda_ready = False
        self._health_state = "checking"
        self._health_notes: list[str] = []
        self._compact = False
        self.update_runtime(None, None)

    def set_compact(self, compact: bool) -> None:
        normalized = bool(compact)
        if normalized == self._compact:
            return
        self._compact = normalized
        self._render()

    def update_runtime(self, capabilities: RuntimeCapabilities | None, policy: ExecutionPolicy | None) -> None:
        if capabilities is None or policy is None:
            self._runtime_label = "Runtime checking…"
            self._runtime_tooltip = ["Checking CPU and CUDA runtime availability."]
            self._runtime_state = "checking"
            self._cuda_ready = False
            self._render()
            return

        label = policy.effective_mode.upper()
        if policy.effective_mode == "cuda" and capabilities.cuda_device_name:
            label = f"CUDA | {capabilities.cuda_device_name}"
        elif policy.effective_mode == "cpu":
            label = "CPU"

        self._runtime_label = label
        self._cuda_ready = policy.effective_mode == "cuda"

        tooltip = [
            f"Preferred: {policy.preferred_mode}",
            f"Effective: {policy.effective_mode}",
            f"Torch: {capabilities.torch_version}",
            f"ONNX: {capabilities.onnx_version or '-'}",
            f"ONNX provider: {policy.onnx_provider}",
            policy.reason,
        ]

        if not capabilities.torch_cuda_build and policy.preferred_mode in {"auto", "cuda"}:
            tooltip.append("CUDA requires a CUDA-enabled Torch build (current Torch is CPU-only).")

        if policy.cuda_required_unavailable and policy.error:
            tooltip.append(policy.error)

        self._runtime_tooltip = [item for item in tooltip if item]
        if policy.cuda_required_unavailable:
            self._runtime_label = "CUDA unavailable"
            self._runtime_state = "unavailable"
        elif policy.preferred_mode == "auto" and policy.effective_mode == "cpu" and policy.reason:
            self._runtime_label = "CPU fallback"
            self._runtime_state = "fallback"
        else:
            self._runtime_state = "ready"
        self._render()

    def set_runtime_failure(self, message: str) -> None:
        """Show a completed local-readiness failure with a concrete retry path."""

        detail = str(message or "Runtime readiness check failed.").strip()
        self._runtime_label = "Runtime failed"
        self._runtime_state = "failed"
        self._cuda_ready = False
        self._runtime_tooltip = [
            detail,
            "Open Settings and choose Rescan GPU Resources to retry the local readiness check.",
        ]
        self._render()

    def set_health(self, state: str, notes: list[str] | tuple[str, ...] | None = None) -> None:
        normalized = str(state or "checking").strip().lower()
        self._health_state = normalized if normalized in {"ok", "warning", "error", "checking"} else "warning"
        self._health_notes = [str(note) for note in (notes or ()) if str(note).strip()]
        self._render()

    def _render(self) -> None:
        suffix = {
            "checking": "Checking",
            "ready": "Ready",
            "fallback": "Fallback",
            "unavailable": "Unavailable",
            "failed": "Failed",
        }.get(self._runtime_state, "Checking")
        compact_label = {
            "checking": "Checking…",
            "ready": "CUDA ready" if self._cuda_ready else "CPU ready",
            "fallback": "CPU fallback",
            "unavailable": "Unavailable",
            "failed": "Failed",
        }.get(self._runtime_state, suffix)
        self.setText(compact_label if self._compact else f"{self._runtime_label} · {suffix}")
        self.setAccessibleName(f"Runtime status: {self._runtime_label}; {suffix}")
        tooltip = list(self._runtime_tooltip)
        if self._cuda_ready and self._health_state == "warning" and self._runtime_state == "ready":
            tooltip.append("CUDA is ready. The application-health notes below do not affect GPU availability.")
        tooltip.extend(self._health_notes)
        self.setAccessibleDescription(". ".join(tooltip))
        self.setToolTip("\n".join(tooltip))
        state = {
            "checking": "warning",
            "ready": "success",
            "fallback": "warning",
            "unavailable": "error",
            "failed": "error",
        }.get(self._runtime_state, "warning")
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)

