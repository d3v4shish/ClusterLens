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
        self._health_state = "checking"
        self._health_notes: list[str] = []
        self.update_runtime(None, None)

    def update_runtime(self, capabilities: RuntimeCapabilities | None, policy: ExecutionPolicy | None) -> None:
        if capabilities is None or policy is None:
            self._runtime_label = "Runtime checking…"
            self._runtime_tooltip = ["Checking CPU and CUDA runtime availability."]
            self._render()
            return

        label = policy.effective_mode.upper()
        if policy.effective_mode == "cuda" and capabilities.cuda_device_name:
            label = f"CUDA | {capabilities.cuda_device_name}"
        elif policy.effective_mode == "cpu":
            label = "CPU"

        self._runtime_label = label

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
        if policy.preferred_mode == "auto" and policy.effective_mode == "cpu" and policy.reason:
            self._runtime_label = "CPU fallback"
        self._render()

    def set_health(self, state: str, notes: list[str] | tuple[str, ...] | None = None) -> None:
        normalized = str(state or "checking").strip().lower()
        self._health_state = normalized if normalized in {"ok", "warning", "error", "checking"} else "warning"
        self._health_notes = [str(note) for note in (notes or ()) if str(note).strip()]
        self._render()

    def _render(self) -> None:
        suffix = {
            "ok": "Ready",
            "warning": "Attention",
            "error": "Unavailable",
            "checking": "Checking",
        }.get(self._health_state, "Attention")
        self.setText(f"{self._runtime_label} · {suffix}")
        self.setAccessibleDescription(". ".join([*self._runtime_tooltip, *self._health_notes]))
        self.setToolTip("\n".join([*self._runtime_tooltip, *self._health_notes]))
        state = "success" if self._health_state == "ok" else "error" if self._health_state == "error" else "warning"
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)

