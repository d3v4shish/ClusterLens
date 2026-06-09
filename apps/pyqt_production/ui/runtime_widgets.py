from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel

from infra.runtime import ExecutionPolicy, RuntimeCapabilities


class RuntimeBadge(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(120)
        self.setContentsMargins(8, 2, 8, 2)
        self.update_runtime(None, None)

    def update_runtime(self, capabilities: RuntimeCapabilities | None, policy: ExecutionPolicy | None) -> None:
        if capabilities is None or policy is None:
            self.setText("Runtime")
            self.setToolTip("")
            self.setStyleSheet("padding: 4px 10px; border-radius: 10px; background: #E9ECEF; color: #333;")
            return

        label = policy.effective_mode.upper()
        if policy.effective_mode == "cuda" and capabilities.cuda_device_name:
            label = f"CUDA | {capabilities.cuda_device_name}"
        elif policy.effective_mode == "directml":
            label = "DIRECTML"
        elif policy.effective_mode == "cpu":
            label = "CPU"

        self.setText(label)

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

        if policy.effective_mode == "directml":
            tooltip.append("DirectML GPU applies only to ONNX-enabled production models (fast_preview/resnet).")
            tooltip.append("ONNX export requires the python package 'onnx'.")

        if (
            sys.platform == "win32"
            and policy.effective_mode == "cpu"
            and capabilities.onnx_providers
            and "DmlExecutionProvider" not in capabilities.onnx_providers
        ):
            tooltip.append("For Windows GPU without CUDA: install onnxruntime-directml and enable ONNX for supported models.")

        self.setToolTip("\n".join(item for item in tooltip if item))

        if policy.effective_mode == "cuda":
            color = "#0B6E4F"
        elif policy.effective_mode == "directml":
            color = "#1F6FEB"
        else:
            color = "#6C757D"

        self.setStyleSheet(
            f"padding: 4px 10px; border-radius: 10px; background: {color}; color: white; font-weight: 600;"
        )

