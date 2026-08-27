from __future__ import annotations

from contextlib import nullcontext
import importlib.util
from pathlib import Path

import numpy as np
import torch

from infra.runtime import ExecutionPolicy, RuntimeCapabilityService, preload_onnx_cuda_runtime_libraries
from infra.settings import get_settings

try:
    import onnxruntime as ort
except Exception:
    ort = None


class OnnxModelService:
    def __init__(
        self,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.model_dir = self.settings.cache_dir / "onnx_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(self.settings.preferred_execution_mode)
        self._sessions: dict[tuple[str, str], object] = {}

    def available(self) -> bool:
        return ort is not None

    @staticmethod
    def can_export() -> bool:
        # torch.onnx.export requires the python package 'onnx'.
        return importlib.util.find_spec("onnx") is not None

    def available_providers(self) -> tuple[str, ...]:
        diagnostics = self.runtime_service.detect()
        return diagnostics.onnx_providers

    def export_supported(self, model_name: str) -> bool:
        return model_name in {"resnet", "convnext", "fast_preview"}

    def session_for(
        self,
        model_name: str,
        model: torch.nn.Module,
        input_size: tuple[int, int],
        provider: str | None = None,
    ):
        if not self.available() or not self.export_supported(model_name):
            return None
        if not self.can_export():
            return None

        selected_provider = self._select_provider(provider)
        cache_key = (model_name, selected_provider)
        if cache_key in self._sessions:
            return self._sessions[cache_key]

        onnx_path = self.model_dir / f"{model_name}_{input_size[0]}x{input_size[1]}.onnx"
        if not onnx_path.exists():
            try:
                self._export(model, input_size, onnx_path, selected_provider)
            except Exception:
                return None

        try:
            self._sessions[cache_key] = self._create_session_with_fallback(
                onnx_path,
                selected_provider,
                allow_cpu_fallback=self.execution_policy.preferred_mode == "auto",
            )
        except Exception:
            return None
        return self._sessions[cache_key]

    def session_from_asset(self, model_name: str, onnx_path: Path, provider: str | None = None):
        if not self.available() or not Path(onnx_path).exists():
            return None
        selected_provider = self._select_provider(provider)
        cache_key = (f"asset:{model_name}:{Path(onnx_path).resolve()}", selected_provider)
        if cache_key in self._sessions:
            return self._sessions[cache_key]
        try:
            self._sessions[cache_key] = self._create_session_with_fallback(
                onnx_path,
                selected_provider,
                allow_cpu_fallback=self.execution_policy.preferred_mode == "auto",
            )
        except Exception:
            return None
        return self._sessions[cache_key]

    def run(self, session, batch_tensor: torch.Tensor) -> torch.Tensor:
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: batch_tensor.detach().cpu().numpy()})
        return torch.from_numpy(np.asarray(outputs[0]))

    def _select_provider(self, preferred_provider: str | None = None) -> str:
        capabilities = self.runtime_service.detect()
        providers = set(capabilities.onnx_providers)
        preferred = str(preferred_provider or self.execution_policy.onnx_provider or "CPUExecutionProvider")
        if self.execution_policy.preferred_mode == "cuda" and preferred != "CUDAExecutionProvider":
            raise RuntimeError(
                self.execution_policy.error
                or "CUDA was explicitly requested, but the ONNX CUDA provider is unavailable."
            )
        if preferred == "CUDAExecutionProvider":
            verified_policy = self.runtime_service.select_policy(self.execution_policy.preferred_mode)
            if verified_policy.onnx_provider != "CUDAExecutionProvider":
                if self.execution_policy.preferred_mode == "cuda":
                    raise RuntimeError(
                        verified_policy.error
                        or "CUDA was explicitly requested, but the ONNX CUDA provider failed verification."
                    )
                return "CPUExecutionProvider"
        if preferred in providers:
            return preferred
        if self.execution_policy.uses_cuda and "CUDAExecutionProvider" in providers:
            return "CUDAExecutionProvider"
        return "CPUExecutionProvider"

    @staticmethod
    def _create_session_with_fallback(onnx_path: Path, provider: str, *, allow_cpu_fallback: bool = True):
        selected = str(provider or "CPUExecutionProvider")
        ordered = [selected, "CPUExecutionProvider"] if selected != "CPUExecutionProvider" else ["CPUExecutionProvider"]
        if selected == "CUDAExecutionProvider":
            preload_onnx_cuda_runtime_libraries()
        try:
            return ort.InferenceSession(str(onnx_path), providers=ordered)
        except Exception:
            if selected == "CPUExecutionProvider" or not allow_cpu_fallback:
                raise
            return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    @staticmethod
    def _export(model: torch.nn.Module, input_size: tuple[int, int], onnx_path: Path, provider: str) -> None:
        device = torch.device("cuda" if provider == "CUDAExecutionProvider" and torch.cuda.is_available() else "cpu")
        sample = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        model = model.to(device).eval()
        with torch.inference_mode(), (
            torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        ):
            torch.onnx.export(
                model,
                sample,
                str(onnx_path),
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
                opset_version=17,
            )

