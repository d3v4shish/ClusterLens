from __future__ import annotations

import base64
import ctypes
from dataclasses import dataclass, field
import importlib
import importlib.util
import os
from pathlib import Path
import site
import sys
import time
import warnings

try:
    from importlib import metadata as _importlib_metadata
except Exception:  # pragma: no cover
    _importlib_metadata = None

_ONNX_CUDA_LIBS_PRELOADED = False
_ONNX_PROVIDER_PROBE_CACHE: dict[str, bool] = {}
_MODULE_UNSET = object()
_TORCH_MODULE: object = _MODULE_UNSET
_ONNX_RUNTIME_MODULE: object = _MODULE_UNSET
_ONNX_PROVIDER_PROBE_MODEL = base64.b64decode(
    "CAg6VQoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRIFcHJvYmVaFwoFaW5wdXQSDgoMCAESCAoCCAEKAggBYhgKBm91dHB1dBIOCgwIARIICgIIAQoCCAFCBAoAEBE="
)


def _load_optional_module(name: str, cache_name: str):
    cached = globals()[cache_name]
    if cached is not _MODULE_UNSET:
        return cached
    try:
        cached = importlib.import_module(name)
    except Exception:
        cached = None
    globals()[cache_name] = cached
    return cached


def _torch_module():
    return _load_optional_module("torch", "_TORCH_MODULE")


def _onnx_runtime_module():
    return _load_optional_module("onnxruntime", "_ONNX_RUNTIME_MODULE")


def _package_version(name: str) -> str:
    if _importlib_metadata is None:
        return ""
    try:
        return str(_importlib_metadata.version(name) or "")
    except Exception:
        return ""


def _first_package_version(*names: str) -> str:
    for name in names:
        version = _package_version(name)
        if version:
            return version
    return ""


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def preload_onnx_cuda_runtime_libraries() -> bool:
    global _ONNX_CUDA_LIBS_PRELOADED
    if _ONNX_CUDA_LIBS_PRELOADED:
        return True

    loaded_any = False
    ort = _onnx_runtime_module()
    preload_dlls = getattr(ort, "preload_dlls", None) if ort is not None else None
    if callable(preload_dlls):
        try:
            preload_dlls(directory="")
            _ONNX_CUDA_LIBS_PRELOADED = True
            return True
        except TypeError:
            try:
                preload_dlls()
                _ONNX_CUDA_LIBS_PRELOADED = True
                return True
            except Exception:
                pass
        except Exception:
            pass

    if os.name != "posix":
        return False

    seen: set[str] = set()
    for site_root in site.getsitepackages():
        nvidia_root = Path(site_root) / "nvidia"
        if not nvidia_root.exists():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            for lib_path in sorted(lib_dir.glob("*.so*")):
                resolved = str(lib_path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                try:
                    ctypes.CDLL(resolved, mode=ctypes.RTLD_GLOBAL)
                    loaded_any = True
                except OSError:
                    continue

    _ONNX_CUDA_LIBS_PRELOADED = loaded_any
    return loaded_any


def available_execution_modes() -> tuple[str, ...]:
    return ("auto", "cuda", "cpu")


def _flash_sdp_enabled() -> bool:
    torch = _torch_module()
    if torch is None:
        return False
    backend = getattr(getattr(torch, "backends", None), "cuda", None)
    probe = getattr(backend, "flash_sdp_enabled", None)
    if callable(probe):
        try:
            return bool(probe())
        except Exception:
            return False
    return False


def _optional_runtime_details() -> dict[str, object]:
    hf_xet_version = _first_package_version("hf_xet", "hf-xet")
    flash_attn_version = _first_package_version("flash-attn", "flash_attn")
    return {
        "hf_xet_installed": bool(hf_xet_version or _module_available("hf_xet")),
        "hf_xet_version": hf_xet_version,
        "flash_attn_version": flash_attn_version,
        "flash_sdp_enabled": _flash_sdp_enabled(),
    }


def _cpu_runtime_details() -> dict[str, object]:
    details: dict[str, object] = {
        "numpy_simd": (),
        "blas": "",
        "blas_threads": 0,
        "libjpeg_turbo": False,
    }
    try:
        import numpy as np

        feature_map = getattr(getattr(np, "core", None), "_multiarray_umath", None)
        feature_map = getattr(feature_map, "__cpu_features__", {})
        priority = ("AVX512_ICL", "AVX512_SKX", "AVX512F", "AVX2", "FMA3", "AVX", "SSE42", "SSE2")
        details["numpy_simd"] = tuple(name for name in priority if bool(feature_map.get(name)))
    except Exception:
        pass
    try:
        from threadpoolctl import threadpool_info

        blas_runtime = next(
            (item for item in threadpool_info() if str(item.get("user_api", "")) == "blas"),
            {},
        )
        details["blas"] = str(
            blas_runtime.get("internal_api")
            or blas_runtime.get("prefix")
            or ""
        )
        details["blas_threads"] = int(blas_runtime.get("num_threads") or 0)
    except Exception:
        pass
    try:
        from PIL import features

        details["libjpeg_turbo"] = bool(features.check_feature("libjpeg_turbo"))
    except Exception:
        pass
    return details


def _build_remediation(
    *,
    capabilities: "RuntimeCapabilities",
    packages: dict[str, str],
    optional_details: dict[str, object],
) -> list[str]:
    remediation: list[str] = []
    if not capabilities.torch_cuda_available:
        if capabilities.torch_cuda_build:
            remediation.append("Torch includes CUDA support, but no CUDA device is available.")
        else:
            remediation.append("Installed Torch build is CPU-only (CUDA requires CUDA Torch wheels).")
    if not packages.get("onnx"):
        remediation.append("Install onnx (python package) to export models for ONNX Runtime CUDA/CPU inference.")
    if not capabilities.has_onnx_cuda:
        remediation.append("Install an ONNX Runtime GPU build (onnxruntime-gpu) for CUDA ONNX inference.")
    if capabilities.torch_cuda_available and not packages.get("cuml"):
        remediation.append("Install the pinned RAPIDS cuML CUDA runtime to accelerate HDBSCAN; native HDBSCAN remains the CPU fallback.")
    if not bool(optional_details.get("hf_xet_installed")):
        remediation.append(
            "Install hf_xet (or huggingface_hub[hf_xet]) for faster Hugging Face downloads, and include it in the packaged exe if you ship one."
        )
    if capabilities.torch_cuda_available and not bool(optional_details.get("flash_sdp_enabled")):
        remediation.append(
            "Current Torch runtime does not expose Flash SDP. If you want flash-attention speedups, install or bundle a Torch/CUDA runtime with flash-attention support and confirm it with Run Runtime Verify."
        )
    return remediation


@dataclass(frozen=True)
class RuntimeCapabilities:
    torch_version: str = ""
    torch_cuda_build: bool = False
    torch_cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str = ""
    cuda_total_memory_mb: int = 0
    onnx_available: bool = False
    onnx_version: str = ""
    onnx_providers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_onnx_cuda(self) -> bool:
        return "CUDAExecutionProvider" in self.onnx_providers

    @property
    def has_onnx_cpu(self) -> bool:
        return "CPUExecutionProvider" in self.onnx_providers


@dataclass(frozen=True)
class RuntimeSelection:
    requested_backend: str = "auto"
    actual_backend: str = "cpu"
    cuda_device: str = ""
    precision: str = "fp32"
    model_artifact: str = ""
    torch_device: str = "cpu"
    onnx_provider: str = "CPUExecutionProvider"
    fallback_reason: str = ""
    error: str = ""


@dataclass(frozen=True)
class ExecutionPolicy:
    preferred_mode: str = "auto"
    effective_mode: str = "cpu"
    torch_device: str = "cpu"
    onnx_provider: str = "CPUExecutionProvider"
    reason: str = ""
    fallback_reason: str = ""
    error: str = ""
    precision: str = "fp32"
    model_artifact: str = ""

    @property
    def uses_cuda(self) -> bool:
        return self.effective_mode == "cuda" and self.torch_device == "cuda"

    @property
    def uses_onnx_cuda(self) -> bool:
        return self.onnx_provider == "CUDAExecutionProvider"

    @property
    def cuda_required_unavailable(self) -> bool:
        return self.preferred_mode == "cuda" and self.effective_mode != "cuda"

    @property
    def runtime_selection(self) -> RuntimeSelection:
        return RuntimeSelection(
            requested_backend=self.preferred_mode,
            actual_backend=self.effective_mode,
            cuda_device="cuda:0" if self.effective_mode == "cuda" else "",
            precision=self.precision,
            model_artifact=self.model_artifact,
            torch_device=self.torch_device,
            onnx_provider=self.onnx_provider,
            fallback_reason=self.fallback_reason,
            error=self.error,
        )


class RuntimeCapabilityService:
    def __init__(self) -> None:
        self._cached: RuntimeCapabilities | None = None
        self._provider_probe_cache = _ONNX_PROVIDER_PROBE_CACHE

    def detect(self, refresh: bool = False) -> RuntimeCapabilities:
        if self._cached is not None and not refresh:
            return self._cached

        torch = _torch_module()
        ort = _onnx_runtime_module()
        torch_version = str(getattr(torch, "__version__", "")) if torch is not None else ""
        cuda_build = bool(getattr(getattr(torch, "version", None), "cuda", None)) if torch is not None else False
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if torch is not None and cuda_available else 0
        cuda_device_name = ""
        cuda_total_memory_mb = 0
        if cuda_available and cuda_device_count > 0:
            try:
                cuda_device_name = str(torch.cuda.get_device_name(0))
            except Exception:
                cuda_device_name = ""
            try:
                props = torch.cuda.get_device_properties(0)
                cuda_total_memory_mb = int(getattr(props, "total_memory", 0) // (1024 * 1024))
            except Exception:
                cuda_total_memory_mb = 0

        onnx_available = ort is not None
        onnx_version = str(getattr(ort, "__version__", "")) if onnx_available else ""
        onnx_providers: tuple[str, ...] = ()
        if onnx_available:
            try:
                onnx_providers = tuple(str(provider) for provider in ort.get_available_providers())
            except Exception:
                onnx_providers = ()

        self._cached = RuntimeCapabilities(
            torch_version=torch_version,
            torch_cuda_build=cuda_build,
            torch_cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            cuda_device_name=cuda_device_name,
            cuda_total_memory_mb=cuda_total_memory_mb,
            onnx_available=onnx_available,
            onnx_version=onnx_version,
            onnx_providers=onnx_providers,
        )
        return self._cached

    def select_policy(self, preferred_mode: str = "auto", refresh: bool = False) -> ExecutionPolicy:
        capabilities = self.detect(refresh=refresh)
        preferred = str(preferred_mode or "auto").strip().lower()
        if preferred not in {"auto", "cuda", "cpu"}:
            preferred = "auto"

        # A CPU-only request must not initialize or probe a CUDA provider. In
        # addition to wasting startup time, probing can print provider errors
        # and briefly reserve GPU runtime resources in download/maintenance
        # workers that deliberately requested CPU execution.
        if preferred == "cpu":
            return ExecutionPolicy(
                preferred_mode=preferred,
                effective_mode="cpu",
                torch_device="cpu",
                onnx_provider="CPUExecutionProvider",
                reason="Using CPU runtime by preference.",
            )

        cuda_onnx_usable = capabilities.has_onnx_cuda and self._onnx_provider_usable(
            "CUDAExecutionProvider",
            refresh=refresh,
        )

        if preferred in {"auto", "cuda"} and capabilities.torch_cuda_available:
            provider = "CUDAExecutionProvider" if cuda_onnx_usable else "CPUExecutionProvider"
            reason = f"Using CUDA Torch on {capabilities.cuda_device_name or 'GPU'}."
            fallback_reason = ""
            if capabilities.has_onnx_cuda and provider != "CUDAExecutionProvider":
                fallback_reason = (
                    "ONNX CUDA provider failed verification; GPU model inference uses CUDA Torch "
                    "instead of CPU ONNX."
                )
                reason += f" {fallback_reason}"
            elif provider != "CUDAExecutionProvider":
                fallback_reason = (
                    "ONNX CUDA provider is unavailable; GPU model inference uses CUDA Torch "
                    "instead of CPU ONNX."
                )
                reason += f" {fallback_reason}"
            return ExecutionPolicy(
                preferred_mode=preferred,
                effective_mode="cuda",
                torch_device="cuda",
                onnx_provider=provider,
                reason=reason,
                fallback_reason=fallback_reason,
            )

        if preferred in {"auto", "cuda"} and cuda_onnx_usable:
            return ExecutionPolicy(
                preferred_mode=preferred,
                effective_mode="cuda",
                torch_device="cpu",
                onnx_provider="CUDAExecutionProvider",
                reason="Using CUDA ONNX Runtime. Torch workloads remain on CPU.",
            )

        if preferred == "cuda":
            return self._cpu_fallback(
                capabilities,
                preferred,
                "CUDA requested but neither CUDA Torch nor a verified CUDA ONNX Runtime is available.",
                error=(
                    "CUDA was explicitly requested, but no compatible CUDA runtime is available. "
                    "Install NVIDIA drivers plus CUDA Torch or onnxruntime-gpu, then rerun Runtime Verify."
                ),
            )

        return self._cpu_fallback(capabilities, preferred, "")

    def _onnx_provider_usable(self, provider: str, *, refresh: bool = False) -> bool:
        normalized = str(provider or "").strip()
        if not normalized or normalized == "CPUExecutionProvider":
            return True
        capabilities = self.detect(refresh=refresh)
        ort = _onnx_runtime_module()
        if ort is None or normalized not in capabilities.onnx_providers:
            return False
        cache_key = normalized.casefold()
        if not refresh and cache_key in self._provider_probe_cache:
            return self._provider_probe_cache[cache_key]
        try:
            if normalized == "CUDAExecutionProvider":
                preload_onnx_cuda_runtime_libraries()
            session = ort.InferenceSession(
                _ONNX_PROVIDER_PROBE_MODEL,
                providers=[normalized, "CPUExecutionProvider"],
            )
            providers = tuple(str(item) for item in session.get_providers())
            usable = normalized in providers and (providers[0] == normalized or len(providers) == 1)
        except Exception:
            usable = False
        self._provider_probe_cache[cache_key] = usable
        return usable

    def diagnostics(self, preferred_mode: str = "auto", *, refresh: bool = False) -> dict[str, object]:
        capabilities = self.detect(refresh=refresh)
        policy = self.select_policy(preferred_mode=preferred_mode, refresh=refresh)
        packages = {
            "torch": _package_version("torch") or capabilities.torch_version,
            "onnxruntime": _package_version("onnxruntime") or capabilities.onnx_version,
            "onnx": _package_version("onnx"),
            "onnxruntime-gpu": _package_version("onnxruntime-gpu"),
            "cuml": _first_package_version("cuml-cu12", "cuml"),
            "cupy": _first_package_version("cupy-cuda12x", "cupy"),
            "hf_xet": _first_package_version("hf_xet", "hf-xet"),
            "flash-attn": _first_package_version("flash-attn", "flash_attn"),
        }
        optional_details = _optional_runtime_details()
        cpu_details = _cpu_runtime_details()
        remediation = _build_remediation(
            capabilities=capabilities,
            packages=packages,
            optional_details=optional_details,
        )

        return {
            "capabilities": capabilities,
            "policy": policy,
            "packages": packages,
            "optional_details": optional_details,
            "cpu_details": cpu_details,
            "remediation": remediation,
        }

    def verify(self, preferred_mode: str = "auto") -> dict[str, object]:
        """Re-probe capabilities and run quick non-IO smoke tests where possible."""
        capabilities = self.detect(refresh=True)
        policy = self.select_policy(preferred_mode=preferred_mode, refresh=True)
        packages = {
            "torch": _package_version("torch") or capabilities.torch_version,
            "onnxruntime": _package_version("onnxruntime") or capabilities.onnx_version,
            "onnx": _package_version("onnx"),
            "onnxruntime-gpu": _package_version("onnxruntime-gpu"),
            "cuml": _first_package_version("cuml-cu12", "cuml"),
            "cupy": _first_package_version("cupy-cuda12x", "cupy"),
            "hf_xet": _first_package_version("hf_xet", "hf-xet"),
            "flash-attn": _first_package_version("flash-attn", "flash_attn"),
        }
        optional_details = _optional_runtime_details()
        cpu_details = _cpu_runtime_details()

        torch = _torch_module()
        ort = _onnx_runtime_module()
        torch_smoke: dict[str, object] | None = None
        if torch is not None and policy.effective_mode == "cuda" and capabilities.torch_cuda_available:
            try:
                t0 = time.perf_counter()
                x = torch.randn(1024, 1024, device="cuda")
                y = x @ x
                _ = float(y[0, 0].item())
                torch.cuda.synchronize()
                torch_smoke = {"ok": True, "ms": round((time.perf_counter() - t0) * 1000.0, 2)}
            except Exception as exc:
                torch_smoke = {"ok": False, "error": str(exc)}

        onnx_smoke: dict[str, object] | None = None
        if ort is not None and packages.get("onnx"):
            try:
                import tempfile
                from pathlib import Path
                import torch.nn as nn

                class _Smoke(nn.Module):
                    def forward(self, x):
                        return x + 1.0

                model = _Smoke().eval()
                tmp = Path(tempfile.gettempdir()) / "imageclustering_smoke.onnx"
                sample = torch.randn(1, 3, 16, 16)
                t0 = time.perf_counter()
                torch.onnx.export(model, sample, str(tmp), input_names=["input"], output_names=["output"], opset_version=17)
                if policy.onnx_provider == "CUDAExecutionProvider":
                    preload_onnx_cuda_runtime_libraries()
                providers = [policy.onnx_provider, "CPUExecutionProvider"]
                session = ort.InferenceSession(str(tmp), providers=providers)
                input_name = session.get_inputs()[0].name
                _ = session.run(None, {input_name: sample.numpy()})
                onnx_smoke = {"ok": True, "provider": session.get_providers()[0], "ms": round((time.perf_counter() - t0) * 1000.0, 2)}
            except Exception as exc:
                onnx_smoke = {"ok": False, "error": str(exc)}

        flash_attention_smoke: dict[str, object] | None = None
        if torch is not None and policy.effective_mode == "cuda" and capabilities.torch_cuda_available:
            try:
                q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.float16)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    _ = torch.nn.functional.scaled_dot_product_attention(q, q, q)
                    torch.cuda.synchronize()
                flash_warning = ""
                for item in caught:
                    message = str(getattr(item, "message", "") or "")
                    if "flash attention" in message.lower():
                        flash_warning = message
                        break
                if flash_warning:
                    flash_attention_smoke = {"ok": False, "error": flash_warning}
                else:
                    flash_attention_smoke = {"ok": True}
            except Exception as exc:
                flash_attention_smoke = {"ok": False, "error": str(exc)}

        remediation = _build_remediation(
            capabilities=capabilities,
            packages=packages,
            optional_details=optional_details,
        )
        if isinstance(flash_attention_smoke, dict) and not flash_attention_smoke.get("ok"):
            remediation.append(
                "Runtime Verify could not confirm Flash Attention support. Install or bundle a matching Torch/CUDA/flash-attention stack for the final exe if this optimization matters."
            )

        return {
            "capabilities": capabilities,
            "policy": policy,
            "packages": packages,
            "optional_details": optional_details,
            "cpu_details": cpu_details,
            "torch_smoke": torch_smoke,
            "onnx_smoke": onnx_smoke,
            "flash_attention_smoke": flash_attention_smoke,
            "remediation": remediation,
        }

    @staticmethod
    def _cpu_fallback(capabilities: RuntimeCapabilities, preferred: str, prefix: str, *, error: str = "") -> ExecutionPolicy:
        parts: list[str] = []
        if prefix:
            parts.append(prefix)
        if not capabilities.torch_cuda_build:
            parts.append(
                f"Current Torch runtime is {capabilities.torch_version or 'unknown'} and does not include CUDA support."
            )
        elif not capabilities.torch_cuda_available:
            parts.append("Torch CUDA runtime is installed, but no CUDA device is usable.")
        if not capabilities.onnx_available:
            parts.append("ONNX Runtime is not installed.")
        elif capabilities.onnx_providers:
            parts.append(f"Available ONNX providers: {', '.join(capabilities.onnx_providers)}.")
        return ExecutionPolicy(
            preferred_mode=preferred or "auto",
            effective_mode="cpu",
            torch_device="cpu",
            onnx_provider="CPUExecutionProvider",
            reason=" ".join(parts).strip() or "Using CPU runtime.",
            fallback_reason=" ".join(parts).strip() or "Using CPU runtime.",
            error=error,
        )
