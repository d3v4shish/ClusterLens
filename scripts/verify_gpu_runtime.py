from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from infra.runtime import RuntimeCapabilityService


def main() -> int:
    result = RuntimeCapabilityService().verify("cuda")
    capabilities = result["capabilities"]
    policy = result["policy"]
    torch_smoke = result.get("torch_smoke") or {}
    onnx_smoke = result.get("onnx_smoke") or {}
    failures: list[str] = []

    if not capabilities.torch_cuda_available:
        failures.append("Torch cannot use CUDA.")
    if policy.torch_device != "cuda":
        failures.append(f"Torch policy selected {policy.torch_device}, not CUDA.")
    if policy.onnx_provider != "CUDAExecutionProvider":
        failures.append(f"ONNX policy selected {policy.onnx_provider}, not CUDAExecutionProvider.")
    if not torch_smoke.get("ok"):
        failures.append(str(torch_smoke.get("error") or "Torch CUDA smoke test failed."))
    if not onnx_smoke.get("ok") or onnx_smoke.get("provider") != "CUDAExecutionProvider":
        failures.append(str(onnx_smoke.get("error") or "ONNX CUDA smoke test failed."))

    print(
        json.dumps(
            {
                "capabilities": asdict(capabilities),
                "policy": asdict(policy),
                "packages": result.get("packages") or {},
                "torch_smoke": torch_smoke,
                "onnx_smoke": onnx_smoke,
                "failures": failures,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
