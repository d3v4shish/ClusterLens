from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apps.pyqt_production.identity import PRODUCTION_APP_ID
from apps.shared.runtime_support import activate_runtime_root, configure_rotating_logging, install_crash_handlers
from app.services.model_assets import (
    MODEL_LICENSES,
    MODEL_MANIFEST_VERSION,
    MODEL_SOURCE_LABELS,
    MODEL_SOURCE_URLS,
    ModelAssetManifestV2,
    sha256_file,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export normalized ONNX bundles for production clustering models.")
    parser.add_argument("--models", default="resnet,convnext,fast_preview")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "build" / "model_assets"))
    parser.add_argument("--preferred-execution-mode", default="cpu")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    runtime_layout = activate_runtime_root(PRODUCTION_APP_ID)
    configure_rotating_logging(runtime_layout)
    install_crash_handlers(runtime_layout)

    from app.services.onnx_models import OnnxModelService
    from infra.performance import select_performance_profile
    from infra.runtime import RuntimeCapabilityService
    from ml.embeddings import ModelManager

    runtime_service = RuntimeCapabilityService()
    execution_policy = runtime_service.select_policy(args.preferred_execution_mode)
    performance_profile = select_performance_profile("balanced")
    model_manager = ModelManager(
        use_onnx=True,
        execution_policy=execution_policy,
        runtime_service=runtime_service,
        performance_profile=performance_profile,
    )
    onnx_service = OnnxModelService(execution_policy=execution_policy, runtime_service=runtime_service)

    requested_models = [item.strip() for item in str(args.models).split(",") if item.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "manifest_version": MODEL_MANIFEST_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(runtime_layout.root),
        "preferred_execution_mode": args.preferred_execution_mode,
        "bundles": [],
        "failures": [],
    }

    for model_name in requested_models:
        if not onnx_service.export_supported(model_name):
            manifest["failures"].append({"model": model_name, "reason": "onnx_export_not_supported"})
            continue
        try:
            bundle = model_manager.get_bundle(model_name, use_onnx=True)
            if bundle.onnx_session is None:
                raise RuntimeError("ONNX session creation failed")
            source_path = onnx_service.model_dir / f"{model_name}_{bundle.input_size[0]}x{bundle.input_size[1]}.onnx"
            if not source_path.exists():
                asset_bundle = model_manager.model_asset_service.find_bundle(model_name)
                if asset_bundle is not None and tuple(asset_bundle.input_size) == tuple(bundle.input_size):
                    source_path = asset_bundle.model_path
            if not source_path.exists():
                raise FileNotFoundError(f"Exported ONNX bundle not found: {source_path}")
            target_dir = output_dir / model_name
            target_dir.mkdir(parents=True, exist_ok=True)
            target_model = target_dir / "model.onnx"
            if not _same_file(source_path, target_model):
                shutil.copy2(source_path, target_model)
            sha256 = sha256_file(target_model)
            runtime_target = "cuda" if execution_policy.effective_mode == "cuda" else "cpu"
            precision = "fp16" if runtime_target == "cuda" else "fp32"
            metadata = ModelAssetManifestV2(
                model_name=model_name,
                purpose="image_embedding",
                format="onnx",
                target=runtime_target,
                min_compute_capability="7.0" if runtime_target == "cuda" else "",
                precision=precision,
                sha256=sha256,
                size_bytes=int(target_model.stat().st_size),
                source_label=MODEL_SOURCE_LABELS.get(model_name, ""),
                source_url=MODEL_SOURCE_URLS.get(model_name, ""),
                license=MODEL_LICENSES.get(model_name, "Review upstream model license before distribution."),
                input_size=(int(bundle.input_size[0]), int(bundle.input_size[1])),
                signature=bundle.signature,
            ).as_dict()
            metadata.update(
                {
                    "family": bundle.family,
                    "source_path": str(source_path),
                    "target_model": str(target_model),
                    "onnx_providers": list(runtime_service.detect().onnx_providers),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            manifest["bundles"].append(metadata)
        except Exception as exc:
            manifest["failures"].append({"model": model_name, "reason": str(exc)})

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))
    return 1 if manifest["failures"] else 0


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.samefile(right)
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
