from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import build_pyqt_binary


def test_release_manifest_declares_cpu_and_gpu_build_variants() -> None:
    manifest = json.loads((REPO_ROOT / "packaging" / "production_release_manifest.json").read_text(encoding="utf-8"))
    variants = {
        str(item["id"]): item
        for item in manifest["installer"]["build_variants"]
    }

    assert set(variants) == {"cpu", "gpu-cu121"}
    assert variants["cpu"]["default_package_mode"] == "onedir"
    assert variants["gpu-cu121"]["default_package_mode"] == "onedir"
    assert variants["cpu"]["requirements"] == "packaging/requirements-build-cpu.txt"
    assert variants["gpu-cu121"]["requirements"] == "packaging/requirements-build-gpu-cu121.txt"


def test_variant_builder_matches_manifest_variants() -> None:
    manifest = json.loads((REPO_ROOT / "packaging" / "production_release_manifest.json").read_text(encoding="utf-8"))
    manifest_variants = {str(item["id"]) for item in manifest["installer"]["build_variants"]}

    assert manifest_variants == set(build_pyqt_binary.BUILD_VARIANTS)
    assert build_pyqt_binary.BUILD_VARIANTS["cpu"].default_package_mode == "onedir"
    assert build_pyqt_binary.BUILD_VARIANTS["gpu-cu121"].default_package_mode == "onedir"


def test_variant_builder_prepares_default_bundled_model_assets() -> None:
    assert build_pyqt_binary.DEFAULT_BUNDLED_MODEL_ASSETS == ("fast_preview", "resnet", "convnext")
    assert build_pyqt_binary.DEFAULT_MODEL_ASSET_OUTPUT_DIR == REPO_ROOT / "build" / "model_assets"


def test_production_build_includes_siglip_tokenizer_runtime() -> None:
    requirements = (REPO_ROOT / "packaging" / "requirements-build-base.txt").read_text(encoding="utf-8")
    spec = (REPO_ROOT / "packaging" / "pyqt_production.spec").read_text(encoding="utf-8")

    assert "sentencepiece>=0.2,<1.0" in requirements
    assert '"sentencepiece"' in spec


def test_production_build_includes_face_model_catalog_metadata() -> None:
    spec = (REPO_ROOT / "packaging" / "pyqt_production.spec").read_text(encoding="utf-8")

    assert "FACE_MODEL_CATALOG_DIR" in spec
    assert "_face_model_catalog_datas()" in spec


def test_production_build_includes_and_verifies_facenet_detector_assets() -> None:
    spec = (REPO_ROOT / "packaging" / "pyqt_production.spec").read_text(encoding="utf-8")

    assert 'collect_data_files("facenet_pytorch", includes=["data/*.pt"])' in spec
    assert "FACENET_PYTORCH_MODELS_ANCHOR" in spec
    assert '"facenet_pytorch/models"' in spec
    assert "*_facenet_pytorch_datas()" in spec
    assert build_pyqt_binary.REQUIRED_FACENET_DETECTOR_ASSETS == ("pnet.pt", "rnet.pt", "onet.pt")


def test_packaged_runtime_asset_verifier_rejects_missing_facenet_detector_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "ClusterLens" / "_internal" / "facenet_pytorch" / "data"
    data_dir.mkdir(parents=True)
    (data_dir.parent / "models").mkdir()
    for name in build_pyqt_binary.REQUIRED_FACENET_DETECTOR_ASSETS:
        (data_dir / name).write_bytes(b"weights")

    build_pyqt_binary.verify_packaged_runtime_assets(
        tmp_path,
        package_mode="onedir",
        executable_name="ClusterLens",
        dry_run=False,
    )
    (data_dir / "pnet.pt").unlink()
    try:
        build_pyqt_binary.verify_packaged_runtime_assets(
            tmp_path,
            package_mode="onedir",
            executable_name="ClusterLens",
            dry_run=False,
        )
    except RuntimeError as exc:
        assert "pnet.pt" in str(exc)
    else:
        raise AssertionError("missing facenet-pytorch detector data must fail the package build")


def test_packaged_runtime_asset_verifier_rejects_missing_facenet_models_directory(tmp_path: Path) -> None:
    data_dir = tmp_path / "ClusterLens" / "_internal" / "facenet_pytorch" / "data"
    data_dir.mkdir(parents=True)
    for name in build_pyqt_binary.REQUIRED_FACENET_DETECTOR_ASSETS:
        (data_dir / name).write_bytes(b"weights")

    try:
        build_pyqt_binary.verify_packaged_runtime_assets(
            tmp_path,
            package_mode="onedir",
            executable_name="ClusterLens",
            dry_run=False,
        )
    except RuntimeError as exc:
        assert "models directory" in str(exc)
        assert "models/../data" in str(exc)
    else:
        raise AssertionError("missing facenet-pytorch models directory must fail the package build")


def test_cpu_requirements_force_cpu_torch_wheels() -> None:
    requirements = (REPO_ROOT / "packaging" / "requirements-build-cpu.txt").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.2.2+cpu" in requirements
    assert "torchvision==0.17.2+cpu" in requirements
    assert "onnxruntime>=" in requirements
    assert "onnxruntime-gpu" not in requirements
    assert "triton" not in requirements.lower()
    assert "nvidia-" not in requirements.lower()


def test_cuda_requirements_force_cuda_wheels_and_provider() -> None:
    requirements = (REPO_ROOT / "packaging" / "requirements-build-gpu-cu121.txt").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cu121" in requirements
    assert "torch==2.2.2" in requirements
    assert "torchvision==0.17.2" in requirements
    assert "onnxruntime-gpu" in requirements


def test_update_manifest_lists_separate_cpu_and_gpu_artifacts() -> None:
    manifest = json.loads((REPO_ROOT / "packaging" / "update_manifest.example.json").read_text(encoding="utf-8"))
    variants = {str(item["variant"]) for item in manifest["artifacts"]}

    assert manifest["manifest_version"] == "UpdateManifestV2"
    assert variants == {"cpu", "gpu-cu121"}
    required_fields = {
        "channel",
        "version",
        "minimum_supported_version",
        "os",
        "architecture",
        "variant",
        "cuda_runtime",
        "size_bytes",
        "sha256",
        "url",
        "ed25519_signature",
    }
    for artifact in manifest["artifacts"]:
        assert required_fields.issubset(artifact)
    gpu = next(item for item in manifest["artifacts"] if item["variant"] == "gpu-cu121")
    cpu = next(item for item in manifest["artifacts"] if item["variant"] == "cpu")
    assert gpu["cuda_runtime"] == "12.1"
    assert cpu["cuda_runtime"] == ""
