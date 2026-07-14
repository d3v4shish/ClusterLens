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
    assert variants["cpu"]["default_package_mode"] == "onefile"
    assert variants["gpu-cu121"]["default_package_mode"] == "onedir"
    assert variants["cpu"]["requirements"] == "packaging/requirements-build-cpu.txt"
    assert variants["gpu-cu121"]["requirements"] == "packaging/requirements-build-gpu-cu121.txt"


def test_variant_builder_matches_manifest_variants() -> None:
    manifest = json.loads((REPO_ROOT / "packaging" / "production_release_manifest.json").read_text(encoding="utf-8"))
    manifest_variants = {str(item["id"]) for item in manifest["installer"]["build_variants"]}

    assert manifest_variants == set(build_pyqt_binary.BUILD_VARIANTS)
    assert build_pyqt_binary.BUILD_VARIANTS["cpu"].default_package_mode == "onefile"
    assert build_pyqt_binary.BUILD_VARIANTS["gpu-cu121"].default_package_mode == "onedir"


def test_variant_builder_prepares_default_bundled_model_assets() -> None:
    assert build_pyqt_binary.DEFAULT_BUNDLED_MODEL_ASSETS == ("fast_preview", "resnet", "convnext")
    assert build_pyqt_binary.DEFAULT_MODEL_ASSET_OUTPUT_DIR == REPO_ROOT / "build" / "model_assets"


def test_cpu_requirements_force_cpu_torch_wheels() -> None:
    requirements = (REPO_ROOT / "packaging" / "requirements-build-cpu.txt").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cpu" in requirements
    assert "torch==2.2.2+cpu" in requirements
    assert "torchvision==0.17.2+cpu" in requirements


def test_update_manifest_lists_separate_cpu_and_gpu_artifacts() -> None:
    manifest = json.loads((REPO_ROOT / "packaging" / "update_manifest.example.json").read_text(encoding="utf-8"))
    variants = {str(item["variant"]) for item in manifest["artifacts"]}

    assert variants == {"cpu", "gpu-cu121"}
