# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


REPO_ROOT = Path(SPEC).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ASSETS_DIR = REPO_ROOT / "apps" / "pyqt_production" / "assets"
MODEL_ASSETS_DIR = Path(os.environ.get("CLUSTERLENS_MODEL_ASSETS_DIR", str(REPO_ROOT / "build" / "model_assets"))).resolve()
FACE_MODEL_CATALOG_DIR = REPO_ROOT / "face_model_assets"
ENTRYPOINT = REPO_ROOT / "apps" / "pyqt_production" / "__main__.py"
BUILD_VARIANT = os.environ.get("CLUSTERLENS_BUILD_VARIANT", "cpu").strip().lower()
PACKAGE_MODE = os.environ.get("CLUSTERLENS_PACKAGE_MODE", "onedir").strip().lower()
EXECUTABLE_NAME = os.environ.get("CLUSTERLENS_EXECUTABLE_NAME", "ClusterLens").strip() or "ClusterLens"
FACENET_PYTORCH_DETECTOR_ASSETS = ("pnet.pt", "rnet.pt", "onet.pt")
FACENET_PYTORCH_MODELS_ANCHOR = REPO_ROOT / "packaging" / "facenet_pytorch_models_runtime_anchor.txt"


def _model_asset_datas():
    if not MODEL_ASSETS_DIR.exists():
        return []
    datas = []
    for path in sorted(MODEL_ASSETS_DIR.rglob("*")):
        if path.is_dir():
            continue
        relative_path = path.relative_to(MODEL_ASSETS_DIR)
        if ".runtime" in relative_path.parts:
            continue
        destination = Path("model_assets") / relative_path.parent
        datas.append((str(path), str(destination)))
    return datas


def _face_model_catalog_datas():
    if not FACE_MODEL_CATALOG_DIR.exists():
        return []
    return [
        (str(path), str(Path("face_model_assets") / path.relative_to(FACE_MODEL_CATALOG_DIR).parent))
        for path in sorted(FACE_MODEL_CATALOG_DIR.rglob("*"))
        if path.is_file()
    ]


def _facenet_pytorch_datas():
    datas = collect_data_files("facenet_pytorch", includes=["data/*.pt"])
    collected_names = {Path(source).name for source, _destination in datas}
    missing = sorted(set(FACENET_PYTORCH_DETECTOR_ASSETS) - collected_names)
    if missing:
        raise RuntimeError(
            "facenet-pytorch detector assets are missing from the build environment: "
            + ", ".join(missing)
        )
    # facenet-pytorch builds its weight path as models/../data/<network>.pt.
    # Python modules live in PyInstaller's PYZ, so the physical `models`
    # directory would otherwise be absent and POSIX traversal fails before
    # it can resolve `..`, even though data/<network>.pt exists.
    if not FACENET_PYTORCH_MODELS_ANCHOR.is_file():
        raise RuntimeError(f"Missing facenet-pytorch runtime anchor: {FACENET_PYTORCH_MODELS_ANCHOR}")
    datas.append((str(FACENET_PYTORCH_MODELS_ANCHOR), "facenet_pytorch/models"))
    return datas


common_excludes = [
    "IPython",
    "matplotlib",
    "notebook",
    "pytest",
    "tensorboard",
    "tensorflow",
    "torch.utils.tensorboard",
]
cpu_excludes = [
    "cupy",
    "nvidia",
    "triton",
    # Torch imports torch.distributed.rpc from torch._jit_internal even in
    # CPU-only wheels. Excluding the whole package produces a packaged app that
    # fails before it can start the worker.
    "torch.distributed.elastic",
]
excludes = list(common_excludes)
gpu_binaries = []
gpu_hiddenimports = []
if BUILD_VARIANT == "cpu":
    excludes.extend(cpu_excludes)
else:
    for package_name in (
        "libcudf",
        "libcuml",
        "libkvikio",
        "libraft",
        "librmm",
        "libucx",
        "libucxx",
    ):
        gpu_binaries.extend(collect_dynamic_libs(package_name))
    gpu_hiddenimports.extend(("cuml.cluster", "cuml.cluster.hdbscan", "cupy"))


a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(REPO_ROOT), str(SRC_DIR)],
    binaries=gpu_binaries,
    datas=[
        (str(ASSETS_DIR / "app_icon.png"), "apps/pyqt_production/assets"),
        (str(REPO_ROOT / "docs" / "USER_FLOWS.md"), "docs"),
        *_model_asset_datas(),
        *_face_model_catalog_datas(),
        *_facenet_pytorch_datas(),
    ],
    hiddenimports=[
        "apps.pyqt_production.worker",
        "app.services.face_region_metadata",
        "sentencepiece",
        "torch.distributed.rpc",
        *gpu_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if PACKAGE_MODE == "onedir":
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=EXECUTABLE_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(ASSETS_DIR / "app_icon.ico"),
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=EXECUTABLE_NAME,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=EXECUTABLE_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(ASSETS_DIR / "app_icon.ico"),
    )
