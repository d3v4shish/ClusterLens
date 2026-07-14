# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


REPO_ROOT = Path(SPEC).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ASSETS_DIR = REPO_ROOT / "apps" / "pyqt_production" / "assets"
MODEL_ASSETS_DIR = Path(os.environ.get("CLUSTERLENS_MODEL_ASSETS_DIR", str(REPO_ROOT / "build" / "model_assets"))).resolve()
ENTRYPOINT = REPO_ROOT / "apps" / "pyqt_production" / "__main__.py"
BUILD_VARIANT = os.environ.get("CLUSTERLENS_BUILD_VARIANT", "cpu").strip().lower()
PACKAGE_MODE = os.environ.get("CLUSTERLENS_PACKAGE_MODE", "onefile").strip().lower()
EXECUTABLE_NAME = os.environ.get("CLUSTERLENS_EXECUTABLE_NAME", "ClusterLens").strip() or "ClusterLens"


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
    "torch.distributed",
    "torch.distributed.elastic",
]
excludes = list(common_excludes)
if BUILD_VARIANT == "cpu":
    excludes.extend(cpu_excludes)


a = Analysis(
    [str(ENTRYPOINT)],
    pathex=[str(REPO_ROOT), str(SRC_DIR)],
    binaries=[],
    datas=[
        (str(ASSETS_DIR / "app_icon.png"), "apps/pyqt_production/assets"),
        *_model_asset_datas(),
    ],
    hiddenimports=[
        "apps.pyqt_production.worker",
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
