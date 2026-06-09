# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path


REPO_ROOT = Path(SPEC).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ASSETS_DIR = REPO_ROOT / "apps" / "pyqt_production" / "assets"
ENTRYPOINT = REPO_ROOT / "apps" / "pyqt_production" / "__main__.py"
BUILD_VARIANT = os.environ.get("CLUSTERLENS_BUILD_VARIANT", "cpu").strip().lower()
PACKAGE_MODE = os.environ.get("CLUSTERLENS_PACKAGE_MODE", "onefile").strip().lower()
EXECUTABLE_NAME = os.environ.get("CLUSTERLENS_EXECUTABLE_NAME", "ClusterLens").strip() or "ClusterLens"


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
