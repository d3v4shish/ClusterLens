from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLED_MODEL_ASSETS = ("fast_preview", "resnet", "convnext")
DEFAULT_MODEL_ASSET_OUTPUT_DIR = REPO_ROOT / "build" / "model_assets"


@dataclass(frozen=True)
class BuildVariant:
    name: str
    requirements: Path
    default_package_mode: str
    description: str


BUILD_VARIANTS = {
    "cpu": BuildVariant(
        name="cpu",
        requirements=REPO_ROOT / "packaging" / "requirements-build-cpu.txt",
        default_package_mode="onefile",
        description="CPU-only Torch runtime. This is the default public build.",
    ),
    "gpu-cu121": BuildVariant(
        name="gpu-cu121",
        requirements=REPO_ROOT / "packaging" / "requirements-build-gpu-cu121.txt",
        default_package_mode="onedir",
        description="CUDA 12.1 Torch runtime. Large separate GPU package.",
    ),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build ClusterLens PyQt production binaries by CPU/GPU variant.")
    parser.add_argument("--variant", choices=["cpu", "gpu-cu121", "all"], default="cpu")
    parser.add_argument(
        "--package-mode",
        choices=["auto", "onefile", "onedir"],
        default="auto",
        help="auto uses onefile for CPU and onedir for GPU.",
    )
    parser.add_argument("--recreate-venv", action="store_true", help="Delete and recreate the variant build venv.")
    parser.add_argument("--skip-install", action="store_true", help="Reuse the existing variant build venv as-is.")
    parser.add_argument("--prepare-only", action="store_true", help="Create/verify the build venv but do not run PyInstaller.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without executing them.")
    parser.add_argument("--dist-root", default=str(REPO_ROOT / "dist" / "production"))
    parser.add_argument("--work-root", default=str(REPO_ROOT / "build" / "pyinstaller"))
    args = parser.parse_args(argv)

    variants = tuple(BUILD_VARIANTS) if args.variant == "all" else (args.variant,)
    for variant_name in variants:
        variant = BUILD_VARIANTS[variant_name]
        package_mode = variant.default_package_mode if args.package_mode == "auto" else args.package_mode
        if build_variant(
            variant,
            package_mode=package_mode,
            dist_root=Path(args.dist_root),
            work_root=Path(args.work_root),
            recreate_venv=bool(args.recreate_venv),
            skip_install=bool(args.skip_install),
            prepare_only=bool(args.prepare_only),
            dry_run=bool(args.dry_run),
        ) != 0:
            return 1
    return 0


def build_variant(
    variant: BuildVariant,
    *,
    package_mode: str,
    dist_root: Path,
    work_root: Path,
    recreate_venv: bool,
    skip_install: bool,
    prepare_only: bool,
    dry_run: bool,
) -> int:
    target = platform_target()
    venv_dir = REPO_ROOT / "build" / "venvs" / variant.name
    python = venv_python(venv_dir)
    dist_dir = dist_root / f"{target}-{variant.name}"
    work_dir = work_root / variant.name
    model_assets_dir = DEFAULT_MODEL_ASSET_OUTPUT_DIR

    print(f"Variant: {variant.name} ({variant.description})")
    print(f"Package mode: {package_mode}")
    print(f"Build venv: {venv_dir}")
    print(f"Dist dir: {dist_dir}")

    if not skip_install:
        if recreate_venv and venv_dir.exists():
            if dry_run:
                print(f"Would remove {venv_dir}")
            else:
                shutil.rmtree(venv_dir)
        if not python.exists():
            if dry_run:
                print(f"Would create venv at {venv_dir}")
            else:
                venv.create(venv_dir, with_pip=True, symlinks=(os.name != "nt"))
        run([str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=dry_run)
        run([str(python), "-m", "pip", "install", "-r", str(variant.requirements)], dry_run=dry_run)
        run([str(python), "-m", "pip", "install", "--no-deps", "-e", str(REPO_ROOT)], dry_run=dry_run)

    verify_variant_environment(python, variant, dry_run=dry_run)
    prepare_bundled_model_assets(python, output_dir=model_assets_dir, dry_run=dry_run)
    if prepare_only:
        return 0

    env = os.environ.copy()
    env["CLUSTERLENS_BUILD_VARIANT"] = variant.name
    env["CLUSTERLENS_PACKAGE_MODE"] = package_mode
    env["CLUSTERLENS_EXECUTABLE_NAME"] = "ClusterLens"
    env["CLUSTERLENS_MODEL_ASSETS_DIR"] = str(model_assets_dir)
    command = [
        str(python),
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        str(REPO_ROOT / "packaging" / "pyqt_production.spec"),
    ]
    run(command, env=env, dry_run=dry_run)
    return 0


def verify_variant_environment(python: Path, variant: BuildVariant, *, dry_run: bool) -> None:
    script = r"""
from __future__ import annotations
import importlib.metadata as md
import sys

names = sorted((dist.metadata.get("Name") or dist.metadata.get("name") or "").lower() for dist in md.distributions())
bad_cuda = [name for name in names if name.startswith("nvidia-") or name == "triton"]
try:
    import torch
except Exception as exc:
    print(f"Unable to import torch: {exc}", file=sys.stderr)
    raise SystemExit(2)
print(f"torch={torch.__version__}")
if sys.argv[1] == "cpu" and bad_cuda:
    print("CPU build environment contains CUDA packages: " + ", ".join(bad_cuda), file=sys.stderr)
    raise SystemExit(3)
"""
    run([str(python), "-c", script, variant.name], dry_run=dry_run)


def prepare_bundled_model_assets(python: Path, *, output_dir: Path, dry_run: bool) -> None:
    run(
        [
            str(python),
            str(REPO_ROOT / "scripts" / "export_prod_onnx_assets.py"),
            "--models",
            ",".join(DEFAULT_BUNDLED_MODEL_ASSETS),
            "--output-dir",
            str(output_dir),
            "--preferred-execution-mode",
            "cpu",
        ],
        dry_run=dry_run,
    )


def platform_target() -> str:
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "x64"
    if sys.platform.startswith("linux"):
        return f"linux-{arch}"
    if sys.platform == "darwin":
        return f"macos-{arch}"
    if os.name == "nt":
        return f"windows-{arch}"
    return f"{sys.platform}-{arch}"


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run(command: list[str], *, env: dict[str, str] | None = None, dry_run: bool = False) -> None:
    print("+ " + " ".join(command))
    if dry_run:
        return
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
