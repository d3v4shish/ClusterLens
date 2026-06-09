from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform release packaging guard. It validates gates/signing inputs before an OS-specific packager runs."
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "dist" / "production"))
    parser.add_argument("--skip-release-gates", action="store_true")
    parser.add_argument("--unsigned", action="store_true", help="Allow local unsigned package tests.")
    args = parser.parse_args(argv)

    manifest = REPO_ROOT / "packaging" / "production_release_manifest.json"
    if not manifest.exists():
        print(f"Missing production release manifest: {manifest}", file=sys.stderr)
        return 2
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    installer = manifest_payload.get("installer") or {}
    required_packaging_files = [
        installer.get("variant_build_script"),
        installer.get("pyinstaller_spec"),
        installer.get("windows_installer_script"),
        installer.get("icon_png"),
        installer.get("icon_ico"),
    ]
    for variant in installer.get("build_variants") or []:
        if isinstance(variant, dict):
            required_packaging_files.append(variant.get("requirements"))
    missing_packaging_files = []
    for relative_path in required_packaging_files:
        if not relative_path:
            continue
        candidate = REPO_ROOT / str(relative_path)
        if not candidate.exists():
            missing_packaging_files.append(candidate)
    if missing_packaging_files:
        print("Packaging assets are missing:", file=sys.stderr)
        for path in missing_packaging_files:
            print(f"  - {path}", file=sys.stderr)
        return 2

    if not args.skip_release_gates:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "apps.pyqt_production.release_gates",
                "--report-dir",
                str(REPO_ROOT / "dist" / "release_gates"),
            ],
            cwd=REPO_ROOT,
        )
        if completed.returncode != 0:
            return int(completed.returncode)

    cert = os.environ.get("IMAGE_CLUSTERING_SIGNING_CERT", "")
    timestamp = os.environ.get("IMAGE_CLUSTERING_TIMESTAMP_SERVER", "")
    if not args.unsigned:
        if not cert:
            print("IMAGE_CLUSTERING_SIGNING_CERT is required unless --unsigned is supplied.", file=sys.stderr)
            return 2
        if not timestamp:
            print("IMAGE_CLUSTERING_TIMESTAMP_SERVER is required unless --unsigned is supplied.", file=sys.stderr)
            return 2

    print(f"Packaging manifest verified: {manifest}")
    print(f"Output directory: {Path(args.output_dir)}")
    print(f"Variant build script: {REPO_ROOT / installer.get('variant_build_script', 'scripts/build_pyqt_binary.py')}")
    print(f"PyInstaller spec: {REPO_ROOT / installer.get('pyinstaller_spec', 'packaging/pyqt_production.spec')}")
    print(f"Windows installer script: {REPO_ROOT / installer.get('windows_installer_script', 'packaging/windows_installer.iss')}")
    print(f"Icon assets: {REPO_ROOT / installer.get('icon_png', 'apps/pyqt_production/assets/app_icon.png')} | {REPO_ROOT / installer.get('icon_ico', 'apps/pyqt_production/assets/app_icon.ico')}")
    print("Next step: run scripts/build_pyqt_binary.py --variant cpu or --variant gpu-cu121, then sign the generated artifact.")
    print("This script intentionally fails early when signing inputs or release gates are missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
