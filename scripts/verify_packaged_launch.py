"""Run an isolated noninteractive smoke check against a built executable.

The verifier requires an empty runtime root and a new report path. It is only
for ignored release evidence and never reuses a user's runtime data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_ENV = "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE"
SMOKE_REPORT_ENV = "CLUSTERLENS_PACKAGED_LAUNCH_REPORT"
SCENARIO = "settings-storage"
REQUIRED_CHECKS = frozenset(
    {
        "frozen_executable",
        "window_icon",
        "settings_dialog_visible",
        "storage_tab",
        "generated_storage_controls",
        "generated_storage_categories_match_service",
    }
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True, help="Built ClusterLens executable to launch.")
    parser.add_argument("--runtime-root", type=Path, required=True, help="Empty disposable runtime directory.")
    parser.add_argument("--report-path", type=Path, required=True, help="New application smoke JSON report path.")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    executable = args.executable.expanduser().resolve()
    runtime_root = _require_empty_directory(args.runtime_root)
    report_path = args.report_path.expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"built executable does not exist: {executable}")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")

    environment = os.environ.copy()
    environment.update(
        {
            "CLUSTERLENS_RUNTIME_ROOT": str(runtime_root),
            "IMAGE_CLUSTERING_APP_DIR": str(runtime_root),
            "XDG_DATA_HOME": str(runtime_root / "xdg-data"),
            "XDG_CONFIG_HOME": str(runtime_root / "xdg-config"),
            "QT_QPA_PLATFORM": environment.get("QT_QPA_PLATFORM", "offscreen"),
            SMOKE_ENV: SCENARIO,
            SMOKE_REPORT_ENV: str(report_path),
        }
    )
    completed = subprocess.run(
        [str(executable)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=max(1.0, float(args.timeout_seconds)),
        check=False,
    )
    payload = _read_and_validate_report(report_path)
    app_log = runtime_root / "logs" / "app.log"
    if not app_log.is_file():
        raise RuntimeError(f"packaged smoke did not create an application log: {app_log}")
    if "app icon applied" not in app_log.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("packaged smoke log did not record a resolved application icon")
    if completed.returncode != 0:
        raise RuntimeError(
            f"packaged executable returned {completed.returncode}; stdout={completed.stdout.strip()!r}; "
            f"stderr={completed.stderr.strip()!r}"
        )

    verifier_report = report_path.with_name("packaged_launch_verifier.json")
    if verifier_report.exists():
        raise FileExistsError(f"refusing to overwrite existing verifier report: {verifier_report}")
    verification = {
        "report_version": "1",
        "scenario": "packaged-launch-settings-storage-verifier",
        "validation": "PASS",
        "executable": str(executable),
        "artifact_sha256": _sha256_file(executable),
        "artifact_size_bytes": executable.stat().st_size,
        "runtime_root": str(runtime_root),
        "application_report": str(report_path),
        "application_log": str(app_log),
        "checks": payload["checks"],
        "network": "not used by launch smoke",
        "verifier_environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "qt_platform": environment["QT_QPA_PLATFORM"],
        },
    }
    verifier_report.write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validation": "PASS", "report": str(verifier_report)}, indent=2))
    return 0


def _read_and_validate_report(report_path: Path) -> dict[str, object]:
    if not report_path.is_file():
        raise RuntimeError(f"packaged executable did not write a smoke report: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if payload.get("validation") != "PASS":
        raise RuntimeError(f"packaged smoke reported failure: {payload.get('error') or payload}")
    if payload.get("scenario") != f"packaged-launch-{SCENARIO}":
        raise RuntimeError(f"unexpected packaged smoke scenario: {payload.get('scenario')!r}")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("packaged smoke report omitted its checks")
    missing = REQUIRED_CHECKS - set(checks)
    failed = sorted(name for name in REQUIRED_CHECKS if checks.get(name) is not True)
    if missing or failed:
        raise RuntimeError(f"packaged smoke checks failed; missing={sorted(missing)} failed={failed}")
    return payload


def _require_empty_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"runtime root must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
