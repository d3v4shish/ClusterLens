"""Run the UX-31 production-shell journey in an isolated process session.

The inner pytest case owns the production window and writes its workload
report.  This wrapper waits for that process to exit and then proves that no
process from the isolated operating-system session survived the window close
or interpreter shutdown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = 1


def _child_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "tests/test_mixed_ui_workload.py",
        "-q",
    ]


def _child_environment(report_dir: Path, mixed_report_path: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("CLUSTERLENS_RUNTIME_ROOT", None)
    environment.update(
        {
            "IMAGE_CLUSTERING_APP_DIR": str(report_dir / "test-runtime"),
            "PYTHONPYCACHEPREFIX": str(report_dir / "pycache"),
            "XDG_CONFIG_HOME": str(report_dir / "settings"),
            "QT_QPA_PLATFORM": environment.get("QT_QPA_PLATFORM", "offscreen"),
            "CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "CLUSTERLENS_UX31_REPORT": str(mixed_report_path),
        }
    )
    network_guard = str(REPO_ROOT / "tests" / "network_guard")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        f"{network_guard}{os.pathsep}{existing_pythonpath}"
        if existing_pythonpath
        else network_guard
    )
    return environment


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_pids(session_id: int) -> list[int]:
    members: list[int] = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return members
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            remainder = raw[raw.rfind(")") + 2 :].split()
            process_session = int(remainder[3])
        except (OSError, ValueError, IndexError):
            continue
        if process_session == int(session_id):
            members.append(int(entry.name))
    return sorted(members)


def _wait_for_session_exit(session_id: int, *, timeout_s: float = 2.0) -> list[int]:
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        members = _session_pids(session_id)
        if not members or time.monotonic() >= deadline:
            return members
        time.sleep(0.05)


def _terminate_session(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5.0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="new directory for lifecycle.json, mixed_ui.json and pytest.log",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=180.0,
        help="bounded watchdog for the isolated mixed-workload test",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise SystemExit(f"refusing to overwrite existing report directory: {report_dir}")
    report_dir.mkdir(parents=True)

    mixed_report_path = report_dir / "mixed_ui.json"
    log_path = report_dir / "pytest.log"
    lifecycle_path = report_dir / "lifecycle.json"
    command = _child_command()
    environment = _child_environment(report_dir, mixed_report_path)

    timed_out = False
    started_at = time.monotonic()
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        session_id = process.pid
        try:
            exit_code = process.wait(timeout=max(1.0, float(args.timeout_seconds)))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_session(process)
            exit_code = process.returncode if process.returncode is not None else 124

    surviving_pids = _wait_for_session_exit(session_id)
    elapsed_s = time.monotonic() - started_at
    mixed_report: dict[str, object] | None = None
    mixed_report_error = ""
    try:
        loaded = json.loads(mixed_report_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("mixed workload report root is not an object")
        mixed_report = loaded
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        mixed_report_error = str(exc)

    procfs_available = Path("/proc").is_dir()
    inner_passed = (
        not timed_out
        and exit_code == 0
        and mixed_report is not None
        and mixed_report.get("status") == "PASS"
    )
    if not procfs_available and inner_passed:
        status = "NOT_RUN"
    elif inner_passed and not surviving_pids:
        status = "PASS"
    else:
        status = "FAIL"
    report = {
        "report_version": REPORT_VERSION,
        "operation": "ux31_production_ui_process_lifecycle",
        "status": status,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "procfs_available": procfs_available,
        },
        "execution": {
            "command": command,
            "cwd": str(REPO_ROOT),
            "elapsed_seconds": round(elapsed_s, 3),
            "exit_code": int(exit_code),
            "timed_out": timed_out,
            "process_session_id": int(session_id),
            "surviving_session_pids": surviving_pids,
            "pytest_log_sha256": _sha256_file(log_path),
        },
        "inner_report": {
            "path": str(mixed_report_path),
            "sha256": _sha256_file(mixed_report_path) if mixed_report_path.is_file() else None,
            "status": mixed_report.get("status") if mixed_report is not None else None,
            "error": mixed_report_error or None,
        },
        "post_close_invariants": {
            "test_process_exited": process.poll() is not None,
            "no_surviving_processes_in_owned_session": not surviving_pids,
            "no_surviving_threads_from_test_process": process.poll() is not None,
            "thread_evidence": (
                "Linux threads cannot survive destruction of their owning process; "
                "the isolated test process exited and its complete process session is empty."
            ),
        },
        "limitations": [
            "Linux /proc process-session evidence; use the native platform verifier on other operating systems",
            "an empty process session proves cleanup after close/exit, not native compositor responsiveness",
        ],
    }
    lifecycle_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if status == "PASS":
        return 0
    if status == "NOT_RUN":
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
