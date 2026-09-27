#!/usr/bin/env python3
"""Run every regression referenced by the durable-operation fault manifest.

The report distinguishes verifier PASS from release qualification: referenced
Linux tests can pass while the manifest remains PARTIAL because classified
fault/platform gaps still exist.  Reports use disposable runtime/settings and
never exercise user media or a user Data Home.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "durable_operation_fault_matrix.json"
REPORT_VERSION = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict[str, object]:
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise ValueError("durable-operation manifest schema_version must be 1")
    operations = payload.get("operations")
    if not isinstance(operations, list) or not operations:
        raise ValueError("durable-operation manifest must contain operations")
    return payload


def _test_nodes(manifest: dict[str, object]) -> tuple[str, ...]:
    nodes = {"tests/test_durable_operation_manifest.py"}
    for operation in manifest["operations"]:
        nodes.update(str(node) for node in operation["tests"])
    return tuple(sorted(nodes))


def _pytest_summary(output: str) -> str:
    for line in reversed(output.splitlines()):
        stripped = line.strip().strip("=").strip()
        if re.search(r"\b(?:passed|failed|error|errors)\b", stripped) and re.search(r"\bin\s+[0-9.]+s\b", stripped):
            return stripped
    return ""


def _load_kill_evidence(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    records: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid killed-process evidence line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"killed-process evidence line {line_number} must be an object")
        records.append(record)
    return records


def _kill_evidence_checks(
    manifest: dict[str, object],
    records: list[dict[str, object]],
) -> dict[str, bool]:
    expected_cases = list(manifest.get("actual_process_exit_cases", []))
    expected_by_id = {str(case["id"]): case for case in expected_cases}
    actual_by_id = {str(record.get("evidence_id", "")): record for record in records}
    digests_valid = all(
        len(str(record.get(field, ""))) == 64
        for record in records
        for field in ("pre_state_sha256", "crash_state_sha256", "recovered_state_sha256")
    )
    cases_valid = all(
        str(record.get("result", "")) == "PASS"
        and str(record.get("fault", "")) == "actual_sigkill"
        and str(record.get("signal", "")) == "SIGKILL"
        and int(record.get("child_exit_code", 0) or 0) == -int(signal.SIGKILL)
        and int(record.get("recovery_attempts", 0) or 0) >= 2
        and str(record.get("operation", "")) == str(expected_by_id[evidence_id]["operation"])
        and str(record.get("boundary", "")) == str(expected_by_id[evidence_id]["checkpoint"])
        for evidence_id, record in actual_by_id.items()
        if evidence_id in expected_by_id
    )
    return {
        "killed_process_evidence_complete": set(actual_by_id) == set(expected_by_id),
        "killed_process_evidence_unique": len(actual_by_id) == len(records),
        "killed_process_state_digests_valid": bool(records) and digests_valid,
        "killed_process_cases_valid": bool(records) and cases_valid,
    }


def _session_pids(session_id: int) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return []
    members: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[3]) == int(session_id):
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members)


def _wait_for_session_exit(session_id: int, *, timeout_s: float = 2.0) -> list[int]:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        members = _session_pids(session_id)
        if not members or time.monotonic() >= deadline:
            return members
        time.sleep(0.05)


def _terminate_session(process: subprocess.Popen[str]) -> None:
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
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args(argv)
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def _new_report_dir(path: Path) -> Path:
    report_dir = path.expanduser().resolve()
    if report_dir == REPO_ROOT or REPO_ROOT in report_dir.parents:
        raise ValueError("durable-fault report directory must be outside the repository")
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {report_dir}")
    report_dir.mkdir(parents=True)
    return report_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_dir = _new_report_dir(args.report_dir)
    report_path = report_dir / "durable_fault_matrix.json"
    log_path = report_dir / "pytest.log"
    kill_evidence_path = report_dir / "killed_process_evidence.jsonl"
    manifest = _load_manifest()
    nodes = _test_nodes(manifest)
    environment = dict(os.environ)
    guard = REPO_ROOT / "tests" / "network_guard"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CLUSTERLENS_RUNTIME_ROOT": str(report_dir / "runtime"),
            "IMAGE_CLUSTERING_APP_DIR": str(report_dir / "runtime"),
            "XDG_CONFIG_HOME": str(report_dir / "settings"),
            "PYTHONPYCACHEPREFIX": str(report_dir / "pycache"),
            "QT_QPA_PLATFORM": "offscreen",
            "CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(guard) + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
            "CLUSTERLENS_DURABLE_KILL_EVIDENCE": str(kill_evidence_path),
        }
    )
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", *nodes]
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    session_id = process.pid
    timed_out = False
    try:
        output, _ = process.communicate(timeout=float(args.timeout_seconds))
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_session(process)
        output, _ = process.communicate()
    exit_code = process.returncode if process.returncode is not None else 124
    log_path.write_text(output, encoding="utf-8")
    survivors = _wait_for_session_exit(session_id)
    try:
        kill_evidence = _load_kill_evidence(kill_evidence_path)
        kill_evidence_error = ""
    except (OSError, ValueError) as exc:
        kill_evidence = []
        kill_evidence_error = str(exc)
    operations = list(manifest["operations"])
    gaps = {
        str(operation["id"]): list(operation["known_gaps"])
        for operation in operations
        if operation["known_gaps"]
    }
    platform_status = {
        name: str(details["status"])
        for name, details in dict(manifest["platforms"]).items()
    }
    checks = {
        "pytest_passed": exit_code == 0,
        "pytest_did_not_time_out": not timed_out,
        "pytest_summary_present": bool(_pytest_summary(output)),
        "process_session_drained": not survivors,
        "all_manifest_operations_exercised": all(operation["tests"] for operation in operations),
        **_kill_evidence_checks(manifest, kill_evidence),
    }
    validation = "PASS" if all(checks.values()) else "FAIL"
    qualification = "PASS" if validation == "PASS" and not gaps and all(
        status == "PASS" for status in platform_status.values()
    ) else "PARTIAL"
    payload = {
        "report_version": REPORT_VERSION,
        "operation": "durable_operation_fault_matrix",
        "validation": validation,
        "qualification": qualification,
        "manifest": str(MANIFEST_PATH),
        "manifest_sha256": _sha256_file(MANIFEST_PATH),
        "coverage_granularity": str(manifest.get("coverage_granularity", "")),
        "operation_count": len(operations),
        "boundary_count": sum(len(operation["boundaries"]) for operation in operations),
        "referenced_test_count": len(nodes) - 1,
        "test_nodes": nodes,
        "actual_killed_process_case_count": len(kill_evidence),
        "actual_killed_process_cases": kill_evidence,
        "actual_killed_process_evidence": str(kill_evidence_path),
        "actual_killed_process_evidence_sha256": (
            _sha256_file(kill_evidence_path) if kill_evidence_path.is_file() else ""
        ),
        "actual_killed_process_evidence_error": kill_evidence_error,
        "known_fault_gaps": gaps,
        "platform_status": platform_status,
        "command": command,
        "exit_code": int(exit_code),
        "timed_out": timed_out,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "pytest_summary": _pytest_summary(output),
        "pytest_log": str(log_path),
        "pytest_log_sha256": _sha256_file(log_path),
        "session_id": int(session_id),
        "surviving_session_pids": survivors,
        "checks": checks,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "qt_platform": environment["QT_QPA_PLATFORM"],
            "private_source_media_used": False,
            "network": "external Python networking disabled; referenced tests use disposable fixtures",
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "qualification": qualification, "report": str(report_path)}, indent=2))
    return 0 if validation == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
