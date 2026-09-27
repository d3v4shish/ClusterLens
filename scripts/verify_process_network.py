#!/usr/bin/env python3
"""Audit a command and every descendant for external native network calls.

The canonical Python guard prevents Python socket use, but native libraries and
executables do not import that guard.  This Linux release verifier follows the
complete process tree with ``strace`` and rejects any outbound AF_INET or
AF_INET6 destination that is not loopback.  Unix sockets and loopback fault
servers remain allowed.  A missing tracer is NOT_RUN (exit 3), never PASS.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = 1
EXIT_NOT_RUN = 3
_OUTBOUND_CALL = re.compile(r"(?:^|\s)(?:connect|sendto|sendmsg|sendmmsg)\(")
_IPV4_ADDRESS = re.compile(r'inet_addr\("([^"\\]+)"\)')
_IPV6_ADDRESS = re.compile(r'inet_pton\(AF_INET6,\s*"([^"\\]+)"\)')


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _external_attempts(trace_text: str) -> list[dict[str, object]]:
    attempts: list[dict[str, object]] = []
    for line_number, line in enumerate(trace_text.splitlines(), start=1):
        if not _OUTBOUND_CALL.search(line):
            continue
        addresses = _IPV4_ADDRESS.findall(line) + _IPV6_ADDRESS.findall(line)
        if "sa_family=AF_INET" in line and not addresses:
            attempts.append(
                {
                    "line": line_number,
                    "address": None,
                    "reason": "unparseable IP destination",
                    "trace": line[:1000],
                }
            )
            continue
        for address in addresses:
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                attempts.append(
                    {
                        "line": line_number,
                        "address": address,
                        "reason": "unparseable IP destination",
                        "trace": line[:1000],
                    }
                )
                continue
            if parsed.is_loopback:
                continue
            attempts.append(
                {
                    "line": line_number,
                    "address": str(parsed),
                    "reason": "non-loopback IP destination",
                    "trace": line[:1000],
                }
            )
    return attempts


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
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command to audit, after --.")
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return args


def _require_external_report_dir(path: Path) -> Path:
    report_dir = path.expanduser().resolve()
    if report_dir == REPO_ROOT or REPO_ROOT in report_dir.parents:
        raise ValueError("network-audit report directory must be outside the repository")
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing report directory: {report_dir}")
    report_dir.mkdir(parents=True)
    return report_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_dir = _require_external_report_dir(args.report_dir)
    report_path = report_dir / "process_network.json"
    trace_path = report_dir / "network.strace"
    command_log = report_dir / "command.log"
    tracer = shutil.which("strace")
    if tracer is None:
        payload = {
            "report_version": REPORT_VERSION,
            "operation": "native_process_network_audit",
            "status": "NOT_RUN",
            "reason": "strace is required to observe native descendant network syscalls",
            "command": args.command,
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": "NOT_RUN", "report": str(report_path)}, indent=2))
        return EXIT_NOT_RUN

    environment = dict(os.environ)
    guard = REPO_ROOT / "tests" / "network_guard"
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment.update(
        {
            "CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(guard) + (os.pathsep + existing_pythonpath if existing_pythonpath else ""),
        }
    )
    traced_command = [
        tracer,
        "-f",
        "-qq",
        "-s",
        "512",
        "-e",
        "trace=%network",
        "-o",
        str(trace_path),
        "--",
        *args.command,
    ]
    started = time.monotonic()
    timed_out = False
    with command_log.open("wb") as log_handle:
        process = subprocess.Popen(
            traced_command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        session_id = process.pid
        try:
            exit_code = process.wait(timeout=float(args.timeout_seconds))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_session(process)
            exit_code = process.returncode if process.returncode is not None else 124
    survivors = _wait_for_session_exit(session_id)
    trace_text = trace_path.read_text(encoding="utf-8", errors="replace") if trace_path.is_file() else ""
    external_attempts = _external_attempts(trace_text)
    checks = {
        "command_passed": exit_code == 0,
        "command_did_not_time_out": not timed_out,
        "trace_created": trace_path.is_file(),
        "no_external_ip_attempts": not external_attempts,
        "process_session_drained": not survivors,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "report_version": REPORT_VERSION,
        "operation": "native_process_network_audit",
        "status": status,
        "command": args.command,
        "traced_command": traced_command,
        "cwd": str(REPO_ROOT),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exit_code": int(exit_code),
        "timed_out": timed_out,
        "session_id": int(session_id),
        "surviving_session_pids": survivors,
        "external_attempts": external_attempts,
        "checks": checks,
        "trace": str(trace_path),
        "trace_sha256": _sha256_file(trace_path) if trace_path.is_file() else None,
        "command_log": str(command_log),
        "command_log_sha256": _sha256_file(command_log),
        "policy": {
            "allowed": ["AF_UNIX", "AF_NETLINK", "IPv4 loopback", "IPv6 loopback"],
            "rejected": ["non-loopback AF_INET", "non-loopback AF_INET6", "unparseable IP destination"],
            "python_guard_inherited": True,
            "model_libraries_forced_offline": True,
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "strace": tracer,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "report": str(report_path)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
