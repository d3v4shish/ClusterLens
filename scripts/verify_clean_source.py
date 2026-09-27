"""Verify the current candidate from two source-only, offline snapshots.

Tracked and non-ignored untracked files are copied without ``.git`` or local
virtual environments.  Each snapshot is built, launch-checked and fully tested
from a different outside working directory with uv forced offline.  The report
records source identity, command logs and process-session cleanup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
    ).stdout


def _candidate_files() -> tuple[Path, ...]:
    raw = _git_bytes("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    relative_paths = {
        Path(part.decode("utf-8", errors="surrogateescape"))
        for part in raw.split(b"\0")
        if part
    }
    return tuple(sorted(relative_paths, key=lambda path: path.as_posix()))


def _snapshot_digest(root: Path, relative_paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for relative in relative_paths:
        path = root / relative
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            kind = b"symlink"
            payload_digest = hashlib.sha256(os.readlink(path).encode("utf-8")).hexdigest()
        elif stat.S_ISREG(mode):
            kind = b"file"
            payload_digest = _sha256_file(path)
        else:
            raise ValueError(f"unsupported source entry: {relative}")
        executable = b"1" if mode & stat.S_IXUSR else b"0"
        digest.update(relative.as_posix().encode("utf-8", errors="surrogateescape"))
        digest.update(b"\0" + kind + b"\0" + executable + b"\0")
        digest.update(payload_digest.encode("ascii") + b"\n")
    return digest.hexdigest()


def _materialize_source(destination: Path, relative_paths: tuple[Path, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for relative in relative_paths:
        source = REPO_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            target.symlink_to(os.readlink(source))
        else:
            shutil.copy2(source, target)


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


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    timeout_s: float,
) -> dict[str, object]:
    started = time.monotonic()
    timed_out = False
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        session_id = process.pid
        try:
            exit_code = process.wait(timeout=max(1.0, timeout_s))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_session(process)
            exit_code = process.returncode if process.returncode is not None else 124
    surviving_pids = _wait_for_session_exit(session_id)
    return {
        "command": command,
        "cwd": str(cwd),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "exit_code": int(exit_code),
        "timed_out": timed_out,
        "session_id": int(session_id),
        "surviving_session_pids": surviving_pids,
        "log": log_path.name,
        "log_sha256": _sha256_file(log_path),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--command-timeout-seconds", type=float, default=300.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise SystemExit(f"refusing to overwrite existing report directory: {report_dir}")
    report_dir.mkdir(parents=True)

    relative_paths = _candidate_files()
    source_digest = _snapshot_digest(REPO_ROOT, relative_paths)
    git_revision = _git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    git_status_sha256 = hashlib.sha256(
        _git_bytes("status", "--porcelain=v1", "-uall")
    ).hexdigest()
    run_reports: list[dict[str, object]] = []
    overall_pass = Path("/proc").is_dir()

    for run_index in range(2):
        with tempfile.TemporaryDirectory(prefix=f"clusterlens-clean-source-{run_index + 1}-") as temp:
            temp_root = Path(temp)
            snapshot_root = temp_root / "candidate"
            outside_cwd = temp_root / "outside-workdir"
            outside_cwd.mkdir()
            _materialize_source(snapshot_root, relative_paths)
            before_digest = _snapshot_digest(snapshot_root, relative_paths)
            environment = dict(os.environ)
            environment.update(
                {
                    "UV_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_DATASETS_OFFLINE": "1",
                }
            )
            environment.pop("PYTHONPATH", None)
            environment.pop("PYTEST_ADDOPTS", None)

            commands = (
                ("build", ["bash", str(snapshot_root / "scripts" / "build.sh")]),
                (
                    "launch_check",
                    ["bash", str(snapshot_root / "scripts" / "run.sh"), "--check-launch"],
                ),
                ("test", ["bash", str(snapshot_root / "scripts" / "test.sh")]),
            )
            command_reports: list[dict[str, object]] = []
            for command_name, command in commands:
                command_report = _run_command(
                    command,
                    cwd=outside_cwd,
                    environment=environment,
                    log_path=report_dir / f"run-{run_index + 1}-{command_name}.log",
                    timeout_s=float(args.command_timeout_seconds),
                )
                command_report["name"] = command_name
                command_reports.append(command_report)
                if (
                    command_report["exit_code"] != 0
                    or command_report["timed_out"]
                    or command_report["surviving_session_pids"]
                ):
                    overall_pass = False
                    break

            after_digest = _snapshot_digest(snapshot_root, relative_paths)
            venv_python = snapshot_root / ".venv" / "bin" / "python"
            run_report = {
                "run": run_index + 1,
                "outside_workdir": str(outside_cwd),
                "source_digest_before": before_digest,
                "source_digest_after": after_digest,
                "source_unchanged": before_digest == after_digest == source_digest,
                "venv_python_present": venv_python.exists(),
                "commands": command_reports,
            }
            run_reports.append(run_report)
            if not run_report["source_unchanged"] or not run_report["venv_python_present"]:
                overall_pass = False

    report = {
        "report_version": REPORT_VERSION,
        "operation": "clean_source_offline_reproducibility",
        "status": "PASS" if overall_pass else "FAIL",
        "candidate": {
            "git_revision": git_revision,
            "git_status_sha256": git_status_sha256,
            "file_count": len(relative_paths),
            "source_digest": source_digest,
            "includes_non_ignored_untracked_files": True,
            "excludes_git_metadata_and_ignored_machine_state": True,
        },
        "policy": {
            "uv_offline": True,
            "frozen_lockfile": True,
            "fresh_virtual_environment_per_run": True,
            "fresh_settings_runtime_and_pycache_from_canonical_test_script": True,
            "procfs_process_session_audit": Path("/proc").is_dir(),
        },
        "runs": run_reports,
        "limitations": [
            "the candidate is an exported dirty-worktree snapshot, not a committed Git revision",
            "offline resolution uses the host's pre-populated uv cache",
            "package/installer artifact reproducibility is a separate UX-34 gate",
        ],
    }
    report_path = report_dir / "clean_source.json"
    report_path.write_text(
        f"{json.dumps(report, indent=2, sort_keys=True)}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
