"""Run the audited UI modules in isolated orders and write reproducible evidence.

The verifier runs the maintained audit modules once from the repository and
once from an outside working directory in reverse order.  Each child receives
empty, report-owned runtime/data/model paths and offline model flags.  The
normal test launcher provides a fresh Qt settings directory for each run.
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
AUDIT_MODULES = (
    "tests/test_theme_system.py",
    "tests/test_work_coordinator.py",
    "tests/test_production_support.py",
    "tests/test_ui_ux_acceptance.py",
    "tests/test_ui_smoke.py",
)
FIXTURE_INPUTS = (
    "tests/test_ui_smoke.py",
    "tests/test_ui_ux_acceptance.py",
    "tests/test_production_support.py",
    "scripts/create_release_fixture.py",
)
MODEL_INPUTS = (
    "packaging/production_release_manifest.json",
    "pyproject.toml",
    "uv.lock",
)
REPORT_VERSION = "1"
FIXTURE_SEED = 20260913
FIXTURE_CONTRACT_TESTS = (
    "test_audit_people_fixture_contains_seeded_named_unnamed_ignored_and_pending_rows",
    "test_all_faces_tab_is_the_default_faces_home_and_loads_album_groups",
    "test_pending_face_review_preview_threshold_accept_and_undo",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(*args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True,
        capture_output=True,
    ).stdout


def _source_snapshot() -> dict[str, object]:
    listed = _git_output("ls-files", "-z", "--cached", "--others", "--exclude-standard")
    paths = sorted(part.decode("utf-8", errors="surrogateescape") for part in listed.split(b"\0") if part)
    aggregate = hashlib.sha256()
    file_count = 0
    for relative in paths:
        path = REPO_ROOT / relative
        if path.is_symlink():
            kind = "symlink"
            digest = _sha256_bytes(os.readlink(path).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            kind = "file"
            digest = _sha256_file(path)
        else:
            continue
        file_count += 1
        aggregate.update(relative.encode("utf-8", errors="surrogateescape"))
        aggregate.update(b"\0")
        aggregate.update(kind.encode("ascii"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    status = _git_output("status", "--porcelain=v1", "-uall")
    diff = _git_output("diff", "--binary", "--", ".")
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "file_count": file_count,
        "status_sha256": _sha256_bytes(status),
        "diff_sha256": _sha256_bytes(diff),
    }


def _input_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in relative_paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"required isolation input is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return hashes


def _runtime_metadata() -> dict[str, str]:
    try:
        from PyQt6.QtCore import PYQT_VERSION_STR, qVersion

        qt_version = str(qVersion())
        pyqt_version = str(PYQT_VERSION_STR)
    except Exception as exc:
        qt_version = f"unavailable: {exc}"
        pyqt_version = f"unavailable: {exc}"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "qt": qt_version,
        "pyqt": pyqt_version,
    }


def _fixture_contract_presence() -> dict[str, bool]:
    source = (REPO_ROOT / "tests" / "test_ui_smoke.py").read_text(encoding="utf-8")
    return {name: f"def {name}(" in source for name in FIXTURE_CONTRACT_TESTS}


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


def _pytest_summary(output: str) -> str:
    for line in reversed(str(output).splitlines()):
        text = line.strip().strip("=").strip()
        if re.search(r"\b(?:passed|failed|error|errors)\b", text):
            return text
    return "pytest summary not found"


def _isolated_environment(run_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    poison_paths = {
        "CLUSTERLENS_RUNTIME_ROOT": run_dir / "poison-runtime",
        "IMAGE_CLUSTERING_APP_DIR": run_dir / "poison-image-app-dir",
        "XDG_DATA_HOME": run_dir / "xdg-data",
        "HF_HOME": run_dir / "huggingface",
        "TORCH_HOME": run_dir / "torch",
    }
    for path in poison_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({key: str(path) for key, path in poison_paths.items()})
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTHONPATH", None)
    declared = {key: environment[key] for key in (*poison_paths.keys(), "QT_QPA_PLATFORM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
    return environment, declared


def _terminate_owned_session(process: subprocess.Popen[bytes]) -> None:
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


def _run_suite(
    *,
    name: str,
    cwd: Path,
    modules: tuple[str, ...],
    report_dir: Path,
    timeout_s: float,
) -> dict[str, object]:
    run_dir = report_dir / name
    run_dir.mkdir(parents=True)
    environment, declared_environment = _isolated_environment(run_dir)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    command = ["bash", str(REPO_ROOT / "scripts" / "test.sh"), "-q", *modules]
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            exit_code = int(process.wait(timeout=max(1.0, float(timeout_s))))
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_owned_session(process)
            exit_code = int(process.returncode if process.returncode is not None else -signal.SIGKILL)
    surviving_pids = _wait_for_session_exit(process.pid)
    stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": name,
        "cwd": str(cwd),
        "command": command,
        "module_order": list(modules),
        "environment": declared_environment,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_s": round(time.monotonic() - started, 3),
        "pytest_summary": _pytest_summary(f"{stdout_text}\n{stderr_text}"),
        "stdout": str(stdout_path),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr": str(stderr_path),
        "stderr_sha256": _sha256_file(stderr_path),
        "surviving_session_pids": surviving_pids,
        "process_group_drained": not surviving_pids,
    }


def _write_markdown(payload: dict[str, object], path: Path) -> None:
    lines = [
        "# Test isolation verification",
        "",
        f"- Validation: **{payload['validation']}**",
        f"- Revision: `{payload['revision']}`",
        f"- Source integrity: `{payload['source_integrity']}`",
        f"- Fixture seed: `{payload['fixture_seed']}`",
        "",
        "## Runs",
        "",
    ]
    for run in payload["runs"]:
        lines.extend(
            [
                f"- **{run['name']}** — `{run['pytest_summary']}`; exit `{run['exit_code']}`; "
                f"process group drained: `{run['process_group_drained']}`; cwd: `{run['cwd']}`",
            ]
        )
    lines.extend(
        [
            "",
            "Synthetic fixture definitions and model/dependency manifests are identified by SHA-256 in the JSON report.",
            "No private media or mutable developer model cache is an input to these runs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True, help="A new output directory outside the repository.")
    parser.add_argument("--outside-workdir", type=Path, default=Path("/tmp"))
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    outside_workdir = args.outside_workdir.expanduser().resolve()
    if report_dir == REPO_ROOT or REPO_ROOT in report_dir.parents:
        raise ValueError("isolation evidence directory must be outside the repository")
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite isolation evidence: {report_dir}")
    if not outside_workdir.is_dir():
        raise NotADirectoryError(f"outside working directory does not exist: {outside_workdir}")
    if outside_workdir == REPO_ROOT or REPO_ROOT in outside_workdir.parents:
        raise ValueError("outside working directory must not be inside the repository")
    report_dir.mkdir(parents=True)

    before = _source_snapshot()
    revision = _git_output("rev-parse", "HEAD").decode("ascii").strip()
    runs = [
        _run_suite(
            name="repository-order",
            cwd=REPO_ROOT,
            modules=AUDIT_MODULES,
            report_dir=report_dir,
            timeout_s=args.timeout_seconds,
        ),
        _run_suite(
            name="outside-reverse-order",
            cwd=outside_workdir,
            modules=tuple(reversed(AUDIT_MODULES)),
            report_dir=report_dir,
            timeout_s=args.timeout_seconds,
        ),
    ]
    after = _source_snapshot()
    source_integrity = "PASS" if before == after else "FAIL"
    fixture_contracts = _fixture_contract_presence()
    checks = {
        "all_runs_passed": all(int(run["exit_code"]) == 0 and not bool(run["timed_out"]) for run in runs),
        "all_process_groups_drained": all(bool(run["process_group_drained"]) for run in runs),
        "source_tree_unchanged": before == after,
        "working_directories_differ": runs[0]["cwd"] != runs[1]["cwd"],
        "module_orders_differ": runs[0]["module_order"] == list(reversed(runs[1]["module_order"])),
        "fixture_contracts_present": all(fixture_contracts.values()),
    }
    payload: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "validation": "PASS" if all(checks.values()) else "FAIL",
        "revision": revision,
        "runtime": _runtime_metadata(),
        "fixture_seed": FIXTURE_SEED,
        "synthetic_coverage": True,
        "private_source_media_used": False,
        "network_policy": "model/download libraries forced offline; tests use local fakes and generated fixtures",
        "fixture_input_sha256": _input_hashes(FIXTURE_INPUTS),
        "fixture_contract_tests": fixture_contracts,
        "model_and_dependency_input_sha256": _input_hashes(MODEL_INPUTS),
        "source_before": before,
        "source_after": after,
        "source_integrity": source_integrity,
        "checks": checks,
        "runs": runs,
    }
    json_path = report_dir / "test_isolation.json"
    markdown_path = report_dir / "test_isolation.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(payload, markdown_path)
    print(json.dumps({"validation": payload["validation"], "report": str(json_path)}, indent=2))
    return 0 if payload["validation"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
