from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_work_conflicts.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_work_conflicts", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_conflict_matrix_is_deterministic_and_complete() -> None:
    verifier = _load_module()

    first = verifier.run_matrix(seed=17)
    second = verifier.run_matrix(seed=17)

    assert first["status"] == "PASS"
    assert first["result_digest"] == second["result_digest"]
    assert first["coverage"] == {
        "conflict_pairs": 6,
        "launch_orders": 2,
        "normal_running_cancel_failure_retry_scenarios": 12,
        "queued_cancel_retry_scenarios": 10,
        "timing_based_synchronization": False,
    }
    assert len(first["results"]) == 22
    assert all(result["status"] == "PASS" for result in first["results"])
    assert all(
        result["resource_counts_after"] == {"queued": 0, "running": 0}
        for result in first["results"]
    )


def test_conflict_matrix_cli_writes_report_and_refuses_overwrite(tmp_path: Path) -> None:
    report_path = tmp_path / "work-conflicts.json"
    environment = os.environ.copy()
    environment["QT_QPA_PLATFORM"] = "offscreen"
    environment["PYTHONPATH"] = os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT)))
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--seed",
        "17",
        "--report",
        str(report_path),
    ]

    completed = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    repeated = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == report
    assert report["status"] == "PASS"
    assert repeated.returncode == 2
    assert "Refusing to overwrite" in repeated.stderr
