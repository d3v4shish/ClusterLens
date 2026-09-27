from pathlib import Path
import signal

import pytest

from scripts import verify_durable_fault_matrix


def test_fault_verifier_collects_every_manifest_test_once() -> None:
    manifest = verify_durable_fault_matrix._load_manifest()
    nodes = verify_durable_fault_matrix._test_nodes(manifest)

    assert nodes == tuple(sorted(set(nodes)))
    assert nodes
    assert "tests/test_durable_operation_manifest.py" in nodes
    assert any("test_face_index_batch_is_atomic" in node for node in nodes)
    assert any("test_relocation_recovers_process_exit" in node for node in nodes)


def test_fault_verifier_summary_uses_terminal_pytest_result() -> None:
    output = "progress\n17 passed, 2 warnings in 1.25s\n"
    assert verify_durable_fault_matrix._pytest_summary(output) == "17 passed, 2 warnings in 1.25s"


def test_fault_verifier_report_must_be_outside_repository(tmp_path: Path) -> None:
    inside = verify_durable_fault_matrix.REPO_ROOT / "ignored-durable-fault-report"
    with pytest.raises(ValueError, match="outside the repository"):
        verify_durable_fault_matrix._new_report_dir(inside)


def test_fault_verifier_requires_every_actual_killed_process_case() -> None:
    manifest = {
        "actual_process_exit_cases": [
            {
                "id": "operation:checkpoint:actual_sigkill",
                "operation": "operation",
                "checkpoint": "checkpoint",
            }
        ]
    }
    record = {
        "evidence_id": "operation:checkpoint:actual_sigkill",
        "operation": "operation",
        "boundary": "checkpoint",
        "fault": "actual_sigkill",
        "signal": "SIGKILL",
        "child_exit_code": -int(signal.SIGKILL),
        "pre_state_sha256": "a" * 64,
        "crash_state_sha256": "b" * 64,
        "recovered_state_sha256": "c" * 64,
        "recovery_attempts": 2,
        "result": "PASS",
    }

    assert all(verify_durable_fault_matrix._kill_evidence_checks(manifest, [record]).values())
    missing = verify_durable_fault_matrix._kill_evidence_checks(manifest, [])
    assert missing["killed_process_evidence_complete"] is False
    assert missing["killed_process_state_digests_valid"] is False


def test_fault_verifier_rejects_duplicate_or_simulated_kill_evidence() -> None:
    manifest = {
        "actual_process_exit_cases": [
            {"id": "operation:checkpoint:actual_sigkill", "operation": "operation", "checkpoint": "checkpoint"}
        ]
    }
    record = {
        "evidence_id": "operation:checkpoint:actual_sigkill",
        "operation": "operation",
        "boundary": "checkpoint",
        "fault": "simulated_exception",
        "signal": "SIGKILL",
        "child_exit_code": 1,
        "pre_state_sha256": "a" * 64,
        "crash_state_sha256": "b" * 64,
        "recovered_state_sha256": "c" * 64,
        "recovery_attempts": 1,
        "result": "PASS",
    }

    checks = verify_durable_fault_matrix._kill_evidence_checks(manifest, [record, record])
    assert checks["killed_process_evidence_unique"] is False
    assert checks["killed_process_cases_valid"] is False
