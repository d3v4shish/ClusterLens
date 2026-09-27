from pathlib import Path

import pytest

from scripts import verify_test_isolation


def test_pytest_summary_uses_the_terminal_result_line() -> None:
    output = "progress\n3 passed, 1 warning in 1.25s\n"
    assert verify_test_isolation._pytest_summary(output) == "3 passed, 1 warning in 1.25s"


def test_input_hashes_are_stable_and_cover_declared_inputs() -> None:
    first = verify_test_isolation._input_hashes(verify_test_isolation.FIXTURE_INPUTS)
    second = verify_test_isolation._input_hashes(verify_test_isolation.FIXTURE_INPUTS)
    assert first == second
    assert set(first) == set(verify_test_isolation.FIXTURE_INPUTS)
    assert all(len(digest) == 64 for digest in first.values())


def test_required_seeded_fixture_contracts_are_present() -> None:
    presence = verify_test_isolation._fixture_contract_presence()
    assert set(presence) == set(verify_test_isolation.FIXTURE_CONTRACT_TESTS)
    assert all(presence.values())


def test_isolation_report_must_be_outside_repository(tmp_path: Path) -> None:
    inside = verify_test_isolation.REPO_ROOT / "ignored-isolation-report"
    with pytest.raises(ValueError, match="outside the repository"):
        verify_test_isolation.main(["--report-dir", str(inside), "--outside-workdir", str(tmp_path)])
