from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from scripts import verify_mixed_ui_lifecycle


def test_session_inventory_contains_current_process_on_linux() -> None:
    if not Path("/proc").is_dir():
        pytest.skip("Linux /proc is required for process-session inventory")
    session_id = os.getsid(0)
    assert os.getpid() in verify_mixed_ui_lifecycle._session_pids(session_id)


def test_verifier_refuses_an_existing_report_directory(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        verify_mixed_ui_lifecycle.main(["--report-dir", str(existing)])


def test_child_uses_current_locked_interpreter_and_isolated_offline_environment(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    mixed_report = report_dir / "mixed.json"

    command = verify_mixed_ui_lifecycle._child_command()
    environment = verify_mixed_ui_lifecycle._child_environment(report_dir, mixed_report)

    assert command[:4] == [sys.executable, "-m", "pytest", "-p"]
    assert "uv" not in command
    assert environment["IMAGE_CLUSTERING_APP_DIR"] == str(report_dir / "test-runtime")
    assert environment["XDG_CONFIG_HOME"] == str(report_dir / "settings")
    assert environment["CLUSTERLENS_UX31_REPORT"] == str(mixed_report)
    assert environment["CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert str(verify_mixed_ui_lifecycle.REPO_ROOT / "tests" / "network_guard") in environment[
        "PYTHONPATH"
    ].split(os.pathsep)
