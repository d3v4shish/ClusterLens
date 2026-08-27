from __future__ import annotations

from pathlib import Path

from apps.pyqt_production.release_gates import _startup_performance_gate


def test_production_startup_stays_lazy_and_within_release_budget(tmp_path: Path) -> None:
    result = _startup_performance_gate(tmp_path)

    assert result.status == "PASS", result.details
    assert (tmp_path / "startup_performance.json").is_file()
