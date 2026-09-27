from __future__ import annotations

import uuid

from PyQt6.QtCore import QCoreApplication

from apps.pyqt_production.single_instance import SingleInstanceCoordinator


def _process_events_until(predicate, *, timeout_ms: int = 1200) -> bool:
    app = QCoreApplication.instance() or QCoreApplication([])
    for _ in range(timeout_ms // 10):
        app.processEvents()
        if predicate():
            return True
    return bool(predicate())


def test_second_launch_hands_off_activation_to_primary(tmp_path):
    runtime_root = tmp_path / uuid.uuid4().hex
    activations: list[str] = []
    primary = SingleInstanceCoordinator(runtime_root)
    secondary = SingleInstanceCoordinator(runtime_root)
    try:
        assert primary.acquire_or_handoff(lambda: activations.append("activate"))
        assert not secondary.acquire_or_handoff(lambda: activations.append("secondary"))
        assert _process_events_until(lambda: activations == ["activate"])
    finally:
        secondary.close()
        primary.close()


def test_closed_primary_allows_a_later_launch_to_take_ownership(tmp_path):
    runtime_root = tmp_path / uuid.uuid4().hex
    first = SingleInstanceCoordinator(runtime_root)
    later = SingleInstanceCoordinator(runtime_root)
    try:
        assert first.acquire_or_handoff(lambda: None)
        first.close()
        assert later.acquire_or_handoff(lambda: None)
    finally:
        later.close()
        first.close()
