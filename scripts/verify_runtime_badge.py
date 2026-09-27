#!/usr/bin/env python3
"""Capture deterministic dark/light evidence for every runtime badge state.

The verifier uses fixture capabilities only. It opens no production runtime,
model, GPU, storage, source media, database, cache, or network resource.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _settle(app, cycles: int = 4) -> None:
    for _index in range(max(1, int(cycles))):
        app.processEvents()


def _states():
    from infra.runtime import ExecutionPolicy, RuntimeCapabilities

    capabilities = RuntimeCapabilities(torch_version="fixture", onnx_version="fixture")
    return (
        ("checking", "Runtime checking… · Checking", "warning", lambda badge: badge.update_runtime(None, None)),
        (
            "ready",
            "CPU · Ready",
            "success",
            lambda badge: badge.update_runtime(
                capabilities,
                ExecutionPolicy(preferred_mode="cpu", effective_mode="cpu", reason="Fixture CPU runtime is ready."),
            ),
        ),
        (
            "fallback",
            "CPU fallback · Fallback",
            "warning",
            lambda badge: badge.update_runtime(
                capabilities,
                ExecutionPolicy(
                    preferred_mode="auto",
                    effective_mode="cpu",
                    reason="Fixture CUDA runtime is unavailable; using CPU.",
                ),
            ),
        ),
        (
            "unavailable",
            "CUDA unavailable · Unavailable",
            "error",
            lambda badge: badge.update_runtime(
                capabilities,
                ExecutionPolicy(
                    preferred_mode="cuda",
                    effective_mode="cpu",
                    error="Fixture CUDA provider is unavailable.",
                ),
            ),
        ),
        (
            "failed",
            "Runtime failed · Failed",
            "error",
            lambda badge: badge.set_runtime_failure("Fixture readiness check failed."),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite runtime badge evidence directory: {report_dir}")

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    for import_path in (REPO_ROOT / "src", REPO_ROOT):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QWidget
    from ui.runtime_widgets import RuntimeBadge
    from ui.theme import apply_app_theme

    app = QApplication.instance() or QApplication([])
    report_dir.mkdir(parents=True)
    report: dict[str, object] = {
        "fixture": "offscreen fixture capabilities; no production runtime or I/O",
        "themes": {},
    }
    try:
        for theme in ("dark", "light"):
            apply_app_theme(app, theme)
            theme_entries: dict[str, object] = {}
            for state, expected_text, expected_property, apply_state in _states():
                host = QWidget()
                host.setWindowTitle(f"Runtime badge: {state}")
                host.resize(620, 96)
                layout = QHBoxLayout(host)
                layout.addWidget(QLabel("Runtime"))
                badge = RuntimeBadge(host)
                apply_state(badge)
                layout.addWidget(badge, stretch=1)
                host.show()
                _settle(app)
                screenshot = report_dir / f"runtime_badge_{theme}_{state}.png"
                expected_accessible_name = f"Runtime status: {badge.text().rsplit(' · ', 1)[0]}; {state.title()}"
                entry = {
                    "text": badge.text(),
                    "accessible_name": badge.accessibleName(),
                    "accessible_description": badge.accessibleDescription(),
                    "state_property": badge.property("state"),
                    "screenshot": str(screenshot),
                    "saved": bool(host.grab().save(str(screenshot), "PNG")),
                    "matches_contract": (
                        badge.text() == expected_text
                        and badge.property("state") == expected_property
                        and badge.accessibleName() == expected_accessible_name
                    ),
                }
                theme_entries[state] = entry
                host.close()
                host.deleteLater()
                _settle(app)
            report["themes"][theme] = theme_entries
        report["validation"] = "PASS" if all(
            entry["saved"] and entry["matches_contract"]
            for theme_entries in report["themes"].values()
            for entry in theme_entries.values()
        ) else "FAIL"
        report_path = report_dir / "runtime_badge.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"validation": report["validation"], "report": str(report_path)}, indent=2))
        return 0 if report["validation"] == "PASS" else 1
    finally:
        apply_app_theme(app, "dark")
        _settle(app)


if __name__ == "__main__":
    raise SystemExit(main())
