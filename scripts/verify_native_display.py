"""Capture native-display evidence from an isolated ClusterLens runtime.

This verifier rejects Qt headless platforms and captures only ClusterLens
widgets. It never captures the surrounding desktop or uses existing runtime
data. Run it from a real release display or clean VM.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import monotonic, sleep

from PyQt6.QtCore import QCoreApplication, QSettings
from PyQt6.QtWidgets import QWidget


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
HEADLESS_QT_PLATFORMS = frozenset({"offscreen", "minimal", "linuxfb", "vnc"})
REQUIRED_MAIN_CONTROLS = (
    "gallery_workspace_button",
    "clustering_workspace_button",
    "faces_workspace_button",
    "names_workspace_button",
    "tags_workspace_button",
    "settings_button",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True, help="New ignored directory for JSON and app-only screenshots.")
    parser.add_argument("--logical-size", default="1920x1080", help="Requested logical window size, WIDTHxHEIGHT.")
    parser.add_argument("--settle-seconds", type=float, default=0.2, help="Qt event-loop settle time before capture.")
    args = parser.parse_args(argv)
    if args.settle_seconds < 0:
        raise ValueError("settle seconds must be non-negative")
    report_dir = args.report_dir.expanduser().resolve()
    report_path = report_dir / "native_display.json"
    main_screenshot = report_dir / "native_main_window.png"
    settings_screenshot = report_dir / "native_settings_dialog.png"
    _require_new_paths(report_path, main_screenshot, settings_screenshot)
    width, height = _parse_logical_size(args.logical_size)
    runtime_root = _require_empty_directory(report_dir / "runtime")
    report_dir.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        {
            "CLUSTERLENS_RUNTIME_ROOT": str(runtime_root),
            "IMAGE_CLUSTERING_APP_DIR": str(runtime_root),
            "XDG_DATA_HOME": str(runtime_root / "xdg-data"),
            "XDG_CONFIG_HOME": str(runtime_root / "xdg-config"),
        }
    )
    for import_path in (REPO_ROOT, SRC_ROOT):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from PyQt6.QtWidgets import QApplication
    from apps.pyqt_production.app import ProductionClusterApp
    from apps.pyqt_production.bootstrap import bootstrap_runtime
    from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG

    app = QApplication.instance() or QApplication([])
    platform_name = str(app.platformName() or "").strip().casefold()
    if platform_name in HEADLESS_QT_PLATFORMS:
        raise RuntimeError(f"native verification refuses Qt platform {platform_name!r}")
    screen = app.primaryScreen()
    if screen is None:
        raise RuntimeError("native verification found no primary Qt screen")
    if screen.availableGeometry().width() < width or screen.availableGeometry().height() < height:
        raise RuntimeError("active display cannot fit the requested logical window size")
    settings = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
    settings.setValue("setup/completed", True)
    settings.sync()
    layout, _ = bootstrap_runtime()
    window = ProductionClusterApp(layout)
    dialog = None
    try:
        window.resize(width, height)
        available = screen.availableGeometry()
        window.move(available.left(), available.top())
        window.show()
        _settle(app, args.settle_seconds)
        checks: dict[str, bool] = {
            "native_qt_platform": platform_name not in HEADLESS_QT_PLATFORMS,
            "main_window_visible": bool(window.isVisible()),
            "main_window_on_screen": _rect_contained(window.frameGeometry(), available),
            "main_window_screenshot": bool(window.grab().save(str(main_screenshot), "PNG")),
        }
        for attribute in REQUIRED_MAIN_CONTROLS:
            checks[f"{attribute}_visible"] = _visible(getattr(window, attribute, None), window)

        dialog = window._create_settings_dialog(parent=window)
        dialog.resize(min(width, 1080), min(height, 760))
        dialog.show()
        _settle(app, args.settle_seconds)
        tab_names = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
        checks.update(
            {
                "settings_dialog_visible": bool(dialog.isVisible()),
                "settings_dialog_on_screen": _rect_contained(dialog.frameGeometry(), available),
                "settings_storage_tab": "Storage" in tab_names,
                "settings_generated_storage_controls": all(
                    getattr(dialog, attribute, None) is not None
                    for attribute in ("clear_runtime_temp_button", "clear_runtime_reports_button", "clear_model_assets_button")
                ),
                "settings_dialog_screenshot": bool(dialog.grab().save(str(settings_screenshot), "PNG")),
            }
        )
        payload = {
            "report_version": "1",
            "scenario": "native-display-layout",
            "validation": "PASS" if all(checks.values()) else "FAIL",
            "runtime_root": str(layout.root),
            "requested_logical_size": {"width": width, "height": height},
            "platform": platform_name,
            "settings_tabs": tab_names,
            "screenshots": {"main_window": str(main_screenshot), "settings_dialog": str(settings_screenshot)},
            "checks": checks,
            "network": "not used by native display verification",
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"validation": payload["validation"], "report": str(report_path)}, indent=2))
        return 0 if payload["validation"] == "PASS" else 1
    finally:
        if dialog is not None:
            dialog.close()
        window.close()
        app.processEvents()


def _require_new_paths(*paths: Path) -> None:
    for path in paths:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite evidence path: {path}")


def _require_empty_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"native verifier runtime root must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _parse_logical_size(value: str) -> tuple[int, int]:
    parts = str(value or "").casefold().split("x")
    if len(parts) != 2:
        raise ValueError("logical size must use WIDTHxHEIGHT, for example 1920x1080")
    try:
        width, height = (int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError("logical size must contain integer dimensions") from exc
    if width < 1 or height < 1:
        raise ValueError("logical size dimensions must be positive")
    return width, height


def _settle(app, seconds: float) -> None:
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        app.processEvents()
        QCoreApplication.processEvents()
        sleep(0.01)


def _visible(widget: QWidget | None, owner: QWidget) -> bool:
    return bool(widget is not None and widget.isVisible() and widget.window() is owner)


def _rect_contained(rect, available) -> bool:
    return bool(available.contains(rect.center()) and rect.left() >= available.left() and rect.top() >= available.top())


if __name__ == "__main__":
    raise SystemExit(main())
