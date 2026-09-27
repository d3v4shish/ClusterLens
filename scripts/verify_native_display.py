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

from PyQt6.QtCore import QCoreApplication, QSettings, Qt
from PyQt6.QtWidgets import QWidget


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
HEADLESS_QT_PLATFORMS = frozenset({"offscreen", "minimal", "linuxfb", "vnc"})
REQUIRED_MAIN_CONTROLS = (
    "library_workspace_button",
    "organize_workspace_button",
    "people_workspace_button",
    "tools_workspace_button",
    "settings_button",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True, help="New ignored directory for JSON and app-only screenshots.")
    parser.add_argument("--logical-size", default="1920x1080", help="Requested logical window size, WIDTHxHEIGHT.")
    parser.add_argument(
        "--text-scale",
        type=int,
        choices=(100, 125, 150, 200),
        default=100,
        help="Persisted ClusterLens text scale to qualify independently of display scaling.",
    )
    parser.add_argument(
        "--require-fractional-scale",
        action="store_true",
        help="Fail unless Qt reports a non-integer device-pixel ratio for the active native display.",
    )
    parser.add_argument(
        "--require-multiple-screens",
        action="store_true",
        help="Move the real window across at least two fitting Qt screens and fail unless every move is observed.",
    )
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
            # Layout verification does not need startup cache reconciliation;
            # avoid unrelated workers while capturing and closing the window.
            "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE": "native-display-layout",
        }
    )
    for import_path in (REPO_ROOT, SRC_ROOT):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from apps.pyqt_production.app import ProductionClusterApp
    from apps.pyqt_production.bootstrap import bootstrap_runtime
    from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG
    from scripts.verify_ui_redesign import (
        _clipped_controls,
        _close_widget,
        _coverage_validation,
        _focus_evidence,
        _palette_contrast_evidence,
        _settle_until as _settle_until_events,
    )
    from ui.list_models import ListEntry

    app = QApplication.instance() or QApplication([])
    platform_name = str(app.platformName() or "").strip().casefold()
    if platform_name in HEADLESS_QT_PLATFORMS:
        raise RuntimeError(f"native verification refuses Qt platform {platform_name!r}")
    screens = list(app.screens())
    if not screens:
        raise RuntimeError("native verification found no primary Qt screen")
    screen = next(
        (
            candidate
            for candidate in screens
            if candidate.availableGeometry().width() >= width
            and candidate.availableGeometry().height() >= height
        ),
        None,
    )
    if screen is None:
        raise RuntimeError("active display cannot fit the requested logical window size")
    fitting_screens = [
        candidate
        for candidate in screens
        if candidate.availableGeometry().width() >= width
        and candidate.availableGeometry().height() >= height
    ]
    if args.require_multiple_screens and len(fitting_screens) < 2:
        raise RuntimeError("native multi-monitor verification requires at least two screens that fit the requested size")
    settings = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
    fixture_media = runtime_root / "native-fixture-media"
    fixture_media.mkdir(parents=True, exist_ok=True)
    fixture_paths: list[str] = []
    for index, color in enumerate((0x446688, 0x885544, 0x557744), start=1):
        path = fixture_media / f"person-{index}.png"
        image = QImage(96, 96, QImage.Format.Format_RGB32)
        image.fill(color)
        image.save(str(path), "PNG")
        fixture_paths.append(str(path))
    settings.setValue("setup/completed", True)
    settings.setValue("workspace/active_roots", json.dumps([str(fixture_media)]))
    settings.setValue("appearance/text_scale", args.text_scale)
    settings.sync()
    layout, _ = bootstrap_runtime()
    window = ProductionClusterApp(layout)
    dialog = None
    try:
        window.apply_screen_geometry_constraints((width, height))
        window.winId()
        if window.windowHandle() is not None:
            window.windowHandle().setScreen(screen)
        window.resize(width, height)
        available = screen.availableGeometry()
        window.move(available.left(), available.top())
        window.show()
        window.raise_()
        window.activateWindow()
        if window.windowHandle() is not None:
            window.windowHandle().requestActivate()
        _settle(app, args.settle_seconds)
        if window.screen() is not screen and window.windowHandle() is not None:
            window.windowHandle().setScreen(screen)
            window.move(available.left(), available.top())
            window.resize(width, height)
            _settle(app, args.settle_seconds)
        _settle_until_events(app, lambda: window.job_manager.active_count() == 0)
        screen_transitions: list[dict[str, object]] = []
        transition_targets = fitting_screens if args.require_multiple_screens else [screen]
        for index, target_screen in enumerate(transition_targets):
            target_available = target_screen.availableGeometry()
            if window.windowHandle() is not None:
                window.windowHandle().setScreen(target_screen)
            window.move(target_available.left(), target_available.top())
            window.resize(width, height)
            window.show()
            window.raise_()
            window.activateWindow()
            if window.windowHandle() is not None:
                window.windowHandle().requestActivate()
            _settle(app, max(args.settle_seconds, 0.1))
            actual_screen = window.screen()
            transition_path = report_dir / f"native_screen_{index}.png"
            screen_transitions.append(
                {
                    "requested_screen": str(target_screen.name()),
                    "actual_screen": str(actual_screen.name()) if actual_screen is not None else "",
                    "requested_geometry": _rect_payload(target_available),
                    "client_geometry": _rect_payload(window.geometry()),
                    "observed_target": actual_screen is target_screen,
                    "contained": bool(
                        actual_screen is target_screen
                        and _rect_contained(window.geometry(), target_available)
                    ),
                    "screenshot": str(transition_path),
                    "saved": bool(window.grab().save(str(transition_path), "PNG")),
                }
            )
        if window.windowHandle() is not None:
            window.windowHandle().setScreen(screen)
        window.move(available.left(), available.top())
        window.resize(width, height)
        _settle(app, max(args.settle_seconds, 0.1))
        window_available = (window.screen() or screen).availableGeometry()
        window.library_workspace_button.setFocus(Qt.FocusReason.TabFocusReason)
        _settle(app, 0.02)
        main_geometry = window.geometry()
        main_screen_geometry = window_available
        checks: dict[str, bool] = {
            "native_qt_platform": platform_name not in HEADLESS_QT_PLATFORMS,
            "main_window_visible": bool(window.isVisible()),
            # Frame geometry includes compositor-owned title bars/shadows.
            # Wayland can also choose the target output after show(). Validate
            # the client surface against that actual output.
            "main_window_on_screen": _rect_contained(window.geometry(), window_available),
            "main_window_screenshot": bool(window.grab().save(str(main_screenshot), "PNG")),
            "requested_client_size": window.width() >= width and window.height() >= height,
            "main_window_no_clipped_copy": not _clipped_controls(window),
            "library_workspace_keyboard_focus": bool(window.library_workspace_button.hasFocus()),
            "workspace_controls_have_accessible_labels": all(
                bool(str(getattr(window, attribute).accessibleName() or getattr(window, attribute).text() or "").strip())
                for attribute in REQUIRED_MAIN_CONTROLS
            ),
            "requested_text_scale_applied": int(window.theme_manager.text_scale) == args.text_scale,
            "supported_window_minimum": window.minimumWidth() <= width and window.minimumHeight() <= height,
        }
        for attribute in REQUIRED_MAIN_CONTROLS:
            checks[f"{attribute}_visible"] = _visible(getattr(window, attribute, None), window)
            control = getattr(window, attribute, None)
            checks[f"{attribute}_text_height"] = bool(
                control is not None and control.height() >= control.fontMetrics().height() + 8
            )

        required_routes = {
            "library": ("timeline", "all_photos", "search", "albums"),
            "organize": ("photos", "tags", "context"),
            "people": ("people", "all_faces", "unnamed", "detect", "review", "review_name", "find", "identities"),
            "tools": ("duplicates", "rename", "recovery"),
        }
        window._build_faces_workspace_widget()
        window._build_names_workspace_widget()

        def seed_people_fixture() -> None:
            if window.faces_pane is not None:
                pane = window.faces_pane
                pane.face_service_provider = None
                pane.face_identity_model.set_source_items(
                    [
                        ListEntry("Alice", subtitle="2 prototype faces", payload="Alice"),
                        ListEntry("Bob", subtitle="1 prototype face", payload="Bob"),
                    ]
                )
                pane.face_pending_model.set_items(
                    [
                        ListEntry("Alice · 91%", subtitle="Pending proposal", payload=(fixture_paths[0], 0)),
                        ListEntry("Bob · 84%", subtitle="Pending proposal", payload=(fixture_paths[1], 0)),
                    ]
                )
                pane.face_results_groups_model.set_items(
                    [ListEntry("Group 1", subtitle="2 faces", payload=0)]
                )
            if window.names_pane is not None:
                window.names_pane.names_model.set_source_items(
                    [
                        ListEntry("Alice", subtitle="2 photos", payload="Alice"),
                        ListEntry("Bob", subtitle="1 photo", payload="Bob"),
                    ]
                )
                window.names_pane._update_count_label()

        seed_people_fixture()
        fixture_counts = {
            "generated_photos": len(fixture_paths),
            "identities": 2,
            "pending_proposals": 2,
            "face_groups": 1,
        }
        route_evidence: dict[str, object] = {}
        for group, sections in required_routes.items():
            for section in sections:
                if group == "library":
                    window.set_active_workspace("library")
                    window.library_pane.set_surface_group("library", section)
                elif group == "organize":
                    window.set_active_workspace(
                        {"photos": "clustering", "tags": "tags", "context": "cluster_context"}[section]
                    )
                elif group == "people":
                    window._pending_people_section = section
                    if section == "people":
                        window.set_active_workspace("names")
                    elif section == "review":
                        window.set_active_workspace("people_cleanup")
                    else:
                        window.set_active_workspace("faces")
                        window._apply_people_section()
                    if section == "review" and window.faces_pane is not None:
                        window.faces_pane.face_pending_items_toggle.setChecked(True)
                else:
                    window.set_active_workspace("tools")
                    window.library_pane.set_surface_group("tools", section)
                _settle(app, args.settle_seconds)
                background_idle = _settle_until_events(app, lambda: window.job_manager.active_count() == 0)
                if group == "people":
                    seed_people_fixture()
                    if window.faces_pane is not None:
                        window.faces_pane.sidebar_scroll.verticalScrollBar().setValue(0)
                    _settle(app, args.settle_seconds)
                route_name = f"{group}.{section}"
                route_path = report_dir / f"native_{group}_{section}.png"
                saved = bool(window.grab().save(str(route_path), "PNG"))
                clipped_controls = _clipped_controls(window)
                focus = _focus_evidence(window)
                route_evidence[route_name] = {
                    "saved": saved,
                    "screenshot": str(route_path),
                    "clipped_controls": clipped_controls,
                    "background_idle": background_idle,
                    "empty_scope_visible": window.workspace_stack.currentWidget() is window.empty_scope_panel,
                    **focus,
                }
        required_route_names = [
            f"{group}.{section}"
            for group, sections in required_routes.items()
            for section in sections
        ]
        route_coverage = _coverage_validation(required_route_names, list(route_evidence))
        checks["route_coverage"] = bool(route_coverage["valid"])
        checks["routes_no_clipped_copy"] = all(not route["clipped_controls"] for route in route_evidence.values())
        checks["routes_keyboard_coverage"] = all(
            bool(route["focus_validation"]["valid"]) for route in route_evidence.values()
        )
        checks["route_screenshots"] = all(bool(route["saved"]) for route in route_evidence.values())
        checks["route_background_idle"] = all(bool(route["background_idle"]) for route in route_evidence.values())
        checks["route_content_visible"] = all(
            not bool(route["empty_scope_visible"]) for route in route_evidence.values()
        )
        checks["populated_fixture"] = all(int(value) > 0 for value in fixture_counts.values())
        root_states: dict[str, object] = {}
        window._edit_active_roots()
        for index, state in enumerate(("sources", "catalog")):
            window.source_pane.tabs.setCurrentIndex(index)
            _settle(app, args.settle_seconds)
            root_path = report_dir / f"native_roots_{state}.png"
            root_states[state] = {
                "saved": bool(window.grab().save(str(root_path), "PNG")),
                "screenshot": str(root_path),
                "clipped_controls": _clipped_controls(window),
                **_focus_evidence(window),
            }
        window._hide_source_pane()
        checks["roots_no_clipped_copy"] = all(not state["clipped_controls"] for state in root_states.values())
        checks["roots_keyboard_coverage"] = all(
            bool(state["focus_validation"]["valid"]) for state in root_states.values()
        )
        checks["roots_screenshots"] = all(bool(state["saved"]) for state in root_states.values())

        device_pixel_ratio = float(screen.devicePixelRatio())
        if args.require_fractional_scale:
            checks["fractional_display_scale"] = abs(device_pixel_ratio - round(device_pixel_ratio)) > 0.01
        if args.require_multiple_screens:
            checks["multi_monitor_movement"] = bool(
                len(screen_transitions) >= 2
                and all(
                    bool(transition["observed_target"])
                    and bool(transition["contained"])
                    and bool(transition["saved"])
                    for transition in screen_transitions
                )
            )

        dialog = window._create_settings_dialog(parent=window)
        dialog.winId()
        if dialog.windowHandle() is not None:
            dialog.windowHandle().setScreen(screen)
        dialog.resize(min(width, 1080), min(height, 760))
        dialog.move(available.left(), available.top())
        dialog.show()
        _settle(app, args.settle_seconds)
        dialog_available = (dialog.screen() or window.screen() or screen).availableGeometry()
        settings_geometry = dialog.geometry()
        settings_screen_geometry = dialog_available
        tab_names = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
        required_settings_sections = [
            "General",
            "Sources & Library",
            "Performance",
            "Clustering Models",
            "Face Models",
            "Storage",
            "Safety & Recovery",
            "Updates",
            "Support",
            "About",
        ]
        settings_sections: dict[str, object] = {}
        for index, section in enumerate(required_settings_sections):
            if not dialog.select_section(section):
                continue
            _settle(app, args.settle_seconds)
            section_path = report_dir / f"native_settings_{index}.png"
            settings_sections[section] = {
                "saved": bool(dialog.grab().save(str(section_path), "PNG")),
                "screenshot": str(section_path),
                "clipped_controls": _clipped_controls(dialog),
                **_focus_evidence(dialog),
            }
        settings_coverage = _coverage_validation(required_settings_sections, list(settings_sections))
        checks.update(
            {
                "settings_dialog_visible": bool(dialog.isVisible()),
                "settings_dialog_on_screen": _rect_contained(dialog.geometry(), dialog_available),
                "settings_dialog_no_clipped_copy": not _clipped_controls(dialog),
                "settings_storage_tab": "Storage" in tab_names and dialog.select_section("Storage"),
                "settings_generated_storage_controls": all(
                    getattr(dialog, attribute, None) is not None
                    for attribute in ("clear_runtime_temp_button", "clear_runtime_reports_button", "clear_model_assets_button")
                ),
            "settings_dialog_screenshot": bool(dialog.grab().save(str(settings_screenshot), "PNG")),
            "settings_section_coverage": bool(settings_coverage["valid"]),
            "settings_sections_no_clipped_copy": all(
                not section["clipped_controls"] for section in settings_sections.values()
            ),
            "settings_keyboard_coverage": all(
                bool(section["focus_validation"]["valid"]) for section in settings_sections.values()
            ),
            "settings_section_screenshots": all(bool(section["saved"]) for section in settings_sections.values()),
            "theme_text_contrast": bool(_palette_contrast_evidence(app)["valid"]),
            }
        )
        shutdown_checks = [_close_widget(app, dialog), _close_widget(app, window)]
        checks["clean_shutdown"] = all(shutdown_checks)
        payload = {
            "report_version": "3",
            "scenario": "native-display-layout",
            "validation": "PASS" if all(checks.values()) else "FAIL",
            "runtime_root": str(layout.root),
            "requested_logical_size": {"width": width, "height": height},
            "requested_text_scale": args.text_scale,
            "display_metrics": {
                "device_pixel_ratio": device_pixel_ratio,
                "logical_dpi_x": float(screen.logicalDotsPerInchX()),
                "logical_dpi_y": float(screen.logicalDotsPerInchY()),
                "physical_dpi_x": float(screen.physicalDotsPerInchX()),
                "physical_dpi_y": float(screen.physicalDotsPerInchY()),
            },
            "screen_transitions": screen_transitions,
            "client_geometries": {
                "main_window": _rect_payload(main_geometry),
                "settings_dialog": _rect_payload(settings_geometry),
                "requested_screen": _rect_payload(available),
                "main_window_screen": _rect_payload(main_screen_geometry),
                "settings_dialog_screen": _rect_payload(settings_screen_geometry),
            },
            "platform": platform_name,
            "settings_tabs": tab_names,
            "routes": route_evidence,
            "route_coverage": route_coverage,
            "fixture_counts": fixture_counts,
            "root_states": root_states,
            "settings_sections": settings_sections,
            "settings_coverage": settings_coverage,
            "qualification": {
                "visual": {"status": "PASS" if all(checks.values()) else "FAIL"},
                "keyboard": {
                    "status": "PASS" if checks["routes_keyboard_coverage"] and checks["settings_keyboard_coverage"] else "FAIL"
                },
                "screen_reader": {
                    "status": "NOT_RUN",
                    "reason": "A real assistive-technology session is a separate UX-32 release gate.",
                },
            },
            "screenshots": {"main_window": str(main_screenshot), "settings_dialog": str(settings_screenshot)},
            "checks": checks,
            "network": "not used by native display verification",
        }
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"validation": payload["validation"], "report": str(report_path)}, indent=2))
        return 0 if payload["validation"] == "PASS" else 1
    finally:
        if dialog is not None:
            try:
                dialog.close()
            except RuntimeError:
                pass
        try:
            window.close()
        except RuntimeError:
            pass
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
    return bool(available.contains(rect))


def _rect_payload(rect) -> dict[str, int]:
    return {"x": rect.x(), "y": rect.y(), "width": rect.width(), "height": rect.height()}


def _contrast_ratio(foreground: str, background: str) -> float:
    def _luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    first, second = sorted((_luminance(foreground), _luminance(background)), reverse=True)
    return (first + 0.05) / (second + 0.05)


if __name__ == "__main__":
    raise SystemExit(main())
