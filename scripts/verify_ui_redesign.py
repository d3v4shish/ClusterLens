"""Create deterministic, app-only UI evidence for the redesigned shell.

The verifier uses an isolated runtime and a generated populated photo root. It
never opens user media, model caches, or existing application settings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import sys
from pathlib import Path
from time import monotonic


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _parse_size(value: str) -> tuple[int, int]:
    parts = str(value or "").casefold().split("x")
    if len(parts) != 2:
        raise ValueError("logical size must use WIDTHxHEIGHT")
    width, height = (int(part.strip()) for part in parts)
    if width < 1280 or height < 720:
        raise ValueError("UI evidence sizes must be at least 1280x720")
    return width, height


def _settle(app, cycles: int = 8) -> None:
    for _index in range(max(1, int(cycles))):
        app.processEvents()


def _settle_until(app, predicate, *, timeout_seconds: float = 5.0) -> bool:
    deadline = monotonic() + max(0.1, float(timeout_seconds))
    while monotonic() < deadline:
        app.processEvents()
        if predicate():
            app.processEvents()
            return True
    return bool(predicate())


def _close_widget(app, widget, *, cycles: int = 300) -> bool:
    for _index in range(max(1, int(cycles))):
        try:
            accepted = bool(widget.close())
        except RuntimeError:
            return True
        app.processEvents()
        if accepted:
            try:
                if not widget.isVisible():
                    return True
            except RuntimeError:
                return True
    return False


def _control_text(widget) -> str:
    from PyQt6.QtWidgets import QAbstractButton, QComboBox, QLineEdit, QTabBar, QTabWidget

    if isinstance(widget, QTabWidget):
        return ", ".join(widget.tabText(index) for index in range(widget.count()))
    if isinstance(widget, QTabBar):
        return ", ".join(widget.tabText(index) for index in range(widget.count()))
    if isinstance(widget, QComboBox):
        return str(widget.currentText() or "")
    if isinstance(widget, QLineEdit):
        return str(widget.placeholderText() or "")
    if isinstance(widget, QAbstractButton):
        return str(widget.text() or "")
    getter = getattr(widget, "text", None)
    if callable(getter):
        return str(getter() or "").replace("\n", " ")
    return ""


def _focus_order(owner) -> list[dict[str, str]]:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QWidget
    from PyQt6.QtTest import QTest

    results: list[dict[str, str]] = []
    focusable = _focusable_controls(owner)
    if not focusable:
        return results
    seen: set[int] = set()
    for seed in focusable:
        if id(seed) in seen:
            continue
        seed.setFocus(Qt.FocusReason.OtherFocusReason)
        QApplication.processEvents()
        widget = QApplication.focusWidget() or owner.focusWidget()
        if widget is not seed:
            continue
        for _index in range(max(16, len(focusable) * 3)):
            if widget is None:
                break
            if widget in _focusable_controls(owner):
                key = id(widget)
                if key in seen:
                    break
                seen.add(key)
                results.append(_focus_entry(widget))
            QTest.keyClick(widget, Qt.Key.Key_Tab)
            QApplication.processEvents()
            next_widget = QApplication.focusWidget() or owner.focusWidget()
            if next_widget is widget and widget not in focusable:
                break
            widget = next_widget
    # A scroll area can expose a different tail of controls after traversal.
    # Confirm those newly painted controls can receive keyboard focus too.
    while True:
        additions = [widget for widget in _focusable_controls(owner) if id(widget) not in seen]
        if not additions:
            break
        added = False
        for widget in additions:
            widget.setFocus(Qt.FocusReason.OtherFocusReason)
            QApplication.processEvents()
            if QApplication.focusWidget() is not widget:
                continue
            seen.add(id(widget))
            results.append(_focus_entry(widget))
            added = True
        if not added:
            break
    return results


def _focusable_controls(owner) -> list[object]:
    """Return the actual visible controls that Qt accepts keyboard focus for."""

    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractButton,
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QTabBar,
        QTextEdit,
    )

    control_types = (
        QAbstractButton,
        QAbstractItemView,
        QComboBox,
        QDoubleSpinBox,
        QLineEdit,
        QPlainTextEdit,
        QSlider,
        QSpinBox,
        QTabBar,
        QTextEdit,
    )

    return [
        widget
        for widget in owner.findChildren(control_types)
        if widget.isVisibleTo(owner)
        and not widget.visibleRegion().isEmpty()
        and widget.isEnabled()
        and widget.focusPolicy() != Qt.FocusPolicy.NoFocus
        and not (
            isinstance(widget, QLineEdit)
            and isinstance(widget.parentWidget(), (QComboBox, QDoubleSpinBox, QSpinBox))
        )
    ]


def _focus_entry(widget) -> dict[str, str]:
    return {
        "class": widget.__class__.__name__,
        "name": str(widget.objectName() or ""),
        "text": _control_text(widget),
        "accessible_name": str(widget.accessibleName() or ""),
    }


def _focus_validation(entries: list[dict[str, str]], controls: list[dict[str, str]]) -> dict[str, object]:
    """Require a nonempty traversal and meaningful labels for actual controls."""

    missing = [
        {"class": entry["class"], "name": entry["name"]}
        for entry in controls
        if not str(entry["accessible_name"] or "").strip() and not str(entry["text"] or "").strip()
    ]
    def key(entry: dict[str, str]) -> tuple[str, str, str, str]:
        return (
            entry["class"],
            entry["name"],
            entry["text"],
            entry["accessible_name"],
        )

    missing_counts = Counter(key(entry) for entry in controls)
    missing_counts.subtract(Counter(key(entry) for entry in entries))
    unreachable = []
    for entry in controls:
        entry_key = key(entry)
        if missing_counts[entry_key] <= 0:
            continue
        unreachable.append(
            {
                "class": entry["class"],
                "name": entry["name"],
                "text": entry["text"],
                "accessible_name": entry["accessible_name"],
            }
        )
        missing_counts[entry_key] -= 1
    return {
        "traversal_controls": len(entries),
        "reachable_controls": len(controls),
        "missing_meaningful_name": missing,
        "missing_from_traversal": unreachable,
        "valid": bool(entries) and bool(controls) and not missing and not unreachable,
    }


def _coverage_validation(required: list[str] | tuple[str, ...], captured: list[str] | tuple[str, ...]) -> dict[str, object]:
    required_set = set(required)
    captured_set = set(captured)
    return {
        "required": list(required),
        "captured": list(captured),
        "missing": sorted(required_set - captured_set),
        "unexpected": sorted(captured_set - required_set),
        "valid": required_set == captured_set,
    }


def _primary_viewport_validation(
    bounds: dict[str, dict[str, int | bool]], required: list[str] | tuple[str, ...]
) -> dict[str, object]:
    failures = [
        {
            "locator": name,
            "visible": bool(bounds.get(name, {}).get("visible", False)),
            "contained": bool(bounds.get(name, {}).get("contained", False)),
        }
        for name in required
        if not bool(bounds.get(name, {}).get("visible", False))
        or not bool(bounds.get(name, {}).get("contained", False))
    ]
    return {"required": list(required), "failures": failures, "valid": not failures}


def _contrast_validation(pairs: list[dict[str, object]]) -> dict[str, object]:
    def luminance(value: str) -> float:
        color = str(value).lstrip("#")
        channels = [int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    results = []
    for pair in pairs:
        foreground = str(pair["foreground"])
        background = str(pair["background"])
        first, second = sorted((luminance(foreground), luminance(background)), reverse=True)
        ratio = (first + 0.05) / (second + 0.05)
        minimum = float(pair.get("minimum", 4.5))
        results.append(
            {
                **pair,
                "ratio": round(ratio, 3),
                "valid": ratio >= minimum,
            }
        )
    return {"pairs": results, "valid": bool(results) and all(bool(pair["valid"]) for pair in results)}


def _palette_contrast_evidence(app) -> dict[str, object]:
    from PyQt6.QtGui import QPalette

    palette = app.palette()
    pairs = []
    for name, foreground_role, background_role, minimum in (
        ("window_text", QPalette.ColorRole.WindowText, QPalette.ColorRole.Window, 4.5),
        ("field_text", QPalette.ColorRole.Text, QPalette.ColorRole.Base, 4.5),
        ("button_text", QPalette.ColorRole.ButtonText, QPalette.ColorRole.Button, 4.5),
        ("selection_text", QPalette.ColorRole.HighlightedText, QPalette.ColorRole.Highlight, 4.5),
    ):
        pairs.append(
            {
                "locator": name,
                "foreground": palette.color(QPalette.ColorGroup.Active, foreground_role).name(),
                "background": palette.color(QPalette.ColorGroup.Active, background_role).name(),
                "minimum": minimum,
            }
        )
    return _contrast_validation(pairs)


def _focus_evidence(owner) -> dict[str, object]:
    order = _focus_order(owner)
    controls = [_focus_entry(widget) for widget in _focusable_controls(owner)]
    return {
        "focus_order": order,
        "focusable_controls": controls,
        "focus_validation": _focus_validation(order, controls),
    }


def _clipped_controls(owner) -> list[dict[str, object]]:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QAbstractButton, QLabel, QLineEdit

    clipped: list[dict[str, object]] = []
    for widget in owner.findChildren((QAbstractButton, QLabel, QLineEdit)):
        # Disabled actions are still user-visible copy and were a specific
        # source of silent clipping in the audit. Eligibility must not exempt
        # their geometry from evidence.
        if not widget.isVisibleTo(owner):
            continue
        if widget.__class__.__name__ == "HelpIconButton":
            # This dedicated 22 px circular glyph deliberately has a larger
            # platform size hint; its single visible “i” is not text copy.
            continue
        text = _control_text(widget).strip()
        if not text:
            continue
        contents = widget.contentsRect()
        available = max(0, int(contents.width()))
        if isinstance(widget, QLineEdit):
            required_height = int(widget.fontMetrics().height()) + 8
            if widget.height() < required_height:
                clipped.append(
                    {
                        "class": widget.__class__.__name__,
                        "name": str(widget.objectName() or ""),
                        "text": text,
                        "available_height": int(widget.height()),
                        "required_height": required_height,
                    }
                )
            # Editable fields scroll long values horizontally. Only their
            # single-line vertical geometry is a clipping contract.
            continue
        if available == 0:
            continue
        if isinstance(widget, QLabel) and widget.wordWrap():
            required_rect = widget.fontMetrics().boundingRect(contents, int(Qt.TextFlag.TextWordWrap), text)
            required = required_rect.width()
            # Native styles can report a word-wrapped bounding box a few
            # pixels taller than QLabel's painted contents because of leading.
            # A four-pixel allowance still catches a missing text line while
            # avoiding a false clipping failure for a fully painted line.
            height_tolerance = max(4, int(widget.fontMetrics().descent()) + 2)
            clipped_text = required_rect.height() > contents.height() + height_tolerance
        elif isinstance(widget, QAbstractButton):
            # Qt's size hint includes the icon, icon/text spacing, style frame,
            # and padding. Text advance alone misses leading characters hidden
            # underneath an icon in a constrained disabled button.
            available = int(widget.width())
            required = max(int(widget.fontMetrics().horizontalAdvance(text)), int(widget.sizeHint().width()))
            clipped_text = required > available + 2
        else:
            required = int(widget.fontMetrics().horizontalAdvance(text))
            clipped_text = required > available + 2
        if clipped_text:
            clipped.append(
                {
                    "class": widget.__class__.__name__,
                    "name": str(widget.objectName() or ""),
                    "text": text,
                    "available_width": available,
                    "required_width": required,
                    **(
                        {
                            "available_height": int(contents.height()),
                            "required_height": int(required_rect.height()),
                            "height_tolerance": int(height_tolerance),
                        }
                        if isinstance(widget, QLabel) and widget.wordWrap()
                        else {}
                    ),
                }
            )
    return clipped


def _primary_bounds(window) -> dict[str, dict[str, int | bool]]:
    owner = window.centralWidget()
    result: dict[str, dict[str, int | bool]] = {}
    for name in (
        "library_workspace_button",
        "organize_workspace_button",
        "people_workspace_button",
        "tools_workspace_button",
        "jobs_widget",
        "settings_button",
        "edit_roots_button",
        "empty_scope_edit_roots_button",
        "workspace_subnav",
        "workspace_stack",
    ):
        widget = getattr(window, name)
        top_left = widget.mapTo(owner, widget.rect().topLeft())
        bottom_right = widget.mapTo(owner, widget.rect().bottomRight())
        contained = (
            top_left.x() >= 0
            and top_left.y() >= 0
            and bottom_right.x() < owner.width()
            and bottom_right.y() < owner.height()
        )
        result[name] = {
            "x": top_left.x(),
            "y": top_left.y(),
            "width": widget.width(),
            "height": widget.height(),
            "contained": contained,
            "visible": widget.isVisibleTo(window),
        }
    return result


def _timeline_layout_evidence(pane) -> dict[str, object]:
    """Record the measured Timeline grid contract without loading media."""

    pane._update_timeline_layout()
    primary_keys = tuple(pane._TIMELINE_PRIMARY_KEYS)
    positions: dict[str, tuple[int, int, int, int]] = {}
    for key in (*primary_keys, "advanced"):
        for index in range(pane.timeline_form_layout.count()):
            if pane.timeline_form_layout.itemAt(index).widget() is pane._timeline_cells[key]:
                positions[key] = tuple(pane.timeline_form_layout.getItemPosition(index))
                break
    mode = pane._timeline_layout_mode
    expected = {
        "wide": {
            "start": (0, 0, 1, 1), "end": (0, 1, 1, 1), "camera": (0, 2, 1, 1),
            "date_source": (0, 3, 1, 1), "grouping": (0, 4, 1, 1), "show": (0, 5, 1, 1),
            "advanced": (1, 0, 1, 6),
        },
        "medium": {
            "start": (0, 0, 1, 1), "end": (0, 1, 1, 1), "camera": (0, 2, 1, 4),
            "date_source": (1, 0, 1, 3), "grouping": (1, 3, 1, 1), "show": (1, 4, 1, 2),
            "advanced": (2, 0, 1, 6),
        },
        "narrow": {
            "start": (0, 0, 1, 1), "end": (1, 0, 1, 1), "camera": (2, 0, 1, 1),
            "date_source": (3, 0, 1, 1), "grouping": (4, 0, 1, 1), "show": (5, 0, 1, 1),
            "advanced": (6, 0, 1, 1),
        },
    }[mode]
    widths = pane._timeline_primary_widths()
    spacing = pane.timeline_form_layout.horizontalSpacing()
    one_row_width = pane._layout_required_width(tuple(widths[key] for key in primary_keys), spacing)
    two_row_width = max(
        pane._layout_required_width(
            tuple(widths[key] for key in pane._TIMELINE_TWO_ROW_FILTER_KEYS), spacing
        ),
        pane._layout_required_width(
            tuple(widths[key] for key in pane._TIMELINE_TWO_ROW_POLICY_KEYS), spacing
        ),
    )
    usable_width = pane._timeline_usable_width()
    expected_mode = pane._timeline_layout_for_width(usable_width, widths)
    return {
        "mode": mode,
        "expected_mode": expected_mode,
        "usable_width": usable_width,
        "one_row_required_width": one_row_width,
        "two_row_required_width": two_row_width,
        "primary_minimum_widths": widths,
        "positions": {key: list(position) for key, position in positions.items()},
        "tab_height": pane.tabs.height(),
        "form_height": pane.timeline_form.height(),
        "vertical_scroll_range": pane.timeline_scroll.verticalScrollBar().maximum(),
        "matches_measured_contract": (
            mode == expected_mode
            and positions == expected
            and (mode != "wide" or pane.timeline_scroll.verticalScrollBar().maximum() == 0)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument(
        "--logical-sizes",
        default="1280x720,1920x1080",
        help="Comma-separated logical window sizes; each must be at least 1280x720.",
    )
    parser.add_argument(
        "--text-scale",
        type=int,
        choices=(100, 125, 150, 200),
        default=100,
        help="Persisted ClusterLens text scale to validate independently of logical window size.",
    )
    args = parser.parse_args(argv)
    report_dir = args.report_dir.expanduser().resolve()
    if report_dir.exists():
        raise FileExistsError(f"refusing to overwrite UI evidence directory: {report_dir}")
    sizes = [_parse_size(value) for value in str(args.logical_sizes).split(",") if value.strip()]
    if not sizes:
        raise ValueError("at least one logical size is required")
    report_dir.mkdir(parents=True)
    runtime_root = report_dir / "runtime"
    os.environ.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "CLUSTERLENS_RUNTIME_ROOT": str(runtime_root),
            "IMAGE_CLUSTERING_APP_DIR": str(runtime_root),
            "XDG_CONFIG_HOME": str(runtime_root / "xdg-config"),
            "XDG_DATA_HOME": str(runtime_root / "xdg-data"),
            # Background recovery/model probes are intentionally out of scope.
            "CLUSTERLENS_PACKAGED_LAUNCH_SMOKE": "ui-redesign-layout",
        }
    )
    for import_path in (REPO_ROOT, SRC_ROOT):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    from PyQt6.QtCore import QSettings
    from PyQt6.QtGui import QImage
    from PyQt6.QtWidgets import QApplication
    from apps.pyqt_production.app import ProductionClusterApp
    from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG
    from apps.shared.runtime_support import activate_runtime_root
    from ui.job_manager import JobManager
    from ui.job_widgets import JobsDialog
    from ui.list_models import ListEntry
    from ui.photo_inspector_dialog import PhotoInspectorDialog
    from ui.theme import apply_app_theme

    app = QApplication.instance() or QApplication([])
    fixture_media = runtime_root / "ui-fixture-media"
    fixture_media.mkdir(parents=True, exist_ok=True)
    fixture_paths: list[str] = []
    for index, color in enumerate((0x446688, 0x885544, 0x557744), start=1):
        path = fixture_media / f"person-{index}.png"
        image = QImage(96, 96, QImage.Format.Format_RGB32)
        image.fill(color)
        image.save(str(path), "PNG")
        fixture_paths.append(str(path))
    settings = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
    settings.setValue("setup/completed", True)
    settings.setValue("workspace/default_view", "library")
    settings.setValue("workspace/active_roots", json.dumps([str(fixture_media)]))
    settings.setValue("appearance/text_scale", args.text_scale)
    settings.sync()
    started = monotonic()
    window = ProductionClusterApp(activate_runtime_root("UiRedesignEvidence"))
    construction_ms = (monotonic() - started) * 1000.0
    report: dict[str, object] = {
        "report_version": "3",
        "fixture": "isolated generated populated UI fixture; no user media",
        "construction_ms": round(construction_ms, 3),
        "text_scale": args.text_scale,
        "sizes": {},
        "theme_contrast": {},
    }
    dialogs = []
    required_routes = {
        "library": ("timeline", "all_photos", "search", "albums"),
        "organize": ("photos", "tags", "context"),
        "people": ("people", "all_faces", "unnamed", "detect", "review", "review_name", "find", "identities"),
        "tools": ("duplicates", "rename", "recovery"),
    }
    captured_routes: set[str] = set()
    try:
        window.show()
        apply_app_theme(app, "dark")
        report["theme_contrast"]["dark"] = _palette_contrast_evidence(app)
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
        report["fixture_counts"] = {
            "generated_photos": len(fixture_paths),
            "identities": 2,
            "pending_proposals": 2,
            "face_groups": 1,
        }
        for width, height in sizes:
            size_key = f"{width}x{height}"
            window.apply_screen_geometry_constraints((width, height))
            window.resize(width, height)
            _settle(app)
            surfaces: dict[str, object] = {}
            for route in ("library", "organize", "people", "tools"):
                route_started = monotonic()
                window.set_active_workspace(route)
                if route == "library":
                    window.library_pane.set_surface_group("library", "timeline")
                _settle(app)
                background_idle = _settle_until(app, lambda: window.job_manager.active_count() == 0)
                screenshot = report_dir / f"{size_key}_{route}.png"
                surfaces[route] = {
                    "first_visible_ms": round((monotonic() - route_started) * 1000.0, 3),
                    "screenshot": str(screenshot),
                    "saved": bool(window.grab().save(str(screenshot), "PNG")),
                    "background_idle": background_idle,
                    "sections": [
                        window.workspace_subnav.tabText(index)
                        for index in range(window.workspace_subnav.count())
                    ],
                    "clipped_controls": _clipped_controls(window),
                    "primary_bounds": _primary_bounds(window),
                    **_focus_evidence(window),
                }
                if route == "library":
                    surfaces[route]["timeline_layout"] = _timeline_layout_evidence(window.library_pane)

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
                    _settle(app)
                    background_idle = _settle_until(app, lambda: window.job_manager.active_count() == 0)
                    if group == "people":
                        seed_people_fixture()
                        if window.faces_pane is not None:
                            window.faces_pane.sidebar_scroll.verticalScrollBar().setValue(0)
                        _settle(app)
                    route_key = f"{group}.{section}"
                    captured_routes.add(route_key)
                    screenshot = report_dir / f"{size_key}_{group}_{section}.png"
                    evidence = {
                        "route": route_key,
                        "screenshot": str(screenshot),
                        "saved": bool(window.grab().save(str(screenshot), "PNG")),
                        "background_idle": background_idle,
                        "empty_scope_visible": window.workspace_stack.currentWidget() is window.empty_scope_panel,
                        "clipped_controls": _clipped_controls(window),
                        "primary_bounds": _primary_bounds(window),
                        **_focus_evidence(window),
                    }
                    surfaces[f"route_{group}_{section}"] = evidence
            # Roots is an on-demand staged editor. These empty fixture folders
            # make the Sources and Catalog evidence independent of user media.
            fixture_root = runtime_root / "evidence-roots" / size_key
            fixture_root.mkdir(parents=True, exist_ok=True)
            window.source_pane.set_selected_directory(str(fixture_root), activate_scope=False)
            window._edit_active_roots()
            window.source_pane.add_active_root(str(fixture_root))
            _settle(app)
            roots_sources_path = report_dir / f"{size_key}_roots_sources.png"
            surfaces["roots_sources"] = {
                "screenshot": str(roots_sources_path),
                "saved": bool(window.grab().save(str(roots_sources_path), "PNG")),
                "draft_dirty": bool(window.source_pane.has_dirty_draft),
                "source_count_label": str(window.source_pane.draft_summary_label.text()),
                "clipped_controls": _clipped_controls(window),
                "primary_bounds": _primary_bounds(window),
                **_focus_evidence(window),
            }
            window.source_pane.discard_draft()
            window._edit_active_roots()
            window.source_pane.tabs.setCurrentIndex(1)
            _settle(app)
            roots_catalog_path = report_dir / f"{size_key}_roots_catalog.png"
            surfaces["roots_catalog"] = {
                "screenshot": str(roots_catalog_path),
                "saved": bool(window.grab().save(str(roots_catalog_path), "PNG")),
                "catalog_rows": int(window.source_pane.active_roots_list.count()),
                "clipped_controls": _clipped_controls(window),
                "primary_bounds": _primary_bounds(window),
                **_focus_evidence(window),
            }
            window._hide_source_pane()

            # The ordinary workspace screenshots intentionally show the
            # photo-first/default states. Capture the dense surfaces separately
            # so typography regressions cannot hide behind an empty gallery.
            window.set_active_workspace("clustering")
            window.set_clustering_mode("advanced")
            _settle(app)
            advanced_path = report_dir / f"{size_key}_organize_advanced_dark.png"
            surfaces["organize_advanced_dark"] = {
                "screenshot": str(advanced_path),
                "saved": bool(window.grab().save(str(advanced_path), "PNG")),
                "clipped_controls": _clipped_controls(window),
                "primary_bounds": _primary_bounds(window),
                **_focus_evidence(window),
            }

            # Production deliberately hides SearchPane's duplicate task tabs.
            # Build the lazily-loaded face surface once and exercise the shell's
            # Unnamed and Find routes through their public contextual mapping.
            window._build_faces_workspace_widget()
            if window.faces_pane is not None:
                for section in ("unnamed", "find"):
                    window._pending_people_section = section
                    window.set_active_workspace("faces")
                    window._apply_people_section()
                    _settle(app)
                    people_path = report_dir / f"{size_key}_people_{section}_dark.png"
                    surfaces[f"people_{section}_dark"] = {
                        "screenshot": str(people_path),
                        "saved": bool(window.grab().save(str(people_path), "PNG")),
                        "clipped_controls": _clipped_controls(window),
                        "primary_bounds": _primary_bounds(window),
                        **_focus_evidence(window),
                    }

            apply_app_theme(app, "light")
            report["theme_contrast"]["light"] = _palette_contrast_evidence(app)
            window.set_active_workspace("clustering")
            window.set_clustering_mode("advanced")
            _settle(app)
            light_path = report_dir / f"{size_key}_organize_advanced_light.png"
            surfaces["organize_advanced_light"] = {
                "screenshot": str(light_path),
                "saved": bool(window.grab().save(str(light_path), "PNG")),
                "clipped_controls": _clipped_controls(window),
                "primary_bounds": _primary_bounds(window),
                **_focus_evidence(window),
            }
            apply_app_theme(app, "dark")
            report["sizes"][size_key] = surfaces

        required_route_keys = [
            f"{group}.{section}"
            for group, sections in required_routes.items()
            for section in sections
        ]
        report["route_coverage"] = _coverage_validation(required_route_keys, sorted(captured_routes))

        settings_dialog = window._create_settings_dialog(parent=window)
        dialogs.append(settings_dialog)
        settings_dialog.resize(1000, 700)
        settings_dialog.show()
        _settle(app)
        settings_sections: dict[str, object] = {}
        settings_categories = [settings_dialog.tabs.tabText(index) for index in range(settings_dialog.tabs.count())]
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
        for index, section in enumerate(required_settings_sections):
            if not settings_dialog.select_section(section):
                continue
            _settle(app)
            section_slug = section.casefold().replace(" & ", "_").replace(" ", "_")
            section_path = report_dir / f"settings_{index}_{section_slug}.png"
            settings_sections[section] = {
                "saved": bool(settings_dialog.grab().save(str(section_path), "PNG")),
                "screenshot": str(section_path),
                "clipped_controls": _clipped_controls(settings_dialog),
                **_focus_evidence(settings_dialog),
            }
        settings_path = report_dir / "settings.png"
        report["settings"] = {
            "categories": settings_categories,
            "sections": settings_sections,
            "clipped_controls": _clipped_controls(settings_dialog),
            "saved": bool(settings_dialog.grab().save(str(settings_path), "PNG")),
            "screenshot": str(settings_path),
        }
        report["settings"]["coverage"] = _coverage_validation(required_settings_sections, list(settings_sections))

        jobs_manager = JobManager()
        unknown_job = jobs_manager.register_job("Discovering active roots", origin="Library")
        zero_job = jobs_manager.register_job("Preparing metadata cache", origin="Library")
        partial_job = jobs_manager.register_job("Indexing face records", origin="People")
        done_job = jobs_manager.register_job("Catalog refresh", origin="Library")
        cancelled_job = jobs_manager.register_job("Face scan", cancel_fn=lambda: None, origin="People")
        failed_job = jobs_manager.register_job("Backup", origin="Tools")
        jobs_manager.update(zero_job, progress=0, text="0/100 assets")
        jobs_manager.update(partial_job, progress=37, text="37/100 assets")
        jobs_manager.finish(done_job)
        jobs_manager.cancel(cancelled_job)
        jobs_manager.finish(cancelled_job, status="cancelled")
        jobs_manager.finish(failed_job, status="failed", error="Destination is read-only")
        jobs_dialog = JobsDialog(jobs_manager, parent=window)
        jobs_dialog.resize(1280, 720)
        dialogs.append(jobs_dialog)
        jobs_dialog.show()
        _settle(app)
        jobs_path = report_dir / "jobs.png"
        jobs_focus_order = _focus_order(jobs_dialog)
        jobs_focus_controls = [_focus_entry(widget) for widget in _focusable_controls(jobs_dialog)]
        jobs_by_id = {job.job_id: job for job in jobs_manager.history(limit=10)}
        jobs_fixture = {
            "unknown": jobs_by_id[unknown_job].status == "running" and jobs_by_id[unknown_job].progress is None,
            "zero": jobs_by_id[zero_job].status == "running" and jobs_by_id[zero_job].progress == 0,
            "partial": jobs_by_id[partial_job].status == "running" and jobs_by_id[partial_job].progress == 37,
            "done": jobs_by_id[done_job].status == "finished",
            "cancelled": jobs_by_id[cancelled_job].status == "cancelled",
            "failed": jobs_by_id[failed_job].status == "failed",
        }
        report["jobs"] = {
            "focus_order": jobs_focus_order,
            "focusable_controls": jobs_focus_controls,
            "state_fixture": jobs_fixture,
            "clipped_controls": _clipped_controls(jobs_dialog),
            "saved": bool(jobs_dialog.grab().save(str(jobs_path), "PNG")),
            "screenshot": str(jobs_path),
        }
        report["jobs"]["focus_validation"] = _focus_validation(jobs_focus_order, jobs_focus_controls)

        inspector_fixture = report_dir / "inspector-focus-fixture.png"
        fixture_image = QImage(16, 16, QImage.Format.Format_RGB32)
        fixture_image.fill(0x4477AA)
        fixture_image.save(str(inspector_fixture), "PNG")
        inspector = PhotoInspectorDialog(image_paths=[str(inspector_fixture)], parent=window)
        dialogs.append(inspector)
        inspector.resize(1280, 720)
        inspector.show()
        _settle(app)
        inspector_panels = [inspector.inspector_tabs.tabText(index) for index in range(inspector.inspector_tabs.count())]
        inspector_sections: dict[str, object] = {}
        for index, panel in enumerate(inspector_panels):
            inspector.inspector_tabs.setCurrentIndex(index)
            _settle(app)
            panel_path = report_dir / f"photo_inspector_{index}_{panel.casefold()}.png"
            inspector_sections[panel] = {
                "saved": bool(inspector.grab().save(str(panel_path), "PNG")),
                "screenshot": str(panel_path),
                "clipped_controls": _clipped_controls(inspector),
                **_focus_evidence(inspector),
            }
        inspector_path = report_dir / "photo_inspector.png"
        report["photo_inspector"] = {
            "panels": inspector_panels,
            "sections": inspector_sections,
            "clipped_controls": _clipped_controls(inspector),
            "saved": bool(inspector.grab().save(str(inspector_path), "PNG")),
            "screenshot": str(inspector_path),
        }
        report["photo_inspector"]["coverage"] = _coverage_validation(inspector_panels, list(inspector_sections))

        primary_checks = []
        for size in report["sizes"].values():
            for surface in size.values():
                bounds = surface["primary_bounds"]
                # Empty scope intentionally has one root action in the shared
                # panel, while a configured scope uses the compact header
                # action. Either is valid; requiring both would hide this
                # first-run safety contract from the verifier.
                root_actions = (
                    bounds["edit_roots_button"],
                    bounds["empty_scope_edit_roots_button"],
                )
                primary_checks.extend(
                    entry["contained"] and entry["visible"]
                    for name, entry in bounds.items()
                    if name not in {"edit_roots_button", "empty_scope_edit_roots_button"}
                    and not (
                        name == "workspace_subnav"
                        and bool(bounds["empty_scope_edit_roots_button"]["visible"])
                    )
                )
                primary_checks.append(
                    sum(bool(entry["contained"] and entry["visible"]) for entry in root_actions) == 1
                )
        screenshot_checks = [
            surface["saved"]
            for size in report["sizes"].values()
            for surface in size.values()
        ]
        screenshot_checks.extend(
            bool(report[key]["saved"]) for key in ("settings", "jobs", "photo_inspector")
        )
        screenshot_checks.extend(
            bool(section["saved"])
            for key in ("settings", "photo_inspector")
            for section in report[key]["sections"].values()
        )
        timeline_checks = [
            bool(surface["timeline_layout"]["matches_measured_contract"])
            for size in report["sizes"].values()
            for surface in size.values()
            if "timeline_layout" in surface
        ]
        clipping_checks = [
            not surface["clipped_controls"]
            for size in report["sizes"].values()
            for surface in size.values()
        ]
        clipping_checks.extend(
            not section["clipped_controls"]
            for key in ("settings", "photo_inspector")
            for section in report[key]["sections"].values()
        )
        focus_checks = [bool(report["jobs"]["focus_validation"]["valid"])]
        focus_checks.extend(
            bool(surface["focus_validation"]["valid"])
            for size in report["sizes"].values()
            for name, surface in size.items()
            if "focus_validation" in surface
        )
        focus_checks.extend(
            bool(section["focus_validation"]["valid"])
            for key in ("settings", "photo_inspector")
            for section in report[key]["sections"].values()
        )
        coverage_checks = [
            bool(report["route_coverage"]["valid"]),
            bool(report["settings"]["coverage"]["valid"]),
            bool(report["photo_inspector"]["coverage"]["valid"]),
            all(int(value) > 0 for value in report["fixture_counts"].values()),
            all(
                not bool(surface.get("empty_scope_visible", False))
                for size in report["sizes"].values()
                for name, surface in size.items()
                if name.startswith("route_")
            ),
        ]
        background_idle_checks = [
            bool(surface.get("background_idle", True))
            for size in report["sizes"].values()
            for surface in size.values()
        ]
        contrast_checks = [
            bool(evidence["valid"])
            for evidence in report["theme_contrast"].values()
        ]
        shutdown_checks = []
        for dialog in reversed(dialogs):
            shutdown_checks.append(_close_widget(app, dialog))
        dialogs.clear()
        shutdown_checks.append(_close_widget(app, window))
        report["shutdown"] = {
            "dialogs_and_window_closed": all(shutdown_checks),
            "checks": shutdown_checks,
        }
        report["validation"] = "PASS" if all(primary_checks) and all(screenshot_checks) and all(timeline_checks) and all(clipping_checks) and all(focus_checks) and all(coverage_checks) and all(background_idle_checks) and all(contrast_checks) and all(shutdown_checks) and all(report["jobs"]["state_fixture"].values()) else "FAIL"
        report_path = report_dir / "ui_redesign.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"validation": report["validation"], "report": str(report_path)}, indent=2))
        return 0 if report["validation"] == "PASS" else 1
    finally:
        for dialog in reversed(dialogs):
            try:
                dialog.close()
            except RuntimeError:
                pass
        try:
            window.close()
        except RuntimeError:
            pass
        _settle(app, 3)


if __name__ == "__main__":
    raise SystemExit(main())
