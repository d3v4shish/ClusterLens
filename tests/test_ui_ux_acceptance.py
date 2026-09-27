from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent, QObject, QItemSelectionModel, QRect, QSettings, Qt, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QLineEdit, QPushButton, QToolButton, QWidget


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from apps.shared.runtime_support import activate_runtime_root
from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG
from infra import settings as settings_mod
from ui.icons import themed_icon
from ui.list_models import ListEntry, PagedListEntryModel


APP = QApplication.instance() or QApplication([])


class UiUxAcceptanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._initial_top_level_widgets = set(APP.topLevelWidgets())
        self._runtime_env_snapshot = {
            key: os.environ.get(key)
            for key in ("CLUSTERLENS_RUNTIME_ROOT", "IMAGE_CLUSTERING_APP_DIR")
        }
        self._runtime_root_fixture = TemporaryDirectory()
        self.addCleanup(self._runtime_root_fixture.cleanup)
        os.environ["CLUSTERLENS_RUNTIME_ROOT"] = self._runtime_root_fixture.name
        os.environ["IMAGE_CLUSTERING_APP_DIR"] = self._runtime_root_fixture.name
        settings_mod._RUNTIME_BASE_DIR = None
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        self._production_settings_snapshot = {
            key: store.value(key)
            for key in store.allKeys()
        }
        store.clear()
        store.setValue("runtime/preferred_mode", "cpu")
        store.setValue("workspace/default_view", "library")
        store.setValue("workspace/faces_mode", "basic")
        store.setValue("workspace/power_user_mode", False)
        store.setValue("setup/completed", True)
        store.sync()

    @staticmethod
    def _wait_for(predicate, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        APP.processEvents()
        if not predicate():
            raise AssertionError("Timed out waiting for UI state")

    def tearDown(self) -> None:
        for widget in [
            candidate
            for candidate in APP.topLevelWidgets()
            if candidate not in self._initial_top_level_widgets
        ]:
            try:
                self._close_widget(widget)
                widget.deleteLater()
            except RuntimeError:
                continue
        for _ in range(4):
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            APP.processEvents()
        settings_mod._RUNTIME_BASE_DIR = None
        for key, value in self._runtime_env_snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.clear()
        for key, value in self._production_settings_snapshot.items():
            store.setValue(key, value)
        store.sync()

    @staticmethod
    def _close_widget(widget) -> None:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                if widget.close():
                    APP.processEvents()
                    APP.processEvents()
                    return
            except RuntimeError:
                return
            APP.processEvents()
            time.sleep(0.005)
        raise AssertionError("Timed out waiting for the application window to close")

    def test_minimum_screen_contract_accepts_landscape_and_portrait(self):
        from apps.pyqt_production.app import _screen_geometry_supported

        self.assertTrue(_screen_geometry_supported(QRect(0, 0, 1280, 720)))
        self.assertTrue(_screen_geometry_supported(QRect(0, 0, 720, 1280)))
        self.assertFalse(_screen_geometry_supported(QRect(0, 0, 1279, 719)))

    def test_ui_verifier_negative_controls_reject_disabled_clipping_and_missing_names(self):
        from scripts.verify_ui_redesign import (
            _clipped_controls,
            _contrast_validation,
            _coverage_validation,
            _focus_validation,
            _primary_viewport_validation,
        )

        owner = QWidget()
        owner.resize(220, 100)
        clipped = QPushButton("Disabled action whose complete label must remain readable", owner)
        clipped.setGeometry(10, 10, 80, 28)
        clipped.setEnabled(False)
        clipped_field = QLineEdit(owner)
        clipped_field.setPlaceholderText("Clipped field")
        clipped_field.setStyleSheet("QLineEdit { min-height: 0px; padding: 0px; }")
        clipped_field.setGeometry(100, 10, 110, 8)
        unnamed = QLabel("", owner)
        unnamed.setGeometry(10, 50, 80, 20)
        owner.show()
        APP.processEvents()
        try:
            failures = _clipped_controls(owner)
            self.assertEqual(2, len(failures))
            self.assertEqual(
                {"Disabled action whose complete label must remain readable", "Clipped field"},
                {failure["text"] for failure in failures},
            )

            validation = _focus_validation(
                [{"class": "QPushButton", "name": "", "text": "", "accessible_name": ""}],
                [{"class": "QPushButton", "name": "", "text": "", "accessible_name": ""}],
            )
            self.assertFalse(validation["valid"])
            self.assertEqual([{"class": "QPushButton", "name": ""}], validation["missing_meaningful_name"])

            incomplete = _focus_validation(
                [{"class": "QPushButton", "name": "first", "text": "First", "accessible_name": ""}],
                [
                    {"class": "QPushButton", "name": "first", "text": "First", "accessible_name": ""},
                    {"class": "QPushButton", "name": "second", "text": "Second", "accessible_name": ""},
                ],
            )
            self.assertFalse(incomplete["valid"])
            self.assertEqual("second", incomplete["missing_from_traversal"][0]["name"])

            coverage = _coverage_validation(("library", "people", "tools"), ("library", "tools"))
            self.assertFalse(coverage["valid"])
            self.assertEqual(["people"], coverage["missing"])

            viewport = _primary_viewport_validation(
                {"primary_search": {"visible": True, "contained": False}},
                ("primary_search",),
            )
            self.assertFalse(viewport["valid"])
            self.assertEqual("primary_search", viewport["failures"][0]["locator"])

            contrast = _contrast_validation(
                [
                    {
                        "locator": "selected_cluster_text",
                        "foreground": "#777777",
                        "background": "#808080",
                        "minimum": 4.5,
                    }
                ]
            )
            self.assertFalse(contrast["valid"])
            self.assertEqual("selected_cluster_text", contrast["pairs"][0]["locator"])
        finally:
            owner.close()

    def test_people_routes_have_named_reachable_keyboard_controls(self):
        from scripts.verify_ui_redesign import _focus_entry, _focusable_controls
        from ui.search_pane import SearchPane

        pane = SearchPane(
            enabled_tabs=["All Faces", "Face Library", "Face Search", "Identities"],
            external_results=False,
        )
        pane.resize(1280, 720)
        pane.show()
        try:
            for tab_index in range(pane.tabs.count()):
                with self.subTest(route=pane.tabs.tabText(tab_index)):
                    pane.tabs.setCurrentIndex(tab_index)
                    APP.processEvents()
                    expected = _focusable_controls(pane)
                    self.assertTrue(expected)
                    unnamed = [
                        _focus_entry(control)
                        for control in expected
                        if not _focus_entry(control)["accessible_name"].strip()
                        and not _focus_entry(control)["text"].strip()
                    ]
                    self.assertEqual([], unnamed)

                    expected_ids = {id(control) for control in expected}
                    observed_ids: set[int] = set()
                    expected[0].setFocus(Qt.FocusReason.OtherFocusReason)
                    for _index in range(max(16, len(expected) * 3)):
                        focused = pane.focusWidget()
                        if focused in expected:
                            observed_ids.add(id(focused))
                        QTest.keyClick(focused or pane, Qt.Key.Key_Tab)
                    self.assertEqual(
                        expected_ids,
                        observed_ids,
                        [_focus_entry(control) for control in expected if id(control) not in observed_ids],
                    )

                    expected[0].setFocus(Qt.FocusReason.OtherFocusReason)
                    before_backtab = pane.focusWidget()
                    QTest.keyClick(pane.focusWidget() or pane, Qt.Key.Key_Backtab)
                    after_backtab = pane.focusWidget()
                    self.assertIsNotNone(after_backtab)
                    self.assertIsNot(after_backtab, before_backtab)
                    self.assertTrue(pane.isAncestorOf(after_backtab))
                    self.assertTrue(after_backtab.isEnabled())
                    pane.tabs.tabBar().setFocus(Qt.FocusReason.OtherFocusReason)
                    APP.processEvents()
        finally:
            focused = APP.focusWidget()
            if focused is not None and pane.isAncestorOf(focused):
                focused.clearFocus()
            pane.close()
            pane.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            APP.processEvents()

    def test_idle_header_is_bounded_and_legacy_controls_are_excluded(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxHeaderTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            header = window.findChild(QWidget, "applicationHeader")
            direct = [
                widget for widget in header.findChildren(
                    QWidget,
                    options=Qt.FindChildOption.FindDirectChildrenOnly,
                )
                if isinstance(widget, (QPushButton, QToolButton, QComboBox))
            ]
            visible_focus_controls = [
                widget for widget in direct
                if not widget.isHidden() and widget.focusPolicy() != Qt.FocusPolicy.NoFocus
            ]

            self.assertLessEqual(len(visible_focus_controls), 9)
            labels = [
                widget.currentText() if isinstance(widget, QComboBox) else widget.text()
                for widget in visible_focus_controls
            ]
            active_mode = window.mode_selector.currentText()
            self.assertIn(active_mode, {"Basic", "Advanced"})
            for label in ("Library", "Organize", "People", "Tools", "Settings"):
                self.assertIn(label, labels)
            for legacy_label in ("Clustering", "Faces", "Names", "Tags"):
                self.assertNotIn(legacy_label, labels)
            self.assertNotIn(active_mode, labels)
            self.assertTrue(window.mode_selector.isHidden())
            self.assertIs(window.jobs_widget.jobs_btn, window.view_button.nextInFocusChain())
            self.assertIs(window.runtime_badge, window.jobs_widget.jobs_btn.nextInFocusChain())
            self.assertIs(window.settings_button, window.runtime_badge.nextInFocusChain())
            window.source_pane.set_active_roots((tmp,))
            window.set_active_workspace("clustering")
            window.set_clustering_mode("advanced")
            self.assertTrue(window.organize_workspace_button.isChecked())
            self.assertFalse(window.mode_selector.isHidden())
            self.assertTrue(all(widget.isHidden() for widget in (
                window.source_toggle,
                window.controls_toggle,
                window.details_toggle,
            )))
            self._close_widget(window)

    def test_four_workspace_shell_exposes_contextual_sections(self):
        from apps.pyqt_production.app import ProductionClusterApp

        def sections(window):
            return [window.workspace_subnav.tabText(index) for index in range(window.workspace_subnav.count())]

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxWorkspaceShellTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.source_pane.set_active_roots((tmp,))

            window.set_active_workspace("library")
            window.show()
            APP.processEvents()
            self.assertEqual(["Timeline", "All Photos", "Search", "Albums"], sections(window))
            window.workspace_subnav.setFocus(Qt.FocusReason.OtherFocusReason)
            QTest.keyClick(window.workspace_subnav, Qt.Key.Key_Tab)
            self.assertIsNot(window.workspace_subnav, window.focusWidget())
            QTest.keyClick(window.focusWidget(), Qt.Key.Key_Backtab)
            self.assertIs(window.workspace_subnav, window.focusWidget())
            window.set_active_workspace("organize")
            self.assertEqual(["Photos", "Tags", "Context"], sections(window))
            window.set_active_workspace("people_cleanup")
            self.assertEqual(
                ["People", "All Faces", "Unnamed", "Detect", "Review", "Review && Name", "Find", "Manage"],
                sections(window),
            )
            review_index = sections(window).index("Review")
            review_name_index = sections(window).index("Review && Name")
            self.assertIn("pending automatic name proposals", window.workspace_subnav.tabToolTip(review_index))
            self.assertIn("durable names", window.workspace_subnav.tabToolTip(review_name_index))

            window.workspace_subnav.setCurrentIndex(sections(window).index("All Faces"))
            self._wait_for(lambda: window.faces_pane is not None)
            window.workspace_subnav.setFocus()
            QTest.keyClick(window.workspace_subnav, Qt.Key.Key_Right)
            APP.processEvents()
            self.assertEqual("Unnamed", window.workspace_subnav.tabText(window.workspace_subnav.currentIndex()))
            self.assertEqual("unlabeled", window.faces_pane.face_album_label_filter_combo.currentData())

            window.workspace_subnav.setCurrentIndex(sections(window).index("Find"))
            APP.processEvents()
            self.assertEqual("faces", window._active_workspace)
            self.assertEqual("find", window.faces_pane._current_people_section())
            window.workspace_subnav.setCurrentIndex(sections(window).index("People"))
            APP.processEvents()
            self.assertEqual("names", window._active_workspace)
            window.workspace_subnav.setCurrentIndex(sections(window).index("Review"))
            APP.processEvents()
            self.assertEqual("people_cleanup", window._active_workspace)

            window.workspace_subnav.setCurrentIndex(sections(window).index("Review && Name"))
            self._wait_for(lambda: window.faces_pane is not None)
            self.assertEqual("Folder Review", window.faces_pane.tabs.tabText(window.faces_pane.tabs.currentIndex()))
            self.assertEqual("Review && Name", window.faces_pane.face_library_tabs.tabText(window.faces_pane.face_library_tabs.currentIndex()))
            window.faces_pane.face_library_tabs.setCurrentIndex(0)
            APP.processEvents()
            self.assertEqual("Detect", window.workspace_subnav.tabText(window.workspace_subnav.currentIndex()))

            window.workspace_subnav.setCurrentIndex(sections(window).index("Unnamed"))
            APP.processEvents()
            self.assertEqual("unlabeled", window.faces_pane.face_album_label_filter_combo.currentData())
            window.faces_pane.face_album_label_filter_combo.setCurrentIndex(
                window.faces_pane.face_album_label_filter_combo.findData("all")
            )
            APP.processEvents()
            self.assertEqual("All Faces", window.workspace_subnav.tabText(window.workspace_subnav.currentIndex()))

            window.workspace_subnav.setCurrentIndex(sections(window).index("Manage"))
            self.assertEqual("Identities", window.faces_pane.tabs.tabText(window.faces_pane.tabs.currentIndex()))

            handoff_photo = Path(tmp) / "handoff.jpg"
            handoff_photo.write_bytes(b"fixture")
            scan_calls: list[bool] = []
            window.faces_pane._scan_face_folder = lambda: scan_calls.append(True)  # type: ignore[method-assign]
            window._review_photo_set_in_faces([str(handoff_photo)])
            APP.processEvents()
            self.assertEqual("Detect", window.workspace_subnav.tabText(window.workspace_subnav.currentIndex()))
            self.assertEqual("detect", window.faces_pane._current_people_section())
            self.assertEqual((str(handoff_photo),), window.faces_pane._explicit_review_scope_paths)
            self.assertEqual([], scan_calls)

            window.set_active_workspace("tools")
            self.assertEqual(["Duplicates", "Rename && Metadata", "Recovery"], sections(window))
            self.assertIs(window.workspace_stack.currentWidget(), window.library_pane)
            self.assertEqual("Tools", window.library_pane.workspace_title.text())
            self.assertTrue(window.tools_workspace_button.isChecked())
            self._close_widget(window)

    def test_people_workspace_state_persists_through_production_reconstruction(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            first = ProductionClusterApp(activate_runtime_root("UiUxPeopleStateRestartTest"))
            self._wait_for(lambda: first._storage_usage_thread is None)
            first.source_pane.set_active_roots((tmp,))
            first.set_active_workspace("faces")
            self._wait_for(lambda: first.faces_pane is not None)
            first.faces_pane.activate_people_section("find")
            first.faces_pane._set_face_walkthrough_visible(False)
            first.faces_pane._set_expander_state(
                first.faces_pane.face_find_options_toggle,
                first.faces_pane.face_find_options_panel,
                True,
            )
            self._close_widget(first)

            stored = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP).value(
                "workspace/people_state_v1", ""
            )
            self.assertIn('"active_route": "find"', str(stored))

            settings_mod._RUNTIME_BASE_DIR = None
            second = ProductionClusterApp(activate_runtime_root("UiUxPeopleStateRestartTest"))
            self.addCleanup(self._close_widget, second)
            self._wait_for(lambda: second.faces_pane is not None)
            self.assertEqual("find", second.faces_pane._current_people_section())
            self.assertTrue(second.faces_pane.face_walkthrough_panel.isHidden())
            self.assertTrue(second.faces_pane.face_find_options_toggle.isChecked())
            self._close_widget(second)

    def test_production_people_workspace_state_rejects_malformed_schema_without_startup_failure(self):
        from types import SimpleNamespace
        from apps.pyqt_production.app import ProductionClusterApp

        for raw in (
            "not-json",
            '{"state_version":"malformed","active_route":"identities"}',
            {"state_version": 99, "active_route": "identities"},
            ["not", "an", "object"],
        ):
            fake = SimpleNamespace(settings_store=SimpleNamespace(value=lambda *_args, _raw=raw: _raw))
            self.assertEqual({}, ProductionClusterApp._load_people_workspace_state(fake))

    def test_production_people_workspace_state_migrates_supported_unversioned_schema_once(self):
        from types import SimpleNamespace
        from apps.pyqt_production.app import (
            LEGACY_PEOPLE_WORKSPACE_STATE_KEY,
            PEOPLE_WORKSPACE_STATE_KEY,
            ProductionClusterApp,
        )

        class FakeSettings:
            def __init__(self):
                self.values = {
                    LEGACY_PEOPLE_WORKSPACE_STATE_KEY: json.dumps(
                        {
                            "route": "search",
                            "walkthrough_dismissed": True,
                            "find_options_expanded": True,
                            "identity_management_expanded": False,
                            "result_paths": ["/must/not/migrate.jpg"],
                        }
                    )
                }
                self.sync_count = 0

            def value(self, key, default=None):
                return self.values.get(key, default)

            def setValue(self, key, value):
                self.values[key] = value

            def remove(self, key):
                self.values.pop(key, None)

            def sync(self):
                self.sync_count += 1

        settings = FakeSettings()
        fake = SimpleNamespace(settings_store=settings)
        migrated = ProductionClusterApp._load_people_workspace_state(fake)
        self.assertEqual(1, migrated["state_version"])
        self.assertEqual("find", migrated["active_route"])
        self.assertTrue(migrated["face_walkthrough_dismissed"])
        self.assertTrue(migrated["face_find_options_expanded"])
        self.assertFalse(migrated["face_identity_management_expanded"])
        self.assertNotIn("result_paths", migrated)
        self.assertNotIn(LEGACY_PEOPLE_WORKSPACE_STATE_KEY, settings.values)
        self.assertEqual(migrated, json.loads(settings.values[PEOPLE_WORKSPACE_STATE_KEY]))
        self.assertEqual(1, settings.sync_count)

    def test_people_workspace_state_survives_missing_and_changed_roots(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            runtime_name = "UiUxPeopleStateChangedRootTest"
            missing_root = Path(tmp) / "removed-root"
            changed_root = Path(tmp) / "changed-root"
            missing_root.mkdir()
            changed_root.mkdir()

            settings_mod._RUNTIME_BASE_DIR = None
            first = ProductionClusterApp(activate_runtime_root(runtime_name))
            self._wait_for(lambda: first._storage_usage_thread is None)
            first.source_pane.set_active_roots((str(missing_root),))
            first.set_active_workspace("faces")
            self._wait_for(lambda: first.faces_pane is not None)
            first.faces_pane.activate_people_section("find")
            first.faces_pane._set_face_walkthrough_visible(False)
            self._close_widget(first)
            missing_root.rmdir()

            settings_mod._RUNTIME_BASE_DIR = None
            second = ProductionClusterApp(activate_runtime_root(runtime_name))
            self.addCleanup(self._close_widget, second)
            second.set_active_workspace("faces")
            self._wait_for(lambda: second.faces_pane is not None)
            self.assertEqual((str(missing_root),), second.source_pane.active_scope.roots)
            self.assertEqual("find", second.faces_pane._current_people_section())
            self.assertTrue(second.faces_pane.face_walkthrough_panel.isHidden())

            second.source_pane.set_active_roots((str(changed_root),))
            APP.processEvents()
            self.assertEqual((str(changed_root),), second.source_pane.active_scope.roots)
            self.assertEqual(str(changed_root), second.faces_pane.face_folder_path.text())
            self.assertEqual("find", second.faces_pane._current_people_section())
            self._close_widget(second)

    def test_people_workspace_state_survives_independent_process_restart(self):
        child_code = r'''
import os
from pathlib import Path
import time

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from apps.shared.runtime_support import activate_runtime_root

layout = activate_runtime_root("UiUxPeopleStateSubprocessTest")
from apps.pyqt_production.app import ProductionClusterApp
from apps.pyqt_production.identity import PRODUCTION_QSETTINGS_APP, PRODUCTION_QSETTINGS_ORG

app = QApplication.instance() or QApplication([])
store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
store.setValue("runtime/preferred_mode", "cpu")
store.setValue("setup/completed", True)
store.sync()

def wait_for(predicate, timeout_s=12.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise RuntimeError("timed out waiting for child UI state")

def close_window(window):
    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if window.close():
            app.processEvents()
            app.processEvents()
            return
        app.processEvents()
        time.sleep(0.005)
    raise RuntimeError("timed out closing child window")

window = ProductionClusterApp(layout)
wait_for(lambda: window._storage_usage_thread is None)
root = Path(os.environ["CLUSTERLENS_RESTART_FIXTURE_ROOT"])
root.mkdir(parents=True, exist_ok=True)
window.source_pane.set_active_roots((str(root),))
window.set_active_workspace("faces")
wait_for(lambda: window.faces_pane is not None)
if os.environ["CLUSTERLENS_RESTART_PHASE"] == "write":
    window.faces_pane.activate_people_section("find")
    window.faces_pane._set_face_walkthrough_visible(False)
    window.faces_pane._set_expander_state(
        window.faces_pane.face_find_options_toggle,
        window.faces_pane.face_find_options_panel,
        True,
    )
    close_window(window)
    print("PEOPLE_STATE_WRITE_OK")
else:
    assert window.faces_pane._current_people_section() == "find"
    assert window.faces_pane.face_walkthrough_panel.isHidden()
    assert window.faces_pane.face_find_options_toggle.isChecked()
    close_window(window)
    print("PEOPLE_STATE_READ_OK")
'''
        with TemporaryDirectory() as tmp:
            environment = os.environ.copy()
            environment.update(
                {
                    "CLUSTERLENS_RUNTIME_ROOT": str(Path(tmp) / "runtime"),
                    "IMAGE_CLUSTERING_APP_DIR": str(Path(tmp) / "runtime"),
                    "CLUSTERLENS_RESTART_FIXTURE_ROOT": str(Path(tmp) / "photos"),
                    "QT_QPA_PLATFORM": "offscreen",
                    "XDG_CONFIG_HOME": str(Path(tmp) / "settings"),
                    "PYTHONPYCACHEPREFIX": str(Path(tmp) / "pycache"),
                    "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
                }
            )
            for phase, marker in (("write", "PEOPLE_STATE_WRITE_OK"), ("read", "PEOPLE_STATE_READ_OK")):
                environment["CLUSTERLENS_RESTART_PHASE"] = phase
                completed = subprocess.run(
                    [sys.executable, "-c", child_code],
                    cwd=ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    0,
                    completed.returncode,
                    msg=f"{phase} child failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
                )
                self.assertIn(marker, completed.stdout)

    def test_workspace_shortcuts_match_the_four_visible_destinations(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxShortcutMapTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.show()
            shortcuts = {shortcut.key().toString(): shortcut for shortcut in window._shell_shortcuts}
            for sequence, button in (
                ("Ctrl+2", window.organize_workspace_button),
                ("Ctrl+3", window.people_workspace_button),
                ("Ctrl+4", window.tools_workspace_button),
                ("Ctrl+1", window.library_workspace_button),
            ):
                shortcuts[sequence].activated.emit()
                APP.processEvents()
                self.assertTrue(button.isChecked())
            shortcuts["Ctrl+F"].activated.emit()
            APP.processEvents()
            self.assertTrue(window.library_workspace_button.isChecked())
            self.assertEqual("Search", window.workspace_subnav.tabText(window.workspace_subnav.currentIndex()))
            self.assertIs(window.library_pane.search_field, window.focusWidget())
            self._close_widget(window)
            APP.processEvents()

    def test_primary_shell_controls_stay_reachable_at_supported_breakpoints(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxResponsiveShellTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.show()
            breakpoint_widths = {}
            for width, height in ((1280, 720), (1366, 768), (1440, 900), (1920, 1080)):
                window.apply_screen_geometry_constraints((width, height))
                window.resize(width, height)
                APP.processEvents()
                breakpoint_widths[width] = window._layout_widths()["workspace"]
                self.assertLessEqual(window.minimumWidth(), width)
                self.assertLessEqual(window.minimumHeight(), height)
                self.assertGreaterEqual(window.workspace_stack.width(), 720)
                for control in (
                    window.library_workspace_button,
                    window.organize_workspace_button,
                    window.people_workspace_button,
                    window.tools_workspace_button,
                    window.jobs_widget,
                    window.settings_button,
                    window.edit_roots_button,
                    window.workspace_subnav,
                ):
                    top_left = control.mapTo(window.centralWidget(), control.rect().topLeft())
                    bottom_right = control.mapTo(window.centralWidget(), control.rect().bottomRight())
                    self.assertGreaterEqual(top_left.x(), 0, control.objectName() or control.__class__.__name__)
                    self.assertGreaterEqual(top_left.y(), 0, control.objectName() or control.__class__.__name__)
                    self.assertLess(bottom_right.x(), window.centralWidget().width(), control.objectName() or control.__class__.__name__)
                    self.assertLess(bottom_right.y(), window.centralWidget().height(), control.objectName() or control.__class__.__name__)
            self.assertEqual(720, breakpoint_widths[1280])
            self.assertEqual(900, breakpoint_widths[1440])
            self.assertEqual(1040, breakpoint_widths[1920])
            self._close_widget(window)
            APP.processEvents()

    def test_primary_shell_remains_reachable_at_200_percent_text_scale(self):
        from apps.pyqt_production.app import ProductionClusterApp

        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.setValue("appearance/text_scale", 200)
        store.sync()
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxLargeTextShellTest"))
            self.addCleanup(self._close_widget, window)
            window.apply_screen_geometry_constraints((1280, 720))
            window.resize(1280, 720)
            window.show()
            APP.processEvents()
            try:
                self.assertLessEqual(window.minimumWidth(), 1280)
                self.assertLessEqual(window.minimumHeight(), 720)
                for control in (
                    window.library_workspace_button,
                    window.organize_workspace_button,
                    window.people_workspace_button,
                    window.tools_workspace_button,
                    window.jobs_widget.jobs_btn,
                    window.settings_button,
                ):
                    top_left = control.mapTo(window.centralWidget(), control.rect().topLeft())
                    bottom_right = control.mapTo(window.centralWidget(), control.rect().bottomRight())
                    self.assertGreaterEqual(top_left.x(), 0, control.accessibleName() or control.text())
                    self.assertLess(bottom_right.x(), window.centralWidget().width(), control.accessibleName() or control.text())
                    self.assertGreaterEqual(
                        control.height(),
                        control.fontMetrics().height() + 8,
                        control.accessibleName() or control.text(),
                    )
            finally:
                if window.theme_manager is not None:
                    window.theme_manager.set_text_scale(100, persist=False)
                self._close_widget(window)

    def test_repeated_text_scale_reflow_preserves_people_route_focus_and_input(self):
        from apps.pyqt_production.app import ProductionClusterApp

        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.setValue("appearance/text_scale", 100)
        store.sync()
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxRepeatedTextScaleTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.source_pane.set_active_roots((tmp,))
            window._build_faces_workspace_widget()
            window._pending_people_section = "find"
            window.set_active_workspace("faces")
            window.apply_screen_geometry_constraints((1280, 720))
            window.resize(1280, 720)
            window.show()
            APP.processEvents()
            pane = window.faces_pane
            self.assertIsNotNone(pane)
            pane.face_query_path.setText("/fixture/a-very-long-query-photo-name.jpg")
            pane.face_query_path.setFocus()
            APP.processEvents()

            try:
                for scale in (200, 125, 150, 100, 200):
                    with self.subTest(scale=scale):
                        window.theme_manager.set_text_scale(scale, persist=False)
                        APP.processEvents()
                        APP.processEvents()
                        self.assertEqual("faces", window._active_workspace)
                        self.assertEqual("find", pane._current_people_section())
                        self.assertEqual(
                            "/fixture/a-very-long-query-photo-name.jpg",
                            pane.face_query_path.text(),
                        )
                        self.assertIs(pane.face_query_path, window.focusWidget())
                        self.assertLessEqual(window.minimumWidth(), 1280)
                        self.assertLessEqual(window.minimumHeight(), 720)
                        overflow = [
                            (
                                child.objectName(),
                                child.__class__.__name__,
                                child.minimumSizeHint().width(),
                                child.width(),
                            )
                            for child in pane.sidebar_content_panel.findChildren(QWidget)
                            if child.isVisibleTo(pane)
                            and child.minimumSizeHint().width() > pane.sidebar_scroll.viewport().width()
                        ]
                        self.assertEqual(0, pane.sidebar_scroll.horizontalScrollBar().maximum(), overflow[:12])
            finally:
                window.theme_manager.set_text_scale(100, persist=False)
                self._close_widget(window)

    def test_settings_cancel_restores_live_appearance_preview_without_persisting(self):
        from apps.pyqt_production.app import ProductionClusterApp

        class PreviewDialog(QObject):
            appearance_preview_requested = pyqtSignal(str, str, int)
            runtime_rescanned = pyqtSignal()
            DialogCode = QDialog.DialogCode

            def __init__(self, manager, focus_thief) -> None:
                super().__init__()
                self.manager = manager
                self.focus_thief = focus_thief
                self.preview_observed = None

            def exec(self):
                self.appearance_preview_requested.emit("light", "middle_gray", 150)
                APP.processEvents()
                self.preview_observed = (
                    self.manager.theme_preference,
                    self.manager.viewer_backdrop,
                    self.manager.text_scale,
                )
                self.focus_thief.setFocus(Qt.FocusReason.OtherFocusReason)
                return self.DialogCode.Rejected

            @staticmethod
            def face_model_inventory_changed() -> bool:
                return False

        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.setValue("appearance/theme", "dark")
        store.setValue("appearance/viewer_backdrop", "black")
        store.setValue("appearance/text_scale", 125)
        store.sync()
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxAppearancePreviewTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.show()
            window.settings_button.setFocus(Qt.FocusReason.OtherFocusReason)
            APP.processEvents()
            dialog = PreviewDialog(window.theme_manager, window.library_workspace_button)

            with patch.object(window, "_create_settings_dialog", return_value=dialog):
                window.open_settings_dialog()

            self.assertEqual(("light", "middle_gray", 150), dialog.preview_observed)
            self.assertEqual("dark", window.theme_manager.theme_preference)
            self.assertEqual("black", window.theme_manager.viewer_backdrop)
            self.assertEqual(125, window.theme_manager.text_scale)
            self.assertEqual("dark", store.value("appearance/theme"))
            self.assertEqual("black", store.value("appearance/viewer_backdrop"))
            self.assertEqual(125, int(store.value("appearance/text_scale")))
            self.assertIs(window.settings_button, window.focusWidget())

    def test_shared_roots_manager_combines_scope_and_library_state(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            source = Path(tmp) / "photos"
            source.mkdir()
            window = ProductionClusterApp(activate_runtime_root("UiUxRootsManagerTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            self.assertTrue(
                Path(window.library_pane.catalog.db_path).is_relative_to(Path(self._runtime_root_fixture.name)),
                window.library_pane.catalog.db_path,
            )
            window.source_pane.set_active_roots([str(source)])
            window.library_pane.catalog.register_root(str(source))
            window.library_pane.refresh_roots()
            self._wait_for(lambda: window.library_pane._catalog_snapshot_job is None)
            window.set_active_workspace("library")
            window._edit_active_roots()
            window.show()
            APP.processEvents()

            self.assertTrue(window.source_pane.isVisible())
            self.assertTrue(window.library_pane.root_list.isHidden())
            window.source_pane.tabs.setCurrentIndex(1)
            self.assertEqual(1, window.source_pane.active_roots_list.count())
            item = window.source_pane.active_roots_list.item(0)
            self.assertIn("online", item.text())
            self.assertIn("0 photos", item.text())
            self.assertFalse(bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable))
            window.source_pane.active_roots_list.setCurrentItem(item)
            APP.processEvents()
            self.assertEqual("Disable", window.source_pane.library_root_action_button.text())
            self.assertTrue(window.source_pane.reveal_root_button.isEnabled())
            self.assertFalse(window.source_pane.pause_library_root_button.isEnabled())
            window.source_pane.set_library_root_job_state(str(source), "refreshing")
            APP.processEvents()
            self.assertIn("refreshing", window.source_pane.active_roots_list.currentItem().text())
            self.assertTrue(window.source_pane.pause_library_root_button.isEnabled())
            self.assertEqual("Storage", window.source_pane.data_home_button.text())
            window.source_pane.tabs.setCurrentIndex(0)
            window.source_pane.add_active_root(str(Path(tmp) / "draft-only"))
            self.assertTrue(window.source_pane.has_dirty_draft)
            self.assertFalse(window.source_pane.library_root_action_button.isEnabled())
            self._close_widget(window)
            APP.processEvents()

    def test_cleanup_duplicate_actions_flow_and_only_show_review_controls_for_a_group(self):
        from app.path_scope import PathScope
        from app.services.duplicate_review import DuplicateCandidate, DuplicateGroup
        from ui.library_pane import LibraryPane

        pane = LibraryPane(lambda: PathScope.from_paths(("/photos",)), lambda: None)
        self.addCleanup(self._close_widget, pane)
        scan_flow = pane.duplicate_scan_actions.layout()
        selection_flow = pane.duplicate_selection_actions.layout()
        self.assertEqual(4, scan_flow.count())
        self.assertEqual(4, selection_flow.count())
        self.assertTrue(pane.duplicate_selection_actions.isHidden())
        self.assertGreater(scan_flow.heightForWidth(240), scan_flow.heightForWidth(1600))
        for action_surface in (
            pane.batch_tool_actions,
            pane.recovery_actions,
            pane.people_review_actions,
        ):
            self.assertGreater(
                action_surface.layout().heightForWidth(220),
                action_surface.layout().heightForWidth(1600),
            )

        candidate = DuplicateCandidate("/photos/other.jpg", "", 1, 1, 1)
        group = DuplicateGroup(
            group_id="fixture",
            kind="exact",
            keeper_path="/photos/keeper.jpg",
            members=(DuplicateCandidate("/photos/keeper.jpg", "", 1, 1, 1), candidate),
            summary="Fixture duplicate group",
            review_keys=("fixture",),
        )
        pane._duplicate_groups = [group]
        pane.duplicate_list.addItem("Hash duplicates · 2 photos")
        pane.duplicate_list.item(0).setData(Qt.ItemDataRole.UserRole, 0)
        pane.duplicate_list.setCurrentRow(0)
        APP.processEvents()
        self.assertFalse(pane.duplicate_selection_actions.isHidden())
        self.assertTrue(pane.open_duplicate_group_button.isEnabled())
        self.assertTrue(pane.not_duplicate_button.isEnabled())
        self.assertEqual(
            ["/photos/keeper.jpg", "/photos/other.jpg"],
            pane.duplicate_preview_gallery._model.all_paths(),  # noqa: SLF001 - verifies the virtual review model
        )
        self.assertEqual(
            "Selected duplicate group photo preview",
            pane.duplicate_preview_gallery.table.accessibleName(),
        )
        opened: list[tuple[list[str], str]] = []
        pane.open_in_gallery_requested.connect(lambda paths, title: opened.append((paths, title)))
        pane.open_duplicate_group_button.click()
        self.assertEqual(
            [(["/photos/keeper.jpg", "/photos/other.jpg"], "Hash duplicate review: keeper.jpg")],
            opened,
        )

    def test_face_detect_actions_wrap_even_when_refresh_actions_are_disabled(self):
        from ui.search_pane import SearchPane

        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        self.addCleanup(self._close_widget, pane)
        pane.face_refresh_faces_button.setEnabled(False)
        pane.face_auto_clean_button.setEnabled(False)
        flow = pane.face_scan_actions_grid

        self.assertEqual(8, flow.count())
        self.assertGreater(flow.heightForWidth(280), flow.heightForWidth(1600))
        self.assertFalse(pane.face_refresh_faces_button.isEnabled())
        self.assertFalse(pane.face_auto_clean_button.isEnabled())
        for index in range(flow.count()):
            button = flow.itemAt(index).widget()
            self.assertTrue(button.toolTip(), button.text())

    def test_library_search_filters_are_disclosed_and_build_a_catalog_query(self):
        from app.path_scope import PathScope
        from ui.library_pane import LibraryPane

        pane = LibraryPane(lambda: PathScope.from_paths(("/photos",)), lambda: None)
        self.addCleanup(self._close_widget, pane)
        pane.tabs.setCurrentIndex(pane.tabs.indexOf(pane.search_page))
        pane.search_advanced_toggle.setChecked(True)
        pane.search_start.setText("2024-01-01")
        pane.search_end.setText("2024-12-31")
        pane.search_camera.setText("Fixture Camera")
        pane.search_folder.setText("/photos/trip")
        pane.search_file_ext.setText(".jpg")
        pane._load_assets = lambda *, reset: None  # type: ignore[method-assign]

        pane.run_search()

        self.assertFalse(pane.search_advanced_filters.isHidden())
        self.assertEqual("2024-01-01", pane._active_query.start_at)
        self.assertEqual("2024-12-31", pane._active_query.end_at)
        self.assertEqual("Fixture Camera", pane._active_query.camera)
        self.assertEqual("/photos/trip", pane._active_query.folder)
        self.assertEqual(".jpg", pane._active_query.file_ext)

    def test_gallery_action_flow_wraps_before_it_elides_actions(self):
        from ui.gallery_pane import GalleryPane

        pane = GalleryPane()
        self.addCleanup(self._close_widget, pane)
        self.assertGreater(pane.action_bar.heightForWidth(420), pane.action_bar.heightForWidth(1400))

    def test_faces_basic_mode_is_human_only_and_hides_technical_administration(self):
        from apps.pyqt_production.app import ProductionClusterApp

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            window = ProductionClusterApp(activate_runtime_root("UiUxFacesBasicTest"))
            self.addCleanup(self._close_widget, window)
            self._wait_for(lambda: window._storage_usage_thread is None)
            window.show()
            window.set_active_workspace("faces")
            self._wait_for(lambda: window.faces_pane is not None)
            pane = window.faces_pane
            pane.set_ui_mode("basic")
            pane.tabs.setCurrentIndex(pane.tab_labels().index("Folder Review"))
            APP.processEvents()

            self.assertEqual(("human",), pane.supported_face_modes)
            self.assertEqual(["All Faces", "Folder Review", "Face Search", "Identities"], pane.tab_labels())
            self.assertTrue(pane.tabs.tabBar().isHidden())
            self.assertEqual(
                ["All Faces", "Detect", "Find", "Manage"],
                [pane.task_navigation.tabText(index) for index in range(pane.task_navigation.count())],
            )
            self.assertTrue(pane.face_library_tabs.tabBar().isTabVisible(1))
            for name in (
                "face_settings_group",
                "face_identity_management_group",
                "face_identity_danger_group",
                "face_delete_db_button",
                "face_purge_data_button",
            ):
                self.assertFalse(getattr(pane, name).isVisibleTo(pane), name)
            for name in (
                "face_db_scope_field",
                "face_upload_to_global_button",
                "face_database_path_label",
                "face_review_source_summary",
                "face_model_status_dashboard_label",
            ):
                self.assertFalse(hasattr(pane, name), name)
            self.assertTrue(pane.face_pipeline_summary_group.isVisibleTo(pane))
            self.assertTrue(pane.face_choose_pipeline_button.isVisibleTo(pane))
            self.assertTrue(pane.face_model_settings_button.isVisibleTo(pane))
            self.assertRegex(pane.face_model_summary_label.text(), r" — (CPU|GPU)$")
            self.assertTrue(pane.face_scan_button.isVisibleTo(pane))
            self.assertNotIn("dog", pane.face_mode_combo.currentText().casefold())
            self.assertNotIn("cat", pane.face_mode_combo.currentText().casefold())
            self._close_widget(window)
            APP.processEvents()

    def test_settings_categories_keep_recovery_human_readable_and_release_gates_out(self):
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
        from app.services.cache_maintenance import CacheClearResult
        from infra.runtime import RuntimeCapabilityService

        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"IMAGE_CLUSTERING_APP_DIR": tmp}, clear=False):
            settings_mod._RUNTIME_BASE_DIR = None
            layout = activate_runtime_root("UiUxSettingsTest")
            store = QSettings("ClusterLensTests", "UiUxSettingsTest")
            with patch.object(ProductionSettingsDialog, "refresh_runtime_diagnostics"), patch.object(
                ProductionSettingsDialog, "refresh_cache_usage"
            ), patch.object(ProductionSettingsDialog, "refresh_model_inventory"), patch.object(
                ProductionSettingsDialog, "refresh_operation_journal"
            ), patch.object(ProductionSettingsDialog, "refresh_log_viewer"):
                dialog = ProductionSettingsDialog(
                    store,
                    RuntimeCapabilityService(),
                    runtime_layout=layout,
                    support_metadata_provider=lambda: {},
                )
            self.addCleanup(self._close_widget, dialog)

            self.assertEqual("system", dialog.application_theme.currentData())
            self.assertEqual("adaptive_neutral", dialog.viewer_backdrop.currentData())
            self.assertEqual(100, dialog.text_scale.currentData())
            dialog.application_theme.setCurrentIndex(dialog.application_theme.findData("light"))
            dialog.viewer_backdrop.setCurrentIndex(dialog.viewer_backdrop.findData("middle_gray"))
            dialog.text_scale.setCurrentIndex(dialog.text_scale.findData(150))
            self.assertEqual("light", dialog.values()["appearance/theme"])
            self.assertEqual("middle_gray", dialog.values()["appearance/viewer_backdrop"])
            self.assertEqual(150, dialog.values()["appearance/text_scale"])
            self.assertIn("Text: 150%", dialog.appearance_summary.text())

            self.assertEqual(
                ["General", "Sources", "Compute", "Storage", "Advanced", "About"],
                [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())],
            )
            for action_surface in (dialog.model_actions, dialog.storage_actions, dialog.journal_actions):
                self.assertGreater(
                    action_surface.layout().heightForWidth(300),
                    action_surface.layout().heightForWidth(1800),
                )
            self.assertEqual(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                dialog.settings_section_tabs["Storage"][2].horizontalScrollBarPolicy(),
            )
            self.assertEqual(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                dialog.settings_section_tabs["Safety & Recovery"][2].horizontalScrollBarPolicy(),
            )
            for section in ("General", "Sources & Library", "Performance", "Updates", "Support"):
                self.assertEqual(
                    Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
                    dialog.settings_section_tabs[section][2].horizontalScrollBarPolicy(),
                    section,
                )
            for action_surface in (
                dialog.support_actions_row,
                dialog.support_log_actions_row,
                dialog.about_actions_row,
            ):
                self.assertEqual("ResponsiveFlowLayout", action_surface.layout().__class__.__name__)
            self.assertGreater(
                dialog.support_actions_row.layout().heightForWidth(300),
                dialog.support_actions_row.layout().heightForWidth(1800),
            )
            self.assertFalse(hasattr(dialog, "run_release_gates_button"))
            safety = dialog.tabs.widget(3)
            safety_buttons = [button.text() for button in safety.findChildren(QPushButton)]
            self.assertNotIn("Open Journal", safety_buttons)
            self.assertIn("Restore (skip conflicts)", safety_buttons)
            self.assertIn("Restore with unique names", safety_buttons)
            self.assertEqual(
                "Incomplete Data Home backups",
                dialog.data_home_backup_recovery_combo.accessibleName(),
            )
            dialog.data_home_manager = SimpleNamespace()
            dialog._set_data_home_backup_recovery_entries(
                (
                    {
                        "status": "failed",
                        "backup_root": "/backups/incomplete",
                        "journal_path": "/runtime/support/data_home_backups/incomplete.json",
                    },
                    {
                        "status": "published",
                        "backup_root": "/backups/published",
                        "journal_path": "/runtime/support/data_home_backups/published.json",
                    },
                )
            )
            self.assertTrue(dialog.discard_backup_staging_button.isEnabled())
            self.assertEqual("Discard Staged Backup", dialog.discard_backup_staging_button.text())
            dialog.data_home_backup_recovery_combo.setCurrentIndex(1)
            APP.processEvents()
            self.assertEqual("Finalize Published Backup", dialog.discard_backup_staging_button.text())
            dialog.data_home_manager = None
            dialog.clear_rebuildable_caches = lambda **_kwargs: CacheClearResult(
                cleared_targets=("cluster_results/",),
                freed_bytes=4096,
                failures=("cluster_results/: cleanup paused",),
                completion_survives_cancellation=True,
                retry_targets=("cluster_results/",),
            )
            dialog.can_clear_rebuildable_caches = lambda: True
            with patch("apps.pyqt_production.settings_dialog.confirmBox", return_value=True), patch(
                "apps.pyqt_production.settings_dialog.errorBox"
            ), patch("apps.pyqt_production.settings_dialog.infoBox"), patch.object(
                dialog, "refresh_cache_usage"
            ):
                dialog._clear_rebuildable_caches()
                self._wait_for(lambda: dialog._cache_clear_thread is None)
            self.assertIn("cleanup is still pending for cluster_results/", dialog.cache_status_label.text())
            self.assertIn("again to retry", dialog.cache_status_label.text())
            support = dialog.tabs.widget(4)
            self.assertIn("Open operation journal", [button.text() for button in support.findChildren(QPushButton)])
            face_models = dialog.tabs.widget(2)
            model_buttons = [button.text() for button in face_models.findChildren(QPushButton)]
            self.assertIn("Install Selected Face Pack", model_buttons)
            self.assertIn("Delete Installed Face Component", model_buttons)
            self.assertIn("Clear Face Download Cache", model_buttons)
            self.assertIn("Choose Face Model Folder", model_buttons)
            self.assertGreaterEqual(dialog.face_model_pack_combo.count(), 8)
            self.assertEqual("latest_gpu", dialog.face_model_pack_combo.currentData())
            self.assertIn("SCRFD 10G + ArcFace R100", dialog.face_model_pack_combo.currentText())
            dialog.settings_search.setText("SCRFD")
            APP.processEvents()
            self.assertEqual("Compute", dialog.tabs.tabText(dialog.tabs.currentIndex()))
            dialog.resize(724, 580)
            self.assertTrue(dialog.select_section("Clustering Models"))
            dialog.show()
            APP.processEvents()
            clustering_models = dialog.settings_section_tabs["Clustering Models"][2]
            self.assertEqual("models_scroll_area", clustering_models.objectName())
            self.assertEqual(Qt.ScrollBarPolicy.ScrollBarAlwaysOff, clustering_models.horizontalScrollBarPolicy())
            self.assertGreaterEqual(clustering_models.verticalScrollBar().maximum(), 0)
            dialog.close()
            APP.processEvents()

    def test_identity_model_pages_and_filters_one_hundred_thousand_rows_within_budget(self):
        model = PagedListEntryModel(page_size=50)
        items = [ListEntry(f"Person {index:06d}", payload=index) for index in range(100_000)]
        model.set_source_items(items)
        self.assertEqual(50, model.rowCount())
        self.assertTrue(model.canFetchMore())

        started = time.perf_counter()
        model.set_filter_text("099999")
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self.assertLessEqual(elapsed_ms, 100.0)
        self.assertEqual(1, model.total_count)
        self.assertEqual(1, model.rowCount())

    def test_operation_journal_model_pages_without_resetting_selection(self):
        from apps.pyqt_production.settings_dialog import OperationJournalTableModel
        from PyQt6.QtWidgets import QTableView

        model = OperationJournalTableModel(page_size=50)
        model.set_entries([{"operation_id": str(index), "operation": "move"} for index in range(125)])
        view = QTableView()
        view.setModel(model)
        selection = view.selectionModel()
        selection.select(model.index(4, 0), QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows)
        selected_id = model.data(selection.selectedRows()[0], Qt.ItemDataRole.UserRole)["operation_id"]

        model.fetchMore()
        self.assertEqual(100, model.rowCount())
        self.assertEqual(selected_id, model.data(selection.selectedRows()[0], Qt.ItemDataRole.UserRole)["operation_id"])
        view.close()

    def test_required_icons_render_at_multiple_logical_sizes(self):
        for name in ("workspace", "folder", "search", "scan", "settings", "jobs", "health", "recovery", "restore", "warning", "delete"):
            icon = themed_icon(name)
            self.assertFalse(icon.isNull(), name)
            for size in (16, 24, 32, 48):
                pixmap = icon.pixmap(size, size)
                self.assertFalse(pixmap.isNull(), f"{name}@{size}")


if __name__ == "__main__":
    unittest.main()
