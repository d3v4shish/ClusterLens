from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QItemSelectionModel, QRect, QSettings, Qt
from PyQt6.QtWidgets import QApplication, QComboBox, QPushButton, QToolButton, QWidget


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
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        self._production_settings_snapshot = {
            key: store.value(key)
            for key in store.allKeys()
        }
        store.setValue("runtime/preferred_mode", "cpu")
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
        settings_mod._RUNTIME_BASE_DIR = None
        store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
        store.clear()
        for key, value in self._production_settings_snapshot.items():
            store.setValue(key, value)
        store.sync()

    @staticmethod
    def _close_widget(widget) -> None:
        try:
            widget.close()
        except RuntimeError:
            pass
        APP.processEvents()

    def test_minimum_screen_contract_accepts_landscape_and_portrait(self):
        from apps.pyqt_production.app import _screen_geometry_supported

        self.assertTrue(_screen_geometry_supported(QRect(0, 0, 1920, 1080)))
        self.assertTrue(_screen_geometry_supported(QRect(0, 0, 1080, 1920)))
        self.assertFalse(_screen_geometry_supported(QRect(0, 0, 1919, 1079)))

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

            self.assertLessEqual(len(visible_focus_controls), 7)
            labels = [
                widget.currentText() if isinstance(widget, QComboBox) else widget.text()
                for widget in visible_focus_controls
            ]
            active_mode = window.mode_selector.currentText()
            self.assertIn(active_mode, {"Basic", "Advanced"})
            for label in ("Clustering", "Faces", active_mode, "View", "Settings"):
                self.assertIn(label, labels)
            self.assertTrue(window.cluster_actions_menu_action.isVisible() is False)
            window.set_active_workspace("clustering")
            window.set_clustering_mode("advanced")
            self.assertTrue(window.cluster_actions_menu_action.isVisible())
            self.assertTrue(all(widget.isHidden() for widget in (
                window.source_toggle,
                window.controls_toggle,
                window.details_toggle,
                window.tag_manager_button,
                window.suggest_tags_button,
                window.cluster_actions_button,
            )))
            window.close()
            APP.processEvents()

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
                ["All Faces", "Detect Faces", "Find a Person", "People & Groups"],
                [pane.task_navigation.tabText(index) for index in range(pane.task_navigation.count())],
            )
            for name in (
                "face_settings_group",
                "face_identity_management_group",
                "face_identity_danger_group",
                "face_db_scope_field",
                "face_delete_db_button",
                "face_purge_data_button",
            ):
                self.assertFalse(getattr(pane, name).isVisibleTo(pane), name)
            self.assertTrue(pane.face_pipeline_summary_group.isVisibleTo(pane))
            self.assertTrue(pane.face_choose_pipeline_button.isVisibleTo(pane))
            self.assertTrue(pane.face_model_settings_button.isVisibleTo(pane))
            self.assertIn("Configured default pending install", pane.face_model_status_dashboard_label.text())
            self.assertIn("scrfd_10g_kps + arcface_r100_glint360k", pane.face_model_status_dashboard_label.text())
            self.assertTrue(pane.face_scan_button.isVisibleTo(pane))
            self.assertNotIn("dog", pane.face_mode_combo.currentText().casefold())
            self.assertNotIn("cat", pane.face_mode_combo.currentText().casefold())
            window.close()
            APP.processEvents()

    def test_settings_categories_keep_recovery_human_readable_and_release_gates_out(self):
        from apps.pyqt_production.settings_dialog import ProductionSettingsDialog
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

            self.assertEqual(
                ["General", "Performance", "Models", "Storage", "Safety & Recovery", "Updates", "Support"],
                [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())],
            )
            self.assertFalse(hasattr(dialog, "run_release_gates_button"))
            safety = dialog.tabs.widget(4)
            safety_buttons = [button.text() for button in safety.findChildren(QPushButton)]
            self.assertNotIn("Open Journal", safety_buttons)
            self.assertIn("Restore (skip conflicts)", safety_buttons)
            self.assertIn("Restore with unique names", safety_buttons)
            support = dialog.tabs.widget(6)
            self.assertIn("Open operation journal", [button.text() for button in support.findChildren(QPushButton)])
            models = dialog.tabs.widget(2)
            model_buttons = [button.text() for button in models.findChildren(QPushButton)]
            self.assertIn("Install Selected Face Pack", model_buttons)
            self.assertIn("Delete Installed Face Component", model_buttons)
            self.assertIn("Clear Face Download Cache", model_buttons)
            self.assertGreaterEqual(dialog.face_model_pack_combo.count(), 8)
            self.assertEqual("latest_gpu", dialog.face_model_pack_combo.currentData())
            self.assertIn("SCRFD 10G + ArcFace R100", dialog.face_model_pack_combo.currentText())
            dialog.resize(724, 580)
            dialog.tabs.setCurrentIndex(2)
            dialog.show()
            APP.processEvents()
            self.assertEqual("models_scroll_area", models.objectName())
            self.assertEqual(Qt.ScrollBarPolicy.ScrollBarAlwaysOff, models.horizontalScrollBarPolicy())
            self.assertGreater(models.verticalScrollBar().maximum(), 0)
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
