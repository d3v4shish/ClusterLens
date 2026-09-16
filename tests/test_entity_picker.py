from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from ui.entity_picker import EntityPicker, EntityPickerDialog


APP = QApplication.instance() or QApplication([])


class EntityPickerTests(unittest.TestCase):
    def test_saved_value_commits_on_enter_and_tab(self) -> None:
        for key, expected in ((Qt.Key.Key_Return, "Alice"), (Qt.Key.Key_Tab, "Bob")):
            picker = EntityPicker(entity_label="person name")
            picker.set_choices(["Alice", "Bob"])
            picker.show()
            try:
                picker.setFocus()
                APP.processEvents()
                completer = picker._name_completer
                completer.setCompletionPrefix(expected[:2].lower())
                completer.complete()
                APP.processEvents()
                self.assertTrue(completer.setCurrentRow(0))
                QTest.keyClick(picker, key)
                APP.processEvents()
                self.assertEqual(expected, picker.text())
                self.assertFalse(completer.popup().isVisible())
            finally:
                picker.close()

    def test_create_row_is_explicit_and_returns_typed_value(self) -> None:
        picker = EntityPicker(entity_label="person name")
        created: list[str] = []
        picker.create_requested.connect(created.append)
        picker.set_choices(["Alice"])
        picker.setText("Cara")
        picker._update_suggestions("Cara")
        self.assertEqual(["Create \u201cCara\u201d"], picker._name_model.stringList())
        picker._commit_completion("Create \u201cCara\u201d")
        self.assertEqual("Cara", picker.text())
        self.assertEqual(["Cara"], created)

    def test_picker_dialog_canonicalizes_saved_values(self) -> None:
        dialog = EntityPickerDialog(
            "Assign person",
            "Choose a person.",
            choices=["Alice"],
            initial="alice",
            entity_label="person name",
        )
        try:
            self.assertEqual("Alice", dialog.selected_value())
        finally:
            dialog.close()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
