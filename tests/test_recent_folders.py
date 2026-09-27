from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.ui.recent_folders import RecentFolderHistory


class _Store:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def value(self, key: str, default=None):
        return self.values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self.values[key] = value


class RecentFolderHistoryTests(unittest.TestCase):
    def test_history_is_persistent_deduplicated_and_bounded(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folders = []
            for index in range(12):
                folder = root / f"folder-{index}"
                folder.mkdir()
                folders.append(folder)
            store = _Store()
            history = RecentFolderHistory(store)
            for folder in folders:
                self.assertTrue(history.record(str(folder)))

            self.assertEqual(10, len(history.paths()))
            self.assertEqual(str(folders[-1].resolve()), history.paths()[0])
            self.assertFalse(history.record(str(folders[-1] / ".")))

            restored = RecentFolderHistory(store)
            self.assertEqual(history.paths(), restored.paths())
            payload = json.loads(str(store.values[history.key]))
            self.assertEqual(1, payload["version"])

    def test_remove_clear_and_invalid_payload_are_safe(self) -> None:
        with TemporaryDirectory() as tmp:
            folder = Path(tmp).resolve()
            store = _Store()
            store.values["workspace/recent_folders_v1"] = "not-json"
            history = RecentFolderHistory(store)
            self.assertEqual([], history.paths())
            self.assertFalse(history.record(str(folder / "missing")))
            self.assertTrue(history.record(str(folder)))
            self.assertTrue(history.remove(str(folder)))
            self.assertEqual([], history.paths())
            self.assertFalse(history.clear())

    def test_restore_is_lexical_and_never_probes_saved_directories(self) -> None:
        store = _Store()
        store.values["workspace/recent_folders_v1"] = json.dumps(
            {"version": 1, "paths": ["/disconnected/archive", "/disconnected/archive"]}
        )

        with patch("src.ui.recent_folders.Path.is_dir", side_effect=AssertionError("unexpected filesystem probe")):
            history = RecentFolderHistory(store)

        self.assertEqual(["/disconnected/archive"], history.paths())


if __name__ == "__main__":
    unittest.main()
