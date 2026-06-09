from __future__ import annotations

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.services.gallery_actions import GalleryActionService


class ProductionSafetyTests(unittest.TestCase):
    def test_gallery_file_operation_journal_can_restore_trash_moves(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            source.write_bytes(b"image-bytes")
            audit_log = root / "logs" / "file_operations.jsonl"
            action_service = GalleryActionService(audit_log_path=audit_log, temp_dir=root / "cache" / "tmp" / "file_ops")

            deleted = action_service.move_to_trash([str(source)])
            self.assertFalse(source.exists())
            self.assertEqual(1, len(deleted.changed_paths))
            trash_path = Path(deleted.changed_paths[0][1])
            self.assertTrue(trash_path.exists())

            restored = action_service.restore_changed_paths(list(deleted.changed_paths))
            self.assertFalse(restored.failures)
            self.assertTrue(source.exists())
            self.assertFalse(trash_path.exists())
            self.assertEqual(b"image-bytes", source.read_bytes())
            entries = action_service.read_audit_entries(limit=10)
            self.assertEqual(["delete_to_trash", "restore"], [entry["operation"] for entry in entries])


if __name__ == "__main__":
    unittest.main()
