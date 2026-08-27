from __future__ import annotations

import sqlite3
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.services.gallery_actions import GalleryActionService


class ProductionSafetyTests(unittest.TestCase):
    def test_gallery_action_service_constructor_performs_no_filesystem_or_database_writes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_log = root / "not-created" / "file_operations.jsonl"
            service = GalleryActionService(audit_log_path=audit_log, temp_dir=root / "not-created" / "temp")

            self.assertFalse(audit_log.parent.exists())
            self.assertFalse(service.journal_path.exists())
            self.assertFalse(service.temp_dir.exists())

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
            self.assertTrue(Path(deleted.journal_path).exists())
            with sqlite3.connect(deleted.journal_path) as db:
                recovery_states = [row[0] for row in db.execute("select recovery_status from operations order by timestamp_utc")]
            self.assertIn("restorable", recovery_states)
            self.assertIn("restored", recovery_states)

    def test_recovery_skip_conflict_preserves_both_files_and_unique_name_restores(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            source.write_bytes(b"moved")
            service = GalleryActionService(audit_log_path=root / "logs" / "file_operations.jsonl")
            deleted = service.move_to_trash([str(source)])
            trash_path = Path(deleted.changed_paths[0][1])
            source.write_bytes(b"replacement")

            skipped = service.restore_changed_paths(list(deleted.changed_paths), conflict_policy="skip")
            self.assertEqual(b"replacement", source.read_bytes())
            self.assertTrue(trash_path.exists())
            self.assertEqual(1, len(skipped.failures))
            self.assertIn("already exists", skipped.failures[0])

            restored = service.restore_changed_paths(list(deleted.changed_paths), conflict_policy="unique_name")
            self.assertFalse(restored.failures)
            unique_path = Path(restored.changed_paths[0][1])
            self.assertNotEqual(source, unique_path)
            self.assertEqual(b"replacement", source.read_bytes())
            self.assertEqual(b"moved", unique_path.read_bytes())
            self.assertFalse(trash_path.exists())

    def test_recovery_reports_missing_restore_source_and_persists_after_restart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit_log = root / "logs" / "file_operations.jsonl"
            source = root / "source.jpg"
            source.write_bytes(b"image")
            service = GalleryActionService(audit_log_path=audit_log)
            deleted = service.move_to_trash([str(source)])
            Path(deleted.changed_paths[0][1]).unlink()

            restored = service.restore_changed_paths(list(deleted.changed_paths))
            self.assertEqual(1, len(restored.failures))
            self.assertIn("restore source does not exist", restored.failures[0])

            reopened = GalleryActionService(audit_log_path=audit_log)
            entries = reopened.read_audit_entries(limit=20)
            self.assertEqual(["delete_to_trash", "restore"], [entry["operation"] for entry in entries])
            self.assertEqual(1, entries[-1]["failure_count"])
            self.assertEqual("failed", entries[-1]["file_results"][0]["status"])

    def test_unavailable_operation_journal_blocks_file_mutation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            source.write_bytes(b"image")
            service = GalleryActionService(audit_log_path=root / "logs" / "file_operations.jsonl")

            with patch.object(service, "_connect_journal", side_effect=sqlite3.OperationalError("read-only storage")):
                result = service.move_to_trash([str(source)])

            self.assertTrue(source.exists())
            self.assertFalse((root / "TrashImages").exists())
            self.assertEqual(1, len(result.failures))
            self.assertIn("no files were changed", result.failures[0])


if __name__ == "__main__":
    unittest.main()
