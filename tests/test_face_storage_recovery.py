from __future__ import annotations

import errno
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.services.face_storage_recovery import FaceStorageRemovalService
from app.services import face_storage_recovery as recovery_module
from infra.cancel import Cancelled


class SimulatedProcessExit(BaseException):
    pass


class FaceStorageRecoveryTests(unittest.TestCase):
    @staticmethod
    def _write_managed_files(root: Path) -> tuple[Path, dict[str, bytes]]:
        database = root / "face_search_human.db"
        manifest: dict[str, bytes] = {}
        for index, path in enumerate(FaceStorageRemovalService.managed_paths(database), start=1):
            payload = f"managed-{index}".encode("utf-8")
            path.write_bytes(payload)
            manifest[path.name] = payload
        return database, manifest

    @staticmethod
    def _assert_manifest(root: Path, manifest: dict[str, bytes], *, present: bool) -> None:
        for name, payload in manifest.items():
            path = root / name
            if present:
                assert path.read_bytes() == payload
            else:
                assert not path.exists()

    def test_committed_removal_is_scoped_recoverable_and_discard_is_explicit(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            database, manifest = self._write_managed_files(root)
            unrelated = root / "family-photo.jpg"
            unrelated.write_bytes(b"user-photo")
            service = FaceStorageRemovalService(root)

            result = service.remove(database)

            self._assert_manifest(root, manifest, present=False)
            self.assertEqual(b"user-photo", unrelated.read_bytes())
            self.assertEqual(set(manifest), {Path(path).name for path in result.staged_paths})
            first = service.recover_incomplete()
            second = service.recover_incomplete()
            self.assertEqual((result.operation_id,), first.committed_operations)
            self.assertEqual(first, second)

            restored = service.restore_committed(result.operation_id)
            self.assertEqual(set(manifest), {Path(path).name for path in restored})
            self._assert_manifest(root, manifest, present=True)
            self.assertEqual(FaceStorageRemovalService(root).recover_incomplete().committed_operations, ())

            second_result = service.remove(database)
            service.discard_committed(second_result.operation_id)
            self.assertFalse(Path(second_result.recovery_path).exists())
            self._assert_manifest(root, manifest, present=False)
            self.assertEqual(b"user-photo", unrelated.read_bytes())

    def test_cancellation_at_every_boundary_restores_before_commit(self):
        checkpoints = self._checkpoint_names()
        for target_checkpoint in checkpoints:
            with self.subTest(checkpoint=target_checkpoint), TemporaryDirectory() as tmp:
                root = Path(tmp)
                database, manifest = self._write_managed_files(root)
                service = FaceStorageRemovalService(root)
                cancelled = False

                def _cancel_at(name: str) -> None:
                    nonlocal cancelled
                    if name == target_checkpoint:
                        cancelled = True

                if target_checkpoint == "after_commit":
                    result = service.remove(database, cancel_check=lambda: cancelled, checkpoint=_cancel_at)
                    self.assertTrue(result.committed)
                    self._assert_manifest(root, manifest, present=False)
                    self.assertEqual((result.operation_id,), service.recover_incomplete().committed_operations)
                else:
                    with self.assertRaises(Cancelled):
                        service.remove(database, cancel_check=lambda: cancelled, checkpoint=_cancel_at)
                    self._assert_manifest(root, manifest, present=True)
                    self.assertEqual((), service.recover_incomplete().restored_operations)
                    self.assertEqual([], list(service.recovery_root.glob("*/journal.json")))

    def test_process_exit_at_every_boundary_recovers_idempotently(self):
        for target_checkpoint in self._checkpoint_names():
            with self.subTest(checkpoint=target_checkpoint), TemporaryDirectory() as tmp:
                root = Path(tmp)
                database, manifest = self._write_managed_files(root)
                service = FaceStorageRemovalService(root)

                def _exit_at(name: str) -> None:
                    if name == target_checkpoint:
                        raise SimulatedProcessExit(name)

                with self.assertRaises(SimulatedProcessExit):
                    service.remove(database, checkpoint=_exit_at)

                first = FaceStorageRemovalService(root).recover_incomplete()
                second = FaceStorageRemovalService(root).recover_incomplete()
                if target_checkpoint == "after_commit":
                    self._assert_manifest(root, manifest, present=False)
                    self.assertEqual(1, len(first.committed_operations))
                    self.assertEqual(first, second)
                else:
                    self._assert_manifest(root, manifest, present=True)
                    self.assertEqual(1, len(first.restored_operations))
                    self.assertEqual((), first.failures)
                    self.assertEqual((), second.restored_operations)
                    self.assertEqual((), second.committed_operations)

    def test_io_failure_rolls_back_without_touching_unrelated_files(self):
        for target_checkpoint in self._checkpoint_names():
            if target_checkpoint == "after_commit":
                continue
            with self.subTest(checkpoint=target_checkpoint), TemporaryDirectory() as tmp:
                root = Path(tmp)
                database, manifest = self._write_managed_files(root)
                unrelated = root / "unrelated.bin"
                unrelated.write_bytes(b"keep")

                def _fail_at(name: str) -> None:
                    if name == target_checkpoint:
                        raise OSError("injected I/O failure")

                with self.assertRaises(OSError):
                    FaceStorageRemovalService(root).remove(database, checkpoint=_fail_at)
                self._assert_manifest(root, manifest, present=True)
                self.assertEqual(b"keep", unrelated.read_bytes())

    def test_permission_and_disk_full_faults_restore_every_precommit_boundary(self):
        fault_factories = (
            lambda: PermissionError(errno.EACCES, "permission denied"),
            lambda: OSError(errno.ENOSPC, "disk full"),
        )
        for target_checkpoint in self._checkpoint_names():
            if target_checkpoint == "after_commit":
                continue
            for fault_factory in fault_factories:
                with self.subTest(checkpoint=target_checkpoint, fault=fault_factory().__class__.__name__), TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    database, manifest = self._write_managed_files(root)
                    unrelated = root / "unrelated.bin"
                    unrelated.write_bytes(b"keep")

                    def _fail_at(name: str) -> None:
                        if name == target_checkpoint:
                            raise fault_factory()

                    with self.assertRaises(OSError):
                        FaceStorageRemovalService(root).remove(database, checkpoint=_fail_at)
                    self._assert_manifest(root, manifest, present=True)
                    self.assertEqual(b"keep", unrelated.read_bytes())

    def test_atomic_journal_write_failure_at_every_boundary_restores(self):
        journal_write_count = 2 * len(FaceStorageRemovalService.managed_paths("faces.db")) + 3
        original_write = recovery_module.atomic_write_text
        for failing_write in range(1, journal_write_count + 1):
            with self.subTest(journal_write=failing_write), TemporaryDirectory() as tmp:
                root = Path(tmp)
                database, manifest = self._write_managed_files(root)
                calls = 0

                def _write(path, payload, *, encoding="utf-8"):
                    nonlocal calls
                    calls += 1
                    if calls == failing_write:
                        raise OSError("injected atomic journal failure")
                    return original_write(path, payload, encoding=encoding)

                with patch.object(recovery_module, "atomic_write_text", side_effect=_write):
                    with self.assertRaises(OSError):
                        FaceStorageRemovalService(root).remove(database)
                self._assert_manifest(root, manifest, present=True)
                recovery_root = FaceStorageRemovalService(root).recovery_root
                self.assertEqual([], list(recovery_root.glob("*/journal.json")))

    def test_recovery_rejects_unmanaged_or_conflicting_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            database, manifest = self._write_managed_files(root)
            service = FaceStorageRemovalService(root)

            def _exit_after_first_rename(name: str) -> None:
                if name.startswith("after_rename:"):
                    raise SimulatedProcessExit(name)

            with self.assertRaises(SimulatedProcessExit):
                service.remove(database, checkpoint=_exit_after_first_rename)
            operation_root = next(path.parent for path in service.recovery_root.glob("*/journal.json"))
            journal_path = operation_root / "journal.json"
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            journal["entries"][0]["name"] = "family-photo.jpg"
            journal_path.write_text(json.dumps(journal), encoding="utf-8")
            unrelated = root / "family-photo.jpg"
            unrelated.write_bytes(b"keep")

            report = service.recover_incomplete()

            self.assertEqual(1, len(report.failures))
            self.assertEqual(b"keep", unrelated.read_bytes())
            self.assertFalse((root / next(iter(manifest))).exists())

    def test_checkpoint_refuses_active_wal_reader(self):
        with TemporaryDirectory() as tmp:
            database = Path(tmp) / "face_search_human.db"
            writer = sqlite3.connect(database)
            reader = sqlite3.connect(database)
            try:
                writer.execute("PRAGMA journal_mode=WAL;")
                writer.execute("PRAGMA wal_autocheckpoint=0;")
                writer.execute("CREATE TABLE faces (value INTEGER)")
                writer.execute("INSERT INTO faces VALUES (1)")
                writer.commit()
                reader.execute("BEGIN")
                self.assertEqual((1,), reader.execute("SELECT value FROM faces").fetchone())
                writer.execute("INSERT INTO faces VALUES (2)")
                writer.commit()

                with self.assertRaisesRegex(RuntimeError, "in use"):
                    FaceStorageRemovalService.checkpoint_database(database)
            finally:
                reader.close()
                writer.close()

    def test_database_must_be_inside_the_exact_managed_root(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside"
            outside.mkdir()
            database = outside / "face_search.db"
            database.write_bytes(b"db")

            with self.assertRaisesRegex(ValueError, "immediate child"):
                FaceStorageRemovalService(root).remove(database)
            self.assertEqual(b"db", database.read_bytes())

    def _checkpoint_names(self) -> tuple[str, ...]:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            database, _manifest = self._write_managed_files(root)
            observed: list[str] = []
            service = FaceStorageRemovalService(root)
            result = service.remove(database, checkpoint=observed.append)
            service.restore_committed(result.operation_id)
            return tuple(observed)


if __name__ == "__main__":
    unittest.main()
