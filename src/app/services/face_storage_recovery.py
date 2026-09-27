from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Callable
from uuid import uuid4

from infra.atomic_io import atomic_write_text
from infra.cancel import Cancelled, raise_if_cancelled


JOURNAL_VERSION = 1
RECOVERY_DIRECTORY_NAME = ".face_storage_recovery"


@dataclass(frozen=True)
class FaceStorageRemovalResult:
    operation_id: str
    db_path: str
    staged_paths: tuple[str, ...]
    recovery_path: str
    committed: bool = True
    completion_survives_cancellation: bool = True


@dataclass(frozen=True)
class FaceStorageRecoveryReport:
    restored_operations: tuple[str, ...]
    committed_operations: tuple[str, ...]
    failures: tuple[str, ...]


class FaceStorageRemovalService:
    """Journal and quarantine the exact files owned by one face database.

    A removal is reversible until its journal is marked ``committed``.  An
    interrupted pre-commit operation is restored on the next recovery pass;
    committed payloads remain in quarantine until an explicit restore or
    discard operation.
    """

    def __init__(self, managed_root: str | Path) -> None:
        self.managed_root = Path(managed_root).resolve(strict=False)
        self.recovery_root = self.managed_root / RECOVERY_DIRECTORY_NAME

    @staticmethod
    def managed_paths(db_path: str | Path) -> tuple[Path, ...]:
        database = Path(db_path)
        return (
            database,
            database.with_name(f"{database.name}-wal"),
            database.with_name(f"{database.name}-shm"),
            Path(f"{database}.faiss"),
            Path(f"{database}.faiss.json"),
        )

    @staticmethod
    def checkpoint_database(db_path: str | Path) -> None:
        database = Path(db_path)
        if not database.exists():
            return
        connection = sqlite3.connect(str(database), timeout=0.1)
        try:
            connection.execute("PRAGMA busy_timeout=100;")
            try:
                row = connection.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
            except sqlite3.DatabaseError as exc:
                message = str(exc).casefold()
                if "locked" in message or "busy" in message:
                    raise RuntimeError(
                        "Face-library storage is in use by another reader or writer; retry when it is idle."
                    ) from exc
                # A corrupt/non-SQLite managed DB still needs safe quarantine;
                # there is no usable WAL state to checkpoint in this case.
                return
            if row is not None and int(row[0] or 0) != 0:
                raise RuntimeError(
                    "Face-library storage is in use by another reader or writer; retry when it is idle."
                )
        finally:
            connection.close()

    def remove(
        self,
        db_path: str | Path,
        *,
        cancel_check=None,
        checkpoint: Callable[[str], None] | None = None,
    ) -> FaceStorageRemovalResult:
        database = self._validate_database_path(db_path)
        raise_if_cancelled(cancel_check)
        existing = tuple(path for path in self.managed_paths(database) if path.exists())
        operation_id = uuid4().hex
        operation_root = self.recovery_root / operation_id
        payload_root = operation_root / "payload"
        payload_root.mkdir(parents=True, exist_ok=False)
        journal = {
            "version": JOURNAL_VERSION,
            "operation_id": operation_id,
            "state": "prepared",
            "db_name": database.name,
            "entries": [
                {"name": path.name, "staged": False}
                for path in existing
            ],
        }
        journal_path = operation_root / "journal.json"
        try:
            self._write_journal(journal_path, journal)
        except Exception:
            shutil.rmtree(operation_root, ignore_errors=True)
            raise
        self._notify(checkpoint, "journal_prepared")

        try:
            raise_if_cancelled(cancel_check)
            for entry in journal["entries"]:
                name = str(entry["name"])
                source = self.managed_root / name
                staged = payload_root / name
                journal["state"] = "staging"
                journal["current"] = name
                self._write_journal(journal_path, journal)
                self._notify(checkpoint, f"before_rename:{name}")
                raise_if_cancelled(cancel_check)
                os.replace(source, staged)
                self._sync_directory(self.managed_root)
                self._sync_directory(payload_root)
                self._notify(checkpoint, f"after_rename:{name}")
                raise_if_cancelled(cancel_check)
                entry["staged"] = True
                self._write_journal(journal_path, journal)
                self._notify(checkpoint, f"journal_after_rename:{name}")
            journal.pop("current", None)
            journal["state"] = "staged"
            self._write_journal(journal_path, journal)
            self._notify(checkpoint, "before_commit")
            raise_if_cancelled(cancel_check)
            committed_journal = {**journal, "state": "committed"}
            self._write_journal(journal_path, committed_journal)
            journal = committed_journal
            self._notify(checkpoint, "after_commit")
        except (Cancelled, Exception):
            # The atomic committed journal is the durable point of no return
            # for this operation.  A late cancellation or observer failure
            # must not turn a completed removal into a misleading rollback.
            if str(journal.get("state", "")) == "committed":
                return self._result(operation_id, database, existing, operation_root)
            self._restore_operation(operation_root, remove_after_restore=True)
            raise

        return self._result(operation_id, database, existing, operation_root)

    @staticmethod
    def _result(
        operation_id: str,
        database: Path,
        existing: tuple[Path, ...],
        operation_root: Path,
    ) -> FaceStorageRemovalResult:
        return FaceStorageRemovalResult(
            operation_id=operation_id,
            db_path=str(database),
            staged_paths=tuple(str(path) for path in existing),
            recovery_path=str(operation_root),
        )

    def recover_incomplete(self) -> FaceStorageRecoveryReport:
        restored: list[str] = []
        committed: list[str] = []
        failures: list[str] = []
        if not self.recovery_root.exists():
            return FaceStorageRecoveryReport((), (), ())
        for operation_root in sorted(path for path in self.recovery_root.iterdir() if path.is_dir()):
            try:
                journal = self._read_journal(operation_root)
                operation_id = str(journal["operation_id"])
                if str(journal.get("state", "")) == "committed":
                    committed.append(operation_id)
                    continue
                self._restore_operation(operation_root, remove_after_restore=True)
                restored.append(operation_id)
            except Exception as exc:
                failures.append(f"{operation_root.name}: {exc}")
        return FaceStorageRecoveryReport(tuple(restored), tuple(committed), tuple(failures))

    def restore_committed(self, operation_id: str) -> tuple[str, ...]:
        operation_root = self._operation_root(operation_id)
        journal = self._read_journal(operation_root)
        if str(journal.get("state", "")) != "committed":
            raise ValueError("Only a committed face-storage removal can be restored explicitly.")
        return self._restore_operation(operation_root, remove_after_restore=True)

    def discard_committed(self, operation_id: str) -> None:
        operation_root = self._operation_root(operation_id)
        journal = self._read_journal(operation_root)
        if str(journal.get("state", "")) != "committed":
            raise ValueError("Only a committed face-storage removal can be discarded.")
        shutil.rmtree(operation_root)

    def _validate_database_path(self, db_path: str | Path) -> Path:
        database = Path(db_path).resolve(strict=False)
        if database.parent != self.managed_root:
            raise ValueError("Face database must be an immediate child of the managed storage root.")
        if not database.name or database.name in {".", ".."}:
            raise ValueError("Face database path is invalid.")
        return database

    def _operation_root(self, operation_id: str) -> Path:
        clean_id = str(operation_id or "").strip()
        if not clean_id or Path(clean_id).name != clean_id:
            raise ValueError("Recovery operation ID is invalid.")
        operation_root = (self.recovery_root / clean_id).resolve(strict=False)
        if operation_root.parent != self.recovery_root.resolve(strict=False):
            raise ValueError("Recovery operation is outside the managed recovery root.")
        return operation_root

    def _restore_operation(self, operation_root: Path, *, remove_after_restore: bool) -> tuple[str, ...]:
        journal = self._read_journal(operation_root)
        restored: list[str] = []
        entries = list(journal.get("entries", []) or [])
        for entry in reversed(entries):
            name = self._validate_entry_name(str(entry.get("name", "")))
            original = self.managed_root / name
            staged = operation_root / "payload" / name
            if staged.exists() and original.exists():
                raise RuntimeError(f"Refusing to restore over an existing managed file: {name}")
            if staged.exists():
                os.replace(staged, original)
                self._sync_directory(staged.parent)
                self._sync_directory(original.parent)
                restored.append(str(original))
        if remove_after_restore:
            shutil.rmtree(operation_root)
        return tuple(reversed(restored))

    def _read_journal(self, operation_root: Path) -> dict[str, object]:
        journal_path = operation_root / "journal.json"
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
        if int(payload.get("version", 0) or 0) != JOURNAL_VERSION:
            raise ValueError("Unsupported face-storage recovery journal version.")
        if str(payload.get("operation_id", "")) != operation_root.name:
            raise ValueError("Face-storage recovery journal identity does not match its directory.")
        database = self._validate_database_path(self.managed_root / str(payload.get("db_name", "")))
        allowed = {path.name for path in self.managed_paths(database)}
        for entry in list(payload.get("entries", []) or []):
            name = self._validate_entry_name(str(entry.get("name", "")))
            if name not in allowed:
                raise ValueError(f"Recovery journal contains an unmanaged file: {name}")
        return payload

    @staticmethod
    def _validate_entry_name(name: str) -> str:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Recovery journal contains an invalid file name.")
        return name

    @staticmethod
    def _write_journal(path: Path, payload: dict[str, object]) -> None:
        atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _notify(checkpoint: Callable[[str], None] | None, name: str) -> None:
        if checkpoint is not None:
            checkpoint(name)

    @staticmethod
    def _sync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except (AttributeError, OSError):
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)
