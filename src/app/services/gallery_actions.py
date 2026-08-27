from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import shutil
from pathlib import Path

from PIL import Image

from infra.settings import get_settings

CLUSTERLENS_TRASH_DIR_NAME = "ClusterLens Trash"


@dataclass
class GalleryActionResult:
    operation_id: str = ""
    operation: str = ""
    changed_paths: list[tuple[str, str]] = field(default_factory=list)
    affected_paths: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cancelled: bool = False
    audit_log_path: str = ""
    journal_path: str = ""


class GalleryActionService:
    def __init__(
        self,
        *,
        audit_log_path: str | Path | None = None,
        journal_path: str | Path | None = None,
        temp_dir: str | Path | None = None,
    ) -> None:
        settings = get_settings()
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else settings.log_dir / "file_operations.jsonl"
        self.journal_path = Path(journal_path) if journal_path is not None else self.audit_log_path.with_suffix(".sqlite3")
        self.temp_dir = Path(temp_dir) if temp_dir is not None else settings.cache_dir / "tmp" / "file_ops"
        # Construction is UI-thread safe. Filesystem and SQLite setup happen
        # lazily inside the background operation that first needs the journal.
        self._journal_schema_ready = False

    def move_to_trash(self, image_paths: list[str], progress_callback=None, cancel_check=None) -> GalleryActionResult:
        result = self._new_result("delete_to_trash")
        if not self._begin_journal_operation(result, requested_paths=image_paths):
            return result
        if not image_paths:
            self._write_audit(result, requested_paths=image_paths)
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Moving to ClusterLens Trash {index}/{total}")
            source = Path(image_path)
            file_entry_id = self._record_journal_file(result, source_path=image_path)
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                trash_dir = source.resolve().parent / CLUSTERLENS_TRASH_DIR_NAME
                trash_dir.mkdir(parents=True, exist_ok=True)
                target = self._unique_target(trash_dir, source.name)
                self._atomic_move(source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths)
        return result

    def move_to_directory(
        self,
        image_paths: list[str],
        destination_directory: str,
        progress_callback=None,
        cancel_check=None,
    ) -> GalleryActionResult:
        destination = Path(destination_directory)
        destination.mkdir(parents=True, exist_ok=True)
        result = self._new_result("move")
        if not self._begin_journal_operation(result, requested_paths=image_paths, destination=str(destination)):
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Moving {index}/{total}")
            source = Path(image_path)
            file_entry_id = self._record_journal_file(result, source_path=image_path)
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                target = self._unique_target(destination, source.name)
                self._update_journal_file(file_entry_id, result, status="pending", target_path=str(target))
                self._safe_move(source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths, destination=str(destination))
        return result

    def copy_to_directory(
        self,
        image_paths: list[str],
        destination_directory: str,
        progress_callback=None,
        cancel_check=None,
    ) -> GalleryActionResult:
        destination = Path(destination_directory)
        destination.mkdir(parents=True, exist_ok=True)
        result = self._new_result("copy")
        if not self._begin_journal_operation(result, requested_paths=image_paths, destination=str(destination)):
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Copying {index}/{total}")
            source = Path(image_path)
            file_entry_id = self._record_journal_file(result, source_path=image_path)
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                target = self._unique_target(destination, source.name)
                self._update_journal_file(file_entry_id, result, status="pending", target_path=str(target))
                self._safe_copy(source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths, destination=str(destination))
        return result

    def write_exif_comment(self, image_path: str, comment: str) -> None:
        self._safe_rewrite_image_exif(image_path, lambda exif: exif.__setitem__(37510, comment))

    def write_exif_metadata_pair(self, image_path: str, key: str, value: str) -> None:
        key = str(key or "").strip()
        value = str(value or "").strip()
        if not key or not value:
            raise ValueError("Both EXIF key and value are required.")

        def _mutate(exif) -> None:
            current = self._coerce_comment_text(exif.get(37510))
            exif[37510] = self.merge_comment_metadata(current, key, value)

        self._safe_rewrite_image_exif(image_path, _mutate)

    @classmethod
    def read_exif_metadata_value(cls, image_path: str, key: str) -> str:
        key = str(key or "").strip()
        if not key:
            return ""
        with Image.open(image_path) as image:
            exif = image.getexif()
            current = cls._coerce_comment_text(exif.get(37510))
        return cls.parse_comment_metadata(current).get(key.casefold(), "")

    def write_exif_comments(self, image_paths: list[str], comment: str, progress_callback=None, cancel_check=None) -> GalleryActionResult:
        result = self._new_result("write_exif_comment")
        if not self._begin_journal_operation(result, requested_paths=image_paths):
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Writing EXIF {index}/{total}")
            file_entry_id = self._record_journal_file(result, source_path=image_path, target_path=image_path)
            try:
                self.write_exif_comment(image_path, comment)
                result.affected_paths.append(str(image_path))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=image_path)
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths)
        return result

    def write_exif_metadata_pairs(
        self,
        image_paths: list[str],
        key: str,
        value: str,
        progress_callback=None,
        cancel_check=None,
    ) -> GalleryActionResult:
        result = self._new_result("write_exif_metadata")
        if not self._begin_journal_operation(result, requested_paths=image_paths):
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Writing EXIF {index}/{total}")
            file_entry_id = self._record_journal_file(result, source_path=image_path, target_path=image_path)
            try:
                self.write_exif_metadata_pair(image_path, key, value)
                result.affected_paths.append(str(image_path))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=image_path)
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths)
        return result

    def cleanup_temp_files(self) -> tuple[int, list[str]]:
        removed = 0
        failures: list[str] = []
        if not self.temp_dir.exists():
            return removed, failures
        for path in self.temp_dir.rglob("*"):
            try:
                if path.is_file():
                    path.unlink(missing_ok=True)
                    removed += 1
            except OSError as exc:
                failures.append(f"{path}: {exc}")
        return removed, failures

    def _connect_journal(self) -> sqlite3.Connection:
        if not self._journal_schema_ready:
            self._ensure_journal_schema()
        connection = sqlite3.connect(str(self.journal_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_journal_schema(self) -> None:
        try:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.journal_path)) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operations (
                        operation_id TEXT PRIMARY KEY,
                        operation TEXT NOT NULL,
                        timestamp_utc TEXT NOT NULL,
                        requested_count INTEGER NOT NULL DEFAULT 0,
                        requested_paths_json TEXT NOT NULL DEFAULT '[]',
                        destination TEXT NOT NULL DEFAULT '',
                        changed_paths_json TEXT NOT NULL DEFAULT '[]',
                        affected_paths_json TEXT NOT NULL DEFAULT '[]',
                        failure_count INTEGER NOT NULL DEFAULT 0,
                        failures_json TEXT NOT NULL DEFAULT '[]',
                        cancelled INTEGER NOT NULL DEFAULT 0,
                        completed INTEGER NOT NULL DEFAULT 0,
                        recovery_status TEXT NOT NULL DEFAULT 'not_applicable',
                        updated_at_utc TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS operation_files (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        source_path TEXT NOT NULL,
                        target_path TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'pending',
                        error TEXT NOT NULL DEFAULT '',
                        recovery_status TEXT NOT NULL DEFAULT 'not_applicable',
                        updated_at_utc TEXT NOT NULL,
                        FOREIGN KEY(operation_id) REFERENCES operations(operation_id)
                    )
                    """
                )
                connection.execute("CREATE INDEX IF NOT EXISTS idx_operation_files_operation ON operation_files(operation_id)")
            self._journal_schema_ready = True
        except sqlite3.Error:
            pass

    def read_audit_entries(self, *, limit: int = 200) -> list[dict[str, object]]:
        """Return recent file-operation audit rows without failing the UI.

        SQLite is the durable operation journal. JSONL remains as a tolerated
        import/export compatibility path for older runs and support bundles.
        """
        entries = self._read_sqlite_journal_entries(limit=limit)
        if entries:
            return entries
        if not self.audit_log_path.exists():
            return []
        recent: deque[str] = deque(maxlen=max(1, int(limit)))
        try:
            with self.audit_log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        recent.append(line)
        except OSError:
            return []
        entries: list[dict[str, object]] = []
        for raw_line in recent:
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _read_sqlite_journal_entries(self, *, limit: int = 200) -> list[dict[str, object]]:
        if not self.journal_path.exists():
            return []
        try:
            with self._connect_journal() as connection:
                rows = connection.execute(
                    """
                    SELECT operation_id, operation, timestamp_utc, requested_count, requested_paths_json,
                           destination, changed_paths_json, affected_paths_json, failure_count,
                           failures_json, cancelled, completed, recovery_status
                    FROM operations
                    ORDER BY timestamp_utc DESC, rowid DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
                operation_ids = [str(row["operation_id"]) for row in rows]
                file_rows = []
                if operation_ids:
                    placeholders = ",".join("?" for _ in operation_ids)
                    file_rows = connection.execute(
                        f"""
                        SELECT operation_id, source_path, target_path, status, error, recovery_status
                        FROM operation_files
                        WHERE operation_id IN ({placeholders})
                        ORDER BY id ASC
                        """,
                        operation_ids,
                    ).fetchall()
        except sqlite3.Error:
            return []
        files_by_operation: dict[str, list[dict[str, object]]] = {}
        for file_row in file_rows:
            files_by_operation.setdefault(str(file_row["operation_id"]), []).append(
                {
                    "source": str(file_row["source_path"] or ""),
                    "target": str(file_row["target_path"] or ""),
                    "status": str(file_row["status"] or ""),
                    "error": str(file_row["error"] or ""),
                    "recovery_status": str(file_row["recovery_status"] or ""),
                }
            )
        entries: list[dict[str, object]] = []
        for row in reversed(rows):
            entries.append(
                {
                    "operation_id": str(row["operation_id"]),
                    "operation": str(row["operation"]),
                    "timestamp_utc": str(row["timestamp_utc"]),
                    "requested_count": int(row["requested_count"]),
                    "requested_paths": _loads_json_list(row["requested_paths_json"]),
                    "destination": str(row["destination"] or ""),
                    "changed_paths": _loads_json_list(row["changed_paths_json"]),
                    "affected_paths": _loads_json_list(row["affected_paths_json"]),
                    "failure_count": int(row["failure_count"]),
                    "failures": _loads_json_list(row["failures_json"]),
                    "cancelled": bool(row["cancelled"]),
                    "completed": bool(row["completed"]),
                    "recovery_status": str(row["recovery_status"] or ""),
                    "journal_path": str(self.journal_path),
                    "audit_log_path": str(self.audit_log_path),
                    "file_results": list(files_by_operation.get(str(row["operation_id"]), ())),
                }
            )
        return entries

    def restore_changed_paths(
        self,
        changed_paths: list[tuple[str, str]],
        progress_callback=None,
        cancel_check=None,
        *,
        conflict_policy: str = "skip",
    ) -> GalleryActionResult:
        """Move files from their operation target back to their original path.

        This is intentionally conservative: it only restores when the current
        target exists. By default, existing originals are left untouched and
        reported as failures to avoid data corruption. Set conflict_policy to
        "unique_name" to restore beside the original using a generated name.
        """
        result = self._new_result("restore")
        total = len(changed_paths)
        requested_paths = [str(dst) for _src, dst in changed_paths]
        if not self._begin_journal_operation(result, requested_paths=requested_paths, destination="original paths"):
            return result
        for index, (original_path, current_path) in enumerate(changed_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Restoring {index}/{total}")
            original = Path(str(original_path)).resolve()
            current = Path(str(current_path)).resolve()
            file_entry_id = self._record_journal_file(result, source_path=str(current), target_path=str(original))
            if not current.is_file():
                failure = f"{current_path}: restore source does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            if original.exists():
                if str(conflict_policy or "skip").strip().lower() == "unique_name":
                    original = self._unique_target(original.parent, original.name)
                    self._update_journal_file(file_entry_id, result, status="pending", target_path=str(original))
                else:
                    failure = f"{original_path}: original path already exists"
                    result.failures.append(failure)
                    self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                    continue
            try:
                original.parent.mkdir(parents=True, exist_ok=True)
                self._safe_move(current, original)
                result.changed_paths.append((str(current), str(original)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(original))
            except Exception as exc:
                failure = f"{current_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=requested_paths, destination="original paths")
        return result

    def _new_result(self, operation: str) -> GalleryActionResult:
        return GalleryActionResult(
            operation_id=uuid.uuid4().hex,
            operation=operation,
            audit_log_path=str(self.audit_log_path),
            journal_path=str(self.journal_path),
        )

    def _begin_journal_operation(self, result: GalleryActionResult, *, requested_paths: list[str], destination: str = "") -> bool:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connect_journal() as connection:
                connection.execute(
                    """
                    INSERT OR REPLACE INTO operations (
                        operation_id, operation, timestamp_utc, requested_count, requested_paths_json,
                        destination, cancelled, completed, recovery_status, updated_at_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                    """,
                    (
                        result.operation_id,
                        result.operation,
                        timestamp,
                        len(requested_paths),
                        json.dumps([str(path) for path in requested_paths], ensure_ascii=False),
                        str(destination or ""),
                        _operation_recovery_status(result.operation, completed=False, changed_paths=[]),
                        timestamp,
                    ),
                )
            return True
        except (OSError, sqlite3.Error) as exc:
            result.failures.append(f"Operation journal is unavailable; no files were changed: {exc}")
            return False

    def _record_journal_file(self, result: GalleryActionResult, *, source_path: str, target_path: str = "") -> int | None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connect_journal() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO operation_files (
                        operation_id, source_path, target_path, status, recovery_status, updated_at_utc
                    )
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        result.operation_id,
                        str(source_path),
                        str(target_path or ""),
                        _file_recovery_status(result.operation, status="pending"),
                        timestamp,
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error:
            return None

    def _update_journal_file(
        self,
        file_entry_id: int | None,
        result: GalleryActionResult,
        *,
        status: str,
        target_path: str = "",
        error: str = "",
    ) -> None:
        if file_entry_id is None:
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connect_journal() as connection:
                connection.execute(
                    """
                    UPDATE operation_files
                    SET target_path = COALESCE(NULLIF(?, ''), target_path),
                        status = ?,
                        error = ?,
                        recovery_status = ?,
                        updated_at_utc = ?
                    WHERE id = ?
                    """,
                    (
                        str(target_path or ""),
                        str(status),
                        str(error or ""),
                        _file_recovery_status(result.operation, status=str(status)),
                        timestamp,
                        int(file_entry_id),
                    ),
                )
        except sqlite3.Error:
            pass

    def _finish_journal_operation(self, result: GalleryActionResult, *, requested_paths: list[str], destination: str = "") -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            with self._connect_journal() as connection:
                connection.execute(
                    """
                    UPDATE operations
                    SET requested_count = ?,
                        requested_paths_json = ?,
                        destination = ?,
                        changed_paths_json = ?,
                        affected_paths_json = ?,
                        failure_count = ?,
                        failures_json = ?,
                        cancelled = ?,
                        completed = 1,
                        recovery_status = ?,
                        updated_at_utc = ?
                    WHERE operation_id = ?
                    """,
                    (
                        len(requested_paths),
                        json.dumps([str(path) for path in requested_paths], ensure_ascii=False),
                        str(destination or ""),
                        json.dumps(list(result.changed_paths), ensure_ascii=False),
                        json.dumps(list(result.affected_paths), ensure_ascii=False),
                        len(result.failures),
                        json.dumps(list(result.failures), ensure_ascii=False),
                        1 if result.cancelled else 0,
                        _operation_recovery_status(result.operation, completed=True, changed_paths=result.changed_paths),
                        timestamp,
                        result.operation_id,
                    ),
                )
        except sqlite3.Error:
            pass

    def _write_audit(
        self,
        result: GalleryActionResult,
        *,
        requested_paths: list[str],
        destination: str = "",
    ) -> None:
        self._finish_journal_operation(result, requested_paths=requested_paths, destination=destination)
        payload = {
            "operation_id": result.operation_id,
            "operation": result.operation,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requested_count": len(requested_paths),
            "requested_paths": [str(path) for path in requested_paths],
            "destination": destination,
            "changed_paths": list(result.changed_paths),
            "affected_paths": list(result.affected_paths),
            "failure_count": len(result.failures),
            "failures": list(result.failures),
            "cancelled": bool(result.cancelled),
        }
        try:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.audit_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        except OSError:
            # Audit logging must never make a user file operation fail after it has safely completed.
            pass

    def _safe_copy(self, source: Path, target: Path) -> None:
        source = source.resolve()
        target = target.resolve()
        if target.exists():
            raise FileExistsError(target)
        temp_target = self._temp_path_for(target)
        try:
            shutil.copy2(str(source), str(temp_target))
            self._verify_copy(source, temp_target)
            os.replace(str(temp_target), str(target))
        finally:
            self._cleanup_file(temp_target)

    def _safe_move(self, source: Path, target: Path) -> None:
        source = source.resolve()
        target = target.resolve()
        if target.exists():
            raise FileExistsError(target)
        try:
            self._atomic_move(source, target)
            return
        except OSError:
            pass
        self._safe_copy(source, target)
        try:
            source.unlink()
        except OSError as exc:
            # Keep the copied destination and report the duplicate so the user can resolve it.
            raise OSError(f"copied to {target}, but original could not be removed: {exc}") from exc

    @staticmethod
    def _atomic_move(source: Path, target: Path) -> None:
        if target.exists():
            raise FileExistsError(target)
        os.replace(str(source), str(target))

    def _safe_rewrite_image_exif(self, image_path: str, mutate_exif) -> None:
        source = Path(image_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        temp_target = self._temp_path_for(source)
        try:
            with Image.open(source) as image:
                exif = image.getexif()
                mutate_exif(exif)
                image.save(temp_target, exif=exif)
            self._verify_image(temp_target)
            os.replace(str(temp_target), str(source))
        finally:
            self._cleanup_file(temp_target)

    def _temp_path_for(self, target: Path) -> Path:
        target.parent.mkdir(parents=True, exist_ok=True)
        suffix = target.suffix or ".tmp"
        return target.parent / f".{target.stem}.{uuid.uuid4().hex}.ic_tmp{suffix}"

    @staticmethod
    def _verify_copy(source: Path, copied: Path) -> None:
        source_stat = source.stat()
        copied_stat = copied.stat()
        if int(source_stat.st_size) != int(copied_stat.st_size):
            raise OSError(f"copy verification failed: {source} -> {copied}")

    @staticmethod
    def _verify_image(path: Path) -> None:
        with Image.open(path) as image:
            image.verify()

    @staticmethod
    def _cleanup_file(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    @staticmethod
    def merge_comment_metadata(existing_comment: str, key: str, value: str) -> str:
        normalized_key = str(key or "").strip()
        normalized_value = str(value or "").strip()
        if not normalized_key or not normalized_value:
            raise ValueError("Both EXIF key and value are required.")

        lines = str(existing_comment or "").splitlines()
        merged_lines: list[str] = []
        replaced = False
        target_key = normalized_key.casefold()

        for raw_line in lines:
            line = str(raw_line).rstrip()
            if ":" not in line:
                if line:
                    merged_lines.append(line)
                continue
            existing_key, existing_value = line.split(":", 1)
            if existing_key.strip().casefold() == target_key:
                if not replaced:
                    merged_lines.append(f"{normalized_key}: {normalized_value}")
                    replaced = True
                continue
            merged_lines.append(f"{existing_key.strip()}: {existing_value.strip()}")

        if not replaced:
            merged_lines.append(f"{normalized_key}: {normalized_value}")
        return "\n".join(line for line in merged_lines if line)

    @staticmethod
    def parse_comment_metadata(existing_comment: str) -> dict[str, str]:
        parsed: dict[str, str] = {}
        for raw_line in str(existing_comment or "").splitlines():
            line = str(raw_line).rstrip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip()
            normalized_value = value.strip()
            if not normalized_key or not normalized_value:
                continue
            parsed[normalized_key.casefold()] = normalized_value
        return parsed

    @staticmethod
    def _coerce_comment_text(raw_value) -> str:
        if raw_value is None:
            return ""
        if isinstance(raw_value, bytes):
            for encoding in ("utf-8", "utf-16", "latin-1"):
                try:
                    return raw_value.decode(encoding).rstrip("\x00")
                except Exception:
                    continue
            return raw_value.decode("utf-8", errors="ignore").rstrip("\x00")
        return str(raw_value)

    @staticmethod
    def _unique_target(directory: Path, file_name: str) -> Path:
        candidate = directory / file_name
        stem = candidate.stem
        suffix = candidate.suffix
        counter = 1
        while candidate.exists():
            candidate = directory / f"{stem}_{counter}{suffix}"
            counter += 1
        return candidate


def _loads_json_list(raw: object) -> list[object]:
    try:
        payload = json.loads(str(raw or "[]"))
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _file_recovery_status(operation: str, *, status: str) -> str:
    if operation in {"move", "delete_to_trash", "restore"}:
        if status == "completed":
            return "restorable" if operation != "restore" else "restored"
        if status == "failed":
            return "failed"
        return "pending"
    return "not_applicable"


def _operation_recovery_status(operation: str, *, completed: bool, changed_paths: list[tuple[str, str]]) -> str:
    if operation == "restore":
        return "restored" if changed_paths else "not_applicable"
    if operation in {"move", "delete_to_trash"}:
        if not completed:
            return "pending"
        return "restorable" if changed_paths else "not_applicable"
    return "not_applicable"
