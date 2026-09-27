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


@dataclass(frozen=True)
class RenamePreviewItem:
    """One source-safe in-place rename proposed by a user-visible preview."""

    source_path: str
    target_path: str = ""
    reason: str = ""

    @property
    def is_ready(self) -> bool:
        return bool(self.source_path and self.target_path and not self.reason)


class GalleryActionService:
    def __init__(
        self,
        *,
        audit_log_path: str | Path | None = None,
        journal_path: str | Path | None = None,
        temp_dir: str | Path | None = None,
    ) -> None:
        settings = get_settings() if audit_log_path is None or temp_dir is None else None
        self.audit_log_path = (
            Path(audit_log_path)
            if audit_log_path is not None
            else settings.log_dir / "file_operations.jsonl"
        )
        self.journal_path = Path(journal_path) if journal_path is not None else self.audit_log_path.with_suffix(".sqlite3")
        self.temp_dir = (
            Path(temp_dir)
            if temp_dir is not None
            else settings.cache_dir / "tmp" / "file_ops"
        )
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
            if file_entry_id is None:
                continue
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                trash_dir = source.resolve().parent / CLUSTERLENS_TRASH_DIR_NAME
                trash_dir.mkdir(parents=True, exist_ok=True)
                target = self._unique_target(trash_dir, source.name)
                self._update_journal_file(file_entry_id, result, status="pending", target_path=str(target))
                self._file_mutation_checkpoint("before_filesystem_mutation", source, target)
                self._atomic_move(source, target)
                self._file_mutation_checkpoint("after_filesystem_mutation", source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
                if cancel_check and cancel_check():
                    result.cancelled = True
                    break
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
        if image_paths:
            self._file_mutation_checkpoint("operation_intent_committed", Path(image_paths[0]), destination)
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Moving {index}/{total}")
            source = Path(image_path)
            file_entry_id = self._record_journal_file(result, source_path=image_path)
            if file_entry_id is None:
                continue
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                target = self._unique_target(destination, source.name)
                self._update_journal_file(file_entry_id, result, status="pending", target_path=str(target))
                self._file_mutation_checkpoint("before_filesystem_mutation", source, target)
                self._safe_move(source, target)
                self._file_mutation_checkpoint("after_filesystem_mutation", source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
                self._file_mutation_checkpoint("file_terminal_recorded", source, target)
                if cancel_check and cancel_check():
                    result.cancelled = True
                    break
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
            if file_entry_id is None:
                continue
            if not source.is_file():
                failure = f"{image_path}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                target = self._unique_target(destination, source.name)
                self._update_journal_file(file_entry_id, result, status="pending", target_path=str(target))
                self._file_mutation_checkpoint("before_filesystem_mutation", source, target)
                self._safe_copy(source, target)
                self._file_mutation_checkpoint("after_filesystem_mutation", source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
                if cancel_check and cancel_check():
                    result.cancelled = True
                    break
            except Exception as exc:
                failure = f"{image_path}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=image_paths, destination=str(destination))
        return result

    @staticmethod
    def preview_renames(
        image_paths: list[str],
        template: str,
        *,
        capture_times: dict[str, str] | None = None,
    ) -> list[RenamePreviewItem]:
        """Build a non-mutating, all-or-nothing in-place rename preview.

        ``template`` produces a filename *stem* and may use ``{stem}``,
        ``{index}``, ``{date}``, ``{year}``, ``{month}``, and ``{day}``.
        The original extension is always retained. Capture dates are supplied
        by the caller when known; otherwise the file's modification date is
        used. Directory changes and collision auto-resolution are deliberately
        excluded so users can review every intended source mutation.
        """
        normalized_template = str(template or "").strip()
        if not normalized_template:
            return [RenamePreviewItem(str(path), reason="A filename template is required.") for path in image_paths]
        capture_times = {str(path): str(value) for path, value in (capture_times or {}).items()}
        preview: list[RenamePreviewItem] = []
        seen_sources: set[str] = set()
        for index, raw_path in enumerate(image_paths, start=1):
            source = Path(str(raw_path)).expanduser().resolve()
            source_text = str(source)
            if source_text in seen_sources:
                continue
            seen_sources.add(source_text)
            if not source.is_file():
                preview.append(RenamePreviewItem(source_text, reason="Source file does not exist."))
                continue
            captured = GalleryActionService._rename_capture_parts(capture_times.get(source_text, ""), source)
            values: dict[str, object] = {
                "stem": source.stem,
                "index": index,
                "date": f"{captured.year:04d}{captured.month:02d}{captured.day:02d}",
                "year": f"{captured.year:04d}",
                "month": f"{captured.month:02d}",
                "day": f"{captured.day:02d}",
            }
            try:
                target_stem = normalized_template.format_map(values).strip()
            except (KeyError, ValueError, IndexError) as exc:
                preview.append(RenamePreviewItem(source_text, reason=f"Invalid template: {exc}"))
                continue
            if not target_stem or target_stem in {".", ".."}:
                preview.append(RenamePreviewItem(source_text, reason="Template produced an empty filename."))
                continue
            if any(separator in target_stem for separator in ("/", "\\", "\x00")):
                preview.append(RenamePreviewItem(source_text, reason="Template may not contain folder separators."))
                continue
            target = source.with_name(f"{target_stem}{source.suffix}")
            if target == source:
                preview.append(RenamePreviewItem(source_text, str(target), "New filename is unchanged."))
                continue
            preview.append(RenamePreviewItem(source_text, str(target)))

        source_paths = {item.source_path for item in preview}
        target_counts: dict[str, int] = {}
        for item in preview:
            if item.target_path:
                target_counts[item.target_path] = target_counts.get(item.target_path, 0) + 1
        checked: list[RenamePreviewItem] = []
        for item in preview:
            if item.reason:
                checked.append(item)
                continue
            target = Path(item.target_path)
            if target_counts.get(item.target_path, 0) > 1:
                checked.append(RenamePreviewItem(item.source_path, item.target_path, "Another planned rename has this target."))
            elif item.target_path in source_paths:
                checked.append(RenamePreviewItem(item.source_path, item.target_path, "Target is another source in this batch; split this rename into separate operations."))
            elif target.exists():
                checked.append(RenamePreviewItem(item.source_path, item.target_path, "Target file already exists."))
            else:
                checked.append(item)
        return checked

    def rename_files(
        self,
        preview: list[RenamePreviewItem],
        progress_callback=None,
        cancel_check=None,
    ) -> GalleryActionResult:
        """Apply only a fully-safe preview and make each rename recoverable."""
        result = self._new_result("rename")
        requested_paths = [item.source_path for item in preview]
        invalid = [item for item in preview if not item.is_ready]
        if invalid:
            result.failures.extend(f"{item.source_path}: {item.reason or 'unsafe rename preview'}" for item in invalid)
            return result
        if not self._begin_journal_operation(result, requested_paths=requested_paths, destination="renamed in place"):
            return result
        total = len(preview)
        for index, item in enumerate(preview, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Renaming {index}/{total}")
            source = Path(item.source_path)
            target = Path(item.target_path)
            file_entry_id = self._record_journal_file(result, source_path=str(source), target_path=str(target))
            if file_entry_id is None:
                continue
            if not source.is_file():
                failure = f"{source}: source file does not exist"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            if target.exists():
                failure = f"{target}: target file already exists"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
                continue
            try:
                self._file_mutation_checkpoint("before_filesystem_mutation", source, target)
                self._atomic_move(source, target)
                self._file_mutation_checkpoint("after_filesystem_mutation", source, target)
                result.changed_paths.append((str(source), str(target)))
                self._update_journal_file(file_entry_id, result, status="completed", target_path=str(target))
                if cancel_check and cancel_check():
                    result.cancelled = True
                    break
            except Exception as exc:
                failure = f"{source}: {exc}"
                result.failures.append(failure)
                self._update_journal_file(file_entry_id, result, status="failed", error=failure)
        self._write_audit(result, requested_paths=requested_paths, destination="renamed in place")
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

    def write_exif_draft(self, image_path: str, draft: dict[str, object]) -> None:
        """Embed only stable, textual fields from an inspector draft.

        Ratings, keywords, locations, and arbitrary custom fields have no
        portable EXIF representation across the supported formats, so they are
        preserved in the ClusterLens sidecar and mirrored in UserComment.
        This avoids corrupting proprietary/MakerNote data while retaining a
        readable embedded copy for JPEG/TIFF users who explicitly request it.
        """
        source = Path(image_path)
        if source.suffix.casefold() not in {".jpg", ".jpeg", ".tif", ".tiff"}:
            raise ValueError("Only JPEG and TIFF originals support embedded inspector metadata.")

        title = str(draft.get("title", "") or "").strip()
        description = str(draft.get("description", "") or "").strip()
        creator = str(draft.get("creator", "") or "").strip()
        copyright_text = str(draft.get("copyright", "") or "").strip()
        captured_at = str(draft.get("captured_at", "") or "").strip()
        try:
            rating = max(0, min(5, int(draft.get("rating", 0) or 0)))
        except (TypeError, ValueError):
            rating = 0
        comment_values: dict[str, str] = {
            "ic_title": title,
            "ic_rating": str(rating),
            "ic_tags": json.dumps([str(tag).strip() for tag in list(draft.get("tags", []) or []) if str(tag).strip()]),
            "ic_location": str(draft.get("location", "") or "").strip(),
        }
        for key, value in dict(draft.get("custom_fields", {}) or {}).items():
            normalized_key = str(key).strip()
            normalized_value = str(value).strip()
            if normalized_key and normalized_value:
                comment_values[f"ic_{normalized_key}"] = normalized_value

        def _mutate(exif) -> None:
            if description or title:
                exif[270] = description or title  # ImageDescription
            if creator:
                exif[315] = creator  # Artist
            if copyright_text:
                exif[33432] = copyright_text
            if captured_at:
                parsed = self._coerce_exif_datetime(captured_at)
                if parsed:
                    exif[36867] = parsed  # DateTimeOriginal
            current = self._coerce_comment_text(exif.get(37510))
            for key, value in comment_values.items():
                if value:
                    current = self.merge_comment_metadata(current, key, value)
            exif[37510] = current

        self._safe_rewrite_image_exif(image_path, _mutate)

    @staticmethod
    def _coerce_exif_datetime(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.strftime("%Y:%m:%d %H:%M:%S")
        except ValueError:
            pass
        if len(raw) >= 19 and raw[4:5] == ":" and raw[7:8] == ":":
            return raw[:19]
        return ""

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
            if file_entry_id is None:
                continue
            try:
                source = Path(image_path).resolve()
                self._metadata_rewrite_checkpoint("rewrite_intent_committed", source, source)
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
            if file_entry_id is None:
                continue
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

    def write_exif_drafts(
        self,
        image_paths: list[str],
        draft: dict[str, object],
        progress_callback=None,
        cancel_check=None,
    ) -> GalleryActionResult:
        """Journal an explicit inspector embed request, one atomic file at a time."""
        result = self._new_result("write_exif_draft")
        if not self._begin_journal_operation(result, requested_paths=image_paths):
            return result
        total = len(image_paths)
        for index, image_path in enumerate(image_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int((index / max(1, total)) * 100), f"Embedding metadata {index}/{total}")
            file_entry_id = self._record_journal_file(result, source_path=image_path, target_path=image_path)
            if file_entry_id is None:
                continue
            try:
                self.write_exif_draft(image_path, draft)
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

    def recover_incomplete_operations(self) -> dict[str, object]:
        """Reconcile interrupted file operations without guessing or moving files.

        Each per-file journal intent is compared with the two physical paths.
        A uniquely observable completed move/copy is published as such; an
        unchanged or ambiguous pair is recorded as a failure for explicit user
        review. Running this repeatedly is idempotent.
        """

        recovered_operations: list[str] = []
        ambiguous_files: list[str] = []
        try:
            with self._connect_journal() as connection:
                operations = connection.execute(
                    """
                    SELECT operation_id, operation, requested_count, requested_paths_json, destination
                    FROM operations
                    WHERE completed = 0
                    ORDER BY timestamp_utc ASC, rowid ASC
                    """
                ).fetchall()
                for operation_row in operations:
                    operation_id = str(operation_row["operation_id"])
                    operation = str(operation_row["operation"])
                    file_rows = connection.execute(
                        """
                        SELECT id, source_path, target_path, status, error
                        FROM operation_files
                        WHERE operation_id = ?
                        ORDER BY id ASC
                        """,
                        (operation_id,),
                    ).fetchall()
                    changed_paths: list[tuple[str, str]] = []
                    failures: list[str] = []
                    for file_row in file_rows:
                        source_text = str(file_row["source_path"] or "")
                        target_text = str(file_row["target_path"] or "")
                        source = Path(source_text) if source_text else None
                        target = Path(target_text) if target_text else None
                        source_exists = bool(source is not None and source.is_file())
                        target_exists = bool(target is not None and target.is_file())
                        status = "failed"
                        error = ""
                        recovery_status = "failed"
                        if not target_text:
                            error = "Interrupted before the operation target was journalled; outcome requires review."
                        elif source_text == target_text:
                            error = "Metadata rewrite was interrupted; the source is usable but commit outcome is unknown."
                            recovery_status = "outcome_unknown"
                        elif operation == "copy" and source_exists and target_exists:
                            status = "completed"
                            recovery_status = "not_applicable"
                            changed_paths.append((source_text, target_text))
                        elif operation in {"move", "rename", "delete_to_trash", "restore"} and target_exists and not source_exists:
                            status = "completed"
                            recovery_status = "restored" if operation == "restore" else "restorable"
                            changed_paths.append((source_text, target_text))
                        elif source_exists and not target_exists:
                            error = "Interrupted before the file move/copy committed; the original remains unchanged."
                            recovery_status = "not_applicable"
                        elif source_exists and target_exists:
                            error = "Both original and target exist after interruption; no file was removed automatically."
                            recovery_status = "outcome_unknown"
                        else:
                            error = "Neither the original nor target file exists; manual recovery is required."
                            recovery_status = "outcome_unknown"
                        if error:
                            failures.append(f"{source_text or target_text}: {error}")
                            ambiguous_files.append(source_text or target_text)
                        connection.execute(
                            """
                            UPDATE operation_files
                            SET status = ?, error = ?, recovery_status = ?, updated_at_utc = ?
                            WHERE id = ?
                            """,
                            (
                                status,
                                error,
                                recovery_status,
                                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                int(file_row["id"]),
                            ),
                        )
                    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    connection.execute(
                        """
                        UPDATE operations
                        SET changed_paths_json = ?, failure_count = ?, failures_json = ?,
                            completed = 1, recovery_status = ?, updated_at_utc = ?
                        WHERE operation_id = ?
                        """,
                        (
                            json.dumps(changed_paths, ensure_ascii=False),
                            len(failures),
                            json.dumps(failures, ensure_ascii=False),
                            _operation_recovery_status(operation, completed=True, changed_paths=changed_paths),
                            timestamp,
                            operation_id,
                        ),
                    )
                    recovered_operations.append(operation_id)
        except (OSError, sqlite3.Error) as exc:
            return {
                "recovered_operations": tuple(recovered_operations),
                "ambiguous_files": tuple(ambiguous_files),
                "failures": (str(exc),),
            }
        return {
            "recovered_operations": tuple(recovered_operations),
            "ambiguous_files": tuple(ambiguous_files),
            "failures": (),
        }

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
            if file_entry_id is None:
                continue
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
        except sqlite3.Error as exc:
            result.failures.append(
                f"Operation journal could not record {source_path}; this file was not changed: {exc}"
            )
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
            self._metadata_rewrite_checkpoint("before_replace", source, temp_target)
            os.replace(str(temp_target), str(source))
            self._metadata_rewrite_checkpoint("after_replace", source, temp_target)
        finally:
            self._cleanup_file(temp_target)

    def _file_mutation_checkpoint(self, _name: str, _source: Path, _target: Path) -> None:
        """Deterministic fault-injection seam around one journalled mutation."""

        return

    def _metadata_rewrite_checkpoint(self, _name: str, _source: Path, _temporary: Path) -> None:
        """Deterministic fault-injection seam around an image replacement."""

        return

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

    @staticmethod
    def _rename_capture_parts(raw_capture: str, source: Path) -> datetime:
        raw = str(raw_capture or "").strip()
        if raw:
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                pass
        return datetime.fromtimestamp(source.stat().st_mtime, timezone.utc)


def _loads_json_list(raw: object) -> list[object]:
    try:
        payload = json.loads(str(raw or "[]"))
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    return []


def _file_recovery_status(operation: str, *, status: str) -> str:
    if operation in {"move", "rename", "delete_to_trash", "restore"}:
        if status == "completed":
            return "restorable" if operation != "restore" else "restored"
        if status == "failed":
            return "failed"
        return "pending"
    return "not_applicable"


def _operation_recovery_status(operation: str, *, completed: bool, changed_paths: list[tuple[str, str]]) -> str:
    if operation == "restore":
        return "restored" if changed_paths else "not_applicable"
    if operation in {"move", "rename", "delete_to_trash"}:
        if not completed:
            return "pending"
        return "restorable" if changed_paths else "not_applicable"
    return "not_applicable"
