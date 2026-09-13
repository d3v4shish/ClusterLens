from __future__ import annotations

import json
import os
import sqlite3
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from infra.cancel import raise_if_cancelled
from infra.logging_config import get_logger
from infra.settings import get_settings
from .gallery_actions import GalleryActionService


LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ClusterTagSummary:
    tag_counts: dict[str, int] = field(default_factory=dict)
    top_tags: tuple[tuple[str, int], ...] = ()
    tagged_image_count: int = 0
    unique_tag_count: int = 0

    def as_context(self) -> dict[str, object]:
        return {
            "tag_counts": dict(self.tag_counts),
            "top_tags": list(self.top_tags),
            "tagged_image_count": int(self.tagged_image_count),
            "unique_tag_count": int(self.unique_tag_count),
        }


@dataclass
class ImageTagEditResult:
    affected_paths: list[str] = field(default_factory=list)
    mirrored_paths: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    cancelled: bool = False


@dataclass(frozen=True)
class TagInventoryItem:
    display_tag: str
    normalized_tag: str
    image_count: int
    sources: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class TagInventoryPage:
    items: tuple[TagInventoryItem, ...] = ()
    total_count: int | None = 0


@dataclass(frozen=True)
class TagPathPage:
    paths: tuple[str, ...] = ()
    total_count: int | None = 0


class ImageTagService:
    exif_key = "ic_tags"

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        action_service: GalleryActionService | None = None,
    ) -> None:
        self.settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else self.settings.image_tags_db
        self.action_service = action_service or GalleryActionService()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=DELETE;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA temp_store=MEMORY;")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create_schema()
        except sqlite3.DatabaseError:
            if not self._quarantine_corrupt_database():
                raise
            self._create_schema()

    def _create_schema(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_tags (
                    image_path TEXT NOT NULL,
                    tag_norm TEXT NOT NULL,
                    display_tag TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    source TEXT NOT NULL DEFAULT 'user',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(image_path, tag_norm)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tags_path ON image_tags(image_path)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_tags_tag_path ON image_tags(tag_norm, image_path)"
            )

    def _quarantine_corrupt_database(self) -> bool:
        if not self.db_path.exists():
            return False
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = self.db_path.with_name(f"{self.db_path.name}.corrupt-{timestamp}")
        try:
            self.db_path.replace(backup_path)
            for suffix in ("-wal", "-shm"):
                sidecar = self.db_path.with_name(f"{self.db_path.name}{suffix}")
                if sidecar.exists():
                    sidecar.replace(backup_path.with_name(f"{backup_path.name}{suffix}"))
        except OSError:
            LOGGER.exception("Could not preserve corrupt image-tag database %s", self.db_path)
            return False
        LOGGER.warning("Preserved corrupt image-tag database at %s", backup_path)
        return True

    @staticmethod
    def normalize_tag(tag: str) -> str:
        text = " ".join(str(tag or "").strip().split())
        return text.casefold()

    @staticmethod
    def clean_display_tag(tag: str) -> str:
        return " ".join(str(tag or "").strip().split())

    @classmethod
    def parse_tag_text(cls, raw_text: str) -> list[str]:
        tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in str(raw_text or "").split(","):
            display_tag = cls.clean_display_tag(raw_tag)
            normalized = cls.normalize_tag(display_tag)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tags.append(display_tag)
        return tags

    @classmethod
    def _canonical_path(cls, image_path: str) -> str:
        text = str(image_path or "").strip()
        if not text:
            return ""
        return os.path.abspath(text)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _stat_for_path(image_path: str) -> tuple[int, int] | None:
        try:
            stat = Path(image_path).stat()
        except FileNotFoundError:
            return None
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _fetch_rows_by_path(
        self,
        image_paths: Iterable[str],
    ) -> dict[str, list[tuple[str, str, str, int, int]]]:
        ordered_paths = [self._canonical_path(path) for path in image_paths if self._canonical_path(path)]
        rows_by_path: dict[str, list[tuple[str, str, str, int, int]]] = {path: [] for path in ordered_paths}
        if not ordered_paths:
            return rows_by_path
        with self._connection() as connection:
            for start in range(0, len(ordered_paths), 900):
                chunk = ordered_paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"""
                    SELECT image_path, tag_norm, display_tag, source, mtime_ns, file_size
                    FROM image_tags
                    WHERE image_path IN ({placeholders})
                    ORDER BY display_tag COLLATE NOCASE, tag_norm
                    """,
                    chunk,
                ).fetchall()
                for row in rows:
                    rows_by_path.setdefault(str(row[0]), []).append(
                        (str(row[1]), str(row[2]), str(row[3]), int(row[4]), int(row[5]))
                    )
        return rows_by_path

    def _load_db_rows_by_path(
        self,
        image_paths: Iterable[str],
    ) -> dict[str, list[tuple[str, str, str, int, int]]]:
        rows_by_path = self._fetch_rows_by_path(image_paths)
        stale_paths: list[str] = []
        for image_path, rows in list(rows_by_path.items()):
            if not rows:
                continue
            stat = self._stat_for_path(image_path)
            if stat is None:
                stale_paths.append(image_path)
                rows_by_path[image_path] = []
                continue
            mtime_ns, file_size = stat
            if any(row[3] != mtime_ns or row[4] != file_size for row in rows):
                stale_paths.append(image_path)
                rows_by_path[image_path] = []
        if stale_paths:
            self.remove_paths(stale_paths)
        return rows_by_path

    @classmethod
    def _rows_to_display_tags(cls, rows: Iterable[tuple[str, str, str, int, int]]) -> tuple[str, ...]:
        display_by_norm: dict[str, str] = {}
        for tag_norm, display_tag, _source, _mtime_ns, _file_size in rows:
            if tag_norm not in display_by_norm:
                display_by_norm[tag_norm] = display_tag
        return tuple(
            display_by_norm[norm]
            for norm in sorted(display_by_norm.keys(), key=lambda value: display_by_norm[value].casefold())
        )

    def _replace_rows_for_path(
        self,
        image_path: str,
        rows: Iterable[tuple[str, str, str]],
    ) -> None:
        canonical_path = self._canonical_path(image_path)
        if not canonical_path:
            return
        normalized_rows: list[tuple[str, str, str]] = []
        seen_norms: set[str] = set()
        for tag_norm, display_tag, source in rows:
            normalized = self.normalize_tag(tag_norm)
            display = self.clean_display_tag(display_tag)
            if not normalized or not display or normalized in seen_norms:
                continue
            seen_norms.add(normalized)
            normalized_rows.append((normalized, display, str(source or "user")))
        with self._connection() as connection:
            connection.execute("DELETE FROM image_tags WHERE image_path=?", (canonical_path,))
            if normalized_rows:
                stat = self._stat_for_path(canonical_path)
                if stat is None:
                    raise FileNotFoundError(canonical_path)
                mtime_ns, file_size = stat
                updated_at = self._utc_now()
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO image_tags(
                        image_path, tag_norm, display_tag, mtime_ns, file_size, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (canonical_path, tag_norm, display_tag, mtime_ns, file_size, source, updated_at)
                        for tag_norm, display_tag, source in normalized_rows
                    ],
                )

    def _db_display_tags_for_paths(self, image_paths: Iterable[str]) -> dict[str, tuple[str, ...]]:
        rows_by_path = self._load_db_rows_by_path(image_paths)
        return {
            image_path: self._rows_to_display_tags(rows)
            for image_path, rows in rows_by_path.items()
        }

    @classmethod
    def _encode_exif_tags(cls, tags: Iterable[str]) -> str:
        return json.dumps(list(tags), ensure_ascii=True)

    @classmethod
    def _decode_exif_tags(cls, payload: str) -> tuple[str, ...]:
        if not str(payload or "").strip():
            return ()
        try:
            loaded = json.loads(payload)
        except json.JSONDecodeError:
            return ()
        if not isinstance(loaded, list):
            return ()
        clean_tags: list[str] = []
        seen: set[str] = set()
        for raw_tag in loaded:
            display_tag = cls.clean_display_tag(str(raw_tag))
            normalized = cls.normalize_tag(display_tag)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            clean_tags.append(display_tag)
        return tuple(clean_tags)

    def _write_exif_tags(self, image_path: str, tags: Iterable[str]) -> None:
        self.action_service.write_exif_metadata_pair(
            image_path,
            self.exif_key,
            self._encode_exif_tags(tags),
        )

    def _read_exif_tags(self, image_path: str) -> tuple[str, ...]:
        payload = self.action_service.read_exif_metadata_value(image_path, self.exif_key)
        return self._decode_exif_tags(payload)

    def _import_exif_tags_for_path(self, image_path: str) -> tuple[str, ...]:
        if not Path(image_path).is_file():
            return ()
        imported_tags = self._read_exif_tags(image_path)
        if not imported_tags:
            return ()
        self._replace_rows_for_path(
            image_path,
            [(self.normalize_tag(tag), tag, "exif") for tag in imported_tags],
        )
        return tuple(sorted(imported_tags, key=str.casefold))

    def load_tags_for_paths(
        self,
        paths: list[str],
        *,
        progress_callback=None,
        cancel_check=None,
        import_missing_exif: bool = True,
    ) -> dict[str, tuple[str, ...]]:
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        db_tags = self._db_display_tags_for_paths(ordered_paths)
        tags_by_path: dict[str, tuple[str, ...]] = {}
        total = len(ordered_paths)
        if total == 0:
            return tags_by_path
        for index, image_path in enumerate(ordered_paths, start=1):
            raise_if_cancelled(cancel_check)
            tags = db_tags.get(image_path, ())
            if tags:
                tags_by_path[image_path] = tags
            elif import_missing_exif:
                try:
                    tags_by_path[image_path] = self._import_exif_tags_for_path(image_path)
                except Exception as exc:
                    LOGGER.warning("Skipping EXIF tag import for unreadable image %s: %s", image_path, exc)
                    tags_by_path[image_path] = ()
            else:
                tags_by_path[image_path] = ()
            if progress_callback and (index == 1 or index == total or index % 25 == 0):
                progress_callback(int(index * 100 / max(1, total)), f"Loading image tags {index}/{total}")
        return tags_by_path

    def list_tag_inventory(self) -> list[TagInventoryItem]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT tag_norm, display_tag, source, COUNT(DISTINCT image_path) AS image_count
                FROM image_tags
                GROUP BY tag_norm, display_tag, source
                ORDER BY display_tag COLLATE NOCASE, tag_norm, source
                """
            ).fetchall()
        grouped: dict[str, dict[str, object]] = {}
        for tag_norm, display_tag, source, image_count in rows:
            key = str(tag_norm)
            item = grouped.setdefault(
                key,
                {
                    "display_tag": str(display_tag),
                    "normalized_tag": key,
                    "image_paths": 0,
                    "sources": Counter(),
                },
            )
            item["display_tag"] = min(str(item["display_tag"]), str(display_tag), key=str.casefold)
            item["image_paths"] = int(item["image_paths"]) + int(image_count)
            item["sources"][str(source or "user")] += int(image_count)
        return [
            TagInventoryItem(
                display_tag=str(item["display_tag"]),
                normalized_tag=str(item["normalized_tag"]),
                image_count=int(item["image_paths"]),
                sources=tuple(sorted(item["sources"].items(), key=lambda value: value[0].casefold())),
            )
            for item in sorted(grouped.values(), key=lambda value: str(value["display_tag"]).casefold())
        ]

    @staticmethod
    def _scope_sql(scope_path: str | Path | None) -> tuple[str, list[object]]:
        """Return a portable directory-prefix clause for canonical database paths."""

        scope = ImageTagService._canonical_path(str(scope_path or ""))
        if not scope:
            return "", []
        prefix = scope.rstrip("/\\") + os.sep
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return "(image_path = ? OR image_path LIKE ? ESCAPE '\\')", [scope, f"{escaped}%"]

    def query_tag_inventory(
        self,
        *,
        query: str = "",
        scope_path: str | Path | None = None,
        limit: int = 100,
        offset: int = 0,
        include_total: bool = True,
    ) -> TagInventoryPage:
        """Return one deterministic inventory page without touching source media."""

        clauses: list[str] = []
        params: list[object] = []
        text = self.normalize_tag(query)
        if text:
            clauses.append("(tag_norm LIKE ? ESCAPE '\\' OR display_tag LIKE ? ESCAPE '\\')")
            escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.extend([f"%{escaped}%", f"%{escaped}%"])
        scope_clause, scope_params = self._scope_sql(scope_path)
        if scope_clause:
            clauses.append(scope_clause)
            params.extend(scope_params)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        page_size = max(1, min(500, int(limit)))
        page_offset = max(0, int(offset))
        with self._connection() as connection:
            total_count = None
            if include_total:
                total_count = int(
                    connection.execute(
                        f"SELECT COUNT(DISTINCT tag_norm) FROM image_tags {where}", params
                    ).fetchone()[0]
                    or 0
                )
            rows = connection.execute(
                f"""
                SELECT tag_norm, MIN(display_tag) AS display_tag, COUNT(DISTINCT image_path) AS image_count
                FROM image_tags
                {where}
                GROUP BY tag_norm
                ORDER BY display_tag COLLATE NOCASE, tag_norm
                LIMIT ? OFFSET ?
                """,
                [*params, page_size, page_offset],
            ).fetchall()
            norms = [str(row[0]) for row in rows]
            sources_by_norm: dict[str, list[tuple[str, int]]] = {norm: [] for norm in norms}
            if norms:
                placeholders = ",".join("?" for _ in norms)
                source_clauses = [f"tag_norm IN ({placeholders})"]
                source_params: list[object] = list(norms)
                if scope_clause:
                    source_clauses.append(scope_clause)
                    source_params.extend(scope_params)
                source_rows = connection.execute(
                    f"""
                    SELECT tag_norm, source, COUNT(DISTINCT image_path)
                    FROM image_tags
                    WHERE {' AND '.join(source_clauses)}
                    GROUP BY tag_norm, source
                    ORDER BY tag_norm, source COLLATE NOCASE
                    """,
                    source_params,
                ).fetchall()
                for tag_norm, source, count in source_rows:
                    sources_by_norm.setdefault(str(tag_norm), []).append((str(source or "user"), int(count)))
        return TagInventoryPage(
            items=tuple(
                TagInventoryItem(
                    display_tag=str(display_tag),
                    normalized_tag=str(tag_norm),
                    image_count=int(image_count),
                    sources=tuple(sources_by_norm.get(str(tag_norm), ())),
                )
                for tag_norm, display_tag, image_count in rows
            ),
            total_count=total_count,
        )

    def query_tagged_paths(
        self,
        tag: str,
        *,
        scope_path: str | Path | None = None,
        limit: int = 200,
        offset: int = 0,
        include_total: bool = True,
    ) -> TagPathPage:
        """Return a bounded page of database-backed photo paths for one tag."""

        normalized = self.normalize_tag(tag)
        if not normalized:
            return TagPathPage()
        clauses = ["tag_norm = ?"]
        params: list[object] = [normalized]
        scope_clause, scope_params = self._scope_sql(scope_path)
        if scope_clause:
            clauses.append(scope_clause)
            params.extend(scope_params)
        where = " AND ".join(clauses)
        page_size = max(1, min(500, int(limit)))
        page_offset = max(0, int(offset))
        with self._connection() as connection:
            total_count = None
            if include_total:
                total_count = int(connection.execute(f"SELECT COUNT(*) FROM image_tags WHERE {where}", params).fetchone()[0] or 0)
            rows = connection.execute(
                f"SELECT image_path FROM image_tags WHERE {where} ORDER BY image_path COLLATE NOCASE LIMIT ? OFFSET ?",
                [*params, page_size, page_offset],
            ).fetchall()
        return TagPathPage(paths=tuple(str(row[0]) for row in rows), total_count=total_count)

    def rename_tag(self, old_tag: str, new_tag: str) -> int:
        old_norm = self.normalize_tag(old_tag)
        new_norm = self.normalize_tag(new_tag)
        new_display = self.clean_display_tag(new_tag)
        if not old_norm or not new_norm or not new_display:
            return 0
        if old_norm == new_norm:
            return 0
        updated_at = self._utc_now()
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT image_path, mtime_ns, file_size, source
                FROM image_tags
                WHERE tag_norm=?
                """,
                (old_norm,),
            ).fetchall()
            affected_paths = {str(row[0]) for row in rows}
            for image_path, mtime_ns, file_size, source in rows:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO image_tags(
                        image_path, tag_norm, display_tag, mtime_ns, file_size, source, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(image_path),
                        new_norm,
                        new_display,
                        int(mtime_ns),
                        int(file_size),
                        str(source or "user"),
                        updated_at,
                    ),
                )
            connection.execute("DELETE FROM image_tags WHERE tag_norm=?", (old_norm,))
        return len(affected_paths)

    def delete_tag(self, tag: str) -> int:
        tag_norm = self.normalize_tag(tag)
        if not tag_norm:
            return 0
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT DISTINCT image_path FROM image_tags WHERE tag_norm=?",
                (tag_norm,),
            ).fetchall()
            connection.execute("DELETE FROM image_tags WHERE tag_norm=?", (tag_norm,))
        return len(rows)

    def apply_tag_edit(
        self,
        paths: list[str],
        *,
        add_tags: list[str] | tuple[str, ...] = (),
        remove_tags: list[str] | tuple[str, ...] = (),
        mirror_to_exif: bool = True,
        progress_callback=None,
        cancel_check=None,
    ) -> ImageTagEditResult:
        result = ImageTagEditResult()
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        add_map: dict[str, str] = {}
        for tag in add_tags:
            normalized = self.normalize_tag(tag)
            display_tag = self.clean_display_tag(tag)
            if not normalized or normalized in add_map:
                continue
            add_map[normalized] = display_tag
        remove_norms = {
            self.normalize_tag(tag)
            for tag in remove_tags
            if self.normalize_tag(tag)
        }
        if not ordered_paths or (not add_map and not remove_norms):
            return result
        current_tags_by_path = self.load_tags_for_paths(ordered_paths)
        total = len(ordered_paths)
        for index, image_path in enumerate(ordered_paths, start=1):
            if cancel_check and cancel_check():
                result.cancelled = True
                break
            if progress_callback:
                progress_callback(int(index * 100 / max(1, total)), f"Updating tags {index}/{total}")
            try:
                if not Path(image_path).is_file():
                    raise FileNotFoundError(image_path)
                current_by_norm: dict[str, str] = {}
                for display_tag in current_tags_by_path.get(image_path, ()):
                    normalized = self.normalize_tag(display_tag)
                    if normalized and normalized not in current_by_norm:
                        current_by_norm[normalized] = display_tag
                for normalized in remove_norms:
                    current_by_norm.pop(normalized, None)
                for normalized, display_tag in add_map.items():
                    current_by_norm.setdefault(normalized, display_tag)
                updated_tags = tuple(
                    current_by_norm[norm]
                    for norm in sorted(current_by_norm.keys(), key=lambda value: current_by_norm[value].casefold())
                )
                self._replace_rows_for_path(
                    image_path,
                    [
                        (normalized, display_tag, "user")
                        for normalized, display_tag in current_by_norm.items()
                    ],
                )
                result.affected_paths.append(image_path)
                if mirror_to_exif:
                    self._write_exif_tags(image_path, updated_tags)
                    result.mirrored_paths.append(image_path)
            except Exception as exc:
                result.failures.append(f"{image_path}: {exc}")
        return result

    def select_paths_by_tags(
        self,
        paths: list[str],
        tags: list[str] | tuple[str, ...],
        match_mode: str,
        *,
        tags_by_path: dict[str, tuple[str, ...]] | None = None,
        progress_callback=None,
        cancel_check=None,
    ) -> list[str]:
        requested = {
            self.normalize_tag(tag)
            for tag in tags
            if self.normalize_tag(tag)
        }
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        if not requested:
            return ordered_paths
        if tags_by_path is None:
            tags_by_path = self.load_tags_for_paths(
                ordered_paths,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        matched_paths: list[str] = []
        require_all = str(match_mode or "Any").strip().casefold() == "all"
        total = max(1, len(ordered_paths))
        for index, image_path in enumerate(ordered_paths, start=1):
            raise_if_cancelled(cancel_check)
            image_tags = {
                self.normalize_tag(tag)
                for tag in tags_by_path.get(image_path, ())
                if self.normalize_tag(tag)
            }
            if not image_tags:
                continue
            if require_all and requested.issubset(image_tags):
                matched_paths.append(image_path)
            elif not require_all and image_tags.intersection(requested):
                matched_paths.append(image_path)
            if progress_callback and (index == 1 or index == total or index % 50 == 0):
                progress_callback(int(index * 100 / total), f"Applying tag filter {index}/{total}")
        return matched_paths

    def summarize_paths(
        self,
        paths: list[str] | tuple[str, ...],
        *,
        tags_by_path: dict[str, tuple[str, ...]] | None = None,
    ) -> ClusterTagSummary:
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        if tags_by_path is None:
            tags_by_path = self.load_tags_for_paths(ordered_paths)
        counter: Counter[str] = Counter()
        display_by_norm: dict[str, str] = {}
        tagged_image_count = 0
        for image_path in ordered_paths:
            seen_norms: set[str] = set()
            for display_tag in tags_by_path.get(image_path, ()):
                normalized = self.normalize_tag(display_tag)
                clean_display = self.clean_display_tag(display_tag)
                if not normalized or not clean_display or normalized in seen_norms:
                    continue
                seen_norms.add(normalized)
                display_by_norm.setdefault(normalized, clean_display)
                counter[normalized] += 1
            if seen_norms:
                tagged_image_count += 1
        sorted_items = sorted(
            counter.items(),
            key=lambda item: (-item[1], display_by_norm[item[0]].casefold()),
        )
        return ClusterTagSummary(
            tag_counts={display_by_norm[norm]: count for norm, count in sorted_items},
            top_tags=tuple((display_by_norm[norm], count) for norm, count in sorted_items[:5]),
            tagged_image_count=tagged_image_count,
            unique_tag_count=len(sorted_items),
        )

    def sync_moved_paths(self, changed_paths: list[tuple[str, str]]) -> None:
        source_paths = [self._canonical_path(source) for source, _target in changed_paths if self._canonical_path(source)]
        rows_by_source = self._fetch_rows_by_path(source_paths)
        for source, target in changed_paths:
            source_path = self._canonical_path(source)
            target_path = self._canonical_path(target)
            if not source_path or not target_path:
                continue
            source_rows = rows_by_source.get(source_path, ())
            if not source_rows:
                continue
            self._replace_rows_for_path(
                target_path,
                [(tag_norm, display_tag, source_name) for tag_norm, display_tag, source_name, _mtime_ns, _file_size in source_rows],
            )
            self.remove_paths([source_path])

    def clone_tags_for_copies(self, changed_paths: list[tuple[str, str]]) -> None:
        source_paths = [self._canonical_path(source) for source, _target in changed_paths if self._canonical_path(source)]
        rows_by_source = self._fetch_rows_by_path(source_paths)
        for source, target in changed_paths:
            source_path = self._canonical_path(source)
            target_path = self._canonical_path(target)
            if not source_path or not target_path:
                continue
            source_rows = rows_by_source.get(source_path, ())
            if not source_rows:
                continue
            self._replace_rows_for_path(
                target_path,
                [(tag_norm, display_tag, source_name) for tag_norm, display_tag, source_name, _mtime_ns, _file_size in source_rows],
            )

    def remove_paths(self, paths: list[str]) -> None:
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        if not ordered_paths:
            return
        with self._connection() as connection:
            for start in range(0, len(ordered_paths), 900):
                chunk = ordered_paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                connection.execute(
                    f"DELETE FROM image_tags WHERE image_path IN ({placeholders})",
                    chunk,
                )

    def import_exif_tags_if_missing(self, path: str) -> tuple[str, ...]:
        image_path = self._canonical_path(path)
        if not image_path:
            return ()
        existing_rows = self._load_db_rows_by_path([image_path]).get(image_path, [])
        if existing_rows:
            return self._rows_to_display_tags(existing_rows)
        return self._import_exif_tags_for_path(image_path)

    def export_exif_tags(
        self,
        paths: list[str],
        *,
        tags_by_path: dict[str, tuple[str, ...]] | None = None,
    ) -> ImageTagEditResult:
        result = ImageTagEditResult()
        ordered_paths = [self._canonical_path(path) for path in paths if self._canonical_path(path)]
        db_tags = tags_by_path or self._db_display_tags_for_paths(ordered_paths)
        total = len(ordered_paths)
        for index, image_path in enumerate(ordered_paths, start=1):
            tags = tuple(db_tags.get(image_path, ()))
            try:
                self._write_exif_tags(image_path, tags)
                result.affected_paths.append(image_path)
                result.mirrored_paths.append(image_path)
            except Exception as exc:
                result.failures.append(f"{image_path}: {exc}")
        return result
