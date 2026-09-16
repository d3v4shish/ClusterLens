from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from uuid import uuid4

from PIL import ExifTags, Image

from app.path_scope import roots_scope_sql
from infra.cancel import Cancelled, raise_if_cancelled
from infra.logging_config import get_logger
from infra.settings import get_settings


LOGGER = get_logger(__name__)
CATALOG_SCHEMA_VERSION = 2
CATALOG_WRITE_BATCH_SIZE = 48
FILENAME_DATE_POLICIES = ("metadata_only", "metadata_or_filename", "prefer_filename", "filename_only")


@dataclass(frozen=True)
class LibraryRoot:
    root_id: str
    path: str
    display_name: str
    enabled: bool
    added_at: str
    last_scan_at: str = ""
    last_error: str = ""


@dataclass(frozen=True)
class CatalogAsset:
    image_path: str
    root_id: str
    captured_at: str
    capture_source: str
    modified_at: str
    camera: str
    width: int
    height: int
    file_size: int
    file_ext: str
    exif: dict[str, str] = field(default_factory=dict)
    xmp_text: str = ""
    mtime_ns: int = 0
    metadata_mtime_ns: int = 0
    metadata_size: int = 0


@dataclass(frozen=True)
class CatalogPage:
    items: tuple[CatalogAsset, ...] = ()
    total_count: int = 0
    next_offset: int | None = None


@dataclass(frozen=True)
class TimelineMonth:
    """One capture-month bucket from the lightweight Library timeline query."""

    year: int
    month: int
    image_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class TimelineYear:
    """A chronologically ordered collection of timeline months."""

    year: int
    months: tuple[TimelineMonth, ...] = ()


@dataclass(frozen=True)
class CatalogTimeline:
    """A full filtered timeline without per-photo EXIF/XMP payloads."""

    years: tuple[TimelineYear, ...] = ()
    total_count: int = 0

    @property
    def image_paths(self) -> tuple[str, ...]:
        """Flattened capture-descending paths for explicit Gallery hand-off."""

        return tuple(
            image_path
            for year in self.years
            for month in year.months
            for image_path in month.image_paths
        )


@dataclass(frozen=True)
class CatalogQuery:
    root_ids: tuple[str, ...] = ()
    # ``None`` keeps the historical all-enabled-roots query. An empty tuple is
    # an explicit active scope with no roots and therefore matches nothing.
    scope_paths: tuple[str, ...] | None = None
    folder: str = ""
    start_at: str = ""
    end_at: str = ""
    camera: str = ""
    file_ext: str = ""
    text: str = ""
    order: str = "captured_desc"
    offset: int = 0
    limit: int = 240


@dataclass(frozen=True)
class SmartAlbum:
    album_id: str
    name: str
    query: dict[str, object]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ClusterContextRecord:
    context_id: str
    cluster_key: str
    representative_path: str
    title: str
    description: str
    keywords: tuple[str, ...]
    provider: str
    model: str
    status: str
    cache_key: str
    created_at: str
    updated_at: str
    members: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClusterContextPage:
    items: tuple[ClusterContextRecord, ...] = ()
    total_count: int = 0


class LibraryCatalogService:
    """Managed, source-read-only catalog for explicitly registered photo roots.

    Discovery and metadata parsing are intentionally called by workers. The
    service only mutates ClusterLens runtime data; it never changes originals,
    XMP sidecars, tags, or face labels.
    """

    def __init__(self, *, db_path: str | Path | None = None, filename_date_policy: str = "metadata_or_filename") -> None:
        settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else settings.cache_dir / "library_catalog.sqlite3"
        self._fts_available = True
        self._filename_date_policy = self.normalize_filename_date_policy(filename_date_policy)
        self._init_db()

    @staticmethod
    def normalize_filename_date_policy(value: str) -> str:
        policy = str(value or "").strip().lower()
        return policy if policy in FILENAME_DATE_POLICIES else "metadata_or_filename"

    @property
    def filename_date_policy(self) -> str:
        return self._filename_date_policy

    def set_filename_date_policy(self, value: str) -> None:
        """Set the source-safe capture-time policy used on the next catalog scan."""
        self._filename_date_policy = self.normalize_filename_date_policy(value)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA temp_store=MEMORY")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS library_roots (
                    root_id TEXT PRIMARY KEY,
                    canonical_path TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    added_at TEXT NOT NULL,
                    last_scan_at TEXT NOT NULL DEFAULT '',
                    last_error TEXT NOT NULL DEFAULT ''
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS catalog_assets (
                    image_path TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL REFERENCES library_roots(root_id) ON DELETE CASCADE,
                    parent_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    captured_at TEXT NOT NULL DEFAULT '',
                    capture_source TEXT NOT NULL DEFAULT 'modified',
                    modified_at TEXT NOT NULL,
                    camera TEXT NOT NULL DEFAULT '',
                    width INTEGER NOT NULL DEFAULT 0,
                    height INTEGER NOT NULL DEFAULT 0,
                    exif_json TEXT NOT NULL DEFAULT '{}',
                    xmp_text TEXT NOT NULL DEFAULT '',
                    metadata_mtime_ns INTEGER NOT NULL DEFAULT 0,
                    metadata_size INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(catalog_assets)").fetchall()}
            if "metadata_mtime_ns" not in columns:
                connection.execute("ALTER TABLE catalog_assets ADD COLUMN metadata_mtime_ns INTEGER NOT NULL DEFAULT 0")
            if "metadata_size" not in columns:
                connection.execute("ALTER TABLE catalog_assets ADD COLUMN metadata_size INTEGER NOT NULL DEFAULT 0")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_catalog_assets_root_capture ON catalog_assets(root_id, captured_at DESC, image_path)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_catalog_assets_parent ON catalog_assets(parent_path)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_catalog_assets_camera ON catalog_assets(camera)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS smart_albums (
                    album_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    query_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_contexts (
                    context_id TEXT PRIMARY KEY,
                    cluster_key TEXT NOT NULL,
                    cache_key TEXT NOT NULL UNIQUE,
                    representative_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cluster_contexts_cluster ON cluster_contexts(cluster_key)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cluster_context_members (
                    context_id TEXT NOT NULL REFERENCES cluster_contexts(context_id) ON DELETE CASCADE,
                    image_path TEXT NOT NULL,
                    member_order INTEGER NOT NULL,
                    PRIMARY KEY(context_id, image_path)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS duplicate_review_feedback (
                    decision_key TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    left_path TEXT NOT NULL,
                    right_path TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_duplicate_feedback_paths ON duplicate_review_feedback(left_path, right_path)")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_cleanup_feedback (
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(image_path, face_index, person_name)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_cleanup_exclusions (
                    left_path TEXT NOT NULL,
                    left_face_index INTEGER NOT NULL,
                    right_path TEXT NOT NULL,
                    right_face_index INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(left_path, left_face_index, right_path, right_face_index)
                )
                """
            )
            try:
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS catalog_asset_fts USING fts5(
                        image_path UNINDEXED, file_name, camera, exif_text, xmp_text,
                        tokenize='unicode61'
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS cluster_context_fts USING fts5(
                        context_id UNINDEXED, title, description, keywords, representative_path,
                        tokenize='unicode61'
                    )
                    """
                )
            except sqlite3.OperationalError:
                self._fts_available = False
                LOGGER.warning("SQLite FTS5 is unavailable; catalog text search will use bounded LIKE matching.")

    @staticmethod
    def canonical_path(path: str | Path) -> str:
        try:
            return str(Path(path).expanduser().resolve(strict=False))
        except (OSError, RuntimeError):
            return os.path.abspath(os.path.normpath(str(path)))

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def register_root(self, path: str | Path, *, display_name: str = "") -> LibraryRoot:
        canonical = self.canonical_path(path)
        candidate = Path(canonical)
        if not candidate.is_dir():
            raise ValueError(f"Library root does not exist or is not a directory: {canonical}")
        now = self._now()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM library_roots WHERE canonical_path=?", (canonical,)).fetchone()
            if row is None:
                root_id = uuid4().hex
                name = str(display_name or candidate.name or canonical).strip()
                connection.execute(
                    "INSERT INTO library_roots(root_id, canonical_path, display_name, enabled, added_at) VALUES (?, ?, ?, 1, ?)",
                    (root_id, canonical, name, now),
                )
            else:
                root_id = str(row["root_id"])
                name = str(display_name or row["display_name"] or candidate.name or canonical).strip()
                connection.execute(
                    "UPDATE library_roots SET display_name=?, enabled=1, last_error='' WHERE root_id=?",
                    (name, root_id),
                )
        return self.get_root(root_id)

    def list_roots(self, *, include_disabled: bool = True) -> list[LibraryRoot]:
        query = "SELECT * FROM library_roots"
        if not include_disabled:
            query += " WHERE enabled=1"
        query += " ORDER BY display_name COLLATE NOCASE, canonical_path"
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [self._root_from_row(row) for row in rows]

    def get_root(self, root_id: str) -> LibraryRoot:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM library_roots WHERE root_id=?", (str(root_id),)).fetchone()
        if row is None:
            raise KeyError(f"Library root not found: {root_id}")
        return self._root_from_row(row)

    def set_root_enabled(self, root_id: str, enabled: bool) -> None:
        with self._connect() as connection:
            changed = connection.execute("UPDATE library_roots SET enabled=? WHERE root_id=?", (int(bool(enabled)), str(root_id))).rowcount
        if not changed:
            raise KeyError(f"Library root not found: {root_id}")

    def remove_root(self, root_id: str) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            changed = connection.execute("DELETE FROM library_roots WHERE root_id=?", (str(root_id),)).rowcount
        if not changed:
            raise KeyError(f"Library root not found: {root_id}")

    def scan_roots(
        self,
        root_ids: Iterable[str] | None = None,
        *,
        force: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        wanted = {str(root_id) for root_id in root_ids or () if str(root_id)}
        roots = [root for root in self.list_roots(include_disabled=False) if not wanted or root.root_id in wanted]
        totals = {"roots": len(roots), "discovered": 0, "updated": 0, "unchanged": 0, "removed": 0, "failures": 0}
        for index, root in enumerate(roots, start=1):
            raise_if_cancelled(cancel_check)
            if progress_callback:
                progress_callback(int((index - 1) * 100 / max(1, len(roots))), f"Scanning library root {index}/{len(roots)}: {root.display_name}")
            try:
                stats = self.scan_root(
                    root.root_id,
                    force=force,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
                for key in ("discovered", "updated", "unchanged", "removed"):
                    totals[key] += int(stats.get(key, 0))
            except Cancelled:
                raise
            except Exception as exc:
                totals["failures"] += 1
                self._set_root_error(root.root_id, str(exc))
                LOGGER.warning("Library root scan failed for %s: %s", root.path, exc)
        if progress_callback:
            progress_callback(100, "Library catalog scan complete")
        return totals

    def scan_root(
        self,
        root_id: str,
        *,
        force: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        root = self.get_root(root_id)
        if not root.enabled:
            return {"discovered": 0, "updated": 0, "unchanged": 0, "removed": 0}
        directory = Path(root.path)
        if not directory.is_dir():
            raise FileNotFoundError(f"Library root is unavailable: {root.path}")
        extensions = {str(ext).casefold() for ext in get_settings().image_extensions}
        paths: list[Path] = []
        for path in directory.rglob("*"):
            raise_if_cancelled(cancel_check)
            if path.is_file() and path.suffix.casefold() in extensions:
                paths.append(path)
        paths.sort(key=lambda value: self.canonical_path(value))
        known = self._asset_fingerprints_for_root(root.root_id)
        current_paths: set[str] = set()
        updated = 0
        unchanged = 0
        total = len(paths)
        pending_upserts: list[tuple[CatalogAsset, int]] = []

        def _flush_pending() -> None:
            if pending_upserts:
                self._upsert_assets(pending_upserts)
                pending_upserts.clear()

        for index, path in enumerate(paths, start=1):
            raise_if_cancelled(cancel_check)
            canonical = self.canonical_path(path)
            current_paths.add(canonical)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            metadata_mtime_ns, metadata_size = self._metadata_sidecar_fingerprint(path)
            fingerprint = (int(stat.st_mtime_ns), int(stat.st_size), metadata_mtime_ns, metadata_size)
            if not force and known.get(canonical) == fingerprint:
                unchanged += 1
                continue
            asset = self._read_asset(
                canonical,
                root.root_id,
                stat,
                metadata_mtime_ns=metadata_mtime_ns,
                metadata_size=metadata_size,
            )
            pending_upserts.append((asset, int(stat.st_mtime_ns)))
            if len(pending_upserts) >= CATALOG_WRITE_BATCH_SIZE:
                _flush_pending()
            updated += 1
            if progress_callback and (index == total or index % 24 == 0):
                progress_callback(int(index * 100 / max(1, total)), f"Cataloging {index}/{total} photos in {root.display_name}")
        _flush_pending()
        stale = sorted(set(known) - current_paths)
        if stale:
            self._delete_assets(stale)
        with self._connect() as connection:
            connection.execute(
                "UPDATE library_roots SET last_scan_at=?, last_error='' WHERE root_id=?",
                (self._now(), root.root_id),
            )
        return {"discovered": total, "updated": updated, "unchanged": unchanged, "removed": len(stale)}

    def query_assets(self, query: CatalogQuery | None = None) -> CatalogPage:
        query = query or CatalogQuery()
        where, args = self._asset_where(query)
        order = "captured_at DESC, image_path" if str(query.order) != "captured_asc" else "captured_at ASC, image_path"
        limit = max(1, min(1000, int(query.limit)))
        offset = max(0, int(query.offset))
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM catalog_assets asset "
                    f"JOIN library_roots root ON asset.root_id=root.root_id WHERE {where}",
                    args,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"SELECT asset.* FROM catalog_assets asset "
                f"JOIN library_roots root ON asset.root_id=root.root_id "
                f"WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [*args, limit, offset],
            ).fetchall()
        items = tuple(self._asset_from_row(row) for row in rows)
        next_offset = offset + len(items) if offset + len(items) < total else None
        return CatalogPage(items=items, total_count=total, next_offset=next_offset)

    def query_timeline(
        self,
        query: CatalogQuery | None = None,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> CatalogTimeline:
        """Read every filtered path/date pair for the virtual Library Timeline.

        This deliberately ignores the paged ``offset`` and ``limit`` fields on
        ``CatalogQuery``.  Timeline is a full filtered catalog view, while
        Search and smart-album pages retain their bounded ``query_assets``
        contract.  Only two indexed scalar columns are selected, and the
        cursor is consumed in bounded batches so worker cancellation remains
        prompt even for a very large registered library.
        """

        query = query or CatalogQuery()
        where, args = self._asset_where(query)
        raise_if_cancelled(cancel_check)
        if progress_callback is not None:
            progress_callback(0, "Reading catalogued capture dates…")

        buckets: dict[tuple[int, int], list[str]] = {}
        with self._connect() as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM catalog_assets asset "
                    f"JOIN library_roots root ON asset.root_id=root.root_id WHERE {where}",
                    args,
                ).fetchone()[0]
            )
            cursor = connection.execute(
                f"SELECT asset.image_path, asset.captured_at FROM catalog_assets asset "
                f"JOIN library_roots root ON asset.root_id=root.root_id "
                f"WHERE {where} ORDER BY asset.captured_at DESC, asset.image_path",
                args,
            )
            processed = 0
            last_progress_at = 0
            while rows := cursor.fetchmany(256):
                raise_if_cancelled(cancel_check)
                for row in rows:
                    image_path = str(row["image_path"] or "")
                    if not image_path:
                        continue
                    bucket = self._timeline_bucket(str(row["captured_at"] or ""))
                    buckets.setdefault(bucket, []).append(image_path)
                processed += len(rows)
                if progress_callback is not None and (processed == total or processed - last_progress_at >= 1024):
                    progress_callback(
                        int(processed * 100 / max(1, total)),
                        f"Reading timeline dates {processed}/{total}…",
                    )
                    last_progress_at = processed
        raise_if_cancelled(cancel_check)
        if progress_callback is not None:
            progress_callback(100, f"Prepared {total} timeline photos")

        years: list[TimelineYear] = []
        months_by_year: dict[int, list[TimelineMonth]] = {}
        for (year, month), image_paths in buckets.items():
            months_by_year.setdefault(year, []).append(TimelineMonth(year, month, tuple(image_paths)))
        for year, months in months_by_year.items():
            years.append(TimelineYear(year, tuple(months)))
        return CatalogTimeline(tuple(years), total)

    def get_asset(self, image_path: str | Path) -> CatalogAsset | None:
        canonical = self.canonical_path(image_path)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM catalog_assets WHERE image_path=?", (canonical,)).fetchone()
        return self._asset_from_row(row) if row is not None else None

    def save_smart_album(self, name: str, query: CatalogQuery | dict[str, object], *, album_id: str = "") -> SmartAlbum:
        clean_name = " ".join(str(name or "").split())
        if not clean_name:
            raise ValueError("Smart album name is required.")
        payload = self.query_to_payload(query) if isinstance(query, CatalogQuery) else self._json_object(query)
        now = self._now()
        with self._connect() as connection:
            if album_id:
                changed = connection.execute(
                    "UPDATE smart_albums SET name=?, query_json=?, updated_at=? WHERE album_id=?",
                    (clean_name, json.dumps(payload, sort_keys=True), now, str(album_id)),
                ).rowcount
                if not changed:
                    raise KeyError(f"Smart album not found: {album_id}")
                record_id = str(album_id)
            else:
                record_id = uuid4().hex
                connection.execute(
                    "INSERT INTO smart_albums(album_id, name, query_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                    (record_id, clean_name, json.dumps(payload, sort_keys=True), now, now),
                )
        return self.get_smart_album(record_id)

    def list_smart_albums(self) -> list[SmartAlbum]:
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM smart_albums ORDER BY name COLLATE NOCASE, album_id").fetchall()
        return [self._album_from_row(row) for row in rows]

    def get_smart_album(self, album_id: str) -> SmartAlbum:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM smart_albums WHERE album_id=?", (str(album_id),)).fetchone()
        if row is None:
            raise KeyError(f"Smart album not found: {album_id}")
        return self._album_from_row(row)

    def delete_smart_album(self, album_id: str) -> None:
        with self._connect() as connection:
            changed = connection.execute("DELETE FROM smart_albums WHERE album_id=?", (str(album_id),)).rowcount
        if not changed:
            raise KeyError(f"Smart album not found: {album_id}")

    def query_smart_album(self, album_id: str, *, offset: int = 0, limit: int = 240) -> CatalogPage:
        album = self.get_smart_album(album_id)
        payload = dict(album.query)
        payload.update({"offset": int(offset), "limit": int(limit)})
        return self.query_assets(self.payload_to_query(payload))

    def store_cluster_context(
        self,
        *,
        cluster_key: str,
        cache_key: str,
        representative_path: str,
        title: str,
        description: str,
        keywords: Iterable[str],
        provider: str,
        model: str,
        status: str,
        members: Iterable[str],
    ) -> ClusterContextRecord:
        clean_keywords = tuple(dict.fromkeys(" ".join(str(item or "").split()) for item in keywords if str(item or "").strip()))
        clean_members = tuple(dict.fromkeys(self.canonical_path(path) for path in members if str(path or "").strip()))
        now = self._now()
        context_id = uuid4().hex
        with self._connect() as connection:
            existing = connection.execute("SELECT context_id, created_at FROM cluster_contexts WHERE cache_key=?", (str(cache_key),)).fetchone()
            if existing is not None:
                context_id = str(existing["context_id"])
                created_at = str(existing["created_at"])
                connection.execute(
                    """
                    UPDATE cluster_contexts SET cluster_key=?, representative_path=?, title=?, description=?, keywords_json=?,
                    provider=?, model=?, status=?, updated_at=? WHERE context_id=?
                    """,
                    (str(cluster_key), self.canonical_path(representative_path), str(title), str(description), json.dumps(clean_keywords), str(provider), str(model), str(status), now, context_id),
                )
                connection.execute("DELETE FROM cluster_context_members WHERE context_id=?", (context_id,))
            else:
                created_at = now
                connection.execute(
                    """
                    INSERT INTO cluster_contexts(context_id, cluster_key, cache_key, representative_path, title, description,
                    keywords_json, provider, model, status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (context_id, str(cluster_key), str(cache_key), self.canonical_path(representative_path), str(title), str(description), json.dumps(clean_keywords), str(provider), str(model), str(status), now, now),
                )
            connection.executemany(
                "INSERT INTO cluster_context_members(context_id, image_path, member_order) VALUES (?, ?, ?)",
                [(context_id, path, index) for index, path in enumerate(clean_members)],
            )
            if self._fts_available:
                connection.execute("DELETE FROM cluster_context_fts WHERE context_id=?", (context_id,))
                connection.execute(
                    "INSERT INTO cluster_context_fts(context_id, title, description, keywords, representative_path) VALUES (?, ?, ?, ?, ?)",
                    (context_id, str(title), str(description), " ".join(clean_keywords), self.canonical_path(representative_path)),
                )
        return ClusterContextRecord(context_id, str(cluster_key), self.canonical_path(representative_path), str(title), str(description), clean_keywords, str(provider), str(model), str(status), str(cache_key), created_at, now, clean_members)

    def load_cluster_context(self, cache_key: str) -> ClusterContextRecord | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cluster_contexts WHERE cache_key=?", (str(cache_key),)).fetchone()
        return self._context_from_row(row) if row is not None else None

    def search_cluster_contexts(self, text: str, *, limit: int = 80) -> ClusterContextPage:
        query = " ".join(str(text or "").split())
        if not query:
            return ClusterContextPage()
        with self._connect() as connection:
            if self._fts_available:
                try:
                    rows = connection.execute(
                        """
                        SELECT context.* FROM cluster_context_fts fts
                        JOIN cluster_contexts context ON context.context_id=fts.context_id
                        WHERE cluster_context_fts MATCH ? ORDER BY bm25(cluster_context_fts), context.updated_at DESC LIMIT ?
                        """,
                        (self._fts_query(query), max(1, min(500, int(limit)))),
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []
            else:
                pattern = f"%{query}%"
                rows = connection.execute(
                    "SELECT * FROM cluster_contexts WHERE title LIKE ? OR description LIKE ? OR keywords_json LIKE ? ORDER BY updated_at DESC LIMIT ?",
                    (pattern, pattern, pattern, max(1, min(500, int(limit)))),
                ).fetchall()
        items = tuple(self._context_from_row(row) for row in rows)
        return ClusterContextPage(items=items, total_count=len(items))

    def catalog_storage_bytes(self) -> int:
        total = 0
        for path in (self.db_path, self.db_path.with_name(f"{self.db_path.name}-wal"), self.db_path.with_name(f"{self.db_path.name}-shm")):
            try:
                total += int(path.stat().st_size)
            except FileNotFoundError:
                pass
        return total

    def clear_cluster_contexts(self) -> int:
        with self._connect() as connection:
            count = int(connection.execute("SELECT COUNT(*) FROM cluster_contexts").fetchone()[0])
            connection.execute("DELETE FROM cluster_context_members")
            connection.execute("DELETE FROM cluster_contexts")
            if self._fts_available:
                connection.execute("DELETE FROM cluster_context_fts")
        return count

    def clear_library_cache(
        self,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ) -> int:
        """Clear derived catalog rows and generated context without source writes.

        Explicit roots, saved albums, and curation feedback are user choices,
        so this leaves them intact. A later root refresh deterministically
        recreates the cached asset metadata and FTS rows.
        """
        raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(0, "Clearing derived Library catalog data")
        with self._connect() as connection:
            asset_count = int(connection.execute("SELECT COUNT(*) FROM catalog_assets").fetchone()[0])
            if self._fts_available:
                connection.execute("DELETE FROM catalog_asset_fts")
                connection.execute("DELETE FROM cluster_context_fts")
            connection.execute("DELETE FROM cluster_context_members")
            connection.execute("DELETE FROM cluster_contexts")
            connection.execute("DELETE FROM catalog_assets")
        raise_if_cancelled(cancel_check)
        if progress_callback:
            progress_callback(70, "Compacting Library catalog storage")
        with self._connect() as connection:
            connection.set_progress_handler(
                lambda: 1 if cancel_check is not None and cancel_check() else 0,
                1_000,
            )
            try:
                connection.execute("VACUUM")
            except sqlite3.OperationalError:
                raise_if_cancelled(cancel_check)
                raise
            finally:
                connection.set_progress_handler(None, 0)
        if progress_callback:
            progress_callback(100, "Library catalog cache cleared")
        return asset_count

    def set_duplicate_feedback(self, *, decision_key: str, kind: str, left_path: str, right_path: str, state: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO duplicate_review_feedback(decision_key, kind, left_path, right_path, state, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(decision_key) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at
                """,
                (str(decision_key), str(kind), self.canonical_path(left_path), self.canonical_path(right_path), str(state), self._now()),
            )

    def duplicate_feedback(self, decision_keys: Iterable[str]) -> dict[str, str]:
        keys = [str(key) for key in decision_keys if str(key)]
        if not keys:
            return {}
        result: dict[str, str] = {}
        with self._connect() as connection:
            for start in range(0, len(keys), 900):
                chunk = keys[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT decision_key, state FROM duplicate_review_feedback WHERE decision_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                result.update({str(row["decision_key"]): str(row["state"]) for row in rows})
        return result

    def set_face_cleanup_feedback(self, image_path: str, face_index: int, person_name: str, state: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO face_cleanup_feedback(image_path, face_index, person_name, state, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_path, face_index, person_name) DO UPDATE SET state=excluded.state, updated_at=excluded.updated_at
                """,
                (self.canonical_path(image_path), int(face_index), str(person_name).strip(), str(state), self._now()),
            )

    def face_cleanup_feedback(self, refs: Iterable[tuple[str, int]]) -> dict[tuple[str, int, str], str]:
        values = [(self.canonical_path(path), int(index)) for path, index in refs if str(path or "").strip()]
        if not values:
            return {}
        result: dict[tuple[str, int, str], str] = {}
        with self._connect() as connection:
            for start in range(0, len(values), 450):
                chunk = values[start : start + 450]
                clauses = " OR ".join("(image_path=? AND face_index=?)" for _path, _index in chunk)
                args = [value for pair in chunk for value in pair]
                rows = connection.execute(
                    f"SELECT image_path, face_index, person_name, state FROM face_cleanup_feedback WHERE {clauses}",
                    args,
                ).fetchall()
                result.update({(str(row["image_path"]), int(row["face_index"]), str(row["person_name"])): str(row["state"]) for row in rows})
        return result

    def add_face_cleanup_exclusion(self, left: tuple[str, int], right: tuple[str, int]) -> None:
        normalized = sorted(((self.canonical_path(left[0]), int(left[1])), (self.canonical_path(right[0]), int(right[1]))))
        if normalized[0] == normalized[1]:
            return
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO face_cleanup_exclusions(left_path, left_face_index, right_path, right_face_index, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(left_path, left_face_index, right_path, right_face_index) DO UPDATE SET updated_at=excluded.updated_at
                """,
                (*normalized[0], *normalized[1], self._now()),
            )

    def face_cleanup_exclusions(self, refs: Iterable[tuple[str, int]]) -> set[tuple[tuple[str, int], tuple[str, int]]]:
        normalized = {(self.canonical_path(path), int(index)) for path, index in refs if str(path or "").strip()}
        if not normalized:
            return set()
        result: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        with self._connect() as connection:
            rows = connection.execute("SELECT left_path, left_face_index, right_path, right_face_index FROM face_cleanup_exclusions").fetchall()
        for row in rows:
            left = (str(row["left_path"]), int(row["left_face_index"]))
            right = (str(row["right_path"]), int(row["right_face_index"]))
            if left in normalized or right in normalized:
                result.add((left, right))
        return result

    def representative_for_members(self, members: Iterable[str]) -> str:
        """Choose a reproducible cluster representative: capture time, then path.

        Catalogued EXIF capture time is preferred. Files outside the catalog use
        their modification time so manual cluster description still works for
        a currently selected folder that has not been registered yet.
        """
        paths = tuple(sorted({self.canonical_path(path) for path in members if str(path or "").strip()}))
        if not paths:
            return ""
        captured_by_path: dict[str, str] = {}
        with self._connect() as connection:
            for start in range(0, len(paths), 900):
                chunk = paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT image_path, captured_at FROM catalog_assets WHERE image_path IN ({placeholders})",
                    chunk,
                ).fetchall()
                captured_by_path.update({str(row["image_path"]): str(row["captured_at"] or "") for row in rows})
        candidates: list[tuple[str, str]] = []
        for path in paths:
            captured = captured_by_path.get(path, "")
            if not captured:
                try:
                    captured = datetime.fromtimestamp(Path(path).stat().st_mtime, timezone.utc).isoformat(timespec="seconds")
                except OSError:
                    captured = "9999-12-31T23:59:59+00:00"
            candidates.append((captured, path))
        return min(candidates)[1]

    @staticmethod
    def build_context_cache_key(
        *,
        cluster_key: str,
        members: Iterable[str],
        representative_path: str,
        metadata_fingerprint: str,
        provider: str,
        model: str,
        prompt_version: str,
    ) -> str:
        payload = {
            "cluster_key": str(cluster_key),
            "members": sorted(LibraryCatalogService.canonical_path(path) for path in members if str(path or "").strip()),
            "representative": LibraryCatalogService.canonical_path(representative_path),
            "metadata": str(metadata_fingerprint),
            "provider": str(provider),
            "model": str(model),
            "prompt_version": str(prompt_version),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _asset_where(self, query: CatalogQuery) -> tuple[str, list[object]]:
        clauses = ["root.enabled=1", "asset.root_id=root.root_id"]
        args: list[object] = []
        if query.root_ids:
            placeholders = ",".join("?" for _ in query.root_ids)
            clauses.append(f"asset.root_id IN ({placeholders})")
            args.extend(str(value) for value in query.root_ids)
        if query.scope_paths is not None:
            scope_clause, scope_args = roots_scope_sql("asset.image_path", query.scope_paths)
            clauses.append(scope_clause)
            args.extend(scope_args)
        if query.folder:
            folder = self.canonical_path(query.folder).rstrip("/\\")
            clauses.append("(asset.parent_path=? OR asset.parent_path LIKE ?)")
            args.extend([folder, f"{folder}{os.sep}%"])
        if query.start_at:
            clauses.append("asset.captured_at>=?")
            args.append(str(query.start_at))
        if query.end_at:
            clauses.append("asset.captured_at<=?")
            args.append(str(query.end_at))
        if query.camera:
            clauses.append("asset.camera LIKE ?")
            args.append(f"%{str(query.camera)}%")
        if query.file_ext:
            clauses.append("asset.file_ext=?")
            args.append(str(query.file_ext).casefold())
        text = " ".join(str(query.text or "").split())
        if text:
            if self._fts_available:
                clauses.append("asset.image_path IN (SELECT image_path FROM catalog_asset_fts WHERE catalog_asset_fts MATCH ?)")
                args.append(self._fts_query(text))
            else:
                clauses.append("(asset.file_name LIKE ? OR asset.camera LIKE ? OR asset.exif_json LIKE ? OR asset.xmp_text LIKE ?)")
                pattern = f"%{text}%"
                args.extend([pattern, pattern, pattern, pattern])
        return " AND ".join(clauses), args

    def _asset_fingerprints_for_root(self, root_id: str) -> dict[str, tuple[int, int, int, int]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path, mtime_ns, file_size, metadata_mtime_ns, metadata_size FROM catalog_assets WHERE root_id=?",
                (str(root_id),),
            ).fetchall()
        return {
            str(row["image_path"]): (
                int(row["mtime_ns"]),
                int(row["file_size"]),
                int(row["metadata_mtime_ns"]),
                int(row["metadata_size"]),
            )
            for row in rows
        }

    def read_asset_metadata(self, image_path: str | Path) -> CatalogAsset:
        """Read one source asset without cataloging or changing its source files."""
        canonical = self.canonical_path(image_path)
        path = Path(canonical)
        stat = path.stat()
        metadata_mtime_ns, metadata_size = self._metadata_sidecar_fingerprint(path)
        return self._read_asset(
            canonical,
            "",
            stat,
            metadata_mtime_ns=metadata_mtime_ns,
            metadata_size=metadata_size,
        )

    def _read_asset(
        self,
        image_path: str,
        root_id: str,
        stat,
        *,
        metadata_mtime_ns: int = 0,
        metadata_size: int = 0,
    ) -> CatalogAsset:
        path = Path(image_path)
        width = height = 0
        camera = ""
        exif: dict[str, str] = {}
        exif_captured_at = ""
        embedded_xmp = ""
        try:
            with Image.open(path) as image:
                width, height = image.size
                for tag, value in dict(image.getexif() or {}).items():
                    label = str(ExifTags.TAGS.get(tag, tag))
                    rendered = self._scalar_text(value)
                    if rendered:
                        exif[label] = rendered
                camera = " ".join(part for part in (exif.get("Make", ""), exif.get("Model", "")) if part).strip()
                raw_date = exif.get("DateTimeOriginal") or exif.get("DateTimeDigitized") or exif.get("DateTime") or ""
                exif_captured_at = self._normalize_exif_datetime(raw_date)
                xmp_candidate = image.info.get("XML:com.adobe.xmp") or image.info.get("xmp") or ""
                embedded_xmp = self._scalar_text(xmp_candidate, max_length=32768)
        except Exception as exc:
            LOGGER.debug("Catalog metadata read failed for %s: %s", image_path, exc)
        captured_at, capture_source = self._resolve_capture_time(
            exif_captured_at=exif_captured_at,
            filename_captured_at=self.filename_capture_datetime(path.name),
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        )
        xmp_text = self._read_xmp_text(path, embedded_xmp)
        return CatalogAsset(
            image_path=image_path,
            root_id=str(root_id),
            captured_at=captured_at,
            capture_source=capture_source,
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            camera=camera,
            width=width,
            height=height,
            file_size=int(stat.st_size),
            file_ext=path.suffix.casefold(),
            exif=exif,
            xmp_text=xmp_text,
            mtime_ns=int(stat.st_mtime_ns),
            metadata_mtime_ns=int(metadata_mtime_ns),
            metadata_size=int(metadata_size),
        )

    def _upsert_asset(self, asset: CatalogAsset, *, mtime_ns: int) -> None:
        self._upsert_assets(((asset, int(mtime_ns)),))

    def _upsert_assets(self, assets: Iterable[tuple[CatalogAsset, int]]) -> None:
        pending = tuple(assets)
        if not pending:
            return
        with self._connect() as connection:
            for asset, mtime_ns in pending:
                self._upsert_asset_on_connection(connection, asset, mtime_ns=int(mtime_ns))

    def _upsert_asset_on_connection(self, connection: sqlite3.Connection, asset: CatalogAsset, *, mtime_ns: int) -> None:
        image_path = self.canonical_path(asset.image_path)
        connection.execute(
            """
            INSERT INTO catalog_assets(image_path, root_id, parent_path, file_name, file_ext, mtime_ns, file_size,
            captured_at, capture_source, modified_at, camera, width, height, exif_json, xmp_text, metadata_mtime_ns, metadata_size)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(image_path) DO UPDATE SET
            root_id=excluded.root_id, parent_path=excluded.parent_path, file_name=excluded.file_name,
            file_ext=excluded.file_ext, mtime_ns=excluded.mtime_ns, file_size=excluded.file_size,
            captured_at=excluded.captured_at, capture_source=excluded.capture_source, modified_at=excluded.modified_at,
            camera=excluded.camera, width=excluded.width, height=excluded.height, exif_json=excluded.exif_json,
            xmp_text=excluded.xmp_text, metadata_mtime_ns=excluded.metadata_mtime_ns, metadata_size=excluded.metadata_size
            """,
            (image_path, asset.root_id, str(Path(image_path).parent), Path(image_path).name, asset.file_ext, int(mtime_ns), asset.file_size, asset.captured_at, asset.capture_source, asset.modified_at, asset.camera, asset.width, asset.height, json.dumps(asset.exif, sort_keys=True), asset.xmp_text, asset.metadata_mtime_ns, asset.metadata_size),
        )
        if self._fts_available:
            connection.execute("DELETE FROM catalog_asset_fts WHERE image_path=?", (image_path,))
            connection.execute(
                "INSERT INTO catalog_asset_fts(image_path, file_name, camera, exif_text, xmp_text) VALUES (?, ?, ?, ?, ?)",
                (image_path, Path(image_path).name, asset.camera, " ".join(f"{key} {value}" for key, value in asset.exif.items()), asset.xmp_text),
            )

    def _delete_assets(self, image_paths: Iterable[str]) -> None:
        paths = [self.canonical_path(path) for path in image_paths if str(path or "").strip()]
        with self._connect() as connection:
            for start in range(0, len(paths), 900):
                chunk = paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                if self._fts_available:
                    connection.execute(f"DELETE FROM catalog_asset_fts WHERE image_path IN ({placeholders})", chunk)
                connection.execute(f"DELETE FROM catalog_assets WHERE image_path IN ({placeholders})", chunk)

    def _set_root_error(self, root_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute("UPDATE library_roots SET last_error=? WHERE root_id=?", (str(error), str(root_id)))

    def _context_from_row(self, row: sqlite3.Row) -> ClusterContextRecord:
        with self._connect() as connection:
            member_rows = connection.execute("SELECT image_path FROM cluster_context_members WHERE context_id=? ORDER BY member_order", (str(row["context_id"]),)).fetchall()
        try:
            keywords = tuple(str(item) for item in json.loads(str(row["keywords_json"]) or "[]") if str(item).strip())
        except (TypeError, ValueError):
            keywords = ()
        return ClusterContextRecord(str(row["context_id"]), str(row["cluster_key"]), str(row["representative_path"]), str(row["title"]), str(row["description"]), keywords, str(row["provider"]), str(row["model"]), str(row["status"]), str(row["cache_key"]), str(row["created_at"]), str(row["updated_at"]), tuple(str(item["image_path"]) for item in member_rows))

    @staticmethod
    def _root_from_row(row: sqlite3.Row) -> LibraryRoot:
        return LibraryRoot(str(row["root_id"]), str(row["canonical_path"]), str(row["display_name"]), bool(row["enabled"]), str(row["added_at"]), str(row["last_scan_at"]), str(row["last_error"]))

    @staticmethod
    def _asset_from_row(row: sqlite3.Row) -> CatalogAsset:
        try:
            exif = {str(key): str(value) for key, value in dict(json.loads(str(row["exif_json"]) or "{}")).items()}
        except (TypeError, ValueError):
            exif = {}
        return CatalogAsset(
            str(row["image_path"]), str(row["root_id"]), str(row["captured_at"]), str(row["capture_source"]),
            str(row["modified_at"]), str(row["camera"]), int(row["width"]), int(row["height"]),
            int(row["file_size"]), str(row["file_ext"]), exif, str(row["xmp_text"]), int(row["mtime_ns"]),
            int(row["metadata_mtime_ns"]), int(row["metadata_size"]),
        )

    @staticmethod
    def _album_from_row(row: sqlite3.Row) -> SmartAlbum:
        try:
            payload = LibraryCatalogService._json_object(json.loads(str(row["query_json"]) or "{}"))
        except (TypeError, ValueError):
            payload = {}
        return SmartAlbum(str(row["album_id"]), str(row["name"]), payload, str(row["created_at"]), str(row["updated_at"]))

    @staticmethod
    def _scalar_text(value: object, *, max_length: int = 8192) -> str:
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8", errors="replace")
            except Exception:
                return ""
        if isinstance(value, (tuple, list, dict, set)):
            return ""
        text = " ".join(str(value or "").replace("\x00", " ").split())
        return text[:max_length]

    @staticmethod
    def _normalize_exif_datetime(value: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        for pattern in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw[:19], pattern).replace(tzinfo=timezone.utc).isoformat(timespec="seconds")
            except ValueError:
                pass
        return ""

    @staticmethod
    def filename_capture_datetime(file_name: str) -> str:
        """Extract an unambiguous year-first date/time from a camera filename.

        Ambiguous locale formats are deliberately ignored.  The result is
        derived catalog data only; choosing it never changes EXIF/XMP.
        """
        match = re.search(
            r"(?<!\d)(?P<year>19\d{2}|20\d{2})[-_.]?(?P<month>0[1-9]|1[0-2])[-_.]?(?P<day>0[1-9]|[12]\d|3[01])"
            r"(?:[T _.-]?(?P<hour>[01]\d|2[0-3])[:._-]?(?P<minute>[0-5]\d)(?:[:._-]?(?P<second>[0-5]\d))?)?(?!\d)",
            str(file_name or ""),
        )
        if match is None:
            return ""
        try:
            captured = datetime(
                int(match.group("year")),
                int(match.group("month")),
                int(match.group("day")),
                int(match.group("hour") or 0),
                int(match.group("minute") or 0),
                int(match.group("second") or 0),
                tzinfo=timezone.utc,
            )
        except ValueError:
            return ""
        return captured.isoformat(timespec="seconds")

    def _resolve_capture_time(
        self,
        *,
        exif_captured_at: str,
        filename_captured_at: str,
        modified_at: str,
    ) -> tuple[str, str]:
        values = {
            "exif": str(exif_captured_at or ""),
            "filename": str(filename_captured_at or ""),
            "modified": str(modified_at or ""),
        }
        priorities = {
            "metadata_only": ("exif", "modified"),
            "metadata_or_filename": ("exif", "filename", "modified"),
            "prefer_filename": ("filename", "exif", "modified"),
            "filename_only": ("filename", "modified"),
        }[self._filename_date_policy]
        for source in priorities:
            if values[source]:
                return values[source], source
        return "", "modified"

    @staticmethod
    def _timeline_bucket(captured_at: str) -> tuple[int, int]:
        """Return a stable year/month bucket without consulting source media."""

        raw = str(captured_at or "").strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return parsed.year, parsed.month
        except ValueError:
            # Cataloging normally falls back to the file modification time,
            # but older/hand-edited databases can contain an empty timestamp.
            # Keep these rows visible in one explicit, chronologically-last
            # group instead of dropping them or touching the source file.
            return 0, 0

    @staticmethod
    def _read_xmp_text(path: Path, embedded_xmp: str) -> str:
        chunks = [embedded_xmp]
        sidecar = LibraryCatalogService._metadata_sidecar_path(path)
        if sidecar.exists():
            try:
                chunks.append(sidecar.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                pass
        return " ".join(" ".join(chunk.replace("\x00", " ").split()) for chunk in chunks if chunk)[:65536]

    @staticmethod
    def _metadata_sidecar_path(path: Path) -> Path:
        adjacent = path.with_suffix(f"{path.suffix}.xmp")
        return adjacent if adjacent.exists() else path.with_suffix(".xmp")

    @classmethod
    def _metadata_sidecar_fingerprint(cls, path: Path) -> tuple[int, int]:
        sidecar = cls._metadata_sidecar_path(path)
        try:
            stat = sidecar.stat()
            return int(stat.st_mtime_ns), int(stat.st_size)
        except OSError:
            return 0, 0

    @staticmethod
    def _fts_query(text: str) -> str:
        terms = [term.replace('"', "") for term in str(text).split() if term.strip()]
        return " AND ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _json_object(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        return json.loads(json.dumps(value, sort_keys=True))

    @staticmethod
    def query_to_payload(query: CatalogQuery | dict[str, object]) -> dict[str, object]:
        if isinstance(query, dict):
            return LibraryCatalogService._json_object(query)
        return {
            "root_ids": list(query.root_ids),
            "scope_paths": list(query.scope_paths) if query.scope_paths is not None else None,
            "folder": query.folder, "start_at": query.start_at,
            "end_at": query.end_at, "camera": query.camera, "file_ext": query.file_ext,
            "text": query.text, "order": query.order,
        }

    @staticmethod
    def payload_to_query(payload: dict[str, object]) -> CatalogQuery:
        return CatalogQuery(
            root_ids=tuple(str(item) for item in list(payload.get("root_ids", []) or []) if str(item)),
            scope_paths=(
                tuple(str(item) for item in list(payload.get("scope_paths", []) or []) if str(item))
                if payload.get("scope_paths") is not None
                else None
            ),
            folder=str(payload.get("folder", "") or ""), start_at=str(payload.get("start_at", "") or ""),
            end_at=str(payload.get("end_at", "") or ""), camera=str(payload.get("camera", "") or ""),
            file_ext=str(payload.get("file_ext", "") or ""), text=str(payload.get("text", "") or ""),
            order=str(payload.get("order", "captured_desc") or "captured_desc"),
            offset=int(payload.get("offset", 0) or 0), limit=int(payload.get("limit", 240) or 240),
        )
