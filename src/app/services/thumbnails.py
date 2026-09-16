from __future__ import annotations

import hashlib
import math
import os
import sqlite3
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from contextlib import closing
from pathlib import Path
from threading import Lock, RLock

from PIL import Image, ImageOps
from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QImageReader, QPainter, QPen

from infra.settings import get_settings

THUMBNAIL_CACHE_VERSION = "exif_v2"
THUMBNAIL_CACHE_INDEX_NAME = ".thumbnail_index.sqlite3"
_PROCESS_DISK_INDEX_LOCK = RLock()
_PROCESS_DISK_INDEX_READY: OrderedDict[str, tuple[int, int, int, int, int]] = OrderedDict()
_PROCESS_DISK_INDEX_READY_LIMIT = 32


def _disk_index_cache_key(cache_dir: Path) -> str:
    try:
        return str(cache_dir.resolve())
    except OSError:
        return str(cache_dir.absolute())


def _disk_cache_directory_signature(cache_dir: Path) -> tuple[int, int, int]:
    entry_count = 0
    newest_mtime_ns = 0
    total_bytes = 0
    try:
        with os.scandir(cache_dir) as entries:
            for entry in entries:
                if not entry.name.casefold().endswith(".webp"):
                    continue
                stat = entry.stat(follow_symlinks=False)
                entry_count += 1
                newest_mtime_ns = max(newest_mtime_ns, int(stat.st_mtime_ns))
                total_bytes += int(stat.st_size)
    except OSError:
        return 0, 0, 0
    return entry_count, newest_mtime_ns, total_bytes


def _disk_index_file_identity(cache_dir: Path) -> tuple[int, int]:
    try:
        stat = (cache_dir / THUMBNAIL_CACHE_INDEX_NAME).stat()
        return int(stat.st_dev), int(stat.st_ino)
    except OSError:
        return 0, 0


def _disk_index_state_signature(cache_dir: Path) -> tuple[int, int, int, int, int]:
    return (*_disk_cache_directory_signature(cache_dir), *_disk_index_file_identity(cache_dir))


def _process_disk_index_is_ready(cache_dir: Path) -> bool:
    key = _disk_index_cache_key(cache_dir)
    signature = _disk_index_state_signature(cache_dir)
    with _PROCESS_DISK_INDEX_LOCK:
        if _PROCESS_DISK_INDEX_READY.get(key) != signature:
            return False
        _PROCESS_DISK_INDEX_READY.move_to_end(key)
        return True


def _mark_process_disk_index_ready(cache_dir: Path) -> None:
    key = _disk_index_cache_key(cache_dir)
    signature = _disk_index_state_signature(cache_dir)
    with _PROCESS_DISK_INDEX_LOCK:
        _PROCESS_DISK_INDEX_READY[key] = signature
        _PROCESS_DISK_INDEX_READY.move_to_end(key)
        while len(_PROCESS_DISK_INDEX_READY) > _PROCESS_DISK_INDEX_READY_LIMIT:
            _PROCESS_DISK_INDEX_READY.popitem(last=False)


def _discard_process_disk_index_state(cache_dir: Path) -> None:
    key = _disk_index_cache_key(cache_dir)
    with _PROCESS_DISK_INDEX_LOCK:
        _PROCESS_DISK_INDEX_READY.pop(key, None)


class ThumbnailService:
    def __init__(self, qimage_cache_size: int | None = None) -> None:
        self.settings = get_settings()
        self.qimage_cache_size = max(32, int(qimage_cache_size or self.settings.thumbnail_cache_size))
        self._qimage_cache: OrderedDict[tuple[str, int], QImage] = OrderedDict()
        self._qimage_cache_lock = Lock()
        self._disk_cache_lock = Lock()
        self._disk_index_initialized = False
        self._last_prune_s = 0.0
        self._new_since_prune = 0
        self._prune_every_n = 128
        self._min_prune_interval_s = 60.0
        self._pending_disk_accesses: dict[str, tuple[int, float]] = {}
        self._last_index_flush_s = 0.0
        self._index_flush_every_n = 64
        self._index_flush_interval_s = 5.0

    def set_qimage_cache_size(self, qimage_cache_size: int) -> None:
        self.qimage_cache_size = max(32, int(qimage_cache_size))
        with self._qimage_cache_lock:
            while len(self._qimage_cache) > self.qimage_cache_size:
                self._qimage_cache.popitem(last=False)

    def clear_memory_cache(self) -> None:
        with self._qimage_cache_lock:
            self._qimage_cache.clear()

    def clear_disk_cache(self) -> None:
        with self._disk_cache_lock:
            cache_dir = self.settings.thumbnail_cache_dir
            try:
                shutil.rmtree(cache_dir)
            except FileNotFoundError:
                pass
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._disk_index_initialized = False
            self._new_since_prune = 0
            self._last_prune_s = 0.0
            self._pending_disk_accesses.clear()
            self._last_index_flush_s = 0.0
            _discard_process_disk_index_state(cache_dir)

    def thumbnail_path(self, image_path: str, size: int) -> Path:
        source = Path(image_path).resolve()
        stat = source.stat()
        digest = hashlib.sha256(f"{THUMBNAIL_CACHE_VERSION}|{source}|{stat.st_mtime_ns}|{size}".encode("utf-8")).hexdigest()
        return self.settings.thumbnail_cache_dir / f"{digest}.webp"

    def invalidate_paths(self, image_paths: Sequence[str], *, sizes: Sequence[int] | None = None) -> None:
        normalized_paths = {str(path or "").strip() for path in image_paths if str(path or "").strip()}
        if not normalized_paths:
            return
        size_filter = {max(1, int(size)) for size in sizes or () if int(size) > 0}
        with self._qimage_cache_lock:
            stale_keys = [
                key
                for key in self._qimage_cache
                if str(key[0]) in normalized_paths and (not size_filter or int(key[1]) in size_filter)
            ]
            for key in stale_keys:
                self._qimage_cache.pop(key, None)
        removed_cache_paths: list[Path] = []
        for image_path in normalized_paths:
            if size_filter:
                for size in size_filter:
                    try:
                        cache_path = self.thumbnail_path(image_path, size)
                        cache_path.unlink(missing_ok=True)
                        removed_cache_paths.append(cache_path)
                    except Exception:
                        continue
        self._remove_disk_cache_entries(removed_cache_paths)

    def ensure_thumbnail(self, image_path: str, size: int) -> Path:
        thumb_path = self.thumbnail_path(image_path, size)
        if thumb_path.exists():
            self._record_disk_cache_entry(thumb_path)
            return thumb_path
        thumb_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            left = (size - image.width) // 2
            top = (size - image.height) // 2
            canvas.paste(image, (left, top))
            canvas.save(thumb_path, format="WEBP", quality=82, method=4)
        self._record_disk_cache_entry(thumb_path, created=True)
        self._new_since_prune += 1
        if self._new_since_prune >= self._prune_every_n and (time.time() - self._last_prune_s) >= self._min_prune_interval_s:
            self.prune_cache()
            self._last_prune_s = time.time()
            self._new_since_prune = 0
        return thumb_path

    def _load_scaled_source_qimage(self, image_path: str, size: int) -> QImage:
        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid() and source_size.width() > 0 and source_size.height() > 0:
            target_size = source_size.scaled(
                QSize(size, size),
                Qt.AspectRatioMode.KeepAspectRatio,
            )
            if target_size.isValid() and target_size.width() > 0 and target_size.height() > 0:
                reader.setScaledSize(target_size)
        image = reader.read()
        if image.isNull():
            return image
        scaled = image.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        canvas = QImage(size, size, QImage.Format.Format_RGB32)
        canvas.fill(QColor("#FFFFFF"))
        painter = QPainter(canvas)
        try:
            offset_x = max(0, (size - scaled.width()) // 2)
            offset_y = max(0, (size - scaled.height()) // 2)
            painter.drawImage(offset_x, offset_y, scaled)
        finally:
            painter.end()
        return canvas

    def load_qimage(self, image_path: str, size: int) -> QImage:
        cache_key = (image_path, size)
        with self._qimage_cache_lock:
            cached = self._qimage_cache.get(cache_key)
            if cached is not None:
                self._qimage_cache.move_to_end(cache_key)
                return QImage(cached)

        thumb_path = self.thumbnail_path(image_path, size)
        image = QImage(str(thumb_path)) if thumb_path.exists() else QImage()
        if not image.isNull() and thumb_path.exists():
            self._record_disk_cache_entry(thumb_path)
        if image.isNull():
            image = self._load_scaled_source_qimage(image_path, size)
        if image.isNull():
            thumb_path = self.ensure_thumbnail(image_path, size)
            image = QImage(str(thumb_path))
        with self._qimage_cache_lock:
            self._qimage_cache[cache_key] = QImage(image)
            self._qimage_cache.move_to_end(cache_key)
            while len(self._qimage_cache) > self.qimage_cache_size:
                self._qimage_cache.popitem(last=False)
        return image

    @staticmethod
    def _normalize_contact_sheet_size(size: QSize | tuple[int, int] | int) -> tuple[int, int]:
        if isinstance(size, QSize):
            return max(1, int(size.width())), max(1, int(size.height()))
        if isinstance(size, tuple):
            width, height = size
            return max(1, int(width)), max(1, int(height))
        edge = max(1, int(size))
        return edge, edge

    def build_contact_sheet_qimage(
        self,
        image_paths: Sequence[str],
        size: QSize | tuple[int, int] | int,
        *,
        max_items: int = 9,
        columns: int = 3,
        cancel_check: Callable[[], bool] | None = None,
    ) -> QImage:
        width, height = self._normalize_contact_sheet_size(size)
        canvas = QImage(width, height, QImage.Format.Format_ARGB32)
        canvas.fill(QColor("#F8FAFC"))

        preview_paths = [str(path) for path in image_paths[: max(1, int(max_items))]]
        if not preview_paths:
            painter = QPainter(canvas)
            try:
                painter.setPen(QPen(QColor("#CBD5E1")))
                placeholder = canvas.rect().adjusted(12, 12, -12, -12)
                painter.drawRect(placeholder)
                painter.drawLine(placeholder.topLeft(), placeholder.bottomRight())
                painter.drawLine(placeholder.bottomLeft(), placeholder.topRight())
            finally:
                painter.end()
            return canvas

        grid_columns = max(1, int(columns))
        grid_rows = max(1, math.ceil(max(1, int(max_items)) / grid_columns))
        gap = 4
        padding = 6
        tile_width = max(1, (width - (padding * 2) - (gap * (grid_columns - 1))) // grid_columns)
        tile_height = max(1, (height - (padding * 2) - (gap * (grid_rows - 1))) // grid_rows)
        thumb_request_size = max(tile_width, tile_height)
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        border_pen = QPen(QColor("#CBD5E1"))
        painter.setPen(border_pen)
        loaded_tiles = 0
        try:
            for tile_index in range(max(1, int(max_items))):
                if cancel_check is not None and cancel_check():
                    return canvas
                row = tile_index // grid_columns
                column = tile_index % grid_columns
                left = padding + (column * (tile_width + gap))
                top = padding + (row * (tile_height + gap))
                tile_rect = QRect(left, top, tile_width, tile_height)
                painter.fillRect(tile_rect, QColor("#E2E8F0"))
                painter.drawRect(tile_rect.adjusted(0, 0, -1, -1))
                if tile_index >= len(preview_paths):
                    continue
                image_path = preview_paths[tile_index]
                tile_image = QImage()
                try:
                    thumb_path = self.ensure_thumbnail(image_path, thumb_request_size)
                    tile_image = QImage(str(thumb_path))
                    if tile_image.isNull():
                        reader = QImageReader(str(image_path))
                        reader.setAutoTransform(True)
                        tile_image = reader.read()
                except Exception:
                    tile_image = QImage()
                if tile_image.isNull():
                    continue
                scaled = tile_image.scaled(
                    tile_width,
                    tile_height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                offset_x = left + max(0, (tile_width - scaled.width()) // 2)
                offset_y = top + max(0, (tile_height - scaled.height()) // 2)
                painter.drawImage(offset_x, offset_y, scaled)
                loaded_tiles += 1
            if loaded_tiles == 0:
                painter.setPen(QPen(QColor("#CBD5E1")))
                placeholder = canvas.rect().adjusted(12, 12, -12, -12)
                painter.drawRect(placeholder)
                painter.drawLine(placeholder.topLeft(), placeholder.bottomRight())
                painter.drawLine(placeholder.bottomLeft(), placeholder.topRight())
        finally:
            painter.end()
        return canvas

    def _disk_index_path(self) -> Path:
        return self.settings.thumbnail_cache_dir / THUMBNAIL_CACHE_INDEX_NAME

    def _connect_disk_index(self) -> sqlite3.Connection:
        self.settings.thumbnail_cache_dir.mkdir(parents=True, exist_ok=True)
        connection: sqlite3.Connection | None = None
        for attempt in range(2):
            try:
                connection = sqlite3.connect(str(self._disk_index_path()), timeout=10.0)
                connection.execute("PRAGMA journal_mode=WAL;")
                connection.execute("PRAGMA synchronous=NORMAL;")
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS thumbnail_cache_entries (
                        cache_path TEXT PRIMARY KEY,
                        size_bytes INTEGER NOT NULL,
                        last_access REAL NOT NULL
                    )
                    """
                )
                if not self._disk_index_initialized or not _process_disk_index_is_ready(
                    self.settings.thumbnail_cache_dir
                ):
                    # Several views own their own ThumbnailService. The first
                    # one reconciles interruption/corruption state; the rest
                    # share that completed result instead of rescanning the
                    # same bounded cache directory on their worker threads.
                    with _PROCESS_DISK_INDEX_LOCK:
                        if not _process_disk_index_is_ready(self.settings.thumbnail_cache_dir):
                            self._reconcile_disk_index(connection)
                            _mark_process_disk_index_ready(self.settings.thumbnail_cache_dir)
                    self._disk_index_initialized = True
                return connection
            except sqlite3.DatabaseError as error:
                if connection is not None:
                    connection.close()
                    connection = None
                if attempt or not self._is_corrupt_disk_index(error):
                    raise
                self._reset_corrupt_disk_index()
        raise RuntimeError("Thumbnail cache index could not be opened")

    @staticmethod
    def _is_corrupt_disk_index(error: sqlite3.DatabaseError) -> bool:
        message = str(error).casefold()
        return "malformed" in message or "not a database" in message or "file is encrypted" in message

    def _reset_corrupt_disk_index(self) -> None:
        index_path = self._disk_index_path()
        for path in (index_path, Path(f"{index_path}-wal"), Path(f"{index_path}-shm")):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
        self._disk_index_initialized = False
        self._pending_disk_accesses.clear()
        self._last_index_flush_s = 0.0
        _discard_process_disk_index_state(self.settings.thumbnail_cache_dir)

    def _managed_disk_cache_path(self, path: Path, *, resolved_cache_root: Path | None = None) -> bool:
        """Return whether ``path`` is a safe thumbnail cache entry.

        Reconciliation visits every normal file immediately below the configured
        cache directory.  That common case needs no costly canonical-path walk:
        it cannot escape the directory unless it is a symlink.  Retain the
        canonical check for symlinks and database-originated paths, where the
        safety boundary matters.
        """

        if path.suffix.casefold() != ".webp":
            return False
        cache_dir = self.settings.thumbnail_cache_dir
        try:
            if path.parent == cache_dir and not path.is_symlink():
                return True
            cache_root = resolved_cache_root or cache_dir.resolve()
            resolved_path = path.resolve(strict=False)
        except OSError:
            return False
        return resolved_path.parent == cache_root

    def _reconcile_disk_index(self, connection: sqlite3.Connection) -> None:
        """Make the thumbnail index match completed cache files after interruption."""

        try:
            resolved_cache_root = self.settings.thumbnail_cache_dir.resolve()
        except OSError:
            return

        indexed_paths = {
            str(cache_path)
            for (cache_path,) in connection.execute("SELECT cache_path FROM thumbnail_cache_entries")
        }
        existing_rows: list[tuple[str, int, float]] = []
        existing_paths: set[str] = set()
        for path in self.settings.thumbnail_cache_dir.glob("*.webp"):
            if not self._managed_disk_cache_path(path, resolved_cache_root=resolved_cache_root):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            path_text = str(path)
            existing_paths.add(path_text)
            if path_text not in indexed_paths:
                existing_rows.append((path_text, int(stat.st_size), float(stat.st_atime)))

        stale_paths = [
            (cache_path,)
            for cache_path in indexed_paths
            if not self._managed_disk_cache_path(Path(cache_path), resolved_cache_root=resolved_cache_root)
            or cache_path not in existing_paths
        ]
        if stale_paths:
            connection.executemany("DELETE FROM thumbnail_cache_entries WHERE cache_path=?", stale_paths)
        if existing_rows:
            connection.executemany(
                "INSERT OR IGNORE INTO thumbnail_cache_entries(cache_path, size_bytes, last_access) VALUES (?, ?, ?)",
                existing_rows,
            )
        connection.commit()

    def _record_disk_cache_entry(self, cache_path: Path, *, created: bool = False) -> None:
        try:
            size_bytes = int(cache_path.stat().st_size)
        except OSError:
            return
        with self._disk_cache_lock:
            access_time = time.time()
            self._pending_disk_accesses[str(cache_path)] = (size_bytes, access_time)
            should_flush = bool(
                created
                or len(self._pending_disk_accesses) >= self._index_flush_every_n
                or (access_time - self._last_index_flush_s) >= self._index_flush_interval_s
            )
            if not should_flush:
                return
            if self._flush_disk_accesses_locked():
                _mark_process_disk_index_ready(self.settings.thumbnail_cache_dir)

    def _flush_disk_accesses_locked(self, connection: sqlite3.Connection | None = None) -> bool:
        if not self._pending_disk_accesses:
            return True
        rows = [
            (cache_path, int(size_bytes), float(last_access))
            for cache_path, (size_bytes, last_access) in self._pending_disk_accesses.items()
        ]
        owns_connection = connection is None
        active_connection = connection
        try:
            if active_connection is None:
                active_connection = self._connect_disk_index()
            active_connection.executemany(
                """
                INSERT INTO thumbnail_cache_entries(cache_path, size_bytes, last_access)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_path) DO UPDATE SET
                    size_bytes=excluded.size_bytes,
                    last_access=excluded.last_access
                """,
                rows,
            )
            if owns_connection:
                active_connection.commit()
        except sqlite3.Error:
            return False
        finally:
            if owns_connection and active_connection is not None:
                active_connection.close()
        self._pending_disk_accesses.clear()
        self._last_index_flush_s = time.time()
        return True

    def _remove_disk_cache_entries(self, cache_paths: Sequence[Path]) -> None:
        values = [(str(path),) for path in cache_paths if str(path)]
        if not values:
            return
        with self._disk_cache_lock:
            for (cache_path,) in values:
                self._pending_disk_accesses.pop(cache_path, None)
            try:
                with closing(self._connect_disk_index()) as connection:
                    with connection:
                        connection.executemany("DELETE FROM thumbnail_cache_entries WHERE cache_path=?", values)
            except sqlite3.Error:
                return
            _mark_process_disk_index_ready(self.settings.thumbnail_cache_dir)

    def prune_cache(self) -> None:
        max_bytes = int(self.settings.thumbnail_cache_max_bytes)
        with self._disk_cache_lock:
            try:
                with closing(self._connect_disk_index()) as connection:
                    with connection:
                        self._flush_disk_accesses_locked(connection)
                        total_size = int(
                            connection.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM thumbnail_cache_entries").fetchone()[0]
                            or 0
                        )
                        if total_size <= max_bytes:
                            return
                        rows = connection.execute(
                            "SELECT cache_path, size_bytes FROM thumbnail_cache_entries ORDER BY last_access ASC"
                        ).fetchall()
                        removed_paths: list[tuple[str]] = []
                        for cache_path_text, recorded_size in rows:
                            cache_path = Path(str(cache_path_text))
                            if not self._managed_disk_cache_path(cache_path):
                                removed_paths.append((str(cache_path),))
                                continue
                            try:
                                actual_size = int(cache_path.stat().st_size)
                                cache_path.unlink(missing_ok=True)
                            except FileNotFoundError:
                                actual_size = int(recorded_size or 0)
                            except OSError:
                                continue
                            removed_paths.append((str(cache_path),))
                            total_size -= max(0, actual_size)
                            if total_size <= max_bytes:
                                break
                        if removed_paths:
                            connection.executemany("DELETE FROM thumbnail_cache_entries WHERE cache_path=?", removed_paths)
                            _mark_process_disk_index_ready(self.settings.thumbnail_cache_dir)
            except sqlite3.Error:
                return
