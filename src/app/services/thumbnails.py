from __future__ import annotations

import hashlib
import math
import shutil
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from threading import Lock

from PIL import Image
from PyQt6.QtCore import QRect, QSize, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen

from infra.settings import get_settings


class ThumbnailService:
    def __init__(self, qimage_cache_size: int | None = None) -> None:
        self.settings = get_settings()
        self.qimage_cache_size = max(32, int(qimage_cache_size or self.settings.thumbnail_cache_size))
        self._qimage_cache: OrderedDict[tuple[str, int], QImage] = OrderedDict()
        self._qimage_cache_lock = Lock()
        self._last_prune_s = 0.0
        self._new_since_prune = 0
        self._prune_every_n = 40
        self._min_prune_interval_s = 3.0

    def set_qimage_cache_size(self, qimage_cache_size: int) -> None:
        self.qimage_cache_size = max(32, int(qimage_cache_size))
        with self._qimage_cache_lock:
            while len(self._qimage_cache) > self.qimage_cache_size:
                self._qimage_cache.popitem(last=False)

    def clear_memory_cache(self) -> None:
        with self._qimage_cache_lock:
            self._qimage_cache.clear()

    def clear_disk_cache(self) -> None:
        cache_dir = self.settings.thumbnail_cache_dir
        try:
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            pass
        cache_dir.mkdir(parents=True, exist_ok=True)

    def thumbnail_path(self, image_path: str, size: int) -> Path:
        source = Path(image_path).resolve()
        stat = source.stat()
        digest = hashlib.sha256(f"{source}|{stat.st_mtime_ns}|{size}".encode("utf-8")).hexdigest()
        return self.settings.thumbnail_cache_dir / f"{digest}.webp"

    def ensure_thumbnail(self, image_path: str, size: int) -> Path:
        thumb_path = self.thumbnail_path(image_path, size)
        if thumb_path.exists():
            return thumb_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image.thumbnail((size, size))
            canvas = Image.new("RGB", (size, size), (255, 255, 255))
            left = (size - image.width) // 2
            top = (size - image.height) // 2
            canvas.paste(image, (left, top))
            canvas.save(thumb_path, format="WEBP", quality=82, method=4)
        self._new_since_prune += 1
        if self._new_since_prune >= self._prune_every_n and (time.time() - self._last_prune_s) >= self._min_prune_interval_s:
            self.prune_cache()
            self._last_prune_s = time.time()
            self._new_since_prune = 0
        return thumb_path

    def load_qimage(self, image_path: str, size: int) -> QImage:
        cache_key = (image_path, size)
        with self._qimage_cache_lock:
            cached = self._qimage_cache.get(cache_key)
            if cached is not None:
                self._qimage_cache.move_to_end(cache_key)
                return QImage(cached)

        thumb_path = self.ensure_thumbnail(image_path, size)
        image = QImage(str(thumb_path))
        if image.isNull():
            image = QImage(image_path)
        scaled = image.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        with self._qimage_cache_lock:
            self._qimage_cache[cache_key] = QImage(scaled)
            self._qimage_cache.move_to_end(cache_key)
            while len(self._qimage_cache) > self.qimage_cache_size:
                self._qimage_cache.popitem(last=False)
        return scaled

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
                painter.setPen(QColor("#64748B"))
                painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "No preview")
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
                        tile_image = QImage(str(image_path))
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
                painter.setPen(QColor("#64748B"))
                painter.drawText(canvas.rect(), Qt.AlignmentFlag.AlignCenter, "Preview unavailable")
        finally:
            painter.end()
        return canvas

    def prune_cache(self) -> None:
        files = [path for path in self.settings.thumbnail_cache_dir.glob("*.webp") if path.is_file()]
        total_size = sum(path.stat().st_size for path in files)
        if total_size <= self.settings.thumbnail_cache_max_bytes:
            return
        files.sort(key=lambda path: path.stat().st_atime if path.exists() else time.time())
        for path in files:
            try:
                file_size = path.stat().st_size
                path.unlink(missing_ok=True)
                total_size -= file_size
                if total_size <= self.settings.thumbnail_cache_max_bytes:
                    break
            except OSError:
                continue
