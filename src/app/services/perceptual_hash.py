from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import imagehash
from PIL import Image

from app.path_scope import folder_scope_sql, path_is_within_scope
from infra.cancel import raise_if_cancelled
from infra.settings import get_settings


@dataclass(frozen=True)
class HashMatch:
    image_path: str
    hash_distance: int
    hash_backend: str


class PerceptualHashIndexService:
    SUPPORTED_HASHES = ("phash", "dhash", "whash")

    def __init__(self, *, db_path: str | Path | None = None, reset_db: bool = False) -> None:
        self.settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else (self.settings.cache_dir / "perceptual_hashes.db")
        if reset_db:
            try:
                self.db_path.unlink(missing_ok=True)
            except Exception:
                pass
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("PRAGMA synchronous=NORMAL;")
        connection.execute("PRAGMA temp_store=MEMORY;")
        # Indexing and duplicate review can overlap in separate workers. Wait
        # briefly for a writer instead of surfacing SQLite's transient lock.
        connection.execute("PRAGMA busy_timeout=5000;")
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS phash_index (
                    image_path TEXT PRIMARY KEY,
                    phash_hex TEXT NOT NULL,
                    dhash_hex TEXT NOT NULL DEFAULT '',
                    whash_hex TEXT NOT NULL DEFAULT '',
                    phash_u64 BLOB NOT NULL DEFAULT X'0000000000000000',
                    dhash_u64 BLOB NOT NULL DEFAULT X'0000000000000000',
                    whash_u64 BLOB NOT NULL DEFAULT X'0000000000000000',
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL
                )
                """
            )
            cols = {row[1] for row in connection.execute("PRAGMA table_info(phash_index)").fetchall()}
            for column_name in ("dhash_hex", "whash_hex"):
                if column_name not in cols:
                    connection.execute(
                        f"ALTER TABLE phash_index ADD COLUMN {column_name} TEXT NOT NULL DEFAULT ''"
                    )
            for column_name in ("phash_u64", "dhash_u64", "whash_u64"):
                if column_name not in cols:
                    connection.execute(
                        f"ALTER TABLE phash_index ADD COLUMN {column_name} BLOB NOT NULL DEFAULT X'0000000000000000'"
                    )

    @staticmethod
    def _u64_blob_from_hex(hex_str: str) -> bytes:
        if not hex_str:
            return b"\x00" * 8
        try:
            value = int(hex_str, 16)
        except Exception:
            return b"\x00" * 8
        return int(value).to_bytes(8, byteorder="big", signed=False)

    def build_index(self, image_paths: list[str], *, cancel_check=None) -> int:
        rows = []
        for image_path in image_paths:
            raise_if_cancelled(cancel_check)
            path = Path(image_path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            hashes = self.compute_hash_hexes(str(path))
            if not hashes.get("phash"):
                continue
            rows.append(
                (
                    str(path),
                    hashes["phash"],
                    hashes["dhash"],
                    hashes["whash"],
                    self._u64_blob_from_hex(hashes["phash"]),
                    self._u64_blob_from_hex(hashes["dhash"]),
                    self._u64_blob_from_hex(hashes["whash"]),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO phash_index(
                    image_path, phash_hex, dhash_hex, whash_hex,
                    phash_u64, dhash_u64, whash_u64,
                    mtime_ns, file_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_path) DO UPDATE SET
                    phash_hex=excluded.phash_hex,
                    dhash_hex=excluded.dhash_hex,
                    whash_hex=excluded.whash_hex,
                    phash_u64=excluded.phash_u64,
                    dhash_u64=excluded.dhash_u64,
                    whash_u64=excluded.whash_u64,
                    mtime_ns=excluded.mtime_ns,
                    file_size=excluded.file_size
                """,
                rows,
            )
        return len(rows)

    def ensure_index(self, image_paths: list[str], *, cancel_check=None) -> int:
        """
        Ensure the given paths exist in the hash index and are up to date.
        Returns number of (re)indexed rows.
        """
        if not image_paths:
            return 0
        stats: dict[str, tuple[int, int]] = {}
        for image_path in image_paths:
            raise_if_cancelled(cancel_check)
            path = Path(image_path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            stats[str(path)] = (int(stat.st_mtime_ns), int(stat.st_size))
        if not stats:
            return 0

        existing: dict[str, tuple[int, int]] = {}
        with self._connect() as connection:
            paths = list(stats)
            for start in range(0, len(paths), 900):
                raise_if_cancelled(cancel_check)
                chunk = paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT image_path, mtime_ns, file_size FROM phash_index WHERE image_path IN ({placeholders})",
                    chunk,
                ).fetchall()
                existing.update({row[0]: (int(row[1]), int(row[2])) for row in rows})

        to_update = [p for p, st in stats.items() if existing.get(p) != st]
        return self.build_index(to_update, cancel_check=cancel_check)

    def sync_directory(
        self,
        directory: str,
        image_paths: list[str],
        *,
        prune_missing: bool = True,
        cancel_check=None,
    ) -> dict[str, int]:
        updated = self.ensure_index(image_paths, cancel_check=cancel_check)
        removed = 0
        if prune_missing:
            scope_clause, scope_args = folder_scope_sql("image_path", str(Path(directory)))
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT image_path FROM phash_index WHERE {scope_clause}",
                    scope_args,
                ).fetchall()
            existing = {row[0] for row in rows if path_is_within_scope(str(row[0]), directory)}
            current = {str(Path(p)) for p in image_paths if p}
            to_remove = sorted(existing - current)
            if to_remove:
                with self._connect() as connection:
                    for start in range(0, len(to_remove), 900):
                        raise_if_cancelled(cancel_check)
                        chunk = to_remove[start : start + 900]
                        placeholders = ",".join("?" for _ in chunk)
                        removed += connection.execute(
                            f"DELETE FROM phash_index WHERE image_path IN ({placeholders})",
                            chunk,
                        ).rowcount
        return {"updated": int(updated), "removed": int(removed)}

    def query_similar(
        self,
        query_image_path: str,
        max_distance: int,
        candidate_paths: set[str] | None = None,
        hash_backend: str = "phash",
        *,
        cancel_check=None,
    ) -> list[HashMatch]:
        if hash_backend not in self.SUPPORTED_HASHES:
            raise ValueError(f"Unsupported hash backend: {hash_backend}")
        query_hash = self.compute_hash_hex(query_image_path, hash_backend)
        if not query_hash:
            return []
        query_value = int(query_hash, 16)
        u64_column = f"{hash_backend}_u64"

        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT image_path, {u64_column} FROM phash_index"
            ).fetchall()

        matches: list[HashMatch] = []
        for image_path, blob in rows:
            raise_if_cancelled(cancel_check)
            if candidate_paths is not None and image_path not in candidate_paths:
                continue
            if blob is None:
                continue
            try:
                cand_value = int.from_bytes(blob, byteorder="big", signed=False)
            except Exception:
                continue
            distance = (query_value ^ cand_value).bit_count()
            if distance <= max_distance:
                matches.append(HashMatch(image_path=image_path, hash_distance=int(distance), hash_backend=hash_backend))
        matches.sort(key=lambda item: (item.hash_distance, item.image_path))
        return matches

    def load_hash_values(
        self,
        image_paths: list[str],
        *,
        hash_backend: str = "phash",
        cancel_check=None,
    ) -> dict[str, int]:
        """Return indexed 64-bit hash values for batch duplicate grouping.

        The caller receives primitive integers so it can form cache-friendly
        candidate buckets without repeatedly opening a SQLite connection for
        every photo pair.
        """
        if hash_backend not in self.SUPPORTED_HASHES:
            raise ValueError(f"Unsupported hash backend: {hash_backend}")
        normalized = [str(Path(path)) for path in image_paths if str(path or "")]
        self.ensure_index(normalized, cancel_check=cancel_check)
        if not normalized:
            return {}
        column = f"{hash_backend}_u64"
        values: dict[str, int] = {}
        with self._connect() as connection:
            for start in range(0, len(normalized), 900):
                raise_if_cancelled(cancel_check)
                chunk = normalized[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT image_path, {column} FROM phash_index WHERE image_path IN ({placeholders})",
                    chunk,
                ).fetchall()
                for image_path, blob in rows:
                    if blob is None:
                        continue
                    try:
                        values[str(image_path)] = int.from_bytes(blob, byteorder="big", signed=False)
                    except (TypeError, ValueError):
                        continue
        return values

    @staticmethod
    def compute_hash_hexes(image_path: str) -> dict[str, str]:
        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
        except Exception:
            return {"phash": "", "dhash": "", "whash": ""}
        return {
            "phash": str(imagehash.phash(rgb, hash_size=8)),
            "dhash": str(imagehash.dhash(rgb, hash_size=8)),
            "whash": str(imagehash.whash(rgb, hash_size=8)),
        }

    @staticmethod
    def compute_hash_hex(image_path: str, hash_backend: str = "phash") -> str:
        return PerceptualHashIndexService.compute_hash_hexes(image_path)[hash_backend]
