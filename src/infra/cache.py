from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .settings import get_settings


LOGGER = logging.getLogger(__name__)


class CacheService:
    def __init__(self, memory_cache_size: int | None = None) -> None:
        self.settings = get_settings()
        self.memory_cache_size = max(1, int(memory_cache_size or self.settings.embedding_memory_cache_size))
        self._memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._initialize_db()

    def _connect(self) -> sqlite3.Connection:
        db_path = Path(self.settings.embedding_cache_db)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            return conn
        except sqlite3.DatabaseError:
            conn.close()
            self._reset_corrupt_db(db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA temp_store=MEMORY")
            return conn

    def _initialize_db(self) -> None:
        self.settings.cache_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    image_path TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    vector_dim INTEGER NOT NULL,
                    dtype TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _reset_corrupt_db(self, db_path: Path) -> None:
        LOGGER.warning("Embedding cache database is invalid; rebuilding %s", db_path)
        for suffix in ("", "-wal", "-shm"):
            target = db_path if not suffix else db_path.with_name(db_path.name + suffix)
            try:
                target.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Failed to remove invalid cache file %s: %s", target, exc)

    @staticmethod
    def build_embedding_key(image_path: str, model_name: str, signature: str) -> str:
        path = Path(image_path)
        stat = path.stat()
        payload = f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{model_name}|{signature}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def build_embedding_key_from_fingerprint(
        image_path: str,
        mtime_ns: int,
        file_size: int,
        model_name: str,
        signature: str,
    ) -> str:
        payload = f"{os.path.abspath(str(image_path))}|{int(mtime_ns)}|{int(file_size)}|{model_name}|{signature}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def build_embedding_keys(
        self,
        image_paths: list[str],
        model_name: str,
        signature: str,
        *,
        path_fingerprints: dict[str, tuple[int, int]] | None = None,
    ) -> list[str]:
        path_fingerprints = path_fingerprints or {}
        keys: list[str] = []
        for image_path in image_paths:
            fingerprint = path_fingerprints.get(str(image_path))
            if fingerprint is None:
                keys.append(self.build_embedding_key(image_path, model_name, signature))
            else:
                mtime_ns, file_size = fingerprint
                keys.append(
                    self.build_embedding_key_from_fingerprint(
                        image_path,
                        mtime_ns,
                        file_size,
                        model_name,
                        signature,
                    )
                )
        return keys

    def get_embedding(self, cache_key: str) -> np.ndarray | None:
        cached = self.get_embeddings_many([cache_key])
        return cached.get(cache_key)

    def get_embeddings_many(self, cache_keys: list[str]) -> dict[str, np.ndarray]:
        if not cache_keys:
            return {}
        found: dict[str, np.ndarray] = {}
        missing: list[str] = []
        for cache_key in cache_keys:
            cached = self._memory_cache.get(cache_key)
            if cached is not None:
                self._memory_cache.move_to_end(cache_key)
                found[cache_key] = cached
            else:
                missing.append(cache_key)
        if not missing:
            return found
        with self._connect() as conn:
            for start in range(0, len(missing), 900):
                chunk = missing[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT cache_key, vector, vector_dim, dtype FROM embeddings WHERE cache_key IN ({placeholders})",
                    chunk,
                ).fetchall()
                for cache_key, vector_blob, vector_dim, dtype in rows:
                    vector = np.frombuffer(vector_blob, dtype=np.dtype(dtype)).copy().reshape(vector_dim)
                    found[cache_key] = vector
                    self._remember(cache_key, vector)
        return found

    def put_embedding(self, cache_key: str, image_path: str, model_name: str, vector: np.ndarray) -> None:
        self.put_embeddings_many([(cache_key, image_path, model_name, vector)])

    def put_embeddings_many(self, rows: list[tuple[str, str, str, np.ndarray]]) -> None:
        if not rows:
            return
        payload = []
        for cache_key, image_path, model_name, vector in rows:
            normalized = np.asarray(vector, dtype=np.float32).reshape(-1)
            payload.append((cache_key, image_path, model_name, normalized.tobytes(), normalized.shape[0], str(normalized.dtype)))
            self._remember(cache_key, normalized)
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embeddings(cache_key, image_path, model_name, vector, vector_dim, dtype)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
            conn.commit()

    def clear_memory_cache(self) -> None:
        self._memory_cache.clear()

    def clear_disk_cache(self) -> None:
        db_path = Path(self.settings.embedding_cache_db)
        for suffix in ("", "-wal", "-shm"):
            try:
                db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)
            except OSError:
                continue
        self._initialize_db()

    def _remember(self, cache_key: str, vector: np.ndarray) -> None:
        self._memory_cache[cache_key] = vector
        self._memory_cache.move_to_end(cache_key)
        while len(self._memory_cache) > self.memory_cache_size:
            self._memory_cache.popitem(last=False)
