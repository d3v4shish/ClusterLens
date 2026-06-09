from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from infra.settings import get_settings

try:
    import faiss
except Exception:
    faiss = None


class EmbeddingIndexService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_dir = self.settings.cache_dir / "embedding_indexes"
        self.index_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_snapshot_key(image_paths: list[str]) -> str:
        payload = []
        for image_path in sorted(image_paths):
            path = Path(image_path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            payload.append(f"{path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}")
        return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()

    @staticmethod
    def build_snapshot_key_from_fingerprints(fingerprints: tuple[tuple[str, int, int], ...] | list[tuple[str, int, int]]) -> str:
        payload = "\n".join(
            f"{path}|{mtime_ns}|{size}"
            for path, mtime_ns, size in sorted(
                ((str(path), int(mtime_ns), int(size)) for path, mtime_ns, size in fingerprints),
                key=lambda item: item[0].lower(),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def save_index(self, snapshot_key: str, model_name: str, ordered_embeddings: list[tuple[str, np.ndarray]]) -> dict[str, str]:
        prefix = self.index_dir / f"{snapshot_key}_{model_name}"
        vector_path = prefix.with_suffix(".npy")
        path_map_path = prefix.with_suffix(".json")
        matrix = np.asarray([vector for _, vector in ordered_embeddings], dtype=np.float32)
        np.save(vector_path, matrix)
        path_map_path.write_text(
            json.dumps([path for path, _ in ordered_embeddings], indent=2),
            encoding="utf-8",
        )
        index_path = ""
        if faiss is not None and len(matrix) > 0:
            index = faiss.IndexHNSWFlat(matrix.shape[1], 32)
            index.hnsw.efConstruction = 64
            index.add(matrix)
            index_path = str(prefix.with_suffix(".faiss"))
            faiss.write_index(index, index_path)
        return {
            "snapshot_key": snapshot_key,
            "vector_path": str(vector_path),
            "path_map_path": str(path_map_path),
            "faiss_index_path": index_path,
        }

    def ensure_index(self, snapshot_key: str, model_name: str, ordered_embeddings: list[tuple[str, np.ndarray]]) -> tuple[dict[str, str], bool]:
        prefix = self.index_dir / f"{snapshot_key}_{model_name}"
        vector_path = prefix.with_suffix(".npy")
        path_map_path = prefix.with_suffix(".json")
        faiss_index_path = prefix.with_suffix(".faiss")
        expects_faiss = faiss is not None and len(ordered_embeddings) > 0
        if vector_path.exists() and path_map_path.exists() and (not expects_faiss or faiss_index_path.exists()):
            return (
                {
                    "snapshot_key": snapshot_key,
                    "vector_path": str(vector_path),
                    "path_map_path": str(path_map_path),
                    "faiss_index_path": str(faiss_index_path) if expects_faiss else "",
                },
                True,
            )
        return self.save_index(snapshot_key, model_name, ordered_embeddings), False
