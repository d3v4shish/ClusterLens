from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from infra.atomic_io import atomic_write_text, atomic_write_with
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

    def save_index(
        self,
        snapshot_key: str,
        model_name: str,
        ordered_embeddings: list[tuple[str, np.ndarray]],
        *,
        embedding_signature: str = "",
    ) -> dict[str, str]:
        prefix = self._prefix(snapshot_key, model_name, embedding_signature)
        vector_path = prefix.with_suffix(".npy")
        path_map_path = prefix.with_suffix(".json")
        matrix = _embedding_matrix(ordered_embeddings)
        atomic_write_with(vector_path, lambda temporary: _save_numpy(temporary, matrix))
        atomic_write_text(path_map_path, json.dumps([path for path, _ in ordered_embeddings], indent=2))
        index_path = ""
        if faiss is not None and len(matrix) > 0:
            index = faiss.IndexHNSWFlat(matrix.shape[1], 32)
            index.hnsw.efConstruction = 64
            index.add(matrix)
            faiss_path = prefix.with_suffix(".faiss")
            atomic_write_with(faiss_path, lambda temporary: faiss.write_index(index, str(temporary)))
            index_path = str(faiss_path)
        return {
            "snapshot_key": snapshot_key,
            "vector_path": str(vector_path),
            "path_map_path": str(path_map_path),
            "faiss_index_path": index_path,
        }

    def ensure_index(
        self,
        snapshot_key: str,
        model_name: str,
        ordered_embeddings: list[tuple[str, np.ndarray]],
        *,
        embedding_signature: str = "",
    ) -> tuple[dict[str, str], bool]:
        prefix = self._prefix(snapshot_key, model_name, embedding_signature)
        vector_path = prefix.with_suffix(".npy")
        path_map_path = prefix.with_suffix(".json")
        faiss_index_path = prefix.with_suffix(".faiss")
        expects_faiss = faiss is not None and len(ordered_embeddings) > 0
        expected_dimension = _embedding_dimension(ordered_embeddings)
        if self._index_is_valid(
            vector_path,
            path_map_path,
            faiss_index_path if expects_faiss else None,
            expected_paths=[str(path) for path, _vector in ordered_embeddings],
            expected_dimension=expected_dimension,
        ):
            return (
                {
                    "snapshot_key": snapshot_key,
                    "vector_path": str(vector_path),
                    "path_map_path": str(path_map_path),
                    "faiss_index_path": str(faiss_index_path) if expects_faiss else "",
                },
                True,
            )
        return self.save_index(
            snapshot_key,
            model_name,
            ordered_embeddings,
            embedding_signature=embedding_signature,
        ), False

    def _prefix(self, snapshot_key: str, model_name: str, embedding_signature: str) -> Path:
        revision_suffix = ""
        if embedding_signature:
            revision_suffix = "_" + hashlib.sha256(str(embedding_signature).encode("utf-8")).hexdigest()[:16]
        return self.index_dir / f"{snapshot_key}_{model_name}{revision_suffix}"

    @staticmethod
    def _index_is_valid(
        vector_path: Path,
        path_map_path: Path,
        faiss_index_path: Path | None,
        *,
        expected_paths: list[str],
        expected_dimension: int,
    ) -> bool:
        try:
            matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
            paths = json.loads(path_map_path.read_text(encoding="utf-8"))
            expected_rows = len(expected_paths)
            if (
                not isinstance(paths, list)
                or [str(path) for path in paths] != list(expected_paths)
                or matrix.ndim != 2
                or int(matrix.shape[0]) != expected_rows
                or int(matrix.shape[1]) != int(expected_dimension)
            ):
                return False
            if faiss_index_path is not None:
                if faiss is None or not faiss_index_path.exists():
                    return False
                index = faiss.read_index(str(faiss_index_path))
                if int(index.ntotal) != int(expected_rows) or int(index.d) != int(expected_dimension):
                    return False
            return True
        except (OSError, ValueError, TypeError, IndexError, RuntimeError, json.JSONDecodeError):
            return False


def _save_numpy(path: Path, matrix: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)


def _embedding_dimension(ordered_embeddings: list[tuple[str, np.ndarray]]) -> int:
    if not ordered_embeddings:
        return 0
    first = np.asarray(ordered_embeddings[0][1], dtype=np.float32).reshape(-1)
    if int(first.size) <= 0:
        raise ValueError("Embedding vectors must not be empty.")
    dimension = int(first.size)
    for _path, vector in ordered_embeddings[1:]:
        if int(np.asarray(vector).size) != dimension:
            raise ValueError("All embedding vectors must have the same dimension.")
    return dimension


def _embedding_matrix(ordered_embeddings: list[tuple[str, np.ndarray]]) -> np.ndarray:
    dimension = _embedding_dimension(ordered_embeddings)
    if not ordered_embeddings:
        return np.empty((0, dimension), dtype=np.float32)
    return np.vstack(
        [np.asarray(vector, dtype=np.float32).reshape(1, dimension) for _path, vector in ordered_embeddings]
    )
