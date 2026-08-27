from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image

from infra.cancel import raise_if_cancelled
from infra.atomic_io import atomic_write_text
from infra.settings import get_settings
from ml.embeddings import EmbeddingService

from .discovery import ImageDiscoveryService
from .perceptual_hash import PerceptualHashIndexService

try:
    import cv2
except Exception:
    cv2 = None

try:
    import faiss
except Exception:
    faiss = None


@dataclass(frozen=True)
class SimilaritySearchRequest:
    query_image_path: str = ""
    query_text: str = ""
    search_mode: str = "image"
    top_k: int = 30
    min_score: float = 0.0
    max_phash_distance: int = 12
    embedding_model: str = "clip"
    filters: dict[str, str] = field(default_factory=dict)
    use_onnx: bool = False
    hash_backend: str = "phash"
    hash_distance: int = 12
    orb_rerank: bool = False
    ann_backend: str = "flat"
    nlist: int = 64
    nprobe: int = 8
    pq_m: int = 8
    candidate_paths: list[str] | None = None


@dataclass(frozen=True)
class SearchResult:
    image_path: str
    score: float
    phash_distance: int
    model_name: str
    match_reason: str
    metadata: dict[str, str]
    hash_distance: int = -1
    hash_backend: str = "phash"
    orb_score: float = 0.0


@dataclass(frozen=True)
class DuplicateReviewGroup:
    query_image_path: str
    candidate_path: str
    score: float
    hash_distance: int
    hash_backend: str
    match_reason: str
    exact: bool


@dataclass
class IndexedImageRecord:
    image_path: str
    embedding: np.ndarray
    embedding_dim: int
    phash_hex: str
    folder: str
    file_ext: str
    width: int
    height: int
    camera: str
    mtime_ns: int
    file_size: int


class GlobalImageIndexService:
    def __init__(self, *, db_path: str | Path | None = None, reset_db: bool = False) -> None:
        self.settings = get_settings()
        self.db_path = Path(db_path) if db_path is not None else (self.settings.cache_dir / "global_image_search.db")
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
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS image_search_index (
                    image_path TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    embedding_dim INTEGER NOT NULL DEFAULT 0,
                    phash_hex TEXT NOT NULL,
                    folder TEXT NOT NULL,
                    file_ext TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    camera TEXT NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    PRIMARY KEY(image_path, model_name)
                )
                """
            )
            cols = {row[1] for row in connection.execute("PRAGMA table_info(image_search_index)").fetchall()}
            if "embedding_dim" not in cols:
                connection.execute("ALTER TABLE image_search_index ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_search_model ON image_search_index(model_name)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_image_search_folder ON image_search_index(folder)"
            )

    def upsert_embeddings(self, ordered_embeddings: list[tuple[str, np.ndarray]], model_name: str) -> int:
        rows = []
        for image_path, embedding in ordered_embeddings:
            path = Path(image_path)
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            metadata = self._read_metadata(path)
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
            rows.append(
                (
                    str(path),
                    model_name,
                    vector.tobytes(),
                    int(vector.shape[0]),
                    PerceptualHashIndexService.compute_hash_hex(str(path), "phash"),
                    str(path.parent),
                    path.suffix.lower(),
                    metadata["width"],
                    metadata["height"],
                    metadata["camera"],
                    stat.st_mtime_ns,
                    stat.st_size,
                )
            )
        if not rows:
            return 0
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO image_search_index(
                    image_path, model_name, embedding, embedding_dim, phash_hex, folder,
                    file_ext, width, height, camera, mtime_ns, file_size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_path, model_name) DO UPDATE SET
                    embedding=excluded.embedding,
                    embedding_dim=excluded.embedding_dim,
                    phash_hex=excluded.phash_hex,
                    folder=excluded.folder,
                    file_ext=excluded.file_ext,
                    width=excluded.width,
                    height=excluded.height,
                    camera=excluded.camera,
                    mtime_ns=excluded.mtime_ns,
                    file_size=excluded.file_size
                """,
                rows,
            )
        return len(rows)

    def load_stats(self, model_name: str, folder_prefix: str | None = None) -> dict[str, tuple[int, int]]:
        query = "SELECT image_path, mtime_ns, file_size FROM image_search_index WHERE model_name=?"
        args: list[object] = [model_name]
        if folder_prefix:
            query += " AND folder LIKE ?"
            args.append(f"{folder_prefix}%")
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return {row[0]: (int(row[1]), int(row[2])) for row in rows}

    def delete_paths(self, model_name: str, image_paths: list[str]) -> int:
        if not image_paths:
            return 0
        removed = 0
        with self._connect() as connection:
            for start in range(0, len(image_paths), 900):
                chunk = image_paths[start : start + 900]
                placeholders = ",".join("?" for _ in chunk)
                removed += connection.execute(
                    f"DELETE FROM image_search_index WHERE model_name=? AND image_path IN ({placeholders})",
                    [model_name, *chunk],
                ).rowcount
        return int(removed)

    def load_records(self, model_name: str, filters: dict[str, str] | None = None) -> list[IndexedImageRecord]:
        filters = filters or {}
        query = """
            SELECT image_path, embedding, embedding_dim, phash_hex, folder, file_ext, width, height, camera, mtime_ns, file_size
            FROM image_search_index
            WHERE model_name=?
        """
        args: list[object] = [model_name]
        if filters.get("folder"):
            query += " AND folder LIKE ?"
            args.append(f"{filters['folder']}%")
        if filters.get("file_ext"):
            query += " AND file_ext=?"
            args.append(filters["file_ext"].lower())
        if filters.get("camera"):
            query += " AND camera LIKE ?"
            args.append(f"%{filters['camera']}%")

        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return [
            IndexedImageRecord(
                image_path=row[0],
                embedding=np.frombuffer(row[1], dtype=np.float32).copy(),
                embedding_dim=int(row[2]) if int(row[2] or 0) > 0 else int(len(row[1]) // 4),
                phash_hex=row[3],
                folder=row[4],
                file_ext=row[5],
                width=int(row[6]),
                height=int(row[7]),
                camera=row[8],
                mtime_ns=int(row[9]),
                file_size=int(row[10]),
            )
            for row in rows
        ]

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, object]:
        width = 0
        height = 0
        camera = ""
        try:
            with Image.open(path) as image:
                width, height = image.size
                exif = image.getexif()
                if exif:
                    tag_lookup = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
                    camera = str(tag_lookup.get("Model", ""))
        except Exception:
            pass
        return {"width": width, "height": height, "camera": camera}


class SimilaritySearchService:
    def __init__(
        self,
        discovery_service: ImageDiscoveryService | None = None,
        embedding_service: EmbeddingService | None = None,
        phash_service: PerceptualHashIndexService | None = None,
        index_service: GlobalImageIndexService | None = None,
    ) -> None:
        self.discovery_service = discovery_service or ImageDiscoveryService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.phash_service = phash_service or PerceptualHashIndexService()
        self.index_service = index_service or GlobalImageIndexService()

    def index_directory(
        self,
        directory: str,
        embedding_model: str,
        recursive: bool = True,
        use_onnx: bool = False,
        *,
        prune_missing: bool = True,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, int]:
        if progress_callback:
            progress_callback(-1, 'Scanning images...')
        image_paths = self.discovery_service.discover(directory, recursive=recursive)
        return self.index_paths(
            image_paths,
            embedding_model=embedding_model,
            directory_hint=directory,
            use_onnx=use_onnx,
            prune_missing=prune_missing,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def index_paths(
        self,
        image_paths: list[str],
        *,
        embedding_model: str,
        directory_hint: str = "",
        use_onnx: bool = False,
        prune_missing: bool = False,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, int]:
        if not image_paths:
            return {'indexed_images': 0, 'updated_images': 0, 'removed_images': 0, 'unchanged_images': 0}
        raise_if_cancelled(cancel_check)

        if progress_callback:
            progress_callback(-1, 'Indexing hashes...')
        hash_metrics: dict[str, int] = {}
        try:
            hash_metrics = self.phash_service.sync_directory(
                directory_hint or str(Path(image_paths[0]).parent),
                image_paths,
                prune_missing=prune_missing,
                cancel_check=cancel_check,
            )
        except Exception:
            # Fallback to full rebuild if the hash service doesn't support incremental sync yet.
            self.phash_service.build_index(image_paths)
            hash_metrics = {'updated': len(image_paths), 'removed': 0}
        raise_if_cancelled(cancel_check)

        folder_prefix = directory_hint or None
        existing = self.index_service.load_stats(embedding_model, folder_prefix=folder_prefix)
        current_stats: dict[str, tuple[int, int]] = {}
        for image_path in image_paths:
            try:
                stat = Path(image_path).stat()
            except FileNotFoundError:
                continue
            current_stats[str(Path(image_path))] = (int(stat.st_mtime_ns), int(stat.st_size))
        to_update = [p for p, st in current_stats.items() if existing.get(p) != st]
        removed = [p for p in existing.keys() if p not in current_stats] if prune_missing else []
        unchanged = max(0, len(existing) - len(to_update) - len(removed))

        def _embed_progress(done: int, total: int, stage: str) -> None:
            if progress_callback:
                pct = 10 + int((done / max(1, total)) * 85)
                progress_callback(pct, stage)
            raise_if_cancelled(cancel_check)

        indexed_count = 0
        embed_metrics: dict[str, object] = {}
        if to_update:
            if progress_callback:
                progress_callback(-1, f'Embedding {len(to_update)} images ({embedding_model})...')
            ordered_embeddings, embed_metrics = self.embedding_service.embed_paths(to_update, embedding_model, use_onnx=use_onnx, progress_callback=_embed_progress)
            indexed_count = self.index_service.upsert_embeddings(ordered_embeddings, embedding_model)

        if removed:
            if progress_callback:
                progress_callback(-1, f'Removing {len(removed)} missing files from index...')
            self.index_service.delete_paths(embedding_model, removed)

        if progress_callback:
            progress_callback(100, 'Index complete')
        return {
            'indexed_images': int(indexed_count),
            'updated_images': int(len(to_update)),
            'removed_images': int(len(removed)),
            'unchanged_images': int(unchanged),
            'hash_updated': int(hash_metrics.get('updated', 0)),
            'hash_removed': int(hash_metrics.get('removed', 0)),
            'effective_mode': str(embed_metrics.get('effective_mode', 'cpu')),
            'onnx_provider': str(embed_metrics.get('onnx_provider', 'CPUExecutionProvider')),
            'runtime_reason': str(embed_metrics.get('runtime_reason', '')),
        }

    def search(self, request: SimilaritySearchRequest, *, cancel_check=None) -> list[SearchResult]:
        records = self.index_service.load_records(request.embedding_model, filters=request.filters)
        if request.candidate_paths:
            allowed_paths = {str(Path(path)) for path in request.candidate_paths if path}
            records = [record for record in records if record.image_path in allowed_paths]
        if not records:
            return []
        hash_candidates: dict[str, int] = {}
        used_hash_prefilter = False
        if request.search_mode == "duplicate":
            if not request.query_image_path:
                raise ValueError("Duplicate search requires query_image_path.")
            candidate_paths = {record.image_path for record in records}
            # Ensure the query hash is present; the folder index job should keep candidates up to date.
            self.phash_service.ensure_index([request.query_image_path], cancel_check=cancel_check)
            matches = self.phash_service.query_similar(
                request.query_image_path,
                max_distance=request.hash_distance,
                candidate_paths=candidate_paths,
                hash_backend=request.hash_backend,
                cancel_check=cancel_check,
            )
            hash_candidates = {match.image_path: match.hash_distance for match in matches}
            if hash_candidates:
                used_hash_prefilter = True
                records = [record for record in records if record.image_path in hash_candidates]
                if not records:
                    return []

        query_embedding = self._build_query_embedding(request)
        query_dim = int(np.asarray(query_embedding, dtype=np.float32).reshape(-1).shape[0])
        kept = [record for record in records if record.embedding.shape[0] == query_dim]
        if not kept:
            dims = sorted({int(getattr(r, "embedding_dim", 0) or r.embedding.shape[0]) for r in records})
            raise ValueError(
                f"Indexed embeddings have incompatible dimensions for model '{request.embedding_model}'. "
                f"Query dim={query_dim}, index dims={dims}. Re-index the folder with the same model."
            )
        records = kept
        records = self._ann_filter_records(records, query_embedding, request)
        query_orb = self._orb_features(request.query_image_path) if request.orb_rerank and request.query_image_path else None

        scored_results = []
        for record in records:
            raise_if_cancelled(cancel_check)
            score = float(np.dot(query_embedding, record.embedding))
            orb_score = 0.0
            if query_orb is not None:
                orb_score = self._orb_match_score(query_orb, self._orb_features(record.image_path))
                score = 0.8 * score + 0.2 * orb_score
            if score < request.min_score:
                continue
            hash_distance = hash_candidates.get(record.image_path, -1)
            scored_results.append(
                SearchResult(
                    image_path=record.image_path,
                    score=round(score, 6),
                    phash_distance=hash_distance,
                    model_name=request.embedding_model,
                    match_reason=self._match_reason("duplicate" if used_hash_prefilter else "image", request.orb_rerank, request.ann_backend),
                    metadata={
                        "folder": record.folder,
                        "file_ext": record.file_ext,
                        "size": f"{record.width}x{record.height}",
                        "camera": record.camera,
                    },
                    hash_distance=hash_distance,
                    hash_backend=request.hash_backend,
                    orb_score=round(orb_score, 6),
                )
            )
        scored_results.sort(
            key=lambda item: (
                -item.score,
                item.hash_distance if item.hash_distance >= 0 else 9999,
                -item.orb_score,
                item.image_path,
            )
        )
        return scored_results[: max(1, request.top_k)]

    @staticmethod
    def build_duplicate_review_groups(
        query_image_path: str,
        results: list[SearchResult],
        *,
        exact_hash_distance: int = 0,
    ) -> list[DuplicateReviewGroup]:
        query = str(query_image_path or "").strip()
        groups: list[DuplicateReviewGroup] = []
        for result in list(results or []):
            candidate = str(result.image_path or "").strip()
            if not candidate or (query and candidate == query):
                continue
            hash_distance = int(getattr(result, "hash_distance", -1))
            if hash_distance < 0:
                hash_distance = int(getattr(result, "phash_distance", -1))
            groups.append(
                DuplicateReviewGroup(
                    query_image_path=query,
                    candidate_path=candidate,
                    score=float(result.score),
                    hash_distance=hash_distance,
                    hash_backend=str(getattr(result, "hash_backend", "phash") or "phash"),
                    match_reason=str(result.match_reason or ""),
                    exact=hash_distance >= 0 and hash_distance <= int(exact_hash_distance),
                )
            )
        groups.sort(
            key=lambda group: (
                0 if group.exact else 1,
                group.hash_distance if group.hash_distance >= 0 else 9999,
                -group.score,
                group.candidate_path,
            )
        )
        return groups

    @staticmethod
    def export_duplicate_decisions(decisions: list[dict[str, object]], output_path: str | Path) -> dict[str, object]:
        clean_decisions: list[dict[str, object]] = []
        for item in list(decisions or []):
            if not isinstance(item, dict):
                continue
            keeper = str(item.get("keeper_path", item.get("keeper", "")) or "").strip()
            duplicate = str(item.get("duplicate_path", item.get("duplicate", "")) or "").strip()
            if not keeper or not duplicate:
                continue
            clean_decisions.append(
                {
                    "keeper_path": keeper,
                    "duplicate_path": duplicate,
                    "action": str(item.get("action", "review") or "review"),
                    "tag": str(item.get("tag", "") or ""),
                    "score": float(item.get("score", 0.0) or 0.0),
                    "hash_distance": int(item.get("hash_distance", -1) or -1),
                    "source_files_changed": False,
                }
            )
        payload = {"version": 1, "decisions": clean_decisions}
        atomic_write_text(Path(output_path), json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def _build_query_embedding(self, request: SimilaritySearchRequest) -> np.ndarray:
        if request.search_mode == "text":
            if not request.query_text.strip():
                raise ValueError("Text search requires query_text.")
            return self.embedding_service.embed_text(
                request.query_text,
                model_name=request.embedding_model,
                use_onnx=request.use_onnx,
            )
        if not request.query_image_path:
            raise ValueError("Image/Duplicate search requires query_image_path.")
        ordered_embeddings, _ = self.embedding_service.embed_paths(
            [request.query_image_path],
            request.embedding_model,
            use_onnx=request.use_onnx,
        )
        if not ordered_embeddings:
            raise ValueError("Could not embed query image.")
        return ordered_embeddings[0][1]

    def _ann_filter_records(
        self,
        records: list[IndexedImageRecord],
        query_embedding: np.ndarray,
        request: SimilaritySearchRequest,
    ) -> list[IndexedImageRecord]:
        if request.ann_backend == "flat" or faiss is None or len(records) <= max(request.top_k, 8):
            return records
        matrix = np.asarray([record.embedding for record in records], dtype=np.float32)
        dim = matrix.shape[1]
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if request.ann_backend == "hnsw":
            index = faiss.IndexHNSWFlat(dim, 32)
            index.hnsw.efConstruction = 64
            index.add(matrix)
        elif request.ann_backend == "ivf_pq":
            nlist = max(1, min(request.nlist, len(records)))
            pq_m = max(1, min(request.pq_m, dim))
            while dim % pq_m != 0 and pq_m > 1:
                pq_m -= 1
            quantizer = faiss.IndexFlatIP(dim)
            if len(records) < max(32, nlist * 4):
                index = faiss.IndexFlatIP(dim)
            else:
                index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8)
                index.train(matrix)
                index.nprobe = max(1, min(request.nprobe, nlist))
            index.add(matrix)
        else:
            return records
        _, indices = index.search(query, min(len(records), max(request.top_k * 4, request.top_k)))
        return [records[int(index)] for index in indices[0] if int(index) >= 0]

    @staticmethod
    def _orb_features(image_path: str):
        if cv2 is None:
            return None
        gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        orb = cv2.ORB_create(nfeatures=600)
        _, descriptors = orb.detectAndCompute(gray, None)
        return descriptors

    @staticmethod
    def _orb_match_score(query_descriptors, candidate_descriptors) -> float:
        if cv2 is None or query_descriptors is None or candidate_descriptors is None:
            return 0.0
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(query_descriptors, candidate_descriptors)
        if not matches:
            return 0.0
        avg_distance = sum(match.distance for match in matches) / len(matches)
        match_ratio = len(matches) / max(len(query_descriptors), len(candidate_descriptors), 1)
        distance_score = max(0.0, 1.0 - (avg_distance / 96.0))
        return float(max(0.0, min(1.0, 0.6 * match_ratio + 0.4 * distance_score)))

    @staticmethod
    def _match_reason(search_mode: str, orb_rerank: bool, ann_backend: str) -> str:
        if search_mode == "text":
            base = "text embedding cosine similarity"
        elif search_mode == "duplicate":
            base = "hash prefilter + embedding rerank"
        else:
            base = "image embedding cosine similarity"
        if orb_rerank:
            base += " + ORB rerank"
        if ann_backend != "flat":
            base += f" + FAISS {ann_backend}"
        return base
