from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

from infra.runtime import ExecutionPolicy, RuntimeCapabilityService
from infra.settings import get_settings
from ml.clustering import ClusteringService

from .discovery import ImageDiscoveryService


@dataclass(frozen=True)
class DetectedFace:
    image_path: str
    bbox: tuple[int, int, int, int]
    confidence: float
    crop: Image.Image


@dataclass(frozen=True)
class FaceSearchRequest:
    query_face_image: str
    query_face_bbox: tuple[int, int, int, int] | None = None
    top_k: int = 30
    min_face_score: float = 0.35
    cluster_people: bool = False
    candidate_paths: list[str] | None = None


@dataclass(frozen=True)
class FaceSearchResult:
    image_path: str
    score: float
    phash_distance: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    model_name: str
    match_reason: str


@dataclass(frozen=True)
class FaceIndexRecord:
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    embedding: np.ndarray


@dataclass(frozen=True)
class FaceLabelRequest:
    person_name: str
    example_image_paths: list[str]
    similarity_threshold: float = 0.72


@dataclass(frozen=True)
class PersonPrototype:
    person_name: str
    embedding: np.ndarray
    similarity_threshold: float
    example_count: int


@dataclass(frozen=True)
class PersonProfile:
    person_name: str
    similarity_threshold: float
    example_count: int
    labeled_count: int
    visible_face_count: int
    notes: str = ""
    tags: tuple[str, ...] = ()
    cover_image_path: str = ""
    cover_face_index: int = 0


@dataclass(frozen=True)
class FaceLabelAssignment:
    person_name: str
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class IndexedFaceRecord:
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    embedding: np.ndarray
    person_name: str = ""
    label_confidence: float = 0.0

class FaceDetectionService:
    def __init__(
        self,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        settings = get_settings()
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self.device = torch.device(self.execution_policy.torch_device if self.execution_policy.uses_cuda else "cpu")
        self._detector: MTCNN | None = None

    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        detector = self._get_detector()
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
        except Exception:
            return []

        boxes, probs = detector.detect(rgb_image)
        if boxes is None or probs is None:
            return []
        faces = []
        for box, prob in zip(boxes, probs):
            if prob is None:
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rgb_image.width, x2)
            y2 = min(rgb_image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    crop=rgb_image.crop((x1, y1, x2, y2)).copy(),
                )
            )
        return faces

    def detect_query_face(self, image_path: str, query_bbox: tuple[int, int, int, int] | None = None) -> DetectedFace | None:
        if query_bbox is not None:
            try:
                with Image.open(image_path) as image:
                    rgb_image = image.convert("RGB")
            except Exception:
                return None
            x1, y1, x2, y2 = query_bbox
            return DetectedFace(
                    image_path=image_path,
                    bbox=query_bbox,
                    confidence=1.0,
                    crop=rgb_image.crop((x1, y1, x2, y2)).copy(),
                )
        faces = self.detect_faces(image_path)
        if not faces:
            return None
        return max(faces, key=lambda face: (face.confidence, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))

    def _get_detector(self) -> MTCNN:
        if self._detector is None:
            self._detector = MTCNN(keep_all=True, device=self.device, post_process=True)
        return self._detector


class FaceEmbeddingService:
    def __init__(
        self,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        settings = get_settings()
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self.device = torch.device(self.execution_policy.torch_device if self.execution_policy.uses_cuda else "cpu")
        self._model: InceptionResnetV1 | None = None
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def embed_faces(self, face_crops: list[Image.Image]) -> np.ndarray:
        if not face_crops:
            return np.zeros((0, 512), dtype=np.float32)
        model = self._get_model()
        batch = torch.stack([self.preprocess(face_crop.convert("RGB")) for face_crop in face_crops]).to(self.device)
        with torch.inference_mode():
            embeddings = model(batch)
            embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
        return embeddings.detach().cpu().numpy().astype(np.float32)

    def _get_model(self) -> InceptionResnetV1:
        if self._model is None:
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        return self._model


class FaceIndexService:
    def __init__(
        self,
        discovery_service: ImageDiscoveryService | None = None,
        detection_service: FaceDetectionService | None = None,
        embedding_service: FaceEmbeddingService | None = None,
        clustering_service: ClusteringService | None = None,
        *,
        db_path: str | Path | None = None,
        reset_db: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.db_path = db_path if db_path is not None else (self.settings.cache_dir / "face_search_index.db")
        self.discovery_service = discovery_service or ImageDiscoveryService()
        self.detection_service = detection_service or FaceDetectionService()
        self.embedding_service = embedding_service or FaceEmbeddingService()
        self.clustering_service = clustering_service or ClusteringService()
        if reset_db:
            try:
                if isinstance(self.db_path, Path) and self.db_path.exists():
                    self.db_path.unlink(missing_ok=True)
            except Exception:
                pass
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL;")
        return connection

    @staticmethod
    def _normalize_path_for_match(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        return os.path.normcase(os.path.normpath(text))

    @classmethod
    def _matches_folder_prefix(cls, image_path: str, folder_prefix: str) -> bool:
        prefix = cls._normalize_path_for_match(folder_prefix)
        if not prefix:
            return True
        candidate = cls._normalize_path_for_match(image_path)
        if candidate == prefix:
            return True
        return candidate.startswith(prefix + os.sep)

    @classmethod
    def _normalized_path_set(cls, candidate_paths: list[str] | None) -> set[str]:
        if not candidate_paths:
            return set()
        return {cls._normalize_path_for_match(path) for path in candidate_paths if str(path or "").strip()}

    @classmethod
    def _path_in_scope(cls, image_path: str, *, folder_prefix: str = "", candidate_paths: set[str] | None = None) -> bool:
        if folder_prefix and not cls._matches_folder_prefix(image_path, folder_prefix):
            return False
        if candidate_paths:
            return cls._normalize_path_for_match(image_path) in candidate_paths
        return True

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_index (
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    bbox_json TEXT NOT NULL,
                    face_confidence REAL NOT NULL,
                    embedding BLOB NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    PRIMARY KEY(image_path, face_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS person_prototypes (
                    person_name TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    similarity_threshold REAL NOT NULL,
                    example_count INTEGER NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS person_profiles (
                    person_name TEXT PRIMARY KEY,
                    notes TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    cover_image_path TEXT NOT NULL DEFAULT '',
                    cover_face_index INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_labels (
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    PRIMARY KEY(image_path, face_index)
                )
                """
            )

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, int]:
        image_paths = self.discovery_service.discover(directory, recursive=recursive)
        return self.index_paths(image_paths, progress_callback=progress_callback, cancel_check=cancel_check)

    def index_paths(
        self,
        image_paths: list[str],
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, int]:
        from infra.cancel import raise_if_cancelled

        total_faces = 0
        done = 0
        total = len(image_paths)
        for image_path in image_paths:
            raise_if_cancelled(cancel_check)
            faces = self.detection_service.detect_faces(image_path)
            if not faces:
                self.remove_image(image_path)
                done += 1
                if progress_callback and total:
                    progress_callback(int(done * 100 / max(1, total)), f"{done}/{total} images, extracted {total_faces} faces")
                continue
            embeddings = self.embedding_service.embed_faces([face.crop for face in faces])
            records = []
            try:
                stat = Path(image_path).stat()
            except FileNotFoundError:
                continue
            for face_index, face in enumerate(faces):
                records.append(
                    FaceIndexRecord(
                        image_path=image_path,
                        face_index=face_index,
                        face_bbox=face.bbox,
                        face_confidence=face.confidence,
                        embedding=embeddings[face_index],
                    )
                )
            self.save_face_records(records, stat.st_mtime_ns, stat.st_size)
            total_faces += len(records)
            done += 1
            if progress_callback and total:
                progress_callback(int(done * 100 / max(1, total)), f"{done}/{total} images, extracted {total_faces} faces")
        if progress_callback:
            progress_callback(100, f"{done}/{total} images, extracted {total_faces} faces")
        return {"images_done": int(done), "images_total": int(total), "faces_indexed": int(total_faces)}

    def save_face_records(self, records: list[FaceIndexRecord], mtime_ns: int, file_size: int) -> None:
        if not records:
            return
        image_path = records[0].image_path
        with self._connect() as connection:
            connection.execute("DELETE FROM face_index WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM face_labels WHERE image_path=?", (image_path,))
            connection.executemany(
                """
                INSERT INTO face_index(image_path, face_index, bbox_json, face_confidence, embedding, mtime_ns, file_size)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.image_path,
                        record.face_index,
                        json.dumps(list(record.face_bbox)),
                        record.face_confidence,
                        np.asarray(record.embedding, dtype=np.float32).tobytes(),
                        mtime_ns,
                        file_size,
                    )
                    for record in records
                ],
            )

    def remove_image(self, image_path: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM face_index WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM face_labels WHERE image_path=?", (image_path,))

    def load_all_records(self) -> list[FaceIndexRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path, face_index, bbox_json, face_confidence, embedding FROM face_index"
            ).fetchall()
        return [
            FaceIndexRecord(
                image_path=row[0],
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
            )
            for row in rows
        ]

    def load_indexed_faces(
        self,
        *,
        folder_prefix: str = "",
        limit: int = 400,
        candidate_paths: list[str] | None = None,
    ) -> list[IndexedFaceRecord]:
        folder_prefix = str(folder_prefix or "").strip()
        limit = max(1, int(limit))
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT
                i.image_path,
                i.face_index,
                i.bbox_json,
                i.face_confidence,
                i.embedding,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            ORDER BY
                CASE WHEN COALESCE(l.person_name, '') = '' THEN 1 ELSE 0 END,
                LOWER(COALESCE(l.person_name, '')),
                i.image_path,
                i.face_index
        """
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        records = [
            IndexedFaceRecord(
                image_path=str(row[0]),
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
                person_name=str(row[5] or ""),
                label_confidence=float(row[6] or 0.0),
            )
            for row in rows
        ]
        records = [
            record
            for record in records
            if self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths)
        ]
        return records[:limit]

    def load_face_record(self, image_path: str, face_index: int) -> IndexedFaceRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    i.image_path,
                    i.face_index,
                    i.bbox_json,
                    i.face_confidence,
                    i.embedding,
                    COALESCE(l.person_name, ''),
                    COALESCE(l.confidence, 0.0)
                FROM face_index i
                LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                WHERE i.image_path=? AND i.face_index=?
                """,
                (str(image_path), int(face_index)),
            ).fetchone()
        if row is None:
            return None
        return IndexedFaceRecord(
            image_path=str(row[0]),
            face_index=int(row[1]),
            face_bbox=tuple(json.loads(row[2])),
            face_confidence=float(row[3]),
            embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
            person_name=str(row[5] or ""),
            label_confidence=float(row[6] or 0.0),
        )

    def search_faces(self, request: FaceSearchRequest) -> list[FaceSearchResult]:
        query_face = self.detection_service.detect_query_face(request.query_face_image, request.query_face_bbox)
        if query_face is None:
            raise ValueError("No face found in query image.")
        query_embedding = self.embedding_service.embed_faces([query_face.crop])[0]
        return self._search_by_embedding(
            query_embedding,
            min_face_score=request.min_face_score,
            top_k=request.top_k,
            candidate_paths=request.candidate_paths,
        )

    def search_similar_face(
        self,
        image_path: str,
        face_index: int,
        *,
        top_k: int = 30,
        min_score: float = 0.35,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
    ) -> list[FaceSearchResult]:
        record = self.load_face_record(image_path, face_index)
        if record is None:
            raise ValueError("The selected indexed face was not found.")
        return self._search_by_embedding(
            record.embedding,
            min_face_score=min_score,
            top_k=top_k,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            exclude={(record.image_path, record.face_index)},
        )

    def _search_by_embedding(
        self,
        query_embedding: np.ndarray,
        *,
        min_face_score: float,
        top_k: int,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        exclude: set[tuple[str, int]] | None = None,
    ) -> list[FaceSearchResult]:
        results = []
        exclude = exclude or set()
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        for record in self.load_all_records():
            if not self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            if (record.image_path, int(record.face_index)) in exclude:
                continue
            score = float(np.dot(query_embedding, record.embedding))
            if score < min_face_score:
                continue
            results.append(
                FaceSearchResult(
                    image_path=record.image_path,
                    score=round(score, 6),
                    phash_distance=-1,
                    face_bbox=record.face_bbox,
                    face_confidence=round(record.face_confidence, 6),
                    model_name="facenet-vggface2",
                    match_reason="face embedding cosine similarity",
                )
            )
        results.sort(key=lambda item: (-item.score, item.image_path, item.face_bbox))
        return results[: max(1, top_k)]

    def cluster_faces(
        self,
        num_clusters: int = 12,
        min_face_score: float = 0.0,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
    ) -> dict[int, list[str]]:
        allowed_paths = self._normalized_path_set(candidate_paths)
        records = [
            record
            for record in self.load_all_records()
            if record.face_confidence >= min_face_score
            and self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths)
        ]
        if len(records) < 2:
            return {}
        num_clusters = min(max(2, num_clusters), len(records))
        embeddings = [record.embedding for record in records]
        clusters, _ = self.clustering_service.cluster(
            embeddings,
            num_clusters=num_clusters,
            backend="cosine-kmeans",
            similarity_mode="cosine",
            outlier_policy="assign",
        )
        return {
            cluster_id: [records[index].image_path for index in indices]
            for cluster_id, indices in clusters.items()
        }

    def label_face_examples(self, request: FaceLabelRequest) -> PersonPrototype:
        if not request.person_name.strip():
            raise ValueError("Person name is required.")
        example_faces = []
        for image_path in request.example_image_paths:
            face = self.detection_service.detect_query_face(image_path)
            if face is not None:
                example_faces.append(face.crop)
        if not example_faces:
            raise ValueError("No faces found in the selected example images.")
        embeddings = self.embedding_service.embed_faces(example_faces)
        return self._save_person_prototype(
            request.person_name,
            embeddings,
            similarity_threshold=request.similarity_threshold,
        )

    def label_indexed_faces(
        self,
        person_name: str,
        face_refs: list[tuple[str, int]],
        *,
        similarity_threshold: float = 0.72,
    ) -> PersonPrototype:
        person_name = str(person_name or "").strip()
        if not person_name:
            raise ValueError("Person name is required.")
        if not face_refs:
            raise ValueError("Select at least one indexed face.")
        records: list[IndexedFaceRecord] = []
        for image_path, face_index in face_refs:
            record = self.load_face_record(image_path, face_index)
            if record is not None:
                records.append(record)
        if not records:
            raise ValueError("The selected indexed faces are no longer available.")
        embeddings = np.asarray([record.embedding for record in records], dtype=np.float32)
        person = self._save_person_prototype(
            person_name,
            embeddings,
            similarity_threshold=similarity_threshold,
        )
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_path, face_index) DO UPDATE SET
                    person_name=excluded.person_name,
                    confidence=excluded.confidence
                """,
                [
                    (record.image_path, int(record.face_index), person.person_name, 1.0)
                    for record in records
                ],
            )
        return person

    def _save_person_prototype(
        self,
        person_name: str,
        embeddings: np.ndarray,
        *,
        similarity_threshold: float,
    ) -> PersonPrototype:
        prototype = embeddings.mean(axis=0)
        prototype = prototype / np.clip(np.linalg.norm(prototype), 1e-12, None)
        person = PersonPrototype(
            person_name=str(person_name).strip(),
            embedding=prototype.astype(np.float32),
            similarity_threshold=float(similarity_threshold),
            example_count=int(len(embeddings)),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO person_prototypes(person_name, embedding, similarity_threshold, example_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(person_name) DO UPDATE SET
                    embedding=excluded.embedding,
                    similarity_threshold=excluded.similarity_threshold,
                    example_count=excluded.example_count
                """,
                (
                    person.person_name,
                    np.asarray(person.embedding, dtype=np.float32).tobytes(),
                    person.similarity_threshold,
                    person.example_count,
                ),
            )
        return person

    def load_person_prototypes(self) -> list[PersonPrototype]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT person_name, embedding, similarity_threshold, example_count FROM person_prototypes"
            ).fetchall()
        return [
            PersonPrototype(
                person_name=row[0],
                embedding=np.frombuffer(row[1], dtype=np.float32).copy(),
                similarity_threshold=float(row[2]),
                example_count=int(row[3]),
            )
            for row in rows
        ]

    def save_person_profile(
        self,
        person_name: str,
        *,
        notes: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        cover_face_ref: tuple[str, int] | None = None,
    ) -> None:
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("Person name is required.")
        clean_tags = tuple(str(tag).strip() for tag in (tags or []) if str(tag).strip())
        cover_image_path = str((cover_face_ref or ("", 0))[0] or "")
        cover_face_index = int((cover_face_ref or ("", 0))[1] or 0)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO person_profiles(person_name, notes, tags_json, cover_image_path, cover_face_index)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(person_name) DO UPDATE SET
                    notes=excluded.notes,
                    tags_json=excluded.tags_json,
                    cover_image_path=excluded.cover_image_path,
                    cover_face_index=excluded.cover_face_index
                """,
                (name, str(notes or ""), json.dumps(list(clean_tags)), cover_image_path, cover_face_index),
            )

    def load_profile_examples(
        self,
        person_name: str,
        *,
        limit: int = 20,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
    ) -> list[IndexedFaceRecord]:
        name = str(person_name or "").strip()
        if not name:
            return []
        allowed_paths = self._normalized_path_set(candidate_paths)
        limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT i.image_path, i.face_index
                FROM face_labels l
                JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
                WHERE l.person_name=?
                ORDER BY l.confidence DESC, i.image_path, i.face_index
                """,
                (name,),
            ).fetchall()
        records: list[IndexedFaceRecord] = []
        for row in rows:
            image_path = str(row[0])
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            record = self.load_face_record(image_path, int(row[1]))
            if record is not None:
                records.append(record)
            if len(records) >= limit:
                break
        return records

    def load_person_profiles(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        limit: int = 200,
    ) -> list[PersonProfile]:
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        prototypes = {profile.person_name: profile for profile in self.load_person_prototypes()}
        labeled_counts = self.label_counts()
        with self._connect() as connection:
            profile_rows = connection.execute(
                "SELECT person_name, notes, tags_json, cover_image_path, cover_face_index FROM person_profiles"
            ).fetchall()
        profile_meta = {
            str(row[0]): {
                "notes": str(row[1] or ""),
                "tags": tuple(str(tag).strip() for tag in json.loads(row[2] or "[]") if str(tag).strip()),
                "cover_image_path": str(row[3] or ""),
                "cover_face_index": int(row[4] or 0),
            }
            for row in profile_rows
        }
        profiles: list[PersonProfile] = []
        for name, prototype in sorted(prototypes.items(), key=lambda item: item[0].lower()):
            visible_count = len(
                self.load_profile_examples(
                    name,
                    limit=max(1, int(limit)),
                    folder_prefix=folder_prefix,
                    candidate_paths=candidate_paths,
                )
            )
            meta = profile_meta.get(name, {})
            profiles.append(
                PersonProfile(
                    person_name=name,
                    similarity_threshold=float(prototype.similarity_threshold),
                    example_count=int(prototype.example_count),
                    labeled_count=int(labeled_counts.get(name, 0)),
                    visible_face_count=int(visible_count),
                    notes=str(meta.get("notes", "")),
                    tags=tuple(meta.get("tags", ())),
                    cover_image_path=str(meta.get("cover_image_path", "")),
                    cover_face_index=int(meta.get("cover_face_index", 0)),
                )
            )
        return profiles[: max(1, int(limit))]

    def label_counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT person_name, COUNT(1) FROM face_labels GROUP BY person_name"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def search_by_person_name(
        self,
        person_name: str,
        *,
        top_k: int = 60,
        min_score: float | None = None,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
    ) -> list[FaceSearchResult]:
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("Person name is required.")
        prototypes = {p.person_name: p for p in self.load_person_prototypes()}
        prototype = prototypes.get(name)
        if prototype is None:
            raise ValueError(f"No saved face prototype found for '{name}'.")

        threshold = float(min_score) if min_score is not None else float(prototype.similarity_threshold)
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)

        query = "SELECT image_path, face_index, bbox_json, face_confidence, embedding FROM face_index"
        args: list[object] = []
        if folder_prefix:
            query += " WHERE image_path LIKE ?"
            args.append(f"{folder_prefix}%")

        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()

        results: list[FaceSearchResult] = []
        for row in rows:
            if not self._path_in_scope(str(row[0]), folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            embedding = np.frombuffer(row[4], dtype=np.float32).copy()
            score = float(np.dot(embedding, prototype.embedding))
            if score < threshold:
                continue
            results.append(
                FaceSearchResult(
                    image_path=str(row[0]),
                    score=round(score, 6),
                    phash_distance=-1,
                    face_bbox=tuple(json.loads(row[2])),
                    face_confidence=round(float(row[3]), 6),
                    model_name="facenet-vggface2",
                    match_reason=f"name match: {name}",
                )
            )
        results.sort(key=lambda item: (-item.score, item.image_path, item.face_bbox))
        return results[: max(1, int(top_k))]

    def auto_propagate_labels(self) -> list[FaceLabelAssignment]:
        prototypes = self.load_person_prototypes()
        records = self.load_all_records()
        if not prototypes or not records:
            return []
        assignments = []
        for record in records:
            best_person = None
            best_score = -1.0
            for prototype in prototypes:
                score = float(np.dot(record.embedding, prototype.embedding))
                if score >= prototype.similarity_threshold and score > best_score:
                    best_person = prototype.person_name
                    best_score = score
            if best_person is not None:
                assignments.append(
                    FaceLabelAssignment(
                        person_name=best_person,
                        image_path=record.image_path,
                        face_index=record.face_index,
                        face_bbox=record.face_bbox,
                        confidence=round(best_score, 6),
                    )
                )
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_path, face_index) DO UPDATE SET
                    person_name=excluded.person_name,
                    confidence=excluded.confidence
                """,
                [
                    (assignment.image_path, assignment.face_index, assignment.person_name, assignment.confidence)
                    for assignment in assignments
                ],
            )
        return assignments

    def list_face_labels(self) -> list[FaceLabelAssignment]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.person_name, f.image_path, f.face_index, i.bbox_json, f.confidence
                FROM face_labels f
                JOIN face_index i ON i.image_path=f.image_path AND i.face_index=f.face_index
                ORDER BY f.person_name, f.confidence DESC, f.image_path
                """
            ).fetchall()
        return [
            FaceLabelAssignment(
                person_name=row[0],
                image_path=row[1],
                face_index=int(row[2]),
                face_bbox=tuple(json.loads(row[3])),
                confidence=float(row[4]),
            )
            for row in rows
        ]

    def merge_person_labels(self, source_name: str, target_name: str) -> None:
        source_name = source_name.strip()
        target_name = target_name.strip()
        if not source_name or not target_name or source_name == target_name:
            return
        with self._connect() as connection:
            connection.execute(
                "UPDATE face_labels SET person_name=? WHERE person_name=?",
                (target_name, source_name),
            )
            connection.execute(
                "DELETE FROM person_prototypes WHERE person_name=?",
                (source_name,),
            )

    def clear_person_labels(self, person_name: str) -> None:
        person_name = person_name.strip()
        if not person_name:
            return
        with self._connect() as connection:
            connection.execute("DELETE FROM face_labels WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM person_prototypes WHERE person_name=?", (person_name,))





