from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from time import monotonic
from typing import Literal

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageOps
from facenet_pytorch import InceptionResnetV1, MTCNN

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None

from infra.logging_config import get_logger
from infra.cancel import raise_if_cancelled
from infra.atomic_io import atomic_write_text, atomic_write_with
from infra.runtime import ExecutionPolicy, RuntimeCapabilityService, preload_onnx_cuda_runtime_libraries
from infra.settings import get_settings
from app.path_scope import folder_scope_sql, normalize_scoped_path, path_is_within_scope
from app.services.cluster_explanations import ClusterExplanation
from app.services.clustering_options import backend_tooltip, clustering_backend_names
from app.services.model_assets import ModelAssetService
from ml.clustering import ClusteringService

from .discovery import ImageDiscoveryService
from .face_model_installer import face_model_runtime_root_dir
from .face_types import EditableFaceInput


LOGGER = get_logger(__name__)
EXIF_ORIENTATION_TAG = 274


FaceMode = Literal["human", "dog", "cat"]
FACE_MODE_LABELS: dict[str, str] = {
    "human": "Human",
    "dog": "Dog",
    "cat": "Cat",
}
ANIMAL_FACE_MODES: tuple[FaceMode, ...] = ("dog", "cat")
BUILTIN_HUMAN_DETECTOR_ID = "mtcnn_builtin"
BUILTIN_HUMAN_EMBEDDER_ID = "vggface2_builtin"
DEFAULT_HUMAN_FACE_PROFILE_ID = "latest_gpu"
DEFAULT_HUMAN_FACE_DETECTOR_ID = "scrfd_10g_kps"
DEFAULT_HUMAN_FACE_EMBEDDER_ID = "arcface_r100_glint360k"
LEGACY_DEFAULT_BUNDLE_ID = "default_bundle"
DEFAULT_FACE_MAX_DETECTIONS = 50
DEFAULT_FACE_SCORE_THRESHOLD = 0.0
DEFAULT_BUNDLED_FACE_PROFILE = DEFAULT_HUMAN_FACE_PROFILE_ID
MAX_FACE_INDEX_PROGRESS_UPDATES = 32
FACE_INDEX_PROGRESS_MIN_INTERVAL_S = 0.15
FACE_SCAN_ROW_PREFETCH_CHUNK = 500

FACE_MODEL_PROFILE_LABELS: dict[str, str] = {
    "accuracy": "Accuracy",
    "balanced": "Balanced",
    "edge": "Edge",
    "latest_gpu": "Latest GPU",
    "max_accuracy": "Max Accuracy",
    "opencv_cpu": "OpenCV CPU",
    "custom": "Custom",
}
FACE_MODEL_PROFILES: dict[str, dict[str, dict[str, object]]] = {
    "accuracy": {
        "human": {
            "detector_id": "scrfd_10g_kps",
            "embedder_id": "adaface_r100",
            "score_threshold": 0.35,
            "max_detections": 50,
        }
    },
    "balanced": {
        "human": {
            "detector_id": "scrfd_2.5g_kps",
            "embedder_id": "arcface_r50",
            "score_threshold": 0.35,
            "max_detections": 50,
        }
    },
    "edge": {
        "human": {
            "detector_id": "scrfd_500m_kps",
            "embedder_id": "mobilefacenet_arcface",
            "score_threshold": 0.30,
            "max_detections": 24,
        }
    },
    "latest_gpu": {
        "human": {
            "detector_id": "scrfd_10g_kps",
            "embedder_id": "arcface_r100_glint360k",
            "score_threshold": 0.35,
            "max_detections": 50,
        }
    },
    "max_accuracy": {
        "human": {
            "detector_id": "scrfd_34gf_kps",
            "embedder_id": "adaface_r100",
            "score_threshold": 0.35,
            "max_detections": 50,
        }
    },
    "opencv_cpu": {
        "human": {
            "detector_id": "yunet_2026may",
            "embedder_id": "sface_2021dec",
            "score_threshold": 0.35,
            "max_detections": 50,
        }
    },
}


def normalize_face_mode(value: str | None) -> FaceMode:
    text = str(value or "human").strip().lower()
    if text in {"dog", "cat"}:
        return text
    return "human"


def face_quality_profile_choices(mode: str | FaceMode) -> list[tuple[str, str]]:
    mode_id = normalize_face_mode(str(mode))
    labels = {
        "high_recall": "High Recall",
        DEFAULT_FACE_QUALITY_PROFILE_ID: "Balanced",
        "high_precision": "High Precision",
    }
    return [(item_id, labels.get(item_id, item_id.replace("_", " ").title())) for item_id in FACE_QUALITY_PROFILES.get(mode_id, {})]


def default_face_quality_profile_id(mode: str | FaceMode) -> str:
    mode_id = normalize_face_mode(str(mode))
    if DEFAULT_FACE_QUALITY_PROFILE_ID in FACE_QUALITY_PROFILES.get(mode_id, {}):
        return DEFAULT_FACE_QUALITY_PROFILE_ID
    for item_id in FACE_QUALITY_PROFILES.get(mode_id, {}):
        return item_id
    return DEFAULT_FACE_QUALITY_PROFILE_ID


def face_quality_profile_config(mode: str | FaceMode, profile_id: str | None) -> dict[str, object]:
    mode_id = normalize_face_mode(str(mode))
    config = FACE_QUALITY_PROFILES.get(mode_id, {})
    item_id = str(profile_id or "").strip().lower() or default_face_quality_profile_id(mode_id)
    return dict(config.get(item_id) or config.get(default_face_quality_profile_id(mode_id)) or {})


def face_detector_policy_choices() -> list[tuple[str, str]]:
    return [
        ("single", "Single detector"),
        ("rescue_on_no_faces", "Fallback on no faces"),
        ("rescue_on_low_confidence", "Fallback on low confidence"),
        ("consensus", "Consensus"),
        ("union_then_verify", "Union then verify"),
    ]


def face_verifier_mode_choices() -> list[tuple[str, str]]:
    return [
        ("off", "Off"),
        ("human_face_verifier", "Human face verifier"),
        ("animal_face_verifier", "Animal face verifier"),
    ]


def face_quality_gate_choices() -> list[tuple[str, str]]:
    return [
        ("clean", "Clean only"),
        ("review", "Review or better"),
        ("reject", "All faces"),
    ]


def face_rerank_policy_choices() -> list[tuple[str, str]]:
    return [
        ("off", "Off"),
        ("quality_score", "Quality score"),
    ]


def face_cluster_backend_choices() -> list[tuple[str, str]]:
    labels = {
        "cosine-kmeans": "Cosine KMeans",
        "hdbscan": "HDBSCAN",
        "graph": "Graph",
    }
    choices = [
        (backend, labels.get(backend, backend.replace("-", " ").title()))
        for backend in clustering_backend_names("production")
    ]
    return choices


def face_cluster_backend_tooltip(backend_id: str) -> str:
    backend = str(backend_id or "").strip().lower()
    return backend_tooltip(backend)


def face_cluster_outlier_policy_choices() -> list[tuple[str, str]]:
    return [
        ("assign", "Assign to nearest cluster"),
        ("isolate", "Keep as outlier cluster"),
        ("keep", "Keep backend outliers as-is"),
    ]


def face_mode_label(value: str | None) -> str:
    mode = normalize_face_mode(value)
    return FACE_MODE_LABELS.get(mode, FACE_MODE_LABELS["human"])


def normalize_face_component_id(value: str | None, fallback: str = "") -> str:
    text = str(value or fallback).strip().lower().replace(" ", "_")
    return text or str(fallback or "").strip().lower().replace(" ", "_")


def face_model_root_dir(model_root: str | Path | None) -> Path | None:
    root_text = str(model_root or "").strip()
    if not root_text:
        return None
    return Path(root_text).expanduser()


def bundled_face_model_root_dirs() -> tuple[Path, ...]:
    roots: list[Path] = []
    env_root = str(os.environ.get("IMAGE_CLUSTERING_FACE_MODEL_ASSETS_DIR", "") or "").strip()
    if env_root:
        roots.append(Path(env_root).expanduser())
    repo_root = Path(__file__).resolve().parents[3]
    roots.append(repo_root / "face_model_assets")
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass) / "face_model_assets")
    try:
        roots.append(Path(sys.executable).resolve().parent / "face_model_assets")
    except Exception:
        pass
    roots.append(repo_root / "build" / "face_model_assets")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return tuple(unique)


def face_model_candidate_roots(model_root: str | Path | None) -> tuple[tuple[str, Path], ...]:
    roots: list[tuple[str, Path]] = [("managed", face_model_runtime_root_dir())]
    external_root = face_model_root_dir(model_root)
    if external_root is not None:
        roots.append(("external", external_root))
    roots.extend(("bundled", root) for root in bundled_face_model_root_dirs())
    unique: list[tuple[str, Path]] = []
    seen: set[tuple[str, str]] = set()
    for source_kind, root in roots:
        key = (source_kind, str(root.resolve()) if root.exists() else str(root))
        if key in seen:
            continue
        seen.add(key)
        unique.append((source_kind, root))
    return tuple(unique)


def animal_model_bundle_dir(model_root: str | Path | None, mode: str | None) -> Path:
    mode_id = normalize_face_mode(mode)
    root_text = str(model_root or "").strip()
    if mode_id not in ANIMAL_FACE_MODES:
        return Path(root_text) if root_text else Path()
    if not root_text:
        return Path() / mode_id
    return Path(root_text).expanduser() / mode_id


def _parse_triplet(values: object, *, key: str) -> tuple[float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError(f"{key} must contain exactly 3 numeric values.")
    try:
        return tuple(float(item) for item in values)
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"{key} must contain numeric values.") from exc


def _parse_size_pair(values: object, *, key: str) -> tuple[int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{key} must contain exactly 2 integer values.")
    try:
        width = max(1, int(values[0]))
        height = max(1, int(values[1]))
    except Exception as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"{key} must contain integer values.") from exc
    return width, height


def face_model_profile_choices(mode: str | None) -> list[tuple[str, str]]:
    mode_id = normalize_face_mode(mode)
    if mode_id != "human":
        return [("custom", FACE_MODEL_PROFILE_LABELS["custom"])]
    profile_ids = ("accuracy", "balanced", "edge", "latest_gpu", "max_accuracy", "opencv_cpu", "custom")
    return [(profile_id, FACE_MODEL_PROFILE_LABELS[profile_id]) for profile_id in profile_ids]


def face_model_profile_config(mode: str | None, profile_id: str | None) -> dict[str, object] | None:
    mode_id = normalize_face_mode(mode)
    profile_key = str(profile_id or "").strip().lower()
    if not profile_key or profile_key == "custom":
        return None
    payload = FACE_MODEL_PROFILES.get(profile_key, {}).get(mode_id)
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def selected_face_model_profile(
    mode: str | None,
    detector_id: str | None,
    embedder_id: str | None,
    *,
    score_threshold: float,
    max_detections: int,
) -> str:
    mode_id = normalize_face_mode(mode)
    if mode_id != "human":
        return "custom"
    detector = normalize_face_component_id(detector_id)
    embedder = normalize_face_component_id(embedder_id)
    score = float(score_threshold)
    max_faces = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))
    for profile_id in ("accuracy", "balanced", "edge", "latest_gpu", "max_accuracy", "opencv_cpu"):
        payload = face_model_profile_config(mode_id, profile_id)
        if not payload:
            continue
        if detector != normalize_face_component_id(str(payload.get("detector_id") or "")):
            continue
        if embedder != normalize_face_component_id(str(payload.get("embedder_id") or "")):
            continue
        if abs(float(payload.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0) - score) > 1e-6:
            continue
        if int(payload.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS) != max_faces:
            continue
        return profile_id
    return "custom"


def _image_exif_orientation(image: Image.Image) -> int:
    try:
        orientation = int(image.getexif().get(EXIF_ORIENTATION_TAG, 1) or 1)
    except Exception:
        orientation = 1
    return orientation if 1 <= orientation <= 8 else 1


def _display_image_size(raw_width: int, raw_height: int, orientation: int) -> tuple[int, int]:
    width = max(0, int(raw_width))
    height = max(0, int(raw_height))
    if int(orientation) in {5, 6, 7, 8}:
        return height, width
    return width, height


def _read_oriented_image_info(image_path: str) -> OrientedImageInfo:
    with Image.open(image_path) as image:
        raw_width = int(image.width)
        raw_height = int(image.height)
        orientation = _image_exif_orientation(image)
    display_width, display_height = _display_image_size(raw_width, raw_height, orientation)
    return OrientedImageInfo(
        raw_width=raw_width,
        raw_height=raw_height,
        display_width=display_width,
        display_height=display_height,
        orientation=orientation,
    )


def _open_display_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _transpose_method_for_orientation(orientation: int):
    return {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }.get(int(orientation))


def _transform_bbox_to_display_space(
    bbox: tuple[int, int, int, int],
    *,
    raw_width: int,
    raw_height: int,
    orientation: int,
) -> tuple[int, int, int, int] | None:
    method = _transpose_method_for_orientation(int(orientation))
    width = max(1, int(raw_width))
    height = max(1, int(raw_height))
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    left = max(0, min(width, min(x1, x2)))
    top = max(0, min(height, min(y1, y2)))
    right = max(0, min(width, max(x1, x2)))
    bottom = max(0, min(height, max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    if method is None:
        return left, top, right, bottom
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rectangle((left, top, max(left, right - 1), max(top, bottom - 1)), fill=255)
    transformed = mask.transpose(method)
    migrated = transformed.getbbox()
    if migrated is None:
        return None
    return tuple(int(value) for value in migrated)


@dataclass(frozen=True)
class AnimalFaceBundle:
    mode: FaceMode
    bundle_dir: Path
    display_name: str
    detector_path: Path
    detector_input_name: str
    detector_input_size: tuple[int, int]
    detector_output_boxes_name: str
    detector_output_scores_name: str
    detector_mean: tuple[float, float, float]
    detector_std: tuple[float, float, float]
    detector_score_threshold: float
    detector_boxes_normalized: bool
    detector_box_format: str
    embedder_path: Path
    embedder_input_name: str
    embedder_input_size: tuple[int, int]
    embedder_output_name: str
    embedder_mean: tuple[float, float, float]
    embedder_std: tuple[float, float, float]

    @property
    def model_name(self) -> str:
        return f"{self.mode}-onnx"


@dataclass(frozen=True)
class FaceDetectorBundle:
    mode: FaceMode
    detector_id: str
    display_name: str
    backend_family: str
    source_kind: str
    bundle_dir: Path = Path()
    detector_path: Path | None = None
    detector_input_name: str = ""
    detector_input_size: tuple[int, int] = (0, 0)
    detector_output_boxes_name: str = ""
    detector_output_scores_name: str = ""
    detector_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    detector_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    detector_score_threshold: float = 0.0
    detector_boxes_normalized: bool = True
    detector_box_format: str = "xyxy"
    detector_max_detections: int = DEFAULT_FACE_MAX_DETECTIONS
    profile: str = "external"
    summary: str = ""
    hardware_class: str = ""
    recommended_pair_id: str = ""
    source_url: str = ""
    license: str = ""
    available: bool = True
    availability_message: str = ""

    @property
    def model_name(self) -> str:
        if self.source_kind == "builtin":
            return str(self.detector_id)
        return f"{self.mode}-{self.detector_id}"


@dataclass(frozen=True)
class FaceEmbedderBundle:
    mode: FaceMode
    embedder_id: str
    display_name: str
    source_kind: str
    bundle_dir: Path = Path()
    embedder_path: Path | None = None
    embedder_input_name: str = ""
    embedder_input_size: tuple[int, int] = (0, 0)
    embedder_output_name: str = ""
    embedder_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    embedder_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    profile: str = "external"
    summary: str = ""
    hardware_class: str = ""
    recommended_pair_id: str = ""
    source_url: str = ""
    license: str = ""
    available: bool = True
    availability_message: str = ""

    @property
    def model_name(self) -> str:
        if self.source_kind == "builtin":
            return str(self.embedder_id)
        return f"{self.mode}-{self.embedder_id}"


def _legacy_animal_face_bundle(model_root: str | Path | None, mode: str | None) -> AnimalFaceBundle:
    mode_id = normalize_face_mode(mode)
    if mode_id not in ANIMAL_FACE_MODES:
        raise ValueError("Animal ONNX bundles are only valid for dog and cat modes.")
    root = face_model_root_dir(model_root)
    if root is None:
        raise ValueError("Animal face model root is not configured.")
    bundle_dir = root / mode_id
    if not bundle_dir.exists():
        raise ValueError(f"Missing {face_mode_label(mode_id)} model bundle directory: {bundle_dir}")
    detector_path = bundle_dir / "detector.onnx"
    embedder_path = bundle_dir / "embedder.onnx"
    metadata_path = bundle_dir / "metadata.json"
    if not detector_path.exists():
        raise ValueError(f"Missing detector model: {detector_path}")
    if not embedder_path.exists():
        raise ValueError(f"Missing embedder model: {embedder_path}")
    if not metadata_path.exists():
        raise ValueError(f"Missing bundle metadata: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read bundle metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Bundle metadata must be a JSON object: {metadata_path}")
    detector = metadata.get("detector")
    embedder = metadata.get("embedder")
    if not isinstance(detector, dict):
        raise ValueError("Bundle metadata is missing a 'detector' object.")
    if not isinstance(embedder, dict):
        raise ValueError("Bundle metadata is missing an 'embedder' object.")
    display_name = str(metadata.get("display_name") or face_mode_label(mode_id)).strip() or face_mode_label(mode_id)
    detector_input_name = str(detector.get("input_name") or "").strip()
    detector_boxes_name = str(detector.get("output_boxes_name") or "").strip()
    detector_scores_name = str(detector.get("output_scores_name") or "").strip()
    detector_box_format = str(detector.get("box_format") or "xyxy").strip().lower()
    if detector_box_format not in {"xyxy", "yxyx"}:
        raise ValueError("detector.box_format must be 'xyxy' or 'yxyx'.")
    embedder_input_name = str(embedder.get("input_name") or "").strip()
    embedder_output_name = str(embedder.get("output_name") or "").strip()
    if not detector_input_name:
        raise ValueError("detector.input_name is required.")
    if not detector_boxes_name:
        raise ValueError("detector.output_boxes_name is required.")
    if not embedder_input_name:
        raise ValueError("embedder.input_name is required.")
    if not embedder_output_name:
        raise ValueError("embedder.output_name is required.")
    return AnimalFaceBundle(
        mode=mode_id,
        bundle_dir=bundle_dir,
        display_name=display_name,
        detector_path=detector_path,
        detector_input_name=detector_input_name,
        detector_input_size=_parse_size_pair(detector.get("input_size"), key="detector.input_size"),
        detector_output_boxes_name=detector_boxes_name,
        detector_output_scores_name=detector_scores_name,
        detector_mean=_parse_triplet(detector.get("mean"), key="detector.mean"),
        detector_std=_parse_triplet(detector.get("std"), key="detector.std"),
        detector_score_threshold=float(detector.get("score_threshold", 0.45)),
        detector_boxes_normalized=bool(detector.get("boxes_normalized", True)),
        detector_box_format=detector_box_format,
        embedder_path=embedder_path,
        embedder_input_name=embedder_input_name,
        embedder_input_size=_parse_size_pair(embedder.get("input_size"), key="embedder.input_size"),
        embedder_output_name=embedder_output_name,
        embedder_mean=_parse_triplet(embedder.get("mean"), key="embedder.mean"),
        embedder_std=_parse_triplet(embedder.get("std"), key="embedder.std"),
    )


def _read_bundle_metadata(metadata_path: Path) -> dict[str, object]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Failed to read bundle metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"Bundle metadata must be a JSON object: {metadata_path}")
    return metadata


def _metadata_supports_mode(metadata: dict[str, object], mode: FaceMode) -> bool:
    supported = metadata.get("supported_modes")
    if not isinstance(supported, (list, tuple)) or not supported:
        return True
    normalized = {normalize_face_mode(str(value)) for value in supported if str(value or "").strip()}
    return mode in normalized


def _component_bundle_roots(model_root: str | Path | None, mode: str | None, kind: str) -> tuple[tuple[str, Path], ...]:
    mode_id = normalize_face_mode(mode)
    return tuple((source_kind, root / mode_id / f"{kind}s") for source_kind, root in face_model_candidate_roots(model_root))


def _load_face_detector_bundle(bundle_dir: Path, mode: FaceMode, detector_id: str, *, source_kind: str) -> FaceDetectorBundle:
    metadata_path = bundle_dir / "metadata.json"
    detector_path = bundle_dir / "detector.onnx"
    if not metadata_path.exists():
        raise ValueError(f"Missing detector metadata: {metadata_path}")
    metadata = _read_bundle_metadata(metadata_path)
    if not _metadata_supports_mode(metadata, mode):
        raise ValueError(f"Detector bundle does not support {face_mode_label(mode)} mode: {bundle_dir}")
    payload = metadata.get("detector") if isinstance(metadata.get("detector"), dict) else metadata
    if not isinstance(payload, dict):
        raise ValueError("Detector bundle metadata is invalid.")
    detector_input_name = str(payload.get("input_name") or "").strip()
    detector_boxes_name = str(payload.get("output_boxes_name") or "").strip()
    detector_scores_name = str(payload.get("output_scores_name") or "").strip()
    detector_box_format = str(payload.get("box_format") or "xyxy").strip().lower()
    if detector_box_format not in {"xyxy", "yxyx"}:
        raise ValueError("detector.box_format must be 'xyxy' or 'yxyx'.")
    if not detector_input_name:
        raise ValueError("detector.input_name is required.")
    if not detector_boxes_name:
        raise ValueError("detector.output_boxes_name is required.")
    available = detector_path.exists()
    return FaceDetectorBundle(
        mode=mode,
        detector_id=normalize_face_component_id(detector_id, detector_id),
        display_name=str(metadata.get("display_name") or bundle_dir.name).strip() or bundle_dir.name,
        backend_family=str(metadata.get("backend_family") or payload.get("backend_family") or "onnx").strip().lower() or "onnx",
        source_kind=source_kind,
        bundle_dir=bundle_dir,
        detector_path=detector_path if available else None,
        detector_input_name=detector_input_name,
        detector_input_size=_parse_size_pair(payload.get("input_size"), key="detector.input_size"),
        detector_output_boxes_name=detector_boxes_name,
        detector_output_scores_name=detector_scores_name,
        detector_mean=_parse_triplet(payload.get("mean"), key="detector.mean"),
        detector_std=_parse_triplet(payload.get("std"), key="detector.std"),
        detector_score_threshold=float(payload.get("score_threshold", 0.0)),
        detector_boxes_normalized=bool(payload.get("boxes_normalized", True)),
        detector_box_format=detector_box_format,
        detector_max_detections=max(1, int(payload.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS)),
        profile=str(metadata.get("profile") or "external").strip().lower() or "external",
        summary=str(metadata.get("summary") or "").strip(),
        hardware_class=str(metadata.get("hardware_class") or "").strip(),
        recommended_pair_id=normalize_face_component_id(str(metadata.get("recommended_pair_id") or ""), ""),
        source_url=str(metadata.get("source_url") or "").strip(),
        license=str(metadata.get("license") or "").strip(),
        available=available,
        availability_message=(
            f"Ready at {bundle_dir}."
            if available
            else (
                f"{bundle_dir.name} is not installed. Use Settings > Face Models > Install Selected Face Pack. "
                f"Managed cache: {face_model_runtime_root_dir()}."
                if source_kind in {"bundled", "managed"}
                else f"Missing detector.onnx in the configured external face-model bundle: {bundle_dir}."
            )
        ),
    )


def _load_face_embedder_bundle(bundle_dir: Path, mode: FaceMode, embedder_id: str, *, source_kind: str) -> FaceEmbedderBundle:
    metadata_path = bundle_dir / "metadata.json"
    embedder_path = bundle_dir / "embedder.onnx"
    if not metadata_path.exists():
        raise ValueError(f"Missing embedder metadata: {metadata_path}")
    metadata = _read_bundle_metadata(metadata_path)
    if not _metadata_supports_mode(metadata, mode):
        raise ValueError(f"Embedder bundle does not support {face_mode_label(mode)} mode: {bundle_dir}")
    payload = metadata.get("embedder") if isinstance(metadata.get("embedder"), dict) else metadata
    if not isinstance(payload, dict):
        raise ValueError("Embedder bundle metadata is invalid.")
    embedder_input_name = str(payload.get("input_name") or "").strip()
    embedder_output_name = str(payload.get("output_name") or "").strip()
    if not embedder_input_name:
        raise ValueError("embedder.input_name is required.")
    if not embedder_output_name:
        raise ValueError("embedder.output_name is required.")
    available = embedder_path.exists()
    return FaceEmbedderBundle(
        mode=mode,
        embedder_id=normalize_face_component_id(embedder_id, embedder_id),
        display_name=str(metadata.get("display_name") or bundle_dir.name).strip() or bundle_dir.name,
        source_kind=source_kind,
        bundle_dir=bundle_dir,
        embedder_path=embedder_path if available else None,
        embedder_input_name=embedder_input_name,
        embedder_input_size=_parse_size_pair(payload.get("input_size"), key="embedder.input_size"),
        embedder_output_name=embedder_output_name,
        embedder_mean=_parse_triplet(payload.get("mean"), key="embedder.mean"),
        embedder_std=_parse_triplet(payload.get("std"), key="embedder.std"),
        profile=str(metadata.get("profile") or "external").strip().lower() or "external",
        summary=str(metadata.get("summary") or "").strip(),
        hardware_class=str(metadata.get("hardware_class") or "").strip(),
        recommended_pair_id=normalize_face_component_id(str(metadata.get("recommended_pair_id") or ""), ""),
        source_url=str(metadata.get("source_url") or "").strip(),
        license=str(metadata.get("license") or "").strip(),
        available=available,
        availability_message=(
            f"Ready at {bundle_dir}."
            if available
            else (
                f"{bundle_dir.name} is not installed. Use Settings > Face Models > Install Selected Face Pack. "
                f"Managed cache: {face_model_runtime_root_dir()}."
                if source_kind in {"bundled", "managed"}
                else f"Missing embedder.onnx in the configured external face-model bundle: {bundle_dir}."
            )
        ),
    )


_RAW_EXTERNAL_DETECTOR_FILENAMES: dict[str, tuple[str, ...]] = {
    "scrfd_500m_kps": ("det_500m.onnx", "scrfd_500m.onnx"),
    "scrfd_2.5g_kps": ("det_2.5g.onnx", "scrfd_2.5g.onnx"),
    "scrfd_10g_kps": ("scrfd_10g_gnkps_fp32.onnx", "scrfd_10g.onnx", "det_10g.onnx"),
    "scrfd_34gf_kps": ("scrfd_34g_gnkps.onnx", "scrfd_34g.onnx", "det_34g.onnx"),
}
_RAW_EXTERNAL_EMBEDDER_FILENAMES: dict[str, tuple[str, ...]] = {
    "arcface_r50": ("w600k_r50.onnx", "arcface_r50.onnx"),
    "arcface_r100_glint360k": ("glintr100.onnx", "w600k_r100.onnx", "arcface_r100.onnx"),
    "mobilefacenet_arcface": ("w600k_mbf.onnx", "mobilefacenet.onnx"),
    "adaface_r100": ("adaface_ir101.onnx", "adaface_r100.onnx"),
}


def _raw_external_model_paths(model_root: str | Path | None, filenames: tuple[str, ...]) -> tuple[Path, ...]:
    """Return direct, user-selected ONNX files without scanning arbitrary folders."""
    root = face_model_root_dir(model_root)
    if root is None or not root.is_dir():
        return ()
    matches: list[Path] = []
    for filename in filenames:
        for candidate in (root / filename, root / "models" / filename):
            if candidate.is_file():
                matches.append(candidate)
                break
    return tuple(matches)


def _prefer_available_detector(
    current: FaceDetectorBundle | None,
    candidate: FaceDetectorBundle,
) -> FaceDetectorBundle:
    if current is None or (candidate.available and not current.available):
        return candidate
    return current


def _prefer_available_embedder(
    current: FaceEmbedderBundle | None,
    candidate: FaceEmbedderBundle,
) -> FaceEmbedderBundle:
    if current is None or (candidate.available and not current.available):
        return candidate
    return current


def list_face_detector_bundles(model_root: str | Path | None, mode: str | None) -> list[FaceDetectorBundle]:
    mode_id = normalize_face_mode(mode)
    bundles: list[FaceDetectorBundle] = []
    if mode_id == "human":
        bundles.append(
            FaceDetectorBundle(
                mode="human",
                detector_id=BUILTIN_HUMAN_DETECTOR_ID,
                display_name="MTCNN (built-in)",
                backend_family="mtcnn",
                source_kind="builtin",
                detector_score_threshold=0.0,
                detector_max_detections=DEFAULT_FACE_MAX_DETECTIONS,
                profile="builtin",
                summary="Built-in Torch fallback detector for human faces.",
                hardware_class="CPU or CUDA",
                available=True,
                availability_message="Built-in human detector is ready.",
            )
        )
    for root_kind, detector_root in _component_bundle_roots(model_root, mode_id, "detector"):
        if not detector_root.exists():
            continue
        for bundle_dir in sorted([path for path in detector_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
            detector_id = normalize_face_component_id(bundle_dir.name, bundle_dir.name)
            try:
                bundles.append(_load_face_detector_bundle(bundle_dir, mode_id, detector_id, source_kind=root_kind))
            except Exception:
                continue
    if mode_id in ANIMAL_FACE_MODES:
        try:
            legacy = _legacy_animal_face_bundle(model_root, mode_id)
            bundles.append(
                FaceDetectorBundle(
                    mode=mode_id,
                    detector_id=LEGACY_DEFAULT_BUNDLE_ID,
                    display_name=f"{legacy.display_name} detector",
                    backend_family="onnx",
                    source_kind="legacy",
                    bundle_dir=legacy.bundle_dir,
                    detector_path=legacy.detector_path,
                    detector_input_name=legacy.detector_input_name,
                    detector_input_size=legacy.detector_input_size,
                    detector_output_boxes_name=legacy.detector_output_boxes_name,
                    detector_output_scores_name=legacy.detector_output_scores_name,
                    detector_mean=legacy.detector_mean,
                    detector_std=legacy.detector_std,
                    detector_score_threshold=legacy.detector_score_threshold,
                    detector_boxes_normalized=legacy.detector_boxes_normalized,
                    detector_box_format=legacy.detector_box_format,
                    detector_max_detections=DEFAULT_FACE_MAX_DETECTIONS,
                )
            )
        except Exception:
            pass
    catalog = {bundle.detector_id: bundle for bundle in bundles}
    if mode_id == "human":
        for detector_id, filenames in _RAW_EXTERNAL_DETECTOR_FILENAMES.items():
            for model_path in _raw_external_model_paths(model_root, filenames):
                template = catalog.get(detector_id)
                if template is None:
                    continue
                bundles.append(
                    replace(
                        template,
                        source_kind="external",
                        bundle_dir=model_path.parent,
                        detector_path=model_path,
                        available=True,
                        availability_message=f"Downloaded external model: {model_path}.",
                    )
                )
                break
    deduped: dict[str, FaceDetectorBundle] = {}
    for bundle in bundles:
        deduped[bundle.detector_id] = _prefer_available_detector(deduped.get(bundle.detector_id), bundle)
    return list(deduped.values())


def list_face_embedder_bundles(model_root: str | Path | None, mode: str | None) -> list[FaceEmbedderBundle]:
    mode_id = normalize_face_mode(mode)
    bundles: list[FaceEmbedderBundle] = []
    if mode_id == "human":
        builtin_ready = ModelAssetService().local_cache_present("facenet")
        bundles.append(
            FaceEmbedderBundle(
                mode="human",
                embedder_id=BUILTIN_HUMAN_EMBEDDER_ID,
                display_name="VGGFace2 FaceNet (built-in)",
                source_kind="builtin",
                profile="builtin",
                summary="Built-in Torch FaceNet embedder trained on VGGFace2.",
                hardware_class="CPU or CUDA",
                available=builtin_ready,
                availability_message=(
                    "Built-in human embedder is ready."
                    if builtin_ready
                    else "FaceNet weights are not installed. Open Settings > Face Models and install FaceNet."
                ),
            )
        )
    for root_kind, embedder_root in _component_bundle_roots(model_root, mode_id, "embedder"):
        if not embedder_root.exists():
            continue
        for bundle_dir in sorted([path for path in embedder_root.iterdir() if path.is_dir()], key=lambda path: path.name.lower()):
            embedder_id = normalize_face_component_id(bundle_dir.name, bundle_dir.name)
            try:
                bundles.append(_load_face_embedder_bundle(bundle_dir, mode_id, embedder_id, source_kind=root_kind))
            except Exception:
                continue
    if mode_id in ANIMAL_FACE_MODES:
        try:
            legacy = _legacy_animal_face_bundle(model_root, mode_id)
            bundles.append(
                FaceEmbedderBundle(
                    mode=mode_id,
                    embedder_id=LEGACY_DEFAULT_BUNDLE_ID,
                    display_name=f"{legacy.display_name} embedder",
                    source_kind="legacy",
                    bundle_dir=legacy.bundle_dir,
                    embedder_path=legacy.embedder_path,
                    embedder_input_name=legacy.embedder_input_name,
                    embedder_input_size=legacy.embedder_input_size,
                    embedder_output_name=legacy.embedder_output_name,
                    embedder_mean=legacy.embedder_mean,
                    embedder_std=legacy.embedder_std,
                )
            )
        except Exception:
            pass
    catalog = {bundle.embedder_id: bundle for bundle in bundles}
    if mode_id == "human":
        for embedder_id, filenames in _RAW_EXTERNAL_EMBEDDER_FILENAMES.items():
            for model_path in _raw_external_model_paths(model_root, filenames):
                template = catalog.get(embedder_id)
                if template is None:
                    continue
                bundles.append(
                    replace(
                        template,
                        source_kind="external",
                        bundle_dir=model_path.parent,
                        embedder_path=model_path,
                        available=True,
                        availability_message=f"Downloaded external model: {model_path}.",
                    )
                )
                break
    deduped: dict[str, FaceEmbedderBundle] = {}
    for bundle in bundles:
        deduped[bundle.embedder_id] = _prefer_available_embedder(deduped.get(bundle.embedder_id), bundle)
    return list(deduped.values())


def default_face_detector_id(model_root: str | Path | None, mode: str | None) -> str:
    mode_id = normalize_face_mode(mode)
    bundles = list_face_detector_bundles(model_root, mode_id)
    if bundles:
        return str(bundles[0].detector_id)
    if mode_id == "human":
        return BUILTIN_HUMAN_DETECTOR_ID
    return LEGACY_DEFAULT_BUNDLE_ID


def default_face_embedder_id(model_root: str | Path | None, mode: str | None) -> str:
    mode_id = normalize_face_mode(mode)
    bundles = list_face_embedder_bundles(model_root, mode_id)
    for bundle in bundles:
        if bundle.available:
            return str(bundle.embedder_id)
    if bundles:
        return str(bundles[0].embedder_id)
    if mode_id == "human":
        return BUILTIN_HUMAN_EMBEDDER_ID
    return LEGACY_DEFAULT_BUNDLE_ID


def resolve_ready_face_pipeline_ids(
    model_root: str | Path | None,
    mode: str | None,
    detector_id: str | None,
    embedder_id: str | None,
) -> tuple[str, str]:
    """Resolve a detector/embedder preference as one ready, atomic pipeline.

    A detector from a partially installed pack must never be paired silently
    with a missing embedder (or vice versa). When either requested component
    is unavailable, another complete managed pair or the built-in safety pair
    is returned.
    """

    mode_id = normalize_face_mode(mode)
    detector_bundles = list_face_detector_bundles(model_root, mode_id)
    embedder_bundles = list_face_embedder_bundles(model_root, mode_id)
    fallback_detector = default_face_detector_id(model_root, mode_id)
    fallback_embedder = default_face_embedder_id(model_root, mode_id)
    requested_detector = normalize_face_component_id(detector_id, fallback_detector)
    requested_embedder = normalize_face_component_id(embedder_id, fallback_embedder)
    detectors = {str(bundle.detector_id): bundle for bundle in detector_bundles}
    embedders = {str(bundle.embedder_id): bundle for bundle in embedder_bundles}

    def _ready_pair(candidate_detector: str, candidate_embedder: str) -> bool:
        detector_bundle = detectors.get(str(candidate_detector))
        embedder_bundle = embedders.get(str(candidate_embedder))
        return bool(
            detector_bundle is not None
            and detector_bundle.available
            and embedder_bundle is not None
            and embedder_bundle.available
        )

    candidates: list[tuple[str, str]] = [(requested_detector, requested_embedder)]
    requested_detector_bundle = detectors.get(requested_detector)
    requested_embedder_bundle = embedders.get(requested_embedder)
    if requested_detector_bundle is not None and requested_detector_bundle.available:
        candidates.append((requested_detector, str(requested_detector_bundle.recommended_pair_id or "")))
    if requested_embedder_bundle is not None and requested_embedder_bundle.available:
        candidates.append((str(requested_embedder_bundle.recommended_pair_id or ""), requested_embedder))

    # Prefer a complete installed pack before the built-in safety pipeline.
    # This preserves acceleration when a saved component disappears but its
    # matching managed pair is already available.
    for bundle in detector_bundles:
        if bundle.available and bundle.recommended_pair_id:
            candidates.append((str(bundle.detector_id), str(bundle.recommended_pair_id)))
    for bundle in embedder_bundles:
        if bundle.available and bundle.recommended_pair_id:
            candidates.append((str(bundle.recommended_pair_id), str(bundle.embedder_id)))
    if mode_id == "human":
        candidates.append((BUILTIN_HUMAN_DETECTOR_ID, BUILTIN_HUMAN_EMBEDDER_ID))
    candidates.append((fallback_detector, fallback_embedder))
    candidates.extend(
        (str(detector_bundle.detector_id), str(embedder_bundle.embedder_id))
        for detector_bundle in detector_bundles
        if detector_bundle.available
        for embedder_bundle in embedder_bundles
        if embedder_bundle.available
    )

    seen: set[tuple[str, str]] = set()
    for candidate_detector, candidate_embedder in candidates:
        pair = (
            normalize_face_component_id(candidate_detector, fallback_detector),
            normalize_face_component_id(candidate_embedder, fallback_embedder),
        )
        if pair in seen:
            continue
        seen.add(pair)
        if _ready_pair(*pair):
            return pair
    return fallback_detector, fallback_embedder


def _hardware_execution_tag(hardware_class: str | None) -> str:
    text = str(hardware_class or "").strip().lower()
    if not text:
        return ""
    has_cpu = "cpu" in text
    has_gpu = any(token in text for token in ("gpu", "cuda"))
    if has_cpu and has_gpu:
        return "CPU/GPU"
    if has_gpu:
        return "GPU"
    if has_cpu:
        return "CPU"
    return ""


def _face_choice_label(
    display_name: str,
    hardware_class: str | None,
    *,
    available: bool,
    source_kind: str,
) -> str:
    label = str(display_name)
    execution_tag = _hardware_execution_tag(hardware_class)
    if execution_tag:
        label = f"{label} [{execution_tag}]"
    if source_kind == "builtin" and available:
        return f"{label} — Built-in"
    if available:
        return f"{label} — Downloaded"
    return f"{label} — Install for {_hardware_execution_tag(hardware_class) or 'CPU/GPU'}"


def face_detector_choices(model_root: str | Path | None, mode: str | None) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for bundle in list_face_detector_bundles(model_root, mode):
        label = _face_choice_label(
            bundle.display_name,
            bundle.hardware_class,
            available=bundle.available,
            source_kind=bundle.source_kind,
        )
        choices.append((str(bundle.detector_id), label))
    return choices


def face_embedder_choices(model_root: str | Path | None, mode: str | None) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for bundle in list_face_embedder_bundles(model_root, mode):
        label = _face_choice_label(
            bundle.display_name,
            bundle.hardware_class,
            available=bundle.available,
            source_kind=bundle.source_kind,
        )
        choices.append((str(bundle.embedder_id), label))
    return choices


def resolve_face_detector_bundle(model_root: str | Path | None, mode: str | None, detector_id: str | None) -> FaceDetectorBundle:
    mode_id = normalize_face_mode(mode)
    if mode_id in ANIMAL_FACE_MODES and not str(model_root or "").strip():
        raise ValueError("Animal face model root is not configured.")
    target_id = normalize_face_component_id(detector_id, default_face_detector_id(model_root, mode_id))
    for bundle in list_face_detector_bundles(model_root, mode_id):
        if bundle.detector_id == target_id:
            return bundle
    raise ValueError(f"Detector '{target_id}' is not available for {face_mode_label(mode_id)} mode.")


def resolve_face_embedder_bundle(model_root: str | Path | None, mode: str | None, embedder_id: str | None) -> FaceEmbedderBundle:
    mode_id = normalize_face_mode(mode)
    if mode_id in ANIMAL_FACE_MODES and not str(model_root or "").strip():
        raise ValueError("Animal face model root is not configured.")
    target_id = normalize_face_component_id(embedder_id, default_face_embedder_id(model_root, mode_id))
    for bundle in list_face_embedder_bundles(model_root, mode_id):
        if bundle.embedder_id == target_id:
            return bundle
    raise ValueError(f"Embedder '{target_id}' is not available for {face_mode_label(mode_id)} mode.")


def _load_animal_face_bundle(model_root: str | Path | None, mode: str | None) -> AnimalFaceBundle:
    return _legacy_animal_face_bundle(model_root, mode)


def inspect_animal_face_bundle(model_root: str | Path | None, mode: str | None) -> tuple[bool, str]:
    mode_id = normalize_face_mode(mode)
    if mode_id == "human":
        return True, "Human mode uses built-in face models."
    if ort is None:
        return False, "onnxruntime is not installed."
    try:
        bundle = _load_animal_face_bundle(model_root, mode_id)
    except Exception as exc:
        return False, str(exc)
    return True, f"{bundle.display_name} bundle ready at {bundle.bundle_dir}."


def _onnx_input_tensor(
    image: Image.Image,
    *,
    input_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> np.ndarray:
    resized = image.convert("RGB").resize(input_size, Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError("Expected an RGB image for ONNX face inference.")
    array = array[:, :, :3]
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.where(std_array == 0.0, 1.0, std_array)
    normalized = (array - mean_array) / std_array
    chw = np.transpose(normalized, (2, 0, 1))
    return chw[np.newaxis, ...].astype(np.float32)


def _flatten_detector_boxes(raw_boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(raw_boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] < 4:
        raise ValueError("Detector boxes output must have shape Nx4 or Nx5.")
    return boxes


def _flatten_detector_scores(raw_scores: np.ndarray | None, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw_scores is None:
        if boxes.shape[1] >= 5:
            return boxes[:, :4], boxes[:, 4]
        return boxes[:, :4], np.ones((boxes.shape[0],), dtype=np.float32)
    scores = np.asarray(raw_scores, dtype=np.float32)
    if scores.size == 0:
        return boxes[:, :4], scores.reshape(0)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    if scores.shape[0] != boxes.shape[0]:
        if boxes.shape[1] >= 5 and boxes.shape[0] == scores.shape[0]:
            return boxes[:, :4], boxes[:, 4]
        raise ValueError("Detector scores output length does not match boxes output.")
    return boxes[:, :4], scores


def _session_input_name(session, configured_name: str) -> str:
    target = str(configured_name or "").strip()
    input_names = [str(item.name or "").strip() for item in session.get_inputs()]
    if target and target in input_names:
        return target
    if not input_names:
        raise ValueError("ONNX session has no inputs.")
    return input_names[0]


def _session_output_name(session, configured_name: str) -> str:
    target = str(configured_name or "").strip()
    output_names = [str(item.name or "").strip() for item in session.get_outputs()]
    if target and target in output_names:
        return target
    if not output_names:
        raise ValueError("ONNX session has no outputs.")
    return output_names[0]


def _session_output_names(session) -> list[str]:
    return [str(item.name or "").strip() for item in session.get_outputs()]


def _create_onnx_session(
    model_path: Path,
    providers: list[str],
    *,
    allow_cpu_fallback: bool = True,
):
    if ort is None:
        raise RuntimeError("onnxruntime is not installed.")
    ordered: list[str] = []
    for provider in providers:
        name = str(provider or "").strip()
        if name and name not in ordered:
            ordered.append(name)
    if not ordered:
        ordered = ["CPUExecutionProvider"]
    if ordered and ordered[0] == "CUDAExecutionProvider":
        preload_onnx_cuda_runtime_libraries()
    try:
        return ort.InferenceSession(str(model_path), providers=ordered)
    except Exception as exc:
        if (
            not allow_cpu_fallback
            or ordered == ["CPUExecutionProvider"]
            or (ordered and ordered[0] == "CPUExecutionProvider")
        ):
            raise
        LOGGER.warning(
            "Falling back to CPU ONNX session for %s after provider %s failed: %s",
            model_path,
            ordered[0],
            exc,
        )
        return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    converted = np.asarray(boxes, dtype=np.float32).copy()
    if converted.size == 0:
        return converted.reshape(0, 4)
    converted[:, 0] = boxes[:, 0] - (boxes[:, 2] / 2.0)
    converted[:, 1] = boxes[:, 1] - (boxes[:, 3] / 2.0)
    converted[:, 2] = boxes[:, 0] + (boxes[:, 2] / 2.0)
    converted[:, 3] = boxes[:, 1] + (boxes[:, 3] / 2.0)
    return converted


def _clip_xyxy(boxes: np.ndarray, *, width: int, height: int) -> np.ndarray:
    clipped = np.asarray(boxes, dtype=np.float32).copy()
    if clipped.size == 0:
        return clipped.reshape(0, 4)
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width)
    clipped[:, 2] = np.clip(clipped[:, 2], 0, width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height)
    clipped[:, 3] = np.clip(clipped[:, 3], 0, height)
    return clipped


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.size == 0 or scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    keep = torchvision.ops.nms(
        torch.from_numpy(np.asarray(boxes, dtype=np.float32)),
        torch.from_numpy(np.asarray(scores, dtype=np.float32)),
        float(iou_threshold),
    )
    return keep.cpu().numpy().astype(np.int64, copy=False)


def _letterbox_rgb_image(
    image: Image.Image,
    *,
    input_size: tuple[int, int],
    fill_value: int = 114,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    source_height, source_width = rgb.shape[:2]
    target_width = max(1, int(input_size[0]))
    target_height = max(1, int(input_size[1]))
    scale = min(target_width / max(1, source_width), target_height / max(1, source_height))
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    if cv2 is not None:
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    else:  # pragma: no cover - OpenCV is a project dependency
        resized = np.asarray(image.convert("RGB").resize((resized_width, resized_height), Image.Resampling.BILINEAR), dtype=np.uint8)
    pad_left = (target_width - resized_width) / 2.0
    pad_top = (target_height - resized_height) / 2.0
    canvas = np.full((target_height, target_width, 3), fill_value, dtype=np.uint8)
    left = int(round(pad_left))
    top = int(round(pad_top))
    canvas[top : top + resized_height, left : left + resized_width] = resized
    return canvas, float(scale), (float(left), float(top))


def _scale_letterboxed_boxes(
    boxes: np.ndarray,
    *,
    scale: float,
    pad: tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    scaled = np.asarray(boxes, dtype=np.float32).copy()
    if scaled.size == 0:
        return scaled.reshape(0, 4)
    scaled[:, [0, 2]] -= float(pad[0])
    scaled[:, [1, 3]] -= float(pad[1])
    divisor = max(float(scale), 1e-12)
    scaled[:, :4] /= divisor
    return _clip_xyxy(scaled[:, :4], width=width, height=height)


def _distance_to_keypoints(anchor_centers: np.ndarray, distance: np.ndarray) -> np.ndarray:
    centers = np.asarray(anchor_centers, dtype=np.float32)
    deltas = np.asarray(distance, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 2 or deltas.ndim != 2 or deltas.shape[1] < 10:
        return np.zeros((0, 5, 2), dtype=np.float32)
    keypoints = np.zeros((deltas.shape[0], 5, 2), dtype=np.float32)
    for point_index in range(5):
        keypoints[:, point_index, 0] = centers[:, 0] + deltas[:, point_index * 2]
        keypoints[:, point_index, 1] = centers[:, 1] + deltas[:, (point_index * 2) + 1]
    return keypoints


@dataclass(frozen=True)
class DetectedFace:
    image_path: str
    bbox: tuple[int, int, int, int]
    confidence: float
    crop: Image.Image
    landmarks: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class FaceSearchRequest:
    query_face_image: str
    query_face_bbox: tuple[int, int, int, int] | None = None
    top_k: int = 30
    min_face_score: float = 0.35
    cluster_people: bool = False
    candidate_paths: list[str] | None = None
    include_tiny_faces: bool = True


@dataclass(frozen=True)
class FaceSearchResult:
    image_path: str
    face_index: int
    score: float
    phash_distance: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    model_name: str
    match_reason: str
    person_name: str = ""
    quality_status: str = "clean"
    quality_reasons: tuple[str, ...] = ()
    hidden: bool = False
    image_mtime: float = 0.0


@dataclass(frozen=True)
class FaceClusterMember:
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    person_name: str = ""
    quality_status: str = "clean"
    hidden: bool = False


@dataclass(frozen=True)
class FaceClusterIdentitySuggestion:
    person_name: str
    support_count: int
    member_count: int
    mean_score: float
    min_score: float
    max_score: float


@dataclass(frozen=True)
class FaceClusteringComparisonResult:
    clusters_by_key: dict[str, dict[int, list[FaceClusterMember]]]
    membership_by_face_ref: dict[tuple[str, int], dict[str, dict[str, object]]]
    metrics_by_key: dict[str, dict[str, object]]
    explanations_by_key: dict[str, dict[int, ClusterExplanation]]
    identity_suggestions_by_key: dict[str, dict[int, FaceClusterIdentitySuggestion]]


@dataclass(frozen=True)
class FaceIndexRecord:
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    embedding: np.ndarray
    quality_status: str = "clean"
    quality_score: float = 1.0
    quality_reasons: tuple[str, ...] = ()
    hidden: bool = False


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
class FaceActionAuditEvent:
    event_id: int
    action: str
    target: str
    details: dict[str, object]
    reversible: bool
    created_at: str


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
    favorite: bool = False
    birth_date: str = ""
    hidden: bool = False


@dataclass(frozen=True)
class PersonPrototypeFace:
    person_name: str
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    quality_status: str = "clean"
    quality_score: float = 1.0
    quality_reasons: tuple[str, ...] = ()
    pinned: bool = False
    label_person_name: str = ""
    label_confidence: float = 0.0


@dataclass(frozen=True)
class FaceLabelAssignment:
    person_name: str
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    confidence: float
    proposal_id: int = 0
    source: str = ""
    created_at: str = ""
    pending: bool = False


@dataclass(frozen=True)
class NamedPhotoSummary:
    """Durable saved-name counts used by the global Names workspace."""

    person_name: str
    face_count: int
    photo_count: int


@dataclass(frozen=True)
class ManualLabelRecoveryResult:
    """Outcome of promoting legacy manual labels that were incorrectly queued."""

    promoted_count: int
    duplicate_count: int
    conflict_count: int


@dataclass(frozen=True)
class FaceLabelAcceptanceBatch:
    accepted: tuple[FaceLabelAssignment, ...]
    previous_labels: tuple[FaceLabelAssignment, ...]


@dataclass(frozen=True)
class IndexedFaceRecord:
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    embedding: np.ndarray
    person_name: str = ""
    label_confidence: float = 0.0
    quality_status: str = "clean"
    quality_score: float = 1.0
    quality_reasons: tuple[str, ...] = ()
    hidden: bool = False
    image_mtime: float = 0.0


@dataclass(frozen=True)
class FaceScanImageRecord:
    image_path: str
    mtime_ns: int
    file_size: int
    face_count: int
    image_width: int = 0
    image_height: int = 0
    indexed_at: str = ""


@dataclass(frozen=True)
class FaceFolderReviewImage:
    image_path: str
    review_status: str
    visible_faces: tuple[IndexedFaceRecord, ...]
    total_face_count: int
    hidden_face_count: int
    image_width: int = 0
    image_height: int = 0


@dataclass(frozen=True)
class FaceAlbumRecord:
    group_id: str
    group_kind: str
    image_path: str
    face_index: int
    face_bbox: tuple[int, int, int, int]
    face_confidence: float
    person_name: str = ""
    label_confidence: float = 0.0
    quality_status: str = "clean"
    quality_score: float = 1.0
    quality_reasons: tuple[str, ...] = ()
    hidden: bool = False
    display_name: str = ""
    source: str = ""
    created_at: str = ""


@dataclass(frozen=True)
class FaceAlbumGroupSummary:
    group_id: str
    group_kind: str
    title: str
    summary: str
    face_count: int
    photo_count: int
    person_name: str = ""


@dataclass(frozen=True)
class FaceAlbumGroupPage:
    items: tuple[FaceAlbumGroupSummary, ...]
    total_count: int
    offset: int
    next_offset: int | None


@dataclass(frozen=True)
class FaceAlbumMemberPage:
    group_id: str
    items: tuple[FaceAlbumRecord, ...]
    total_count: int
    offset: int
    next_offset: int | None


@dataclass(frozen=True)
class OrientedImageInfo:
    raw_width: int
    raw_height: int
    display_width: int
    display_height: int
    orientation: int = 1


MIN_INDEXED_FACE_SIDE_PX = 20
VISIBLE_FACE_SIDE_PX = 28
VISIBLE_FACE_AREA_PX = 900
FACE_QUALITY_METADATA_REVISION = 1
FACE_TINY_VISIBILITY_REVISION = 1
MAX_EAGER_FACE_REVIEW_MIGRATION_PATHS = 256
DEFAULT_FACE_QUALITY_PROFILE_ID = "balanced"
FACE_QUALITY_PROFILES: dict[FaceMode, dict[str, dict[str, object]]] = {
    "human": {
        "high_recall": {
            "min_face_side_px": 22,
            "min_face_area_px": 700,
            "review_confidence": 0.42,
            "reject_confidence": 0.18,
            "max_edge_clip_fraction": 0.18,
            "max_reject_edge_clip_fraction": 0.30,
            "min_bbox_aspect_ratio": 0.48,
            "max_bbox_aspect_ratio": 2.05,
            "reject_min_bbox_aspect_ratio": 0.38,
            "reject_max_bbox_aspect_ratio": 2.45,
            "min_sharpness_clean": 10.0,
            "min_sharpness_review": 4.0,
            "min_luma": 20.0,
            "max_luma": 240.0,
            "min_contrast": 10.0,
            "landmarks_policy": "prefer",
            "alignment_policy": "prefer",
            "duplicate_iou_threshold": 0.75,
        },
        "balanced": {
            "min_face_side_px": 28,
            "min_face_area_px": 900,
            "review_confidence": 0.55,
            "reject_confidence": 0.28,
            "max_edge_clip_fraction": 0.12,
            "max_reject_edge_clip_fraction": 0.26,
            "min_bbox_aspect_ratio": 0.58,
            "max_bbox_aspect_ratio": 1.95,
            "reject_min_bbox_aspect_ratio": 0.42,
            "reject_max_bbox_aspect_ratio": 2.35,
            "min_sharpness_clean": 18.0,
            "min_sharpness_review": 8.0,
            "min_luma": 28.0,
            "max_luma": 230.0,
            "min_contrast": 14.0,
            "landmarks_policy": "prefer",
            "alignment_policy": "prefer",
            "duplicate_iou_threshold": 0.70,
        },
        "high_precision": {
            "min_face_side_px": 34,
            "min_face_area_px": 1200,
            "review_confidence": 0.64,
            "reject_confidence": 0.36,
            "max_edge_clip_fraction": 0.08,
            "max_reject_edge_clip_fraction": 0.18,
            "min_bbox_aspect_ratio": 0.64,
            "max_bbox_aspect_ratio": 1.80,
            "reject_min_bbox_aspect_ratio": 0.50,
            "reject_max_bbox_aspect_ratio": 2.10,
            "min_sharpness_clean": 26.0,
            "min_sharpness_review": 12.0,
            "min_luma": 34.0,
            "max_luma": 220.0,
            "min_contrast": 18.0,
            "landmarks_policy": "require",
            "alignment_policy": "require",
            "duplicate_iou_threshold": 0.65,
        },
    },
    "dog": {
        "high_recall": {
            "min_face_side_px": 28,
            "min_face_area_px": 900,
            "review_confidence": 0.40,
            "reject_confidence": 0.16,
            "max_edge_clip_fraction": 0.20,
            "max_reject_edge_clip_fraction": 0.32,
            "min_bbox_aspect_ratio": 0.42,
            "max_bbox_aspect_ratio": 2.40,
            "reject_min_bbox_aspect_ratio": 0.32,
            "reject_max_bbox_aspect_ratio": 2.80,
            "min_sharpness_clean": 8.0,
            "min_sharpness_review": 3.0,
            "min_luma": 18.0,
            "max_luma": 242.0,
            "min_contrast": 8.0,
            "landmarks_policy": "off",
            "alignment_policy": "off",
            "duplicate_iou_threshold": 0.75,
        },
        "balanced": {
            "min_face_side_px": 34,
            "min_face_area_px": 1200,
            "review_confidence": 0.52,
            "reject_confidence": 0.26,
            "max_edge_clip_fraction": 0.14,
            "max_reject_edge_clip_fraction": 0.26,
            "min_bbox_aspect_ratio": 0.50,
            "max_bbox_aspect_ratio": 2.20,
            "reject_min_bbox_aspect_ratio": 0.38,
            "reject_max_bbox_aspect_ratio": 2.60,
            "min_sharpness_clean": 14.0,
            "min_sharpness_review": 6.0,
            "min_luma": 24.0,
            "max_luma": 235.0,
            "min_contrast": 12.0,
            "landmarks_policy": "off",
            "alignment_policy": "off",
            "duplicate_iou_threshold": 0.70,
        },
        "high_precision": {
            "min_face_side_px": 40,
            "min_face_area_px": 1500,
            "review_confidence": 0.60,
            "reject_confidence": 0.34,
            "max_edge_clip_fraction": 0.10,
            "max_reject_edge_clip_fraction": 0.20,
            "min_bbox_aspect_ratio": 0.56,
            "max_bbox_aspect_ratio": 2.00,
            "reject_min_bbox_aspect_ratio": 0.42,
            "reject_max_bbox_aspect_ratio": 2.30,
            "min_sharpness_clean": 20.0,
            "min_sharpness_review": 9.0,
            "min_luma": 30.0,
            "max_luma": 225.0,
            "min_contrast": 16.0,
            "landmarks_policy": "prefer",
            "alignment_policy": "prefer",
            "duplicate_iou_threshold": 0.65,
        },
    },
    "cat": {
        "high_recall": {
            "min_face_side_px": 28,
            "min_face_area_px": 900,
            "review_confidence": 0.40,
            "reject_confidence": 0.16,
            "max_edge_clip_fraction": 0.20,
            "max_reject_edge_clip_fraction": 0.32,
            "min_bbox_aspect_ratio": 0.42,
            "max_bbox_aspect_ratio": 2.35,
            "reject_min_bbox_aspect_ratio": 0.32,
            "reject_max_bbox_aspect_ratio": 2.75,
            "min_sharpness_clean": 8.0,
            "min_sharpness_review": 3.0,
            "min_luma": 18.0,
            "max_luma": 242.0,
            "min_contrast": 8.0,
            "landmarks_policy": "off",
            "alignment_policy": "off",
            "duplicate_iou_threshold": 0.75,
        },
        "balanced": {
            "min_face_side_px": 34,
            "min_face_area_px": 1200,
            "review_confidence": 0.52,
            "reject_confidence": 0.26,
            "max_edge_clip_fraction": 0.14,
            "max_reject_edge_clip_fraction": 0.26,
            "min_bbox_aspect_ratio": 0.50,
            "max_bbox_aspect_ratio": 2.15,
            "reject_min_bbox_aspect_ratio": 0.38,
            "reject_max_bbox_aspect_ratio": 2.55,
            "min_sharpness_clean": 14.0,
            "min_sharpness_review": 6.0,
            "min_luma": 24.0,
            "max_luma": 235.0,
            "min_contrast": 12.0,
            "landmarks_policy": "off",
            "alignment_policy": "off",
            "duplicate_iou_threshold": 0.70,
        },
        "high_precision": {
            "min_face_side_px": 40,
            "min_face_area_px": 1500,
            "review_confidence": 0.60,
            "reject_confidence": 0.34,
            "max_edge_clip_fraction": 0.10,
            "max_reject_edge_clip_fraction": 0.20,
            "min_bbox_aspect_ratio": 0.56,
            "max_bbox_aspect_ratio": 1.95,
            "reject_min_bbox_aspect_ratio": 0.42,
            "reject_max_bbox_aspect_ratio": 2.25,
            "min_sharpness_clean": 20.0,
            "min_sharpness_review": 9.0,
            "min_luma": 30.0,
            "max_luma": 225.0,
            "min_contrast": 16.0,
            "landmarks_policy": "prefer",
            "alignment_policy": "prefer",
            "duplicate_iou_threshold": 0.65,
        },
    },
}
_CANONICAL_5PT_LANDMARKS = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _face_bbox_metrics(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    return width, height, min(width, height), width * height


def _normalize_landmarks(
    landmarks: object,
    *,
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float], ...]:
    try:
        array = np.asarray(landmarks, dtype=np.float32)
    except Exception:
        return ()
    if array.size == 0:
        return ()
    if array.ndim == 1 and array.size % 2 == 0:
        array = array.reshape((-1, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        return ()
    normalized: list[tuple[float, float]] = []
    for point in array[:5]:
        x = float(np.clip(point[0], 0.0, float(max(0, image_width))))
        y = float(np.clip(point[1], 0.0, float(max(0, image_height))))
        normalized.append((x, y))
    return tuple(normalized)


def _aligned_face_crop(
    rgb_image: Image.Image,
    bbox: tuple[int, int, int, int],
    landmarks: tuple[tuple[float, float], ...] = (),
    *,
    output_size: tuple[int, int] = (112, 112),
) -> Image.Image:
    if len(landmarks) != 5 or cv2 is None:
        return rgb_image.crop(bbox).copy()
    source = np.asarray(rgb_image.convert("RGB"), dtype=np.uint8)
    src = np.asarray(landmarks, dtype=np.float32)
    dst = _CANONICAL_5PT_LANDMARKS.copy()
    if tuple(output_size) != (112, 112):
        dst[:, 0] *= float(output_size[0]) / 112.0
        dst[:, 1] *= float(output_size[1]) / 112.0
    try:
        matrix, _inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    except Exception:
        matrix = None
    if matrix is None:
        return rgb_image.crop(bbox).copy()
    try:
        aligned = cv2.warpAffine(
            source,
            matrix,
            tuple(int(v) for v in output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
    except Exception:
        return rgb_image.crop(bbox).copy()
    return Image.fromarray(aligned.astype(np.uint8), mode="RGB")


def _crop_detected_face(
    rgb_image: Image.Image,
    bbox: tuple[int, int, int, int],
    landmarks: tuple[tuple[float, float], ...] = (),
) -> Image.Image:
    return _aligned_face_crop(rgb_image, bbox, landmarks)


def _passes_index_bbox_filter(bbox: tuple[int, int, int, int]) -> bool:
    _width, _height, min_side, area = _face_bbox_metrics(bbox)
    return min_side >= MIN_INDEXED_FACE_SIDE_PX and area > 0


def _is_visible_face_bbox(bbox: tuple[int, int, int, int]) -> bool:
    _width, _height, min_side, area = _face_bbox_metrics(bbox)
    return min_side >= VISIBLE_FACE_SIDE_PX and area >= VISIBLE_FACE_AREA_PX


def _normalize_face_bbox(
    bbox: tuple[int, int, int, int] | list[int] | tuple[float, float, float, float] | list[float],
    *,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    except Exception:
        return None
    x1 = max(0, min(int(image_width), x1))
    y1 = max(0, min(int(image_height), y1))
    x2 = max(0, min(int(image_width), x2))
    y2 = max(0, min(int(image_height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    normalized = (x1, y1, x2, y2)
    if not _passes_index_bbox_filter(normalized):
        return None
    return normalized


def _face_crop_quality_metrics(crop: Image.Image) -> tuple[float, float, float]:
    try:
        gray = np.asarray(crop.convert("L"), dtype=np.float32)
    except Exception:
        return 0.0, 0.0, 0.0
    if gray.size == 0:
        return 0.0, 0.0, 0.0
    luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    if cv2 is not None:
        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    else:  # pragma: no cover - OpenCV is a project dependency
        grad_x = np.diff(gray, axis=1)
        grad_y = np.diff(gray, axis=0)
        sharpness = float(np.var(grad_x)) + float(np.var(grad_y))
    return sharpness, luma, contrast


def _bbox_edge_clip_fraction(
    bbox: tuple[int, int, int, int],
    *,
    image_width: int,
    image_height: int,
) -> float:
    x1, y1, x2, y2 = [int(value) for value in bbox]
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)
    left_clip = max(0, -x1)
    top_clip = max(0, -y1)
    right_clip = max(0, x2 - int(image_width))
    bottom_clip = max(0, y2 - int(image_height))
    clipped_px = left_clip + top_clip + right_clip + bottom_clip
    perimeter = max(1, (width * 2) + (height * 2))
    return float(clipped_px) / float(perimeter)


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = [int(value) for value in left]
    rx1, ry1, rx2, ry2 = [int(value) for value in right]
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union_area = max(1, left_area + right_area - inter_area)
    return float(inter_area) / float(union_area)

class FaceDetectionService:
    def __init__(
        self,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
        *,
        detector_id: str = BUILTIN_HUMAN_DETECTOR_ID,
        display_name: str = "MTCNN (built-in)",
        score_threshold: float = DEFAULT_FACE_SCORE_THRESHOLD,
        max_detections: int = DEFAULT_FACE_MAX_DETECTIONS,
    ) -> None:
        settings = get_settings()
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self.device = torch.device(self.execution_policy.torch_device if self.execution_policy.uses_cuda else "cpu")
        self._detector: MTCNN | None = None
        self.mode: FaceMode = "human"
        self.mode_label = face_mode_label("human")
        self.detector_id = normalize_face_component_id(detector_id, BUILTIN_HUMAN_DETECTOR_ID)
        self.display_name = str(display_name or "MTCNN (built-in)")
        self.model_name = "facenet-vggface2"
        self.score_threshold = float(max(0.0, score_threshold))
        self.max_detections = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))

    def configure_options(self, *, score_threshold: float | None = None, max_detections: int | None = None) -> None:
        if score_threshold is not None:
            self.score_threshold = float(max(0.0, score_threshold))
        if max_detections is not None:
            self.max_detections = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))

    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        detector = self._get_detector()
        try:
            rgb_image = _open_display_rgb_image(image_path)
        except Exception:
            return []

        landmarks = None
        try:
            boxes, probs, landmarks = detector.detect(rgb_image, landmarks=True)
        except TypeError:
            boxes, probs = detector.detect(rgb_image)
        if boxes is None or probs is None:
            return []
        faces = []
        for index, (box, prob) in enumerate(zip(boxes, probs)):
            if prob is None:
                continue
            if float(prob) < float(self.score_threshold):
                continue
            x1, y1, x2, y2 = [int(round(value)) for value in box.tolist()]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(rgb_image.width, x2)
            y2 = min(rgb_image.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if not _passes_index_bbox_filter((x1, y1, x2, y2)):
                continue
            face_landmarks = ()
            if landmarks is not None and index < len(landmarks):
                face_landmarks = _normalize_landmarks(
                    landmarks[index],
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                )
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(prob),
                    crop=_crop_detected_face(rgb_image, (x1, y1, x2, y2), face_landmarks),
                    landmarks=face_landmarks,
                )
            )
        faces.sort(key=lambda face: (-float(face.confidence), -((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))))
        return faces[: self.max_detections]

    def detect_query_face(self, image_path: str, query_bbox: tuple[int, int, int, int] | None = None) -> DetectedFace | None:
        if query_bbox is not None:
            try:
                rgb_image = _open_display_rgb_image(image_path)
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

    def is_ready(self) -> bool:
        return True

    def readiness_message(self) -> str:
        return f"Detector: {self.display_name}."


class FaceEmbeddingService:
    def __init__(
        self,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
        *,
        embedder_id: str = BUILTIN_HUMAN_EMBEDDER_ID,
        display_name: str = "VGGFace2 FaceNet (built-in)",
    ) -> None:
        settings = get_settings()
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        torch_home = Path(settings.cache_dir) / "torch"
        torch_home.mkdir(parents=True, exist_ok=True)
        os.environ["TORCH_HOME"] = str(torch_home)
        self.device = torch.device(self.execution_policy.torch_device if self.execution_policy.uses_cuda else "cpu")
        self._model: InceptionResnetV1 | None = None
        self.mode: FaceMode = "human"
        self.mode_label = face_mode_label("human")
        self.embedder_id = normalize_face_component_id(embedder_id, BUILTIN_HUMAN_EMBEDDER_ID)
        self.display_name = str(display_name or "VGGFace2 FaceNet (built-in)")
        self.model_name = "facenet-vggface2"
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
            if not ModelAssetService().local_cache_present("facenet"):
                raise RuntimeError(
                    "FaceNet VGGFace2 weights are not installed. Open Settings > Face Models, select FaceNet, "
                    "and choose Download / Install Selected. The download will appear in Jobs and can be cancelled."
                )
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        return self._model

    def is_ready(self) -> bool:
        return ModelAssetService().local_cache_present("facenet")

    def readiness_message(self) -> str:
        if self.is_ready():
            return f"Embedder: {self.display_name}."
        return "FaceNet weights are not installed. Open Settings > Face Models and install FaceNet."


class AnimalFaceDetectionService:
    def __init__(
        self,
        *,
        mode: str,
        model_root: str | Path | None,
        detector_id: str | None = None,
        detector_bundle: FaceDetectorBundle | None = None,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        self.mode = normalize_face_mode(mode)
        self.mode_label = face_mode_label(self.mode)
        self.model_root = str(model_root or "").strip()
        self.detector_id = normalize_face_component_id(detector_id, default_face_detector_id(self.model_root, self.mode))
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        settings = get_settings()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self._bundle: FaceDetectorBundle | None = detector_bundle
        self._session = None
        self.display_name = str(getattr(detector_bundle, "display_name", "") or self.detector_id or f"{self.mode}-onnx")
        self.backend_family = str(getattr(detector_bundle, "backend_family", "") or "onnx")
        self.model_name = str(getattr(detector_bundle, "model_name", "") or f"{self.mode}-{self.detector_id}")
        self.score_threshold = float(getattr(detector_bundle, "detector_score_threshold", 0.0) or 0.0)
        self.max_detections = max(1, int(getattr(detector_bundle, "detector_max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS))

    def is_ready(self) -> bool:
        try:
            bundle = self._get_bundle()
        except Exception:
            return False
        return bool(bundle.available and ort is not None)

    def readiness_message(self) -> str:
        try:
            bundle = self._get_bundle()
        except Exception as exc:
            return str(exc)
        if not bundle.available:
            return bundle.availability_message or "Detector bundle is not installed."
        if ort is None:
            return "onnxruntime is not installed."
        return f"Detector: {bundle.display_name} ({bundle.backend_family})."

    def configure_options(self, *, score_threshold: float | None = None, max_detections: int | None = None) -> None:
        if score_threshold is not None:
            self.score_threshold = float(max(0.0, score_threshold))
        if max_detections is not None:
            self.max_detections = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))

    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        bundle = self._get_bundle()
        session = self._get_session()
        try:
            rgb_image = _open_display_rgb_image(image_path)
        except Exception:
            return []
        if bundle.backend_family == "scrfd":
            faces = self._detect_scrfd(session, rgb_image, image_path)
            faces.sort(key=lambda face: (-float(face.confidence), -((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))))
            return faces[: self.max_detections]
        if bundle.backend_family == "yolo":
            faces = self._detect_yolo(session, rgb_image, image_path)
            faces.sort(key=lambda face: (-float(face.confidence), -((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))))
            return faces[: self.max_detections]
        return self._detect_generic(bundle, session, rgb_image, image_path)

    def _detect_generic(self, bundle: FaceDetectorBundle, session, rgb_image: Image.Image, image_path: str) -> list[DetectedFace]:
        input_name = _session_input_name(session, bundle.detector_input_name)
        output_names = [_session_output_name(session, bundle.detector_output_boxes_name)]
        score_output = str(bundle.detector_output_scores_name or "").strip()
        if score_output:
            output_names.append(_session_output_name(session, score_output))
        inputs = {
            input_name: _onnx_input_tensor(
                rgb_image,
                input_size=bundle.detector_input_size,
                mean=bundle.detector_mean,
                std=bundle.detector_std,
            )
        }
        outputs = session.run(output_names, inputs)
        raw_boxes = outputs[0] if outputs else np.zeros((0, 4), dtype=np.float32)
        raw_scores = outputs[1] if len(outputs) > 1 else None
        boxes = _flatten_detector_boxes(raw_boxes)
        boxes, scores = _flatten_detector_scores(raw_scores, boxes)
        faces: list[DetectedFace] = []
        width_scale = float(rgb_image.width)
        height_scale = float(rgb_image.height)
        for box, score in zip(boxes, scores):
            confidence = float(score)
            if confidence < float(self.score_threshold):
                continue
            x1, y1, x2, y2 = [float(value) for value in box[:4]]
            if bundle.detector_box_format == "yxyx":
                y1, x1, y2, x2 = [float(value) for value in box[:4]]
            if bundle.detector_boxes_normalized:
                x1 *= width_scale
                x2 *= width_scale
                y1 *= height_scale
                y2 *= height_scale
            bbox = (
                max(0, int(round(x1))),
                max(0, int(round(y1))),
                min(rgb_image.width, int(round(x2))),
                min(rgb_image.height, int(round(y2))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if not _passes_index_bbox_filter(bbox):
                continue
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=bbox,
                    confidence=confidence,
                    crop=_crop_detected_face(rgb_image, bbox),
                )
            )
        faces.sort(key=lambda face: (-float(face.confidence), -((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))))
        return faces[: self.max_detections]

    def _detect_scrfd(self, session, rgb_image: Image.Image, image_path: str) -> list[DetectedFace]:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for SCRFD face detection.")
        input_meta = session.get_inputs()[0]
        input_shape = list(input_meta.shape or ())
        static_size: tuple[int, int] | None = None
        if len(input_shape) >= 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
            static_size = (int(input_shape[3]), int(input_shape[2]))
        input_size = static_size or (640, 640)
        source = np.asarray(rgb_image, dtype=np.uint8)
        im_ratio = float(source.shape[0]) / max(1.0, float(source.shape[1]))
        model_ratio = float(input_size[1]) / max(1.0, float(input_size[0]))
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / max(1, source.shape[0])
        resized = cv2.resize(source, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        det_image = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_image,
            scalefactor=1.0 / 128.0,
            size=input_size,
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        output_names = _session_output_names(session)
        outputs = session.run(output_names, {_session_input_name(session, ""): blob})
        if not outputs:
            return []
        parsed_outputs: dict[int, dict[str, np.ndarray]] = {}
        for output_name, output_value in zip(output_names, outputs):
            match = re.search(r"(\d+)$", str(output_name or ""))
            if match is None:
                continue
            stride = int(match.group(1))
            array = np.asarray(output_value, dtype=np.float32)
            if array.ndim == 3 and array.shape[0] == 1:
                array = array[0]
            elif array.ndim == 1:
                array = array.reshape(-1, 1)
            output_key = ""
            lowered_name = str(output_name or "").lower()
            last_dim = int(array.shape[-1]) if array.ndim >= 2 else 0
            if "score" in lowered_name or last_dim == 1:
                output_key = "score"
            elif "box" in lowered_name or "bbox" in lowered_name or last_dim == 4:
                output_key = "bbox"
            elif "kps" in lowered_name or "lmk" in lowered_name or last_dim == 10:
                output_key = "kps"
            if output_key:
                parsed_outputs.setdefault(stride, {})[output_key] = array
        use_named_layout = bool(parsed_outputs) and all(
            "score" in group and "bbox" in group for group in parsed_outputs.values()
        )
        batched = len(outputs[0].shape) == 3
        use_kps = len(outputs) in {9, 15}
        if use_named_layout:
            feature_strides = sorted(parsed_outputs)
            feature_maps = len(feature_strides)
        elif len(outputs) in {6, 9}:
            feature_strides = [8, 16, 32]
            feature_maps = 3
        elif len(outputs) in {10, 15}:
            feature_strides = [8, 16, 32, 64, 128]
            feature_maps = 5
        else:
            raise RuntimeError(f"Unsupported SCRFD output layout with {len(outputs)} tensors.")
        center_cache: dict[tuple[int, int, int], np.ndarray] = {}
        score_rows: list[np.ndarray] = []
        box_rows: list[np.ndarray] = []
        keypoint_rows: list[np.ndarray] = []
        input_height = int(blob.shape[2])
        input_width = int(blob.shape[3])
        for index, stride in enumerate(feature_strides):
            if use_named_layout:
                group = parsed_outputs.get(int(stride), {})
                score_map = np.asarray(group.get("score"), dtype=np.float32).reshape(-1)
                bbox_map = np.asarray(group.get("bbox"), dtype=np.float32) * float(stride)
                if "kps" in group:
                    use_kps = True
                    kps_map = np.asarray(group.get("kps"), dtype=np.float32) * float(stride)
            else:
                if batched:
                    score_map = np.asarray(outputs[index][0], dtype=np.float32).reshape(-1)
                    bbox_map = np.asarray(outputs[index + feature_maps][0], dtype=np.float32) * float(stride)
                    if use_kps:
                        kps_map = np.asarray(outputs[index + (feature_maps * 2)][0], dtype=np.float32) * float(stride)
                else:
                    score_map = np.asarray(outputs[index], dtype=np.float32).reshape(-1)
                    bbox_map = np.asarray(outputs[index + feature_maps], dtype=np.float32) * float(stride)
                    if use_kps:
                        kps_map = np.asarray(outputs[index + (feature_maps * 2)], dtype=np.float32) * float(stride)
            height = input_height // stride
            width = input_width // stride
            locations = max(1, int(height * width))
            num_anchors = max(1, int(round(float(score_map.shape[0]) / float(locations))))
            cache_key = (height, width, stride)
            anchor_centers = center_cache.get(cache_key)
            if anchor_centers is None:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
                center_cache[cache_key] = anchor_centers
            positive = np.where(score_map >= float(self.score_threshold))[0]
            if positive.size == 0:
                continue
            decoded = np.stack(
                [
                    anchor_centers[:, 0] - bbox_map[:, 0],
                    anchor_centers[:, 1] - bbox_map[:, 1],
                    anchor_centers[:, 0] + bbox_map[:, 2],
                    anchor_centers[:, 1] + bbox_map[:, 3],
                ],
                axis=-1,
            )
            score_rows.append(score_map[positive].reshape(-1, 1))
            box_rows.append(decoded[positive])
            if use_kps:
                decoded_kps = _distance_to_keypoints(anchor_centers, kps_map)
                keypoint_rows.append(decoded_kps[positive])
        if not score_rows or not box_rows:
            return []
        scores = np.vstack(score_rows).astype(np.float32, copy=False).reshape(-1)
        boxes = (np.vstack(box_rows).astype(np.float32, copy=False) / max(det_scale, 1e-12))
        keypoints = (
            np.vstack(keypoint_rows).astype(np.float32, copy=False) / max(det_scale, 1e-12)
            if keypoint_rows
            else None
        )
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]
        if keypoints is not None:
            keypoints = keypoints[order]
        keep = _nms_xyxy(boxes, scores, 0.4)
        faces: list[DetectedFace] = []
        for result_index, (bbox_values, confidence) in enumerate(zip(boxes[keep], scores[keep])):
            bbox = (
                max(0, int(round(float(bbox_values[0])))),
                max(0, int(round(float(bbox_values[1])))),
                min(rgb_image.width, int(round(float(bbox_values[2])))),
                min(rgb_image.height, int(round(float(bbox_values[3])))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if not _passes_index_bbox_filter(bbox):
                continue
            face_landmarks = ()
            if keypoints is not None and result_index < len(keep):
                face_landmarks = _normalize_landmarks(
                    keypoints[keep[result_index]],
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                )
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=bbox,
                    confidence=float(confidence),
                    crop=_crop_detected_face(rgb_image, bbox, face_landmarks),
                    landmarks=face_landmarks,
                )
            )
        return faces

    def _detect_yolo(self, session, rgb_image: Image.Image, image_path: str) -> list[DetectedFace]:
        bundle = self._get_bundle()
        letterboxed, scale, pad = _letterbox_rgb_image(rgb_image, input_size=bundle.detector_input_size)
        tensor = np.transpose(letterboxed.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, ...]
        output_name = _session_output_name(session, bundle.detector_output_boxes_name)
        predictions = session.run([output_name], {_session_input_name(session, bundle.detector_input_name): tensor})[0]
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim == 3:
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 6:
            raise RuntimeError("YOLO face detector output must be a 2D array with box and confidence columns.")
        class_scores = predictions[:, 15:] if predictions.shape[1] > 15 else np.ones((predictions.shape[0], 1), dtype=np.float32)
        class_conf = class_scores.max(axis=1)
        scores = predictions[:, 4] * class_conf
        keep_mask = scores >= float(self.score_threshold)
        if not np.any(keep_mask):
            return []
        filtered = predictions[keep_mask]
        scores = scores[keep_mask]
        boxes = _xywh_to_xyxy(filtered[:, :4])
        raw_landmarks = filtered[:, 5:15] if filtered.shape[1] >= 15 else None
        keep = _nms_xyxy(boxes, scores, 0.45)
        if keep.size == 0:
            return []
        boxes = _scale_letterboxed_boxes(boxes[keep], scale=scale, pad=pad, width=rgb_image.width, height=rgb_image.height)
        scores = scores[keep]
        landmarks = raw_landmarks[keep] if raw_landmarks is not None else None
        faces: list[DetectedFace] = []
        for index, (bbox_values, confidence) in enumerate(zip(boxes, scores)):
            bbox = (
                max(0, int(round(float(bbox_values[0])))),
                max(0, int(round(float(bbox_values[1])))),
                min(rgb_image.width, int(round(float(bbox_values[2])))),
                min(rgb_image.height, int(round(float(bbox_values[3])))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if not _passes_index_bbox_filter(bbox):
                continue
            face_landmarks = ()
            if landmarks is not None and index < len(landmarks):
                points = np.asarray(landmarks[index], dtype=np.float32).reshape((-1, 2))
                points[:, 0] -= float(pad[0])
                points[:, 1] -= float(pad[1])
                points[:, 0] /= max(float(scale), 1e-12)
                points[:, 1] /= max(float(scale), 1e-12)
                face_landmarks = _normalize_landmarks(
                    points,
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                )
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=bbox,
                    confidence=float(confidence),
                    crop=_crop_detected_face(rgb_image, bbox, face_landmarks),
                    landmarks=face_landmarks,
                )
            )
        return faces

    def detect_query_face(self, image_path: str, query_bbox: tuple[int, int, int, int] | None = None) -> DetectedFace | None:
        if query_bbox is not None:
            try:
                rgb_image = _open_display_rgb_image(image_path)
            except Exception:
                return None
            return DetectedFace(
                image_path=image_path,
                bbox=query_bbox,
                confidence=1.0,
                crop=rgb_image.crop(query_bbox).copy(),
            )
        faces = self.detect_faces(image_path)
        if not faces:
            return None
        return max(faces, key=lambda face: (face.confidence, (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])))

    def _get_bundle(self) -> FaceDetectorBundle:
        if self._bundle is None:
            self._bundle = resolve_face_detector_bundle(self.model_root, self.mode, self.detector_id)
            self.display_name = self._bundle.display_name
            self.backend_family = self._bundle.backend_family
            self.model_name = self._bundle.model_name
            self.score_threshold = float(self._bundle.detector_score_threshold or 0.0)
            self.max_detections = max(1, int(self._bundle.detector_max_detections or DEFAULT_FACE_MAX_DETECTIONS))
        return self._bundle

    def _get_session(self):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")
        if self._session is None:
            bundle = self._get_bundle()
            if not bundle.available or bundle.detector_path is None:
                raise RuntimeError(bundle.availability_message or "Detector bundle is not installed.")
            provider = str(self.execution_policy.onnx_provider or "CPUExecutionProvider")
            if self.execution_policy.preferred_mode == "cuda" and provider != "CUDAExecutionProvider":
                raise RuntimeError(
                    self.execution_policy.error
                    or "CUDA was explicitly requested, but the face detector has no verified CUDA provider."
                )
            providers = [provider, "CPUExecutionProvider"]
            self._session = _create_onnx_session(
                Path(bundle.detector_path),
                providers,
                allow_cpu_fallback=self.execution_policy.preferred_mode == "auto",
            )
        return self._session


class AnimalFaceEmbeddingService:
    def __init__(
        self,
        *,
        mode: str,
        model_root: str | Path | None,
        embedder_id: str | None = None,
        embedder_bundle: FaceEmbedderBundle | None = None,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        self.mode = normalize_face_mode(mode)
        self.mode_label = face_mode_label(self.mode)
        self.model_root = str(model_root or "").strip()
        self.embedder_id = normalize_face_component_id(embedder_id, default_face_embedder_id(self.model_root, self.mode))
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        settings = get_settings()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self._bundle: FaceEmbedderBundle | None = embedder_bundle
        self._session = None
        self.display_name = str(getattr(embedder_bundle, "display_name", "") or self.embedder_id or f"{self.mode}-onnx")
        self.model_name = str(getattr(embedder_bundle, "model_name", "") or f"{self.mode}-{self.embedder_id}")

    def is_ready(self) -> bool:
        try:
            bundle = self._get_bundle()
        except Exception:
            return False
        return bool(bundle.available and ort is not None)

    def readiness_message(self) -> str:
        try:
            bundle = self._get_bundle()
        except Exception as exc:
            return str(exc)
        if not bundle.available:
            return bundle.availability_message or "Embedder bundle is not installed."
        if ort is None:
            return "onnxruntime is not installed."
        return f"Embedder: {bundle.display_name}."

    def embed_faces(self, face_crops: list[Image.Image]) -> np.ndarray:
        if not face_crops:
            return np.zeros((0, 512), dtype=np.float32)
        bundle = self._get_bundle()
        session = self._get_session()
        tensors = [
            _onnx_input_tensor(
                crop,
                input_size=bundle.embedder_input_size,
                mean=bundle.embedder_mean,
                std=bundle.embedder_std,
            )[0]
            for crop in face_crops
        ]
        batch = np.stack(tensors, axis=0).astype(np.float32)
        output_name = _session_output_name(session, bundle.embedder_output_name)
        input_name = _session_input_name(session, bundle.embedder_input_name)
        outputs = session.run([output_name], {input_name: batch})
        embeddings = np.asarray(outputs[0], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError("Animal face embedder output must be a 2D tensor.")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (embeddings / norms).astype(np.float32)

    def _get_bundle(self) -> FaceEmbedderBundle:
        if self._bundle is None:
            self._bundle = resolve_face_embedder_bundle(self.model_root, self.mode, self.embedder_id)
            self.display_name = self._bundle.display_name
            self.model_name = self._bundle.model_name
        return self._bundle

    def _get_session(self):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")
        if self._session is None:
            bundle = self._get_bundle()
            if not bundle.available or bundle.embedder_path is None:
                raise RuntimeError(bundle.availability_message or "Embedder bundle is not installed.")
            provider = str(self.execution_policy.onnx_provider or "CPUExecutionProvider")
            if self.execution_policy.preferred_mode == "cuda" and provider != "CUDAExecutionProvider":
                raise RuntimeError(
                    self.execution_policy.error
                    or "CUDA was explicitly requested, but the face embedder has no verified CUDA provider."
                )
            providers = [provider, "CPUExecutionProvider"]
            self._session = _create_onnx_session(
                Path(bundle.embedder_path),
                providers,
                allow_cpu_fallback=self.execution_policy.preferred_mode == "auto",
            )
        return self._session


class YuNetFaceDetectionService:
    def __init__(
        self,
        *,
        mode: str,
        model_root: str | Path | None,
        detector_id: str | None = None,
        detector_bundle: FaceDetectorBundle | None = None,
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
    ) -> None:
        self.mode = normalize_face_mode(mode)
        self.mode_label = face_mode_label(self.mode)
        self.model_root = str(model_root or "").strip()
        self.detector_id = normalize_face_component_id(detector_id, default_face_detector_id(self.model_root, self.mode))
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        settings = get_settings()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(settings.preferred_execution_mode)
        self._bundle: FaceDetectorBundle | None = detector_bundle
        self._detector = None
        self._detector_key: tuple[int, int, float, int] | None = None
        self.display_name = str(getattr(detector_bundle, "display_name", "") or self.detector_id or f"{self.mode}-yunet")
        self.backend_family = "yunet"
        self.model_name = str(getattr(detector_bundle, "model_name", "") or f"{self.mode}-{self.detector_id}")
        self.score_threshold = float(getattr(detector_bundle, "detector_score_threshold", 0.0) or 0.0)
        self.max_detections = max(1, int(getattr(detector_bundle, "detector_max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS))
        self.nms_threshold = 0.3

    def is_ready(self) -> bool:
        try:
            bundle = self._get_bundle()
        except Exception:
            return False
        return bool(bundle.available and cv2 is not None and hasattr(cv2, "FaceDetectorYN"))

    def readiness_message(self) -> str:
        try:
            bundle = self._get_bundle()
        except Exception as exc:
            return str(exc)
        if not bundle.available:
            return bundle.availability_message or "Detector bundle is not installed."
        if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
            return "OpenCV FaceDetectorYN is not available."
        return f"Detector: {bundle.display_name} (OpenCV YuNet)."

    def configure_options(self, *, score_threshold: float | None = None, max_detections: int | None = None) -> None:
        if score_threshold is not None:
            self.score_threshold = float(max(0.0, score_threshold))
        if max_detections is not None:
            self.max_detections = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))
        self._detector_key = None

    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        bundle = self._get_bundle()
        detector = self._get_detector()
        try:
            rgb_image = _open_display_rgb_image(image_path)
        except Exception:
            return []
        if hasattr(detector, "setInputSize"):
            try:
                detector.setInputSize((int(rgb_image.width), int(rgb_image.height)))
            except Exception:
                pass
        source = np.asarray(rgb_image, dtype=np.uint8)
        bgr_image = source[:, :, ::-1]
        try:
            retval, faces = detector.detect(bgr_image)
        except Exception:
            return []
        if retval is None or int(retval) <= 0 or faces is None:
            return []
        faces_array = np.asarray(faces, dtype=np.float32)
        if faces_array.size == 0:
            return []
        if faces_array.ndim == 1:
            faces_array = faces_array.reshape(1, -1)
        if faces_array.ndim == 3 and faces_array.shape[0] == 1:
            faces_array = faces_array[0]
        if faces_array.ndim != 2 or faces_array.shape[1] < 5:
            return []
        faces: list[DetectedFace] = []
        for row in faces_array:
            x, y, width, height = [float(value) for value in row[:4]]
            confidence = float(row[-1]) if row.size else 0.0
            if confidence < float(self.score_threshold):
                continue
            bbox = (
                max(0, int(round(x))),
                max(0, int(round(y))),
                min(rgb_image.width, int(round(x + width))),
                min(rgb_image.height, int(round(y + height))),
            )
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if not _passes_index_bbox_filter(bbox):
                continue
            face_landmarks = ()
            if row.size >= 14:
                face_landmarks = _normalize_landmarks(
                    np.asarray(row[4:14], dtype=np.float32).reshape((-1, 2)),
                    image_width=rgb_image.width,
                    image_height=rgb_image.height,
                )
            faces.append(
                DetectedFace(
                    image_path=image_path,
                    bbox=bbox,
                    confidence=confidence,
                    crop=_crop_detected_face(rgb_image, bbox, face_landmarks),
                    landmarks=face_landmarks,
                )
            )
        faces.sort(key=lambda face: (-float(face.confidence), -((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))))
        return faces[: self.max_detections]

    def detect_query_face(self, image_path: str, query_bbox: tuple[int, int, int, int] | None = None) -> DetectedFace | None:
        if query_bbox is not None:
            try:
                rgb_image = _open_display_rgb_image(image_path)
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

    def _get_bundle(self) -> FaceDetectorBundle:
        if self._bundle is None:
            self._bundle = resolve_face_detector_bundle(self.model_root, self.mode, self.detector_id)
            self.display_name = self._bundle.display_name
            self.backend_family = self._bundle.backend_family
            self.model_name = self._bundle.model_name
            self.score_threshold = float(self._bundle.detector_score_threshold or 0.0)
            self.max_detections = max(1, int(self._bundle.detector_max_detections or DEFAULT_FACE_MAX_DETECTIONS))
        return self._bundle

    def _get_detector(self):
        if cv2 is None or not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV FaceDetectorYN is not available.")
        bundle = self._get_bundle()
        input_size = tuple(int(value) for value in (bundle.detector_input_size or (320, 320)))
        key = (input_size[0], input_size[1], float(self.score_threshold), int(self.max_detections))
        if self._detector is None or self._detector_key != key:
            self._detector = cv2.FaceDetectorYN.create(
                str(bundle.detector_path or ""),
                "",
                input_size,
                float(self.score_threshold),
                float(self.nms_threshold),
                int(self.max_detections),
            )
            self._detector_key = key
        return self._detector


class FaceIndexService:
    def __init__(
        self,
        discovery_service: ImageDiscoveryService | None = None,
        detection_service: FaceDetectionService | None = None,
        embedding_service: FaceEmbeddingService | None = None,
        clustering_service: ClusteringService | None = None,
        *,
        mode: str = "human",
        execution_policy: ExecutionPolicy | None = None,
        runtime_service: RuntimeCapabilityService | None = None,
        model_root: str | Path | None = None,
        detector_id: str | None = None,
        embedder_id: str | None = None,
        detector_score_threshold: float | None = None,
        detector_max_detections: int | None = None,
        quality_profile_id: str | None = None,
        quality_thresholds: dict[str, object] | None = None,
        detector_policy: str | None = None,
        fallback_detector_id: str | None = None,
        verifier_mode: str | None = None,
        search_quality_min: str | None = None,
        cluster_quality_min: str | None = None,
        prototype_quality_min: str | None = None,
        recognition_min_score: float | None = None,
        auto_label_min_score: float | None = None,
        rerank_policy: str | None = None,
        rerank_top_n: int | None = None,
        animal_model_root: str | Path | None = None,
        db_path: str | Path | None = None,
        reset_db: bool = False,
    ) -> None:
        self.settings = get_settings()
        self.mode: FaceMode = normalize_face_mode(mode)
        self.mode_label = face_mode_label(self.mode)
        self.runtime_service = runtime_service or RuntimeCapabilityService()
        self.execution_policy = execution_policy or self.runtime_service.select_policy(self.settings.preferred_execution_mode)
        self.model_root = str(model_root or animal_model_root or "").strip()
        self.animal_model_root = self.model_root
        self.detector_id = normalize_face_component_id(detector_id, default_face_detector_id(self.model_root, self.mode))
        self.embedder_id = normalize_face_component_id(embedder_id, default_face_embedder_id(self.model_root, self.mode))
        self.detector_score_threshold = float(
            max(
                0.0,
                detector_score_threshold
                if detector_score_threshold is not None
                else DEFAULT_FACE_SCORE_THRESHOLD,
            )
        )
        self.detector_max_detections = max(
            1,
            int(
                detector_max_detections
                if detector_max_detections is not None
                else DEFAULT_FACE_MAX_DETECTIONS
            ),
        )
        self.quality_profile_id = str(quality_profile_id or default_face_quality_profile_id(self.mode)).strip().lower()
        self.quality_thresholds = face_quality_profile_config(self.mode, self.quality_profile_id)
        if quality_thresholds:
            self.quality_thresholds.update({str(key): value for key, value in dict(quality_thresholds).items()})
        self.detector_policy = str(detector_policy or "single").strip().lower()
        if self.detector_policy not in {item_id for item_id, _label in face_detector_policy_choices()}:
            self.detector_policy = "single"
        self.fallback_detector_id = normalize_face_component_id(
            fallback_detector_id,
            self.detector_id,
        )
        self.verifier_mode = str(verifier_mode or "off").strip().lower()
        if self.verifier_mode not in {item_id for item_id, _label in face_verifier_mode_choices()}:
            self.verifier_mode = "off"
        self.search_quality_min = str(search_quality_min or "clean").strip().lower()
        self.cluster_quality_min = str(cluster_quality_min or "clean").strip().lower()
        self.prototype_quality_min = str(prototype_quality_min or "clean").strip().lower()
        self.recognition_min_score = float(recognition_min_score if recognition_min_score is not None else 0.35)
        self.auto_label_min_score = float(auto_label_min_score if auto_label_min_score is not None else 0.72)
        self.rerank_policy = str(rerank_policy or "off").strip().lower()
        if self.rerank_policy not in {item_id for item_id, _label in face_rerank_policy_choices()}:
            self.rerank_policy = "off"
        self.rerank_top_n = max(1, int(rerank_top_n if rerank_top_n is not None else 25))
        self._last_pending_face_acceptance_batch: FaceLabelAcceptanceBatch | None = None
        self._recent_migrated_face_paths: set[str] = set()
        self._identity_cache_generation = 0
        self._cached_person_prototypes: list[PersonPrototype] | None = None
        self._cached_duplicate_warnings: dict[float, dict[str, tuple[tuple[str, float], ...]]] = {}
        self._cached_person_prototype_faces: dict[tuple[str, int, bool, bool], list[PersonPrototypeFace]] = {}
        self._face_recognition_enabled_cache: bool | None = None
        if db_path is None:
            if self.mode == "human":
                self.db_path = self.settings.cache_dir / "face_search_index.db"
            else:
                self.db_path = self.settings.cache_dir / f"face_search_{self.mode}.db"
        else:
            self.db_path = db_path
        self.discovery_service = discovery_service or ImageDiscoveryService()
        self._provided_detection_service = detection_service
        detector_bundle: FaceDetectorBundle | None = None
        embedder_bundle: FaceEmbedderBundle | None = None
        try:
            detector_bundle = resolve_face_detector_bundle(self.model_root, self.mode, self.detector_id)
        except Exception:
            detector_bundle = None
        try:
            embedder_bundle = resolve_face_embedder_bundle(self.model_root, self.mode, self.embedder_id)
        except Exception:
            embedder_bundle = None
        if detection_service is not None:
            self.detection_service = detection_service
        elif self.mode == "human" and (detector_bundle is None or detector_bundle.source_kind == "builtin"):
            self.detection_service = FaceDetectionService(
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
                detector_id=self.detector_id,
                display_name=str(getattr(detector_bundle, "display_name", "") or self._resolved_detector_display_name()),
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )
        elif detector_bundle is not None and detector_bundle.backend_family == "yunet":
            self.detection_service = YuNetFaceDetectionService(
                mode=self.mode,
                model_root=self.model_root,
                detector_id=self.detector_id,
                detector_bundle=detector_bundle,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
            )
            self.detection_service.configure_options(
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )
        else:
            self.detection_service = AnimalFaceDetectionService(
                mode=self.mode,
                model_root=self.model_root,
                detector_id=self.detector_id,
                detector_bundle=detector_bundle,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
            )
            self.detection_service.configure_options(
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )
        if embedding_service is not None:
            self.embedding_service = embedding_service
        elif self.mode == "human" and (embedder_bundle is None or embedder_bundle.source_kind == "builtin"):
            self.embedding_service = FaceEmbeddingService(
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
                embedder_id=self.embedder_id,
                display_name=str(getattr(embedder_bundle, "display_name", "") or self._resolved_embedder_display_name()),
            )
        else:
            self.embedding_service = AnimalFaceEmbeddingService(
                mode=self.mode,
                model_root=self.model_root,
                embedder_id=self.embedder_id,
                embedder_bundle=embedder_bundle,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
            )
        self.fallback_detection_service = self._build_detector_service(self.fallback_detector_id)
        self.clustering_service = clustering_service or ClusteringService()
        self.detector_display_name = str(
            getattr(self.detection_service, "display_name", "") or self._resolved_detector_display_name()
        )
        self.embedder_display_name = str(
            getattr(self.embedding_service, "display_name", "") or self._resolved_embedder_display_name()
        )
        self.model_name = str(
            getattr(self.embedding_service, "model_name", "")
            or getattr(self.detection_service, "model_name", "")
            or f"{self.mode}-{self.detector_id}-{self.embedder_id}"
        )
        if reset_db:
            try:
                if isinstance(self.db_path, Path) and self.db_path.exists():
                    self.db_path.unlink(missing_ok=True)
            except Exception:
                pass
        self._ann_index = None
        self._ann_records: list[tuple[str, int]] = []
        self._ann_fingerprint: tuple[int, int] | None = None
        self._init_db()

    def _resolved_detector_display_name(self) -> str:
        try:
            return str(resolve_face_detector_bundle(self.model_root, self.mode, self.detector_id).display_name)
        except Exception:
            if self.mode == "human":
                return "MTCNN (built-in)"
            return self.detector_id or f"{self.mode}-detector"

    def _resolved_embedder_display_name(self) -> str:
        try:
            return str(resolve_face_embedder_bundle(self.model_root, self.mode, self.embedder_id).display_name)
        except Exception:
            if self.mode == "human":
                return "VGGFace2 FaceNet (built-in)"
            return self.embedder_id or f"{self.mode}-embedder"

    def _build_detector_service(self, detector_id: str | None):
        target_id = normalize_face_component_id(detector_id, self.detector_id)
        if self._provided_detection_service is not None and target_id == self.detector_id:
            return self.detection_service
        detector_bundle: FaceDetectorBundle | None = None
        try:
            detector_bundle = resolve_face_detector_bundle(self.model_root, self.mode, target_id)
        except Exception:
            detector_bundle = None
        if self.mode == "human" and (detector_bundle is None or detector_bundle.source_kind == "builtin"):
            return FaceDetectionService(
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
                detector_id=target_id,
                display_name=str(getattr(detector_bundle, "display_name", "") or self._resolved_detector_display_name()),
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )
        if detector_bundle is not None and detector_bundle.backend_family == "yunet":
            service = YuNetFaceDetectionService(
                mode=self.mode,
                model_root=self.model_root,
                detector_id=target_id,
                detector_bundle=detector_bundle,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
            )
        else:
            service = AnimalFaceDetectionService(
                mode=self.mode,
                model_root=self.model_root,
                detector_id=target_id,
                detector_bundle=detector_bundle,
                execution_policy=self.execution_policy,
                runtime_service=self.runtime_service,
            )
        configure_fn = getattr(service, "configure_options", None)
        if callable(configure_fn):
            configure_fn(
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )
        return service

    def configure_detector_options(
        self,
        *,
        score_threshold: float | None = None,
        max_detections: int | None = None,
    ) -> None:
        if score_threshold is not None:
            self.detector_score_threshold = float(max(0.0, score_threshold))
        if max_detections is not None:
            self.detector_max_detections = max(1, int(max_detections or DEFAULT_FACE_MAX_DETECTIONS))
        configure_fn = getattr(self.detection_service, "configure_options", None)
        if callable(configure_fn):
            configure_fn(
                score_threshold=self.detector_score_threshold,
                max_detections=self.detector_max_detections,
            )

    def configure_quality_options(
        self,
        *,
        profile_id: str | None = None,
        thresholds: dict[str, object] | None = None,
    ) -> None:
        if profile_id:
            self.quality_profile_id = str(profile_id).strip().lower()
            self.quality_thresholds = face_quality_profile_config(self.mode, self.quality_profile_id)
        if thresholds:
            self.quality_thresholds.update({str(key): value for key, value in dict(thresholds).items()})

    def configure_cascade_options(
        self,
        *,
        detector_policy: str | None = None,
        fallback_detector_id: str | None = None,
        verifier_mode: str | None = None,
    ) -> None:
        if detector_policy is not None:
            target = str(detector_policy or "single").strip().lower()
            if target in {item_id for item_id, _label in face_detector_policy_choices()}:
                self.detector_policy = target
        if fallback_detector_id is not None:
            self.fallback_detector_id = normalize_face_component_id(fallback_detector_id, self.detector_id)
            self.fallback_detection_service = self._build_detector_service(self.fallback_detector_id)
        if verifier_mode is not None:
            target = str(verifier_mode or "off").strip().lower()
            if target in {item_id for item_id, _label in face_verifier_mode_choices()}:
                self.verifier_mode = target

    def configure_recognition_options(
        self,
        *,
        search_quality_min: str | None = None,
        cluster_quality_min: str | None = None,
        prototype_quality_min: str | None = None,
        recognition_min_score: float | None = None,
        auto_label_min_score: float | None = None,
        rerank_policy: str | None = None,
        rerank_top_n: int | None = None,
    ) -> None:
        if search_quality_min is not None:
            target = str(search_quality_min or "clean").strip().lower()
            if target in {item_id for item_id, _label in face_quality_gate_choices()}:
                self.search_quality_min = target
        if cluster_quality_min is not None:
            target = str(cluster_quality_min or "clean").strip().lower()
            if target in {item_id for item_id, _label in face_quality_gate_choices()}:
                self.cluster_quality_min = target
        if prototype_quality_min is not None:
            target = str(prototype_quality_min or "clean").strip().lower()
            if target in {item_id for item_id, _label in face_quality_gate_choices()}:
                self.prototype_quality_min = target
        if recognition_min_score is not None:
            self.recognition_min_score = float(max(-1.0, recognition_min_score))
        if auto_label_min_score is not None:
            self.auto_label_min_score = float(max(-1.0, auto_label_min_score))
        if rerank_policy is not None:
            target = str(rerank_policy or "off").strip().lower()
            if target in {item_id for item_id, _label in face_rerank_policy_choices()}:
                self.rerank_policy = target
        if rerank_top_n is not None:
            self.rerank_top_n = max(1, int(rerank_top_n))

    def copy_for_job(self, **overrides) -> "FaceIndexService":
        service = FaceIndexService(
            discovery_service=self.discovery_service,
            detection_service=self.detection_service,
            embedding_service=self.embedding_service,
            clustering_service=self.clustering_service,
            runtime_service=self.runtime_service,
            execution_policy=self.execution_policy,
            mode=self.mode,
            model_root=self.model_root,
            detector_id=self.detector_id,
            embedder_id=self.embedder_id,
            detector_score_threshold=self.detector_score_threshold,
            detector_max_detections=self.detector_max_detections,
            quality_profile_id=self.quality_profile_id,
            quality_thresholds=dict(self.quality_thresholds),
            detector_policy=str(overrides.get("detector_policy", self.detector_policy)),
            fallback_detector_id=str(overrides.get("fallback_detector_id", self.fallback_detector_id)),
            verifier_mode=str(overrides.get("verifier_mode", self.verifier_mode)),
            search_quality_min=self.search_quality_min,
            cluster_quality_min=self.cluster_quality_min,
            prototype_quality_min=self.prototype_quality_min,
            recognition_min_score=self.recognition_min_score,
            auto_label_min_score=self.auto_label_min_score,
            rerank_policy=self.rerank_policy,
            rerank_top_n=self.rerank_top_n,
            animal_model_root=self.animal_model_root,
            db_path=self.db_path,
        )
        return service

    def _quality_threshold(self, key: str, fallback: float) -> float:
        try:
            return float(self.quality_thresholds.get(str(key), fallback))
        except Exception:
            return float(fallback)

    def _quality_policy(self, key: str, fallback: str = "off") -> str:
        try:
            value = str(self.quality_thresholds.get(str(key), fallback) or fallback).strip().lower()
        except Exception:
            value = fallback
        if value not in {"off", "prefer", "require"}:
            return fallback
        return value

    def _assess_face_quality(
        self,
        *,
        image_path: str,
        bbox: tuple[int, int, int, int],
        confidence: float,
        crop: Image.Image | None = None,
        landmarks: tuple[tuple[float, float], ...] = (),
        image_width: int = 0,
        image_height: int = 0,
    ) -> tuple[str, float, tuple[str, ...]]:
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        box_width, box_height, min_side, area = _face_bbox_metrics(bbox)
        aspect = float(box_width) / max(1.0, float(box_height))
        edge_clip_fraction = _bbox_edge_clip_fraction(bbox, image_width=width, image_height=height)
        face_crop = crop
        if face_crop is None:
            try:
                face_crop = _open_display_rgb_image(image_path).crop(tuple(bbox))
            except Exception:
                face_crop = None
        sharpness = 0.0
        luma = 0.0
        contrast = 0.0
        if face_crop is not None:
            sharpness, luma, contrast = _face_crop_quality_metrics(face_crop)

        reasons: list[str] = []
        score = 1.0
        reject_confidence = self._quality_threshold("reject_confidence", 0.28)
        review_confidence = self._quality_threshold("review_confidence", 0.55)
        min_face_side_px = self._quality_threshold("min_face_side_px", float(VISIBLE_FACE_SIDE_PX))
        min_face_area_px = self._quality_threshold("min_face_area_px", float(VISIBLE_FACE_AREA_PX))
        max_edge_clip_fraction = self._quality_threshold("max_edge_clip_fraction", 0.12)
        max_reject_edge_clip_fraction = self._quality_threshold("max_reject_edge_clip_fraction", 0.26)
        min_bbox_aspect_ratio = self._quality_threshold("min_bbox_aspect_ratio", 0.58)
        max_bbox_aspect_ratio = self._quality_threshold("max_bbox_aspect_ratio", 1.95)
        reject_min_bbox_aspect_ratio = self._quality_threshold("reject_min_bbox_aspect_ratio", 0.42)
        reject_max_bbox_aspect_ratio = self._quality_threshold("reject_max_bbox_aspect_ratio", 2.35)
        min_sharpness_clean = self._quality_threshold("min_sharpness_clean", 18.0)
        min_sharpness_review = self._quality_threshold("min_sharpness_review", 8.0)
        min_luma = self._quality_threshold("min_luma", 28.0)
        max_luma = self._quality_threshold("max_luma", 230.0)
        min_contrast = self._quality_threshold("min_contrast", 14.0)

        if float(confidence) < reject_confidence:
            reasons.append("low_detector_score")
            score -= 0.35
        elif float(confidence) < review_confidence:
            reasons.append("review_confidence")
            score -= 0.18
        if float(min_side) < min_face_side_px:
            reasons.append("too_small")
            score -= 0.28
        if float(area) < min_face_area_px:
            reasons.append("low_area")
            score -= 0.18
        if float(edge_clip_fraction) > max_reject_edge_clip_fraction:
            reasons.append("edge_clipped")
            score -= 0.28
        elif float(edge_clip_fraction) > max_edge_clip_fraction:
            reasons.append("touches_edge")
            score -= 0.12
        if float(aspect) < reject_min_bbox_aspect_ratio or float(aspect) > reject_max_bbox_aspect_ratio:
            reasons.append("bad_aspect_ratio")
            score -= 0.25
        elif float(aspect) < min_bbox_aspect_ratio or float(aspect) > max_bbox_aspect_ratio:
            reasons.append("unusual_aspect_ratio")
            score -= 0.12
        if float(sharpness) < min_sharpness_review:
            reasons.append("blurred")
            score -= 0.22
        elif float(sharpness) < min_sharpness_clean:
            reasons.append("soft_focus")
            score -= 0.10
        if float(luma) < min_luma:
            reasons.append("too_dark")
            score -= 0.08
        elif float(luma) > max_luma:
            reasons.append("too_bright")
            score -= 0.08
        if float(contrast) < min_contrast:
            reasons.append("low_contrast")
            score -= 0.08
        landmarks_policy = self._quality_policy("landmarks_policy", "off")
        alignment_policy = self._quality_policy("alignment_policy", "off")
        if landmarks_policy != "off" and len(tuple(landmarks or ())) != 5:
            reasons.append("missing_landmarks")
            score -= 0.12 if landmarks_policy == "prefer" else 0.22
        elif len(tuple(landmarks or ())) == 5:
            xs = [float(point[0]) for point in landmarks]
            ys = [float(point[1]) for point in landmarks]
            spread_x = max(xs) - min(xs)
            spread_y = max(ys) - min(ys)
            if spread_x <= 0.0 or spread_y <= 0.0:
                reasons.append("invalid_landmarks")
                score -= 0.18
        if alignment_policy != "off" and len(tuple(landmarks or ())) != 5:
            reasons.append("alignment_unavailable")
            score -= 0.10 if alignment_policy == "prefer" else 0.18

        status = "clean"
        if any(
            reason in reasons
            for reason in {"low_detector_score", "too_small", "edge_clipped", "bad_aspect_ratio", "blurred"}
        ):
            status = "reject"
        elif reasons:
            status = "review"
        score = max(0.0, min(1.0, float(score)))
        return status, score, tuple(dict.fromkeys(reasons))

    def assess_face_bbox(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int],
        *,
        confidence: float = 1.0,
    ) -> tuple[str, float, tuple[str, ...]]:
        width, height = self._image_dimensions(str(image_path))
        normalized = _normalize_face_bbox(tuple(bbox), image_width=width, image_height=height)
        if normalized is None:
            return "reject", 0.0, ("invalid_bbox",)
        return self._assess_face_quality(
            image_path=str(image_path),
            bbox=normalized,
            confidence=float(confidence),
            image_width=width,
            image_height=height,
        )

    def _dedupe_detected_faces(self, faces: list[DetectedFace]) -> list[DetectedFace]:
        threshold = self._quality_threshold("duplicate_iou_threshold", 0.70)
        kept: list[DetectedFace] = []
        for face in sorted(list(faces or []), key=lambda item: float(item.confidence), reverse=True):
            if any(_bbox_iou(tuple(face.bbox), tuple(existing.bbox)) >= threshold for existing in kept):
                continue
            kept.append(face)
        return kept

    @staticmethod
    def _quality_rank(status: str) -> int:
        normalized = str(status or "clean").strip().lower()
        if normalized == "clean":
            return 3
        if normalized == "review":
            return 2
        return 1

    def _quality_allows(self, status: str, minimum: str) -> bool:
        target = str(minimum or "clean").strip().lower()
        if target not in {"clean", "review", "reject"}:
            target = "clean"
        return self._quality_rank(status) >= self._quality_rank(target)

    def _face_quality_from_detected(self, image_path: str, face: DetectedFace) -> tuple[str, float, tuple[str, ...]]:
        width, height = self._image_dimensions(image_path)
        return self._assess_face_quality(
            image_path=image_path,
            bbox=tuple(face.bbox),
            confidence=float(face.confidence or 0.0),
            crop=face.crop,
            landmarks=tuple(getattr(face, "landmarks", ()) or ()),
            image_width=width,
            image_height=height,
        )

    def _verify_detected_face(self, image_path: str, face: DetectedFace) -> bool:
        if self.verifier_mode == "off":
            return True
        status, score, reasons = self._face_quality_from_detected(image_path, face)
        if "not_face_verifier_failed" in reasons:
            return False
        if self.verifier_mode == "human_face_verifier":
            return status == "clean" or (status == "review" and score >= 0.45)
        if self.verifier_mode == "animal_face_verifier":
            return status != "reject" and score >= 0.35
        return True

    def _usable_detected_faces(self, image_path: str, faces: list[DetectedFace]) -> list[DetectedFace]:
        usable: list[DetectedFace] = []
        for face in list(faces or []):
            status, _score, _reasons = self._face_quality_from_detected(image_path, face)
            if status != "reject":
                usable.append(face)
        return usable

    def _consensus_faces(self, image_path: str, primary_faces: list[DetectedFace], fallback_faces: list[DetectedFace]) -> list[DetectedFace]:
        threshold = self._quality_threshold("duplicate_iou_threshold", 0.70)
        merged: list[DetectedFace] = []
        used_fallback: set[int] = set()
        review_confidence = max(0.0, self._quality_threshold("review_confidence", 0.55) - 0.01)
        for primary in list(primary_faces or []):
            matched_index = -1
            best_iou = 0.0
            for index, fallback in enumerate(list(fallback_faces or [])):
                if index in used_fallback:
                    continue
                overlap = _bbox_iou(tuple(primary.bbox), tuple(fallback.bbox))
                if overlap >= threshold and overlap > best_iou:
                    matched_index = index
                    best_iou = overlap
            if matched_index >= 0:
                used_fallback.add(matched_index)
                partner = fallback_faces[matched_index]
                merged.append(primary if float(primary.confidence or 0.0) >= float(partner.confidence or 0.0) else partner)
            else:
                merged.append(
                    DetectedFace(
                        image_path=primary.image_path,
                        bbox=tuple(primary.bbox),
                        confidence=min(float(primary.confidence or 0.0), review_confidence),
                        crop=primary.crop,
                        landmarks=tuple(getattr(primary, "landmarks", ()) or ()),
                    )
                )
        for index, fallback in enumerate(list(fallback_faces or [])):
            if index in used_fallback:
                continue
            merged.append(
                DetectedFace(
                    image_path=fallback.image_path,
                    bbox=tuple(fallback.bbox),
                    confidence=min(float(fallback.confidence or 0.0), review_confidence),
                    crop=fallback.crop,
                    landmarks=tuple(getattr(fallback, "landmarks", ()) or ()),
                )
            )
        return self._dedupe_detected_faces(merged)

    def _detect_faces_with_policy(self, image_path: str) -> list[DetectedFace]:
        primary_faces = self._dedupe_detected_faces(list(self.detection_service.detect_faces(str(image_path)) or []))
        fallback_service = getattr(self, "fallback_detection_service", None)
        fallback_id = normalize_face_component_id(getattr(fallback_service, "detector_id", self.fallback_detector_id), self.detector_id)
        if (
            self.detector_policy == "single"
            or fallback_service is None
            or fallback_service is self.detection_service
            or fallback_id == self.detector_id
        ):
            return primary_faces
        fallback_faces: list[DetectedFace] = []
        should_run_fallback = False
        if self.detector_policy == "rescue_on_no_faces":
            should_run_fallback = not self._usable_detected_faces(str(image_path), primary_faces)
        elif self.detector_policy == "rescue_on_low_confidence":
            best_confidence = max((float(face.confidence or 0.0) for face in primary_faces), default=0.0)
            should_run_fallback = best_confidence < self._quality_threshold("review_confidence", 0.55)
        elif self.detector_policy in {"consensus", "union_then_verify"}:
            should_run_fallback = True
        if should_run_fallback:
            fallback_faces = self._dedupe_detected_faces(list(fallback_service.detect_faces(str(image_path)) or []))
        if self.detector_policy == "rescue_on_no_faces":
            return primary_faces if primary_faces else fallback_faces
        if self.detector_policy == "rescue_on_low_confidence":
            if not fallback_faces:
                return primary_faces
            primary_best = max((float(face.confidence or 0.0) for face in primary_faces), default=0.0)
            fallback_best = max((float(face.confidence or 0.0) for face in fallback_faces), default=0.0)
            return fallback_faces if fallback_best > primary_best else primary_faces
        if self.detector_policy == "consensus":
            return self._consensus_faces(str(image_path), primary_faces, fallback_faces)
        if self.detector_policy == "union_then_verify":
            union_faces = self._dedupe_detected_faces(list(primary_faces) + list(fallback_faces))
            return [face for face in union_faces if self._verify_detected_face(str(image_path), face)]
        return primary_faces

    def detect_query_face(self, image_path: str, query_bbox: tuple[int, int, int, int] | None = None) -> DetectedFace | None:
        if query_bbox is not None:
            return self.detection_service.detect_query_face(str(image_path), query_bbox=query_bbox)
        faces = self._detect_faces_with_policy(str(image_path))
        if not faces:
            return None
        return max(
            faces,
            key=lambda face: (
                self._quality_rank(self._face_quality_from_detected(str(image_path), face)[0]),
                float(face.confidence or 0.0),
                (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
            ),
        )

    def is_ready(self) -> bool:
        detection_ready = True
        embedder_ready = True
        try:
            ready_fn = getattr(self.detection_service, "is_ready", None)
            if callable(ready_fn):
                detection_ready = bool(ready_fn())
        except Exception:
            detection_ready = False
        try:
            ready_fn = getattr(self.embedding_service, "is_ready", None)
            if callable(ready_fn):
                embedder_ready = bool(ready_fn())
        except Exception:
            embedder_ready = False
        return bool(detection_ready and embedder_ready)

    def readiness_message(self) -> str:
        detection_message = ""
        embedder_message = ""
        try:
            message_fn = getattr(self.detection_service, "readiness_message", None)
            if callable(message_fn):
                detection_message = str(message_fn() or "").strip()
        except Exception:
            detection_message = ""
        try:
            message_fn = getattr(self.embedding_service, "readiness_message", None)
            if callable(message_fn):
                embedder_message = str(message_fn() or "").strip()
        except Exception:
            embedder_message = ""
        if not self.is_ready():
            combined = " ".join(part for part in (detection_message, embedder_message) if part).strip()
            return combined or f"{self.mode_label} models are not ready."
        combined = " ".join(part for part in (detection_message, embedder_message) if part).strip()
        return combined or f"{self.mode_label} models are ready."

    def face_recognition_enabled(self) -> bool:
        if self._face_recognition_enabled_cache is not None:
            return bool(self._face_recognition_enabled_cache)
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT value FROM face_settings WHERE key='face_recognition_enabled'"
                ).fetchone()
        except Exception:
            return True
        if row is None:
            self._face_recognition_enabled_cache = True
            return True
        self._face_recognition_enabled_cache = str(row[0] or "1").strip().lower() not in {"0", "false", "no", "off"}
        return bool(self._face_recognition_enabled_cache)

    def set_face_recognition_enabled(self, enabled: bool) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO face_settings(key, value)
                VALUES ('face_recognition_enabled', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                ("1" if bool(enabled) else "0",),
            )
            self._record_action_audit(
                "set_face_recognition_enabled",
                target="face_recognition_enabled",
                details={"enabled": bool(enabled)},
                reversible=True,
                connection=connection,
            )
        self._face_recognition_enabled_cache = bool(enabled)

    def _record_action_audit(
        self,
        action: str,
        *,
        target: str = "",
        details: dict[str, object] | None = None,
        reversible: bool = False,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        payload = json.dumps(dict(details or {}), sort_keys=True)
        values = (str(action), str(target or ""), payload, 1 if bool(reversible) else 0)
        sql = """
            INSERT INTO face_action_audit(action, target, details_json, reversible)
            VALUES (?, ?, ?, ?)
        """
        if connection is not None:
            connection.execute(sql, values)
            return
        with self._connect() as audit_connection:
            audit_connection.execute(sql, values)

    def load_face_action_audit(self, *, limit: int = 50) -> list[FaceActionAuditEvent]:
        limit = max(1, int(limit))
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT event_id, action, target, details_json, reversible, created_at
                FROM face_action_audit
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        events: list[FaceActionAuditEvent] = []
        for row in rows:
            try:
                details = json.loads(row[3] or "{}")
            except Exception:
                details = {}
            events.append(
                FaceActionAuditEvent(
                    event_id=int(row[0]),
                    action=str(row[1] or ""),
                    target=str(row[2] or ""),
                    details=dict(details or {}),
                    reversible=bool(row[4]),
                    created_at=str(row[5] or ""),
                )
            )
        return events

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path))
        connection.execute("PRAGMA journal_mode=WAL;")
        return connection

    def _ann_index_path(self) -> Path:
        return Path(f"{self.db_path}.faiss")

    def _ann_meta_path(self) -> Path:
        return Path(f"{self.db_path}.faiss.json")

    def _db_fingerprint(self) -> tuple[int, int]:
        db_file = Path(self.db_path)
        if not db_file.exists():
            return (0, 0)
        stat = db_file.stat()
        return int(stat.st_mtime_ns), int(stat.st_size)

    def _invalidate_ann_index(self) -> None:
        self._ann_index = None
        self._ann_records = []
        self._ann_fingerprint = None
        for path in (self._ann_index_path(), self._ann_meta_path()):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass

    @staticmethod
    def _normalize_path_for_match(value: str) -> str:
        return normalize_scoped_path(value)

    @classmethod
    def _candidate_path_query_values(cls, candidate_paths: list[str] | None) -> list[str]:
        if not candidate_paths:
            return []
        deduped: dict[str, str] = {}
        for path in candidate_paths:
            raw_path = str(path or "").strip()
            if not raw_path:
                continue
            normalized = cls._normalize_path_for_match(raw_path)
            if normalized and normalized not in deduped:
                deduped[normalized] = raw_path
        return list(deduped.values())

    @staticmethod
    def _chunked_values(values: list[object], size: int = 900) -> list[list[object]]:
        chunk_size = max(1, int(size))
        return [list(values[index : index + chunk_size]) for index in range(0, len(values), chunk_size)]

    @staticmethod
    def _append_sql_condition(query: str, condition: str) -> str:
        base = str(query).rstrip()
        clause_match = re.search(r"\b(GROUP BY|ORDER BY|LIMIT|HAVING)\b", base, flags=re.IGNORECASE)
        if clause_match is None:
            head = base
            tail = ""
        else:
            head = base[: clause_match.start()].rstrip()
            tail = base[clause_match.start() :]
        if re.search(r"\bWHERE\b", head, flags=re.IGNORECASE):
            return f"{head} AND {condition}{tail}"
        return f"{head} WHERE {condition}{tail}"

    def _execute_scoped_query(
        self,
        connection: sqlite3.Connection,
        query: str,
        params: list[object] | tuple[object, ...] | None = None,
        *,
        image_column: str,
        candidate_paths: list[str] | None = None,
    ) -> list[tuple]:
        base_params = list(params or [])
        query_values = self._candidate_path_query_values(candidate_paths)
        if not query_values:
            return connection.execute(query, base_params).fetchall()
        rows: list[tuple] = []
        for chunk in self._chunked_values(list(query_values), size=900):
            placeholders = ", ".join("?" for _value in chunk)
            chunk_query = self._append_sql_condition(query, f"{image_column} IN ({placeholders})")
            rows.extend(connection.execute(chunk_query, [*base_params, *chunk]).fetchall())
        return rows

    def _invalidate_identity_caches(self) -> None:
        self._identity_cache_generation += 1
        self._cached_person_prototypes = None
        self._cached_duplicate_warnings = {}
        self._cached_person_prototype_faces = {}

    @classmethod
    def _matches_folder_prefix(cls, image_path: str, folder_prefix: str) -> bool:
        if not str(folder_prefix or "").strip():
            return True
        return path_is_within_scope(image_path, folder_prefix)

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

    @staticmethod
    def _face_is_visible(face_bbox: tuple[int, int, int, int], *, include_tiny_faces: bool) -> bool:
        return bool(include_tiny_faces or _is_visible_face_bbox(face_bbox))

    @staticmethod
    def _map_face_indexes(
        old_faces: list[tuple[int, tuple[int, int, int, int]]],
        new_records: list[FaceIndexRecord],
    ) -> dict[int, int]:
        old_by_index = [(int(index), tuple(bbox)) for index, bbox in old_faces]
        new_by_index = [(int(record.face_index), tuple(record.face_bbox)) for record in new_records]
        mapped: dict[int, int] = {}
        used_new: set[int] = set()
        for old_index, old_bbox in old_by_index:
            for new_index, new_bbox in new_by_index:
                if new_index in used_new:
                    continue
                if old_bbox == new_bbox:
                    mapped[old_index] = new_index
                    used_new.add(new_index)
                    break
        for old_index, old_bbox in old_by_index:
            if old_index in mapped:
                continue
            best_index = -1
            best_iou = 0.0
            for new_index, new_bbox in new_by_index:
                if new_index in used_new:
                    continue
                overlap = _bbox_iou(old_bbox, new_bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_index = new_index
            if best_index >= 0 and best_iou >= 0.45:
                mapped[old_index] = best_index
                used_new.add(best_index)
        return mapped

    @staticmethod
    def _empty_indexed_face_record(image_path: str) -> IndexedFaceRecord:
        return IndexedFaceRecord(
            image_path=str(image_path),
            face_index=-1,
            face_bbox=(0, 0, 0, 0),
            face_confidence=0.0,
            embedding=np.empty(0, dtype=np.float32),
        )

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
                    quality_status TEXT NOT NULL DEFAULT 'clean',
                    quality_score REAL NOT NULL DEFAULT 1.0,
                    quality_reasons_json TEXT NOT NULL DEFAULT '[]',
                    quality_revision INTEGER NOT NULL DEFAULT 0,
                    is_tiny INTEGER NOT NULL DEFAULT 0,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    PRIMARY KEY(image_path, face_index)
                )
                """
            )
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(face_index)").fetchall()
            }
            if "quality_status" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN quality_status TEXT NOT NULL DEFAULT 'clean'")
            if "quality_score" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN quality_score REAL NOT NULL DEFAULT 1.0")
            if "quality_reasons_json" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN quality_reasons_json TEXT NOT NULL DEFAULT '[]'")
            if "quality_revision" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN quality_revision INTEGER NOT NULL DEFAULT 0")
            if "hidden" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            if "is_tiny" not in columns:
                connection.execute("ALTER TABLE face_index ADD COLUMN is_tiny INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_scan_images (
                    image_path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    file_size INTEGER NOT NULL,
                    face_count INTEGER NOT NULL,
                    image_width INTEGER NOT NULL DEFAULT 0,
                    image_height INTEGER NOT NULL DEFAULT 0,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
            profile_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(person_profiles)").fetchall()
            }
            if "favorite" not in profile_columns:
                connection.execute("ALTER TABLE person_profiles ADD COLUMN favorite INTEGER NOT NULL DEFAULT 0")
            if "birth_date" not in profile_columns:
                connection.execute("ALTER TABLE person_profiles ADD COLUMN birth_date TEXT NOT NULL DEFAULT ''")
            if "hidden" not in profile_columns:
                connection.execute("ALTER TABLE person_profiles ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS person_prototype_faces (
                    person_name TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    sort_order INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(person_name, image_path, face_index)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rejected_face_labels (
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(image_path, face_index, person_name, source)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS rejected_face_labels_person_name_idx
                ON rejected_face_labels(person_name)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS person_prototype_faces_person_name_idx
                ON person_prototype_faces(person_name)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS person_prototype_faces_lookup_idx
                ON person_prototype_faces(person_name, pinned DESC, sort_order ASC, image_path ASC, face_index ASC)
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
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS face_labels_person_name_idx
                ON face_labels(person_name)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS face_labels_image_path_idx
                ON face_labels(image_path)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_face_labels (
                    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    face_index INTEGER NOT NULL,
                    person_name TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_path, face_index, source)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS pending_face_labels_person_name_idx
                ON pending_face_labels(person_name)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS pending_face_labels_image_path_idx
                ON pending_face_labels(image_path)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS pending_face_labels_created_at_idx
                ON pending_face_labels(created_at DESC, proposal_id DESC)
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS face_action_audit (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    target TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}',
                    reversible INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS face_action_audit_created_at_idx
                ON face_action_audit(created_at)
                """
            )
            self._backfill_face_tiny_visibility(connection)

    @staticmethod
    def _image_dimensions(image_path: str) -> tuple[int, int]:
        try:
            info = _read_oriented_image_info(image_path)
            return int(info.display_width), int(info.display_height)
        except Exception:
            return 0, 0

    def _load_existing_face_scan_rows(self, image_paths: list[str]) -> dict[str, tuple[int, int]]:
        normalized_paths = [str(path or "").strip() for path in image_paths if str(path or "").strip()]
        if not normalized_paths:
            return {}
        rows_by_path: dict[str, tuple[int, int]] = {}
        with self._connect() as connection:
            for start in range(0, len(normalized_paths), FACE_SCAN_ROW_PREFETCH_CHUNK):
                chunk = normalized_paths[start : start + FACE_SCAN_ROW_PREFETCH_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                query = f"SELECT image_path, mtime_ns, file_size FROM face_scan_images WHERE image_path IN ({placeholders})"
                for row in connection.execute(query, chunk).fetchall():
                    rows_by_path[str(row[0] or "")] = (int(row[1] or 0), int(row[2] or 0))
        return rows_by_path

    def _backfill_face_tiny_visibility(self, connection: sqlite3.Connection) -> None:
        revision_row = connection.execute(
            "SELECT value FROM face_settings WHERE key=?",
            ("face_index_tiny_visibility_revision",),
        ).fetchone()
        current_revision = int(revision_row[0] or 0) if revision_row is not None else 0
        if current_revision >= int(FACE_TINY_VISIBILITY_REVISION):
            return
        rows = connection.execute(
            "SELECT image_path, face_index, bbox_json FROM face_index"
        ).fetchall()
        updates: list[tuple[int, str, int]] = []
        for row in rows:
            try:
                bbox = tuple(int(value) for value in json.loads(row[2] or "[]"))
            except Exception:
                bbox = (0, 0, 0, 0)
            updates.append(
                (
                    0 if _is_visible_face_bbox(bbox) else 1,
                    str(row[0] or ""),
                    int(row[1] or 0),
                )
            )
        if updates:
            connection.executemany(
                "UPDATE face_index SET is_tiny=? WHERE image_path=? AND face_index=?",
                updates,
            )
        connection.execute(
            """
            INSERT INTO face_settings(key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            ("face_index_tiny_visibility_revision", str(int(FACE_TINY_VISIBILITY_REVISION))),
        )

    def _replace_image_records(
        self,
        image_path: str,
        records: list[FaceIndexRecord],
        *,
        mtime_ns: int,
        file_size: int,
        image_width: int = 0,
        image_height: int = 0,
    ) -> None:
        with self._connect() as connection:
            old_face_rows = [
                (int(row[0]), tuple(json.loads(row[1] or "[]")))
                for row in connection.execute(
                    "SELECT face_index, bbox_json FROM face_index WHERE image_path=? ORDER BY face_index",
                    (str(image_path),),
                ).fetchall()
            ]
            index_map = self._map_face_indexes(old_face_rows, list(records or []))
            label_rows = [
                (int(row[0]), str(row[1]), float(row[2]))
                for row in connection.execute(
                    "SELECT face_index, person_name, confidence FROM face_labels WHERE image_path=?",
                    (str(image_path),),
                ).fetchall()
            ]
            pending_rows = [
                (int(row[0]), str(row[1]), float(row[2]), str(row[3]), str(row[4] or ""))
                for row in connection.execute(
                    """
                    SELECT face_index, person_name, confidence, source, created_at
                    FROM pending_face_labels
                    WHERE image_path=?
                    """,
                    (str(image_path),),
                ).fetchall()
            ]
            prototype_rows = [
                (str(row[0]), int(row[1]), int(row[2]), int(row[3]))
                for row in connection.execute(
                    """
                    SELECT person_name, face_index, pinned, sort_order
                    FROM person_prototype_faces
                    WHERE image_path=?
                    """,
                    (str(image_path),),
                ).fetchall()
            ]
            rejected_rows = [
                (int(row[0]), str(row[1]), str(row[2]), str(row[3] or ""))
                for row in connection.execute(
                    """
                    SELECT face_index, person_name, source, created_at
                    FROM rejected_face_labels
                    WHERE image_path=?
                    """,
                    (str(image_path),),
                ).fetchall()
            ]
            connection.execute("DELETE FROM face_index WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM face_labels WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM pending_face_labels WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM person_prototype_faces WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM rejected_face_labels WHERE image_path=?", (image_path,))
            if records:
                connection.executemany(
                    """
                    INSERT INTO face_index(
                        image_path,
                        face_index,
                        bbox_json,
                        face_confidence,
                        embedding,
                        quality_status,
                        quality_score,
                        quality_reasons_json,
                        quality_revision,
                        hidden,
                        is_tiny,
                        mtime_ns,
                        file_size
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record.image_path,
                            record.face_index,
                            json.dumps(list(record.face_bbox)),
                            record.face_confidence,
                            np.asarray(record.embedding, dtype=np.float32).tobytes(),
                            str(record.quality_status or "clean"),
                            float(record.quality_score),
                            json.dumps(list(record.quality_reasons or ())),
                            int(FACE_QUALITY_METADATA_REVISION),
                            1 if bool(getattr(record, "hidden", False)) else 0,
                            0 if _is_visible_face_bbox(tuple(record.face_bbox)) else 1,
                            mtime_ns,
                            file_size,
                        )
                        for record in records
                    ],
                )
            if index_map:
                mapped_labels = [
                    (str(image_path), int(index_map[old_index]), person_name, confidence)
                    for old_index, person_name, confidence in label_rows
                    if old_index in index_map
                ]
                if mapped_labels:
                    connection.executemany(
                        """
                        INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(image_path, face_index) DO UPDATE SET
                            person_name=excluded.person_name,
                            confidence=excluded.confidence
                        """,
                        mapped_labels,
                    )
                mapped_pending = [
                    (str(image_path), int(index_map[old_index]), person_name, confidence, source, created_at)
                    for old_index, person_name, confidence, source, created_at in pending_rows
                    if old_index in index_map
                ]
                if mapped_pending:
                    connection.executemany(
                        """
                        INSERT INTO pending_face_labels(image_path, face_index, person_name, confidence, source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(image_path, face_index, source) DO UPDATE SET
                            person_name=excluded.person_name,
                            confidence=excluded.confidence,
                            created_at=excluded.created_at
                        """,
                        mapped_pending,
                    )
                mapped_prototypes = [
                    (person_name, str(image_path), int(index_map[old_index]), pinned, sort_order)
                    for person_name, old_index, pinned, sort_order in prototype_rows
                    if old_index in index_map
                ]
                if mapped_prototypes:
                    connection.executemany(
                        """
                        INSERT INTO person_prototype_faces(person_name, image_path, face_index, pinned, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(person_name, image_path, face_index) DO UPDATE SET
                            pinned=excluded.pinned,
                            sort_order=excluded.sort_order
                        """,
                        mapped_prototypes,
                    )
                mapped_rejections = [
                    (str(image_path), int(index_map[old_index]), person_name, source, created_at)
                    for old_index, person_name, source, created_at in rejected_rows
                    if old_index in index_map
                ]
                if mapped_rejections:
                    connection.executemany(
                        """
                        INSERT INTO rejected_face_labels(image_path, face_index, person_name, source, created_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(image_path, face_index, person_name, source) DO UPDATE SET
                            created_at=excluded.created_at
                        """,
                        mapped_rejections,
                    )
            connection.execute(
                """
                INSERT INTO face_scan_images(image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(image_path) DO UPDATE SET
                    mtime_ns=excluded.mtime_ns,
                    file_size=excluded.file_size,
                    face_count=excluded.face_count,
                    image_width=excluded.image_width,
                    image_height=excluded.image_height,
                    indexed_at=CURRENT_TIMESTAMP
                """,
                (
                    str(image_path),
                    int(mtime_ns),
                    int(file_size),
                    int(len(records)),
                    max(0, int(image_width)),
                    max(0, int(image_height)),
                ),
            )
        self._invalidate_identity_caches()
        self._invalidate_ann_index()

    def _assess_face_records(
        self,
        image_path: str,
        records: list[FaceIndexRecord],
        *,
        image_width: int,
        image_height: int,
    ) -> list[FaceIndexRecord]:
        if not records:
            return []
        assessed: list[FaceIndexRecord] = []
        rgb_image: Image.Image | None = None
        try:
            rgb_image = _open_display_rgb_image(image_path)
        except Exception:
            rgb_image = None
        for record in records:
            crop = None
            if rgb_image is not None:
                try:
                    crop = rgb_image.crop(tuple(record.face_bbox))
                except Exception:
                    crop = None
            status, score, reasons = self._assess_face_quality(
                image_path=image_path,
                bbox=tuple(record.face_bbox),
                confidence=float(record.face_confidence or 0.0),
                crop=crop,
                image_width=image_width,
                image_height=image_height,
            )
            assessed.append(
                FaceIndexRecord(
                    image_path=str(record.image_path),
                    face_index=int(record.face_index),
                    face_bbox=tuple(record.face_bbox),
                    face_confidence=float(record.face_confidence or 0.0),
                    embedding=np.asarray(record.embedding, dtype=np.float32).copy(),
                    quality_status=str(status or "clean"),
                    quality_score=float(score),
                    quality_reasons=tuple(str(value) for value in reasons or ()),
                )
            )
        if rgb_image is not None:
            try:
                rgb_image.close()
            except Exception:
                pass
        return assessed

    def _backfill_quality_metadata_for_paths(self, image_paths: list[str]) -> None:
        targets = [str(path or "").strip() for path in image_paths if str(path or "").strip()]
        if not targets:
            return
        rows: list[tuple] = []
        with self._connect() as connection:
            for chunk in self._chunked_values(targets, size=900):
                placeholders = ", ".join(["?"] * len(chunk))
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT image_path, face_index, bbox_json, face_confidence
                        FROM face_index
                        WHERE image_path IN ({placeholders}) AND quality_revision < ?
                        ORDER BY image_path, face_index
                        """,
                        [*chunk, int(FACE_QUALITY_METADATA_REVISION)],
                    ).fetchall()
                )
        if not rows:
            return
        updates: list[tuple[str, float, str, int, str, int]] = []
        dimensions: dict[str, tuple[int, int]] = {}
        for row in rows:
            image_path = str(row[0])
            if image_path not in dimensions:
                dimensions[image_path] = self._image_dimensions(image_path)
            bbox = tuple(json.loads(row[2]))
            status, score, reasons = self.assess_face_bbox(
                image_path,
                bbox,
                confidence=float(row[3] or 0.0),
            )
            updates.append(
                (
                    str(status or "clean"),
                    float(score),
                    json.dumps(list(reasons or ())),
                    int(FACE_QUALITY_METADATA_REVISION),
                    image_path,
                    int(row[1]),
                )
            )
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE face_index
                SET quality_status=?, quality_score=?, quality_reasons_json=?, quality_revision=?
                WHERE image_path=? AND face_index=?
                """,
                updates,
            )

    def _build_ann_index(self, *, cancel_check=None) -> tuple[object | None, list[tuple[str, int]], tuple[int, int]]:
        from infra.cancel import raise_if_cancelled

        raise_if_cancelled(cancel_check)
        if faiss is None:
            return None, [], self._db_fingerprint()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path, face_index, embedding FROM face_index ORDER BY image_path, face_index"
            ).fetchall()
        if not rows:
            return None, [], self._db_fingerprint()
        raise_if_cancelled(cancel_check)
        vectors = np.vstack(
            [np.frombuffer(row[2], dtype=np.float32).copy() for row in rows]
        ).astype(np.float32, copy=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        index = faiss.IndexHNSWFlat(int(vectors.shape[1]), 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 128
        raise_if_cancelled(cancel_check)
        index.add(vectors)
        refs = [(str(row[0]), int(row[1])) for row in rows]
        fingerprint = self._db_fingerprint()
        try:
            raise_if_cancelled(cancel_check)
            atomic_write_with(
                self._ann_index_path(),
                lambda temporary: faiss.write_index(index, str(temporary)),
            )
            raise_if_cancelled(cancel_check)
            atomic_write_text(
                self._ann_meta_path(),
                json.dumps(
                    {
                        "fingerprint": [int(fingerprint[0]), int(fingerprint[1])],
                        "records": [[image_path, int(face_index)] for image_path, face_index in refs],
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
        except Exception:
            pass
        return index, refs, fingerprint

    def build_ann_index(self, *, cancel_check=None) -> dict[str, int]:
        index, refs, fingerprint = self._build_ann_index(cancel_check=cancel_check)
        self._ann_index = index
        self._ann_records = list(refs)
        self._ann_fingerprint = fingerprint
        return {"faces_indexed": int(len(refs)), "ann_available": int(index is not None)}

    def _load_ann_index(self) -> tuple[object | None, list[tuple[str, int]], tuple[int, int]]:
        fingerprint = self._db_fingerprint()
        if self._ann_fingerprint == fingerprint and self._ann_index is not None:
            return self._ann_index, list(self._ann_records), fingerprint
        index_path = self._ann_index_path()
        meta_path = self._ann_meta_path()
        if faiss is not None and index_path.exists() and meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                saved_fingerprint = tuple(int(value) for value in payload.get("fingerprint", []))
                refs = [
                    (str(item[0]), int(item[1]))
                    for item in payload.get("records", [])
                    if isinstance(item, list) and len(item) == 2
                ]
                if saved_fingerprint == fingerprint and refs:
                    index = faiss.read_index(str(index_path))
                    if int(getattr(index, "ntotal", -1)) != len(refs):
                        raise ValueError("Face ANN index record count does not match its metadata.")
                    if hasattr(index, "hnsw"):
                        index.hnsw.efSearch = 128
                    self._ann_index = index
                    self._ann_records = list(refs)
                    self._ann_fingerprint = fingerprint
                    return index, list(refs), fingerprint
            except Exception:
                pass
        self._ann_index = None
        self._ann_records = []
        self._ann_fingerprint = fingerprint
        return None, [], fingerprint

    def _search_by_embedding_bruteforce(
        self,
        query_embedding: np.ndarray,
        *,
        min_face_score: float,
        top_k: int,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        exclude: set[tuple[str, int]] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceSearchResult]:
        hits: list[tuple[FaceSearchResult, IndexedFaceRecord]] = []
        exclude = exclude or set()
        folder_prefix = str(folder_prefix or "").strip()
        for record in self.load_all_records(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
        ):
            raise_if_cancelled(cancel_check)
            if (record.image_path, int(record.face_index)) in exclude:
                continue
            if not self._quality_allows(record.quality_status, self.search_quality_min):
                continue
            record_embedding = np.asarray(record.embedding, dtype=np.float32)
            record_norm = float(np.linalg.norm(record_embedding))
            if record_norm > 1e-12:
                record_embedding = record_embedding / record_norm
            score = float(np.dot(query_embedding, record_embedding))
            if score < min_face_score:
                continue
            hits.append(
                (
                    FaceSearchResult(
                        image_path=record.image_path,
                        face_index=int(record.face_index),
                        score=round(score, 6),
                        phash_distance=-1,
                        face_bbox=record.face_bbox,
                        face_confidence=round(record.face_confidence, 6),
                        model_name=self.model_name,
                        match_reason=f"{self.mode_label.lower()} face embedding cosine similarity",
                        person_name=str(record.person_name or ""),
                        quality_status=str(record.quality_status or "clean"),
                        quality_reasons=tuple(record.quality_reasons or ()),
                        hidden=bool(record.hidden),
                        image_mtime=float(record.image_mtime or 0.0),
                    ),
                    record,
                )
            )
        return self._finalize_search_hits(hits, top_k=top_k)

    def _finalize_search_hits(
        self,
        hits: list[tuple[FaceSearchResult, IndexedFaceRecord]],
        *,
        top_k: int,
    ) -> list[FaceSearchResult]:
        sorted_hits = sorted(
            list(hits or []),
            key=lambda item: (-float(item[0].score), item[0].image_path, item[0].face_bbox),
        )
        if self.rerank_policy == "quality_score" and sorted_hits:
            rerank_top_n = min(max(1, int(self.rerank_top_n)), len(sorted_hits))
            reranked = sorted(
                sorted_hits[:rerank_top_n],
                key=lambda item: (-float(item[1].quality_score), -float(item[0].score), item[0].image_path, item[0].face_bbox),
            )
            sorted_hits = reranked + sorted_hits[rerank_top_n:]
        return [result for result, _record in sorted_hits[: max(1, top_k)]]

    def index_directory(
        self,
        directory: str,
        recursive: bool = True,
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> dict[str, int]:
        LOGGER.info(
            "FaceIndexService index_directory mode=%s db=%s directory=%s recursive=%s",
            self.mode,
            self.db_path,
            directory,
            recursive,
        )
        image_paths = self.discovery_service.discover(directory, recursive=recursive)
        metrics = self.index_paths(image_paths, progress_callback=progress_callback, cancel_check=cancel_check)
        LOGGER.info(
            "FaceIndexService index_directory_complete mode=%s db=%s directory=%s images=%s faces=%s",
            self.mode,
            self.db_path,
            directory,
            int(metrics.get("images_done", 0)),
            int(metrics.get("faces_indexed", 0)),
        )
        return metrics

    def index_paths(
        self,
        image_paths: list[str],
        *,
        progress_callback=None,
        cancel_check=None,
        force: bool = False,
    ) -> dict[str, int]:
        from infra.cancel import raise_if_cancelled

        total_faces = 0
        done = 0
        skipped_unchanged = 0
        skipped_disabled = 0
        total = len(image_paths)
        progress_step = max(1, (max(1, total) + MAX_FACE_INDEX_PROGRESS_UPDATES - 1) // MAX_FACE_INDEX_PROGRESS_UPDATES)
        last_progress_done = -1
        last_progress_s = monotonic()

        def _emit_progress(status: str, *, force: bool = False) -> None:
            nonlocal last_progress_done, last_progress_s
            if progress_callback is None:
                return
            now = monotonic()
            if not force and total > 0 and done >= total:
                return
            if not force and total > 0 and last_progress_done >= 0 and done < total:
                if (done - last_progress_done) < progress_step and (now - last_progress_s) < FACE_INDEX_PROGRESS_MIN_INTERVAL_S:
                    return
            progress_callback(
                100 if total <= 0 else int(done * 100 / max(1, total)),
                str(status),
            )
            last_progress_done = done
            last_progress_s = now

        if not self.face_recognition_enabled():
            for _image_path in image_paths:
                raise_if_cancelled(cancel_check)
                done += 1
                skipped_disabled += 1
                _emit_progress(f"{done}/{total} images, face recognition disabled")
            _emit_progress(f"{done}/{total} images skipped because face recognition is disabled", force=True)
            return {
                "images_done": int(done),
                "images_total": int(total),
                "faces_indexed": 0,
                "skipped_unchanged": 0,
                "skipped_recognition_disabled": int(skipped_disabled),
            }
        existing_scan_rows = self._load_existing_face_scan_rows(image_paths)
        for image_path in image_paths:
            raise_if_cancelled(cancel_check)
            try:
                stat = Path(image_path).stat()
            except FileNotFoundError:
                continue
            scan_row = existing_scan_rows.get(str(image_path))
            if (
                not bool(force)
                and scan_row is not None
                and int(scan_row[0]) == int(stat.st_mtime_ns)
                and int(scan_row[1]) == int(stat.st_size)
            ):
                skipped_unchanged += 1
                done += 1
                _emit_progress(f"{done}/{total} images, skipped {skipped_unchanged} unchanged, extracted {total_faces} faces")
                continue
            image_width, image_height = self._image_dimensions(image_path)
            faces = self._detect_faces_with_policy(image_path)
            if not faces:
                self._replace_image_records(
                    image_path,
                    [],
                    mtime_ns=stat.st_mtime_ns,
                    file_size=stat.st_size,
                    image_width=image_width,
                    image_height=image_height,
                )
                done += 1
                _emit_progress(f"{done}/{total} images, extracted {total_faces} faces")
                continue
            embeddings = self.embedding_service.embed_faces([face.crop for face in faces])
            records = []
            for face_index, face in enumerate(faces):
                status, score, reasons = self._assess_face_quality(
                    image_path=image_path,
                    bbox=tuple(face.bbox),
                    confidence=float(face.confidence or 0.0),
                    crop=face.crop,
                    landmarks=tuple(getattr(face, "landmarks", ()) or ()),
                    image_width=image_width,
                    image_height=image_height,
                )
                records.append(
                    FaceIndexRecord(
                        image_path=image_path,
                        face_index=face_index,
                        face_bbox=face.bbox,
                        face_confidence=face.confidence,
                        embedding=embeddings[face_index],
                        quality_status=status,
                        quality_score=score,
                        quality_reasons=reasons,
                    )
                )
            self.save_face_records(
                records,
                stat.st_mtime_ns,
                stat.st_size,
                image_width=image_width,
                image_height=image_height,
                assess_quality=False,
            )
            total_faces += len(records)
            done += 1
            _emit_progress(f"{done}/{total} images, extracted {total_faces} faces")
        _emit_progress(f"{done}/{total} images, extracted {total_faces} faces", force=True)
        return {
            "images_done": int(done),
            "images_total": int(total),
            "faces_indexed": int(total_faces),
            "skipped_unchanged": int(skipped_unchanged),
            "skipped_recognition_disabled": int(skipped_disabled),
        }

    def save_face_records(
        self,
        records: list[FaceIndexRecord],
        mtime_ns: int,
        file_size: int,
        *,
        image_width: int = 0,
        image_height: int = 0,
        assess_quality: bool = True,
    ) -> None:
        if not records:
            return
        image_path = records[0].image_path
        width = int(image_width)
        height = int(image_height)
        if width <= 0 or height <= 0:
            width, height = self._image_dimensions(image_path)
        if assess_quality:
            records = self._assess_face_records(
                image_path,
                list(records),
                image_width=width,
                image_height=height,
            )
        self._replace_image_records(
            image_path,
            records,
            mtime_ns=mtime_ns,
            file_size=file_size,
            image_width=width,
            image_height=height,
        )

    def remove_image(self, image_path: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM face_index WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM face_labels WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM pending_face_labels WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM face_scan_images WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM person_prototype_faces WHERE image_path=?", (image_path,))
            connection.execute("DELETE FROM rejected_face_labels WHERE image_path=?", (image_path,))
        self._invalidate_identity_caches()
        self._invalidate_ann_index()

    def load_all_records(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[IndexedFaceRecord]:
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT
                i.image_path,
                i.face_index,
                i.bbox_json,
                i.face_confidence,
                i.embedding,
                i.quality_status,
                i.quality_score,
                i.quality_reasons_json,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0),
                COALESCE(i.hidden, 0),
                COALESCE(pp.hidden, 0),
                COALESCE(i.mtime_ns, 0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        query += " ORDER BY i.image_path, i.face_index"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        records = [
            IndexedFaceRecord(
                image_path=row[0],
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
                quality_status=str(row[5] or "clean"),
                quality_score=float(row[6] or 1.0),
                quality_reasons=tuple(str(value) for value in json.loads(row[7] or "[]")),
                person_name=str(row[8] or ""),
                label_confidence=float(row[9] or 0.0),
                hidden=bool(row[10] or row[11]),
                image_mtime=float(row[12] or 0.0) / 1_000_000_000.0,
            )
            for row in rows
        ]
        return [
            record
            for record in records
            if self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths)
            and self._face_is_visible(record.face_bbox, include_tiny_faces=include_tiny_faces)
            and (include_hidden or not bool(record.hidden))
        ]

    @staticmethod
    def _face_album_group_id(group_kind: str, person_name: str = "") -> str:
        kind = str(group_kind or "").strip().lower()
        if kind == "named":
            return f"person:{str(person_name or '').strip()}"
        if kind == "pending":
            return "pending"
        if kind == "hidden":
            return "hidden"
        return "unlabeled"

    def load_face_album_members(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceAlbumRecord]:
        raise_if_cancelled(cancel_check)
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT
                i.image_path,
                i.face_index,
                i.bbox_json,
                i.face_confidence,
                i.quality_status,
                i.quality_score,
                i.quality_reasons_json,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0),
                COALESCE(i.hidden, 0),
                COALESCE(pp.hidden, 0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        if not include_tiny_faces:
            query = self._append_sql_condition(query, "COALESCE(i.is_tiny, 0)=0")
        query += " ORDER BY i.image_path, i.face_index"
        with self._connect() as connection:
            rows = self._execute_scoped_query(
                connection,
                query,
                args,
                image_column="i.image_path",
                candidate_paths=candidate_paths,
            )
        album_records: list[FaceAlbumRecord] = []
        record_by_ref: dict[tuple[str, int], FaceAlbumRecord] = {}
        for row in rows:
            raise_if_cancelled(cancel_check)
            image_path = str(row[0] or "")
            face_index = int(row[1] or 0)
            face_bbox = tuple(json.loads(row[2] or "[]"))
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            if not self._face_is_visible(face_bbox, include_tiny_faces=include_tiny_faces):
                continue
            person_name = str(row[7] or "").strip()
            hidden = bool(row[9] or row[10])
            if hidden:
                group_kind = "hidden"
            elif person_name:
                group_kind = "named"
            else:
                group_kind = "unlabeled"
            album_record = FaceAlbumRecord(
                group_id=self._face_album_group_id(group_kind, person_name),
                group_kind=group_kind,
                image_path=image_path,
                face_index=face_index,
                face_bbox=face_bbox,
                face_confidence=float(row[3] or 0.0),
                person_name=person_name,
                label_confidence=float(row[8] or 0.0),
                quality_status=str(row[4] or "clean"),
                quality_score=float(row[5] or 1.0),
                quality_reasons=tuple(str(value) for value in json.loads(row[6] or "[]")),
                hidden=hidden,
                display_name=person_name or "Unlabeled",
            )
            record_by_ref[(image_path, face_index)] = album_record
            album_records.append(album_record)
        for assignment in self.load_pending_face_labels(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
            include_hidden=True,
        ):
            raise_if_cancelled(cancel_check)
            ref = (str(assignment.image_path), int(assignment.face_index))
            record = record_by_ref.get(ref)
            if record is None:
                continue
            if not self._path_in_scope(str(record.image_path), folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            if bool(record.hidden):
                continue
            album_records.append(
                FaceAlbumRecord(
                    group_id=self._face_album_group_id("pending"),
                    group_kind="pending",
                    image_path=str(record.image_path),
                    face_index=int(record.face_index),
                    face_bbox=tuple(record.face_bbox),
                    face_confidence=float(record.face_confidence),
                    person_name=str(assignment.person_name or "").strip(),
                    label_confidence=float(assignment.confidence),
                    quality_status=str(record.quality_status or "clean"),
                    quality_score=float(record.quality_score),
                    quality_reasons=tuple(str(value) for value in record.quality_reasons or ()),
                    hidden=False,
                    display_name=str(assignment.person_name or "").strip() or str(record.person_name or "").strip() or "Pending label",
                    source=str(assignment.source or ""),
                    created_at=str(assignment.created_at or ""),
                )
            )
        return album_records

    def load_face_album_snapshot(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> tuple[list[FaceAlbumGroupSummary], list[FaceAlbumRecord]]:
        members = self.load_face_album_members(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
            cancel_check=cancel_check,
        )
        raise_if_cancelled(cancel_check)
        groups = self._summarize_face_album_members(
            members,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
        )
        return groups, members

    def _prepare_face_album_scope_table(
        self,
        connection: sqlite3.Connection,
        candidate_paths: list[str] | None,
    ) -> bool:
        values = self._candidate_path_query_values(candidate_paths)
        if not values:
            return False
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS face_album_scope_paths (image_path TEXT PRIMARY KEY)")
        connection.execute("DELETE FROM face_album_scope_paths")
        connection.executemany(
            "INSERT OR IGNORE INTO face_album_scope_paths(image_path) VALUES (?)",
            [(value,) for value in values],
        )
        return True

    @staticmethod
    def _face_album_scope_sql(
        column: str,
        *,
        folder_prefix: str,
        candidate_scope: bool,
        include_tiny_faces: bool,
    ) -> tuple[str, list[object]]:
        conditions: list[str] = []
        args: list[object] = []
        if str(folder_prefix or "").strip():
            scope_clause, scope_args = folder_scope_sql(column, folder_prefix)
            conditions.append(scope_clause)
            args.extend(scope_args)
        if candidate_scope:
            conditions.append(f"{column} IN (SELECT image_path FROM face_album_scope_paths)")
        if not include_tiny_faces:
            conditions.append("COALESCE(i.is_tiny, 0)=0")
        return (" AND ".join(conditions) if conditions else "1=1"), args

    def load_face_album_group_page(
        self,
        *,
        offset: int = 0,
        limit: int = 100,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> FaceAlbumGroupPage:
        raise_if_cancelled(cancel_check)
        page_offset = max(0, int(offset))
        page_limit = max(1, min(500, int(limit)))
        with self._connect() as connection:
            candidate_scope = self._prepare_face_album_scope_table(connection, candidate_paths)
            base_scope, base_args = self._face_album_scope_sql(
                "i.image_path",
                folder_prefix=folder_prefix,
                candidate_scope=candidate_scope,
                include_tiny_faces=include_tiny_faces,
            )
            pending_scope, pending_args = self._face_album_scope_sql(
                "p.image_path",
                folder_prefix=folder_prefix,
                candidate_scope=candidate_scope,
                include_tiny_faces=include_tiny_faces,
            )
            member_rows_sql = f"""
                SELECT
                    CASE
                        WHEN COALESCE(i.hidden, 0)=1 OR COALESCE(pp.hidden, 0)=1 THEN 'hidden'
                        WHEN TRIM(COALESCE(l.person_name, '')) <> '' THEN 'named'
                        ELSE 'unlabeled'
                    END AS group_kind,
                    CASE
                        WHEN COALESCE(i.hidden, 0)=1 OR COALESCE(pp.hidden, 0)=1 THEN 'hidden'
                        WHEN TRIM(COALESCE(l.person_name, '')) <> '' THEN 'person:' || TRIM(l.person_name)
                        ELSE 'unlabeled'
                    END AS group_id,
                    CASE
                        WHEN COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0 THEN TRIM(COALESCE(l.person_name, ''))
                        ELSE ''
                    END AS person_name,
                    i.image_path,
                    COALESCE(pp.favorite, 0) AS favorite,
                    COALESCE(pp.notes, '') AS notes
                FROM face_index i
                LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                WHERE {base_scope}

                UNION ALL

                SELECT
                    'pending' AS group_kind,
                    'pending' AS group_id,
                    '' AS person_name,
                    p.image_path,
                    0 AS favorite,
                    '' AS notes
                FROM pending_face_labels p
                JOIN face_index i ON i.image_path=p.image_path AND i.face_index=p.face_index
                LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                WHERE {pending_scope}
                    AND COALESCE(i.hidden, 0)=0
                    AND COALESCE(pp.hidden, 0)=0
            """
            grouped_sql = f"""
                SELECT
                    group_id,
                    group_kind,
                    person_name,
                    COUNT(*) AS face_count,
                    COUNT(DISTINCT image_path) AS photo_count,
                    MAX(favorite) AS favorite,
                    MAX(notes) AS notes
                FROM ({member_rows_sql}) album_members
                GROUP BY group_id, group_kind, person_name
            """
            params = [*base_args, *pending_args]
            total_count = int(
                connection.execute(f"SELECT COUNT(*) FROM ({grouped_sql}) grouped", params).fetchone()[0] or 0
            )
            rows = connection.execute(
                f"""
                {grouped_sql}
                ORDER BY
                    CASE group_kind WHEN 'named' THEN 0 WHEN 'unlabeled' THEN 1 WHEN 'pending' THEN 2 ELSE 3 END,
                    CASE WHEN group_kind='named' THEN favorite ELSE 0 END DESC,
                    LOWER(person_name),
                    group_id
                LIMIT ? OFFSET ?
                """,
                [*params, page_limit, page_offset],
            ).fetchall()
        summaries: list[FaceAlbumGroupSummary] = []
        for group_id, group_kind, person_name, face_count, photo_count, favorite, notes in rows:
            raise_if_cancelled(cancel_check)
            kind = str(group_kind or "unlabeled")
            name = str(person_name or "").strip()
            count = int(face_count or 0)
            photos = int(photo_count or 0)
            if kind == "named":
                title = f"{'[Favorite] ' if bool(favorite) else ''}{name} | {count} face(s)"
                summary = f"{name}: {count} face(s) across {photos} photo(s)."
                if str(notes or "").strip():
                    summary += f" {str(notes).strip()[:120]}"
            elif kind == "pending":
                title = f"Pending Labels | {count} face(s)"
                summary = f"{count} pending face assignment(s) across {photos} photo(s)."
            elif kind == "hidden":
                title = f"Hidden / Rejected | {count} face(s)"
                summary = f"{count} hidden face(s) across {photos} photo(s)."
            else:
                title = f"Unlabeled | {count} face(s)"
                summary = f"{count} unlabeled face(s) across {photos} photo(s)."
            summaries.append(
                FaceAlbumGroupSummary(
                    group_id=str(group_id),
                    group_kind=kind,
                    title=title,
                    summary=summary,
                    face_count=count,
                    photo_count=photos,
                    person_name=name,
                )
            )
        next_offset = page_offset + len(summaries)
        return FaceAlbumGroupPage(
            items=tuple(summaries),
            total_count=total_count,
            offset=page_offset,
            next_offset=next_offset if next_offset < total_count else None,
        )

    def load_face_album_member_page(
        self,
        group_id: str,
        *,
        offset: int = 0,
        limit: int = 200,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> FaceAlbumMemberPage:
        raise_if_cancelled(cancel_check)
        normalized_group_id = str(group_id or "").strip()
        page_offset = max(0, int(offset))
        page_limit = max(1, min(1000, int(limit)))
        with self._connect() as connection:
            candidate_scope = self._prepare_face_album_scope_table(connection, candidate_paths)
            path_column = "p.image_path" if normalized_group_id == "pending" else "i.image_path"
            scope_clause, scope_args = self._face_album_scope_sql(
                path_column,
                folder_prefix=folder_prefix,
                candidate_scope=candidate_scope,
                include_tiny_faces=include_tiny_faces,
            )
            args: list[object] = list(scope_args)
            if normalized_group_id == "pending":
                from_sql = """
                    FROM pending_face_labels p
                    JOIN face_index i ON i.image_path=p.image_path AND i.face_index=p.face_index
                    LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                    LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                """
                condition = f"{scope_clause} AND COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0"
                select_sql = """
                    SELECT p.image_path, p.face_index, i.bbox_json, i.face_confidence,
                           i.quality_status, i.quality_score, i.quality_reasons_json,
                           COALESCE(p.person_name, ''), COALESCE(p.confidence, 0.0),
                           COALESCE(p.source, ''), COALESCE(p.created_at, '')
                """
                group_kind = "pending"
            else:
                from_sql = """
                    FROM face_index i
                    LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                    LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                """
                hidden_expression = "(COALESCE(i.hidden, 0)=1 OR COALESCE(pp.hidden, 0)=1)"
                if normalized_group_id.startswith("person:"):
                    group_kind = "named"
                    condition = f"{scope_clause} AND NOT {hidden_expression} AND TRIM(COALESCE(l.person_name, ''))=?"
                    args.append(normalized_group_id.split(":", 1)[1])
                elif normalized_group_id == "hidden":
                    group_kind = "hidden"
                    condition = f"{scope_clause} AND {hidden_expression}"
                else:
                    group_kind = "unlabeled"
                    condition = f"{scope_clause} AND NOT {hidden_expression} AND TRIM(COALESCE(l.person_name, ''))=''"
                select_sql = """
                    SELECT i.image_path, i.face_index, i.bbox_json, i.face_confidence,
                           i.quality_status, i.quality_score, i.quality_reasons_json,
                           COALESCE(l.person_name, ''), COALESCE(l.confidence, 0.0),
                           '', ''
                """
            total_count = int(
                connection.execute(f"SELECT COUNT(*) {from_sql} WHERE {condition}", args).fetchone()[0] or 0
            )
            rows = connection.execute(
                f"{select_sql} {from_sql} WHERE {condition} ORDER BY 1, 2 LIMIT ? OFFSET ?",
                [*args, page_limit, page_offset],
            ).fetchall()
        items = tuple(
            FaceAlbumRecord(
                group_id=normalized_group_id,
                group_kind=group_kind,
                image_path=str(row[0] or ""),
                face_index=int(row[1] or 0),
                face_bbox=tuple(json.loads(row[2] or "[]")),
                face_confidence=float(row[3] or 0.0),
                quality_status=str(row[4] or "clean"),
                quality_score=float(row[5] or 1.0),
                quality_reasons=tuple(str(value) for value in json.loads(row[6] or "[]")),
                person_name=str(row[7] or "").strip(),
                label_confidence=float(row[8] or 0.0),
                hidden=group_kind == "hidden",
                display_name=str(row[7] or "").strip() or ("Pending label" if group_kind == "pending" else "Unlabeled"),
                source=str(row[9] or ""),
                created_at=str(row[10] or ""),
            )
            for row in rows
        )
        raise_if_cancelled(cancel_check)
        next_offset = page_offset + len(items)
        return FaceAlbumMemberPage(
            group_id=normalized_group_id,
            items=items,
            total_count=total_count,
            offset=page_offset,
            next_offset=next_offset if next_offset < total_count else None,
        )

    def load_face_album_groups(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceAlbumGroupSummary]:
        members = self.load_face_album_members(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
            cancel_check=cancel_check,
        )
        return self._summarize_face_album_members(
            members,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
        )

    def _summarize_face_album_members(
        self,
        members: list[FaceAlbumRecord],
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
    ) -> list[FaceAlbumGroupSummary]:
        members_by_group: dict[str, list[FaceAlbumRecord]] = defaultdict(list)
        for item in members:
            members_by_group[str(item.group_id)].append(item)
        profile_by_name = {
            str(profile.person_name): profile
            for profile in self.load_person_profiles(
                folder_prefix=folder_prefix,
                candidate_paths=candidate_paths,
                limit=max(1, len(members) + 100),
                include_tiny_faces=include_tiny_faces,
                include_hidden=True,
            )
        }
        rejected_count = 0
        allowed_paths = self._normalized_path_set(candidate_paths)
        for assignment in self.list_rejected_face_label_corrections():
            image_path = str(assignment.image_path or "")
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            if not self._face_is_visible(tuple(assignment.face_bbox), include_tiny_faces=include_tiny_faces):
                continue
            rejected_count += 1
        named_summaries: list[FaceAlbumGroupSummary] = []
        bucket_summaries: list[FaceAlbumGroupSummary] = []
        for group_id, group_members in members_by_group.items():
            first = group_members[0]
            photo_count = len({str(item.image_path) for item in group_members})
            face_count = len(group_members)
            if first.group_kind == "named":
                person_name = str(first.person_name or "").strip()
                profile = profile_by_name.get(person_name)
                prefix = "[Favorite] " if bool(getattr(profile, "favorite", False)) else ""
                title = f"{prefix}{person_name} | {face_count} face(s)"
                summary = f"{person_name}: {face_count} face(s) across {photo_count} photo(s)."
                if profile is not None and str(getattr(profile, "notes", "") or "").strip():
                    summary += f" {str(getattr(profile, 'notes', '')).strip()[:120]}"
                named_summaries.append(
                    FaceAlbumGroupSummary(
                        group_id=group_id,
                        group_kind="named",
                        title=title,
                        summary=summary,
                        face_count=face_count,
                        photo_count=photo_count,
                        person_name=person_name,
                    )
                )
                continue
            if first.group_kind == "pending":
                bucket_summaries.append(
                    FaceAlbumGroupSummary(
                        group_id=group_id,
                        group_kind="pending",
                        title=f"Pending Labels | {face_count} face(s)",
                        summary=f"{face_count} pending face assignment(s) across {photo_count} photo(s).",
                        face_count=face_count,
                        photo_count=photo_count,
                    )
                )
                continue
            if first.group_kind == "hidden":
                hidden_summary = f"{face_count} hidden face(s) across {photo_count} photo(s)."
                if rejected_count:
                    hidden_summary += f" {rejected_count} rejected correction(s) recorded."
                bucket_summaries.append(
                    FaceAlbumGroupSummary(
                        group_id=group_id,
                        group_kind="hidden",
                        title=f"Hidden / Rejected | {face_count} face(s)",
                        summary=hidden_summary,
                        face_count=face_count,
                        photo_count=photo_count,
                    )
                )
                continue
            bucket_summaries.append(
                FaceAlbumGroupSummary(
                    group_id=group_id,
                    group_kind="unlabeled",
                    title=f"Unlabeled | {face_count} face(s)",
                    summary=f"{face_count} unlabeled face(s) across {photo_count} photo(s).",
                    face_count=face_count,
                    photo_count=photo_count,
                )
            )
        named_summaries.sort(
            key=lambda item: (
                not bool(getattr(profile_by_name.get(item.person_name), "favorite", False)),
                str(item.person_name or "").lower(),
            )
        )
        bucket_order = {"unlabeled": 0, "pending": 1, "hidden": 2}
        bucket_summaries.sort(key=lambda item: (bucket_order.get(str(item.group_kind or ""), 99), str(item.title or "").lower()))
        return named_summaries + bucket_summaries

    def _migrate_legacy_face_coordinate_space(self, image_paths: list[str]) -> set[str]:
        targets = []
        seen: set[str] = set()
        for image_path in image_paths:
            normalized = self._normalize_path_for_match(image_path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            targets.append(str(image_path))
        if not targets:
            return set()
        migrated_paths: set[str] = set()
        with self._connect() as connection:
            for image_path in targets:
                row = connection.execute(
                    "SELECT image_width, image_height FROM face_scan_images WHERE image_path=?",
                    (str(image_path),),
                ).fetchone()
                if row is None:
                    continue
                saved_width = int(row[0] or 0)
                saved_height = int(row[1] or 0)
                try:
                    info = _read_oriented_image_info(str(image_path))
                except Exception:
                    continue
                display_dims = (int(info.display_width), int(info.display_height))
                raw_dims = (int(info.raw_width), int(info.raw_height))
                if (saved_width, saved_height) == display_dims:
                    continue
                if int(info.orientation) == 1:
                    if saved_width <= 0 or saved_height <= 0:
                        connection.execute(
                            "UPDATE face_scan_images SET image_width=?, image_height=? WHERE image_path=?",
                            (display_dims[0], display_dims[1], str(image_path)),
                        )
                        migrated_paths.add(str(image_path))
                    continue
                source_dims = (saved_width, saved_height) if saved_width > 0 and saved_height > 0 else raw_dims
                if source_dims != raw_dims:
                    LOGGER.warning(
                        "FaceIndexService skipped legacy face bbox migration mode=%s db=%s image=%s saved_dims=%s raw_dims=%s display_dims=%s orientation=%s",
                        self.mode,
                        self.db_path,
                        image_path,
                        source_dims,
                        raw_dims,
                        display_dims,
                        int(info.orientation),
                    )
                    continue
                face_rows = connection.execute(
                    "SELECT face_index, bbox_json, face_confidence FROM face_index WHERE image_path=? ORDER BY face_index",
                    (str(image_path),),
                ).fetchall()
                display_image: Image.Image | None = None
                if face_rows:
                    try:
                        display_image = _open_display_rgb_image(str(image_path))
                    except Exception:
                        display_image = None
                updates: list[tuple[str, str, float, str, int, str, int]] = []
                for face_index, bbox_json, face_confidence in face_rows:
                    try:
                        bbox = tuple(int(value) for value in json.loads(bbox_json))
                    except Exception:
                        continue
                    migrated_bbox = _transform_bbox_to_display_space(
                        bbox,
                        raw_width=info.raw_width,
                        raw_height=info.raw_height,
                        orientation=info.orientation,
                    )
                    if migrated_bbox is None:
                        continue
                    crop = None
                    if display_image is not None:
                        try:
                            crop = display_image.crop(migrated_bbox)
                        except Exception:
                            crop = None
                    status, score, reasons = self._assess_face_quality(
                        image_path=str(image_path),
                        bbox=migrated_bbox,
                        confidence=float(face_confidence or 0.0),
                        crop=crop,
                        image_width=display_dims[0],
                        image_height=display_dims[1],
                    )
                    updates.append(
                        (
                            json.dumps(list(migrated_bbox)),
                            str(status or "clean"),
                            float(score),
                            json.dumps(list(reasons or ())),
                            int(FACE_QUALITY_METADATA_REVISION),
                            str(image_path),
                            int(face_index),
                        )
                    )
                if display_image is not None:
                    try:
                        display_image.close()
                    except Exception:
                        pass
                if updates:
                    connection.executemany(
                        """
                        UPDATE face_index
                        SET bbox_json=?, quality_status=?, quality_score=?, quality_reasons_json=?, quality_revision=?
                        WHERE image_path=? AND face_index=?
                        """,
                        updates,
                    )
                connection.execute(
                    "UPDATE face_scan_images SET image_width=?, image_height=? WHERE image_path=?",
                    (display_dims[0], display_dims[1], str(image_path)),
                )
                migrated_paths.add(str(image_path))
                LOGGER.info(
                    "FaceIndexService migrated legacy face coordinate space mode=%s db=%s image=%s orientation=%s raw_dims=%s display_dims=%s faces=%s",
                    self.mode,
                    self.db_path,
                    image_path,
                    int(info.orientation),
                    raw_dims,
                    display_dims,
                    len(updates),
                )
        if migrated_paths:
            self._recent_migrated_face_paths.update(migrated_paths)
        return migrated_paths

    def consume_recent_migrated_face_paths(self) -> list[str]:
        paths = sorted(self._recent_migrated_face_paths)
        self._recent_migrated_face_paths.clear()
        return paths

    def load_scan_image_records(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
    ) -> list[FaceScanImageRecord]:
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at
            FROM face_scan_images
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        query += " ORDER BY image_path"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        records = [
            FaceScanImageRecord(
                image_path=str(row[0]),
                mtime_ns=int(row[1] or 0),
                file_size=int(row[2] or 0),
                face_count=int(row[3] or 0),
                image_width=int(row[4] or 0),
                image_height=int(row[5] or 0),
                indexed_at=str(row[6] or ""),
            )
            for row in rows
        ]
        return [
            record
            for record in records
            if self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths)
        ]

    def load_folder_review_images(
        self,
        directory: str,
        *,
        recursive: bool = True,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
        cancel_check=None,
    ) -> list[FaceFolderReviewImage]:
        raise_if_cancelled(cancel_check)
        directory = str(directory or "").strip()
        if not directory:
            return []
        LOGGER.info(
            "FaceIndexService load_folder_review_images mode=%s db=%s directory=%s recursive=%s include_tiny=%s candidate_paths=%s",
            self.mode,
            self.db_path,
            directory,
            recursive,
            include_tiny_faces,
            0 if not candidate_paths else len(candidate_paths),
        )
        if candidate_paths is not None:
            discovered_paths = [
                str(path)
                for path in list(candidate_paths or [])
                if self._path_in_scope(str(path), folder_prefix=directory)
            ]
        else:
            discovered_paths = self.discovery_service.discover_result(
                directory,
                recursive=recursive,
                cancel_check=cancel_check,
            ).paths
        raise_if_cancelled(cancel_check)
        allowed_paths = self._normalized_path_set(candidate_paths)
        if allowed_paths:
            discovered_paths = [
                path for path in discovered_paths
                if self._normalize_path_for_match(path) in allowed_paths
            ]
        if len(discovered_paths) <= int(MAX_EAGER_FACE_REVIEW_MIGRATION_PATHS):
            self._migrate_legacy_face_coordinate_space(discovered_paths)
        elif discovered_paths:
            LOGGER.info(
                "FaceIndexService skipping eager folder-review coordinate migration mode=%s db=%s directory=%s paths=%s",
                self.mode,
                self.db_path,
                directory,
                len(discovered_paths),
            )
        self._backfill_quality_metadata_for_paths(discovered_paths)
        raise_if_cancelled(cancel_check)
        query = """
            SELECT
                i.image_path,
                i.face_index,
                i.bbox_json,
                i.face_confidence,
                i.quality_status,
                i.quality_score,
                i.quality_reasons_json,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0),
                COALESCE(i.hidden, 0),
                COALESCE(pp.hidden, 0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if directory:
            scope_clause, scope_args = folder_scope_sql("i.image_path", directory)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        query += """
            ORDER BY
                CASE WHEN COALESCE(l.person_name, '') = '' THEN 1 ELSE 0 END,
                LOWER(COALESCE(l.person_name, '')),
                i.image_path,
                i.face_index
        """
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        scan_state_by_path = {
            record.image_path: record
            for record in self.load_scan_image_records(folder_prefix=directory, candidate_paths=candidate_paths)
        }
        records_by_path: dict[str, list[IndexedFaceRecord]] = {}
        empty_embedding = np.empty(0, dtype=np.float32)
        for row in rows:
            raise_if_cancelled(cancel_check)
            image_path = str(row[0])
            if not self._path_in_scope(image_path, folder_prefix=directory, candidate_paths=allowed_paths):
                continue
            record = IndexedFaceRecord(
                image_path=image_path,
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=empty_embedding.copy(),
                quality_status=str(row[4] or "clean"),
                quality_score=float(row[5] or 1.0),
                quality_reasons=tuple(str(value) for value in json.loads(row[6] or "[]")),
                person_name=str(row[7] or ""),
                label_confidence=float(row[8] or 0.0),
                hidden=bool(row[9] or row[10]),
            )
            if record.hidden and not include_hidden:
                continue
            records_by_path.setdefault(image_path, []).append(record)
        if not discovered_paths:
            discovered_paths = sorted(
                set(scan_state_by_path).union(records_by_path),
                key=str.lower,
            )

        review_images: list[FaceFolderReviewImage] = []
        for image_path in discovered_paths:
            raise_if_cancelled(cancel_check)
            image_records = list(records_by_path.get(image_path, ()))
            visible_records = [
                record
                for record in image_records
                if self._face_is_visible(record.face_bbox, include_tiny_faces=include_tiny_faces)
            ]
            hidden_face_count = max(0, len(image_records) - len(visible_records))
            scan_state = scan_state_by_path.get(image_path)
            if image_records:
                if visible_records:
                    review_status = "detected"
                elif hidden_face_count > 0:
                    review_status = "tiny_hidden"
                else:
                    review_status = "detected"
            elif scan_state is not None:
                review_status = "no_faces"
            else:
                review_status = "not_scanned"
            review_images.append(
                FaceFolderReviewImage(
                    image_path=image_path,
                    review_status=review_status,
                    visible_faces=tuple(visible_records),
                    total_face_count=max(len(image_records), int(getattr(scan_state, "face_count", 0) or 0)),
                    hidden_face_count=hidden_face_count,
                    image_width=int(getattr(scan_state, "image_width", 0) or 0),
                    image_height=int(getattr(scan_state, "image_height", 0) or 0),
                )
            )
        LOGGER.info(
            "FaceIndexService load_folder_review_images_complete mode=%s db=%s directory=%s discovered=%s review_images=%s",
            self.mode,
            self.db_path,
            directory,
            len(discovered_paths),
            len(review_images),
        )
        return review_images

    def count_indexed_faces(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> int:
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT i.image_path, i.bbox_json, COALESCE(i.hidden, 0), COALESCE(pp.hidden, 0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        query += " ORDER BY i.image_path, i.face_index"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        total = 0
        for row in rows:
            image_path = str(row[0])
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            if not self._face_is_visible(tuple(json.loads(row[1])), include_tiny_faces=include_tiny_faces):
                continue
            if (bool(row[2]) or bool(row[3])) and not include_hidden:
                continue
            total += 1
        return int(total)

    def load_indexed_faces(
        self,
        *,
        folder_prefix: str = "",
        limit: int = 400,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[IndexedFaceRecord]:
        folder_prefix = str(folder_prefix or "").strip()
        limit = max(1, int(limit))
        if candidate_paths:
            self._migrate_legacy_face_coordinate_space([str(path) for path in candidate_paths])
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT
                i.image_path,
                i.face_index,
                i.bbox_json,
                i.face_confidence,
                i.embedding,
                i.quality_status,
                i.quality_score,
                i.quality_reasons_json,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0),
                COALESCE(i.hidden, 0),
                COALESCE(pp.hidden, 0),
                COALESCE(i.mtime_ns, 0)
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        query += """
            ORDER BY
                CASE WHEN COALESCE(l.person_name, '') = '' THEN 1 ELSE 0 END,
                LOWER(COALESCE(l.person_name, '')),
                i.image_path,
                i.face_index
        """
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        records = [
            IndexedFaceRecord(
                image_path=str(row[0]),
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
                quality_status=str(row[5] or "clean"),
                quality_score=float(row[6] or 1.0),
                quality_reasons=tuple(str(value) for value in json.loads(row[7] or "[]")),
                person_name=str(row[8] or ""),
                label_confidence=float(row[9] or 0.0),
                hidden=bool(row[10] or row[11]),
                image_mtime=float(row[12] or 0.0) / 1_000_000_000.0,
            )
            for row in rows
        ]
        records = [
            record
            for record in records
            if self._path_in_scope(record.image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths)
            and self._face_is_visible(record.face_bbox, include_tiny_faces=include_tiny_faces)
            and (include_hidden or not bool(record.hidden))
        ]
        return records[:limit]

    def load_face_record(self, image_path: str, face_index: int, *, include_hidden: bool = False) -> IndexedFaceRecord | None:
        self._migrate_legacy_face_coordinate_space([str(image_path)])
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    i.image_path,
                    i.face_index,
                    i.bbox_json,
                    i.face_confidence,
                    i.embedding,
                    i.quality_status,
                    i.quality_score,
                    i.quality_reasons_json,
                    COALESCE(l.person_name, ''),
                    COALESCE(l.confidence, 0.0),
                    COALESCE(i.hidden, 0),
                    COALESCE(pp.hidden, 0),
                    COALESCE(i.mtime_ns, 0)
                FROM face_index i
                LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                WHERE i.image_path=? AND i.face_index=?
                """,
                (str(image_path), int(face_index)),
            ).fetchone()
        if row is None:
            return None
        hidden = bool(row[10] or row[11])
        if hidden and not include_hidden:
            return None
        return IndexedFaceRecord(
            image_path=str(row[0]),
            face_index=int(row[1]),
            face_bbox=tuple(json.loads(row[2])),
            face_confidence=float(row[3]),
            embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
            quality_status=str(row[5] or "clean"),
            quality_score=float(row[6] or 1.0),
            quality_reasons=tuple(str(value) for value in json.loads(row[7] or "[]")),
            person_name=str(row[8] or ""),
            label_confidence=float(row[9] or 0.0),
            hidden=hidden,
            image_mtime=float(row[12] or 0.0) / 1_000_000_000.0,
        )

    def load_face_metadata_many(
        self,
        face_refs: list[tuple[str, int]],
        *,
        include_hidden: bool = False,
    ) -> dict[tuple[str, int], tuple[str, str]]:
        refs = list(
            dict.fromkeys(
                (str(image_path), int(face_index))
                for image_path, face_index in (face_refs or [])
                if str(image_path or "").strip() and int(face_index) >= 0
            )
        )
        if not refs:
            return {}
        self._migrate_legacy_face_coordinate_space([image_path for image_path, _face_index in refs])
        metadata_by_ref: dict[tuple[str, int], tuple[str, str]] = {}
        with self._connect() as connection:
            for chunk in self._chunked_values(list(refs), size=250):
                conditions = " OR ".join("(i.image_path=? AND i.face_index=?)" for _item in chunk)
                params: list[object] = []
                for image_path, face_index in chunk:
                    params.extend([str(image_path), int(face_index)])
                query = f"""
                    SELECT
                        i.image_path,
                        i.face_index,
                        COALESCE(l.person_name, ''),
                        COALESCE(i.quality_status, 'clean'),
                        COALESCE(i.hidden, 0),
                        COALESCE(pp.hidden, 0)
                    FROM face_index i
                    LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
                    LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
                    WHERE {conditions}
                """
                rows = connection.execute(query, params).fetchall()
                for row in rows:
                    hidden = bool(row[4] or row[5])
                    if hidden and not include_hidden:
                        continue
                    metadata_by_ref[(str(row[0] or ""), int(row[1] or 0))] = (
                        str(row[2] or "").strip(),
                        str(row[3] or "clean").strip().lower(),
                    )
        return metadata_by_ref

    def detect_faces_for_image(self, image_path: str, *, include_tiny_faces: bool = False) -> list[DetectedFace]:
        LOGGER.info(
            "FaceIndexService detect_faces_for_image mode=%s db=%s image=%s include_tiny=%s",
            self.mode,
            self.db_path,
            image_path,
            include_tiny_faces,
        )
        faces = self._detect_faces_with_policy(str(image_path))
        LOGGER.info(
            "FaceIndexService detect_faces_for_image_complete mode=%s db=%s image=%s faces=%s",
            self.mode,
            self.db_path,
            image_path,
            len(faces),
        )
        return faces

    def load_image_faces(self, image_path: str, *, include_tiny_faces: bool = True, include_hidden: bool = False) -> list[IndexedFaceRecord]:
        target = self._normalize_path_for_match(image_path)
        if not target:
            return []
        self._migrate_legacy_face_coordinate_space([str(image_path)])
        return [
            record
            for record in self.load_indexed_faces(
                candidate_paths=[str(image_path)],
                limit=1000,
                include_tiny_faces=include_tiny_faces,
                include_hidden=include_hidden,
            )
            if self._normalize_path_for_match(record.image_path) == target
        ]

    def save_image_faces(
        self,
        image_path: str,
        face_inputs: list[EditableFaceInput],
        *,
        preserve_labels: bool = True,
    ) -> list[IndexedFaceRecord]:
        image_path = str(image_path or "").strip()
        if not image_path:
            raise ValueError("Image path is required.")
        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(image_path)
        stat = image_file.stat()
        image_width, image_height = self._image_dimensions(image_path)
        existing_records = self.load_image_faces(image_path, include_tiny_faces=True, include_hidden=True)
        existing_by_bbox = {
            tuple(record.face_bbox): record
            for record in existing_records
        }
        normalized_inputs: list[EditableFaceInput] = []
        seen_bboxes: set[tuple[int, int, int, int]] = set()
        for face_input in list(face_inputs or []):
            bbox = _normalize_face_bbox(
                tuple(face_input.bbox),
                image_width=image_width,
                image_height=image_height,
            )
            if bbox is None or bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            normalized_inputs.append(
                EditableFaceInput(
                    bbox=bbox,
                    confidence=float(max(0.0, face_input.confidence)),
                )
            )
        if face_inputs and not normalized_inputs:
            raise ValueError("No valid face boxes were provided for this image.")

        LOGGER.info(
            "FaceIndexService save_image_faces mode=%s db=%s image=%s faces=%s preserve_labels=%s",
            self.mode,
            self.db_path,
            image_path,
            len(normalized_inputs),
            preserve_labels,
        )

        new_face_crops: list[Image.Image] = []
        for face_input in normalized_inputs:
            if tuple(face_input.bbox) in existing_by_bbox:
                continue
            detected_face = self.detection_service.detect_query_face(image_path, query_bbox=tuple(face_input.bbox))
            if detected_face is None:
                raise ValueError(f"Could not extract the drawn face crop for {image_path}.")
            new_face_crops.append(detected_face.crop)
        new_embeddings = self.embedding_service.embed_faces(new_face_crops) if new_face_crops else np.zeros((0, 0), dtype=np.float32)

        records: list[FaceIndexRecord] = []
        label_rows: list[tuple[str, int, str, float]] = []
        new_embedding_pos = 0
        for face_index, face_input in enumerate(normalized_inputs):
            existing = existing_by_bbox.get(tuple(face_input.bbox))
            if existing is not None:
                embedding = np.asarray(existing.embedding, dtype=np.float32).copy()
                confidence = float(existing.face_confidence or face_input.confidence or 1.0)
            else:
                embedding = np.asarray(new_embeddings[new_embedding_pos], dtype=np.float32).copy()
                new_embedding_pos += 1
                confidence = float(face_input.confidence or 1.0)
            records.append(
                FaceIndexRecord(
                    image_path=image_path,
                    face_index=int(face_index),
                    face_bbox=tuple(face_input.bbox),
                    face_confidence=confidence,
                    embedding=embedding,
                )
            )
            if preserve_labels and existing is not None and str(existing.person_name or "").strip():
                label_rows.append(
                    (
                        image_path,
                        int(face_index),
                        str(existing.person_name),
                        float(existing.label_confidence or 1.0),
                    )
                )

        records = self._assess_face_records(
            image_path,
            records,
            image_width=image_width,
            image_height=image_height,
        )
        self._replace_image_records(
            image_path,
            records,
            mtime_ns=stat.st_mtime_ns,
            file_size=stat.st_size,
            image_width=image_width,
            image_height=image_height,
        )
        if label_rows:
            with self._connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index) DO UPDATE SET
                        person_name=excluded.person_name,
                        confidence=excluded.confidence
                    """,
                    label_rows,
                )
        saved = self.load_image_faces(image_path, include_tiny_faces=True)
        LOGGER.info(
            "FaceIndexService save_image_faces_complete mode=%s db=%s image=%s faces=%s",
            self.mode,
            self.db_path,
            image_path,
            len(saved),
        )
        return saved

    def search_faces(self, request: FaceSearchRequest, *, cancel_check=None) -> list[FaceSearchResult]:
        raise_if_cancelled(cancel_check)
        query_face = self.detect_query_face(request.query_face_image, request.query_face_bbox)
        if query_face is None:
            raise ValueError(f"No {self.mode_label.lower()} face found in query image.")
        query_embedding = self.embedding_service.embed_faces([query_face.crop])[0]
        raise_if_cancelled(cancel_check)
        return self._search_by_embedding(
            query_embedding,
            min_face_score=max(float(request.min_face_score), float(self.recognition_min_score)),
            top_k=request.top_k,
            candidate_paths=request.candidate_paths,
            include_tiny_faces=request.include_tiny_faces,
            cancel_check=cancel_check,
        )

    @staticmethod
    def _mean_face_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
        if not embeddings:
            raise ValueError("At least one face embedding is required.")
        matrix = np.asarray([np.asarray(item, dtype=np.float32) for item in embeddings], dtype=np.float32)
        merged = matrix.mean(axis=0)
        norm = float(np.linalg.norm(merged))
        if norm > 1e-12:
            merged = merged / norm
        return merged.astype(np.float32)

    def search_query_faces(
        self,
        query_face_image: str,
        query_face_bboxes: list[tuple[int, int, int, int]],
        *,
        top_k: int = 30,
        min_face_score: float = 0.35,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceSearchResult]:
        faces: list[DetectedFace] = []
        for bbox in list(query_face_bboxes or []):
            raise_if_cancelled(cancel_check)
            face = self.detect_query_face(query_face_image, bbox)
            if face is not None:
                faces.append(face)
        if not faces:
            raise ValueError(f"No {self.mode_label.lower()} faces found in the selected query photo.")
        query_embedding = self._mean_face_embedding(self.embedding_service.embed_faces([face.crop for face in faces]))
        raise_if_cancelled(cancel_check)
        return self._search_by_embedding(
            query_embedding,
            min_face_score=max(float(min_face_score), float(self.recognition_min_score)),
            top_k=top_k,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
            cancel_check=cancel_check,
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
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceSearchResult]:
        LOGGER.info(
            "FaceIndexService search_similar_face mode=%s db=%s image=%s face_index=%s top_k=%s min_score=%s folder_prefix=%s include_tiny=%s",
            self.mode,
            self.db_path,
            image_path,
            face_index,
            top_k,
            min_score,
            folder_prefix,
            include_tiny_faces,
        )
        raise_if_cancelled(cancel_check)
        record = self.load_face_record(image_path, face_index)
        if record is None:
            raise ValueError("The selected indexed face was not found.")
        results = self._search_by_embedding(
            record.embedding,
            min_face_score=max(float(min_score), float(self.recognition_min_score)),
            top_k=top_k,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            exclude={(record.image_path, record.face_index)},
            include_tiny_faces=include_tiny_faces,
            cancel_check=cancel_check,
        )
        LOGGER.info(
            "FaceIndexService search_similar_face_complete mode=%s db=%s image=%s face_index=%s results=%s",
            self.mode,
            self.db_path,
            image_path,
            face_index,
            len(results),
        )
        return results

    def search_similar_faces(
        self,
        face_refs: list[tuple[str, int]],
        *,
        top_k: int = 30,
        min_score: float = 0.35,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceSearchResult]:
        refs = list(dict.fromkeys((str(image_path), int(face_index)) for image_path, face_index in (face_refs or []) if str(image_path or "").strip()))
        if not refs:
            raise ValueError("At least one indexed face is required.")
        embeddings: list[np.ndarray] = []
        exclude: set[tuple[str, int]] = set()
        for image_path, face_index in refs:
            raise_if_cancelled(cancel_check)
            record = self.load_face_record(image_path, face_index)
            if record is None:
                continue
            embeddings.append(record.embedding)
            exclude.add((record.image_path, int(record.face_index)))
        if not embeddings:
            raise ValueError("None of the selected indexed faces could be loaded.")
        query_embedding = self._mean_face_embedding(embeddings)
        return self._search_by_embedding(
            query_embedding,
            min_face_score=max(float(min_score), float(self.recognition_min_score)),
            top_k=top_k,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            exclude=exclude,
            include_tiny_faces=include_tiny_faces,
            cancel_check=cancel_check,
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
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> list[FaceSearchResult]:
        raise_if_cancelled(cancel_check)
        exclude = exclude or set()
        folder_prefix = str(folder_prefix or "").strip()
        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        query_vector = query_vector / np.clip(norms, 1e-12, None)
        ann_index, ann_refs, _fingerprint = self._load_ann_index()
        if ann_index is None or not ann_refs:
            return self._search_by_embedding_bruteforce(
                query_vector[0],
                min_face_score=min_face_score,
                top_k=top_k,
                folder_prefix=folder_prefix,
                candidate_paths=candidate_paths,
                exclude=exclude,
                include_tiny_faces=include_tiny_faces,
                cancel_check=cancel_check,
            )
        allowed_paths = self._normalized_path_set(candidate_paths)
        cached_records: dict[tuple[str, int], IndexedFaceRecord | None] = {}
        hits: list[tuple[FaceSearchResult, IndexedFaceRecord]] = []
        total = len(ann_refs)
        search_k = min(total, max(int(top_k) * 8, 128))
        while search_k > 0:
            raise_if_cancelled(cancel_check)
            try:
                scores, indices = ann_index.search(query_vector.astype(np.float32, copy=False), search_k)
            except Exception:
                self._invalidate_ann_index()
                return self._search_by_embedding_bruteforce(
                    query_vector[0],
                    min_face_score=min_face_score,
                    top_k=top_k,
                    folder_prefix=folder_prefix,
                    candidate_paths=candidate_paths,
                    exclude=exclude,
                    include_tiny_faces=include_tiny_faces,
                    cancel_check=cancel_check,
                )
            hits.clear()
            seen: set[tuple[str, int]] = set()
            for score, raw_index in zip(scores[0], indices[0]):
                raise_if_cancelled(cancel_check)
                index_id = int(raw_index)
                if index_id < 0 or index_id >= total:
                    continue
                ref = ann_refs[index_id]
                if ref in seen or ref in exclude:
                    continue
                seen.add(ref)
                image_path, face_index = ref
                if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                    continue
                record = cached_records.get(ref)
                if record is None and ref not in cached_records:
                    record = self.load_face_record(image_path, face_index)
                    cached_records[ref] = record
                if record is None:
                    continue
                if not self._face_is_visible(record.face_bbox, include_tiny_faces=include_tiny_faces):
                    continue
                if not self._quality_allows(record.quality_status, self.search_quality_min):
                    continue
                similarity = float(score)
                if similarity < min_face_score:
                    continue
                hits.append(
                    (
                        FaceSearchResult(
                            image_path=record.image_path,
                            face_index=int(record.face_index),
                            score=round(similarity, 6),
                            phash_distance=-1,
                            face_bbox=record.face_bbox,
                            face_confidence=round(record.face_confidence, 6),
                                model_name=self.model_name,
                                match_reason=f"{self.mode_label.lower()} face embedding cosine similarity",
                                person_name=str(record.person_name or ""),
                                quality_status=str(record.quality_status or "clean"),
                                quality_reasons=tuple(record.quality_reasons or ()),
                                hidden=bool(record.hidden),
                                image_mtime=float(record.image_mtime or 0.0),
                            ),
                            record,
                        )
                )
                if len(hits) >= max(1, top_k):
                    break
            if len(hits) >= max(1, top_k) or search_k >= total:
                break
            search_k = min(total, search_k * 2)
        if not hits:
            return self._search_by_embedding_bruteforce(
                query_vector[0],
                min_face_score=min_face_score,
                top_k=top_k,
                folder_prefix=folder_prefix,
                candidate_paths=candidate_paths,
                exclude=exclude,
                include_tiny_faces=include_tiny_faces,
                cancel_check=cancel_check,
            )
        return self._finalize_search_hits(hits, top_k=top_k)

    def cluster_faces(
        self,
        num_clusters: int = 12,
        min_face_score: float = 0.0,
        *,
        backend: str = "hdbscan",
        outlier_policy: str = "assign",
        backend_options: dict[str, object] | None = None,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        face_refs: list[tuple[str, int]] | None = None,
        include_tiny_faces: bool = True,
    ) -> dict[int, list[FaceClusterMember]]:
        LOGGER.info(
            "FaceIndexService cluster_faces mode=%s db=%s num_clusters=%s backend=%s outlier_policy=%s min_face_score=%s folder_prefix=%s include_tiny=%s",
            self.mode,
            self.db_path,
            num_clusters,
            backend,
            outlier_policy,
            min_face_score,
            folder_prefix,
            include_tiny_faces,
        )
        records = self._clusterable_face_records(
            min_face_score=min_face_score,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            face_refs=face_refs,
            include_tiny_faces=include_tiny_faces,
        )
        if len(records) < 2:
            return {}
        compare = self.cluster_faces_compare(
            num_clusters=num_clusters,
            min_face_score=min_face_score,
            backends=[backend],
            outlier_policy=outlier_policy,
            backend_options_by_backend={str(backend or "hdbscan").strip().lower(): dict(backend_options or {})},
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            face_refs=face_refs,
            include_tiny_faces=include_tiny_faces,
        )
        key = next(iter(compare.clusters_by_key.keys()), str(backend or "hdbscan").strip().lower() or "hdbscan")
        clusters_by_id = dict(compare.clusters_by_key.get(key) or {})
        LOGGER.info(
            "FaceIndexService cluster_faces_complete mode=%s db=%s records=%s clusters=%s",
            self.mode,
            self.db_path,
            len(records),
            len(clusters_by_id),
        )
        return clusters_by_id

    def cluster_faces_compare(
        self,
        num_clusters: int = 12,
        min_face_score: float = 0.0,
        *,
        backends: list[str] | tuple[str, ...] | None = None,
        outlier_policy: str = "isolate",
        backend_options_by_backend: dict[str, dict[str, object]] | None = None,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        face_refs: list[tuple[str, int]] | None = None,
        include_tiny_faces: bool = True,
        cancel_check=None,
    ) -> FaceClusteringComparisonResult:
        raise_if_cancelled(cancel_check)
        records = self._clusterable_face_records(
            min_face_score=min_face_score,
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            face_refs=face_refs,
            include_tiny_faces=include_tiny_faces,
        )
        normalized_backends = self._normalized_face_cluster_backends(backends)
        if len(records) < 2 or not normalized_backends:
            return FaceClusteringComparisonResult({}, {}, {}, {}, {})
        raise_if_cancelled(cancel_check)
        cluster_count = min(max(2, int(num_clusters)), len(records))
        embeddings = [record.embedding for record in records]
        prepared_matrix, prepared_info = self.clustering_service.prepare_matrix_with_info(embeddings, similarity_mode="cosine")
        prototypes = self.load_person_prototypes()
        raise_if_cancelled(cancel_check)
        clusters_by_key: dict[str, dict[int, list[FaceClusterMember]]] = {}
        membership_by_face_ref: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
        metrics_by_key: dict[str, dict[str, object]] = {}
        explanations_by_key: dict[str, dict[int, ClusterExplanation]] = {}
        suggestions_by_key: dict[str, dict[int, FaceClusterIdentitySuggestion]] = {}
        for backend_id in normalized_backends:
            raise_if_cancelled(cancel_check)
            clusters, metrics = self._cluster_prepared_faces_for_backend(
                prepared_matrix,
                cluster_count,
                backend=backend_id,
                outlier_policy=outlier_policy,
                backend_options=dict((backend_options_by_backend or {}).get(str(backend_id), {}) or {}),
            )
            members_by_cluster = self._face_cluster_members_from_indices(records, clusters)
            clusters_by_key[backend_id] = members_by_cluster
            metrics_by_key[backend_id] = dict(metrics)
            metrics_by_key[backend_id]["requested_backend"] = backend_id
            metrics_by_key[backend_id]["outlier_policy"] = str(outlier_policy or "assign").strip().lower()
            metrics_by_key[backend_id]["backend_options"] = dict((backend_options_by_backend or {}).get(str(backend_id), {}) or {})
            explanations_by_key[backend_id] = self.clustering_service.build_cluster_explanations(
                prepared_matrix,
                clusters,
                cluster_quality_score=metrics.get("cluster_quality_score"),
                prepared_info=prepared_info,
            )
            suggestions_by_key[backend_id] = self._cluster_identity_suggestions(records, clusters, prototypes)
            for cluster_id, indices in clusters.items():
                raise_if_cancelled(cancel_check)
                cluster_size = len(indices)
                cluster_quality = metrics.get("cluster_quality_score")
                for rank, record_index in enumerate(indices, start=1):
                    raise_if_cancelled(cancel_check)
                    record = records[int(record_index)]
                    membership_by_face_ref.setdefault((record.image_path, int(record.face_index)), {})[backend_id] = {
                        "backend": backend_id,
                        "cluster_id": int(cluster_id),
                        "cluster_size": int(cluster_size),
                        "rank": int(rank),
                        "outlier": int(cluster_id) == -1,
                        "cluster_quality_score": cluster_quality,
                    }
        raise_if_cancelled(cancel_check)
        return FaceClusteringComparisonResult(
            clusters_by_key=clusters_by_key,
            membership_by_face_ref=membership_by_face_ref,
            metrics_by_key=metrics_by_key,
            explanations_by_key=explanations_by_key,
            identity_suggestions_by_key=suggestions_by_key,
        )

    def _clusterable_face_records(
        self,
        *,
        min_face_score: float,
        folder_prefix: str,
        candidate_paths: list[str] | None,
        face_refs: list[tuple[str, int]] | None,
        include_tiny_faces: bool,
    ) -> list[IndexedFaceRecord]:
        selected_refs = {
            (str(image_path), int(face_index))
            for image_path, face_index in (face_refs or [])
            if str(image_path or "").strip()
        }
        records: list[IndexedFaceRecord] = []
        for record in self.load_all_records(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
        ):
            if selected_refs and (record.image_path, int(record.face_index)) not in selected_refs:
                continue
            if record.face_confidence < float(min_face_score):
                continue
            if not self._quality_allows(record.quality_status, self.cluster_quality_min):
                continue
            records.append(record)
        return records

    @staticmethod
    def _normalized_face_cluster_backends(backends: list[str] | tuple[str, ...] | None) -> list[str]:
        allowed = {item_id for item_id, _label in face_cluster_backend_choices()}
        normalized: list[str] = []
        for backend in backends or ():
            item_id = str(backend or "").strip().lower()
            if item_id in allowed and item_id not in normalized:
                normalized.append(item_id)
        return normalized or ["hdbscan"]

    def _cluster_prepared_faces_for_backend(
        self,
        prepared_matrix: np.ndarray,
        num_clusters: int,
        *,
        backend: str,
        outlier_policy: str,
        backend_options: dict[str, object] | None = None,
    ) -> tuple[dict[int, list[int]], dict[str, object]]:
        backend_id = str(backend or "hdbscan").strip().lower()
        clusters, raw_metrics = self.clustering_service.cluster_prepared(
            prepared_matrix,
            num_clusters,
            backend=backend_id,
            outlier_policy=outlier_policy,
            backend_options=backend_options,
        )
        metrics = dict(raw_metrics)
        metrics["fallback_used"] = False
        return clusters, metrics

    @staticmethod
    def _face_cluster_members_from_indices(
        records: list[IndexedFaceRecord],
        clusters: dict[int, list[int]],
    ) -> dict[int, list[FaceClusterMember]]:
        return {
            int(cluster_id): [
                FaceClusterMember(
                    image_path=records[index].image_path,
                    face_index=int(records[index].face_index),
                    face_bbox=tuple(records[index].face_bbox),
                    face_confidence=float(records[index].face_confidence),
                    person_name=str(getattr(records[index], "person_name", "") or ""),
                    quality_status=str(getattr(records[index], "quality_status", "clean") or "clean"),
                    hidden=bool(getattr(records[index], "hidden", False)),
                )
                for index in indices
            ]
            for cluster_id, indices in clusters.items()
        }

    def _cluster_identity_suggestions(
        self,
        records: list[IndexedFaceRecord],
        clusters: dict[int, list[int]],
        prototypes: list[PersonPrototype],
    ) -> dict[int, FaceClusterIdentitySuggestion]:
        named_prototypes = [
            prototype
            for prototype in list(prototypes or [])
            if str(prototype.person_name or "").strip() and np.asarray(prototype.embedding).size > 0
        ]
        if not named_prototypes:
            return {}
        suggestions: dict[int, FaceClusterIdentitySuggestion] = {}
        prototype_names = [str(prototype.person_name) for prototype in named_prototypes]
        prototype_matrix = np.asarray([np.asarray(prototype.embedding, dtype=np.float32) for prototype in named_prototypes], dtype=np.float32)
        thresholds = np.asarray(
            [
                max(float(prototype.similarity_threshold), float(self.recognition_min_score))
                for prototype in named_prototypes
            ],
            dtype=np.float32,
        )
        for cluster_id, indices in clusters.items():
            if int(cluster_id) == -1 or not indices:
                continue
            cluster_embeddings = np.asarray(
                [np.asarray(records[int(index)].embedding, dtype=np.float32) for index in indices],
                dtype=np.float32,
            )
            if cluster_embeddings.size <= 0:
                continue
            similarity_matrix = cluster_embeddings @ prototype_matrix.T
            best_indexes = np.argmax(similarity_matrix, axis=1)
            best_scores = similarity_matrix[np.arange(similarity_matrix.shape[0]), best_indexes]
            scores_by_person: dict[str, list[float]] = defaultdict(list)
            for best_index, best_score in zip(best_indexes.tolist(), best_scores.tolist()):
                if float(best_score) < float(thresholds[int(best_index)]):
                    continue
                scores_by_person[prototype_names[int(best_index)]].append(float(best_score))
            if not scores_by_person:
                continue
            best_person, score_values = max(
                scores_by_person.items(),
                key=lambda item: (len(item[1]), sum(item[1]) / max(1, len(item[1]))),
            )
            support_count = len(score_values)
            member_count = len(indices)
            required_support = 1 if member_count == 1 else max(2, (member_count + 1) // 2)
            if support_count < required_support:
                continue
            suggestions[int(cluster_id)] = FaceClusterIdentitySuggestion(
                person_name=best_person,
                support_count=support_count,
                member_count=member_count,
                mean_score=float(sum(score_values) / max(1, len(score_values))),
                min_score=float(min(score_values)),
                max_score=float(max(score_values)),
            )
        return suggestions

    def label_face_examples(self, request: FaceLabelRequest) -> PersonPrototype:
        if not request.person_name.strip():
            raise ValueError("Identity name is required.")
        example_faces = []
        prototype_face_refs: list[tuple[str, int]] = []
        for image_path in request.example_image_paths:
            face = self.detect_query_face(image_path)
            if face is not None:
                status, _score, _reasons = self._face_quality_from_detected(str(image_path), face)
                if self._quality_allows(status, self.prototype_quality_min):
                    example_faces.append(face.crop)
                    matching_records = [
                        record
                        for record in self.load_image_faces(str(image_path), include_tiny_faces=True)
                        if tuple(record.face_bbox) == tuple(face.bbox)
                    ]
                    if matching_records:
                        matched_record = matching_records[0]
                        prototype_face_refs.append((matched_record.image_path, int(matched_record.face_index)))
        if not example_faces:
            raise ValueError(f"No {self.mode_label.lower()} faces found in the selected example images.")
        embeddings = self.embedding_service.embed_faces(example_faces)
        return self._save_person_prototype(
            request.person_name,
            embeddings,
            similarity_threshold=request.similarity_threshold,
            prototype_face_refs=prototype_face_refs,
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
            raise ValueError("Identity name is required.")
        if not face_refs:
            raise ValueError("Select at least one indexed face.")
        records: list[IndexedFaceRecord] = []
        for image_path, face_index in face_refs:
            record = self.load_face_record(image_path, face_index)
            if record is not None and self._quality_allows(record.quality_status, self.prototype_quality_min):
                records.append(record)
        if not records:
            raise ValueError("The selected indexed faces do not meet the current prototype quality gate.")
        embeddings = np.asarray([record.embedding for record in records], dtype=np.float32)
        person = self._save_person_prototype(
            person_name,
            embeddings,
            similarity_threshold=similarity_threshold,
            prototype_face_refs=[(record.image_path, int(record.face_index)) for record in records],
        )
        self._queue_face_label_assignments(
            [
                FaceLabelAssignment(
                    person_name=person.person_name,
                    image_path=record.image_path,
                    face_index=int(record.face_index),
                    face_bbox=record.face_bbox,
                    confidence=1.0,
                    source="manual_selected_faces",
                    pending=True,
                )
                for record in records
            ]
        )
        return person

    def label_indexed_faces_immediately(
        self,
        person_name: str,
        face_refs: list[tuple[str, int]],
        *,
        similarity_threshold: float = 0.72,
        source: str = "manual_cluster_name",
    ) -> PersonPrototype:
        person_name = str(person_name or "").strip()
        if not person_name:
            raise ValueError("Identity name is required.")
        normalized_refs = list(
            dict.fromkeys(
                (str(image_path), int(face_index))
                for image_path, face_index in (face_refs or [])
                if str(image_path or "").strip()
            )
        )
        if not normalized_refs:
            raise ValueError("Select at least one indexed face.")

        records: list[IndexedFaceRecord] = []
        for image_path, face_index in normalized_refs:
            record = self.load_face_record(image_path, face_index)
            if record is not None and self._quality_allows(record.quality_status, self.prototype_quality_min):
                records.append(record)
        if not records:
            raise ValueError("The selected indexed faces do not meet the current prototype quality gate.")

        embeddings = np.asarray([record.embedding for record in records], dtype=np.float32)
        person = self._save_person_prototype(
            person_name,
            embeddings,
            similarity_threshold=similarity_threshold,
            prototype_face_refs=[(record.image_path, int(record.face_index)) for record in records],
        )

        label_rows = [
            (
                record.image_path,
                int(record.face_index),
                str(person.person_name),
                1.0,
            )
            for record in records
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_path, face_index) DO UPDATE SET
                    person_name=excluded.person_name,
                    confidence=excluded.confidence
                """,
                label_rows,
            )
            connection.executemany(
                "DELETE FROM pending_face_labels WHERE image_path=? AND face_index=?",
                [(record.image_path, int(record.face_index)) for record in records],
            )
            self._record_action_audit(
                "label_indexed_faces_immediate",
                target=str(person.person_name),
                details={
                    "face_count": len(records),
                    "source": str(source or "manual_cluster_name"),
                    "prototype_face_count": len(records),
                },
                reversible=False,
                connection=connection,
            )
        return person

    def _save_person_prototype(
        self,
        person_name: str,
        embeddings: np.ndarray,
        *,
        similarity_threshold: float,
        prototype_face_refs: list[tuple[str, int]] | None = None,
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
            if prototype_face_refs is not None:
                unique_refs: list[tuple[str, int]] = []
                seen_refs: set[tuple[str, int]] = set()
                for image_path, face_index in list(prototype_face_refs or []):
                    ref = (str(image_path), int(face_index))
                    if not ref[0] or ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    unique_refs.append(ref)
                connection.execute(
                    "DELETE FROM person_prototype_faces WHERE person_name=?",
                    (person.person_name,),
                )
                for order, (image_path, face_index) in enumerate(unique_refs):
                    connection.execute(
                        """
                        INSERT INTO person_prototype_faces(person_name, image_path, face_index, pinned, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            person.person_name,
                            image_path,
                            int(face_index),
                            1 if order == 0 else 0,
                            int(order),
                        ),
                    )
                if unique_refs:
                    self._set_person_profile_cover(
                        connection,
                        person.person_name,
                        cover_face_ref=unique_refs[0],
                    )
        self._invalidate_identity_caches()
        return person

    def load_person_prototypes(self) -> list[PersonPrototype]:
        cached = self._cached_person_prototypes
        if cached is not None:
            return list(cached)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT person_name, embedding, similarity_threshold, example_count FROM person_prototypes"
            ).fetchall()
        prototypes = [
            PersonPrototype(
                person_name=row[0],
                embedding=np.frombuffer(row[1], dtype=np.float32).copy(),
                similarity_threshold=float(row[2]),
                example_count=int(row[3]),
            )
            for row in rows
        ]
        self._cached_person_prototypes = list(prototypes)
        return list(prototypes)

    def _set_person_profile_cover(
        self,
        connection: sqlite3.Connection,
        person_name: str,
        *,
        cover_face_ref: tuple[str, int] | None,
    ) -> None:
        row = connection.execute(
            "SELECT notes, tags_json, favorite, birth_date, hidden FROM person_profiles WHERE person_name=?",
            (str(person_name),),
        ).fetchone()
        notes = str(row[0] or "") if row is not None else ""
        tags_json = str(row[1] or "[]") if row is not None else "[]"
        favorite = int(row[2] or 0) if row is not None else 0
        birth_date = str(row[3] or "") if row is not None else ""
        hidden = int(row[4] or 0) if row is not None else 0
        cover_image_path = str((cover_face_ref or ("", 0))[0] or "")
        cover_face_index = int((cover_face_ref or ("", 0))[1] or 0)
        connection.execute(
            """
            INSERT INTO person_profiles(person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_name) DO UPDATE SET
                notes=excluded.notes,
                tags_json=excluded.tags_json,
                cover_image_path=excluded.cover_image_path,
                cover_face_index=excluded.cover_face_index,
                favorite=excluded.favorite,
                birth_date=excluded.birth_date,
                hidden=excluded.hidden
            """,
            (str(person_name), notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden),
        )

    def save_person_profile(
        self,
        person_name: str,
        *,
        notes: str = "",
        tags: list[str] | tuple[str, ...] | None = None,
        cover_face_ref: tuple[str, int] | None = None,
        favorite: bool | None = None,
        birth_date: str | None = None,
        hidden: bool | None = None,
    ) -> None:
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("Identity name is required.")
        clean_tags = tuple(str(tag).strip() for tag in (tags or []) if str(tag).strip())
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cover_image_path, cover_face_index, favorite, birth_date, hidden
                FROM person_profiles
                WHERE person_name=?
                """,
                (name,),
            ).fetchone()
            if cover_face_ref is None and row is not None:
                cover_image_path = str(row[0] or "")
                cover_face_index = int(row[1] or 0)
            else:
                cover_image_path = str((cover_face_ref or ("", 0))[0] or "")
                cover_face_index = int((cover_face_ref or ("", 0))[1] or 0)
            favorite_value = int(row[2] or 0) if row is not None else 0
            birth_date_value = str(row[3] or "") if row is not None else ""
            hidden_value = int(row[4] or 0) if row is not None else 0
            if favorite is not None:
                favorite_value = 1 if bool(favorite) else 0
            if birth_date is not None:
                birth_date_value = str(birth_date or "").strip()
            if hidden is not None:
                hidden_value = 1 if bool(hidden) else 0
            connection.execute(
                """
                INSERT INTO person_profiles(person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_name) DO UPDATE SET
                    notes=excluded.notes,
                    tags_json=excluded.tags_json,
                    cover_image_path=excluded.cover_image_path,
                    cover_face_index=excluded.cover_face_index,
                    favorite=excluded.favorite,
                    birth_date=excluded.birth_date,
                    hidden=excluded.hidden
                """,
                (
                    name,
                    str(notes or ""),
                    json.dumps(list(clean_tags)),
                    cover_image_path,
                    cover_face_index,
                    int(favorite_value),
                    birth_date_value,
                    int(hidden_value),
                ),
            )
            self._record_action_audit(
                "save_person_profile",
                target=name,
                details={
                    "favorite": bool(favorite_value),
                    "birth_date": birth_date_value,
                    "hidden": bool(hidden_value),
                    "tag_count": len(clean_tags),
                    "cover_image_path": cover_image_path,
                    "cover_face_index": int(cover_face_index),
                },
                reversible=False,
                connection=connection,
            )
        self._invalidate_identity_caches()

    def set_person_hidden(self, person_name: str, hidden: bool = True) -> None:
        name = str(person_name or "").strip()
        if not name:
            return
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date
                FROM person_profiles
                WHERE person_name=?
                """,
                (name,),
            ).fetchone()
            values = (
                name,
                str(row[0] or "") if row is not None else "",
                str(row[1] or "[]") if row is not None else "[]",
                str(row[2] or "") if row is not None else "",
                int(row[3] or 0) if row is not None else 0,
                int(row[4] or 0) if row is not None else 0,
                str(row[5] or "") if row is not None else "",
                1 if bool(hidden) else 0,
            )
            connection.execute(
                """
                INSERT INTO person_profiles(person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(person_name) DO UPDATE SET
                    hidden=excluded.hidden
                """,
                values,
            )
            self._record_action_audit(
                "hide_person" if bool(hidden) else "unhide_person",
                target=name,
                details={"person_name": name, "hidden": bool(hidden)},
                reversible=True,
                connection=connection,
            )
        self._invalidate_identity_caches()

    def hide_person(self, person_name: str) -> None:
        self.set_person_hidden(person_name, True)

    def unhide_person(self, person_name: str) -> None:
        self.set_person_hidden(person_name, False)

    def set_face_hidden(self, image_path: str, face_index: int, hidden: bool = True) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE face_index SET hidden=? WHERE image_path=? AND face_index=?",
                (1 if bool(hidden) else 0, str(image_path), int(face_index)),
            )
            self._record_action_audit(
                "hide_face" if bool(hidden) else "unhide_face",
                target=f"{image_path}#{int(face_index)}",
                details={"image_path": str(image_path), "face_index": int(face_index), "hidden": bool(hidden)},
                reversible=True,
                connection=connection,
            )
        self._invalidate_identity_caches()
        self._invalidate_ann_index()

    def hide_face(self, image_path: str, face_index: int) -> None:
        self.set_face_hidden(image_path, face_index, True)

    def unhide_face(self, image_path: str, face_index: int) -> None:
        self.set_face_hidden(image_path, face_index, False)

    def age_at_photo(self, person_name: str, image_path: str) -> int | None:
        name = str(person_name or "").strip()
        if not name:
            return None
        with self._connect() as connection:
            row = connection.execute(
                "SELECT birth_date FROM person_profiles WHERE person_name=?",
                (name,),
            ).fetchone()
        birth_date_text = str(row[0] or "").strip() if row is not None else ""
        if not birth_date_text:
            return None
        try:
            born = date.fromisoformat(birth_date_text)
        except ValueError:
            return None
        try:
            photo_date = datetime.fromtimestamp(Path(image_path).stat().st_mtime).date()
        except Exception:
            return None
        age = photo_date.year - born.year - ((photo_date.month, photo_date.day) < (born.month, born.day))
        return max(0, int(age))

    def load_person_prototype_faces(
        self,
        person_name: str,
        *,
        limit: int = 100,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[PersonPrototypeFace]:
        name = str(person_name or "").strip()
        if not name:
            return []
        limit = max(1, int(limit))
        cache_key = (name, limit, bool(include_tiny_faces), bool(include_hidden))
        cached = self._cached_person_prototype_faces.get(cache_key)
        if cached is not None:
            return list(cached)
        query = """
            SELECT
                p.person_name,
                p.image_path,
                p.face_index,
                p.pinned,
                i.bbox_json,
                i.face_confidence,
                i.quality_status,
                i.quality_score,
                i.quality_reasons_json,
                COALESCE(l.person_name, ''),
                COALESCE(l.confidence, 0.0)
            FROM person_prototype_faces p
            JOIN face_index i ON i.image_path=p.image_path AND i.face_index=p.face_index
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=p.person_name
            WHERE p.person_name=?
        """
        args: list[object] = [name]
        if not include_tiny_faces:
            query = self._append_sql_condition(query, "COALESCE(i.is_tiny, 0)=0")
        if not include_hidden:
            query = self._append_sql_condition(query, "COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0")
        query += """
            ORDER BY p.pinned DESC, p.sort_order ASC, p.image_path ASC, p.face_index ASC
            LIMIT ?
        """
        args.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        faces: list[PersonPrototypeFace] = []
        for row in rows:
            faces.append(
                PersonPrototypeFace(
                    person_name=str(row[0]),
                    image_path=str(row[1]),
                    face_index=int(row[2]),
                    pinned=bool(row[3]),
                    face_bbox=tuple(json.loads(row[4])),
                    face_confidence=float(row[5]),
                    quality_status=str(row[6] or "clean"),
                    quality_score=float(row[7] or 1.0),
                    quality_reasons=tuple(str(value) for value in json.loads(row[8] or "[]")),
                    label_person_name=str(row[9] or ""),
                    label_confidence=float(row[10] or 0.0),
                )
            )
        self._cached_person_prototype_faces[cache_key] = list(faces)
        return list(faces)

    def remove_person_prototype_face(self, person_name: str, image_path: str, face_index: int) -> PersonPrototype:
        name = str(person_name or "").strip()
        path = str(image_path or "").strip()
        index = int(face_index)
        if not name or not path:
            raise ValueError("Identity and prototype face are required.")
        with self._connect() as connection:
            refs = [
                (str(row[0]), int(row[1]), bool(row[2]))
                for row in connection.execute(
                    """
                    SELECT image_path, face_index, pinned
                    FROM person_prototype_faces
                    WHERE person_name=?
                    ORDER BY pinned DESC, sort_order ASC, image_path ASC, face_index ASC
                    """,
                    (name,),
                ).fetchall()
            ]
            remaining = [(ref_path, ref_index) for ref_path, ref_index, _pinned in refs if (ref_path, ref_index) != (path, index)]
            if len(remaining) <= 0:
                raise ValueError("At least one prototype example is required for a saved identity.")
            connection.execute(
                "DELETE FROM person_prototype_faces WHERE person_name=? AND image_path=? AND face_index=?",
                (name, path, index),
            )
            for order, (ref_path, ref_index) in enumerate(remaining):
                connection.execute(
                    """
                    UPDATE person_prototype_faces
                    SET pinned=?, sort_order=?
                    WHERE person_name=? AND image_path=? AND face_index=?
                    """,
                    (1 if order == 0 else 0, int(order), name, ref_path, int(ref_index)),
                )
            self._set_person_profile_cover(connection, name, cover_face_ref=remaining[0])
        self._invalidate_identity_caches()
        rebuilt = self._rebuild_person_prototype_from_refs(name)
        if rebuilt is None:
            raise ValueError("Failed to rebuild the saved identity after removing the prototype example.")
        return rebuilt

    def pin_person_prototype_face(self, person_name: str, image_path: str, face_index: int) -> None:
        name = str(person_name or "").strip()
        path = str(image_path or "").strip()
        index = int(face_index)
        if not name or not path:
            raise ValueError("Identity and prototype face are required.")
        with self._connect() as connection:
            connection.execute(
                "UPDATE person_prototype_faces SET pinned=0 WHERE person_name=?",
                (name,),
            )
            connection.execute(
                """
                UPDATE person_prototype_faces
                SET pinned=1
                WHERE person_name=? AND image_path=? AND face_index=?
                """,
                (name, path, index),
            )
            self._set_person_profile_cover(connection, name, cover_face_ref=(path, index))
        self._invalidate_identity_caches()

    def identity_duplicate_warnings(self, *, similarity_threshold: float = 0.92) -> dict[str, tuple[tuple[str, float], ...]]:
        threshold = float(similarity_threshold)
        cached = self._cached_duplicate_warnings.get(threshold)
        if cached is not None:
            return dict(cached)
        prototypes = [
            prototype
            for prototype in list(self.load_person_prototypes())
            if np.asarray(prototype.embedding).size > 0
        ]
        warnings: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for left_index, left in enumerate(prototypes):
            for right in prototypes[left_index + 1 :]:
                if left.embedding.size <= 0 or right.embedding.size <= 0:
                    continue
                score = float(np.dot(left.embedding, right.embedding))
                if score < threshold:
                    continue
                warnings[left.person_name].append((right.person_name, score))
                warnings[right.person_name].append((left.person_name, score))
        result = {
            str(name): tuple(sorted(values, key=lambda item: (-float(item[1]), str(item[0]).lower())))
            for name, values in warnings.items()
        }
        self._cached_duplicate_warnings[threshold] = dict(result)
        return dict(result)

    def merge_person_identities(self, source_name: str, target_name: str) -> None:
        source = str(source_name or "").strip()
        target = str(target_name or "").strip()
        if not source or not target or source == target:
            return
        source_prototype = {item.person_name: item for item in self.load_person_prototypes()}.get(source)
        target_prototype = {item.person_name: item for item in self.load_person_prototypes()}.get(target)
        source_refs = [(item.image_path, int(item.face_index), bool(item.pinned)) for item in self.load_person_prototype_faces(source, include_tiny_faces=True)]
        target_refs = [(item.image_path, int(item.face_index), bool(item.pinned)) for item in self.load_person_prototype_faces(target, include_tiny_faces=True)]
        with self._connect() as connection:
            connection.execute(
                "UPDATE face_labels SET person_name=? WHERE person_name=?",
                (target, source),
            )
            connection.execute(
                "UPDATE pending_face_labels SET person_name=? WHERE person_name=?",
                (target, source),
            )
            connection.execute(
                "UPDATE rejected_face_labels SET person_name=? WHERE person_name=?",
                (target, source),
            )
            source_meta = connection.execute(
                "SELECT notes, tags_json, cover_image_path, cover_face_index FROM person_profiles WHERE person_name=?",
                (source,),
            ).fetchone()
            target_meta = connection.execute(
                "SELECT notes, tags_json, cover_image_path, cover_face_index FROM person_profiles WHERE person_name=?",
                (target,),
            ).fetchone()
            merged_notes = str((target_meta[0] if target_meta is not None else "") or "").strip()
            source_notes = str((source_meta[0] if source_meta is not None else "") or "").strip()
            if source_notes and source_notes not in merged_notes:
                merged_notes = source_notes if not merged_notes else f"{merged_notes}\n\n{source_notes}"
            merged_tags: list[str] = []
            for tag in list(json.loads((target_meta[1] if target_meta is not None else "[]") or "[]")) + list(json.loads((source_meta[1] if source_meta is not None else "[]") or "[]")):
                clean_tag = str(tag).strip()
                if clean_tag and clean_tag not in merged_tags:
                    merged_tags.append(clean_tag)
            merged_refs: list[tuple[str, int]] = []
            seen_refs: set[tuple[str, int]] = set()
            preferred_cover: tuple[str, int] | None = None
            for ref_path, ref_index, pinned in list(target_refs) + list(source_refs):
                ref = (str(ref_path), int(ref_index))
                if not ref[0] or ref in seen_refs:
                    continue
                seen_refs.add(ref)
                if preferred_cover is None and pinned:
                    preferred_cover = ref
                merged_refs.append(ref)
            if preferred_cover is None and merged_refs:
                preferred_cover = merged_refs[0]
            connection.execute("DELETE FROM person_prototype_faces WHERE person_name IN (?, ?)", (source, target))
            for order, (ref_path, ref_index) in enumerate(merged_refs):
                connection.execute(
                    """
                    INSERT INTO person_prototype_faces(person_name, image_path, face_index, pinned, sort_order)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (target, ref_path, int(ref_index), 1 if preferred_cover == (ref_path, int(ref_index)) else 0, int(order)),
                )
            self._set_person_profile_cover(connection, target, cover_face_ref=preferred_cover)
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
                (
                    target,
                    merged_notes,
                    json.dumps(merged_tags),
                    str((preferred_cover or ("", 0))[0] or ""),
                    int((preferred_cover or ("", 0))[1] or 0),
                ),
            )
            connection.execute("DELETE FROM person_profiles WHERE person_name=?", (source,))
            connection.execute("DELETE FROM person_prototypes WHERE person_name=?", (source,))
            self._record_action_audit(
                "merge_person_identities",
                target=target,
                details={"source": source, "target": target, "prototype_ref_count": len(merged_refs)},
                reversible=False,
                connection=connection,
            )
        self._invalidate_identity_caches()
        rebuilt = self._rebuild_person_prototype_from_refs(target)
        if rebuilt is None and source_prototype is not None and target_prototype is not None:
            embeddings = np.vstack(
                [
                    np.repeat(target_prototype.embedding[np.newaxis, :], max(1, int(target_prototype.example_count)), axis=0),
                    np.repeat(source_prototype.embedding[np.newaxis, :], max(1, int(source_prototype.example_count)), axis=0),
                ]
            )
            self._save_person_prototype(
                target,
                embeddings,
                similarity_threshold=min(float(target_prototype.similarity_threshold), float(source_prototype.similarity_threshold)),
                prototype_face_refs=merged_refs,
            )
        self._invalidate_identity_caches()

    def _rebuild_person_prototype_from_refs(self, person_name: str) -> PersonPrototype | None:
        refs = self.load_person_prototype_faces(person_name, include_tiny_faces=True)
        if not refs:
            return None
        records: list[IndexedFaceRecord] = []
        for ref in refs:
            record = self.load_face_record(ref.image_path, ref.face_index)
            if record is not None:
                records.append(record)
        if not records:
            return None
        prototype = self._save_person_prototype(
            str(person_name),
            np.asarray([record.embedding for record in records], dtype=np.float32),
            similarity_threshold=float(self.find_person_prototype(str(person_name)).similarity_threshold if self.find_person_prototype(str(person_name)) is not None else self.recognition_min_score),
            prototype_face_refs=[(record.image_path, int(record.face_index)) for record in records],
        )
        pinned_ref = next(((ref.image_path, int(ref.face_index)) for ref in refs if ref.pinned), None)
        if pinned_ref is not None:
            self.pin_person_prototype_face(str(person_name), pinned_ref[0], pinned_ref[1])
        return prototype

    def find_person_prototype(self, person_name: str) -> PersonPrototype | None:
        target = str(person_name or "").strip()
        if not target:
            return None
        for prototype in self.load_person_prototypes():
            if str(prototype.person_name) == target:
                return prototype
        return None

    def load_profile_examples(
        self,
        person_name: str,
        *,
        limit: int = 20,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[IndexedFaceRecord]:
        name = str(person_name or "").strip()
        if not name:
            return []
        allowed_paths = self._normalized_path_set(candidate_paths)
        limit = max(1, int(limit))
        query = """
            SELECT i.image_path, i.face_index, i.bbox_json, i.face_confidence, i.embedding, COALESCE(i.hidden, 0), COALESCE(pp.hidden, 0)
            FROM face_labels l
            JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
            WHERE l.person_name=?
        """
        args: list[object] = [name]
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" AND {scope_clause}"
            args.extend(scope_args)
        query += " ORDER BY l.confidence DESC, i.image_path, i.face_index"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        records: list[IndexedFaceRecord] = []
        for row in rows:
            image_path = str(row[0])
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            record = IndexedFaceRecord(
                image_path=image_path,
                face_index=int(row[1]),
                face_bbox=tuple(json.loads(row[2])),
                face_confidence=float(row[3]),
                embedding=np.frombuffer(row[4], dtype=np.float32).copy(),
                person_name=name,
                label_confidence=1.0,
                hidden=bool(row[5] or row[6]),
            )
            if not self._face_is_visible(record.face_bbox, include_tiny_faces=include_tiny_faces):
                continue
            if record.hidden and not include_hidden:
                continue
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
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
        cancel_check=None,
    ) -> list[PersonProfile]:
        raise_if_cancelled(cancel_check)
        folder_prefix = str(folder_prefix or "").strip()
        prototypes = {profile.person_name: profile for profile in self.load_person_prototypes()}
        labeled_counts = self.label_counts()
        visible_counts: dict[str, int] = defaultdict(int)
        query = """
            SELECT l.person_name, COUNT(1)
            FROM face_labels l
            JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("i.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        if not include_tiny_faces:
            query = self._append_sql_condition(query, "COALESCE(i.is_tiny, 0)=0")
        if not include_hidden:
            query = self._append_sql_condition(query, "COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0")
        query += " GROUP BY l.person_name"
        with self._connect() as connection:
            rows = self._execute_scoped_query(
                connection,
                query,
                args,
                image_column="i.image_path",
                candidate_paths=candidate_paths,
            )
        for row in rows:
            raise_if_cancelled(cancel_check)
            person_name = str(row[0] or "")
            visible_counts[person_name] += int(row[1] or 0)
        with self._connect() as connection:
            profile_rows = connection.execute(
                "SELECT person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden FROM person_profiles"
            ).fetchall()
        profile_meta = {
            str(row[0]): {
                "notes": str(row[1] or ""),
                "tags": tuple(str(tag).strip() for tag in json.loads(row[2] or "[]") if str(tag).strip()),
                "cover_image_path": str(row[3] or ""),
                "cover_face_index": int(row[4] or 0),
                "favorite": bool(row[5]),
                "birth_date": str(row[6] or ""),
                "hidden": bool(row[7]),
            }
            for row in profile_rows
        }
        profiles: list[PersonProfile] = []
        for name, prototype in sorted(
            prototypes.items(),
            key=lambda item: (not bool(profile_meta.get(item[0], {}).get("favorite", False)), item[0].lower()),
        ):
            raise_if_cancelled(cancel_check)
            meta = profile_meta.get(name, {})
            if bool(meta.get("hidden", False)) and not include_hidden:
                continue
            if not include_hidden and int(labeled_counts.get(name, 0)) > 0 and int(visible_counts.get(name, 0)) <= 0:
                continue
            profiles.append(
                PersonProfile(
                    person_name=name,
                    similarity_threshold=float(prototype.similarity_threshold),
                    example_count=int(prototype.example_count),
                    labeled_count=int(labeled_counts.get(name, 0)),
                    visible_face_count=int(visible_counts.get(name, 0)),
                    notes=str(meta.get("notes", "")),
                    tags=tuple(meta.get("tags", ())),
                    cover_image_path=str(meta.get("cover_image_path", "")),
                    cover_face_index=int(meta.get("cover_face_index", 0)),
                    favorite=bool(meta.get("favorite", False)),
                    birth_date=str(meta.get("birth_date", "")),
                    hidden=bool(meta.get("hidden", False)),
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
        include_tiny_faces: bool = True,
    ) -> list[FaceSearchResult]:
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("Identity name is required.")
        prototypes = {p.person_name: p for p in self.load_person_prototypes()}
        prototype = prototypes.get(name)
        if prototype is None:
            raise ValueError(f"No saved face prototype found for identity '{name}'.")

        threshold = float(min_score) if min_score is not None else float(prototype.similarity_threshold)
        threshold = max(threshold, float(self.recognition_min_score))
        results = self._search_by_embedding(
            prototype.embedding,
            min_face_score=threshold,
            top_k=max(1, int(top_k)),
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
        )
        return [
            FaceSearchResult(
                image_path=result.image_path,
                face_index=int(result.face_index),
                score=result.score,
                phash_distance=result.phash_distance,
                face_bbox=result.face_bbox,
                face_confidence=result.face_confidence,
                model_name=result.model_name,
                match_reason=f"identity match: {name}",
                person_name=str(result.person_name or name),
                quality_status=str(result.quality_status or "clean"),
                quality_reasons=tuple(result.quality_reasons or ()),
                hidden=bool(result.hidden),
                image_mtime=float(result.image_mtime or 0.0),
            )
            for result in results
        ]

    def find_face_records_by_people(
        self,
        person_names: list[str] | tuple[str, ...],
        *,
        require_all: bool = True,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
    ) -> list[IndexedFaceRecord]:
        target_names = [str(value or "").strip() for value in person_names if str(value or "").strip()]
        if not target_names:
            return []
        targets = {name.lower() for name in target_names}
        by_image: dict[str, list[IndexedFaceRecord]] = defaultdict(list)
        for record in self.load_all_records(candidate_paths=candidate_paths, include_tiny_faces=include_tiny_faces):
            by_image[str(record.image_path)].append(record)
        matches: list[IndexedFaceRecord] = []
        for records in by_image.values():
            present = {str(record.person_name or "").strip().lower() for record in records if str(record.person_name or "").strip()}
            qualifies = targets.issubset(present) if require_all else bool(targets & present)
            if not qualifies:
                continue
            matches.extend(
                [
                    record
                    for record in records
                    if str(record.person_name or "").strip().lower() in targets
                ]
            )
        return matches

    def find_face_records_with_named_people(
        self,
        *,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
    ) -> list[IndexedFaceRecord]:
        return [
            record
            for record in self.load_all_records(candidate_paths=candidate_paths, include_tiny_faces=include_tiny_faces)
            if str(record.person_name or "").strip()
        ]

    def find_face_records_with_unknown_people(
        self,
        *,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
    ) -> list[IndexedFaceRecord]:
        by_image: dict[str, list[IndexedFaceRecord]] = defaultdict(list)
        for record in self.load_all_records(candidate_paths=candidate_paths, include_tiny_faces=include_tiny_faces):
            by_image[str(record.image_path)].append(record)
        matches: list[IndexedFaceRecord] = []
        for records in by_image.values():
            if any(not str(record.person_name or "").strip() for record in records):
                matches.extend([record for record in records if not str(record.person_name or "").strip()])
        return matches

    def find_face_records_with_primary_person(
        self,
        person_name: str,
        *,
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
    ) -> list[IndexedFaceRecord]:
        target = str(person_name or "").strip().lower()
        if not target:
            return []
        by_image: dict[str, list[IndexedFaceRecord]] = defaultdict(list)
        for record in self.load_all_records(candidate_paths=candidate_paths, include_tiny_faces=include_tiny_faces):
            by_image[str(record.image_path)].append(record)
        matches: list[IndexedFaceRecord] = []
        for records in by_image.values():
            primary = max(
                records,
                key=lambda record: max(0, int(record.face_bbox[2]) - int(record.face_bbox[0]))
                * max(0, int(record.face_bbox[3]) - int(record.face_bbox[1])),
                default=None,
            )
            if primary is not None and str(primary.person_name or "").strip().lower() == target:
                matches.append(primary)
        return matches

    @staticmethod
    def _parse_face_query_tokens(query: str) -> dict[str, list[str]]:
        tokens: dict[str, list[str]] = defaultdict(list)
        for match in re.finditer(r"([A-Za-z_][\w-]*):(\"[^\"]*\"|\S+)", str(query or "")):
            key = str(match.group(1) or "").strip().lower()
            raw_value = str(match.group(2) or "").strip()
            if raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2:
                raw_value = raw_value[1:-1]
            if key:
                tokens[key].append(raw_value.strip())
        return dict(tokens)

    def search_face_query(
        self,
        query: str,
        *,
        candidate_paths: list[str] | None = None,
        folder_prefix: str = "",
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[IndexedFaceRecord]:
        tokens = self._parse_face_query_tokens(query)
        wants_hidden = any(str(value).strip().lower() in {"1", "true", "yes", "on"} for value in tokens.get("hidden", ()))
        records = self.load_all_records(
            folder_prefix=folder_prefix,
            candidate_paths=candidate_paths,
            include_tiny_faces=include_tiny_faces,
            include_hidden=include_hidden or wants_hidden,
        )
        if wants_hidden:
            records = [record for record in records if bool(record.hidden)]
        by_image: dict[str, list[IndexedFaceRecord]] = defaultdict(list)
        for record in records:
            by_image[str(record.image_path)].append(record)

        for value in tokens.get("faces", ()):
            normalized = str(value or "").strip().lower()
            if normalized in {"true", "yes", "1"}:
                continue
            if normalized in {"false", "no", "0"}:
                scan_records = self.load_scan_image_records(folder_prefix=folder_prefix, candidate_paths=candidate_paths)
                return [
                    self._empty_indexed_face_record(record.image_path)
                    for record in scan_records
                    if int(record.face_count or 0) == 0
                ]
            try:
                expected_count = int(normalized)
            except ValueError:
                continue
            scan_counts = {
                record.image_path: int(record.face_count or 0)
                for record in self.load_scan_image_records(folder_prefix=folder_prefix, candidate_paths=candidate_paths)
            }
            records = [record for record in records if int(scan_counts.get(record.image_path, len(by_image.get(record.image_path, ())))) == expected_count]
            by_image = defaultdict(list)
            for record in records:
                by_image[str(record.image_path)].append(record)

        for value in tokens.get("person", ()):
            text = str(value or "").strip()
            if not text:
                continue
            if "&" in text or "|" in text:
                require_all = "&" in text
                names = [part.strip().lower() for part in re.split(r"[&|]", text) if part.strip()]
                if not names:
                    continue
                matching_images: set[str] = set()
                for image_path, image_records in by_image.items():
                    present = {str(record.person_name or "").strip().lower() for record in image_records if str(record.person_name or "").strip()}
                    if names and (set(names).issubset(present) if require_all else bool(set(names) & present)):
                        matching_images.add(image_path)
                records = [
                    record
                    for record in records
                    if record.image_path in matching_images
                    and str(record.person_name or "").strip().lower() in set(names)
                ]
            else:
                target = text.lower()
                records = [record for record in records if str(record.person_name or "").strip().lower() == target]
            by_image = defaultdict(list)
            for record in records:
                by_image[str(record.image_path)].append(record)

        for value in tokens.get("people", ()):
            needle = str(value or "").strip().lower()
            if needle:
                records = [record for record in records if needle in str(record.person_name or "").strip().lower()]

        for value in tokens.get("unknown", ()):
            if str(value or "").strip().lower() in {"1", "true", "yes", "on"}:
                records = [record for record in records if not str(record.person_name or "").strip()]

        return sorted(records, key=lambda record: (record.image_path, int(record.face_index)))

    def export_person_photo_set(self, person_name: str, output_path: str | Path | None = None) -> dict[str, object]:
        name = str(person_name or "").strip()
        if not name:
            raise ValueError("Identity name is required.")
        labels = [
            assignment
            for assignment in self.list_face_labels(include_tiny_faces=True, include_hidden=True)
            if str(assignment.person_name or "") == name
        ]
        refs = [
            {
                "image_path": str(assignment.image_path),
                "face_index": int(assignment.face_index),
                "bbox": list(assignment.face_bbox),
                "source": "label",
            }
            for assignment in labels
        ]
        for face in self.load_person_prototype_faces(name, limit=1000, include_tiny_faces=True, include_hidden=True):
            refs.append(
                {
                    "image_path": str(face.image_path),
                    "face_index": int(face.face_index),
                    "bbox": list(face.face_bbox),
                    "source": "prototype",
                    "pinned": bool(face.pinned),
                }
            )
        unique_refs: list[dict[str, object]] = []
        seen_refs: set[tuple[str, int, str]] = set()
        for ref in refs:
            key = (str(ref["image_path"]), int(ref["face_index"]), str(ref["source"]))
            if key in seen_refs:
                continue
            seen_refs.add(key)
            unique_refs.append(ref)
        payload = {
            "person_name": name,
            "image_paths": sorted({str(ref["image_path"]) for ref in unique_refs}),
            "face_refs": unique_refs,
        }
        if output_path is not None:
            atomic_write_text(Path(output_path), json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def export_identity_data(self) -> dict[str, object]:
        with self._connect() as connection:
            face_rows = connection.execute(
                """
                SELECT image_path, face_index, bbox_json, face_confidence, embedding, quality_status,
                       quality_score, quality_reasons_json, quality_revision, mtime_ns, file_size, hidden
                FROM face_index
                ORDER BY image_path, face_index
                """
            ).fetchall()
            scan_rows = connection.execute(
                """
                SELECT image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at
                FROM face_scan_images
                ORDER BY image_path
                """
            ).fetchall()
            prototype_rows = connection.execute(
                "SELECT person_name, embedding, similarity_threshold, example_count FROM person_prototypes ORDER BY person_name"
            ).fetchall()
            profile_rows = connection.execute(
                """
                SELECT person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden
                FROM person_profiles
                ORDER BY person_name
                """
            ).fetchall()
            prototype_face_rows = connection.execute(
                """
                SELECT person_name, image_path, face_index, pinned, sort_order
                FROM person_prototype_faces
                ORDER BY person_name, sort_order, image_path, face_index
                """
            ).fetchall()
            label_rows = connection.execute(
                "SELECT image_path, face_index, person_name, confidence FROM face_labels ORDER BY person_name, image_path, face_index"
            ).fetchall()
            pending_rows = connection.execute(
                """
                SELECT image_path, face_index, person_name, confidence, source, created_at
                FROM pending_face_labels
                ORDER BY created_at, image_path, face_index
                """
            ).fetchall()
            correction_rows = connection.execute(
                """
                SELECT image_path, face_index, person_name, source, created_at
                FROM rejected_face_labels
                ORDER BY created_at, person_name, image_path, face_index
                """
            ).fetchall()
        profiles_by_name = {
            str(row[0]): {
                "person_name": str(row[0]),
                "notes": str(row[1] or ""),
                "tags": list(json.loads(row[2] or "[]")),
                "cover_image_path": str(row[3] or ""),
                "cover_face_index": int(row[4] or 0),
                "favorite": bool(row[5]),
                "birth_date": str(row[6] or ""),
                "hidden": bool(row[7]),
            }
            for row in profile_rows
        }
        identities = []
        for row in prototype_rows:
            name = str(row[0])
            profile = dict(profiles_by_name.get(name, {"person_name": name}))
            profile.update(
                {
                    "similarity_threshold": float(row[2]),
                    "example_count": int(row[3]),
                    "prototype_embedding": np.frombuffer(row[1], dtype=np.float32).astype(float).tolist(),
                    "prototype_refs": [
                        {
                            "image_path": str(ref[1]),
                            "face_index": int(ref[2]),
                            "pinned": bool(ref[3]),
                            "sort_order": int(ref[4]),
                        }
                        for ref in prototype_face_rows
                        if str(ref[0]) == name
                    ],
                }
            )
            identities.append(profile)
        return {
            "version": 1,
            "mode": self.mode,
            "identities": identities,
            "face_index": [
                {
                    "image_path": str(row[0]),
                    "face_index": int(row[1]),
                    "bbox": list(json.loads(row[2] or "[]")),
                    "face_confidence": float(row[3]),
                    "embedding": np.frombuffer(row[4], dtype=np.float32).astype(float).tolist(),
                    "quality_status": str(row[5] or "clean"),
                    "quality_score": float(row[6] or 1.0),
                    "quality_reasons": list(json.loads(row[7] or "[]")),
                    "quality_revision": int(row[8] or 0),
                    "mtime_ns": int(row[9] or 0),
                    "file_size": int(row[10] or 0),
                    "hidden": bool(row[11]),
                }
                for row in face_rows
            ],
            "face_scan_images": [
                {
                    "image_path": str(row[0]),
                    "mtime_ns": int(row[1] or 0),
                    "file_size": int(row[2] or 0),
                    "face_count": int(row[3] or 0),
                    "image_width": int(row[4] or 0),
                    "image_height": int(row[5] or 0),
                    "indexed_at": str(row[6] or ""),
                }
                for row in scan_rows
            ],
            "labels": [
                {
                    "image_path": str(row[0]),
                    "face_index": int(row[1]),
                    "person_name": str(row[2]),
                    "confidence": float(row[3]),
                }
                for row in label_rows
            ],
            "pending_labels": [
                {
                    "image_path": str(row[0]),
                    "face_index": int(row[1]),
                    "person_name": str(row[2]),
                    "confidence": float(row[3]),
                    "source": str(row[4]),
                    "created_at": str(row[5] or ""),
                }
                for row in pending_rows
            ],
            "corrections": [
                {
                    "image_path": str(row[0]),
                    "face_index": int(row[1]),
                    "person_name": str(row[2]),
                    "source": str(row[3]),
                    "created_at": str(row[4] or ""),
                }
                for row in correction_rows
            ],
        }

    def export_identity_data_to_file(self, output_path: str | Path) -> dict[str, object]:
        payload = self.export_identity_data()
        atomic_write_text(Path(output_path), json.dumps(payload, indent=2, sort_keys=True))
        return payload

    def preview_identity_import(self, payload: dict[str, object]) -> dict[str, object]:
        identities = [item for item in list(payload.get("identities", []) or []) if isinstance(item, dict)]
        labels = [item for item in list(payload.get("labels", []) or []) if isinstance(item, dict)]
        face_rows = [item for item in list(payload.get("face_index", []) or []) if isinstance(item, dict)]
        pending = [item for item in list(payload.get("pending_labels", []) or []) if isinstance(item, dict)]
        corrections = [item for item in list(payload.get("corrections", []) or []) if isinstance(item, dict)]
        incoming_names = {str(item.get("person_name", "") or "").strip() for item in identities}
        incoming_names.discard("")
        incoming_label_refs = [
            (str(item.get("image_path", "") or ""), int(item.get("face_index", 0) or 0), str(item.get("person_name", "") or ""))
            for item in labels
        ]
        conflicts: list[dict[str, object]] = []
        with self._connect() as connection:
            if incoming_names:
                placeholders = ",".join("?" for _ in incoming_names)
                existing_profiles = connection.execute(
                    f"SELECT person_name FROM person_profiles WHERE person_name IN ({placeholders})",
                    tuple(sorted(incoming_names)),
                ).fetchall()
                existing_prototypes = connection.execute(
                    f"SELECT person_name FROM person_prototypes WHERE person_name IN ({placeholders})",
                    tuple(sorted(incoming_names)),
                ).fetchall()
                for row in existing_profiles:
                    conflicts.append({"type": "identity", "person_name": str(row[0]), "target": "profile"})
                for row in existing_prototypes:
                    conflicts.append({"type": "identity", "person_name": str(row[0]), "target": "prototype"})
            for image_path, face_index, person_name in incoming_label_refs:
                row = connection.execute(
                    "SELECT person_name FROM face_labels WHERE image_path=? AND face_index=?",
                    (image_path, int(face_index)),
                ).fetchone()
                if row is not None and str(row[0] or "") != str(person_name or ""):
                    conflicts.append(
                        {
                            "type": "label",
                            "image_path": image_path,
                            "face_index": int(face_index),
                            "existing_person_name": str(row[0] or ""),
                            "incoming_person_name": str(person_name or ""),
                        }
                    )
        prototype_refs = sum(len(list(item.get("prototype_refs", []) or [])) for item in identities)
        prototype_embeddings = sum(1 for item in identities if list(item.get("prototype_embedding", []) or []))
        hidden_items = sum(1 for item in identities if bool(item.get("hidden", False))) + sum(1 for item in face_rows if bool(item.get("hidden", False)))
        return {
            "counts": {
                "identities": len(identities),
                "labels": len(labels),
                "prototypes": int(prototype_refs + prototype_embeddings),
                "hidden_items": int(hidden_items),
                "corrections": len(corrections),
                "pending_labels": len(pending),
                "face_index": len(face_rows),
            },
            "conflicts": conflicts,
            "conflict_count": len(conflicts),
        }

    def preview_identity_import_from_file(self, input_path: str | Path) -> dict[str, object]:
        return self.preview_identity_import(json.loads(Path(input_path).read_text(encoding="utf-8")))

    @staticmethod
    def _clear_identity_import_tables(connection) -> None:
        for table in (
            "face_labels",
            "pending_face_labels",
            "rejected_face_labels",
            "person_prototype_faces",
            "person_prototypes",
            "person_profiles",
            "face_scan_images",
            "face_index",
        ):
            connection.execute(f"DELETE FROM {table}")

    def import_identity_data(self, payload: dict[str, object], *, mode: str = "merge") -> None:
        mode = str(mode or "merge").strip().lower()
        if mode not in {"merge", "replace"}:
            raise ValueError("Identity import mode must be 'merge' or 'replace'.")
        with self._connect() as connection:
            if mode == "replace":
                self._clear_identity_import_tables(connection)
            for item in list(payload.get("face_index", []) or []):
                embedding = np.asarray(item.get("embedding", []), dtype=np.float32)
                connection.execute(
                    """
                    INSERT INTO face_index(
                        image_path, face_index, bbox_json, face_confidence, embedding, quality_status,
                        quality_score, quality_reasons_json, quality_revision, mtime_ns, file_size, hidden, is_tiny
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index) DO UPDATE SET
                        bbox_json=excluded.bbox_json,
                        face_confidence=excluded.face_confidence,
                        embedding=excluded.embedding,
                        quality_status=excluded.quality_status,
                        quality_score=excluded.quality_score,
                        quality_reasons_json=excluded.quality_reasons_json,
                        quality_revision=excluded.quality_revision,
                        mtime_ns=excluded.mtime_ns,
                        file_size=excluded.file_size,
                        hidden=excluded.hidden,
                        is_tiny=excluded.is_tiny
                    """,
                    (
                        str(item.get("image_path", "")),
                        int(item.get("face_index", 0)),
                        json.dumps(list(item.get("bbox", ()) or ())),
                        float(item.get("face_confidence", 0.0)),
                        embedding.astype(np.float32).tobytes(),
                        str(item.get("quality_status", "clean") or "clean"),
                        float(item.get("quality_score", 1.0) or 1.0),
                        json.dumps(list(item.get("quality_reasons", ()) or ())),
                        int(item.get("quality_revision", FACE_QUALITY_METADATA_REVISION) or 0),
                        int(item.get("mtime_ns", 0) or 0),
                        int(item.get("file_size", 0) or 0),
                        1 if bool(item.get("hidden", False)) else 0,
                        0 if _is_visible_face_bbox(tuple(int(value) for value in (item.get("bbox", ()) or (0, 0, 0, 0)))) else 1,
                    ),
                )
            for item in list(payload.get("face_scan_images", []) or []):
                connection.execute(
                    """
                    INSERT INTO face_scan_images(image_path, mtime_ns, file_size, face_count, image_width, image_height, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_path) DO UPDATE SET
                        mtime_ns=excluded.mtime_ns,
                        file_size=excluded.file_size,
                        face_count=excluded.face_count,
                        image_width=excluded.image_width,
                        image_height=excluded.image_height,
                        indexed_at=excluded.indexed_at
                    """,
                    (
                        str(item.get("image_path", "")),
                        int(item.get("mtime_ns", 0) or 0),
                        int(item.get("file_size", 0) or 0),
                        int(item.get("face_count", 0) or 0),
                        int(item.get("image_width", 0) or 0),
                        int(item.get("image_height", 0) or 0),
                        str(item.get("indexed_at", "") or ""),
                    ),
                )
            for identity in list(payload.get("identities", []) or []):
                name = str(identity.get("person_name", "") or "").strip()
                if not name:
                    continue
                embedding = np.asarray(identity.get("prototype_embedding", []), dtype=np.float32)
                if embedding.size:
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
                            name,
                            embedding.astype(np.float32).tobytes(),
                            float(identity.get("similarity_threshold", self.recognition_min_score) or self.recognition_min_score),
                            int(identity.get("example_count", 1) or 1),
                        ),
                    )
                connection.execute(
                    """
                    INSERT INTO person_profiles(person_name, notes, tags_json, cover_image_path, cover_face_index, favorite, birth_date, hidden)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_name) DO UPDATE SET
                        notes=excluded.notes,
                        tags_json=excluded.tags_json,
                        cover_image_path=excluded.cover_image_path,
                        cover_face_index=excluded.cover_face_index,
                        favorite=excluded.favorite,
                        birth_date=excluded.birth_date,
                        hidden=excluded.hidden
                    """,
                    (
                        name,
                        str(identity.get("notes", "") or ""),
                        json.dumps(list(identity.get("tags", ()) or ())),
                        str(identity.get("cover_image_path", "") or ""),
                        int(identity.get("cover_face_index", 0) or 0),
                        1 if bool(identity.get("favorite", False)) else 0,
                        str(identity.get("birth_date", "") or ""),
                        1 if bool(identity.get("hidden", False)) else 0,
                    ),
                )
                for ref in list(identity.get("prototype_refs", []) or []):
                    connection.execute(
                        """
                        INSERT INTO person_prototype_faces(person_name, image_path, face_index, pinned, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(person_name, image_path, face_index) DO UPDATE SET
                            pinned=excluded.pinned,
                            sort_order=excluded.sort_order
                        """,
                        (
                            name,
                            str(ref.get("image_path", "") or ""),
                            int(ref.get("face_index", 0) or 0),
                            1 if bool(ref.get("pinned", False)) else 0,
                            int(ref.get("sort_order", 0) or 0),
                        ),
                    )
            for item in list(payload.get("labels", []) or []):
                connection.execute(
                    """
                    INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index) DO UPDATE SET
                        person_name=excluded.person_name,
                        confidence=excluded.confidence
                    """,
                    (
                        str(item.get("image_path", "") or ""),
                        int(item.get("face_index", 0) or 0),
                        str(item.get("person_name", "") or ""),
                        float(item.get("confidence", 0.0) or 0.0),
                    ),
                )
            for item in list(payload.get("pending_labels", []) or []):
                connection.execute(
                    """
                    INSERT INTO pending_face_labels(image_path, face_index, person_name, confidence, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index, source) DO UPDATE SET
                        person_name=excluded.person_name,
                        confidence=excluded.confidence,
                        created_at=excluded.created_at
                    """,
                    (
                        str(item.get("image_path", "") or ""),
                        int(item.get("face_index", 0) or 0),
                        str(item.get("person_name", "") or ""),
                        float(item.get("confidence", 0.0) or 0.0),
                        str(item.get("source", "pending") or "pending"),
                        str(item.get("created_at", "") or ""),
                    ),
                )
            for item in list(payload.get("corrections", []) or []):
                connection.execute(
                    """
                    INSERT INTO rejected_face_labels(image_path, face_index, person_name, source, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index, person_name, source) DO UPDATE SET
                        created_at=excluded.created_at
                    """,
                    (
                        str(item.get("image_path", "") or ""),
                        int(item.get("face_index", 0) or 0),
                        str(item.get("person_name", "") or ""),
                        str(item.get("source", "pending") or "pending"),
                        str(item.get("created_at", "") or ""),
                    ),
                )
        self._invalidate_identity_caches()
        self._invalidate_ann_index()

    def import_identity_data_from_file(self, input_path: str | Path, *, mode: str = "merge") -> None:
        self.import_identity_data(json.loads(Path(input_path).read_text(encoding="utf-8")), mode=mode)

    def prepare_face_data_purge(self) -> dict[str, object]:
        table_counts: dict[str, int] = {}
        tables = [
            "face_index",
            "face_scan_images",
            "person_prototypes",
            "person_profiles",
            "person_prototype_faces",
            "face_labels",
            "pending_face_labels",
            "rejected_face_labels",
        ]
        with self._connect() as connection:
            for table in tables:
                try:
                    table_counts[table] = int(connection.execute(f"SELECT COUNT(1) FROM {table}").fetchone()[0] or 0)
                except Exception:
                    table_counts[table] = 0
        ann_files = [str(path) for path in (self._ann_index_path(), self._ann_meta_path()) if path.exists()]
        model_targets: list[str] = []
        try:
            target = face_model_runtime_root_dir() / self.mode
            if target.exists():
                model_targets.append(str(target))
        except Exception:
            model_targets = []
        return {
            "table_counts": table_counts,
            "ann_files": ann_files,
            "model_cache_targets": model_targets,
        }

    def purge_face_data(self, *, confirm: bool = False) -> dict[str, object]:
        if not confirm:
            raise ValueError("Face data purge requires explicit confirmation.")
        report = self.prepare_face_data_purge()
        with self._connect() as connection:
            for table in [
                "face_index",
                "face_scan_images",
                "person_prototypes",
                "person_profiles",
                "person_prototype_faces",
                "face_labels",
                "pending_face_labels",
                "rejected_face_labels",
            ]:
                connection.execute(f"DELETE FROM {table}")
            self._record_action_audit(
                "purge_face_data",
                target=self.mode,
                details={
                    "table_counts": dict(report.get("table_counts", {}) or {}),
                    "ann_file_count": len(list(report.get("ann_files", []) or [])),
                    "model_cache_target_count": len(list(report.get("model_cache_targets", []) or [])),
                },
                reversible=False,
                connection=connection,
            )
        removed_files: list[str] = []
        for path_text in list(report.get("ann_files", []) or []):
            path = Path(str(path_text))
            try:
                path.unlink(missing_ok=True)
                removed_files.append(str(path))
            except Exception:
                pass
        removed_targets: list[str] = []
        for path_text in list(report.get("model_cache_targets", []) or []):
            path = Path(str(path_text))
            try:
                if path.exists():
                    shutil.rmtree(path)
                    removed_targets.append(str(path))
            except Exception:
                pass
        self._invalidate_identity_caches()
        self._ann_index = None
        self._ann_records = []
        self._ann_fingerprint = None
        return {
            "before": report,
            "removed_ann_files": removed_files,
            "removed_model_cache_targets": removed_targets,
        }

    def auto_propagate_labels(self, *, include_tiny_faces: bool = True) -> list[FaceLabelAssignment]:
        prototypes = self.load_person_prototypes()
        records = [
            record
            for record in self.load_all_records(include_tiny_faces=include_tiny_faces)
            if self._quality_allows(record.quality_status, self.search_quality_min)
        ]
        if not prototypes or not records:
            return []
        assignments = []
        for record in records:
            if str(record.person_name or "").strip():
                continue
            best_person = None
            best_score = -1.0
            for prototype in prototypes:
                score = float(np.dot(record.embedding, prototype.embedding))
                threshold = max(float(prototype.similarity_threshold), float(self.auto_label_min_score))
                if score >= threshold and score > best_score:
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
                        source="auto_propagate",
                        pending=True,
                    )
                )
        return self._queue_face_label_assignments(assignments)

    def _queue_face_label_assignments(self, assignments: list[FaceLabelAssignment]) -> list[FaceLabelAssignment]:
        queued: list[FaceLabelAssignment] = []
        rows = [
            (
                assignment.image_path,
                int(assignment.face_index),
                str(assignment.person_name),
                float(assignment.confidence),
                str(assignment.source or "pending"),
            )
            for assignment in assignments
            if str(assignment.person_name or "").strip()
        ]
        if not rows:
            return []
        with self._connect() as connection:
            filtered_rows: list[tuple[str, int, str, float, str]] = []
            for image_path, face_index, person_name, confidence, source in rows:
                rejected = connection.execute(
                    """
                    SELECT 1
                    FROM rejected_face_labels
                    WHERE image_path=? AND face_index=? AND person_name=? AND source=?
                    """,
                    (str(image_path), int(face_index), str(person_name), str(source)),
                ).fetchone()
                if rejected is None:
                    filtered_rows.append((str(image_path), int(face_index), str(person_name), float(confidence), str(source)))
            rows = filtered_rows
            if not rows:
                return []
            connection.executemany(
                """
                INSERT INTO pending_face_labels(image_path, face_index, person_name, confidence, source)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_path, face_index, source) DO UPDATE SET
                    person_name=excluded.person_name,
                    confidence=excluded.confidence,
                    created_at=CURRENT_TIMESTAMP
                """,
                rows,
            )
        self._invalidate_identity_caches()
        refs = {(row[0], int(row[1]), row[4]) for row in rows}
        for assignment in self.load_pending_face_labels(include_tiny_faces=True):
            if (assignment.image_path, int(assignment.face_index), str(assignment.source or "")) in refs:
                queued.append(assignment)
        return queued

    def load_pending_face_labels(
        self,
        *,
        folder_prefix: str = "",
        candidate_paths: list[str] | None = None,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[FaceLabelAssignment]:
        folder_prefix = str(folder_prefix or "").strip()
        allowed_paths = self._normalized_path_set(candidate_paths)
        query = """
            SELECT p.proposal_id, p.person_name, p.image_path, p.face_index, i.bbox_json, p.confidence, p.source, p.created_at
            FROM pending_face_labels p
            JOIN face_index i ON i.image_path=p.image_path AND i.face_index=p.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=p.person_name
        """
        args: list[object] = []
        if folder_prefix:
            scope_clause, scope_args = folder_scope_sql("p.image_path", folder_prefix)
            query += f" WHERE {scope_clause}"
            args.extend(scope_args)
        if not include_tiny_faces:
            query = self._append_sql_condition(query, "COALESCE(i.is_tiny, 0)=0")
        if not include_hidden:
            query = self._append_sql_condition(query, "COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0")
        query += " ORDER BY p.created_at DESC, p.proposal_id DESC"
        with self._connect() as connection:
            rows = self._execute_scoped_query(
                connection,
                query,
                args,
                image_column="p.image_path",
                candidate_paths=candidate_paths,
            )
        rows.sort(key=lambda row: (str(row[7] or ""), int(row[0] or 0)), reverse=True)
        assignments: list[FaceLabelAssignment] = []
        for row in rows:
            bbox = tuple(json.loads(row[4]))
            image_path = str(row[2] or "")
            if not self._path_in_scope(image_path, folder_prefix=folder_prefix, candidate_paths=allowed_paths):
                continue
            assignments.append(
                FaceLabelAssignment(
                    proposal_id=int(row[0] or 0),
                    person_name=str(row[1] or ""),
                    image_path=image_path,
                    face_index=int(row[3] or 0),
                    face_bbox=bbox,
                    confidence=float(row[5] or 0.0),
                    source=str(row[6] or ""),
                    created_at=str(row[7] or ""),
                    pending=True,
                )
            )
        return assignments

    def accept_pending_face_labels_batch(self, proposal_ids: list[int]) -> FaceLabelAcceptanceBatch:
        ids = sorted({int(value) for value in proposal_ids if int(value) > 0})
        if not ids:
            return FaceLabelAcceptanceBatch((), ())
        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT p.proposal_id, p.person_name, p.image_path, p.face_index, i.bbox_json, p.confidence, p.source, p.created_at
                FROM pending_face_labels
                p JOIN face_index i ON i.image_path=p.image_path AND i.face_index=p.face_index
                WHERE p.proposal_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            if not rows:
                return FaceLabelAcceptanceBatch((), ())
            previous_rows = connection.execute(
                f"""
                SELECT l.image_path, l.face_index, i.bbox_json, l.person_name, l.confidence
                FROM face_labels l
                JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
                WHERE (l.image_path, l.face_index) IN (
                    SELECT image_path, face_index FROM pending_face_labels WHERE proposal_id IN ({placeholders})
                )
                """,
                ids,
            ).fetchall()
            connection.executemany(
                """
                INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(image_path, face_index) DO UPDATE SET
                    person_name=excluded.person_name,
                    confidence=excluded.confidence
                """,
                [(str(row[2]), int(row[3]), str(row[1]), float(row[5])) for row in rows],
            )
            connection.execute(
                f"DELETE FROM pending_face_labels WHERE proposal_id IN ({placeholders})",
                ids,
            )
            self._record_action_audit(
                "accept_pending_face_labels",
                target=", ".join(sorted({str(row[1]) for row in rows})),
                details={"proposal_ids": ids, "accepted_count": len(rows)},
                reversible=True,
                connection=connection,
            )
        accepted: list[FaceLabelAssignment] = []
        for row in rows:
            accepted.append(
                FaceLabelAssignment(
                    proposal_id=int(row[0] or 0),
                    person_name=str(row[1] or ""),
                    image_path=str(row[2] or ""),
                    face_index=int(row[3] or 0),
                    face_bbox=tuple(json.loads(row[4] or "[]")),
                    confidence=float(row[5] or 0.0),
                    source=str(row[6] or ""),
                    created_at=str(row[7] or ""),
                )
            )
        previous_labels = [
            FaceLabelAssignment(
                person_name=str(row[3] or ""),
                image_path=str(row[0] or ""),
                face_index=int(row[1] or 0),
                face_bbox=tuple(json.loads(row[2] or "[]")),
                confidence=float(row[4] or 0.0),
            )
            for row in previous_rows
        ]
        batch = FaceLabelAcceptanceBatch(
            accepted=tuple(accepted),
            previous_labels=tuple(previous_labels),
        )
        self._last_pending_face_acceptance_batch = batch
        self._invalidate_identity_caches()
        return batch

    def accept_pending_face_labels(self, proposal_ids: list[int]) -> list[FaceLabelAssignment]:
        return list(self.accept_pending_face_labels_batch(proposal_ids).accepted)

    def undo_last_pending_face_acceptance(self) -> list[FaceLabelAssignment]:
        batch = self._last_pending_face_acceptance_batch
        if batch is None or not batch.accepted:
            return []
        accepted = list(batch.accepted)
        previous_labels = list(batch.previous_labels)
        refs = [(assignment.image_path, int(assignment.face_index)) for assignment in accepted]
        with self._connect() as connection:
            connection.executemany(
                "DELETE FROM face_labels WHERE image_path=? AND face_index=?",
                refs,
            )
            if previous_labels:
                connection.executemany(
                    """
                    INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index) DO UPDATE SET
                        person_name=excluded.person_name,
                        confidence=excluded.confidence
                    """,
                    [
                        (
                            assignment.image_path,
                            int(assignment.face_index),
                            str(assignment.person_name),
                            float(assignment.confidence),
                        )
                        for assignment in previous_labels
                    ],
                )
            connection.executemany(
                """
                INSERT INTO pending_face_labels(proposal_id, image_path, face_index, person_name, confidence, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_path, face_index, source) DO UPDATE SET
                    proposal_id=excluded.proposal_id,
                    person_name=excluded.person_name,
                    confidence=excluded.confidence,
                    created_at=excluded.created_at
                """,
                [
                    (
                        int(assignment.proposal_id),
                        str(assignment.image_path),
                        int(assignment.face_index),
                        str(assignment.person_name),
                        float(assignment.confidence),
                        str(assignment.source or "pending"),
                        str(assignment.created_at or ""),
                    )
                    for assignment in accepted
                    if int(assignment.proposal_id or 0) > 0
                ],
            )
            self._record_action_audit(
                "undo_pending_face_acceptance",
                target=", ".join(sorted({str(item.person_name) for item in accepted})),
                details={"restored_count": len(accepted)},
                reversible=False,
                connection=connection,
            )
        self._last_pending_face_acceptance_batch = None
        self._invalidate_identity_caches()
        return accepted

    def reject_pending_face_labels(self, proposal_ids: list[int]) -> int:
        ids = sorted({int(value) for value in proposal_ids if int(value) > 0})
        if not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT image_path, face_index, person_name, source
                FROM pending_face_labels
                WHERE proposal_id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            if rows:
                connection.executemany(
                    """
                    INSERT INTO rejected_face_labels(image_path, face_index, person_name, source)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index, person_name, source) DO UPDATE SET
                        created_at=CURRENT_TIMESTAMP
                    """,
                    [(str(row[0]), int(row[1]), str(row[2]), str(row[3] or "pending")) for row in rows],
                )
            cursor = connection.execute(
                f"DELETE FROM pending_face_labels WHERE proposal_id IN ({placeholders})",
                ids,
            )
            self._record_action_audit(
                "reject_pending_face_labels",
                target=", ".join(sorted({str(row[2]) for row in rows})),
                details={"proposal_ids": ids, "rejected_count": int(getattr(cursor, "rowcount", 0) or 0)},
                reversible=False,
                connection=connection,
            )
        self._invalidate_identity_caches()
        return int(getattr(cursor, "rowcount", 0) or 0)

    def clear_face_label_corrections(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM rejected_face_labels")

    def list_rejected_face_label_corrections(self) -> list[FaceLabelAssignment]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.person_name, r.image_path, r.face_index, COALESCE(i.bbox_json, '[0,0,0,0]'), r.source, r.created_at
                FROM rejected_face_labels r
                LEFT JOIN face_index i ON i.image_path=r.image_path AND i.face_index=r.face_index
                ORDER BY r.created_at DESC, r.person_name, r.image_path, r.face_index
                """
            ).fetchall()
        return [
            FaceLabelAssignment(
                person_name=str(row[0] or ""),
                image_path=str(row[1] or ""),
                face_index=int(row[2] or 0),
                face_bbox=tuple(json.loads(row[3] or "[0,0,0,0]")),
                confidence=0.0,
                source=str(row[4] or ""),
                created_at=str(row[5] or ""),
                pending=False,
            )
            for row in rows
        ]

    def list_face_labels(self, *, include_tiny_faces: bool = True, include_hidden: bool = False) -> list[FaceLabelAssignment]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.person_name, f.image_path, f.face_index, i.bbox_json, f.confidence,
                       COALESCE(i.hidden, 0), COALESCE(pp.hidden, 0)
                FROM face_labels f
                JOIN face_index i ON i.image_path=f.image_path AND i.face_index=f.face_index
                LEFT JOIN person_profiles pp ON pp.person_name=f.person_name
                ORDER BY f.person_name, f.confidence DESC, f.image_path
                """
            ).fetchall()
        assignments: list[FaceLabelAssignment] = []
        for row in rows:
            bbox = tuple(json.loads(row[3]))
            if not self._face_is_visible(bbox, include_tiny_faces=include_tiny_faces):
                continue
            if (bool(row[5]) or bool(row[6])) and not include_hidden:
                continue
            assignments.append(
                FaceLabelAssignment(
                    person_name=row[0],
                    image_path=row[1],
                    face_index=int(row[2]),
                    face_bbox=bbox,
                    confidence=float(row[4]),
                )
            )
        return assignments

    def list_named_photo_summaries(
        self,
        *,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[NamedPhotoSummary]:
        """Return durable names with visible face and distinct-photo counts.

        This intentionally reads ``face_labels`` rather than prototypes or the
        review queue.  A person can therefore remain visible in Names even if
        an older database contains labels without a matching profile row.
        """

        query = """
            SELECT l.person_name, COUNT(1), COUNT(DISTINCT l.image_path)
            FROM face_labels l
            JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
        """
        conditions: list[str] = []
        if not include_tiny_faces:
            conditions.append("COALESCE(i.is_tiny, 0)=0")
        if not include_hidden:
            conditions.append("COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY l.person_name ORDER BY lower(l.person_name), l.person_name"
        with self._connect() as connection:
            rows = connection.execute(query).fetchall()
        return [
            NamedPhotoSummary(
                person_name=str(row[0] or ""),
                face_count=int(row[1] or 0),
                photo_count=int(row[2] or 0),
            )
            for row in rows
            if str(row[0] or "").strip()
        ]

    def list_named_photo_paths(
        self,
        person_name: str,
        *,
        include_tiny_faces: bool = True,
        include_hidden: bool = False,
    ) -> list[str]:
        """Return unique global photo paths with a durable label for ``person_name``."""

        name = str(person_name or "").strip()
        if not name:
            return []
        query = """
            SELECT DISTINCT l.image_path
            FROM face_labels l
            JOIN face_index i ON i.image_path=l.image_path AND i.face_index=l.face_index
            LEFT JOIN person_profiles pp ON pp.person_name=l.person_name
            WHERE l.person_name=?
        """
        args: list[object] = [name]
        if not include_tiny_faces:
            query += " AND COALESCE(i.is_tiny, 0)=0"
        if not include_hidden:
            query += " AND COALESCE(i.hidden, 0)=0 AND COALESCE(pp.hidden, 0)=0"
        query += " ORDER BY l.image_path"
        with self._connect() as connection:
            rows = connection.execute(query, args).fetchall()
        return list(dict.fromkeys(str(row[0] or "") for row in rows if str(row[0] or "").strip()))

    def label_unlabeled_faces_in_images(self, person_name: str, image_paths: list[str]) -> int:
        """Save ``person_name`` on visible, currently-unlabeled faces in selected photos.

        Photo selection is intentionally only a way to choose a bounded set of
        face records. Existing labels in a mixed-person photo are left alone.
        """

        target = str(person_name or "").strip()
        if not target:
            raise ValueError("Identity name is required.")
        refs = self._selected_image_face_refs(image_paths, unlabeled_only=True)
        return self._apply_selected_image_label_change(
            target_name=target,
            face_refs=refs,
            action="name_selected_images",
        )

    def rename_labeled_faces_in_images(self, source_name: str, target_name: str, image_paths: list[str]) -> int:
        """Rename only ``source_name`` face labels in the selected photos."""

        source = str(source_name or "").strip()
        target = str(target_name or "").strip()
        if not source or not target:
            raise ValueError("Both the current and new identity names are required.")
        if source == target:
            return 0
        refs = self._selected_image_face_refs(image_paths, person_name=source)
        return self._apply_selected_image_label_change(
            source_name=source,
            target_name=target,
            face_refs=refs,
            action="rename_selected_image_labels",
        )

    def unlabel_labeled_faces_in_images(self, person_name: str, image_paths: list[str]) -> int:
        """Remove only ``person_name`` face labels in the selected photos."""

        source = str(person_name or "").strip()
        if not source:
            raise ValueError("The current identity name is required.")
        refs = self._selected_image_face_refs(image_paths, person_name=source)
        return self._apply_selected_image_label_change(
            source_name=source,
            face_refs=refs,
            action="unlabel_selected_image_labels",
        )

    def _selected_image_face_refs(
        self,
        image_paths: list[str],
        *,
        person_name: str = "",
        unlabeled_only: bool = False,
    ) -> list[tuple[str, int]]:
        """Return visible face rows in explicitly selected images only."""

        paths = self._candidate_path_query_values(image_paths)
        if not paths:
            return []
        name = str(person_name or "").strip()
        query = """
            SELECT i.image_path, i.face_index
            FROM face_index i
            LEFT JOIN face_labels l ON l.image_path=i.image_path AND l.face_index=i.face_index
            WHERE COALESCE(i.hidden, 0)=0
        """
        params: list[object] = []
        if name:
            query += " AND l.person_name=?"
            params.append(name)
        elif unlabeled_only:
            query += " AND l.person_name IS NULL"
        query += " ORDER BY i.image_path, i.face_index"
        with self._connect() as connection:
            rows = self._execute_scoped_query(
                connection,
                query,
                params,
                image_column="i.image_path",
                candidate_paths=paths,
            )
        return list(dict.fromkeys((str(row[0]), int(row[1])) for row in rows if str(row[0] or "").strip()))

    def _apply_selected_image_label_change(
        self,
        *,
        source_name: str = "",
        target_name: str = "",
        face_refs: list[tuple[str, int]],
        action: str,
    ) -> int:
        """Persist a bounded face-label mutation and repair touched identities.

        ``face_labels`` is authoritative for Names. Prototype rows are derived
        identity data, so they are reconciled immediately after the durable
        transaction and never used to broaden a photo-level selection.
        """

        source = str(source_name or "").strip()
        target = str(target_name or "").strip()
        normalized_refs = list(
            dict.fromkeys(
                (str(image_path), int(face_index))
                for image_path, face_index in list(face_refs or [])
                if str(image_path or "").strip()
            )
        )
        if not normalized_refs:
            return 0

        records: list[IndexedFaceRecord] = []
        for image_path, face_index in normalized_refs:
            record = self.load_face_record(image_path, face_index)
            if record is not None and self._quality_allows(record.quality_status, self.prototype_quality_min):
                records.append(record)
        if not records:
            return 0
        refs = [(record.image_path, int(record.face_index)) for record in records]

        with self._connect() as connection:
            # The Name action promises not to overwrite an already-saved
            # label. Serialize this check with the subsequent write so a
            # concurrent face workflow cannot turn a previously-unlabeled
            # selected row into somebody else's label between the two steps.
            connection.execute("BEGIN IMMEDIATE")
            applicable_refs: list[tuple[str, int]] = []
            for image_path, face_index in refs:
                row = connection.execute(
                    "SELECT person_name FROM face_labels WHERE image_path=? AND face_index=?",
                    (image_path, face_index),
                ).fetchone()
                existing_name = str(row[0] or "").strip() if row is not None else ""
                if source:
                    if existing_name == source:
                        applicable_refs.append((image_path, face_index))
                elif not existing_name:
                    applicable_refs.append((image_path, face_index))
            if not applicable_refs:
                return 0

            if target:
                target_existing = connection.execute(
                    "SELECT COUNT(1) FROM person_prototype_faces WHERE person_name=?",
                    (target,),
                ).fetchone()
                next_sort_row = connection.execute(
                    "SELECT COALESCE(MAX(sort_order), -1) FROM person_prototype_faces WHERE person_name=?",
                    (target,),
                ).fetchone()
                next_sort_order = int((next_sort_row or (-1,))[0]) + 1
                target_has_example = bool(int((target_existing or [0])[0] or 0))
                connection.executemany(
                    """
                    INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(image_path, face_index) DO UPDATE SET
                        person_name=excluded.person_name,
                        confidence=excluded.confidence
                    """,
                    [(image_path, face_index, target, 1.0) for image_path, face_index in applicable_refs],
                )
                connection.executemany(
                    "DELETE FROM pending_face_labels WHERE image_path=? AND face_index=?",
                    applicable_refs,
                )
                for offset, (image_path, face_index) in enumerate(applicable_refs):
                    connection.execute(
                        """
                        INSERT INTO person_prototype_faces(person_name, image_path, face_index, pinned, sort_order)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(person_name, image_path, face_index) DO NOTHING
                        """,
                        (
                            target,
                            image_path,
                            face_index,
                            1 if not target_has_example and offset == 0 else 0,
                            next_sort_order + offset,
                        ),
                    )
            else:
                connection.executemany(
                    "DELETE FROM face_labels WHERE image_path=? AND face_index=? AND person_name=?",
                    [(image_path, face_index, source) for image_path, face_index in applicable_refs],
                )

            if source:
                connection.executemany(
                    "DELETE FROM person_prototype_faces WHERE person_name=? AND image_path=? AND face_index=?",
                    [(source, image_path, face_index) for image_path, face_index in applicable_refs],
                )
            self._record_action_audit(
                action,
                target=target or source,
                details={
                    "source_name": source,
                    "target_name": target,
                    "face_count": len(applicable_refs),
                },
                reversible=False,
                connection=connection,
            )

        self._invalidate_identity_caches()
        if target:
            self._reconcile_identity_after_selected_label_change(target)
        if source and source != target:
            self._reconcile_identity_after_selected_label_change(source)
        self._invalidate_identity_caches()
        return len(applicable_refs)

    def _reconcile_identity_after_selected_label_change(self, person_name: str) -> None:
        """Keep one touched identity's search prototype consistent with labels."""

        name = str(person_name or "").strip()
        if not name:
            return
        with self._connect() as connection:
            has_labels = connection.execute(
                "SELECT 1 FROM face_labels WHERE person_name=? LIMIT 1",
                (name,),
            ).fetchone()
            if has_labels is None:
                connection.execute("DELETE FROM person_prototype_faces WHERE person_name=?", (name,))
                connection.execute("DELETE FROM person_prototypes WHERE person_name=?", (name,))
                connection.execute("DELETE FROM person_profiles WHERE person_name=?", (name,))
                return
            rows = connection.execute(
                """
                SELECT p.image_path, p.face_index, p.pinned
                FROM person_prototype_faces p
                JOIN face_labels l ON l.image_path=p.image_path AND l.face_index=p.face_index
                WHERE p.person_name=? AND l.person_name=?
                ORDER BY p.pinned DESC, p.sort_order ASC, p.image_path ASC, p.face_index ASC
                """,
                (name, name),
            ).fetchall()
            if not rows:
                rows = connection.execute(
                    """
                    SELECT l.image_path, l.face_index, 0
                    FROM face_labels l
                    WHERE l.person_name=?
                    ORDER BY l.confidence DESC, l.image_path ASC, l.face_index ASC
                    LIMIT 64
                    """,
                    (name,),
                ).fetchall()

        pinned_ref = next(
            ((str(row[0]), int(row[1])) for row in rows if bool(row[2])),
            None,
        )
        records: list[IndexedFaceRecord] = []
        for image_path, face_index, _pinned in rows:
            record = self.load_face_record(str(image_path), int(face_index))
            if record is not None and self._quality_allows(record.quality_status, self.prototype_quality_min):
                records.append(record)
        if not records:
            return
        existing = self.find_person_prototype(name)
        threshold = float(existing.similarity_threshold) if existing is not None else self.recognition_min_score
        self._save_person_prototype(
            name,
            np.asarray([record.embedding for record in records], dtype=np.float32),
            similarity_threshold=threshold,
            prototype_face_refs=[(record.image_path, int(record.face_index)) for record in records],
        )
        if pinned_ref is not None and pinned_ref in {(record.image_path, int(record.face_index)) for record in records}:
            self.pin_person_prototype_face(name, pinned_ref[0], pinned_ref[1])

    def recover_legacy_manual_face_labels(self) -> ManualLabelRecoveryResult:
        """Promote old explicit manual labels that were accidentally queued.

        Earlier UI versions routed ``manual_selected_faces`` through the
        review queue.  Explicit user input should be durable, but recovery
        must never overwrite a label saved later by the user.  Conflicting
        proposals are deliberately left pending for the existing review UI.
        """

        promoted: list[tuple[str, int, str, float]] = []
        resolved_ids: list[int] = []
        duplicate_count = 0
        conflict_count = 0
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT proposal_id, image_path, face_index, person_name, confidence
                FROM pending_face_labels
                WHERE source='manual_selected_faces'
                ORDER BY proposal_id
                """
            ).fetchall()
            for row in rows:
                proposal_id = int(row[0] or 0)
                image_path = str(row[1] or "")
                face_index = int(row[2] or 0)
                person_name = str(row[3] or "").strip()
                confidence = float(row[4] or 1.0)
                if not proposal_id or not image_path or not person_name:
                    continue
                existing = connection.execute(
                    "SELECT person_name FROM face_labels WHERE image_path=? AND face_index=?",
                    (image_path, face_index),
                ).fetchone()
                if existing is not None:
                    if str(existing[0] or "").strip() == person_name:
                        resolved_ids.append(proposal_id)
                        duplicate_count += 1
                    else:
                        conflict_count += 1
                    continue
                face_exists = connection.execute(
                    "SELECT 1 FROM face_index WHERE image_path=? AND face_index=?",
                    (image_path, face_index),
                ).fetchone()
                if face_exists is None:
                    continue
                promoted.append((image_path, face_index, person_name, confidence))
                resolved_ids.append(proposal_id)
            if promoted:
                connection.executemany(
                    """
                    INSERT INTO face_labels(image_path, face_index, person_name, confidence)
                    VALUES (?, ?, ?, ?)
                    """,
                    promoted,
                )
            if resolved_ids:
                connection.executemany(
                    "DELETE FROM pending_face_labels WHERE proposal_id=?",
                    [(proposal_id,) for proposal_id in resolved_ids],
                )
            if promoted or duplicate_count or conflict_count:
                self._record_action_audit(
                    "recover_legacy_manual_face_labels",
                    details={
                        "promoted_count": len(promoted),
                        "duplicate_count": duplicate_count,
                        "conflict_count": conflict_count,
                    },
                    reversible=False,
                    connection=connection,
                )
        if promoted or duplicate_count:
            self._invalidate_identity_caches()
        return ManualLabelRecoveryResult(
            promoted_count=len(promoted),
            duplicate_count=duplicate_count,
            conflict_count=conflict_count,
        )

    def merge_person_labels(self, source_name: str, target_name: str) -> None:
        self.merge_person_identities(source_name, target_name)

    def clear_person_labels(self, person_name: str) -> None:
        person_name = person_name.strip()
        if not person_name:
            return
        with self._connect() as connection:
            connection.execute("DELETE FROM face_labels WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM pending_face_labels WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM rejected_face_labels WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM person_prototypes WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM person_prototype_faces WHERE person_name=?", (person_name,))
            connection.execute("DELETE FROM person_profiles WHERE person_name=?", (person_name,))
        self._invalidate_identity_caches()





