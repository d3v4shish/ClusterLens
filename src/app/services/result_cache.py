from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from infra.settings import get_settings
from infra.atomic_io import atomic_write_text
from .similarity_modes import SIMILARITY_SPACE_VERSION


LOGGER = logging.getLogger(__name__)
RESULT_CACHE_SCHEMA_VERSION = 2


class ResultCacheService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.cache_dir = self.settings.cache_dir / "cluster_results"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_result_key(
        snapshot_key: str,
        embedding_model: str,
        similarity_mode: str,
        clustering_backend: str,
        num_clusters: int,
        outlier_policy: str,
        use_onnx: bool,
        embedding_signature: str = "",
        similarity_space_version: str = SIMILARITY_SPACE_VERSION,
    ) -> str:
        payload = json.dumps(
            {
                "snapshot_key": snapshot_key,
                "embedding_model": embedding_model,
                "similarity_mode": similarity_mode,
                "similarity_space_version": similarity_space_version,
                "clustering_backend": clustering_backend,
                "num_clusters": num_clusters,
                "outlier_policy": outlier_policy,
                "use_onnx": use_onnx,
                "embedding_signature": str(embedding_signature or ""),
                "cache_schema_version": RESULT_CACHE_SCHEMA_VERSION,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, result_key: str) -> tuple[dict[int, list[str]], dict[str, Any]] | None:
        path = self.cache_dir / f"{result_key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or int(payload.get("schema_version", 0) or 0) != RESULT_CACHE_SCHEMA_VERSION:
                return None
            raw_clusters = payload.get("clusters")
            if not isinstance(raw_clusters, dict):
                return None
            clusters: dict[int, list[str]] = {}
            for key, value in raw_clusters.items():
                if not isinstance(value, (list, tuple)) or not all(isinstance(path, str) for path in value):
                    return None
                cluster_id = int(key)
                if cluster_id in clusters:
                    return None
                clusters[cluster_id] = list(value)
            metrics = payload.get("metrics", {})
            return clusters, dict(metrics) if isinstance(metrics, dict) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            LOGGER.warning("Ignoring invalid clustering result cache %s: %s", path, exc)
            return None

    def save(self, result_key: str, clusters: dict[int, list[str]], metrics: dict[str, Any]) -> None:
        path = self.cache_dir / f"{result_key}.json"
        atomic_write_text(
            path,
            json.dumps(
                {
                    "schema_version": RESULT_CACHE_SCHEMA_VERSION,
                    "clusters": clusters,
                    "metrics": metrics,
                },
                indent=2,
            ),
        )
