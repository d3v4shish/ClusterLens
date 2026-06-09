from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from infra.settings import get_settings
from .similarity_modes import SIMILARITY_SPACE_VERSION


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
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def load(self, result_key: str) -> tuple[dict[int, list[str]], dict[str, Any]] | None:
        path = self.cache_dir / f"{result_key}.json"
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        clusters = {int(key): value for key, value in payload["clusters"].items()}
        return clusters, payload.get("metrics", {})

    def save(self, result_key: str, clusters: dict[int, list[str]], metrics: dict[str, Any]) -> None:
        path = self.cache_dir / f"{result_key}.json"
        path.write_text(
            json.dumps({"clusters": clusters, "metrics": metrics}, indent=2),
            encoding="utf-8",
        )
