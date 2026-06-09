from __future__ import annotations

from dataclasses import asdict, dataclass, field


SUPPORTED_SIMILARITY_MODES = ("semantic", "cosine")


@dataclass(frozen=True)
class ProductionClusterRequest:
    directory: str
    embedding_models: list[str]
    num_clusters: int
    clustering_backends: list[str]
    recursive: bool
    similarity_mode: str
    outlier_policy: str
    use_onnx: bool
    reuse_result_cache: bool
    use_embedding_cache_lookup: bool
    source_paths: list[str] | None = None
    performance_profile: str = "balanced"
    preferred_execution_mode: str = "auto"
    tag_filter: list[str] = field(default_factory=list)
    tag_match: str = "Any"
    similarity_modes: list[str] = field(default_factory=list)
    generate_cluster_meanings: bool = False
    generate_cluster_explanations: bool = True
    cluster_meaning_model: str = "auto"
    allow_model_downloads: bool = True

    def __post_init__(self) -> None:
        modes = _normalize_similarity_modes(self.similarity_modes, self.similarity_mode)
        object.__setattr__(self, "similarity_modes", modes)
        object.__setattr__(self, "similarity_mode", modes[0])
        object.__setattr__(self, "allow_model_downloads", bool(self.allow_model_downloads))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def json_line(event_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {"type": event_type, "payload": payload}


def _normalize_similarity_modes(similarity_modes: list[str] | None, legacy_similarity_mode: str | None) -> list[str]:
    raw_modes = list(similarity_modes or [])
    if not raw_modes and legacy_similarity_mode:
        raw_modes = [legacy_similarity_mode]
    normalized: list[str] = []
    for mode in raw_modes or ["semantic"]:
        cleaned = str(mode or "").strip().lower()
        if cleaned not in SUPPORTED_SIMILARITY_MODES:
            cleaned = "semantic"
        if cleaned not in normalized:
            normalized.append(cleaned)
    return normalized or ["semantic"]
