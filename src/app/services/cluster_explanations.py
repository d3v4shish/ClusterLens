from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterExplanation:
    cluster_id: int
    cluster_size: int
    is_outlier: bool
    cohesion_mean: float | None = None
    cohesion_median: float | None = None
    cohesion_min: float | None = None
    cohesion_max: float | None = None
    nearest_cluster_id: int | None = None
    nearest_cluster_similarity: float | None = None
    separation_margin: float | None = None
    representative_images_note: str = ""
    cluster_quality_score: float | None = None
    similarity_mode: str = ""
    similarity_space: str = ""
    input_dimension: int | None = None
    prepared_dimension: int | None = None
    pca_components: int | None = None
    member_cohesion_scores: tuple[float, ...] = ()

    def as_context(self) -> dict[str, object]:
        return {
            "cluster_id": int(self.cluster_id),
            "cluster_size": int(self.cluster_size),
            "is_outlier": bool(self.is_outlier),
            "cohesion_mean": self.cohesion_mean,
            "cohesion_median": self.cohesion_median,
            "cohesion_min": self.cohesion_min,
            "cohesion_max": self.cohesion_max,
            "nearest_cluster_id": self.nearest_cluster_id,
            "nearest_cluster_similarity": self.nearest_cluster_similarity,
            "separation_margin": self.separation_margin,
            "representative_images_note": str(self.representative_images_note),
            "cluster_quality_score": self.cluster_quality_score,
            "similarity_mode": str(self.similarity_mode),
            "similarity_space": str(self.similarity_space),
            "input_dimension": self.input_dimension,
            "prepared_dimension": self.prepared_dimension,
            "pca_components": self.pca_components,
            "member_cohesion_scores": [float(score) for score in self.member_cohesion_scores],
        }

    @classmethod
    def from_context(cls, context: dict[str, object]) -> "ClusterExplanation":
        return cls(
            cluster_id=int(context.get("cluster_id", -1) or -1),
            cluster_size=int(context.get("cluster_size", 0) or 0),
            is_outlier=bool(context.get("is_outlier", False)),
            cohesion_mean=_maybe_float(context.get("cohesion_mean")),
            cohesion_median=_maybe_float(context.get("cohesion_median")),
            cohesion_min=_maybe_float(context.get("cohesion_min")),
            cohesion_max=_maybe_float(context.get("cohesion_max")),
            nearest_cluster_id=_maybe_int(context.get("nearest_cluster_id")),
            nearest_cluster_similarity=_maybe_float(context.get("nearest_cluster_similarity")),
            separation_margin=_maybe_float(context.get("separation_margin")),
            representative_images_note=str(context.get("representative_images_note", "") or ""),
            cluster_quality_score=_maybe_float(context.get("cluster_quality_score")),
            similarity_mode=str(context.get("similarity_mode", "") or ""),
            similarity_space=str(context.get("similarity_space", "") or ""),
            input_dimension=_maybe_int(context.get("input_dimension")),
            prepared_dimension=_maybe_int(context.get("prepared_dimension")),
            pca_components=_maybe_int(context.get("pca_components")),
            member_cohesion_scores=tuple(
                score
                for score in (_maybe_float(value) for value in context.get("member_cohesion_scores", []) or [])
                if score is not None
            ),
        )


def _maybe_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _maybe_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None
