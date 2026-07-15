from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


def _job_value(job, key: str, default=None):
    if isinstance(job, dict):
        return job.get(key, default)
    return getattr(job, key, default)


def _first(metrics: dict[str, object], *keys: str, default=None):
    for key in keys:
        value = metrics.get(key)
        if value is not None and value != "":
            return value
    return default


def _as_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PerformanceDashboardSnapshot:
    model_load_state: str = "unknown"
    cache_hits: int = 0
    cache_misses: int = 0
    skipped_unchanged: int = 0
    detector_calls: int = 0
    embedder_calls: int = 0
    ann_state: str = "unknown"
    cancellation_status: str = "idle"
    active_job_labels: tuple[str, ...] = field(default_factory=tuple)

    def to_text(self) -> str:
        jobs = ", ".join(self.active_job_labels) if self.active_job_labels else "none"
        return (
            f"Jobs: {jobs} | Cancel: {self.cancellation_status} | "
            f"Model: {self.model_load_state} | "
            f"Cache: {self.cache_hits} hit/{self.cache_misses} miss | "
            f"Skipped: {self.skipped_unchanged} unchanged | "
            f"Face calls: detector={self.detector_calls}, embedder={self.embedder_calls} | "
            f"ANN: {self.ann_state}"
        )


class PerformanceDashboardService:
    def snapshot(
        self,
        metrics: dict[str, object] | None = None,
        *,
        jobs: Iterable[object] = (),
    ) -> PerformanceDashboardSnapshot:
        metrics = dict(metrics or {})
        job_list = list(jobs or ())
        active_jobs = [
            job for job in job_list
            if str(_job_value(job, "status", "")).casefold() in {"running", "cancelling"}
        ]
        active_labels = tuple(
            str(_job_value(job, "label", "")).strip()
            for job in active_jobs
            if str(_job_value(job, "label", "")).strip()
        )
        statuses = {str(_job_value(job, "status", "")).casefold() for job in job_list}
        metric_status = str(metrics.get("status") or "").casefold()
        if "cancelling" in statuses:
            cancellation_status = "cancelling"
        elif "cancelled" in statuses or metric_status == "cancelled":
            cancellation_status = "cancelled"
        elif active_jobs:
            cancellation_status = "running"
        else:
            cancellation_status = "idle"

        model_load_state = str(_first(metrics, "model_load_state", default="")).strip()
        if not model_load_state:
            if metrics.get("embedding_stage_skipped"):
                model_load_state = "skipped"
            elif "model_load_time_s" in metrics:
                model_load_state = "loaded" if float(metrics.get("model_load_time_s") or 0.0) > 0 else "reused"
            else:
                model_load_state = "unknown"

        ann_state = str(_first(metrics, "ann_state", "ann_index_state", default="")).strip()
        if not ann_state:
            if bool(metrics.get("ann_build_required")):
                ann_state = "build required"
            elif metrics.get("ann_files"):
                ann_state = "ready"
            else:
                ann_state = "unknown"

        return PerformanceDashboardSnapshot(
            model_load_state=model_load_state,
            cache_hits=_as_int(_first(metrics, "cache_hits", default=0)),
            cache_misses=_as_int(_first(metrics, "cache_misses", default=0)),
            skipped_unchanged=_as_int(
                _first(
                    metrics,
                    "skipped_unchanged",
                    "unchanged_skipped",
                    "unchanged_skipped_count",
                    "skipped_unchanged_files",
                    "cached_image_count",
                    default=0,
                )
            ),
            detector_calls=_as_int(
                _first(metrics, "detector_calls", "face_detector_calls", "detector_call_count", default=0)
            ),
            embedder_calls=_as_int(
                _first(metrics, "embedder_calls", "face_embedder_calls", "embedder_call_count", default=0)
            ),
            ann_state=ann_state,
            cancellation_status=cancellation_status,
            active_job_labels=active_labels,
        )
