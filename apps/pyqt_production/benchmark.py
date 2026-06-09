from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from apps.pyqt_production.bootstrap import bootstrap_runtime
from apps.shared.benchmark_schema import BenchmarkStage, build_report, write_report
from apps.shared.profile_support import current_memory_rss_mb, profile_call
from apps.shared.runtime_support import configure_rotating_logging, install_crash_handlers


LOGGER = logging.getLogger(__name__)


@dataclass
class BenchmarkStack:
    cache_maintenance: object
    pipeline: object


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark the production PyQt clustering stack.")
    parser.add_argument("--bench", choices=["cluster"], default="cluster")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--models", default="dino")
    parser.add_argument("--backends", default="cosine-kmeans")
    parser.add_argument("--passes", default="cold,warm,cache-hit")
    parser.add_argument("--report-dir", default="")
    parser.add_argument("--performance-profile", default="balanced")
    parser.add_argument("--preferred-execution-mode", default="auto")
    parser.add_argument("--num-clusters", type=int, default=12)
    parser.add_argument("--use-onnx", action="store_true")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Permit missing model files to be downloaded during this non-interactive benchmark run.",
    )
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir).resolve() if args.report_dir else (Path.cwd() / "benchmarks" / "pyqt_production")
    runtime_root = report_dir / "runtime" / "pyqt_production"
    os.environ["IMAGE_CLUSTERING_APP_DIR"] = str(runtime_root)

    runtime_layout, _repo_root = bootstrap_runtime()
    configure_rotating_logging(runtime_layout)
    install_crash_handlers(runtime_layout)

    stages: list[BenchmarkStage] = []
    summary: dict[str, object] = {}
    stack: BenchmarkStack | None = None
    state: dict[str, bool] = {"warm_ready": False, "cache_hit_ready": False}

    stage_names = [item.strip() for item in str(args.passes).split(",") if item.strip()]
    for stage_name in stage_names:
        stack, stage = _run_stage(
            stage_name=stage_name,
            stack=stack,
            state=state,
            folder=args.folder,
            models=[item.strip() for item in str(args.models).split(",") if item.strip()],
            backends=[item.strip() for item in str(args.backends).split(",") if item.strip()],
            performance_profile=args.performance_profile,
            preferred_execution_mode=args.preferred_execution_mode,
            num_clusters=int(args.num_clusters),
            use_onnx=bool(args.use_onnx),
            allow_model_downloads=bool(args.allow_downloads),
        )
        stages.append(stage)
        summary[f"{stage_name}_pipeline_wall_time_s"] = float(stage.details.get("pipeline_wall_time_s", 0.0) or 0.0)
        summary[f"{stage_name}_benchmark_wall_time_s"] = float(stage.details.get("benchmark_wall_time_s", 0.0) or 0.0)
        if "memory_rss_mb_after" in stage.details:
            summary[f"{stage_name}_memory_rss_mb_after"] = stage.details["memory_rss_mb_after"]

    cold_time = float(summary.get("cold_pipeline_wall_time_s", 0.0) or 0.0)
    for stage_name in ("warm", "cache-hit"):
        stage_time = float(summary.get(f"{stage_name}_pipeline_wall_time_s", 0.0) or 0.0)
        if cold_time > 0.0 and stage_time > 0.0:
            summary[f"{stage_name}_speedup_vs_cold"] = round(cold_time / stage_time, 3)

    report = build_report(
        app_id="pyqt_production",
        variant="production",
        scenario="cluster",
        runtime_root=str(runtime_layout.root),
        stages=stages,
        summary=summary,
        metadata={
            "folder": args.folder,
            "models": args.models,
            "backends": args.backends,
            "performance_profile": args.performance_profile,
            "preferred_execution_mode": args.preferred_execution_mode,
            "num_clusters": int(args.num_clusters),
            "use_onnx": bool(args.use_onnx),
            "passes": stage_names,
        },
    )
    json_path, markdown_path = write_report(report, report_dir, "pyqt_production_benchmark")
    LOGGER.info("Benchmark report written to %s and %s", json_path, markdown_path)
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(markdown_path)}, indent=2))
    return 0


def _run_stage(
    *,
    stage_name: str,
    stack: BenchmarkStack | None,
    state: dict[str, bool],
    folder: str,
    models: list[str],
    backends: list[str],
    performance_profile: str,
    preferred_execution_mode: str,
    num_clusters: int,
    use_onnx: bool,
    allow_model_downloads: bool,
) -> tuple[BenchmarkStack, BenchmarkStage]:
    normalized = str(stage_name or "").strip().lower()
    if normalized not in {"cold", "warm", "cache-hit"}:
        raise ValueError(f"Unsupported benchmark pass: {stage_name}")

    if normalized == "cold":
        stack = _build_stack(
            performance_profile=performance_profile,
            preferred_execution_mode=preferred_execution_mode,
            use_onnx=use_onnx,
            allow_model_downloads=allow_model_downloads,
        )
        _clear_rebuildable_cache(stack)
        state["warm_ready"] = False
        state["cache_hit_ready"] = False
    elif stack is None:
        stack = _build_stack(
            performance_profile=performance_profile,
            preferred_execution_mode=preferred_execution_mode,
            use_onnx=use_onnx,
            allow_model_downloads=allow_model_downloads,
        )
        _clear_rebuildable_cache(stack)

    if normalized == "warm" and not state["warm_ready"]:
        _run_cluster_pass(
            stack=stack,
            folder=folder,
            models=models,
            backends=backends,
            performance_profile=performance_profile,
            num_clusters=num_clusters,
            use_onnx=use_onnx,
            reuse_result_cache=False,
        )
        state["warm_ready"] = True

    if normalized == "cache-hit" and not state["cache_hit_ready"]:
        _run_cluster_pass(
            stack=stack,
            folder=folder,
            models=models,
            backends=backends,
            performance_profile=performance_profile,
            num_clusters=num_clusters,
            use_onnx=use_onnx,
            reuse_result_cache=True,
        )
        state["warm_ready"] = True
        state["cache_hit_ready"] = True

    memory_before = current_memory_rss_mb()
    result_metrics, profile_result = profile_call(
        lambda: _run_cluster_pass(
            stack=stack,
            folder=folder,
            models=models,
            backends=backends,
            performance_profile=performance_profile,
            num_clusters=num_clusters,
            use_onnx=use_onnx,
            reuse_result_cache=(normalized == "cache-hit"),
        )
    )
    metrics = dict(result_metrics)
    metrics["benchmark_stage"] = normalized
    metrics["benchmark_wall_time_s"] = profile_result.elapsed_s
    metrics["hotspots"] = profile_result.hotspots
    metrics["profile_stats_text"] = profile_result.stats_text
    metrics["result_cache_enabled"] = normalized == "cache-hit"
    metrics["memory_rss_mb_before"] = memory_before
    metrics["memory_rss_mb_after"] = current_memory_rss_mb()
    if normalized == "cold":
        state["warm_ready"] = True
        state["cache_hit_ready"] = False
    elif normalized == "warm":
        state["warm_ready"] = True
        state["cache_hit_ready"] = False
    else:
        state["warm_ready"] = True
        state["cache_hit_ready"] = True
    return stack, BenchmarkStage(
        name=normalized,
        elapsed_s=float(metrics.get("pipeline_wall_time_s", 0.0) or profile_result.elapsed_s),
        details=metrics,
    )


def _build_stack(
    *,
    performance_profile: str,
    preferred_execution_mode: str,
    use_onnx: bool,
    allow_model_downloads: bool,
) -> BenchmarkStack:
    from app.services.cache_maintenance import CacheMaintenanceService
    from app.services.clustering_pipeline import ClusteringPipelineService
    from infra.performance import select_performance_profile
    from infra.runtime import RuntimeCapabilityService
    from ml.embeddings import EmbeddingService, ModelManager

    runtime_service = RuntimeCapabilityService()
    execution_policy = runtime_service.select_policy(preferred_execution_mode)
    profile = select_performance_profile(performance_profile)
    model_manager = ModelManager(
        use_onnx=use_onnx,
        execution_policy=execution_policy,
        runtime_service=runtime_service,
        performance_profile=profile,
        allow_model_downloads=allow_model_downloads,
    )
    embedding_service = EmbeddingService(model_manager=model_manager, performance_profile=profile)
    pipeline = ClusteringPipelineService(embedding_service=embedding_service)
    cache_maintenance = CacheMaintenanceService()
    return BenchmarkStack(
        cache_maintenance=cache_maintenance,
        pipeline=pipeline,
    )


def _clear_rebuildable_cache(stack: BenchmarkStack) -> None:
    embedding_service = getattr(stack.pipeline, "embedding_service", None)
    cache_service = getattr(embedding_service, "cache_service", None)
    cleared_targets: list[str] = []
    failures: list[str] = []
    if cache_service is not None:
        try:
            cache_service.clear_disk_cache()
            cleared_targets.append("embeddings.sqlite3")
        except Exception as exc:
            failures.append(f"embeddings.sqlite3: {exc}")
        try:
            cache_service.clear_memory_cache()
        except Exception:
            LOGGER.exception("Failed to clear embedding memory cache")
    disk_cleared, disk_failures = stack.cache_maintenance.clear_rebuildable_disk_targets(exclude={"embeddings.sqlite3"})
    cleared_targets.extend(disk_cleared)
    failures.extend(disk_failures)
    if failures:
        LOGGER.warning("Cache clear completed with failures: %s", tuple(failures))
    LOGGER.info("Cleared benchmark rebuildable caches: %s", tuple(cleared_targets))


def _run_cluster_pass(
    *,
    stack: BenchmarkStack,
    folder: str,
    models: list[str],
    backends: list[str],
    performance_profile: str,
    num_clusters: int,
    use_onnx: bool,
    reuse_result_cache: bool,
) -> dict[str, object]:
    from app.services.clustering_pipeline import ClusteringRequest

    _result, metrics = stack.pipeline.run(
        ClusteringRequest(
            directory=folder,
            embedding_models=models,
            num_clusters=num_clusters,
            clustering_backends=backends,
            recursive=True,
            similarity_mode="semantic",
            outlier_policy="assign",
            use_onnx=use_onnx,
            reuse_result_cache=reuse_result_cache,
            use_embedding_cache_lookup=True,
            performance_profile=performance_profile,
            generate_cluster_explanations=False,
        )
    )
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
