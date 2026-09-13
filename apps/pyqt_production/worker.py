from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

from apps.pyqt_production.bootstrap import bootstrap_runtime
from apps.pyqt_production.worker_protocol import ProductionClusterRequest, json_line
from apps.shared.runtime_support import configure_rotating_logging, install_crash_handlers


RUNTIME_LAYOUT, _REPO_ROOT = bootstrap_runtime()
os.environ["IMAGE_CLUSTERING_MODEL_ASSETS_DIR"] = str(RUNTIME_LAYOUT.model_assets_dir)
configure_rotating_logging(RUNTIME_LAYOUT)
install_crash_handlers(RUNTIME_LAYOUT)

from app.services.clustering_pipeline import ClusteringPipelineService, ClusteringRequest  # noqa: E402
from app.services.image_tags import ClusterTagSummary, ImageTagService  # noqa: E402
from infra.performance import apply_performance_overrides, select_performance_profile  # noqa: E402
from infra.runtime import RuntimeCapabilityService  # noqa: E402
from ml.embeddings import EmbeddingService, ModelManager  # noqa: E402


configure_rotating_logging(RUNTIME_LAYOUT, force=True)
LOGGER = logging.getLogger(__name__)


def _emit_event(event_type: str, payload: dict[str, object]) -> None:
    data = (json.dumps(json_line(event_type, payload)) + "\n").encode("utf-8", errors="replace")
    stream = getattr(sys, "stdout", None)
    if stream is not None:
        try:
            stream.write(data.decode("utf-8"))
            stream.flush()
            return
        except Exception:
            LOGGER.exception("Worker stdout write failed; falling back to fd 1")
    try:
        os.write(1, data)
    except OSError:
        LOGGER.exception("Worker stdout fd write failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Production PyQt clustering worker")
    parser.add_argument("--request-json", help="Path to a JSON request file.")
    parser.add_argument("--model-request-json", help="Path to a model download request file.")
    parser.add_argument("--daemon", action="store_true", help="Keep the worker process alive and accept request JSON lines on stdin.")
    args = parser.parse_args(argv)

    if args.daemon:
        LOGGER.info("persistent worker startup begin | app_log=%s", RUNTIME_LAYOUT.app_log)
        return _run_daemon()
    if args.model_request_json:
        LOGGER.info("model worker startup begin | request_json=%s app_log=%s", args.model_request_json, RUNTIME_LAYOUT.app_log)
        return _run_model_download_request(Path(args.model_request_json))
    if not args.request_json:
        parser.error("--request-json or --model-request-json is required unless --daemon is used")

    LOGGER.info("worker startup begin | request_json=%s app_log=%s", args.request_json, RUNTIME_LAYOUT.app_log)
    request_payload = json.loads(Path(args.request_json).read_text(encoding="utf-8"))
    request = ProductionClusterRequest(**request_payload)
    return _run_cluster_request(request)


def _run_model_download_request(request_path: Path) -> int:
    from app.services.model_downloads import ModelDownloadItem, ModelDownloadService

    try:
        payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
        raw_items = list(payload.get("items") or []) if isinstance(payload, dict) else []
        items = [
            ModelDownloadItem(
                model_name=str(item.get("model_name") or ""),
                require_text=bool(item.get("require_text")),
            )
            for item in raw_items
            if isinstance(item, dict)
        ]
        if not items:
            raise ValueError("The model download request is empty.")

        result = ModelDownloadService().acquire(
            items,
            progress_callback=lambda value, status: _emit_event(
                "progress",
                {"value": int(value), "status": str(status)},
            ),
        )
        _emit_event("result", result.as_dict())
        return 0
    except Exception as exc:
        LOGGER.exception("Model download worker failed")
        _emit_event("error", {"message": str(exc), "traceback": traceback.format_exc()})
        return 1


@dataclass
class WorkerStack:
    key: tuple[object, ...]
    pipeline: ClusteringPipelineService
    tag_service: ImageTagService


class PersistentWorkerRuntime:
    def __init__(self) -> None:
        self._runtime_service = RuntimeCapabilityService()
        self._stack: WorkerStack | None = None

    def stack_for(self, request: ProductionClusterRequest) -> WorkerStack:
        key = (
            str(request.preferred_execution_mode or "auto"),
            str(request.performance_profile or "balanced"),
            bool(request.use_onnx),
            bool(request.allow_model_downloads),
            int(request.batch_size_cpu),
            int(request.batch_size_gpu),
            int(request.preprocess_workers),
            int(request.vram_headroom_mb),
        )
        if self._stack is not None and self._stack.key == key:
            return self._stack
        execution_policy = self._runtime_service.select_policy(request.preferred_execution_mode)
        if execution_policy.cuda_required_unavailable:
            raise RuntimeError(execution_policy.error or execution_policy.reason)
        performance_profile = _performance_profile_for_request(request)
        model_manager = ModelManager(
            use_onnx=request.use_onnx,
            execution_policy=execution_policy,
            runtime_service=self._runtime_service,
            performance_profile=performance_profile,
            allow_model_downloads=bool(request.allow_model_downloads),
        )
        embedding_service = EmbeddingService(model_manager=model_manager, performance_profile=performance_profile)
        self._stack = WorkerStack(
            key=key,
            pipeline=ClusteringPipelineService(embedding_service=embedding_service),
            tag_service=ImageTagService(),
        )
        LOGGER.info(
            "Prepared persistent worker stack | mode=%s profile=%s use_onnx=%s allow_downloads=%s",
            key[0],
            key[1],
            key[2],
            key[3],
        )
        return self._stack


def _run_daemon() -> int:
    runtime = PersistentWorkerRuntime()
    _emit_event("ready", {"status": "persistent worker ready"})
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            envelope = json.loads(line)
            event_type = str(envelope.get("type", "request"))
            if event_type == "shutdown":
                LOGGER.info("persistent worker shutdown requested")
                return 0
            payload = dict(envelope.get("payload") or envelope)
            request = ProductionClusterRequest(**payload)
            _run_cluster_request(request, runtime=runtime)
        except Exception as exc:
            LOGGER.exception("Persistent worker request dispatch failed")
            _emit_event(
                "error",
                {
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                    "preferred_execution_mode": "",
                },
            )
    return 0


def _build_one_shot_stack(request: ProductionClusterRequest) -> WorkerStack:
    runtime_service = RuntimeCapabilityService()
    execution_policy = runtime_service.select_policy(request.preferred_execution_mode)
    if execution_policy.cuda_required_unavailable:
        raise RuntimeError(execution_policy.error or execution_policy.reason)
    performance_profile = _performance_profile_for_request(request)
    model_manager = ModelManager(
        use_onnx=request.use_onnx,
        execution_policy=execution_policy,
        runtime_service=runtime_service,
        performance_profile=performance_profile,
        allow_model_downloads=bool(request.allow_model_downloads),
    )
    embedding_service = EmbeddingService(model_manager=model_manager, performance_profile=performance_profile)
    return WorkerStack(
        key=(
            request.preferred_execution_mode,
            request.performance_profile,
            bool(request.use_onnx),
            bool(request.allow_model_downloads),
            int(request.batch_size_cpu),
            int(request.batch_size_gpu),
            int(request.preprocess_workers),
            int(request.vram_headroom_mb),
        ),
        pipeline=ClusteringPipelineService(embedding_service=embedding_service),
        tag_service=ImageTagService(),
    )


def _performance_profile_for_request(request: ProductionClusterRequest):
    return apply_performance_overrides(
        select_performance_profile(request.performance_profile),
        cpu_batch_size=request.batch_size_cpu or None,
        gpu_batch_size=request.batch_size_gpu or None,
        embedding_preprocess_workers=request.preprocess_workers or None,
        vram_headroom_mb=request.vram_headroom_mb or None,
    )


def _run_cluster_request(request: ProductionClusterRequest, *, runtime: PersistentWorkerRuntime | None = None) -> int:
    stack = runtime.stack_for(request) if runtime is not None else _build_one_shot_stack(request)

    def _progress(value: int, status: str) -> None:
        _emit_event("progress", {"value": int(value), "status": str(status)})

    try:
        _progress(-1, "Worker started. Preparing clustering pipeline...")
        result, metrics = stack.pipeline.run(
            ClusteringRequest(
                directory=request.directory,
                embedding_models=list(request.embedding_models),
                num_clusters=int(request.num_clusters),
                clustering_backends=list(request.clustering_backends),
                recursive=bool(request.recursive),
                similarity_mode=request.similarity_mode,
                similarity_modes=list(request.similarity_modes),
                outlier_policy=request.outlier_policy,
                use_onnx=bool(request.use_onnx),
                reuse_result_cache=bool(request.reuse_result_cache),
                use_embedding_cache_lookup=bool(request.use_embedding_cache_lookup),
                source_paths=list(request.source_paths) if request.source_paths else None,
                source_fingerprints=list(request.source_fingerprints) if request.source_fingerprints else None,
                source_snapshot_key=str(request.source_snapshot_key or ""),
                performance_profile=request.performance_profile,
                generate_cluster_meanings=bool(request.generate_cluster_meanings),
                generate_cluster_explanations=bool(request.generate_cluster_explanations),
                cluster_meaning_model=request.cluster_meaning_model,
                backend_options_by_backend={
                    str(backend): dict(options or {})
                    for backend, options in request.backend_options_by_backend.items()
                },
            ),
            progress_callback=_progress,
        )
        cluster_summaries = _summaries_for_result(stack.tag_service, result.clusters_by_key)
        payload = {
            "clusters_by_key": _stringify_int_keys(result.clusters_by_key),
            "membership_by_image": _stringify_int_keys(result.membership_by_image),
            "metrics_by_key": result.metrics_by_key,
            "cluster_explanations_by_key": {
                comparison_key: {
                    str(cluster_id): explanation.as_context()
                    for cluster_id, explanation in explanations.items()
                }
                for comparison_key, explanations in result.cluster_explanations_by_key.items()
            },
            "cluster_meanings_by_key": {
                comparison_key: {
                    str(cluster_id): meaning.as_context()
                    for cluster_id, meaning in meanings.items()
                }
                for comparison_key, meanings in result.cluster_meanings_by_key.items()
            },
            "metrics": metrics,
            "cluster_summaries": {
                comparison_key: {
                    str(cluster_id): summary.as_context()
                    for cluster_id, summary in summaries.items()
                }
                for comparison_key, summaries in cluster_summaries.items()
            },
        }
        _emit_event("result", payload)
        return 0
    except Exception as exc:
        LOGGER.exception("Worker clustering failed")
        payload = {
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "preferred_execution_mode": request.preferred_execution_mode,
        }
        _emit_event("error", payload)
        return 1


def _summaries_for_result(
    tag_service: ImageTagService,
    clusters_by_key: dict[str, dict[int, list[str]]],
) -> dict[str, dict[int, ClusterTagSummary]]:
    summaries: dict[str, dict[int, ClusterTagSummary]] = {}
    all_paths = sorted({path for clusters in clusters_by_key.values() for paths in clusters.values() for path in paths})
    tags_by_path = tag_service.load_tags_for_paths(all_paths)
    for comparison_key, clusters in clusters_by_key.items():
        summaries[comparison_key] = {
            int(cluster_id): tag_service.summarize_paths(image_paths, tags_by_path=tags_by_path)
            for cluster_id, image_paths in clusters.items()
        }
    return summaries


def _stringify_int_keys(value):
    if isinstance(value, dict):
        return {str(key): _stringify_int_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_stringify_int_keys(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
