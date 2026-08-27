from __future__ import annotations

import inspect
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from infra.logging_config import get_logger
from infra.performance import select_performance_profile
from app.services.cluster_explanations import ClusterExplanation
from app.services.cluster_meanings import ClusterMeaning, ClusterMeaningCancelled, ClusterMeaningService
from ml.clustering import ClusteringService, PreparedMatrixInfo
from ml.embeddings import EmbeddingService

from .discovery import DiscoveryResult, ImageDiscoveryService
from .embedding_index import EmbeddingIndexService
from .result_cache import ResultCacheService
from .similarity_modes import SIMILARITY_SPACE_VERSION, normalize_similarity_modes

LOGGER = get_logger(__name__)


class RunCancelled(RuntimeError):
    pass


def _prepare_matrix_with_info(
    clustering_service: ClusteringService,
    embeddings,
    *,
    pca_dim: int,
    similarity_mode: str,
) -> tuple[object, PreparedMatrixInfo]:
    prepare_with_info = getattr(clustering_service, "prepare_matrix_with_info", None)
    if callable(prepare_with_info):
        return prepare_with_info(embeddings, pca_dim=pca_dim, similarity_mode=similarity_mode)

    prepared_matrix = clustering_service.prepare_matrix(
        embeddings,
        pca_dim=pca_dim,
        similarity_mode=similarity_mode,
    )
    return prepared_matrix, PreparedMatrixInfo.from_matrix(prepared_matrix, similarity_mode)


def _prepared_info_metrics(prepared_info: PreparedMatrixInfo) -> dict[str, object]:
    return {
        "similarity_space": prepared_info.space_label,
        "input_dimension": prepared_info.input_dimension,
        "prepared_dimension": prepared_info.prepared_dimension,
        "pca_components": prepared_info.pca_components or 0,
    }


@dataclass
class ClusteringRequest:
    directory: str
    embedding_models: list[str]
    num_clusters: int
    clustering_backends: list[str]
    recursive: bool
    similarity_mode: str
    outlier_policy: str
    use_onnx: bool
    reuse_result_cache: bool
    use_embedding_cache_lookup: bool = True
    source_paths: list[str] | None = None
    source_fingerprints: list[tuple[str, int, int]] | None = None
    source_snapshot_key: str = ""
    performance_profile: str = "balanced"
    similarity_modes: list[str] | None = None
    generate_cluster_meanings: bool = False
    generate_cluster_explanations: bool = True
    cluster_meaning_model: str = "auto"

    def normalized_similarity_modes(self) -> list[str]:
        return normalize_similarity_modes(self.similarity_modes, self.similarity_mode)


@dataclass
class MultiBackendClusteringResult:
    # Key format for new runs: "{embedding_model}::{similarity_mode}::{backend}".
    clusters_by_key: dict[str, dict[int, list[str]]] = field(default_factory=dict)
    membership_by_image: dict[str, dict[str, dict[str, object]]] = field(default_factory=dict)
    metrics_by_key: dict[str, dict[str, object]] = field(default_factory=dict)
    cluster_explanations_by_key: dict[str, dict[int, ClusterExplanation]] = field(default_factory=dict)
    cluster_meanings_by_key: dict[str, dict[int, ClusterMeaning]] = field(default_factory=dict)


class ClusteringPipelineService:
    def __init__(
        self,
        discovery_service: ImageDiscoveryService | None = None,
        embedding_service: EmbeddingService | None = None,
        clustering_service: ClusteringService | None = None,
        embedding_index_service: EmbeddingIndexService | None = None,
        result_cache_service: ResultCacheService | None = None,
        cluster_meaning_service: ClusterMeaningService | None = None,
    ) -> None:
        self.discovery_service = discovery_service or ImageDiscoveryService()
        self.embedding_service = embedding_service or EmbeddingService()
        self.clustering_service = clustering_service or ClusteringService()
        self.embedding_index_service = embedding_index_service or EmbeddingIndexService()
        self.result_cache_service = result_cache_service or ResultCacheService()
        self.cluster_meaning_service = cluster_meaning_service or ClusterMeaningService()

    def run(
        self,
        request: ClusteringRequest,
        progress_callback=None,
        cancel_check=None,
    ) -> tuple[MultiBackendClusteringResult, dict]:
        pipeline_start = time.perf_counter()
        backends = [backend for backend in request.clustering_backends if backend]
        if not backends:
            raise ValueError("Select at least one clustering backend.")

        models = [model for model in request.embedding_models if model]
        if not models:
            raise ValueError("Select at least one embedding model.")
        similarity_modes = request.normalized_similarity_modes()

        metrics: dict[str, object] = {
            "embedding_models": ", ".join(models),
            "similarity_mode": similarity_modes[0],
            "similarity_modes": ", ".join(similarity_modes),
            "clustering_backends": ", ".join(backends),
            "outlier_policy": request.outlier_policy,
            "use_onnx": request.use_onnx,
            "embedding_cache_lookup": "enabled" if request.use_embedding_cache_lookup else "disabled",
            "performance_profile": request.performance_profile,
            "similarity_space_version": SIMILARITY_SPACE_VERSION,
            "cluster_meanings": "enabled" if request.generate_cluster_meanings else "disabled",
            "cluster_meaning_model": request.cluster_meaning_model,
        }
        performance_profile = select_performance_profile(request.performance_profile)

        scan_start = time.perf_counter()
        if request.source_paths:
            if progress_callback:
                progress_callback(-1, f"Preparing {len(request.source_paths)} selected image(s)...")
            image_paths = sorted({str(path) for path in request.source_paths if path})
            image_path_set = set(image_paths)
            provided_fingerprints = {
                str(path): (str(path), int(mtime_ns), int(size))
                for path, mtime_ns, size in (request.source_fingerprints or [])
                if str(path) in image_path_set
            }
            fingerprints = tuple(
                provided_fingerprints[path]
                for path in image_paths
                if path in provided_fingerprints
            )
            if len(fingerprints) != len(image_paths):
                fingerprints = self.discovery_service.collect_fingerprints(image_paths)
            discovery = DiscoveryResult(
                paths=tuple(path for path, _mtime_ns, _size in fingerprints),
                snapshot_key=(
                    str(request.source_snapshot_key)
                    if request.source_snapshot_key and len(fingerprints) == len(image_paths)
                    else self.embedding_index_service.build_snapshot_key_from_fingerprints(fingerprints)
                ),
                image_count=len(fingerprints),
                fingerprints=fingerprints,
            )
            image_paths = list(discovery.paths)
        else:
            if progress_callback:
                progress_callback(-1, f"Scanning folder: {request.directory}")
            discovery = self._discover_result(
                request.directory,
                recursive=request.recursive,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
            image_paths = list(discovery.paths)
        discovery_time_s = round(time.perf_counter() - scan_start, 3)
        metrics["scan_time_s"] = discovery_time_s
        metrics["discovery_time_s"] = discovery_time_s
        metrics["image_count"] = len(image_paths)
        metrics["source_scope"] = "working_set" if request.source_paths else "folder"
        if len(image_paths) < 2:
            raise ValueError("Need at least two images to cluster.")
        if request.num_clusters > len(image_paths):
            raise ValueError("Cluster count cannot exceed image count.")
        if cancel_check and cancel_check():
            raise RunCancelled()

        snapshot_key = discovery.snapshot_key
        metrics["snapshot_key"] = snapshot_key[:16]
        path_fingerprint_map = {
            str(path): (int(mtime_ns), int(size))
            for path, mtime_ns, size in getattr(discovery, "fingerprints", ())
        }
        cached_only = self._try_result_cache_only(
            request,
            models=models,
            similarity_modes=similarity_modes,
            backends=backends,
            snapshot_key=snapshot_key,
            metrics=metrics,
            image_paths=image_paths,
            pipeline_start=pipeline_start,
        )
        if cached_only is not None:
            return cached_only

        def embedding_progress(done: int, total: int, stage: str) -> None:
            if progress_callback:
                progress = int((done / max(1, total)) * 65)
                progress_callback(progress, stage)
            if cancel_check and cancel_check():
                raise RunCancelled()

        result = MultiBackendClusteringResult()

        embed_total_start = time.perf_counter()
        cache_hits = 0
        cache_misses = 0
        model_load_time_s = 0.0
        model_device = ""
        embedding_backend = ""
        amp_enabled = False
        effective_mode = ""
        onnx_provider = ""
        runtime_reason = ""
        last_batch_size = None
        backend_hit_flags: dict[str, bool] = {}
        total_prepared_matrix_s = 0.0
        total_cluster_s = 0.0
        total_index_persist_s = 0.0
        index_reuse_flags: dict[str, bool] = {}
        total_cache_lookup_s = 0.0
        total_preprocess_s = 0.0
        total_inference_s = 0.0
        oom_backoff_used = False
        total_embedding_stage_s = 0.0
        total_backend_wall_s = 0.0

        for model_index, model_name in enumerate(models, start=1):
            if progress_callback:
                progress_callback(0, f"Embedding {model_name} ({model_index}/{len(models)})")

            embed_start = time.perf_counter()
            embed_kwargs = {
                "use_onnx": request.use_onnx,
                "progress_callback": embedding_progress,
                "use_cache_lookup": request.use_embedding_cache_lookup,
                "path_fingerprints": path_fingerprint_map,
            }
            try:
                embed_parameters = inspect.signature(self.embedding_service.embed_paths).parameters
            except (TypeError, ValueError):
                embed_parameters = {}
            if "cancel_check" in embed_parameters:
                embed_kwargs["cancel_check"] = cancel_check
            ordered_embeddings, embed_metrics = self.embedding_service.embed_paths(
                image_paths,
                model_name,
                **embed_kwargs,
            )
            embed_elapsed_s = round(time.perf_counter() - embed_start, 3)
            total_embedding_stage_s += embed_elapsed_s
            metrics[f"embedding_time_s[{model_name}]"] = embed_elapsed_s
            usable_count = len(ordered_embeddings)
            if usable_count < 2:
                raise ValueError("Need at least two readable images to cluster. Some files may be corrupted/unreadable.")
            if request.num_clusters > usable_count:
                raise ValueError(f"Cluster count {request.num_clusters} exceeds readable image count {usable_count}.")

            cache_hits += int(embed_metrics.get("cache_hits", 0) or 0)
            cache_misses += int(embed_metrics.get("cache_misses", 0) or 0)
            model_load_time_s += float(embed_metrics.get("model_load_time_s", 0.0) or 0.0)
            if not model_device:
                model_device = str(embed_metrics.get("model_device", ""))
            if not embedding_backend:
                embedding_backend = str(embed_metrics.get("embedding_backend", ""))
            if "amp_enabled" in embed_metrics:
                amp_enabled = bool(embed_metrics.get("amp_enabled"))
            if not effective_mode:
                effective_mode = str(embed_metrics.get("effective_mode", ""))
            if not onnx_provider:
                onnx_provider = str(embed_metrics.get("onnx_provider", ""))
            if not runtime_reason:
                runtime_reason = str(embed_metrics.get("runtime_reason", ""))
            if "batch_size" in embed_metrics:
                last_batch_size = embed_metrics.get("batch_size")
            total_cache_lookup_s += float(embed_metrics.get("cache_lookup_time_s", 0.0) or 0.0)
            total_preprocess_s += float(embed_metrics.get("preprocess_time_s", 0.0) or 0.0)
            total_inference_s += float(embed_metrics.get("inference_time_s", 0.0) or 0.0)
            oom_backoff_used = oom_backoff_used or bool(embed_metrics.get("oom_backoff"))

            paths_by_index = [image_path for image_path, _ in ordered_embeddings]
            embedding_signature = self._embedding_signature(model_name, request.use_onnx)
            cache_plan: dict[str, dict[str, tuple[dict[int, list[str]], dict[str, object]]]] = {}
            any_backend_compute = False
            for similarity_mode in similarity_modes:
                cached_by_backend: dict[str, tuple[dict[int, list[str]], dict[str, object]]] = {}
                backends_to_compute: list[str] = []
                for backend in backends:
                    result_key = self.result_cache_service.build_result_key(
                        snapshot_key=snapshot_key,
                        embedding_model=model_name,
                        similarity_mode=similarity_mode,
                        clustering_backend=backend,
                        num_clusters=request.num_clusters,
                        outlier_policy=request.outlier_policy,
                        use_onnx=request.use_onnx,
                        embedding_signature=embedding_signature,
                        similarity_space_version=SIMILARITY_SPACE_VERSION,
                    )
                    if request.reuse_result_cache:
                        cached = self.result_cache_service.load(result_key)
                        if cached is not None:
                            cached_by_backend[backend] = (cached[0], dict(cached[1]))
                            continue
                    backends_to_compute.append(backend)
                cache_plan[similarity_mode] = cached_by_backend
                any_backend_compute = any_backend_compute or bool(backends_to_compute)

            if any_backend_compute:
                persist_start = time.perf_counter()
                ensure_index_kwargs: dict[str, object] = {}
                try:
                    ensure_index_parameters = inspect.signature(
                        self.embedding_index_service.ensure_index
                    ).parameters
                except (TypeError, ValueError):
                    ensure_index_parameters = {}
                if "embedding_signature" in ensure_index_parameters:
                    ensure_index_kwargs["embedding_signature"] = embedding_signature
                index_paths, index_reused = self.embedding_index_service.ensure_index(
                    snapshot_key,
                    model_name,
                    ordered_embeddings,
                    **ensure_index_kwargs,
                )
                index_persist_time_s = round(time.perf_counter() - persist_start, 3)
                total_index_persist_s += index_persist_time_s
                index_reuse_flags[model_name] = index_reused
                metrics[f"index_persist_time_s[{model_name}]"] = index_persist_time_s
                metrics[f"index_reused[{model_name}]"] = index_reused
                metrics.update({f"{key}[{model_name}]": value for key, value in index_paths.items()})
            else:
                metrics[f"index_persist_time_s[{model_name}]"] = 0.0
                metrics[f"index_reused[{model_name}]"] = True

            index_by_path = {str(path): index for index, path in enumerate(paths_by_index)}

            def run_backend(
                backend: str,
                similarity_mode: str,
                prepared_matrix,
                prepared_info: PreparedMatrixInfo,
            ):
                key = self._comparison_key(model_name, similarity_mode, backend)
                cached_by_backend = cache_plan[similarity_mode]
                if backend in cached_by_backend:
                    clustered_images, cached_metrics = cached_by_backend[backend]
                    backend_metrics = dict(cached_metrics)
                    backend_metrics["cached_clustering_time_s"] = float(backend_metrics.get("clustering_time_s", 0.0) or 0.0)
                    backend_metrics["clustering_time_s"] = 0.0
                    backend_metrics["embedding_model"] = model_name
                    backend_metrics["similarity_mode"] = similarity_mode
                    backend_metrics["backend"] = backend
                    backend_metrics.update(_prepared_info_metrics(prepared_info))
                    backend_metrics["result_cache_hit"] = True
                    backend_metrics["backend_wall_time_s"] = 0.0
                    backend_metrics["backend_compute_time_s"] = 0.0
                    clusters = self._clusters_to_indices(clustered_images, index_by_path)
                    return key, backend, clustered_images, clusters, backend_metrics, True, 0.0

                backend_start = time.perf_counter()
                cluster_start = time.perf_counter()
                clusters, cluster_metrics = self.clustering_service.cluster_prepared(
                    prepared_matrix,
                    request.num_clusters,
                    backend=backend,
                    outlier_policy=request.outlier_policy,
                    performance_profile=request.performance_profile,
                )
                backend_metrics = dict(cluster_metrics)
                backend_metrics["clustering_time_s"] = round(time.perf_counter() - cluster_start, 3)
                backend_metrics["embedding_model"] = model_name
                backend_metrics["similarity_mode"] = similarity_mode
                backend_metrics["backend"] = backend
                backend_metrics.update(_prepared_info_metrics(prepared_info))
                backend_metrics["result_cache_hit"] = False

                clustered_images = {
                    cluster_id: [paths_by_index[index] for index in indices]
                    for cluster_id, indices in clusters.items()
                }
                if request.reuse_result_cache:
                    result_key = self.result_cache_service.build_result_key(
                        snapshot_key=snapshot_key,
                        embedding_model=model_name,
                        similarity_mode=similarity_mode,
                        clustering_backend=backend,
                        num_clusters=request.num_clusters,
                        outlier_policy=request.outlier_policy,
                        use_onnx=request.use_onnx,
                        embedding_signature=embedding_signature,
                        similarity_space_version=SIMILARITY_SPACE_VERSION,
                    )
                    self.result_cache_service.save(result_key, clustered_images, backend_metrics)
                backend_wall_time_s = round(time.perf_counter() - backend_start, 3)
                backend_metrics["backend_wall_time_s"] = backend_wall_time_s
                backend_metrics["backend_compute_time_s"] = float(backend_metrics.get("clustering_time_s", 0.0) or 0.0)
                return key, backend, clustered_images, clusters, backend_metrics, False, backend_wall_time_s

            model_prepare_time_s = 0.0
            total_mode_backend_count = max(1, len(similarity_modes) * len(backends))
            completed_mode_backends = 0
            for similarity_mode in similarity_modes:
                prepare_start = time.perf_counter()
                prepared_matrix, prepared_info = _prepare_matrix_with_info(
                    self.clustering_service,
                    [embedding for _, embedding in ordered_embeddings],
                    pca_dim=50,
                    similarity_mode=similarity_mode,
                )
                prepared_matrix_time_s = round(time.perf_counter() - prepare_start, 3)
                model_prepare_time_s += prepared_matrix_time_s
                total_prepared_matrix_s += prepared_matrix_time_s
                metrics[f"prepared_matrix_time_s[{model_name}::{similarity_mode}]"] = prepared_matrix_time_s
                metrics[f"prepared_matrix_dim[{model_name}::{similarity_mode}]"] = prepared_info.prepared_dimension
                metrics[f"input_embedding_dim[{model_name}::{similarity_mode}]"] = prepared_info.input_dimension
                metrics[f"pca_components[{model_name}::{similarity_mode}]"] = prepared_info.pca_components or 0
                metrics[f"similarity_space[{model_name}::{similarity_mode}]"] = prepared_info.space_label

                if progress_callback:
                    progress_callback(70, f"Clustering {model_name} / {similarity_mode}")

                max_backend_workers = performance_profile.backend_workers_for(len(backends))
                with ThreadPoolExecutor(max_workers=max_backend_workers) as executor:
                    futures = [
                        executor.submit(run_backend, backend, similarity_mode, prepared_matrix, prepared_info)
                        for backend in backends
                    ]
                    for future in as_completed(futures):
                        if cancel_check and cancel_check():
                            raise RunCancelled()
                        key, backend, clustered_images, clusters_by_index, backend_metrics, cache_hit, backend_wall_time_s = future.result()
                        result.clusters_by_key[key] = clustered_images
                        result.metrics_by_key[key] = backend_metrics
                        if request.generate_cluster_explanations:
                            result.cluster_explanations_by_key[key] = self.clustering_service.build_cluster_explanations(
                                prepared_matrix,
                                clusters_by_index,
                                cluster_quality_score=backend_metrics.get("cluster_quality_score"),
                                prepared_info=prepared_info,
                            )
                        backend_hit_flags[key] = cache_hit
                        total_cluster_s += float(backend_metrics.get("backend_compute_time_s", 0.0) or 0.0)
                        total_backend_wall_s += float(backend_wall_time_s or 0.0)
                        metrics[f"backend_wall_time_s[{key}]"] = float(backend_metrics.get("backend_wall_time_s", 0.0) or 0.0)
                        metrics[f"backend_compute_time_s[{key}]"] = float(backend_metrics.get("backend_compute_time_s", 0.0) or 0.0)
                        metrics[f"backend_cache_hit[{key}]"] = cache_hit
                        self._merge_membership(
                            result.membership_by_image,
                            key,
                            clustered_images,
                            model_name=model_name,
                            similarity_mode=similarity_mode,
                            backend=backend,
                        )
                        completed_mode_backends += 1
                        if progress_callback:
                            progress = 70 + int((completed_mode_backends / total_mode_backend_count) * 30)
                            progress_callback(progress, f"Finished {model_name} / {similarity_mode} / {backend}")
            metrics[f"prepared_matrix_time_s[{model_name}]"] = round(model_prepare_time_s, 3)

        if request.generate_cluster_meanings:
            meaning_start = time.perf_counter()
            if progress_callback:
                progress_callback(90, "Generating cluster meanings")

            def meaning_progress(value: int, status: str) -> None:
                if progress_callback:
                    progress_callback(90 + int(max(0, min(100, value)) * 10 / 100), status)
                if cancel_check and cancel_check():
                    raise RunCancelled()

            try:
                meanings, meaning_metrics = self.cluster_meaning_service.generate(
                    clusters_by_key=result.clusters_by_key,
                    all_image_paths=image_paths,
                    snapshot_key=snapshot_key,
                    path_fingerprints=path_fingerprint_map,
                    embedding_service=self.embedding_service,
                    selected_models=models,
                    requested_model=request.cluster_meaning_model,
                    use_embedding_cache_lookup=request.use_embedding_cache_lookup,
                    progress_callback=meaning_progress,
                    cancel_check=cancel_check,
                )
                result.cluster_meanings_by_key = meanings
                metrics.update(meaning_metrics)
            except RunCancelled:
                raise
            except ClusterMeaningCancelled as exc:
                raise RunCancelled(str(exc)) from exc
            except Exception as exc:
                LOGGER.exception("Cluster meaning sidecar failed")
                metrics["cluster_meaning_status"] = "unavailable"
                metrics["cluster_meaning_error"] = str(exc)
            metrics["cluster_meaning_wall_time_s"] = round(time.perf_counter() - meaning_start, 3)

        metrics["embedding_time_s"] = round(total_embedding_stage_s, 3)
        metrics["embedding_stage_time_s"] = round(total_embedding_stage_s, 3)
        metrics["cache_hits"] = cache_hits
        metrics["cache_misses"] = cache_misses
        metrics["model_load_time_s"] = round(model_load_time_s, 3)
        metrics["model_device"] = model_device
        metrics["embedding_backend"] = embedding_backend
        metrics["amp_enabled"] = amp_enabled
        metrics["effective_mode"] = effective_mode
        metrics["onnx_provider"] = onnx_provider
        metrics["runtime_reason"] = runtime_reason
        if last_batch_size is not None:
            metrics["batch_size"] = last_batch_size
        metrics["cache_lookup_time_s"] = round(total_cache_lookup_s, 3)
        metrics["preprocess_time_s"] = round(total_preprocess_s, 3)
        metrics["inference_time_s"] = round(total_inference_s, 3)
        metrics["prepared_matrix_time_s"] = round(total_prepared_matrix_s, 3)
        metrics["clustering_time_s"] = round(total_cluster_s, 3)
        metrics["backend_wall_time_s"] = round(total_backend_wall_s, 3)
        metrics["index_persist_time_s"] = round(total_index_persist_s, 3)
        metrics["index_reuse"] = ", ".join(
            f"{model}:{'reused' if reused else 'written'}" for model, reused in sorted(index_reuse_flags.items())
        )
        metrics["oom_backoff"] = oom_backoff_used
        metrics["backend_cache_hits"] = ", ".join(
            f"{key}:{'hit' if backend_hit_flags.get(key) else 'miss'}" for key in sorted(backend_hit_flags.keys())
        )
        metrics["pipeline_wall_time_s"] = round(time.perf_counter() - pipeline_start, 3)
        metrics["run_stage_wall_time_s"] = round(time.perf_counter() - embed_total_start, 3)

        LOGGER.info("Run metrics: %s", metrics)
        return result, metrics

    def _discover_result(
        self,
        directory: str,
        *,
        recursive: bool,
        progress_callback=None,
        cancel_check=None,
    ) -> DiscoveryResult:
        discover_result = self.discovery_service.discover_result
        try:
            parameters = inspect.signature(discover_result).parameters
        except (TypeError, ValueError):
            parameters = {}
        kwargs = {"recursive": recursive}
        if "progress_callback" in parameters:
            kwargs["progress_callback"] = progress_callback
        if "cancel_check" in parameters:
            kwargs["cancel_check"] = cancel_check
        return discover_result(directory, **kwargs)

    def _embedding_signature(self, model_name: str, use_onnx: bool) -> str:
        manager = getattr(self.embedding_service, "model_manager", None)
        signature_fn = getattr(manager, "static_signature", None)
        if not callable(signature_fn):
            return ""
        try:
            return str(signature_fn(model_name, use_onnx) or "")
        except Exception:
            return ""

    def _try_result_cache_only(
        self,
        request: ClusteringRequest,
        *,
        models: list[str],
        similarity_modes: list[str],
        backends: list[str],
        snapshot_key: str,
        metrics: dict[str, object],
        image_paths: list[str],
        pipeline_start: float,
    ) -> tuple[MultiBackendClusteringResult, dict] | None:
        if not request.reuse_result_cache:
            return None
        if request.generate_cluster_explanations or request.generate_cluster_meanings:
            return None

        cached_items: list[tuple[str, str, str, str, dict[int, list[str]], dict[str, object]]] = []
        for model_name in models:
            embedding_signature = self._embedding_signature(model_name, request.use_onnx)
            for similarity_mode in similarity_modes:
                for backend in backends:
                    result_key = self.result_cache_service.build_result_key(
                        snapshot_key=snapshot_key,
                        embedding_model=model_name,
                        similarity_mode=similarity_mode,
                        clustering_backend=backend,
                        num_clusters=request.num_clusters,
                        outlier_policy=request.outlier_policy,
                        use_onnx=request.use_onnx,
                        embedding_signature=embedding_signature,
                        similarity_space_version=SIMILARITY_SPACE_VERSION,
                    )
                    cached = self.result_cache_service.load(result_key)
                    if cached is None:
                        return None
                    clustered_images, cached_metrics = cached
                    cached_items.append(
                        (
                            self._comparison_key(model_name, similarity_mode, backend),
                            model_name,
                            similarity_mode,
                            backend,
                            clustered_images,
                            dict(cached_metrics),
                        )
                    )

        result = MultiBackendClusteringResult()
        backend_hit_flags: dict[str, bool] = {}
        for key, model_name, similarity_mode, backend, clustered_images, cached_metrics in cached_items:
            backend_metrics = dict(cached_metrics)
            backend_metrics["cached_clustering_time_s"] = float(backend_metrics.get("clustering_time_s", 0.0) or 0.0)
            backend_metrics["clustering_time_s"] = 0.0
            backend_metrics["backend_wall_time_s"] = 0.0
            backend_metrics["backend_compute_time_s"] = 0.0
            backend_metrics["result_cache_hit"] = True
            backend_metrics["embedding_model"] = model_name
            backend_metrics["similarity_mode"] = similarity_mode
            backend_metrics["backend"] = backend
            result.clusters_by_key[key] = clustered_images
            result.metrics_by_key[key] = backend_metrics
            backend_hit_flags[key] = True
            metrics[f"backend_wall_time_s[{key}]"] = 0.0
            metrics[f"backend_compute_time_s[{key}]"] = 0.0
            metrics[f"backend_cache_hit[{key}]"] = True
            if "prepared_dimension" in backend_metrics:
                metrics[f"prepared_matrix_dim[{model_name}::{similarity_mode}]"] = backend_metrics["prepared_dimension"]
            if "input_dimension" in backend_metrics:
                metrics[f"input_embedding_dim[{model_name}::{similarity_mode}]"] = backend_metrics["input_dimension"]
            if "pca_components" in backend_metrics:
                metrics[f"pca_components[{model_name}::{similarity_mode}]"] = backend_metrics["pca_components"]
            if "similarity_space" in backend_metrics:
                metrics[f"similarity_space[{model_name}::{similarity_mode}]"] = backend_metrics["similarity_space"]
            self._merge_membership(
                result.membership_by_image,
                key,
                clustered_images,
                model_name=model_name,
                similarity_mode=similarity_mode,
                backend=backend,
            )

        metrics.update(
            {
                "full_result_cache_hit": True,
                "embedding_stage_skipped": "result-cache-hit",
                "embedding_time_s": 0.0,
                "embedding_stage_time_s": 0.0,
                "cache_hits": 0,
                "cache_misses": 0,
                "model_load_time_s": 0.0,
                "model_device": "",
                "embedding_backend": "skipped",
                "amp_enabled": False,
                "effective_mode": "",
                "onnx_provider": "",
                "runtime_reason": "Skipped embedding/model load because all requested clustering results were cached.",
                "cache_lookup_time_s": 0.0,
                "preprocess_time_s": 0.0,
                "inference_time_s": 0.0,
                "prepared_matrix_time_s": 0.0,
                "clustering_time_s": 0.0,
                "backend_wall_time_s": 0.0,
                "index_persist_time_s": 0.0,
                "index_reuse": "skipped-result-cache-hit",
                "oom_backoff": False,
                "backend_cache_hits": ", ".join(f"{key}:hit" for key in sorted(backend_hit_flags)),
                "pipeline_wall_time_s": round(time.perf_counter() - pipeline_start, 3),
                "run_stage_wall_time_s": 0.0,
                "cached_image_count": len(image_paths),
            }
        )
        LOGGER.info("Run metrics: %s", metrics)
        return result, metrics

    @staticmethod
    def _merge_membership(
        membership_by_image: dict[str, dict[str, dict[str, object]]],
        key: str,
        clustered_images: dict[int, list[str]],
        model_name: str,
        similarity_mode: str,
        backend: str,
    ) -> None:
        for cluster_id, image_paths in clustered_images.items():
            cluster_size = len(image_paths)
            for rank, image_path in enumerate(image_paths, start=1):
                membership_by_image.setdefault(image_path, {})[key] = {
                    "embedding_model": model_name,
                    "similarity_mode": similarity_mode,
                    "backend": backend,
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "rank": rank,
                    "outlier": cluster_id == -1,
                }

    @staticmethod
    def _comparison_key(model_name: str, similarity_mode: str, backend: str) -> str:
        return f"{model_name}::{similarity_mode}::{backend}"

    @staticmethod
    def _clusters_to_indices(
        clustered_images: dict[int, list[str]],
        index_by_path: dict[str, int],
    ) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {}
        for cluster_id, image_paths in clustered_images.items():
            indices = [int(index_by_path[path]) for path in image_paths if path in index_by_path]
            if indices:
                clusters[int(cluster_id)] = indices
        return clusters
