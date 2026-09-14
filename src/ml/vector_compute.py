from __future__ import annotations

import importlib
from dataclasses import dataclass
from threading import RLock

import numpy as np

from infra.logging_config import get_logger
from infra.runtime import ExecutionPolicy

LOGGER = get_logger(__name__)

# GPU vector jobs can be submitted by concurrent clustering backends. Keeping
# one bounded workspace active at a time avoids transient VRAM spikes and makes
# operation-level CPU fallback predictable.
_CUDA_COMPUTE_LOCK = RLock()
_CUDA_WORKSPACE_BYTES = 256 * 1024 * 1024
_CUDA_MAX_ROWS_PER_BATCH = 65_536


@dataclass(frozen=True)
class VectorComputeInfo:
    device: str
    implementation: str
    fallback_reason: str = ""


def _load_torch():
    return importlib.import_module("torch")


def _load_cuml_hdbscan():
    return importlib.import_module("cuml.cluster").HDBSCAN


def _contiguous_float32(matrix: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(matrix, dtype=np.float32)


class VectorComputeService:
    """CUDA-first dense vector operations with a vectorized CPU fallback."""

    def __init__(self, execution_policy: ExecutionPolicy | None = None) -> None:
        self.execution_policy = execution_policy or ExecutionPolicy()
        self._cuml_hdbscan_provider = None
        self._cuml_hdbscan_probe_complete = False
        self._cuml_hdbscan_error = ""

    @property
    def cuda_enabled(self) -> bool:
        return bool(self.execution_policy.uses_cuda)

    @property
    def cuml_hdbscan_available(self) -> bool:
        return self._load_cuml_hdbscan_provider() is not None

    def _load_cuml_hdbscan_provider(self):
        """Load cuML once per compute service and retain failed probes too."""

        if self._cuml_hdbscan_probe_complete:
            return self._cuml_hdbscan_provider
        self._cuml_hdbscan_probe_complete = True
        try:
            self._cuml_hdbscan_provider = _load_cuml_hdbscan()
        except Exception as exc:
            self._cuml_hdbscan_provider = None
            self._cuml_hdbscan_error = f"cuML HDBSCAN is unavailable: {exc}"
        return self._cuml_hdbscan_provider

    def cosine_scores(
        self,
        matrix: np.ndarray,
        query: np.ndarray,
        *,
        normalize: bool = False,
    ) -> tuple[np.ndarray, VectorComputeInfo]:
        scores, compute_info = self.cosine_matrix(
            matrix,
            np.asarray(query, dtype=np.float32).reshape(1, -1),
            normalize=normalize,
        )
        return np.asarray(scores[:, 0], dtype=np.float32), compute_info

    def cosine_matrix(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        normalize: bool = False,
    ) -> tuple[np.ndarray, VectorComputeInfo]:
        left_matrix = _contiguous_float32(left)
        right_matrix = _contiguous_float32(right)
        if self.cuda_enabled:
            try:
                with _CUDA_COMPUTE_LOCK:
                    torch = self._verified_cuda_torch()
                    gpu_left = torch.as_tensor(left_matrix, dtype=torch.float32, device="cuda")
                    gpu_right = torch.as_tensor(right_matrix, dtype=torch.float32, device="cuda")
                    if normalize:
                        gpu_left = gpu_left / torch.clamp(
                            torch.linalg.vector_norm(gpu_left, dim=1, keepdim=True),
                            min=1e-12,
                        )
                        gpu_right = gpu_right / torch.clamp(
                            torch.linalg.vector_norm(gpu_right, dim=1, keepdim=True),
                            min=1e-12,
                        )
                    scores = torch.matmul(gpu_left, gpu_right.T).cpu().numpy()
                return np.asarray(scores, dtype=np.float32), VectorComputeInfo(
                    device="cuda",
                    implementation="torch-cuda-matmul",
                )
            except Exception as exc:
                reason = self._cuda_fallback_reason("dense similarity scoring", exc)
                LOGGER.warning(reason)
                self._release_cuda_cache()
                fallback_reason = reason
        else:
            fallback_reason = ""

        if normalize:
            left_matrix = left_matrix / np.clip(
                np.linalg.norm(left_matrix, axis=1, keepdims=True),
                1e-12,
                None,
            )
            right_matrix = right_matrix / np.clip(
                np.linalg.norm(right_matrix, axis=1, keepdims=True),
                1e-12,
                None,
            )
        scores = left_matrix @ right_matrix.T
        return np.asarray(scores, dtype=np.float32), VectorComputeInfo(
            device="cpu",
            implementation="numpy-blas-matmul",
            fallback_reason=fallback_reason,
        )

    def pca_project(
        self,
        matrix: np.ndarray,
        component_count: int,
    ) -> tuple[np.ndarray | None, VectorComputeInfo]:
        if not self.cuda_enabled:
            return None, VectorComputeInfo("cpu", "sklearn-pca")

        contiguous = _contiguous_float32(matrix)
        try:
            with _CUDA_COMPUTE_LOCK:
                torch = self._verified_cuda_torch()
                data = torch.as_tensor(contiguous, dtype=torch.float32, device="cuda")
                centered = data - torch.mean(data, dim=0, keepdim=True)
                _u, _singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                projected = torch.matmul(centered, vh[: int(component_count)].T)
                projected = projected / torch.clamp(
                    torch.linalg.vector_norm(projected, dim=1, keepdim=True),
                    min=1e-12,
                )
                result = projected.cpu().numpy()
            return np.ascontiguousarray(result, dtype=np.float32), VectorComputeInfo(
                device="cuda",
                implementation="torch-cuda-svd-pca",
            )
        except Exception as exc:
            reason = self._cuda_fallback_reason("semantic PCA", exc)
            LOGGER.warning(reason)
            self._release_cuda_cache()
            return None, VectorComputeInfo(
                device="cpu",
                implementation="sklearn-pca",
                fallback_reason=reason,
            )

    def cosine_neighbors(
        self,
        matrix: np.ndarray,
        neighbor_count: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, VectorComputeInfo]:
        contiguous = _contiguous_float32(matrix)
        if not self.cuda_enabled:
            return None, None, VectorComputeInfo(
                device="cpu",
                implementation="sklearn-nearest-neighbors",
            )

        try:
            with _CUDA_COMPUTE_LOCK:
                torch = self._verified_cuda_torch()
                gpu_matrix = torch.as_tensor(contiguous, dtype=torch.float32, device="cuda")
                gpu_matrix = gpu_matrix / torch.clamp(
                    torch.linalg.vector_norm(gpu_matrix, dim=1, keepdim=True),
                    min=1e-12,
                )
                count = max(1, min(int(neighbor_count), int(gpu_matrix.shape[0])))
                batch_rows = self._row_batch_size(int(gpu_matrix.shape[0]))
                score_parts = []
                index_parts = []
                for start in range(0, int(gpu_matrix.shape[0]), batch_rows):
                    similarities = torch.matmul(gpu_matrix[start : start + batch_rows], gpu_matrix.T)
                    scores, indices = torch.topk(similarities, count, dim=1, largest=True, sorted=True)
                    score_parts.append(scores.cpu())
                    index_parts.append(indices.cpu())
                scores = torch.cat(score_parts, dim=0).numpy()
                indices = torch.cat(index_parts, dim=0).numpy()
            distances = np.asarray(1.0 - scores, dtype=np.float32)
            return distances, np.asarray(indices, dtype=np.int64), VectorComputeInfo(
                device="cuda",
                implementation="torch-cuda-chunked-topk",
            )
        except Exception as exc:
            reason = self._cuda_fallback_reason("graph neighbor search", exc)
            LOGGER.warning(reason)
            self._release_cuda_cache()
            return None, None, VectorComputeInfo(
                device="cpu",
                implementation="sklearn-nearest-neighbors",
                fallback_reason=reason,
            )

    def kmeans_labels(
        self,
        matrix: np.ndarray,
        cluster_count: int,
        *,
        seed: int = 42,
        max_iter: int = 80,
        n_init: int = 3,
    ) -> tuple[np.ndarray | None, VectorComputeInfo]:
        if not self.cuda_enabled:
            return None, VectorComputeInfo(
                device="cpu",
                implementation="sklearn-native-kmeans",
            )

        contiguous = _contiguous_float32(matrix)
        try:
            with _CUDA_COMPUTE_LOCK:
                labels = self._cuda_kmeans_labels(
                    contiguous,
                    cluster_count,
                    seed=seed,
                    max_iter=max_iter,
                    n_init=n_init,
                )
            return labels, VectorComputeInfo(
                device="cuda",
                implementation="torch-cuda-kmeans",
            )
        except Exception as exc:
            reason = self._cuda_fallback_reason("cosine K-means", exc)
            LOGGER.warning(reason)
            self._release_cuda_cache()
            return None, VectorComputeInfo(
                device="cpu",
                implementation="sklearn-native-kmeans",
                fallback_reason=reason,
            )

    def silhouette_score(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        *,
        sample_size: int,
        seed: int = 42,
    ) -> tuple[float | None, VectorComputeInfo]:
        if not self.cuda_enabled:
            return None, VectorComputeInfo("cpu", "sklearn-silhouette")

        contiguous = _contiguous_float32(matrix)
        label_values = np.ascontiguousarray(labels, dtype=np.int64)
        if len(contiguous) > int(sample_size):
            selection = np.random.RandomState(seed).permutation(len(contiguous))[: int(sample_size)]
            contiguous = np.ascontiguousarray(contiguous[selection], dtype=np.float32)
            label_values = np.ascontiguousarray(label_values[selection], dtype=np.int64)
        unique_labels = np.unique(label_values)
        if len(unique_labels) < 2 or len(contiguous) <= len(unique_labels):
            return None, VectorComputeInfo("cuda", "torch-cuda-silhouette")

        try:
            with _CUDA_COMPUTE_LOCK:
                torch = self._verified_cuda_torch()
                data = torch.as_tensor(contiguous, dtype=torch.float32, device="cuda")
                gpu_labels = torch.as_tensor(label_values, dtype=torch.long, device="cuda")
                distances = torch.cdist(data, data, p=2.0)
                _unique, inverse = torch.unique(gpu_labels, sorted=True, return_inverse=True)
                membership = torch.nn.functional.one_hot(
                    inverse,
                    num_classes=int(len(unique_labels)),
                ).to(torch.float32)
                counts = torch.sum(membership, dim=0)
                distance_sums = torch.matmul(distances, membership)
                own_sums = torch.gather(distance_sums, 1, inverse.unsqueeze(1)).squeeze(1)
                own_counts = counts[inverse]
                intra = own_sums / torch.clamp(own_counts - 1.0, min=1.0)
                mean_to_clusters = distance_sums / counts.unsqueeze(0)
                mean_to_clusters.scatter_(1, inverse.unsqueeze(1), float("inf"))
                nearest_other = torch.min(mean_to_clusters, dim=1).values
                denominator = torch.maximum(intra, nearest_other)
                silhouettes = torch.where(
                    (own_counts > 1.0) & (denominator > 0),
                    (nearest_other - intra) / denominator,
                    torch.zeros_like(intra),
                )
                score = float(torch.mean(silhouettes).item())
            return score, VectorComputeInfo("cuda", "torch-cuda-silhouette")
        except Exception as exc:
            reason = self._cuda_fallback_reason("silhouette scoring", exc)
            LOGGER.warning(reason)
            self._release_cuda_cache()
            return None, VectorComputeInfo(
                "cpu",
                "sklearn-silhouette",
                reason,
            )

    def hdbscan_labels(
        self,
        matrix: np.ndarray,
        *,
        min_cluster_size: int,
        min_samples: int | None,
        cluster_selection_epsilon: float,
        allow_single_cluster: bool,
    ) -> tuple[np.ndarray | None, VectorComputeInfo]:
        if not self.cuda_enabled:
            return None, VectorComputeInfo("cpu", "hdbscan-native")

        contiguous = _contiguous_float32(matrix)
        provider = self._load_cuml_hdbscan_provider()
        if provider is None:
            return None, VectorComputeInfo(
                "cpu",
                "hdbscan-native",
                self._cuml_hdbscan_error or "cuML HDBSCAN is unavailable in this CUDA runtime.",
            )
        try:
            with _CUDA_COMPUTE_LOCK:
                self._verified_cuda_torch()
                self._release_cuda_cache()
                estimator = provider(
                    min_cluster_size=int(min_cluster_size),
                    min_samples=min_samples,
                    metric="euclidean",
                    cluster_selection_epsilon=float(cluster_selection_epsilon),
                    cluster_selection_method="eom",
                    allow_single_cluster=bool(allow_single_cluster),
                    gen_min_span_tree=False,
                    prediction_data=False,
                    output_type="numpy",
                )
                labels = np.asarray(estimator.fit_predict(contiguous), dtype=np.int32).reshape(-1)
                if len(labels) != len(contiguous):
                    raise RuntimeError(
                        f"cuML returned {len(labels)} labels for {len(contiguous)} samples"
                    )
            return labels, VectorComputeInfo("cuda", "cuml-hdbscan")
        except Exception as exc:
            reason = self._cuda_fallback_reason("HDBSCAN", exc)
            LOGGER.warning(reason)
            self._release_cuda_cache()
            return None, VectorComputeInfo(
                "cpu",
                "hdbscan-native",
                reason,
            )

    def _cuda_kmeans_labels(
        self,
        matrix: np.ndarray,
        cluster_count: int,
        *,
        seed: int,
        max_iter: int,
        n_init: int,
    ) -> np.ndarray:
        torch = self._verified_cuda_torch()
        data = torch.as_tensor(matrix, dtype=torch.float32, device="cuda")
        sample_count = int(data.shape[0])
        cluster_count = max(1, min(int(cluster_count), sample_count))
        row_norms = torch.sum(data * data, dim=1)
        batch_rows = self._row_batch_size(cluster_count)
        best_labels = None
        best_inertia = float("inf")

        for init_index in range(max(1, int(n_init))):
            generator = torch.Generator(device="cuda")
            generator.manual_seed(int(seed) + init_index)
            centroids = self._cuda_kmeans_plus_plus(
                torch,
                data,
                row_norms,
                cluster_count,
                generator,
            )
            previous_labels = None

            for _iteration in range(max(1, int(max_iter))):
                labels = torch.empty(sample_count, dtype=torch.long, device="cuda")
                sums = torch.zeros_like(centroids)
                counts = torch.zeros(cluster_count, dtype=torch.float32, device="cuda")
                centroid_norms = torch.sum(centroids * centroids, dim=1)
                for start in range(0, sample_count, batch_rows):
                    stop = min(sample_count, start + batch_rows)
                    scores = 2.0 * torch.matmul(data[start:stop], centroids.T) - centroid_norms
                    block_labels = torch.argmax(scores, dim=1)
                    labels[start:stop] = block_labels
                    sums.index_add_(0, block_labels, data[start:stop])
                    counts.add_(torch.bincount(block_labels, minlength=cluster_count).to(torch.float32))

                nonempty = counts > 0
                updated = centroids.clone()
                updated[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
                converged = previous_labels is not None and bool(torch.equal(labels, previous_labels))
                centroids = updated
                previous_labels = labels
                if converged:
                    break

            centroid_norms = torch.sum(centroids * centroids, dim=1)
            inertia = torch.zeros((), dtype=torch.float32, device="cuda")
            final_labels = torch.empty(sample_count, dtype=torch.long, device="cuda")
            for start in range(0, sample_count, batch_rows):
                stop = min(sample_count, start + batch_rows)
                scores = 2.0 * torch.matmul(data[start:stop], centroids.T) - centroid_norms
                block_scores, block_labels = torch.max(scores, dim=1)
                final_labels[start:stop] = block_labels
                inertia.add_(torch.sum(row_norms[start:stop] - block_scores))
            inertia_value = float(inertia.item())
            if inertia_value < best_inertia:
                best_inertia = inertia_value
                best_labels = final_labels.clone()

        if best_labels is None:
            raise RuntimeError("CUDA K-means did not produce labels.")
        return np.asarray(best_labels.cpu().numpy(), dtype=np.int32)

    @staticmethod
    def _cuda_kmeans_plus_plus(torch, data, row_norms, cluster_count: int, generator):
        sample_count = int(data.shape[0])
        first = int(torch.randint(sample_count, (1,), generator=generator, device="cuda").item())
        centroid_rows = [data[first].clone()]
        closest_distances = torch.clamp(
            row_norms + torch.sum(centroid_rows[0] * centroid_rows[0])
            - 2.0 * torch.matmul(data, centroid_rows[0]),
            min=0.0,
        )
        for offset in range(1, cluster_count):
            total = torch.sum(closest_distances)
            if not bool(torch.isfinite(total)) or float(total.item()) <= 1e-12:
                next_index = (first + offset) % sample_count
            else:
                next_index = int(
                    torch.multinomial(
                        closest_distances / total,
                        1,
                        replacement=False,
                        generator=generator,
                    ).item()
                )
            centroid = data[next_index].clone()
            centroid_rows.append(centroid)
            distances = torch.clamp(
                row_norms + torch.sum(centroid * centroid) - 2.0 * torch.matmul(data, centroid),
                min=0.0,
            )
            closest_distances = torch.minimum(closest_distances, distances)
        return torch.stack(centroid_rows, dim=0)

    @staticmethod
    def _row_batch_size(output_columns: int) -> int:
        bytes_per_row = max(1, int(output_columns)) * np.dtype(np.float32).itemsize
        workspace_rows = max(1, _CUDA_WORKSPACE_BYTES // bytes_per_row)
        return max(1, min(_CUDA_MAX_ROWS_PER_BATCH, int(workspace_rows)))

    @staticmethod
    def _cuda_fallback_reason(operation: str, exc: Exception) -> str:
        detail = str(exc).strip() or type(exc).__name__
        return f"CUDA {operation} failed; using the vectorized CPU fallback: {detail}"

    @staticmethod
    def _release_cuda_cache() -> None:
        try:
            torch = _load_torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @staticmethod
    def _verified_cuda_torch():
        torch = _load_torch()
        if not bool(torch.cuda.is_available()):
            raise RuntimeError("the selected CUDA Torch device is no longer available")
        return torch
