from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from app.services.cluster_explanations import ClusterExplanation
from app.services.similarity_modes import SUPPORTED_SIMILARITY_MODES
from app.services.similarity_graph import SimilarityGraphService
from infra.logging_config import get_logger
from infra.settings import get_settings

LOGGER = get_logger(__name__)
DEFAULT_SEMANTIC_PCA_DIM = 50
CLUSTER_EXPLANATION_SCORE_LIMIT = 24
MIN_SEMANTIC_PCA_SAMPLES = 4


@dataclass(frozen=True)
class PreparedMatrixInfo:
    similarity_mode: str
    input_dimension: int
    prepared_dimension: int
    pca_components: int | None
    normalized: bool
    space_label: str

    @classmethod
    def from_matrix(cls, matrix: np.ndarray, similarity_mode: str) -> "PreparedMatrixInfo":
        dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        mode = _normalize_similarity_mode(similarity_mode)
        return cls(
            similarity_mode=mode,
            input_dimension=dimension,
            prepared_dimension=dimension,
            pca_components=None,
            normalized=True,
            space_label=f"{mode} full vector | input {dimension} -> full {dimension} | normalized",
        )


try:
    import faiss
except Exception:
    faiss = None

try:
    import hdbscan
except Exception:
    hdbscan = None


def _normalize_similarity_mode(similarity_mode: str) -> str:
    mode = str(similarity_mode or "").strip().lower()
    if mode not in SUPPORTED_SIMILARITY_MODES:
        return "semantic"
    return mode


def _safe_semantic_pca_components(matrix: np.ndarray, target_dim: int) -> int | None:
    if matrix.ndim != 2:
        return None
    sample_count, input_dimension = matrix.shape
    if sample_count < MIN_SEMANTIC_PCA_SAMPLES or input_dimension <= 2:
        return None
    max_components = min(sample_count - 1, input_dimension)
    components = min(max(2, target_dim), max_components)
    if components >= input_dimension:
        return None
    return int(components)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class ClusteringService:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.graph_service = SimilarityGraphService()

    def cluster(
        self,
        embeddings: list[np.ndarray],
        num_clusters: int,
        backend: str = "cosine-kmeans",
        pca_dim: int | None = None,
        similarity_mode: str = "semantic",
        outlier_policy: str = "assign",
        backend_options: dict[str, object] | None = None,
    ) -> tuple[dict[int, list[int]], dict]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape[0] == 0:
            return {}, {"backend": backend, "outlier_count": 0}
        if matrix.shape[0] < num_clusters and backend not in {"hdbscan", "graph"}:
            raise ValueError("Cluster count cannot exceed image count.")

        metric_matrix = self.prepare_matrix(matrix, pca_dim, similarity_mode)
        return self.cluster_prepared(
            metric_matrix,
            num_clusters,
            backend=backend,
            outlier_policy=outlier_policy,
            backend_options=backend_options,
        )

    def prepare_matrix(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        pca_dim: int | None = None,
        similarity_mode: str = "semantic",
    ) -> np.ndarray:
        prepared, _info = self.prepare_matrix_with_info(embeddings, pca_dim, similarity_mode)
        return prepared

    def prepare_matrix_with_info(
        self,
        embeddings: list[np.ndarray] | np.ndarray,
        pca_dim: int | None = None,
        similarity_mode: str = "semantic",
    ) -> tuple[np.ndarray, PreparedMatrixInfo]:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.shape[0] == 0:
            return matrix, PreparedMatrixInfo.from_matrix(matrix, similarity_mode)
        return self._prepare_matrix_with_info(matrix, pca_dim, similarity_mode)

    def cluster_prepared(
        self,
        metric_matrix: np.ndarray,
        num_clusters: int,
        *,
        backend: str = "cosine-kmeans",
        outlier_policy: str = "assign",
        performance_profile: str | None = None,
        backend_options: dict[str, object] | None = None,
    ) -> tuple[dict[int, list[int]], dict]:
        labels, used_backend = self._labels_for_backend(
            metric_matrix,
            num_clusters,
            backend,
            performance_profile=performance_profile,
            backend_options=backend_options,
        )
        clusters = self._clusters_from_labels(metric_matrix, labels, num_clusters, outlier_policy)
        clusters = self._merge_tiny_clusters(metric_matrix, clusters)
        clusters = self._rerank_cluster_members(metric_matrix, clusters)

        metrics = {
            "backend": used_backend,
            "outlier_count": len(clusters.get(-1, [])),
            "cluster_quality_score": self._cluster_quality(metric_matrix, labels),
        }
        return clusters, metrics

    def build_cluster_explanations(
        self,
        metric_matrix: np.ndarray,
        clusters: dict[int, list[int]],
        *,
        cluster_quality_score: float | None = None,
        prepared_info: PreparedMatrixInfo | None = None,
    ) -> dict[int, ClusterExplanation]:
        matrix = np.asarray(metric_matrix, dtype=np.float32)
        if matrix.shape[0] == 0 or not clusters:
            return {}

        centroids = self._cluster_centroids(matrix, {cluster_id: members for cluster_id, members in clusters.items() if cluster_id != -1})
        explanations: dict[int, ClusterExplanation] = {}
        for cluster_id, members in sorted(clusters.items(), key=lambda item: int(item[0])):
            normalized_members = [int(index) for index in members]
            if not normalized_members:
                continue
            is_outlier = int(cluster_id) == -1
            if is_outlier or cluster_id not in centroids:
                explanations[int(cluster_id)] = ClusterExplanation(
                    cluster_id=int(cluster_id),
                    cluster_size=len(normalized_members),
                    is_outlier=True,
                    representative_images_note="Preview list preserves the backend member order for this outlier cluster.",
                    cluster_quality_score=cluster_quality_score,
                    similarity_mode=prepared_info.similarity_mode if prepared_info is not None else "",
                    similarity_space=prepared_info.space_label if prepared_info is not None else "",
                    input_dimension=prepared_info.input_dimension if prepared_info is not None else None,
                    prepared_dimension=prepared_info.prepared_dimension if prepared_info is not None else None,
                    pca_components=prepared_info.pca_components if prepared_info is not None else None,
                )
                continue

            centroid = centroids[cluster_id]
            member_vectors = matrix[np.asarray(normalized_members, dtype=np.int32)]
            member_scores = np.asarray(member_vectors @ centroid, dtype=np.float32)
            mean_similarity = float(member_scores.mean()) if member_scores.size else None
            median_similarity = float(np.median(member_scores)) if member_scores.size else None
            min_similarity = float(member_scores.min()) if member_scores.size else None
            max_similarity = float(member_scores.max()) if member_scores.size else None

            nearest_cluster_id = None
            nearest_cluster_similarity = None
            separation_margin = None
            competing_centroids = {
                other_cluster_id: other_centroid
                for other_cluster_id, other_centroid in centroids.items()
                if int(other_cluster_id) != int(cluster_id)
            }
            if competing_centroids:
                nearest_cluster_id = max(
                    competing_centroids,
                    key=lambda other_cluster_id: float(np.dot(centroid, competing_centroids[other_cluster_id])),
                )
                nearest_cluster_similarity = float(np.dot(centroid, competing_centroids[nearest_cluster_id]))
                if mean_similarity is not None:
                    separation_margin = float(mean_similarity - nearest_cluster_similarity)

            explanations[int(cluster_id)] = ClusterExplanation(
                cluster_id=int(cluster_id),
                cluster_size=len(normalized_members),
                is_outlier=False,
                cohesion_mean=mean_similarity,
                cohesion_median=median_similarity,
                cohesion_min=min_similarity,
                cohesion_max=max_similarity,
                nearest_cluster_id=int(nearest_cluster_id) if nearest_cluster_id is not None else None,
                nearest_cluster_similarity=nearest_cluster_similarity,
                separation_margin=separation_margin,
                representative_images_note="Preview list is already ordered by centroid similarity.",
                cluster_quality_score=cluster_quality_score,
                similarity_mode=prepared_info.similarity_mode if prepared_info is not None else "",
                similarity_space=prepared_info.space_label if prepared_info is not None else "",
                input_dimension=prepared_info.input_dimension if prepared_info is not None else None,
                prepared_dimension=prepared_info.prepared_dimension if prepared_info is not None else None,
                pca_components=prepared_info.pca_components if prepared_info is not None else None,
                member_cohesion_scores=tuple(
                    float(score) for score in member_scores[:CLUSTER_EXPLANATION_SCORE_LIMIT]
                ),
            )
        return explanations

    def _prepare_matrix(self, matrix: np.ndarray, pca_dim: int | None, similarity_mode: str) -> np.ndarray:
        prepared, _info = self._prepare_matrix_with_info(matrix, pca_dim, similarity_mode)
        return prepared

    def _prepare_matrix_with_info(
        self,
        matrix: np.ndarray,
        pca_dim: int | None,
        similarity_mode: str,
    ) -> tuple[np.ndarray, PreparedMatrixInfo]:
        mode = _normalize_similarity_mode(similarity_mode)
        matrix = np.asarray(matrix, dtype=np.float32)
        input_dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        pca_components = None
        pca_applied = False

        if mode == "semantic":
            target_dim = int(pca_dim or DEFAULT_SEMANTIC_PCA_DIM)
            pca_components = _safe_semantic_pca_components(matrix, target_dim)
            if pca_components is not None:
                try:
                    matrix = PCA(n_components=pca_components, random_state=42).fit_transform(matrix)
                    pca_applied = True
                except Exception as exc:
                    LOGGER.warning(
                        "Semantic PCA projection failed; falling back to full normalized vectors: samples=%s input_dim=%s target_dim=%s error=%s",
                        matrix.shape[0],
                        input_dimension,
                        target_dim,
                        exc,
                    )
                    pca_components = None

        matrix = _l2_normalize(matrix)
        prepared_dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
        if mode == "semantic" and pca_applied:
            space_label = (
                "semantic projection | "
                f"input {input_dimension} -> PCA {prepared_dimension} | normalized"
            )
        elif mode == "semantic":
            space_label = (
                "semantic projection fallback | "
                f"input {input_dimension} -> full {prepared_dimension} | normalized"
            )
        else:
            space_label = f"cosine full vector | input {input_dimension} -> full {prepared_dimension} | normalized"

        info = PreparedMatrixInfo(
            similarity_mode=mode,
            input_dimension=input_dimension,
            prepared_dimension=prepared_dimension,
            pca_components=pca_components if pca_applied else None,
            normalized=True,
            space_label=space_label,
        )
        return np.asarray(matrix, dtype=np.float32), info

    def _labels_for_backend(
        self,
        matrix: np.ndarray,
        num_clusters: int,
        backend: str,
        *,
        performance_profile: str | None = None,
        backend_options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, str]:
        profile = str(performance_profile or "balanced").strip().lower()
        if backend == "faiss" and faiss is not None:
            LOGGER.info("Clustering with FAISS KMeans")
            return self._faiss_kmeans(matrix, num_clusters), "faiss"
        if backend == "hdbscan" and hdbscan is not None:
            LOGGER.info("Clustering with HDBSCAN")
            options = dict(backend_options or {})
            min_cluster_size = max(
                2,
                self._backend_option_int(
                    options,
                    "min_cluster_size",
                    self.settings.min_graph_cluster_size,
                    minimum=2,
                ),
            )
            min_samples_value = self._backend_option_int(options, "min_samples", 0, minimum=0)
            min_samples = None if min_samples_value <= 0 else min_samples_value
            cluster_selection_epsilon = self._backend_option_float(options, "cluster_selection_epsilon", 0.0, minimum=0.0)
            allow_single_cluster = self._backend_option_bool(options, "allow_single_cluster", False)
            labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_epsilon=cluster_selection_epsilon,
                cluster_selection_method="eom",
                allow_single_cluster=allow_single_cluster,
            ).fit_predict(matrix)
            return labels.astype(np.int32), "hdbscan"
        if backend == "graph":
            LOGGER.info("Clustering with kNN graph")
            labels = self.graph_service.graph_cluster(
                matrix,
                min_similarity=self.settings.min_graph_similarity,
                min_cluster_size=self.settings.min_graph_cluster_size,
            )
            return labels.astype(np.int32), "graph"
        if backend == "cosine-kmeans":
            LOGGER.info("Clustering with cosine MiniBatch/KMeans")
            matrix = np.asarray(matrix, dtype=np.float32)
        if profile == "max_speed" and backend in {"cosine-kmeans", "sklearn"} and matrix.shape[0] >= 512:
            labels = MiniBatchKMeans(
                n_clusters=num_clusters,
                random_state=42,
                batch_size=max(1024, num_clusters * 32),
                n_init=3,
                max_iter=80,
            ).fit_predict(matrix)
            return labels.astype(np.int32), "minibatch-kmeans-fast" if backend == "sklearn" else "cosine-kmeans-fast"
        if matrix.shape[0] >= self.settings.minibatch_kmeans_threshold:
            labels = MiniBatchKMeans(
                n_clusters=num_clusters,
                random_state=42,
                batch_size=max(1024, num_clusters * 16),
                n_init="auto",
            ).fit_predict(matrix)
            return labels.astype(np.int32), "minibatch-kmeans" if backend == "sklearn" else backend
        labels = KMeans(n_clusters=num_clusters, random_state=42, n_init=10).fit_predict(matrix)
        return labels.astype(np.int32), "sklearn" if backend == "sklearn" else backend

    @staticmethod
    def _backend_option_int(
        options: dict[str, object],
        key: str,
        default: int,
        *,
        minimum: int | None = None,
    ) -> int:
        try:
            value = int(options.get(key, default) or default)
        except Exception:
            value = int(default)
        if minimum is not None:
            value = max(int(minimum), value)
        return value

    @staticmethod
    def _backend_option_float(
        options: dict[str, object],
        key: str,
        default: float,
        *,
        minimum: float | None = None,
    ) -> float:
        try:
            value = float(options.get(key, default) or default)
        except Exception:
            value = float(default)
        if minimum is not None:
            value = max(float(minimum), value)
        return value

    @staticmethod
    def _backend_option_bool(options: dict[str, object], key: str, default: bool) -> bool:
        value = options.get(key, default)
        if isinstance(value, bool):
            return value
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    def _clusters_from_labels(
        self,
        matrix: np.ndarray,
        labels: np.ndarray,
        num_clusters: int,
        outlier_policy: str,
    ) -> dict[int, list[int]]:
        clusters: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(index)
        if -1 not in clusters and outlier_policy == "isolate" and len(matrix) >= max(4, num_clusters):
            clusters = self._isolate_outliers(matrix, clusters)
        if outlier_policy == "assign" and -1 in clusters and len(clusters) > 1:
            outliers = clusters.pop(-1)
            centroids = self._cluster_centroids(matrix, clusters)
            for index in outliers:
                target = self._nearest_cluster(matrix[index], centroids)
                clusters.setdefault(target, []).append(index)
        return {cluster_id: members for cluster_id, members in clusters.items() if members}

    def _isolate_outliers(self, matrix: np.ndarray, clusters: dict[int, list[int]]) -> dict[int, list[int]]:
        centroids = self._cluster_centroids(matrix, clusters)
        all_scores = []
        owner = {}
        for cluster_id, members in clusters.items():
            centroid = centroids[cluster_id]
            for index in members:
                score = float(np.dot(matrix[index], centroid))
                all_scores.append(score)
                owner[index] = cluster_id
        if not all_scores:
            return clusters
        threshold = np.quantile(np.asarray(all_scores), 1.0 - self.settings.outlier_quantile)
        updated = {cluster_id: [] for cluster_id in clusters}
        updated[-1] = []
        for index, cluster_id in owner.items():
            if float(np.dot(matrix[index], centroids[cluster_id])) < threshold:
                updated[-1].append(index)
            else:
                updated[cluster_id].append(index)
        return {cluster_id: members for cluster_id, members in updated.items() if members}

    def _merge_tiny_clusters(self, matrix: np.ndarray, clusters: dict[int, list[int]]) -> dict[int, list[int]]:
        min_size = max(2, self.settings.tiny_cluster_min_size)
        if len(clusters) <= 1:
            return clusters
        large = {cluster_id: members for cluster_id, members in clusters.items() if cluster_id == -1 or len(members) >= min_size}
        tiny = {cluster_id: members for cluster_id, members in clusters.items() if cluster_id != -1 and len(members) < min_size}
        if not tiny:
            return clusters
        if not any(cluster_id != -1 for cluster_id in large):
            return clusters
        centroids = self._cluster_centroids(matrix, {k: v for k, v in large.items() if k != -1})
        for members in tiny.values():
            for index in members:
                target = self._nearest_cluster(matrix[index], centroids)
                large.setdefault(target, []).append(index)
        return {cluster_id: members for cluster_id, members in large.items() if members}

    def _rerank_cluster_members(self, matrix: np.ndarray, clusters: dict[int, list[int]]) -> dict[int, list[int]]:
        reranked = {}
        centroids = self._cluster_centroids(matrix, {k: v for k, v in clusters.items() if k != -1})
        for cluster_id, members in clusters.items():
            if cluster_id == -1 or cluster_id not in centroids:
                reranked[cluster_id] = members
                continue
            centroid = centroids[cluster_id]
            reranked[cluster_id] = sorted(
                members,
                key=lambda index: float(np.dot(matrix[index], centroid)),
                reverse=True,
            )
        return reranked

    def _cluster_quality(self, matrix: np.ndarray, labels: np.ndarray) -> float | None:
        unique_labels = {int(label) for label in labels if int(label) != -1}
        if len(unique_labels) < 2 or matrix.shape[0] <= len(unique_labels):
            return None
        try:
            sample_size = min(self.settings.silhouette_sample_size, matrix.shape[0])
            return float(silhouette_score(matrix, labels, sample_size=sample_size, random_state=42))
        except Exception:
            return None

    @staticmethod
    def _cluster_centroids(matrix: np.ndarray, clusters: dict[int, list[int]]) -> dict[int, np.ndarray]:
        centroids = {}
        for cluster_id, members in clusters.items():
            if not members:
                continue
            centroid = matrix[np.asarray(members, dtype=np.int32)].mean(axis=0)
            centroid = centroid / np.clip(np.linalg.norm(centroid), 1e-12, None)
            centroids[cluster_id] = centroid.astype(np.float32)
        return centroids

    @staticmethod
    def _nearest_cluster(vector: np.ndarray, centroids: dict[int, np.ndarray]) -> int:
        return max(centroids, key=lambda cluster_id: float(np.dot(vector, centroids[cluster_id])))

    @staticmethod
    def _faiss_kmeans(embeddings: np.ndarray, num_clusters: int) -> np.ndarray:
        d = embeddings.shape[1]
        kmeans = faiss.Kmeans(d, num_clusters, niter=20, verbose=False)
        kmeans.train(embeddings)
        _, assignments = kmeans.index.search(embeddings, 1)
        return assignments.reshape(-1)
