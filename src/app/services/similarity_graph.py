from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from infra.runtime import ExecutionPolicy


class SimilarityGraphService:
    def __init__(self, execution_policy: ExecutionPolicy | None = None) -> None:
        from ml.vector_compute import VectorComputeService

        self.vector_compute = VectorComputeService(execution_policy)

    def graph_cluster(
        self,
        matrix: np.ndarray,
        min_similarity: float,
        min_cluster_size: int,
        k_neighbors: int = 20,
    ) -> np.ndarray:
        labels, _compute_info = self.graph_cluster_with_info(
            matrix,
            min_similarity=min_similarity,
            min_cluster_size=min_cluster_size,
            k_neighbors=k_neighbors,
        )
        return labels

    def graph_cluster_with_info(
        self,
        matrix: np.ndarray,
        min_similarity: float,
        min_cluster_size: int,
        k_neighbors: int = 20,
    ) -> tuple[np.ndarray, object]:
        if len(matrix) == 0:
            from ml.vector_compute import VectorComputeInfo

            return np.asarray([], dtype=np.int32), VectorComputeInfo("cpu", "empty")
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        n_neighbors = min(max(2, k_neighbors), len(matrix))
        distances, indices, compute_info = self.vector_compute.cosine_neighbors(matrix, n_neighbors)
        if distances is None or indices is None:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(matrix)
            distances, indices = nbrs.kneighbors(matrix)
        labels = np.full(len(matrix), -1, dtype=np.int32)
        cluster_id = 0
        for start in range(len(matrix)):
            if labels[start] != -1:
                continue
            stack = [start]
            component = []
            labels[start] = -2
            while stack:
                node = stack.pop()
                component.append(node)
                for distance, neighbor in zip(distances[node], indices[node]):
                    if 1.0 - float(distance) < min_similarity:
                        continue
                    if labels[neighbor] == -1:
                        labels[neighbor] = -2
                        stack.append(int(neighbor))
            if len(component) >= min_cluster_size:
                for node in component:
                    labels[node] = cluster_id
                cluster_id += 1
            else:
                for node in component:
                    labels[node] = -1
        return labels, compute_info
