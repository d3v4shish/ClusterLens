from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SimilarityGraphService:
    def graph_cluster(
        self,
        matrix: np.ndarray,
        min_similarity: float,
        min_cluster_size: int,
        k_neighbors: int = 20,
    ) -> np.ndarray:
        if len(matrix) == 0:
            return np.asarray([], dtype=np.int32)
        n_neighbors = min(max(2, k_neighbors), len(matrix))
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
        return labels
