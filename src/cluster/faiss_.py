from ml.clustering import ClusteringService


class Clustering:
    def __init__(self, num_threads=8):
        self.service = ClusteringService()

    def cluster(self, embeddings, num_clusters, method="sklearn", pca_dim=None, num_neighbors=None, progress_bar=None):
        clusters, _ = self.service.cluster(embeddings, num_clusters, backend=method, pca_dim=pca_dim)
        return clusters
