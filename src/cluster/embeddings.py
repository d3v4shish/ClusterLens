from ml.embeddings import EmbeddingService, ModelManager


class EmbeddingGenerator:
    def __init__(self, model_name="resnet"):
        self.model_name = model_name
        self.service = EmbeddingService(ModelManager())

    def get(self, image_path):
        embeddings, _ = self.service.embed_paths([image_path], self.model_name)
        return embeddings[0][1]
