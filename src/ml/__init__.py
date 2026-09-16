"""Machine-learning package with lazy public exports.

Importing a light-weight submodule such as :mod:`ml.vector_compute` must not
initialize Torch or ONNX.  Those libraries are loaded only by an actual
embedding/model request.
"""

from __future__ import annotations

from typing import Any

__all__ = ("ClusteringService", "EmbeddingService", "ModelManager")


def __getattr__(name: str) -> Any:
    if name == "ClusteringService":
        from .clustering import ClusteringService

        return ClusteringService
    if name in {"EmbeddingService", "ModelManager"}:
        from .embeddings import EmbeddingService, ModelManager

        return {"EmbeddingService": EmbeddingService, "ModelManager": ModelManager}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
