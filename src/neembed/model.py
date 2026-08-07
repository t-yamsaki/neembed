"""Sentence Transformer integration for manifold-valued embeddings."""

from collections.abc import Sequence
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from neembed.manifolds import get_manifold


class ManifoldSentenceTransformer(nn.Module):
    """Map pretrained sentence embeddings onto the configured manifold."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        manifold: str = "poincare",
        embedding_dim: int | None = None,
        curvature: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name_or_path)

        encoder_dim = self.encoder.get_embedding_dimension()
        if encoder_dim is None:
            raise ValueError("Sentence Transformer embedding dimension is unknown")

        self.projection: nn.Module
        if embedding_dim is None:
            self.projection = nn.Identity()
            self.embedding_dim = encoder_dim
        else:
            self.projection = nn.Linear(encoder_dim, embedding_dim)
            self.embedding_dim = embedding_dim

        self.manifold = get_manifold(manifold, curvature)

    def forward(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode sentences and map their embeddings from the origin tangent space."""
        features = self.encoder.preprocess(list(sentences))
        features = {
            key: value.to(self.encoder.device) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        encoder_output: dict[str, Any] = self.encoder(features)
        tangent = self.projection(encoder_output["sentence_embedding"])
        return self.manifold.expmap0(tangent)
