"""Sentence Transformer integration for manifold-valued embeddings."""

from collections.abc import Sequence
import json
from pathlib import Path
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

        self._projection_dim = embedding_dim
        self.projection: nn.Module
        if embedding_dim is None:
            self.projection = nn.Identity()
            self.embedding_dim = encoder_dim
        else:
            self.projection = nn.Linear(encoder_dim, embedding_dim)
            self.embedding_dim = embedding_dim
        self.projection.to(self.encoder.device)

        self.manifold_name = manifold
        self.curvature = float(curvature)
        self.manifold = get_manifold(self.manifold_name, self.curvature)
        self.manifold.to(self.encoder.device)

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

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
    ) -> Any:
        """Return manifold embeddings for one text or a batch of texts."""
        single_input = isinstance(sentences, str)
        batch = [sentences] if single_input else list(sentences)

        self.eval()
        with torch.inference_mode():
            embeddings = self(batch)

        if single_input:
            embeddings = embeddings[0]
        if convert_to_tensor:
            return embeddings
        return embeddings.cpu().numpy()

    def distance(self, a: Any, b: Any) -> torch.Tensor:
        """Return the geodesic distance between two manifold embeddings."""
        reference = next(self.parameters())
        a_tensor = torch.as_tensor(a, device=reference.device, dtype=reference.dtype)
        b_tensor = torch.as_tensor(b, device=reference.device, dtype=reference.dtype)

        with torch.inference_mode():
            return self.manifold.dist(a_tensor, b_tensor)

    def save_pretrained(self, output_path: str | Path) -> None:
        """Save the encoder, projection weights, and neembed configuration."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(str(output_path / "encoder"))
        config = {
            "embedding_dim": self._projection_dim,
            "manifold": self.manifold_name,
            "curvature": self.curvature,
        }
        (output_path / "neembed_config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        torch.save(self.projection.state_dict(), output_path / "projection.pt")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
    ) -> "ManifoldSentenceTransformer":
        """Load a model previously saved with :meth:`save_pretrained`."""
        model_path = Path(model_path)
        config = json.loads(
            (model_path / "neembed_config.json").read_text(encoding="utf-8")
        )
        model = cls(
            str(model_path / "encoder"),
            manifold=config["manifold"],
            embedding_dim=config["embedding_dim"],
            curvature=config["curvature"],
        )
        projection_state = torch.load(
            model_path / "projection.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.projection.load_state_dict(projection_state)
        return model
