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
    """Map pretrained sentence embeddings onto a configured manifold.

    Args:
        model_name_or_path: Sentence Transformer model name or local model path.
        manifold: Manifold backend name. Supports ``"poincare"`` and ``"lorentz"``.
        embedding_dim: Optional intrinsic output dimension for a learned linear
            projection. If omitted, the encoder embedding dimension is preserved.
            Lorentz embeddings use one additional ambient coordinate.
        curvature: Positive, finite magnitude of the negative sectional curvature.
        learnable_curvature: When ``True``, optimize curvature jointly with the
            ordinary model parameters. Fixed curvature remains the default.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        manifold: str = "poincare",
        embedding_dim: int | None = None,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
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
        self.learnable_curvature = bool(learnable_curvature)
        self.manifold = get_manifold(
            self.manifold_name,
            float(curvature),
            self.learnable_curvature,
        )
        self.manifold.to(self.encoder.device)

    @property
    def curvature(self) -> float:
        """Return the current public curvature magnitude as a Python float."""
        if self.manifold_name == "poincare":
            curvature = self.manifold.c
        else:
            curvature = self.manifold.k.reciprocal()
        return float(curvature.detach().cpu())

    def forward(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode a batch and map embeddings from the origin tangent space.

        Args:
            sentences: Batch of input texts.

        Returns:
            Manifold-valued embeddings. Poincaré output has shape
            ``(batch_size, embedding_dim)``. Lorentz output has shape
            ``(batch_size, embedding_dim + 1)`` because the hyperboloid uses one
            additional ambient time-like coordinate. Lorentz geometry is computed
            in double precision for numerical stability.
        """
        features = self.encoder.preprocess(list(sentences))
        features = {
            key: value.to(self.encoder.device) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        encoder_output: dict[str, Any] = self.encoder(features)
        tangent = self.projection(encoder_output["sentence_embedding"])
        if self.manifold_name == "lorentz":
            tangent = tangent.to(dtype=torch.float64)
            tangent = torch.cat((torch.zeros_like(tangent[..., :1]), tangent), dim=-1)
        return self.manifold.expmap0(tangent)

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
    ) -> Any:
        """Encode text as manifold-valued embeddings for inference.

        Args:
            sentences: A single text or a sequence of texts.
            convert_to_tensor: Return a ``torch.Tensor`` instead of a NumPy array.

        Returns:
            A single manifold embedding for string input or a batch for sequence
            input. The last dimension is ``embedding_dim`` for Poincaré and
            ``embedding_dim + 1`` for Lorentz. NumPy arrays are returned by default;
            tensors are returned when ``convert_to_tensor=True``. Lorentz outputs
            use ``float64`` for the manifold geometry path.

        Notes:
            Encoding switches the model to evaluation mode and runs under
            ``torch.inference_mode()``, so returned embeddings do not track
            gradients.
        """
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
        """Return the geodesic distance between two manifold embeddings.

        Args:
            a: First manifold embedding or array-like value.
            b: Second manifold embedding or array-like value.

        Returns:
            A tensor containing the manifold geodesic distance.

        Notes:
            This is an inference helper. Inputs are moved to the model device and
            geometry dtype, and the distance is computed under ``torch.no_grad()``.
            Lorentz distance is evaluated in ``float64``; Poincaré keeps the model
            parameter dtype.
        """
        reference = next(self.parameters())
        geometry_dtype = (
            torch.float64 if self.manifold_name == "lorentz" else reference.dtype
        )
        a_tensor = torch.as_tensor(
            a,
            device=reference.device,
            dtype=geometry_dtype,
        )
        b_tensor = torch.as_tensor(
            b,
            device=reference.device,
            dtype=geometry_dtype,
        )

        with torch.no_grad():
            return self.manifold.dist(a_tensor, b_tensor)

    def save_pretrained(self, output_path: str | Path) -> None:
        """Save the encoder, projection weights, and neembed configuration.

        Args:
            output_path: Directory in which to save the model.
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(str(output_path / "encoder"))
        config = {
            "embedding_dim": self._projection_dim,
            "manifold": self.manifold_name,
            "curvature": self.curvature,
        }
        if self.learnable_curvature:
            config["learnable_curvature"] = True
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
        """Load a model previously saved with :meth:`save_pretrained`.

        Args:
            model_path: Directory containing a saved neembed model.

        Returns:
            The reconstructed manifold sentence model.
        """
        model_path = Path(model_path)
        config = json.loads(
            (model_path / "neembed_config.json").read_text(encoding="utf-8")
        )
        model = cls(
            str(model_path / "encoder"),
            manifold=config["manifold"],
            embedding_dim=config["embedding_dim"],
            curvature=config["curvature"],
            learnable_curvature=config.get("learnable_curvature", False),
        )
        projection_state = torch.load(
            model_path / "projection.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.projection.load_state_dict(projection_state)
        return model
