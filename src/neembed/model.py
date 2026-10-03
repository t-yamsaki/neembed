"""Sentence Transformer integration for manifold-valued embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer
from torch import nn

from neembed.manifolds import get_manifold
from neembed.product_config import (
    ProductComponentConfig,
    ProductConfig,
    normalize_product_config,
)
from neembed.product_runtime import (
    build_product_manifold,
    map_product_tangent,
    product_geometry_device,
    product_geometry_dtype,
    product_requires_double,
)


_STEREOGRAPHIC_DOUBLE_MANIFOLDS = {"sphere_projection", "stereographic"}
_DOUBLE_GEOMETRY_MANIFOLDS = {
    "lorentz",
    "sphere_projection",
    "stereographic",
}


def _select_geometry_device(
    manifold_name: str,
    encoder_device: torch.device | str,
) -> torch.device:
    """Choose a device that can represent the configured geometry dtype.

    Apple MPS does not support float64 tensors. SphereProjection and generic
    Stereographic therefore keep the encoder/projection on MPS but run the
    float64 manifold path on CPU. Other manifolds preserve their existing
    device behavior.
    """
    device = torch.device(encoder_device)
    if manifold_name in _STEREOGRAPHIC_DOUBLE_MANIFOLDS and device.type == "mps":
        return torch.device("cpu")
    return device


def _apply_closure_values(fn) -> dict[str, Any]:
    """Return closure values for a standard PyTorch ``Module._apply`` transform."""
    code = getattr(fn, "__code__", None)
    closure = getattr(fn, "__closure__", None)
    if code is None or closure is None:
        return {}

    values: dict[str, Any] = {}
    for name, cell in zip(code.co_freevars, closure):
        try:
            values[name] = cell.cell_contents
        except ValueError:
            continue
    return values


def _apply_requests_empty(fn) -> bool:
    """Return whether ``fn`` represents PyTorch's empty-allocation transform."""
    qualname = getattr(fn, "__qualname__", "")
    code = getattr(fn, "__code__", None)
    names = () if code is None else code.co_names
    return "Module.to_empty" in qualname or "empty_like" in names


def _apply_requested_dtype(fn) -> torch.dtype | None:
    """Return an explicitly requested floating dtype, when one is detectable."""
    values = _apply_closure_values(fn)
    dtype = values.get("dtype")
    if isinstance(dtype, torch.dtype):
        return dtype

    qualname = getattr(fn, "__qualname__", "")
    dtype_methods = {
        "Module.float": torch.float32,
        "Module.double": torch.float64,
        "Module.half": torch.float16,
        "Module.bfloat16": torch.bfloat16,
    }
    for marker, requested_dtype in dtype_methods.items():
        if marker in qualname:
            return requested_dtype

    dst_type = values.get("dst_type")
    if dst_type is not None:
        try:
            return torch.empty(0, dtype=torch.float64).type(dst_type).dtype
        except (RuntimeError, TypeError):
            return None
    return None


def _fixed_double_apply_fn(
    manifold_name: str,
    fn,
    target_device: torch.device | str,
    *,
    geometry_device: torch.device | str | None = None,
):
    """Translate a module transform while preserving fixed float64 geometry.

    Device movement and empty-allocation semantics are preserved, but requests
    to narrow floating tensors are deliberately ignored for fixed-double
    geometry state. Product geometry can supply an explicit common geometry
    device because every component must be packed on the same device.
    """
    target_device = torch.device(target_device)
    if geometry_device is None:
        geometry_device = _select_geometry_device(manifold_name, target_device)
    else:
        geometry_device = torch.device(geometry_device)

    def geometry_dtype(tensor: torch.Tensor) -> torch.dtype:
        return torch.float64 if tensor.is_floating_point() else tensor.dtype

    if _apply_requests_empty(fn):
        return lambda tensor: torch.empty_like(
            tensor,
            device=geometry_device,
            dtype=geometry_dtype(tensor),
        )

    requested_dtype = _apply_requested_dtype(fn)
    if target_device.type == "mps" or (
        requested_dtype is not None and requested_dtype != torch.float64
    ):
        def preserve_double(tensor: torch.Tensor) -> torch.Tensor:
            dtype = geometry_dtype(tensor)
            if tensor.device == geometry_device and tensor.dtype == dtype:
                return tensor
            return tensor.to(device=geometry_device, dtype=dtype)

        return preserve_double

    return fn


class ManifoldSentenceTransformer(nn.Module):
    """Map pretrained sentence embeddings onto a configured manifold.

    Args:
        model_name_or_path: Sentence Transformer model name or local model path.
        manifold: Single-manifold backend name. Supports ``"poincare"``,
            ``"lorentz"``, ``"euclidean"``, ``"sphere_projection"``, and
            ``"stereographic"``. ``"product"`` is accepted only together with
            ``product_config``; otherwise the legacy default remains Poincare.
        embedding_dim: Optional intrinsic output dimension for a learned linear
            projection. In product mode the normalized product ``projection_dim``
            owns this width; an explicitly supplied value must match it.
        curvature: Positive, finite magnitude of the negative sectional curvature.
            The same public meaning is used for Poincare and Lorentz geometry.
            This legacy argument is not reinterpreted for new v0.9 geometry.
        learnable_curvature: When ``True``, optimize the positive scalar curvature
            state jointly with the ordinary model parameters. Fixed curvature
            remains the default. Learnable curvature is limited to single
            Poincare/Lorentz geometry and is not supported by product mode.
        sectional_curvature: Signed sectional curvature for new v0.9 constant-
            curvature geometry. Euclidean accepts ``None`` or exactly ``0.0``;
            SphereProjection requires a finite positive value; Stereographic
            requires an explicit finite value of any sign.
        product_config: Ordered v0.10 mixed-curvature product configuration. The
            component dimensions define how projected encoder features are split,
            mapped, and packed into a Geoopt ``ProductManifold`` point.

    Notes:
        The returned sentence embeddings are geometry-valued outputs, while the
        encoder and projection weights remain ordinary Euclidean parameters.
        Product components share one packed tensor, so a product containing any
        Lorentz, SphereProjection, or Stereographic component uses ``float64`` for
        every component. On Apple MPS, such a fixed-double product geometry runs
        on CPU while the encoder/projection remain on MPS. Component ``scale``
        values are persisted by the v0.10 contract but are not applied until the
        scaled-distance follow-on work.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        manifold: str = "poincare",
        embedding_dim: int | None = None,
        curvature: float = 1.0,
        learnable_curvature: bool = False,
        sectional_curvature: float | None = None,
        product_config: ProductConfig
        | Sequence[ProductComponentConfig | Mapping[str, Any]]
        | Mapping[str, Any]
        | None = None,
    ) -> None:
        super().__init__()
        self.encoder = SentenceTransformer(model_name_or_path)

        encoder_dim = self.encoder.get_embedding_dimension()
        if encoder_dim is None:
            raise ValueError("Sentence Transformer embedding dimension is unknown")

        self.product_config = (
            normalize_product_config(product_config)
            if product_config is not None
            else None
        )
        if self.product_config is not None:
            if manifold not in {"poincare", "product"}:
                raise ValueError(
                    "manifold must remain at its default or be 'product' when "
                    "product_config is provided"
                )
            if embedding_dim is not None and embedding_dim != self.product_config.projection_dim:
                raise ValueError(
                    "embedding_dim must match product_config.projection_dim"
                )
            if curvature != 1.0:
                raise ValueError(
                    "top-level curvature is not used in product mode; configure "
                    "curvature per component"
                )
            if learnable_curvature:
                raise ValueError(
                    "learnable_curvature is not supported in product mode"
                )
            if sectional_curvature is not None:
                raise ValueError(
                    "top-level sectional_curvature is not used in product mode; "
                    "configure sectional_curvature per component"
                )
        elif manifold == "product":
            raise ValueError("manifold='product' requires product_config")

        self.projection: nn.Module
        if self.product_config is not None:
            projection_dim = self.product_config.projection_dim
            self._projection_dim = projection_dim
            self.projection = nn.Linear(encoder_dim, projection_dim)
            self.embedding_dim = projection_dim
        else:
            self._projection_dim = embedding_dim
            if embedding_dim is None:
                self.projection = nn.Identity()
                self.embedding_dim = encoder_dim
            else:
                self.projection = nn.Linear(encoder_dim, embedding_dim)
                self.embedding_dim = embedding_dim
        self.projection.to(self.encoder.device)

        if self.product_config is not None:
            self.manifold_name = "product"
            self.learnable_curvature = False
            projection_dtype = self.projection.weight.dtype
            geometry_device = product_geometry_device(
                self.product_config,
                self.encoder.device,
            )
            geometry_dtype = product_geometry_dtype(
                self.product_config,
                projection_dtype,
            )
            self.manifold = build_product_manifold(
                self.product_config,
                device=geometry_device,
                dtype=geometry_dtype,
            )
            return

        self.manifold_name = manifold
        self.learnable_curvature = bool(learnable_curvature)
        if self.manifold_name in {"poincare", "lorentz"}:
            if sectional_curvature is not None:
                raise ValueError(
                    "sectional_curvature must be None for poincare and lorentz"
                )
            self.manifold = get_manifold(
                self.manifold_name,
                float(curvature),
                self.learnable_curvature,
            )
        else:
            self.manifold = get_manifold(
                self.manifold_name,
                float(curvature),
                self.learnable_curvature,
                sectional_curvature=sectional_curvature,
            )
        self.manifold.to(
            _select_geometry_device(self.manifold_name, self.encoder.device)
        )

    def _apply(self, fn, recurse: bool = True):
        """Apply module transforms while preserving fixed-double geometry state."""
        manifold = self._modules.get("manifold")
        product_double = (
            self.product_config is not None
            and product_requires_double(self.product_config)
        )
        protect_manifold = manifold is not None and (
            self.manifold_name in _STEREOGRAPHIC_DOUBLE_MANIFOLDS
            or product_double
        )
        if not protect_manifold or not recurse:
            return super()._apply(fn, recurse=recurse)

        self._modules["manifold"] = None
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._modules["manifold"] = manifold

        explicit_geometry_device = None
        if self.product_config is not None:
            explicit_geometry_device = product_geometry_device(
                self.product_config,
                self.encoder.device,
            )
        geometry_fn = _fixed_double_apply_fn(
            self.manifold_name,
            fn,
            self.encoder.device,
            geometry_device=explicit_geometry_device,
        )
        manifold._apply(geometry_fn, recurse=True)

        if self.product_config is not None:
            for component_manifold in manifold.manifolds:
                if (
                    component_manifold.dtype is not None
                    and component_manifold.dtype != torch.float64
                ):
                    raise RuntimeError(
                        "fixed-double product manifold state must remain float64"
                    )
        elif manifold.k.dtype != torch.float64:
            raise RuntimeError("fixed stereographic curvature must remain float64")
        return result

    @property
    def curvature(self) -> float:
        """Return the current legacy public curvature magnitude as a Python float."""
        if self.manifold_name == "poincare":
            curvature = self.manifold.c
        elif self.manifold_name == "lorentz":
            curvature = self.manifold.k.reciprocal()
        elif self.manifold_name == "product":
            raise AttributeError(
                "product curvature is component-specific; inspect product_config"
            )
        else:
            raise AttributeError(
                "curvature is only defined for poincare and lorentz; "
                "use sectional_curvature for v0.9 geometry"
            )
        return float(curvature.detach().cpu())

    @property
    def sectional_curvature(self) -> float:
        """Return signed sectional curvature for supported v0.9 geometry."""
        if self.manifold_name == "euclidean":
            return 0.0
        if self.manifold_name in {"sphere_projection", "stereographic"}:
            return float(self.manifold.k.detach().cpu())
        if self.manifold_name == "product":
            raise AttributeError(
                "product sectional curvature is component-specific; inspect product_config"
            )
        raise AttributeError(
            "sectional_curvature is defined only for v0.9 geometry; "
            "poincare and lorentz keep the legacy curvature magnitude API"
        )

    def forward(self, sentences: Sequence[str]) -> torch.Tensor:
        """Encode a batch and map embeddings into the configured geometry."""
        features = self.encoder.preprocess(list(sentences))
        features = {
            key: value.to(self.encoder.device) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        encoder_output: dict[str, Any] = self.encoder(features)
        tangent = self.projection(encoder_output["sentence_embedding"])

        if self.product_config is not None:
            geometry_device = product_geometry_device(
                self.product_config,
                self.encoder.device,
            )
            geometry_dtype = product_geometry_dtype(
                self.product_config,
                tangent.dtype,
            )
            return map_product_tangent(
                self.manifold,
                self.product_config,
                tangent,
                device=geometry_device,
                dtype=geometry_dtype,
            )

        if self.manifold_name == "euclidean":
            return tangent
        if self.manifold_name in _DOUBLE_GEOMETRY_MANIFOLDS:
            tangent = tangent.to(
                device=_select_geometry_device(
                    self.manifold_name,
                    self.encoder.device,
                ),
                dtype=torch.float64,
            )
        if self.manifold_name == "lorentz":
            tangent = torch.cat((torch.zeros_like(tangent[..., :1]), tangent), dim=-1)
        return self.manifold.expmap0(tangent)

    def encode(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
    ) -> Any:
        """Encode text as geometry-valued embeddings for inference.

        Product output uses ``product_config.ambient_dim`` as its packed final
        width. NumPy arrays are returned by default; tensors are returned when
        ``convert_to_tensor=True``.
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
        """Return the geodesic distance between two geometry embeddings."""
        reference = next(self.parameters())
        if self.product_config is not None:
            geometry_dtype = product_geometry_dtype(
                self.product_config,
                reference.dtype,
            )
            geometry_device = product_geometry_device(
                self.product_config,
                self.encoder.device,
            )
        else:
            use_double_geometry = self.manifold_name in _DOUBLE_GEOMETRY_MANIFOLDS
            geometry_dtype = torch.float64 if use_double_geometry else reference.dtype
            geometry_device = _select_geometry_device(
                self.manifold_name,
                self.encoder.device,
            )

        a_tensor = torch.as_tensor(
            a,
            device=geometry_device,
            dtype=geometry_dtype,
        )
        b_tensor = torch.as_tensor(
            b,
            device=geometry_device,
            dtype=geometry_dtype,
        )

        with torch.no_grad():
            return self.manifold.dist(a_tensor, b_tensor)

    def rank(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, str | int | float]]:
        """Rank an in-memory candidate list by geodesic distance to a query."""
        if isinstance(candidates, str):
            raise ValueError(
                "candidates must be a sequence of strings, not a single string"
            )
        candidate_list = list(candidates)
        if not candidate_list:
            raise ValueError("candidates must contain at least one item")
        if top_k is None:
            result_count = len(candidate_list)
        else:
            if (
                isinstance(top_k, bool)
                or not isinstance(top_k, int)
                or not 1 <= top_k <= len(candidate_list)
            ):
                raise ValueError(
                    "top_k must be an integer between 1 and the candidate count"
                )
            result_count = top_k

        with torch.no_grad():
            query_embedding = self.encode(query, convert_to_tensor=True)
            candidate_embeddings = self.encode(
                candidate_list,
                convert_to_tensor=True,
            )
            distances = self.distance(
                query_embedding.unsqueeze(0),
                candidate_embeddings,
            )

        distance_values = distances.detach().cpu().tolist()
        ranked_indices = sorted(
            range(len(candidate_list)),
            key=distance_values.__getitem__,
        )[:result_count]
        return [
            {
                "candidate": candidate_list[index],
                "index": index,
                "distance": float(distance_values[index]),
            }
            for index in ranked_indices
        ]

    def save_pretrained(self, output_path: str | Path) -> None:
        """Save the encoder, projection, and sentence-model geometry state."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(str(output_path / "encoder"))
        config: dict[str, Any] = {
            "embedding_dim": self._projection_dim,
            "manifold": self.manifold_name,
        }
        if self.product_config is not None:
            config["product_config"] = self.product_config.to_dict()
        elif self.manifold_name in {"poincare", "lorentz"}:
            config["curvature"] = self.curvature
            if self.learnable_curvature:
                config["learnable_curvature"] = True
        else:
            config["sectional_curvature"] = self.sectional_curvature
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
        manifold_name = config["manifold"]

        if manifold_name == "product":
            product_config = normalize_product_config(config["product_config"])
            saved_projection_dim = config["embedding_dim"]
            if saved_projection_dim != product_config.projection_dim:
                raise ValueError(
                    "saved embedding_dim does not match product_config.projection_dim"
                )
            kwargs: dict[str, Any] = {
                "manifold": "product",
                "embedding_dim": saved_projection_dim,
                "product_config": product_config,
            }
        else:
            kwargs = {
                "manifold": manifold_name,
                "embedding_dim": config["embedding_dim"],
            }
            if manifold_name in {"poincare", "lorentz"}:
                kwargs["curvature"] = config["curvature"]
                kwargs["learnable_curvature"] = config.get(
                    "learnable_curvature",
                    False,
                )
            else:
                kwargs["sectional_curvature"] = config["sectional_curvature"]

        model = cls(str(model_path / "encoder"), **kwargs)
        projection_state = torch.load(
            model_path / "projection.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.projection.load_state_dict(projection_state)
        return model
