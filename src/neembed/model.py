"""Sentence Transformer integration for manifold-valued embeddings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any, Literal

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
    validate_product_scale,
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
        manifold: Manifold backend name. Supports ``"poincare"``, ``"lorentz"``,
            ``"euclidean"``, ``"sphere_projection"``, and ``"stereographic"``.
            ``"product"`` is accepted with ``product_config``.
        embedding_dim: Optional intrinsic output dimension for a learned linear
            projection. If omitted, the encoder embedding dimension is preserved.
            Lorentz embeddings use one additional ambient coordinate. In product
            mode an explicit value must match ``product_config.projection_dim``.
        curvature: Positive, finite magnitude of the negative sectional curvature.
            The same public meaning is used for Poincare and Lorentz geometry.
            This legacy argument is not reinterpreted for new v0.9 geometry.
        learnable_curvature: When ``True``, optimize the positive scalar curvature
            state jointly with the ordinary model parameters. Fixed curvature
            remains the default. Learnable curvature is limited to Poincare and
            Lorentz and is not itself a manifold-valued point.
        sectional_curvature: Signed sectional curvature for new v0.9 constant-
            curvature geometry. Euclidean accepts ``None`` or exactly ``0.0``;
            SphereProjection requires a finite positive value; Stereographic
            requires an explicit finite value of any sign.
        product_config: Ordered flat product configuration. Component dimensions
            determine the projection width and split/map/pack layout. Curvature
            is fixed and configured per component; top-level curvature arguments
            must retain their defaults.
        revision: Encoder repository branch, tag, or commit ID. Use a commit ID
            to pin a remote encoder. This does not select a neembed checkpoint.
        local_files_only: Load the encoder only from local files or the existing
            cache when ``True``. Defaults to Sentence Transformers' normal
            loading behavior without preventing downloads.
        cache_folder: Optional Sentence Transformers download/cache directory.
        device: Optional encoder device, such as ``"cpu"`` or ``"cuda:0"``.
            Projection and geometry follow the existing device policies.

    Notes:
        The returned sentence embeddings are geometry-valued outputs, while the
        encoder and optional projection weights remain ordinary Euclidean
        parameters. True manifold-valued trainable coordinates are introduced
        separately through :class:`neembed.ManifoldPrototypes`. SphereProjection
        and Stereographic geometry operations use ``float64`` even when the
        encoder and projection remain in their ordinary model dtype. On Apple MPS,
        those float64 geometry operations fall back to CPU while the encoder and
        projection remain on MPS. A product containing Lorentz, SphereProjection,
        or Stereographic uses a common float64 geometry dtype, with CPU fallback
        on MPS. Component scales are fixed distance multipliers applied through Geoopt.
        Encoder loading options apply only to the current load and are not
        written to neembed metadata. Unspecified options are left to Sentence
        Transformers; remote-code execution is not enabled by neembed.
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
        revision: str | None = None,
        local_files_only: bool = False,
        cache_folder: str | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__()
        encoder_kwargs: dict[str, Any] = {}
        if revision is not None:
            encoder_kwargs["revision"] = revision
        if local_files_only:
            encoder_kwargs["local_files_only"] = local_files_only
        if cache_folder is not None:
            encoder_kwargs["cache_folder"] = cache_folder
        if device is not None:
            encoder_kwargs["device"] = device
        self.encoder = SentenceTransformer(model_name_or_path, **encoder_kwargs)

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

        if self.product_config is not None:
            embedding_dim = self.product_config.projection_dim
        self._projection_dim = embedding_dim
        self.projection: nn.Module
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
        product_config = getattr(self, "product_config", None)
        product_double = (
            product_config is not None
            and product_requires_double(product_config)
        )
        protect_manifold = manifold is not None and (
            self.manifold_name in _STEREOGRAPHIC_DOUBLE_MANIFOLDS
            or product_double
        )
        # Validate before transforming any state, so a rejected narrowing cast
        # leaves the entire model unchanged. Fixed-double products do not narrow.
        if manifold is not None and product_config is not None and not product_double and recurse:
            requested_dtype = _apply_requested_dtype(fn)
            if requested_dtype is not None:
                for component, geometry in zip(product_config.components, manifold.manifolds):
                    if component.scale != 1.0:
                        scale = geometry.scale
                        if scale.is_meta:
                            scale = torch.tensor(component.scale, dtype=requested_dtype)
                        else:
                            scale = scale.detach().to(device="cpu", dtype=requested_dtype)
                        validate_product_scale(scale)
        if not protect_manifold or not recurse:
            return super()._apply(fn, recurse=recurse)

        # Ordinary model state follows the requested transform unchanged. The
        # protected geometry receives an equivalent transform that preserves its
        # fixed float64 policy and translates unsupported MPS storage to CPU.
        self._modules["manifold"] = None
        try:
            result = super()._apply(fn, recurse=recurse)
        finally:
            self._modules["manifold"] = manifold

        explicit_geometry_device = None
        if product_config is not None:
            explicit_geometry_device = product_geometry_device(
                product_config,
                self.encoder.device,
            )
        geometry_fn = _fixed_double_apply_fn(
            self.manifold_name,
            fn,
            self.encoder.device,
            geometry_device=explicit_geometry_device,
        )
        manifold._apply(geometry_fn, recurse=True)

        if product_config is not None:
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
        """Return the current legacy public curvature magnitude as a Python float.

        The value is the positive magnitude of negative sectional curvature for
        Poincare and Lorentz. New v0.9 geometry uses the distinct signed
        ``sectional_curvature`` property instead.
        """
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

    def _resolve_input_prompt(
        self,
        task: Literal["query", "document"] | None,
        prompt_name: str | None,
        prompt: str | None,
    ) -> str | None:
        """Choose a prompt without changing the legacy role-free path."""
        if task is not None and task not in ("query", "document"):
            raise ValueError("task must be 'query', 'document', or None")
        if prompt is not None and not isinstance(prompt, str):
            raise TypeError("prompt must be a string or None")
        if prompt_name is not None and not isinstance(prompt_name, str):
            raise TypeError("prompt_name must be a string or None")
        if prompt is not None and prompt_name is not None:
            raise ValueError("provide either prompt or prompt_name, not both")
        if prompt is not None:
            return prompt
        # Historical forward/encode never applied encoder.default_prompt_name.
        if task is None and prompt_name is None:
            return None

        prompts = getattr(self.encoder, "prompts", {})
        if not isinstance(prompts, Mapping):
            raise ValueError("encoder.prompts must be a mapping")
        if prompt_name is None:
            candidates = ("query",) if task == "query" else (
                "document", "passage", "corpus",
            )
            prompt_name = next((name for name in candidates if name in prompts), None)
            if prompt_name is None:
                prompt_name = getattr(self.encoder, "default_prompt_name", None)
        if prompt_name is None:
            return None
        if prompt_name not in prompts:
            raise ValueError(f"unknown encoder prompt name: {prompt_name!r}")
        resolved = prompts[prompt_name]
        if not isinstance(resolved, str):
            raise TypeError("selected encoder prompt must be a string")
        return resolved

    def forward(
        self,
        sentences: Sequence[str],
        *,
        task: Literal["query", "document"] | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> torch.Tensor:
        """Encode a batch and map embeddings into the configured geometry.

        Args:
            sentences: Batch of input texts.
            task: Optional query/document role, forwarded to encoder preprocessing
                and forward for task-aware routing. Unspecified keeps legacy behavior.
            prompt_name: Name from ``encoder.prompts``. Unknown names are rejected.
            prompt: Explicit prefix, including ``""`` to disable prompt selection.
                Cannot be combined with ``prompt_name``. Otherwise the role prompt
                wins over the encoder default; with no role/name/prompt, no prompt
                is selected. See the inference guide for document fallback names.

        Returns:
            Geometry-valued embeddings. Poincare, Euclidean, SphereProjection,
            and Stereographic output have shape ``(batch_size, embedding_dim)``.
            Lorentz output has shape ``(batch_size, embedding_dim + 1)`` because
            the hyperboloid uses one additional ambient time-like coordinate.
            Euclidean output is the encoder/projection output directly;
            Poincare, SphereProjection, and Stereographic map the projected tangent
            vector through the origin exponential map. Lorentz, SphereProjection,
            and Stereographic geometry operations use double precision for
            numerical stability. SphereProjection/Stereographic use CPU for that
            double-precision geometry path when the encoder runs on Apple MPS.

        Product output has final width ``product_config.ambient_dim``.
        """
        resolved_prompt = self._resolve_input_prompt(task, prompt_name, prompt)
        preprocess_kwargs: dict[str, Any] = {}
        encoder_kwargs: dict[str, Any] = {}
        if resolved_prompt is not None:
            preprocess_kwargs["prompt"] = resolved_prompt
        if task is not None:
            preprocess_kwargs["task"] = task
            encoder_kwargs["task"] = task
        # Delegate prompt insertion and prompt-length metadata to the encoder.
        # Do not call encoder.encode(), which would disable training gradients.
        features = self.encoder.preprocess(list(sentences), **preprocess_kwargs)
        features = {
            key: value.to(self.encoder.device) if torch.is_tensor(value) else value
            for key, value in features.items()
        }
        encoder_output: dict[str, Any] = self.encoder(features, **encoder_kwargs)
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
        task: Literal["query", "document"] | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Encode text as geometry-valued embeddings for inference.

        Args:
            sentences: A single text or a sequence of texts.
            convert_to_tensor: Return a ``torch.Tensor`` instead of a NumPy array.
            task: Optional query/document role, with the same rules as ``forward``.
            prompt_name: Saved encoder prompt name, exclusive with ``prompt``.
            prompt: Explicit prefix; ``""`` disables automatic prompt selection.

        Returns:
            A single geometry embedding for string input or a batch for sequence
            input. The last dimension is ``embedding_dim`` for Poincare,
            Euclidean, SphereProjection, and Stereographic and
            ``embedding_dim + 1`` for Lorentz. NumPy arrays are returned by
            default; tensors are returned when ``convert_to_tensor=True``.
            Lorentz, SphereProjection, and Stereographic outputs use ``float64``
            for the manifold geometry path. SphereProjection/Stereographic tensor
            outputs are CPU tensors when the encoder uses Apple MPS.

        Notes:
            Encoding switches the model to evaluation mode and runs under
            ``torch.inference_mode()``, so returned embeddings do not track
            gradients.

        Product output has final width ``product_config.ambient_dim``.
        """
        single_input = isinstance(sentences, str)
        batch = [sentences] if single_input else list(sentences)

        input_kwargs: dict[str, Any] = {}
        if task is not None:
            input_kwargs["task"] = task
        if prompt_name is not None:
            input_kwargs["prompt_name"] = prompt_name
        if prompt is not None:
            input_kwargs["prompt"] = prompt

        self.eval()
        with torch.inference_mode():
            embeddings = self(batch, **input_kwargs)

        if single_input:
            embeddings = embeddings[0]
        if convert_to_tensor:
            return embeddings
        return embeddings.cpu().numpy()

    def encode_query(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Encode queries with ``task='query'`` and shared prompt selection.

        Uses the saved ``query`` prompt, then the saved default if no explicit
        prompt/name is supplied. Return shape and inference mode match ``encode``.
        """
        return self.encode(
            sentences, convert_to_tensor=convert_to_tensor, task="query",
            prompt_name=prompt_name, prompt=prompt,
        )

    def encode_document(
        self,
        sentences: str | Sequence[str],
        *,
        convert_to_tensor: bool = False,
        prompt_name: str | None = None,
        prompt: str | None = None,
    ) -> Any:
        """Encode documents with ``task='document'`` and shared prompt selection.

        Tries saved ``document``, ``passage``, ``corpus`` prompts in that order,
        then the saved default if no explicit prompt/name is supplied. Return
        shape and inference mode match ``encode``.
        """
        return self.encode(
            sentences, convert_to_tensor=convert_to_tensor, task="document",
            prompt_name=prompt_name, prompt=prompt,
        )

    def _distance_tensors(self, a: Any, b: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert distance inputs using the configured geometry policy."""
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

        return a_tensor, b_tensor

    def distance(self, a: Any, b: Any) -> torch.Tensor:
        """Return the geodesic distance between two geometry embeddings.

        Args:
            a: First geometry embedding or array-like value.
            b: Second geometry embedding or array-like value.

        Returns:
            A tensor containing the configured geometry distance.

        Notes:
            This is an inference helper. Inputs are moved to the geometry device
            and dtype, and the distance is computed under ``torch.no_grad()``.
            Lorentz, SphereProjection, and Stereographic distance are evaluated in
            ``float64``. Poincare and Euclidean retain the model parameter dtype.
            SphereProjection/Stereographic distance uses CPU when the encoder is
            on Apple MPS because MPS does not support ``float64`` tensors.

        Product distance uses the common geometry dtype/device and configured scales.
        """
        a_tensor, b_tensor = self._distance_tensors(a, b)

        with torch.no_grad():
            return self.manifold.dist(a_tensor, b_tensor)

    @torch.no_grad()
    def product_distance_diagnostics(
        self, a: Any, b: Any,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """Report total and named component geodesic distances for a product.

        Args:
            a: Packed product embedding or array-like batch. The final dimension
                must equal ``product_config.ambient_dim``.
            b: Second packed product embedding or batch, with broadcast-compatible
                leading dimensions. Encode text with ``encode()`` before calling.

        Returns:
            A dictionary with ``total_distance`` from Geoopt ProductManifold
            and ``component_distances``, an ordered name-to-tensor dictionary
            matching product configuration order. Component distances include
            fixed Scaled multipliers. Each tensor has the broadcast batch shape.

        Raises:
            ValueError: If the model is not a product or an input has the wrong
                packed embedding width.

        Notes:
            All computation is no-grad and uses the same geometry dtype/device
            as ``distance()``. This helper does not change model mode or training
            losses. The total is obtained directly from Geoopt, preserving its
            numerical safeguards rather than recomputing it from diagnostics.
        """
        if self.product_config is None:
            raise ValueError("product_distance_diagnostics requires a product model")
        a_tensor, b_tensor = self._distance_tensors(a, b)
        for name, tensor in (("a", a_tensor), ("b", b_tensor)):
            if tensor.ndim == 0 or tensor.shape[-1] != self.product_config.ambient_dim:
                raise ValueError(
                    f"{name} must have final dimension {self.product_config.ambient_dim}"
                )
        component_distances = {}
        for index, (component, manifold) in enumerate(zip(
            self.product_config.components, self.manifold.manifolds,
        )):
            component_distances[component.name] = manifold.dist(
                self.manifold.take_submanifold_value(a_tensor, index),
                self.manifold.take_submanifold_value(b_tensor, index),
            )
        return {
            "total_distance": self.manifold.dist(a_tensor, b_tensor),
            "component_distances": component_distances,
        }

    def rank(
        self,
        query: str,
        candidates: Sequence[str],
        *,
        top_k: int | None = None,
    ) -> list[dict[str, str | int | float]]:
        """Rank an in-memory candidate list by geodesic distance to a query.

        Args:
            query: Query text to encode.
            candidates: Non-empty sequence of candidate texts to rerank.
            top_k: Number of ranked candidates to return. ``None`` returns the
                full list. Integer values must be between 1 and the candidate
                count, inclusive.

        Returns:
            Plain Python dictionaries ordered by ascending geodesic distance.
            Each result contains the original ``candidate``, its input ``index``,
            and the scalar ``distance``. Equal-distance candidates retain their
            original input order.

        Raises:
            ValueError: If ``candidates`` is a bare string or empty, or ``top_k``
                is invalid.

        Notes:
            This helper is intended for small in-memory reranking. It does not
            build or persist a search index. Encoding and distance calculation run
            without gradient tracking through the existing inference helpers.
        """
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
        """Save the encoder, projection, and sentence-model geometry state.

        Args:
            output_path: Directory in which to save the model.

        Notes:
            Poincare/Lorentz keep the legacy public ``curvature`` metadata. New
            v0.9 geometry stores the distinct signed ``sectional_curvature`` value.
            External modules such as :class:`neembed.ManifoldPrototypes`,
            hierarchy metadata, and optimizer state are not included by this
            helper and should be saved separately when needed.
        """
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
        *,
        revision: str | None = None,
        local_files_only: bool = False,
        cache_folder: str | None = None,
        device: str | None = None,
    ) -> "ManifoldSentenceTransformer":
        """Load a model previously saved with :meth:`save_pretrained`.

        Args:
            model_path: Directory containing a saved neembed model.
            revision: Forwarded to the saved encoder loader; does not select a
                version of the local neembed directory. Usually omitted here.
            local_files_only: Restrict encoder loading to local files/cache.
            cache_folder: Optional encoder cache directory for this load only.
            device: Optional encoder device; projection and geometry follow the
                existing device policies.

        Returns:
            The reconstructed geometry-aware sentence model. Poincare/Lorentz
            retain the saved legacy curvature magnitude and trainability; new
            v0.9 geometry restores signed sectional-curvature metadata. External
            prototype modules must be reconstructed and loaded separately.
        """
        model_path = Path(model_path)
        config = json.loads(
            (model_path / "neembed_config.json").read_text(encoding="utf-8")
        )
        manifold_name = config["manifold"]
        kwargs: dict[str, Any] = {
            "manifold": manifold_name,
            "embedding_dim": config["embedding_dim"],
            "revision": revision,
            "local_files_only": local_files_only,
            "cache_folder": cache_folder,
            "device": device,
        }
        if manifold_name == "product":
            product_config = ProductConfig.from_dict(config["product_config"])
            if config["embedding_dim"] != product_config.projection_dim:
                raise ValueError(
                    "saved embedding_dim does not match product_config.projection_dim"
                )
            kwargs["product_config"] = product_config
        elif manifold_name in {"poincare", "lorentz"}:
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
