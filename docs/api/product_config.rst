Product configuration
=====================

v0.10 supports flat mixed-curvature sentence embeddings through the existing
:class:`neembed.ManifoldSentenceTransformer` encoder/projection architecture.
A normalized ``product_config`` defines how projected encoder features are split,
mapped into component manifolds, and packed into Geoopt ``ProductManifold``
points.

A product is an ordered flat list of vector-valued components. Each component
has a stable name, manifold type, intrinsic projection width, geometry-specific
curvature metadata, and an optional positive distance scale. The normalized
configuration exposes both the total encoder ``projection_dim`` and packed
``ambient_dim``. Lorentz is the only current component whose ambient width is
larger than its intrinsic width because it adds one time-like coordinate.

For example, ``H^8 x S^8 x R^16`` can be represented and executed as::

   from neembed import ManifoldSentenceTransformer, normalize_product_config

   config = normalize_product_config(
       [
           {
               "name": "hierarchy",
               "manifold": "poincare",
               "intrinsic_dim": 8,
               "curvature": 0.5,
           },
           {
               "name": "spherical",
               "manifold": "sphere_projection",
               "intrinsic_dim": 8,
               "sectional_curvature": 0.25,
           },
           {
               "name": "residual",
               "manifold": "euclidean",
               "intrinsic_dim": 16,
           },
       ]
   )

   assert config.projection_dim == 32
   assert config.ambient_dim == 32

   model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       product_config=config,
   )
   embeddings = model.encode(["dog", "mammal"], convert_to_tensor=True)
   assert embeddings.shape == (2, config.ambient_dim)

The projection layer outputs ``projection_dim`` ordinary Euclidean features.
Product execution splits that tensor in component order, maps each chunk through
its configured origin map, and packs the resulting component points with Geoopt
``ProductManifold.pack_point``. Product distance is Geoopt's product geodesic
distance.

The curvature names intentionally preserve the v0.9 contract: Poincare and
Lorentz use positive legacy ``curvature`` magnitude, while Euclidean,
SphereProjection, and Stereographic use signed ``sectional_curvature``. A product
containing Lorentz, SphereProjection, or Stereographic uses a common ``float64``
geometry dtype so all packed components share one dtype. On Apple MPS, such a
fixed-double product geometry falls back to CPU while the encoder/projection can
remain on MPS.

``scale`` defaults to ``1.0``. Non-unit scales wrap each component in Geoopt
``Scaled`` with a fixed positive distance multiplier. For component distances
``d_i`` and scales ``s_i``, the product metric is ``sqrt(sum((s_i * d_i)**2))``,
computed entirely by Geoopt. Scaling changes metric contribution, not component
dimensionality or encoded coordinates: the encoder tangent is mapped using the
underlying component manifold, then distances use the scaled product. Unit
scales retain the unwrapped component and the existing unscaled behavior.

Scales use the common geometry dtype/device and must be representable as positive
finite values in that dtype, as must their squared metric factors. Dtype changes
that violate this requirement are rejected before model state is changed.
Fixed scales are preserved in the existing version-1
configuration metadata on save/load. Learnable scales are not exposed in this
release. Models saved during Issue #130 with non-unit metadata-only scales now
apply those scales when loaded; unit-scale models retain their prior behavior.
Nested products and SPD/Stiefel/Siegel components are not supported.
``ManifoldPrototypes`` does not yet support product models and rejects them
explicitly; this runtime support covers sentence embeddings.
The radial hierarchy APIs ``ManifoldDepthLoss``, ``ManifoldRadialOrderLoss``,
``ManifoldHierarchyTripletLoss``, and ``ManifoldHierarchyEvaluator`` also reject
product models at construction because Geoopt ``ProductManifold`` has no
``dist0`` method. Product origin-distance support is not included here.

Product models also support the existing retrieval losses, ``ManifoldTrainer``,
``rank()``, ``exact_corpus_search()``, binary/graded corpus evaluation, and offline
hard-negative mining. See :doc:`../user_guide/retrieval` for a complete workflow.

``ProductConfig.to_dict()`` returns versioned JSON-compatible metadata and
``ProductConfig.from_dict()`` validates the schema, component order, dimensions,
curvature fields, scale values, and stable names when loading. Model
``save_pretrained()`` / ``from_pretrained()`` persist and restore the normalized
product metadata together with encoder and projection state.

Component distance diagnostics
------------------------------

``model.product_distance_diagnostics(a, b)`` accepts two packed embeddings or
embedding batches and reports a simple dictionary:

- ``total_distance`` is the distance returned directly by Geoopt
  ``ProductManifold.dist``.
- ``component_distances`` maps each configured component name to its geodesic
  distance, preserving configuration order. Distances include any fixed
  ``Scaled`` multiplier.

Inputs use the same geometry dtype/device conversion as ``model.distance()``.
Their final width must equal ``product_config.ambient_dim``, including Lorentz's
additional ambient coordinate. Leading dimensions broadcast as for ``distance()``:
aligned batches produce one distance per pair, while singleton batch dimensions
produce a query-by-candidate matrix.

For text batches, encode first using the existing inference API:

.. code-block:: python

   queries = model.encode(["dog", "cat"])
   candidates = model.encode(["mammal", "vehicle"])
   report = model.product_distance_diagnostics(
       queries[:, None, :], candidates[None, :, :],
   )
   assert report["total_distance"].shape == (2, 2)
   assert list(report["component_distances"]) == [
       component.name for component in model.product_config.components
   ]
   for name, distances in report["component_distances"].items():
       print(name, distances)

For the earlier H x S x E configuration, names are ``hierarchy``, ``spherical``,
and ``residual``. Tensor outputs stay on the geometry device; use
``tensor.detach().cpu().tolist()`` when a plain Python display is needed.

The diagnostic always runs without gradients and does not change model mode,
parameters, or loss behavior. Calling ``encode()`` still has its existing effect
of switching the model to evaluation mode. For training, use the model's normal
forward path and retrieval losses. Single-manifold models reject this
product-specific helper.

The total is not recomputed from the component results: Geoopt's own squared
distance aggregation and numerical safeguards remain authoritative, including
at coincident points. These distances describe geometry, not attribution,
learned component importance, or an automatic scale-selection rule.

.. autoclass:: neembed.ProductComponentConfig
   :members:

.. autoclass:: neembed.ProductConfig
   :members:

.. autofunction:: neembed.normalize_product_config
