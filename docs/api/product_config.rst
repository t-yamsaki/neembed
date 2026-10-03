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

``scale`` defaults to ``1.0`` and is persisted, but v0.10 Issue #130 does not yet
apply component scales to distances; scaled product distance is follow-on work.
Nested products and SPD/Stiefel/Siegel components are not supported.

``ProductConfig.to_dict()`` returns versioned JSON-compatible metadata and
``ProductConfig.from_dict()`` validates the schema, component order, dimensions,
curvature fields, scale values, and stable names when loading. Model
``save_pretrained()`` / ``from_pretrained()`` persist and restore the normalized
product metadata together with encoder and projection state.

.. autoclass:: neembed.ProductComponentConfig
   :members:

.. autoclass:: neembed.ProductConfig
   :members:

.. autofunction:: neembed.normalize_product_config