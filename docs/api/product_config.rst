Product configuration
=====================

v0.10 begins mixed-curvature support with a configuration contract only. Product
embedding execution is added by follow-on work; existing single-manifold model
construction remains unchanged.

A product is an ordered flat list of vector-valued components. Each component
has a stable name, manifold type, intrinsic projection width, geometry-specific
curvature metadata, and an optional positive distance scale. The normalized
configuration exposes both the total encoder ``projection_dim`` and packed
``ambient_dim``. Lorentz is the only current component whose ambient width is
larger than its intrinsic width because it adds one time-like coordinate.

For example, ``H^8 x S^8 x R^16`` can be represented as::

   from neembed import normalize_product_config

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

The curvature names intentionally preserve the v0.9 contract: Poincare and
Lorentz use positive legacy ``curvature`` magnitude, while Euclidean,
SphereProjection, and Stereographic use signed ``sectional_curvature``.
``scale`` defaults to ``1.0`` and is persisted now so later ``Scaled`` support
can consume the same normalized contract; this issue does not yet apply scale to
distances.

``ProductConfig.to_dict()`` returns versioned JSON-compatible metadata and
``ProductConfig.from_dict()`` validates the schema, component order, dimensions,
curvature fields, scale values, and stable names when loading.

.. autoclass:: neembed.ProductComponentConfig
   :members:

.. autoclass:: neembed.ProductConfig
   :members:

.. autofunction:: neembed.normalize_product_config
