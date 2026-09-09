Constant-curvature semantics for v0.9
=====================================

This page is the implementation contract for the v0.9 constant-curvature
follow-on work. It defines names, validation, and persistence semantics only;
it does not add new manifold backends by itself.

Existing Poincare and Lorentz contract
--------------------------------------

The existing public ``curvature`` argument does **not** carry a sign. For
``manifold="poincare"`` and ``manifold="lorentz"`` it remains the positive,
finite magnitude :math:`c` of negative sectional curvature:

.. math::

   K = -c, \qquad c > 0.

This behavior is frozen for backward compatibility.

* Poincare passes ``curvature=c`` to ``geoopt.PoincareBall(c=c)``.
* Lorentz maps the same public magnitude to Geoopt's squared hyperboloid radius
  ``k = 1 / c``. The resulting sectional curvature is ``-1 / k = -c``.
* ``model.curvature`` continues to report the positive magnitude ``c``.
* ``learnable_curvature=True`` continues to mean a trainable positive magnitude
  for these existing hyperbolic backends.

Neither the constructor meaning nor saved-model meaning of ``curvature`` may be
changed to a signed value in v0.9.

Signed sectional curvature for new geometry
-------------------------------------------

New constant-curvature geometry uses the explicit public name
``sectional_curvature``. It is a finite **signed sectional curvature**
:math:`K`, following Geoopt's generic ``Stereographic(k=K)`` convention:

* ``sectional_curvature < 0`` means hyperbolic stereographic geometry.
* ``sectional_curvature == 0`` means Euclidean geometry.
* ``sectional_curvature > 0`` means spherical stereographic geometry.

The public name is deliberately not ``k``. Geoopt's Lorentz backend already
uses ``k`` for the positive squared hyperboloid radius, where ``k = 1 / c``;
reusing that name publicly would make the two meanings ambiguous.

Constructor and factory contract
--------------------------------

The existing calls remain valid without modification:

.. code-block:: python

   ManifoldSentenceTransformer(
       model_name,
       manifold="poincare",
       curvature=2.0,
   )

   ManifoldSentenceTransformer(
       model_name,
       manifold="lorentz",
       curvature=2.0,
   )

The v0.9 follow-on implementations add one keyword-only argument to the shared
model constructor and manifold factory:

.. code-block:: text

   sectional_curvature: float | None = None

The existing ``curvature=1.0`` and ``learnable_curvature=False`` defaults stay
unchanged. ``sectional_curvature`` is the only new public signed-curvature
keyword; there is no public ``k`` argument.

The same keyword name is used by both entry points:

.. code-block:: python

   ManifoldSentenceTransformer(
       model_name,
       manifold="stereographic",
       sectional_curvature=-2.0,
   )

   get_manifold(
       "stereographic",
       sectional_curvature=-2.0,
   )

The new manifold names and validation rules are fixed as follows.

``manifold="poincare"`` and ``manifold="lorentz"``
   Continue to use ``curvature``. ``sectional_curvature`` must be ``None``.
   Existing ``learnable_curvature`` behavior is unchanged.

``manifold="euclidean"``
   Sectional curvature is exactly ``0.0``. ``sectional_curvature`` may be
   omitted or explicitly set to ``0.0``. ``learnable_curvature=True`` is
   invalid. The legacy ``curvature`` value is not used as Euclidean curvature.

``manifold="sphere_projection"``
   Requires a finite positive ``sectional_curvature`` and fixed curvature.
   This corresponds to Geoopt
   ``SphereProjection(k=sectional_curvature)``.

``manifold="stereographic"``
   Requires a finite ``sectional_curvature`` and accepts negative, zero, or
   positive values with fixed curvature. This corresponds to Geoopt
   ``Stereographic(k=sectional_curvature)``.

Because the shared constructor and ``get_manifold`` retain the legacy
``curvature=1.0`` default for source compatibility, new manifold backends do not
interpret that default as their curvature. If a caller supplies a non-default
legacy ``curvature`` value together with ``euclidean``, ``sphere_projection``,
or ``stereographic``, the v0.9 implementation must raise ``ValueError`` rather
than silently ignore or reinterpret it. This makes accidental use of the old
keyword visible while preserving calls that rely on the existing default.

For v0.9, signed curvature is fixed. There is no
``learnable_sectional_curvature`` contract and no support for learning through
zero. Existing ``learnable_curvature`` remains limited to the established
Poincare/Lorentz positive-magnitude semantics.

Validation matrix
-----------------

.. list-table::
   :header-rows: 1

   * - Manifold name
     - Public curvature field
     - Valid values
     - Sectional curvature
   * - ``poincare``
     - ``curvature``
     - finite ``> 0``
     - ``K = -curvature``
   * - ``lorentz``
     - ``curvature``
     - finite ``> 0``
     - ``K = -curvature``
   * - ``euclidean``
     - ``sectional_curvature``
     - omitted or exactly ``0.0``
     - ``K = 0``
   * - ``sphere_projection``
     - ``sectional_curvature``
     - finite ``> 0``
     - ``K = sectional_curvature``
   * - ``stereographic``
     - ``sectional_curvature``
     - any finite value
     - ``K = sectional_curvature``

Persistence contract
--------------------

Existing Poincare and Lorentz files keep their current
``neembed_config.json`` schema. In particular, v0.9 must not rewrite existing
saved models to a new field:

.. code-block:: json

   {
     "embedding_dim": 2,
     "manifold": "poincare",
     "curvature": 2.0
   }

Lorentz uses the same public ``"curvature"`` magnitude field. The Geoopt
squared-radius ``k`` is reconstructed from it and is not persisted as a public
curvature value.

New v0.9 geometries use the distinct ``"sectional_curvature"`` key:

.. code-block:: json

   {
     "embedding_dim": 2,
     "manifold": "euclidean",
     "sectional_curvature": 0.0
   }

.. code-block:: json

   {
     "embedding_dim": 2,
     "manifold": "sphere_projection",
     "sectional_curvature": 2.0
   }

.. code-block:: json

   {
     "embedding_dim": 2,
     "manifold": "stereographic",
     "sectional_curvature": -2.0
   }

Loaders select the curvature field from the saved ``manifold`` name. They must
not infer signed curvature by changing the meaning of an old ``"curvature"``
field, and new geometry must not persist Geoopt's implementation-level ``k`` as
a public field. No configuration-version framework is required for this
separation because the distinct field names are sufficient.

The follow-on implementations may expose the signed value on new geometry as a
read-only ``model.sectional_curvature`` property. They must not change the
meaning of the existing ``model.curvature`` property for Poincare/Lorentz.
Persistence is defined by the keys above rather than by requiring one unified
curvature property across every geometry.

Equivalence relationships for follow-on tests
---------------------------------------------

The v0.9 geometry issues use the following relationships for parity tests:

* ``poincare(curvature=c)`` and
  ``stereographic(sectional_curvature=-c)`` represent the same negative
  constant sectional curvature, subject to their coordinate/API conventions.
* ``euclidean`` corresponds to ``stereographic(sectional_curvature=0.0)``.
* ``sphere_projection(sectional_curvature=c)`` corresponds to
  ``stereographic(sectional_curvature=c)`` for ``c > 0``.
* ``lorentz(curvature=c)`` also has sectional curvature ``-c``, but it uses
  hyperboloid coordinates with one extra ambient coordinate; do not require
  coordinate-wise equality with stereographic representations.

Scope boundary
--------------

This contract does not introduce learnable signed curvature crossing zero,
product or mixed-curvature models, SPD/Siegel/Stiefel spaces, or a generalized
geometry-configuration framework. Those remain separate work.
