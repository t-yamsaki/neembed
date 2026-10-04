Constant-curvature semantics for v0.9
=====================================

This page is the canonical guide for choosing and configuring the v0.9
constant-curvature backends. It defines their intended roles, public curvature
names, validation, persistence, numerical dtype policy, and compatibility with
the existing training and retrieval surface.

Choosing a geometry
-------------------

The five backends expose different coordinate models and curvature regimes. A
geometry choice changes the metric used by losses and retrieval; it does **not**
by itself guarantee better embedding quality. Compare geometries under matched
data, initialization, objective, and evaluation before drawing task-specific
conclusions.

``manifold="euclidean"``
   Use as the flat :math:`K=0` baseline when manifold curvature is not required.
   It keeps the ordinary model dtype and uses Euclidean geodesic distance.

``manifold="poincare"``
   Use the established Poincare-ball hyperbolic path when you want negative
   curvature with ball coordinates or need backward compatibility with existing
   neembed hyperbolic models. Its public ``curvature`` value is the positive
   magnitude :math:`c` of sectional curvature :math:`K=-c`.

``manifold="lorentz"``
   Use the established hyperboloid representation when Lorentz coordinates are
   preferable. It represents the same negative-curvature magnitude convention
   as Poincare but adds one ambient coordinate and uses the existing float64
   geometry path.

``manifold="sphere_projection"``
   Use the dedicated positive-curvature stereographic backend when
   :math:`K>0`. It requires fixed positive ``sectional_curvature`` and uses the
   v0.9 float64 stereographic geometry policy.

``manifold="stereographic"``
   Use the generic signed-curvature backend when one API should cover
   :math:`K<0`, :math:`K=0`, and :math:`K>0`. The public value is the signed
   ``sectional_curvature`` itself. This backend is fixed-curvature in v0.9 and
   uses float64 geometry operations.

Two distinctions matter when comparing these backends:

* Poincare and Lorentz retain the old positive-magnitude ``curvature`` API.
  Generic Stereographic and SphereProjection use signed
  ``sectional_curvature``; Euclidean is fixed at zero.
* ``euclidean`` and ``stereographic(sectional_curvature=0.0)`` describe flat
  constant curvature but are not numerically interchangeable distance APIs in
  the current Geoopt convention. Their ``expmap0`` coordinates agree, while
  generic ``Stereographic(k=0)`` has conformal factor ``2`` and therefore
  reports geodesic distances equal to twice the corresponding Euclidean
  :math:`L_2` distance.

Quick configuration examples
----------------------------

.. code-block:: python

   from neembed import ManifoldSentenceTransformer

   euclidean = ManifoldSentenceTransformer(
       model_name,
       manifold="euclidean",
   )

   poincare = ManifoldSentenceTransformer(
       model_name,
       manifold="poincare",
       curvature=0.5,  # K = -0.5
   )

   lorentz = ManifoldSentenceTransformer(
       model_name,
       manifold="lorentz",
       curvature=0.5,  # K = -0.5
   )

   sphere = ManifoldSentenceTransformer(
       model_name,
       manifold="sphere_projection",
       sectional_curvature=0.5,  # K = +0.5
   )

   signed = ManifoldSentenceTransformer(
       model_name,
       manifold="stereographic",
       sectional_curvature=-0.5,  # K = -0.5
   )

The detailed constructor signature is generated in :doc:`../api/model`.

Compatibility with existing workflows
-------------------------------------

The core v0.4-v0.8 objectives, evaluators, and exact-retrieval helpers operate on
``model(...)``, ``model.distance()``, ``manifold.dist()``, or
``manifold.dist0()`` rather than branching on a specific constant-curvature
backend. The table below records the supported v0.9 API surface.
For v0.10 mixed-curvature products, see the :ref:`product-compatibility` matrix
and :doc:`mixed_curvature`.

``Supported`` means the API is intended to run with that backend under its dtype
and device policy. It is not a claim that every geometry is equally suitable for
a dataset, that every curvature magnitude is numerically benign, or that a
geometry will improve quality.

.. list-table:: Constant-curvature compatibility matrix
   :header-rows: 1
   :stub-columns: 1

   * - API surface
     - Euclidean
     - Poincare
     - Lorentz
     - SphereProjection
     - Stereographic
   * - ``encode()``, ``distance()``, ``rank()``
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - MNRL and symmetric MNRL
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - Triplet, MarginMSE, DistanceMSE
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - Corpus / graded retrieval evaluators
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - ``exact_corpus_search()`` and ``mine_hard_negatives()``
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - Radial/depth hierarchy objectives and hierarchy evaluator
     - Supported
     - Supported
     - Supported
     - Supported
     - Supported
   * - Learnable curvature
     - No
     - Supported
     - Supported
     - No
     - No

The retrieval objectives referred to above are
:class:`neembed.ManifoldMultipleNegativesRankingLoss`,
:class:`neembed.ManifoldSymmetricMultipleNegativesRankingLoss`,
:class:`neembed.ManifoldTripletLoss`,
:class:`neembed.ManifoldMarginMSELoss`, and
:class:`neembed.ManifoldDistanceMSELoss`. Retrieval evaluation includes
:class:`neembed.ManifoldCorpusRetrievalEvaluator` and
:class:`neembed.ManifoldGradedCorpusRetrievalEvaluator`; exact search and mining
are documented in :doc:`retrieval`.

The radial/depth hierarchy row covers
:class:`neembed.ManifoldRadialOrderLoss`,
:class:`neembed.ManifoldDepthLoss`,
:class:`neembed.ManifoldHierarchyTripletLoss`,
:class:`neembed.ManifoldRetrievalHierarchyLoss`, and
:class:`neembed.ManifoldHierarchyEvaluator`. These APIs use geodesic distance
from the configured origin through ``dist0``. On positive curvature, the
available geodesic radius is bounded, so callers should choose radial margins and
``radial_scale`` values that fit the intended spherical regime rather than
assuming hyperbolic-style unbounded radial capacity.

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
* ``sectional_curvature == 0`` means flat stereographic geometry.
* ``sectional_curvature > 0`` means spherical stereographic geometry.

The public name is deliberately not ``k``. Geoopt's Lorentz backend already
uses ``k`` for the positive squared hyperboloid radius, where ``k = 1 / c``;
reusing that name publicly would make the two meanings ambiguous. For generic
Stereographic, Geoopt's implementation-level ``k`` equals the signed public
``sectional_curvature``. For Lorentz, Geoopt's ``k`` is instead the positive
squared radius. Callers should therefore use neembed's public field names rather
than transporting a raw ``k`` value between backends.

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

The v0.9 implementations add one keyword-only argument to the shared model
constructor and manifold factory:

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

The manifold names and validation rules are fixed as follows.

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
``neembed_config.json`` schema. In particular, v0.9 does not rewrite existing
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

Loaders select the curvature field from the saved ``manifold`` name. They do not
infer signed curvature by changing the meaning of an old ``"curvature"`` field,
and new geometry does not persist Geoopt's implementation-level ``k`` as a
public field. No configuration-version framework is required for this separation
because the distinct field names are sufficient.

The signed value on new geometry is exposed through the read-only
``model.sectional_curvature`` property. This does not change the meaning of the
existing ``model.curvature`` property for Poincare/Lorentz. Persistence is
defined by the keys above rather than by requiring one unified curvature
property across every geometry.

Equivalence relationships
-------------------------

The v0.9 geometry implementations use the following relationships for parity
regressions:

* ``poincare(curvature=c)`` and
  ``stereographic(sectional_curvature=-c)`` represent the same negative
  constant sectional curvature, subject to their coordinate/API conventions.
* ``euclidean`` and ``stereographic(sectional_curvature=0.0)`` are both flat.
  Their ``expmap0`` coordinates agree, but the generic Stereographic distance is
  twice Euclidean :math:`L_2` distance under Geoopt's conformal convention.
* ``sphere_projection(sectional_curvature=c)`` corresponds to
  ``stereographic(sectional_curvature=c)`` for ``c > 0``.
* ``lorentz(curvature=c)`` also has sectional curvature ``-c``, but it uses
  hyperboloid coordinates with one extra ambient coordinate; do not require
  coordinate-wise equality with stereographic representations.

Numerical dtype and device policy
---------------------------------

Geoopt strongly recommends double precision for the stereographic model because
its projection, conformal-factor, inverse-trigonometric, and distance operations
can become numerically fragile in ``float32``. neembed therefore treats
``sphere_projection`` and ``stereographic`` as **float64 geometry paths**.

The policy is deliberately narrower than converting the whole sentence model to
double precision:

* encoder and optional projection parameters retain their existing dtype, which
  is commonly ``float32``;
* after the projection, the tangent vector is promoted to ``torch.float64``
  before ``expmap0`` for ``sphere_projection`` and ``stereographic``;
* the Geoopt sectional-curvature tensor ``k`` for those backends is constructed
  as ``float64``;
* encoded SphereProjection/Stereographic manifold points are therefore
  ``float64`` tensors (or ``float64`` NumPy arrays when tensor output is not
  requested);
* ``model.distance()`` converts external inputs to ``float64`` before evaluating
  SphereProjection/Stereographic geodesic distance;
* exact corpus search, corpus evaluation, and hard-negative mining preserve the
  encoded ``float64`` values while staging embeddings on CPU and while moving
  active distance blocks to the geometry device.

The cast after the projection remains differentiable: gradients from a
``float64`` manifold loss flow through the cast back to ordinary projection and
encoder parameters in their original dtype. The policy therefore improves the
geometry calculation without doubling the storage of the whole transformer.

On Apple MPS, ``float64`` tensors are unavailable. SphereProjection and generic
Stereographic therefore keep the encoder/projection path on MPS but execute the
float64 manifold path on CPU; their encoded manifold tensors and geodesic
distance work are CPU-resident in that configuration. Module dtype/device
transforms preserve this fixed-double geometry policy rather than silently
narrowing the stereographic state. Lorentz keeps its pre-v0.9 float64 behavior;
v0.9 does not add the SphereProjection/Stereographic MPS fallback policy to the
legacy Lorentz backend.

Poincare and Euclidean retain their existing model-parameter dtype in v0.9 for
backward compatibility. The cost of the fixed-double stereographic path is that
its embeddings and distance blocks use roughly twice the memory of their
``float32`` equivalents, and double precision may be slower on some devices.
``float64`` improves robustness but is not a guarantee against every extreme
radius/curvature regime. v0.9 does not add automatic curvature clipping, radius
clipping, arbitrary precision, or a mixed-precision geometry policy; callers
should treat non-finite values as a failed numerical regime rather than silently
accepting them.

Matched comparison example
--------------------------

Run the deterministic v0.9 engineering comparison from the repository root:

.. code-block:: bash

   python examples/v09_constant_curvature_comparison.py

``examples/v09_constant_curvature_comparison.py`` holds the tiny retrieval data,
MNRL objective, temperature, projection dimension, seed, and retrieval evaluator
fixed while exercising Euclidean, Poincare, Lorentz, SphereProjection, and
generic Stereographic at negative, zero, and positive signed curvature. It
reports finite loss/retrieval diagnostics plus curvature and dtype/device
metadata.

The example is an engineering regression reference, **not a benchmark or a
geometry-superiority claim**. Its purpose is to make configuration and numerical
behavior comparable under one controlled workflow; it does not establish which
geometry should be chosen for a real dataset.

Scope boundary
--------------

This guide covers single-manifold geometry. Flat mixed-curvature products are
documented in :doc:`mixed_curvature`. Learnable signed curvature crossing zero,
SPD/Siegel/Stiefel spaces, and geometry-selection theory remain outside scope.
