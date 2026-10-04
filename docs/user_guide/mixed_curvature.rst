Mixed-curvature product embeddings
==================================

v0.10 development adds flat Geoopt ProductManifold embeddings to the existing
sentence encoder, projection, losses, trainer, and save/load workflow. Component
types, dimensions, names, curvatures, and scales are explicit caller choices.
More components or more curvature do not guarantee better representations.
Keep :doc:`constant_curvature_semantics` as a single-manifold baseline and
compare held-out task metrics under matched data, encoder, training budget, and
total intrinsic dimension.

Configure an ordered product
----------------------------

The minimal fields are ``manifold`` and positive integer ``intrinsic_dim``,
plus the curvature field required by that geometry. Normalization can generate
``name``, but explicit unique stable names make diagnostics and hierarchy
selection easier to maintain. ``scale`` is a positive finite distance multiplier
and defaults to ``1.0``.

.. code-block:: python

   from neembed import ManifoldSentenceTransformer, normalize_product_config

   config = normalize_product_config([
       {"name": "hierarchy", "manifold": "poincare", "intrinsic_dim": 8,
        "curvature": 0.5, "scale": 1.5},
       {"name": "semantic", "manifold": "sphere_projection", "intrinsic_dim": 8,
        "sectional_curvature": 0.25, "scale": 0.75},
       {"name": "residual", "manifold": "euclidean", "intrinsic_dim": 16},
   ])
   model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2", product_config=config,
   )
   assert config.projection_dim == 32
   assert config.ambient_dim == 32
   points = model.encode(["dog", "mammal"], convert_to_tensor=True)
   assert points.shape == (2, config.ambient_dim)

These names do not make components learn their roles automatically.
``product_config`` also accepts the ordered list directly or the versioned
dictionary returned by ``config.to_dict()``. See :doc:`../api/product_config`
for the normalized schema and validation API.

.. list-table:: Component curvature and dimensions
   :header-rows: 1

   * - ``manifold``
     - Curvature field
     - Intrinsic / ambient width
   * - ``poincare``
     - ``curvature=c>0``, sectional curvature ``K=-c``; default ``c=1``
     - ``D / D``
   * - ``lorentz``
     - Same positive magnitude; Geoopt squared radius ``k=1/c``
     - ``D / (D+1)``
   * - ``euclidean``
     - ``sectional_curvature=0``, optional; rejects ``curvature``
     - ``D / D``
   * - ``sphere_projection``
     - Required finite ``sectional_curvature>0``
     - ``D / D``
   * - ``stereographic``
     - Required finite signed ``sectional_curvature``, negative, zero, or positive
     - ``D / D``

The learned Euclidean projection emits the sum of intrinsic widths,
``config.projection_dim``. Geoopt packs points in configuration order with final
width ``config.ambient_dim``, the sum of ambient widths. Replacing the example's
8-dimensional Poincare component with Lorentz keeps projection width 32 but
makes packed width 33. Component slices then become ``0:9``, ``9:17``, and
``17:33``. The extra time coordinate is not intrinsic capacity; do not split a
Lorentz product by intrinsic widths.

Supplying ``product_config`` selects product mode with the default constructor
arguments; explicit ``manifold="product"`` is also accepted. If supplied,
``embedding_dim`` must equal the total projection width. Leave top-level
``curvature`` at its default, omit top-level ``sectional_curvature``, and
configure curvature per component. Product ``learnable_curvature`` is unsupported.
Scalar ``model.curvature`` and ``model.sectional_curvature`` properties reject
products: inspect ``model.product_config.components`` instead.

Geoopt mapping, distance, and scales
-----------------------------------

neembed splits the projection in order, applies each underlying component's
origin map, and delegates packing and total distance to Geoopt. Euclidean mapping
is the identity; Lorentz prepends a zero time-like tangent coordinate. Non-unit
scales wrap components in ``geoopt.Scaled(base, scale, learnable=False)``; unit
scales leave them unwrapped. Mapping uses the base manifold's ``expmap0`` instead
of ``Scaled.expmap0``, so changing a scale weights distances between the same
encoded points instead of rescaling tangent coordinates.

For unscaled distances ``d_i`` and scales ``s_i``, the metric is
``sqrt(sum((s_i * d_i)**2))``. ``model.distance()`` calls Geoopt
``ProductManifold.dist`` with its squared-distance aggregation and numerical
safeguards. Use that total for objectives and ranking: reconstructing it from
component diagnostics can differ numerically, including at coincident points.

All components share one geometry dtype/device. Any Lorentz, SphereProjection,
or Stereographic component makes the entire product use ``float64``, even if
the encoder/projection is cast to lower precision. Poincare/Euclidean products
follow the projection dtype. On Apple MPS, fixed-double product geometry runs
on CPU while the encoder/projection can remain on MPS. Scales and their squared
metric factors must be positive finite values representable in the geometry
dtype. Invalid scales are rejected at construction; incompatible dtype narrowing
is rejected before model state changes. Monitor finite distances and losses
during training: finite configuration values alone do not ensure stability.

Train retrieval and optional hierarchy
--------------------------------------

Retrieval uses the full scaled product with the existing losses and trainer:

.. code-block:: python

   from neembed import ManifoldMultipleNegativesRankingLoss, ManifoldTrainer

   retrieval = ManifoldMultipleNegativesRankingLoss(model)
   trainer = ManifoldTrainer(model, retrieval, learning_rate=1e-3)
   trainer.fit([
       (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
   ], epochs=2)

Temperature, margins, and teacher distance targets use the resulting scaled
metric; they are not calibrated automatically when dimensions, curvature, or
scales change. The default optimizer is ordinary AdamW over the Euclidean
encoder/projection parameters. Manifold-valued outputs alone do not require a
Riemannian optimizer.

Optional hierarchy supervision must designate one compatible component:

.. code-block:: python

   from neembed import (
       ManifoldHierarchyTripletLoss, ManifoldRetrievalHierarchyLoss,
   )

   hierarchy = ManifoldHierarchyTripletLoss(model, component="hierarchy")
   combined = ManifoldRetrievalHierarchyLoss(
       retrieval, hierarchy, hierarchy_weight=0.3,
   )
   trainer = ManifoldTrainer(model, combined, learning_rate=1e-3)
   trainer.fit([
       (
           (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
           (["animal", "animal"], ["dog", "cat"], ["car", "vehicle"]),
       ),
   ], epochs=2)

``component`` accepts a stable name or zero-based integer index and is required
for product hierarchy APIs, even when only one component is compatible. Initially
only components configured as ``poincare`` or ``lorentz`` are supported.
Euclidean, SphereProjection, and generic Stereographic are rejected here,
including negative-curvature Stereographic.

Radial order/depth use the selected scaled component's distance from its origin.
Directed triplets use that same component for proximity and radial terms.
Targets and margins include its scale. Retrieval in the composite objective
still uses the full product. Direct hierarchy gradients reach the selected
projection rows, but the shared encoder can change other representations too;
this does not establish independent roles. See :doc:`hierarchy` for caller-owned
labels, directed negatives, batch composition, and single-manifold behavior.

.. _product-compatibility:

Public API compatibility
------------------------

This reference table describes executable support, not expected quality.
Single-manifold support remains in :doc:`constant_curvature_semantics`.

.. list-table:: Product compatibility matrix
   :header-rows: 1
   :widths: 45 55

   * - API surface
     - Product behavior
   * - ``encode()``, ``distance()``, ``rank()``, ``exact_corpus_search()``
     - Full scaled product; lower distance ranks first
   * - ``ManifoldMultipleNegativesRankingLoss`` and ``ManifoldSymmetricMultipleNegativesRankingLoss``
     - Full product; MNRL retains optional explicit negatives
   * - ``ManifoldTripletLoss``, ``ManifoldMarginMSELoss``, ``ManifoldDistanceMSELoss``
     - Full product distance
   * - ``ManifoldTrainer``
     - Existing trainer and ordinary AdamW default
   * - ``ManifoldEmbeddingEvaluator``
     - Full product aligned retrieval evaluation
   * - ``ManifoldCorpusRetrievalEvaluator`` and ``ManifoldGradedCorpusRetrievalEvaluator``
     - Full product MRR/Recall@K; graded path also reports nDCG@K
   * - ``mine_hard_negatives()``
     - Full product exact mining with declared exclusions
   * - ``ManifoldDepthLoss``, ``ManifoldRadialOrderLoss``, ``ManifoldHierarchyTripletLoss``
     - Explicit ``component=`` Poincare/Lorentz name or index required
   * - ``ManifoldHierarchyEvaluator``
     - Selected Poincare/Lorentz component radial/depth diagnostics
   * - ``ManifoldRetrievalHierarchyLoss``
     - Full-product retrieval with selected-component hierarchy
   * - ``product_distance_diagnostics()``
     - No-gradient total and named scaled component distances
   * - ``save_pretrained()`` / ``from_pretrained()``
     - Normalized product configuration, encoder, and projection
   * - ``ManifoldPrototypes``, ``ManifoldPrototypeHierarchyLoss``, ``ManifoldPrototypeAssignmentEvaluator``
     - Unsupported for products; product prototype construction rejects them
   * - Learnable component curvature or scales
     - Unsupported; all product geometry metadata is fixed

Evaluate and inspect
--------------------

``rank()`` handles small in-memory lists; ``exact_corpus_search()`` and corpus
evaluators encode in bounded chunks and evaluate distance blocks. Equal distances
keep corpus-index order. IDs, binary/graded relevance, positive/self exclusions,
and hierarchy labels remain caller-owned; see :doc:`retrieval` and
:doc:`retrieval_objectives`.

.. code-block:: python

   from neembed import ManifoldCorpusRetrievalEvaluator

   metrics = ManifoldCorpusRetrievalEvaluator(
       model=model,
       query_ids=["q-dog", "q-cat"], queries=["Shiba Inu", "Siamese cat"],
       corpus_ids=["dog", "cat", "car"], corpus=["dog", "cat", "car"],
       relevance={"q-dog": ["dog"], "q-cat": ["cat"]},
       recall_at_k=(1, 3),
   )()
   print(metrics)  # Full-product MRR and Recall@K

For diagnostics, encode text before passing packed points:

.. code-block:: python

   queries = model.encode(["dog", "cat"], convert_to_tensor=True)
   candidates = model.encode(["mammal", "vehicle"], convert_to_tensor=True)
   report = model.product_distance_diagnostics(
       queries[:, None, :], candidates[None, :, :],
   )
   assert report["total_distance"].shape == (2, 2)
   assert list(report["component_distances"]) == [
       "hierarchy", "semantic", "residual",
   ]

Aligned batches return one distance per pair; broadcast singleton dimensions
produce query-by-candidate matrices. The total comes directly from Geoopt.
Component tensors include fixed scale multipliers and preserve configuration
order. They describe geometry, not attribution, learned importance, or automatic
scale selection. The helper always runs without gradients and leaves model
mode/state unchanged; ``encode()`` retains its usual switch to evaluation mode.
See :doc:`../api/product_config` for validation and output details.

Save, load, and reproduce
-------------------------

``model.save_pretrained("./saved_product")`` keeps the existing layout:
``encoder/``, ``neembed_config.json``, and ``projection.pt``. Version-1 product
metadata records names, order, dimensions, curvatures, and fixed scales; loading
reconstructs components and restores encoder/projection state. No checkpoint
schema bump or separate scale checkpoint is needed. Learnable scales are not
exposed, so there are no scale optimizer parameters, trainability flags, or
optimizer state to restore for them.

After loading, reconstruct hierarchy losses/evaluators using the persisted
stable component name. Selectors, supervision, objective hyperparameters, and
optimizer state are external to the sentence-model checkpoint; see
:doc:`saving_loading`. Early Issue #130 checkpoints with non-unit metadata-only
scales now apply those scales on load; unit-scale checkpoints keep prior behavior.

.. code-block:: python

   model.save_pretrained("./saved_product")
   loaded = ManifoldSentenceTransformer.from_pretrained("./saved_product")
   assert loaded.product_config == model.product_config
   loaded_hierarchy = ManifoldHierarchyTripletLoss(loaded, component="hierarchy")

Run the v0.10 regression from the repository root:

.. code-block:: bash

   python examples/v10_mixed_curvature_workflow.py
   python examples/v10_mixed_curvature_workflow.py --model ./local_encoder --output ./saved_product

The `example source
<https://github.com/t-yamsaki/neembed/blob/main/examples/v10_mixed_curvature_workflow.py>`_
trains a small named Poincare x SphereProjection x Euclidean product, reports
before/after retrieval and selected hierarchy metrics, inspects component
distances, and checks save/load parity. The default encoder may need a download;
CI uses a deterministic tiny encoder without downloads. Without ``--output``,
the temporary checkpoint is removed. This is an engineering regression, not a
benchmark or evidence of geometry superiority or guaranteed metric improvement.

Limitations
-----------

Products are flat ordered vector-valued components, not an arbitrary
manifold-composition language. Nested products, SPD/Siegel/Stiefel/Birkhoff
components, product prototypes, global product radial hierarchy semantics,
multi-component hierarchy supervision, and learnable component curvature/scales
are outside the current API. ANN/vector-database integration, distributed
retrieval, architecture search, and automatic role/geometry discovery are also
outside scope. Use held-out task metrics to justify extra configuration and
numerical cost.
