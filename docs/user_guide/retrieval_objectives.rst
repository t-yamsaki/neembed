Retrieval objectives and graded evaluation
==========================================

v0.7 adds several retrieval objectives and graded corpus evaluation without
coupling either choice to a particular manifold. The objective answers **what
supervision signal should the model optimize?**; the manifold answers **which
geometry should represent the embeddings?**. Treat those as separate
experimental factors.

All objectives below use the configured Geoopt geodesic distance. The current
Poincare and Lorentz model paths therefore keep the same distance semantics used
by inference and exact retrieval. Choosing a different objective does not
silently change the manifold, curvature, optimizer policy, or retrieval ranking
rule.

Choosing an objective
---------------------

Use the smallest objective that matches the supervision you actually own.

.. list-table:: v0.7 retrieval objectives
   :header-rows: 1
   :widths: 22 25 26 27

   * - Objective
     - Input contract
     - Use it when
     - Target semantics
   * - ``ManifoldMultipleNegativesRankingLoss``
     - ``(anchors, positives)`` or optional ``(anchors, positives, negatives)``
     - You have aligned positives and want in-batch contrastive retrieval training.
     - The aligned positive at the same index is correct; off-diagonal positives
       and any explicit negatives are candidates to rank behind it.
   * - ``ManifoldSymmetricMultipleNegativesRankingLoss``
     - Same two- or three-sequence contract as MNRL.
     - Both anchor-to-positive and positive-to-anchor retrieval directions matter.
     - The loss averages the two directional cross-entropies. Explicit negatives
       are added only to the forward anchor-to-candidate direction because they
       have no aligned reverse target.
   * - ``ManifoldTripletLoss``
     - ``(anchors, positives, negatives)`` with aligned triplets.
     - You have one explicit negative per example and want a direct relative
       distance constraint rather than a batch-wide candidate softmax.
     - Enforce ``d(anchor, positive) + margin <= d(anchor, negative)`` through a
       hinge penalty.
   * - ``ManifoldMarginMSELoss``
     - ``(anchors, positives, negatives, target_margin)``.
     - A teacher or external scorer provides a graded positive-vs-negative
       preference margin.
     - neembed predicts ``d(anchor, negative) - d(anchor, positive)``. Positive
       target margins prefer the positive, zero means no preference, and negative
       margins prefer the supplied negative.
   * - ``ManifoldDistanceMSELoss``
     - ``(texts_a, texts_b, target_distance)``.
     - Supervision gives desired pairwise geodesic distances directly.
     - Targets must be finite and non-negative. ``0`` means coincident points;
       smaller targets mean closer pairs.

MNRL and symmetric MNRL
~~~~~~~~~~~~~~~~~~~~~~~

MNRL uses every off-diagonal positive in a batch as a negative candidate. This
is efficient when a batch naturally supplies useful negatives, but it also means
batch construction is part of the supervision. Avoid duplicate or semantically
valid cross-example positives that would become false negatives.

The symmetric variant keeps the same candidate semantics in the forward
direction and adds a second positive-to-anchor retrieval term. It is opt-in; the
original MNRL remains one-directional. Supplying explicit negatives does **not**
create reverse targets for them, so they participate only in the forward term.

Triplet versus batch-wide ranking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Triplet loss uses only each aligned ``(anchor, positive, negative)`` relation.
That makes the supervision easier to audit when negatives have an explicit
per-example meaning. MNRL instead compares an anchor against a candidate pool,
so its training signal also depends on the other examples in the batch.

Neither formulation is universally better. Compare them under matched encoder,
data, manifold, seed, and evaluation when the research question concerns the
objective itself.

Regression objectives
~~~~~~~~~~~~~~~~~~~~~

``ManifoldMarginMSELoss`` is appropriate when teacher scores encode relative
preference. If larger teacher scores are better, pass
``teacher_positive_score - teacher_negative_score`` directly as the target
margin.

``ManifoldDistanceMSELoss`` is appropriate when the supervision is already a
distance target rather than a ranking preference. neembed does not convert
cosine similarity, probabilities, or teacher logits into distances
automatically; target construction and calibration remain caller-owned.

For batching, target tensors or sequences must align with the corresponding text
pairs or triplets. See :doc:`training` for the trainer contract and optimizer
boundary, and the API reference for exact validation rules:

- :class:`neembed.ManifoldMultipleNegativesRankingLoss`
- :class:`neembed.ManifoldSymmetricMultipleNegativesRankingLoss`
- :class:`neembed.ManifoldTripletLoss`
- :class:`neembed.ManifoldMarginMSELoss`
- :class:`neembed.ManifoldDistanceMSELoss`

Choosing evaluation metrics
---------------------------

Use binary metrics when relevance is binary and nDCG when ordering different
levels of relevance matters.

``MRR``
   Measures how early the **first** relevant corpus item appears. It is useful
   when one early correct result is the main concern, but it does not reward the
   ordering of additional relevant items after the first hit.

``Recall@K``
   Measures what fraction of a query's binary relevant corpus items appear in
   the first K results. It is useful when retrieving multiple acceptable items
   matters.

``nDCG@K``
   Uses caller-owned non-negative relevance grades and discounts gains at lower
   ranks. It is useful when a highly relevant result should count more than a
   weakly relevant result and their ordering matters.

Binary corpus evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`neembed.ManifoldCorpusRetrievalEvaluator` when every relevant item
is simply relevant or not relevant. ``relevance`` maps each query ID to one or
more corpus IDs, and the evaluator reports MRR and configured Recall@K values.

Graded corpus evaluation
~~~~~~~~~~~~~~~~~~~~~~~~

Use :class:`neembed.ManifoldGradedCorpusRetrievalEvaluator` when corpus items
have grades:

.. code-block:: python

   from neembed import ManifoldGradedCorpusRetrievalEvaluator

   evaluator = ManifoldGradedCorpusRetrievalEvaluator(
       model=model,
       query_ids=["q-dog"],
       queries=["Shiba Inu"],
       corpus_ids=["dog", "canine", "vehicle"],
       corpus=["dog", "canine", "vehicle"],
       graded_relevance={
           "q-dog": {
               "dog": 3.0,
               "canine": 1.0,
           },
       },
       recall_at_k=(1, 3),
       ndcg_at_k=(1, 3),
   )
   metrics = evaluator()

For MRR and Recall@K, any grade greater than zero is treated as binary relevant.
For nDCG@K, the documented gain is

.. math::

   g(r) = 2^r - 1,

with logarithmic rank discount

.. math::

   w(i) = \frac{1}{\log_2(i + 1)}.

Corpus IDs omitted from ``graded_relevance`` have grade zero. Internally, gains
are normalized by a query-local common scale when needed so large finite grades
do not overflow; this does not change nDCG because the same positive scale is
applied to DCG and IDCG. A cutoff larger than the corpus size evaluates the full
corpus.

Both binary and graded corpus evaluators keep the exact geodesic ranking and the
stable ``(distance, corpus index)`` tie rule used by
:func:`neembed.exact_corpus_search`. See :doc:`evaluation` for the broader
evaluator contracts and :doc:`retrieval` for exact-search and mining behavior.

Objective and manifold are independent factors
----------------------------------------------

Do not infer a geometry choice from an objective choice. For example, triplet
loss is not inherently "hyperbolic", and MNRL is not inherently tied to one
coordinate model. The same public objective consumes whichever manifold distance
the configured model exposes.

For controlled experiments, change one factor at a time:

1. To compare objectives, hold encoder, data, seed, manifold, curvature, and
   evaluator fixed while swapping the loss.
2. To compare manifolds, hold encoder, data, seed, objective, and evaluator fixed
   while changing the geometry.

Raw geodesic distance magnitudes can differ across geometry/configuration, so do
not interpret them as directly comparable quality scores without an explicitly
matched experiment.

v0.7 objective-comparison regression
------------------------------------

Run the deterministic reference from the repository root:

.. code-block:: bash

   python examples/v07_objective_comparison.py

The example keeps the encoder setup, tiny local data, Poincare manifold, seed,
and graded exact retrieval evaluation fixed while swapping MNRL, Triplet,
MarginMSE, and DistanceMSE. It reports finite training losses plus MRR,
Recall@K, and nDCG@K before and after the tiny training run.

The command is an **engineering regression diagnostic**, not a benchmark and not
evidence that one objective is superior. The normal test suite replaces the
Sentence Transformer with a deterministic tiny encoder so CI does not require a
network download.

The existing v0.4-v0.6 examples remain separate and unchanged. See the
`v0.7 example source
<https://github.com/t-yamsaki/neembed/blob/main/examples/v07_objective_comparison.py>`_
for the complete composition.
