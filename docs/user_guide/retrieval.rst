Retrieval workflow
==================

v0.5 adds a small retrieval-oriented workflow on top of the existing manifold
sentence-model API. The pieces are intentionally separate: train with optional
caller-supplied hard negatives, evaluate aligned retrieval ranks, rerank a small
candidate list by geodesic distance, and add prototype assignment diagnostics
only when learned manifold structure is part of the task.

All of these paths use the configured Geoopt geodesic distance. Poincare and
Lorentz therefore keep the same geometry semantics used by ``encode()`` and
``distance()`` elsewhere in neembed.

End-to-end reference
--------------------

Run the compact v0.5 reference workflow from the repository root:

.. code-block:: bash

   python examples/v05_retrieval_workflow.py

The script composes the public APIs for explicit hard-negative training,
Recall@K / MRR evaluation, in-memory geodesic reranking, and prototype
assignment evaluation. The corpus, hierarchy metadata, and seed are defined in
the script itself.

This is an engineering and regression reference, not a benchmark claim. The
user-facing script intentionally uses one Poincare configuration; lower-level
tests provide the geometry parity coverage for Poincare and Lorentz.

1. Train with two or three sequences
------------------------------------

The original ranking contract remains supported:

``(anchors, positives)``
   Uses aligned anchor-positive pairs. Off-diagonal positive candidates act as
   in-batch negatives.

``(anchors, positives, negatives)``
   Adds a caller-supplied explicit-negative sequence. Every explicit negative is
   appended to the candidate pool for every anchor in the batch.

The third sequence is optional and does not replace the original two-sequence
path. neembed does not mine hard negatives automatically; callers choose the
negative texts. See :doc:`training` for the loss definition, batch validation,
and optimizer behavior.

For ordinary model-only fine-tuning, including fixed-curvature models and the
opt-in learnable-curvature scalar path, the trainer's default ``AdamW`` behavior
remains valid. A manifold-valued model output does not by itself require a
Riemannian optimizer.

2. Evaluate aligned retrieval
-----------------------------

``ManifoldEmbeddingEvaluator`` treats every supplied positive as a retrieval
candidate for every anchor and uses the same-index positive as the single
relevant target. It reports the existing distance metrics plus:

``Recall@K``
   Fraction of anchors whose aligned target is among the K nearest positive
   candidates. ``recall_at_1`` is the same quantity as
   ``retrieval_accuracy`` under this single-relevant aligned contract.

``MRR``
   Mean reciprocal rank of the aligned target.

Candidate ranks are ordered by ascending Geoopt geodesic distance. See
:doc:`evaluation` for the exact definitions, cutoff behavior, and prototype
assignment metrics.

3. Rerank a small candidate set
-------------------------------

``ManifoldSentenceTransformer.rank()`` encodes one query and a supplied
candidate sequence, then returns plain Python results ordered by ascending
geodesic distance. It is intended for small in-memory reranking only.

The helper does **not** build an ANN index, persist a corpus, cache embeddings,
or provide a vector-database integration. For a large corpus, candidate
retrieval or indexing remains outside neembed; a small externally produced
shortlist can then be scored or reranked with the existing manifold inference
helpers. See :doc:`inference` for the ``rank()`` result contract and ``top_k``
validation.

4. Add prototype diagnostics when needed
----------------------------------------

``ManifoldPrototypeAssignmentEvaluator`` is optional. Use it when the task also
contains learned ``ManifoldPrototypes`` and caller-owned prototype IDs. It
assigns each sentence to the nearest prototype by Geoopt geodesic distance and
reports assignment accuracy plus mean distance to the selected prototype.

Prototype coordinates are true manifold-valued trainable parameters. When they
are optimized, use a Geoopt Riemannian optimizer such as
``RiemannianAdam``. This requirement is specific to manifold-valued trainable
parameters; it does not change the ordinary AdamW model-only path. See
:doc:`learnable_structure` for the parameter categories, persistence boundary,
and joint learnable-curvature / prototype guidance.

What v0.5 does not add
----------------------

The retrieval additions deliberately stop before a retrieval framework. They do
not implement:

- automatic hard-negative mining;
- ANN or vector-database indexing;
- persistent corpus management or embedding caches;
- distributed retrieval infrastructure;
- multi-relevance metrics such as nDCG;
- a claim that the tiny v0.5 example establishes geometry superiority.

This boundary keeps the public API focused on manifold representation,
geodesic scoring, and lightweight evaluation while allowing larger retrieval
systems to own their indexing and candidate-generation layers.
