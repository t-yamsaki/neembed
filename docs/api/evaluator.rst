Evaluator
=========

Binary and graded corpus relevance use separate evaluator paths so existing
``ManifoldCorpusRetrievalEvaluator`` callers keep their MRR/Recall@K contract
unchanged. Use ``ManifoldGradedCorpusRetrievalEvaluator`` when corpus items have
caller-owned non-negative relevance grades and nDCG@K is required.

For graded evaluation, grades greater than zero are treated as binary relevant
for MRR and Recall@K. nDCG uses gain ``2**grade - 1`` and logarithmic discount
``1 / log2(rank + 1)``. Corpus IDs omitted from ``graded_relevance`` have grade
zero. ``ndcg_at_k`` cutoffs larger than the corpus size evaluate the full corpus,
and exact geodesic ranking keeps the same corpus-index tie ordering as
``exact_corpus_search()``.

Hierarchy evaluation is independent of retrieval quality. Parent-child radial
order is strict: equal-radius ties are not counted as correct, while their
violation magnitude is zero because ``max(parent_radius - child_radius, 0)`` is
used. When caller-owned depths are supplied, ``depth_radius_spearman`` is a
tie-aware descriptive rank association between depth and geodesic radius; it
does not imply causality. Degenerate depth/radius rankings report ``0.0`` so the
metric remains finite.

.. autoclass:: neembed.ManifoldEmbeddingEvaluator
   :members:

.. autoclass:: neembed.ManifoldCorpusRetrievalEvaluator
   :members:

.. autoclass:: neembed.ManifoldGradedCorpusRetrievalEvaluator
   :members:

.. autoclass:: neembed.ManifoldHierarchyEvaluator
   :members:
