Retrieval workflow
==================

v0.6 extends the lightweight retrieval path introduced in v0.5 with exact text
corpus search, corpus-level evaluation with explicit IDs and multi-positive
relevance, and caller-invoked offline hard-negative mining. The pieces remain
separate on purpose: neembed provides exact manifold scoring and small utilities,
not a retrieval framework.

All retrieval paths use the configured Geoopt geodesic distance. Poincare and
Lorentz therefore keep the same geometry semantics used by ``encode()`` and
``distance()`` elsewhere in neembed.

Choose the retrieval path
-------------------------

``ManifoldSentenceTransformer.rank()``
   Use this for one query and a small in-memory candidate list. It encodes the
   supplied candidates and returns them ordered by exact geodesic distance.

``exact_corpus_search()``
   Use this when you own a text corpus and want exact geodesic top-k search over
   one or more queries without materializing the full query-by-corpus distance
   matrix. Text encoding and distance evaluation are processed in bounded
   batches/blocks.

ANN or vector-database retrieval
   Use an external system when the corpus is too large for exact exhaustive
   search. v0.6 does not add FAISS, HNSW, a vector database, persistent indexes,
   or approximate search. An external system may generate candidates, after
   which neembed can still score or rerank them with its manifold helpers.

The distinction is about scale and ownership, not distance semantics:
``rank()`` and ``exact_corpus_search()`` both use exact configured Geoopt
geodesic distance.

v0.6 end-to-end reference
-------------------------

Run the compact v0.6 workflow from the repository root:

.. code-block:: bash

   python examples/v06_exact_retrieval_workflow.py

The script composes public APIs only:

1. exact corpus search;
2. corpus MRR / Recall@K evaluation with explicit IDs;
3. offline hard-negative mining with positive exclusions;
4. the existing ``(anchors, positives, negatives)`` training contract;
5. the same retrieval diagnostics after training.

The tiny corpus and fixed seed live in the script. This is an engineering and
regression reference, not a benchmark claim and not evidence that training must
improve the reported metrics.

See the `v0.6 example source
<https://github.com/t-yamsaki/neembed/blob/main/examples/v06_exact_retrieval_workflow.py>`_
for the full composition.

Exact corpus search
-------------------

``exact_corpus_search()`` accepts multiple query texts and a corpus of candidate
texts:

.. code-block:: python

   from neembed import exact_corpus_search

   results = exact_corpus_search(
       model,
       ["Shiba Inu", "Siamese cat"],
       ["dog", "cat", "bird", "vehicle"],
       top_k=2,
       query_chunk_size=2,
       corpus_chunk_size=3,
   )

One ranked list is returned per query. Every result contains the original
``candidate`` text, its corpus ``index``, and exact geodesic ``distance``.
Results are ordered by ascending distance; equal-distance candidates preserve
corpus input order through the stable ``(distance, index)`` ordering rule.

``top_k=None`` returns a full corpus ranking for every query. For a bounded
result set, prefer an integer ``top_k`` so the search retains only the best K
candidates seen for each query.

Chunking is exact, not approximate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``query_chunk_size`` and ``corpus_chunk_size`` control both text-encoding batch
sizes and the active geodesic distance blocks. Completed embeddings are staged
on CPU and the full query-by-corpus distance matrix is not materialized.

Smaller chunks generally reduce peak encoder/device working memory, while more
chunks can increase runtime because more forward passes and distance blocks are
processed. Changing a chunk size does **not** change exact-search semantics or
introduce approximation; it only changes how the same exhaustive computation is
scheduled.

The encoded query and corpus embeddings themselves still exist for the duration
of the call, so chunking is not a substitute for ANN or persistent indexing when
the corpus becomes very large. See :doc:`inference` for the concise inference
contract and :func:`neembed.exact_corpus_search` for argument details.

Corpus retrieval evaluation
----------------------------

``ManifoldCorpusRetrievalEvaluator`` evaluates a real corpus rather than the
older aligned-positive candidate matrix. Query IDs and corpus IDs are explicit
caller-owned strings, and ``relevance`` maps every query ID to one or more
relevant corpus IDs:

.. code-block:: python

   from neembed import ManifoldCorpusRetrievalEvaluator

   evaluator = ManifoldCorpusRetrievalEvaluator(
       model=model,
       query_ids=["q-dog", "q-cat"],
       queries=["Shiba Inu", "Siamese cat"],
       corpus_ids=["dog", "cat", "bird", "vehicle"],
       corpus=["dog", "cat", "bird", "vehicle"],
       relevance={
           "q-dog": ["dog"],
           "q-cat": ["cat", "bird"],
       },
       recall_at_k=(1, 2, 4),
   )

For each query, Recall@K is the fraction of that query's relevant corpus items
found in the first K exact results. The evaluator then averages that fraction
across queries. MRR uses the rank of the first relevant result for each query.
This differs intentionally from ``ManifoldEmbeddingEvaluator``, whose aligned
single-positive contract keeps ``retrieval_accuracy == recall_at_1``.

IDs must be unique, every query must have a non-empty relevance set, and all
relevance IDs must exist in the supplied corpus. See :doc:`evaluation` for the
metric definitions and validation boundary.

Offline hard-negative mining
----------------------------

``mine_hard_negatives()`` is an explicit preprocessing step. neembed does not mine hard negatives automatically.
The helper runs only when the caller invokes it; it does not run inside
``ManifoldTrainer.fit()`` and it does not create a background or online mining
loop.

.. code-block:: python

   from neembed import mine_hard_negatives

   mined = mine_hard_negatives(
       model,
       queries=["Shiba Inu", "Siamese cat"],
       corpus=["dog", "cat", "wolf", "tiger"],
       query_ids=["q-dog", "q-cat"],
       corpus_ids=["dog", "cat", "wolf", "tiger"],
       positive_corpus_ids={
           "q-dog": ["dog"],
           "q-cat": ["cat"],
       },
       num_negatives=1,
   )

Known positives are never returned. ``excluded_corpus_ids`` can add caller-owned
exclusions, and a corpus item whose ID equals the current query ID is excluded
as an explicit self item. The nearest remaining candidates are selected by exact
geodesic distance, with corpus input order as the equal-distance tie-breaker.
Returned dictionaries include ``corpus_id``, ``candidate``, original ``index``,
and ``distance`` so mining decisions can be audited.

When mined negatives are fed into one three-sequence ranking batch, remember
that ``ManifoldMultipleNegativesRankingLoss`` makes every explicit negative a
candidate for every anchor in that batch. If one query's mined negative is a
positive for another query in the same batch, exclude the union of that batch's
positive IDs during mining or split the triples into compatible batches. The
v0.6 reference example demonstrates the union-exclusion approach.

See :doc:`training` for the three-sequence loss semantics and
:func:`neembed.mine_hard_negatives` for the focused API contract.

Training composition
--------------------

After mining, the existing trainer API remains unchanged:

.. code-block:: python

   mined_negative_texts = tuple(items[0]["candidate"] for items in mined)

   history = trainer.fit(
       [(anchors, positives, mined_negative_texts)],
       epochs=3,
   )

The trainer does not know whether the third sequence was mined, manually chosen,
or produced by another system. This separation keeps hard-negative policy under
caller control and avoids turning the trainer into an online retrieval
framework.

For ordinary model-only fine-tuning, including fixed curvature and the opt-in
learnable-curvature scalar path, the default AdamW path remains valid. A
manifold-valued model output does not by itself require a Riemannian optimizer.
True manifold-valued trainable parameters such as ``ManifoldPrototypes`` still
require an appropriate Geoopt optimizer such as ``RiemannianAdam``. See
:doc:`training` and :doc:`learnable_structure` for that optimizer boundary.

Aligned retrieval and prototype diagnostics
-------------------------------------------

The v0.5 utilities remain supported. ``ManifoldEmbeddingEvaluator`` is useful
for aligned anchor-positive evaluation, while
``ManifoldPrototypeAssignmentEvaluator`` remains optional for tasks with learned
``ManifoldPrototypes``. v0.6 does not redefine those contracts; it adds corpus
retrieval and mining beside them.

What v0.6 still does not add
----------------------------

The retrieval additions deliberately stop before a general retrieval system.
They do not implement:

- ANN, FAISS, HNSW, or vector-database integrations;
- persistent indexes, corpus stores, or embedding caches;
- online or asynchronous hard-negative mining;
- distributed retrieval or cross-device mining;
- nDCG, MAP, or graded relevance;
- benchmark superiority claims.

This boundary keeps neembed focused on manifold representation, exact geodesic
scoring, lightweight evaluation, and caller-controlled training composition.
