Retrieval workflow
==================

For v0.10 component configuration, full-product retrieval, diagnostics, and
compatibility, see :doc:`mixed_curvature`.

v0.6 extends the lightweight retrieval path introduced in v0.5 with exact text
corpus search, corpus-level evaluation with explicit IDs and multi-positive
relevance, and caller-invoked offline hard-negative mining. The pieces remain
separate on purpose: neembed provides exact manifold scoring and small utilities,
not a retrieval framework.

All retrieval paths use the configured Geoopt geodesic distance. Poincare and
Lorentz therefore keep the same geometry semantics used by ``encode()`` and
``distance()`` elsewhere in neembed.

Consistent query/document preprocessing
---------------------------------------

``rank()``, exact corpus search, binary/graded retrieval evaluation, and offline
mining accept optional ``input_options``. Like the :doc:`retrieval_objectives`
training API, this maps each text input name to ``task``, ``prompt_name``, and/or
``prompt`` options. Reuse the same query and document settings across training
and inference; only the outer input names differ:

.. list-table:: Input names for role/prompt configuration
   :header-rows: 1
   :widths: 65 35

   * - API
     - ``input_options`` keys
   * - ``model.rank()``
     - ``query``, ``candidates``
   * - ``exact_corpus_search()``, ``mine_hard_negatives()``
     - ``queries``, ``corpus``
   * - Binary and graded corpus retrieval evaluators
     - ``queries``, ``corpus``
   * - ``ManifoldEmbeddingEvaluator``
     - ``anchors``, ``positives``
   * - ``ManifoldHierarchyEvaluator``
     - ``texts``
   * - ``ManifoldPrototypeAssignmentEvaluator``
     - ``sentences``

For an encoder with saved query/document prompts:

.. code-block:: python

   from neembed import (
       ManifoldMultipleNegativesRankingLoss, ManifoldCorpusRetrievalEvaluator,
       exact_corpus_search, mine_hard_negatives,
   )

   query = {"task": "query"}
   document = {"task": "document"}
   # For explicit prefixes, use settings appropriate for your encoder, e.g.
   # query = {"task": "query", "prompt": "query: "}
   # document = {"task": "document", "prompt": "passage: "}
   corpus_options = {"queries": query, "corpus": document}
   queries = ["What is a dog?"]
   corpus = ["A dog is a mammal.", "A car is a vehicle."]
   query_ids, corpus_ids = ["q-dog"], ["dog", "car"]
   relevance = {"q-dog": ["dog"]}

   loss = ManifoldMultipleNegativesRankingLoss(model, input_options={
       "anchors": query, "positives": document, "negatives": document,
   })
   reranked = model.rank(queries[0], corpus, input_options={
       "query": query, "candidates": document,
   })
   results = exact_corpus_search(
       model, queries, corpus, top_k=2, query_chunk_size=1, corpus_chunk_size=1,
       input_options=corpus_options,
   )
   evaluator = ManifoldCorpusRetrievalEvaluator(
       model=model, queries=queries, corpus=corpus, query_ids=query_ids,
       corpus_ids=corpus_ids, relevance=relevance, input_options=corpus_options,
   )
   metrics = evaluator()
   negatives = mine_hard_negatives(
       model, queries, corpus, query_ids=query_ids, corpus_ids=corpus_ids,
       positive_corpus_ids=relevance, input_options=corpus_options,
   )

``ManifoldGradedCorpusRetrievalEvaluator`` uses the same settings for both its
binary metrics and nDCG rankings. Every encoding chunk gets its input's settings;
the batch/block sizes, CPU staging, geodesic distances, query order, and stable
corpus-index tie ordering remain unchanged. Relevance and exclusion IDs refer to
the caller's original corpus, and returned candidate strings are the original
texts. Prefixes are applied once by the encoder, not stored in IDs or text lists.

Missing input keys, empty settings, and ``None`` option values retain raw-text
encoding, even when the encoder has a default prompt. An input never inherits
another input's options. A named prompt can be selected without a task, and
``prompt=""`` disables prompt insertion while retaining an explicit route. See
:doc:`inference` for precedence and supported encoder/Router signatures.

Evaluators copy their settings at construction. Function/method helpers use a
copy for the current call. Unknown input names and option keys are rejected;
prompt names and values are checked by the model when that input is encoded.
These options are caller-owned and are not persisted by model saving.
Generic pair, hierarchy, and prototype evaluation have no implicit roles;
choose their text semantics explicitly. Hierarchy roles do not select a product
geometry component or alter node IDs, edges, or depths.

Product embeddings
------------------

A model with ``product_config`` works with the same retrieval APIs and trainer.
MNRL (including explicit negatives), symmetric MNRL, Triplet, MarginMSE, and
DistanceMSE optimize Geoopt's product geodesic distance. Non-unit fixed component
scales affect both training distances and retrieval rankings. Temperatures,
triplet margins, and regression targets are caller-chosen in that scaled metric;
neembed does not calibrate them automatically.

For example, this H x E workflow uses caller-owned IDs for evaluation and
negative exclusions:

.. code-block:: python

   from neembed import (
       ManifoldSentenceTransformer, ManifoldMultipleNegativesRankingLoss,
       ManifoldTrainer, ManifoldCorpusRetrievalEvaluator,
       exact_corpus_search, mine_hard_negatives,
   )

   model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       product_config=[
           {"name": "hierarchy", "manifold": "poincare",
            "intrinsic_dim": 8, "curvature": 0.5, "scale": 0.7},
           {"name": "residual", "manifold": "euclidean",
            "intrinsic_dim": 8, "scale": 1.6},
       ],
   )
   queries = ["Shiba Inu", "electric car"]
   corpus = ["dog", "canine", "vehicle", "battery"]
   query_ids = ["q-dog", "q-car"]
   corpus_ids = ["dog", "canine", "vehicle", "battery"]
   relevance = {"q-dog": ["dog", "canine"], "q-car": ["vehicle"]}

   ranking = model.rank(queries[0], corpus, top_k=2)
   results = exact_corpus_search(
       model, queries, corpus, top_k=2,
       query_chunk_size=1, corpus_chunk_size=2,
   )
   evaluator = ManifoldCorpusRetrievalEvaluator(
       model=model, queries=queries, query_ids=query_ids,
       corpus=corpus, corpus_ids=corpus_ids, relevance=relevance,
       recall_at_k=(1, 2), query_chunk_size=1, corpus_chunk_size=2,
   )
   metrics = evaluator()
   negatives = mine_hard_negatives(
       model, queries, corpus, query_ids=query_ids, corpus_ids=corpus_ids,
       positive_corpus_ids=relevance, num_negatives=1,
       query_chunk_size=1, corpus_chunk_size=2,
   )
   trainer = ManifoldTrainer(
       model, ManifoldMultipleNegativesRankingLoss(model, temperature=0.5),
   )
   history = trainer.fit(
       [(queries, ["dog", "vehicle"],
         [rows[0]["candidate"] for rows in negatives])],
       epochs=1, evaluator=evaluator,
   )

For H x S, replace the Euclidean component with ``sphere_projection`` and a
positive ``sectional_curvature``. SphereProjection and Lorentz components
make the entire product geometry use float64; on Apple MPS that geometry runs
on CPU. The trainer remains the existing ``ManifoldTrainer``.

Exact search and mining encode text in bounded batches and stream distance
blocks. Binary corpus evaluation uses two passes of those blocks and retains
only relevant-item rank state; it does not build full corpus rankings. Graded
evaluation also supports products through
:class:`neembed.ManifoldGradedCorpusRetrievalEvaluator` and retains top-k results
for nDCG. As for single manifolds, identical distances retain corpus input order,
even across chunk boundaries. IDs and relevance remain external metadata:
duplicate texts may have different IDs, and only declared positive, excluded,
or matching self IDs are filtered by the miner.

See :doc:`../api/product_config` for component dimensions, curvature fields,
scale validation, and persistence. Product prototypes remain unsupported;
component-targeted hierarchy supervision can be combined with full-product
retrieval as described in :doc:`hierarchy`.

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
