Inference
=========

The model-level public inference helpers are ``encode()``, ``distance()``, and
``rank()`` on :class:`neembed.ManifoldSentenceTransformer`. v0.6 also exposes
:func:`neembed.exact_corpus_search` for exact multi-query text-corpus search.

Encoding text
-------------

``encode()`` accepts either one string or a sequence of strings.

.. code-block:: python

   single = model.encode("Shiba Inu")
   batch = model.encode(["Shiba Inu", "dog", "mammal"])

For ``manifold="poincare"``, a single string has shape ``(embedding_dim,)`` and
a sequence has shape ``(batch_size, embedding_dim)``.

For ``manifold="lorentz"``, ``embedding_dim`` remains the intrinsic projected
dimension, while the hyperboloid representation adds one ambient time-like
coordinate. The corresponding shapes are therefore ``(embedding_dim + 1,)``
and ``(batch_size, embedding_dim + 1)``.

NumPy and Tensor output
-----------------------

By default, ``encode()`` returns a NumPy array on CPU:

.. code-block:: python

   embeddings = model.encode(["dog", "cat"])

Set ``convert_to_tensor=True`` to keep the result as a ``torch.Tensor``:

.. code-block:: python

   embeddings = model.encode(
       ["dog", "cat"],
       convert_to_tensor=True,
   )

The Tensor stays on the model device. Lorentz manifold outputs use ``float64``
for numerical stability; Poincare outputs keep the model's ordinary parameter
dtype.

Inference mode
--------------

``encode()`` switches the model to evaluation mode and executes the forward
pass inside ``torch.inference_mode()``. Returned embeddings therefore do not
track gradients and the helper is intended for inference rather than training.

Training code should call the model through its normal ``forward()`` path; the
provided loss does this automatically.

Geodesic distance
-----------------

``distance()`` computes the Geoopt manifold distance between two already
encoded manifold embeddings:

.. code-block:: python

   embeddings = model.encode(["Shiba Inu", "dog"])
   distance = model.distance(embeddings[0], embeddings[1])
   print(float(distance))

Array-like inputs are converted to tensors on the model device. Poincare
distance uses the model parameter dtype, while Lorentz distance is evaluated in
``float64`` to preserve the precision used by the Lorentz manifold path. The
distance calculation runs under ``torch.no_grad()`` and returns a Tensor.
This helper is therefore also inference-oriented; the training loss calls the
manifold distance directly so gradients remain available during optimization.

In-memory geodesic reranking
----------------------------

``rank()`` is a small convenience helper for reranking a supplied candidate
list by ascending manifold geodesic distance:

.. code-block:: python

   results = model.rank(
       "Shiba Inu",
       ["dog", "cat", "mammal"],
       top_k=2,
   )

Each result is a plain Python dictionary containing the original ``candidate``,
its input ``index``, and the scalar ``distance``. The index keeps duplicate text
candidates distinguishable. Equal-distance candidates retain their original
input order.

``top_k=None`` returns the complete ranked list. An integer ``top_k`` must be
between 1 and the number of supplied candidates, inclusive. The candidate list
must be non-empty.

The helper uses the existing ``encode()`` and ``distance()`` inference paths, so
Poincare and Lorentz models use their configured Geoopt geodesic distance. It is
intentionally limited to small in-memory candidate lists: it does not create an
ANN index, persist a corpus, cache embeddings, or integrate with a vector
database.

Exact text-corpus search
------------------------

Use :func:`neembed.exact_corpus_search` when you want exhaustive geodesic search
across one or more queries and a caller-owned text corpus:

.. code-block:: python

   from neembed import exact_corpus_search

   results = exact_corpus_search(
       model,
       queries=["Shiba Inu", "Siamese cat"],
       corpus=["dog", "cat", "bird", "vehicle"],
       top_k=2,
       query_chunk_size=2,
       corpus_chunk_size=3,
   )

The result shape is one ranked list per query. Search uses the same exact Geoopt
geodesic distance as ``rank()``. ``query_chunk_size`` and
``corpus_chunk_size`` bound encoding batches and active distance blocks; they
change the memory/runtime tradeoff but do not approximate the ranking. Equal
distances use corpus input order as the deterministic tie-breaker.

The full query-by-corpus distance matrix is not materialized, but the encoded
query and corpus embeddings are retained for the call. v0.6 still does not
provide ANN, FAISS/HNSW, persistent indexing, or vector-database integration.
For the decision boundary between ``rank()``, exact corpus search, and external
ANN retrieval, see :doc:`retrieval`.
