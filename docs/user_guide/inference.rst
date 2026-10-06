Inference
=========

The model-level public inference helpers include ``encode()``, ``encode_query()``,
``encode_document()``, ``distance()``, and ``rank()`` on
:class:`neembed.ManifoldSentenceTransformer`. v0.6 also exposes
:func:`neembed.exact_corpus_search` for exact multi-query text-corpus search.

For product models, ``product_distance_diagnostics()`` reports named component
distances alongside the unchanged Geoopt total. Encode text first and pass the
packed embeddings to the helper. See :doc:`../api/product_config` for
broadcasting, fixed scale semantics, and a diagnostic example.

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

Query/document roles and prompts
--------------------------------

Use ``encode_query()`` and ``encode_document()`` to declare the input role.
Both call the same geometry mapping as ``encode()`` and share prompt selection
with differentiable ``forward()``:

.. code-block:: python

   # Explicit prefixes for a model that requires query/passage instructions.
   queries = model.encode_query(["What is a dog?"], prompt="query: ")
   documents = model.encode_document(["A dog is a mammal."], prompt="passage: ")

   # Saved prompts are selected from model.encoder.prompts.
   queries = model.encode(["What is a dog?"], task="query")
   documents = model.encode_document(["A dog is a mammal."])

``forward()`` and ``encode()`` accept keyword-only ``task``, ``prompt_name``,
and ``prompt``. Supported tasks are ``"query"``, ``"document"``, or ``None``.
The two inference wrappers set the corresponding task. Prompt selection follows
these rules:

1. An explicit ``prompt`` is used as-is. ``prompt=""`` disables automatic
   prompt selection while retaining the task for routing.
2. An explicit ``prompt_name`` selects that key in ``model.encoder.prompts``.
3. With a task but no explicit prompt/name, query selects ``query``; document
   selects the first existing key in ``document``, ``passage``, ``corpus`` order.
4. If no role prompt exists, the saved ``encoder.default_prompt_name`` is used
   when configured. Otherwise the input has no prompt.

``prompt`` and ``prompt_name`` together are rejected, even when the explicit
prompt is empty. Unknown prompt names and unsupported tasks raise ``ValueError``.
Selected prompts must be strings. To configure custom named prompts without
another configuration layer, update ``model.encoder.prompts`` directly; saved
encoder prompts are restored through Sentence Transformers' own persistence.
Per-call prompt/task arguments do not become saved defaults.

For backward compatibility, **a call with no task, prompt, or prompt_name uses
the original raw-text path**, even if the encoder has a default prompt. An
explicit name or prefix without a task applies that prompt without selecting
a role. An empty saved role prompt is a valid selection and does not fall through
to the default.

Pass raw texts: neembed delegates prompt application once to encoder
``preprocess(prompt=...)`` and retains its prompt-length metadata for modules
that exclude prompt tokens during pooling. It does not also concatenate a prefix
or attempt to infer whether caller text was already manually prefixed.

For training, use the normal module call with the same options:

.. code-block:: python

   model.train()
   query_embeddings = model(["What is a dog?"], task="query", prompt="query: ")
   document_embeddings = model(["A dog is a mammal."], task="document", prompt="passage: ")
   # Build a differentiable loss from these embeddings, then call backward().

This path calls encoder preprocessing and ``forward`` directly, preserving
gradients through the encoder, projection, and any learnable curvature. It never
calls encoder ``encode()``. Task is forwarded to both preprocessing and encoder
forward so task-aware modules can choose the same route in both stages.

The role-aware path is verified with Sentence Transformers 6.1 text encoders
and a local query/document Router with matching output dimensions. Encoders must
accept the selected prompt/task arguments in preprocessing and task in forward.
Unsupported custom signatures or missing routes report their errors; neembed
does not retry by dropping the task. Other Router layouts, unequal route output
dimensions, arbitrary task names, and multimodal/chat inputs are outside this
text-only contract. The existing role-free API still works with legacy encoder
signatures. Dependency-version qualification is tracked separately in Issue #181.

Built-in losses accept input-specific ``input_options`` through the same
differentiable forward path; see :doc:`retrieval_objectives` for training
examples. The ``rank()``, corpus search, evaluator, and mining convenience
APIs also accept optional input-specific ``input_options``; see :doc:`retrieval`
for their input names and a consistent training/search/mining example. Calls
without settings retain their original raw-text behavior.

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
