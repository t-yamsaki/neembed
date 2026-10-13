Encoder and dependency compatibility
====================================

The role-aware text APIs use Sentence Transformers' public ``preprocess()`` and
``get_embedding_dimension()`` interfaces. These appeared with the v5.4 shared
base-package changes; v5.3 has neither interface on ``SentenceTransformer``.
neembed's supported floor is the patched v5.4.1 release, rather than v5.4.0.
See the upstream `migration guide <https://sbert.net/docs/migration_guide.html#migrating-from-v5-x-to-v5-4>`_
and `v5.4.1 release notes <https://github.com/huggingface/sentence-transformers/releases/tag/v5.4.1>`_.

Qualified dependency profiles
-----------------------------

The small ``Encoder compatibility`` CI matrix runs the complete unit suite and
the pinned real-model gate with these two profiles. Exact CI constraints live in
``tests/compatibility/minimum.txt`` and ``tests/compatibility/current.txt``.

.. list-table:: Linux CPU qualification (2026-10-13)
   :header-rows: 1
   :widths: 22 39 39

   * - Dependency
     - Minimum profile
     - Current profile
   * - Python
     - 3.10
     - 3.12
   * - PyTorch
     - 2.3.0 (CPU wheel)
     - 2.14.1 (CPU wheel)
   * - Sentence Transformers
     - 5.4.1
     - 6.1.0
   * - Transformers
     - 4.41.0
     - 5.19.0
   * - Geoopt
     - 0.5.1
     - 0.5.1
   * - NumPy / Hub
     - 1.26.4 / 0.23.0
     - Resolved by the current profile; logged by the gate

Package lower bounds are ``torch>=2.3.0``, ``sentence-transformers>=5.4.1``, and
``geoopt>=0.5.1``. Existing prototype parameter transfers use
``torch.utils.swap_tensors``, introduced in `PyTorch 2.3 <https://docs.pytorch.org/tutorials/recipes/recipes/swap_tensors.html>`_.
Testing 2.2 reproduced four failures in that contract; those tests remain active
in the minimum-profile unit suite. Geoopt's floor is the qualified lower bound
for neembed's complete stack, not an audit of every earlier Geoopt release.
Sentence Transformers supplies its own Transformers/Hub requirements; v6
raises those floors to Transformers 5 and Hub 1.3. No additional upper bounds
are imposed by neembed.

The matrix qualifies these combinations, not every cross-product of dependency
versions. The minimum profile pins NumPy 1.26.4 because PyTorch 2.3's NumPy bridge
predates NumPy 2; use the complete constraints file to reproduce that profile.
The ordinary Python 3.10/3.11/3.12 unit CI additionally checks the resolver's
current dependencies with Hub access disabled during test execution.

Pinned encoder qualification
----------------------------

``tests/integration/role_models.json`` is the shared manifest used by both the
preloader and the tests:

.. list-table:: Text encoders
   :header-rows: 1
   :widths: 35 65

   * - Encoder
     - Immutable revision
   * - sentence-transformers/all-MiniLM-L6-v2
     - ``1110a243fdf4706b3f48f1d95db1a4f5529b4d41``
   * - intfloat/e5-small-v2
     - ``ffb93f3bd4047442299a41ebb6fa998a38507c52``

MiniLM uses empty role prompts. E5 uses ``"query: "`` for queries and ``"passage: "``
for documents, as specified in the `E5 model card <https://huggingface.co/intfloat/e5-small-v2>`_.
Configure these explicitly; neembed does not infer prefixes from model IDs:

.. code-block:: python

   model.encoder.prompts.update({"query": "query: ", "document": "passage: "})
   queries = model.encode_query(query_texts, batch_size=16)
   documents = model.encode_document(document_texts, batch_size=16)

Pass unprefixed text so the instruction is inserted once. The gate inspects the
actual token IDs against tokenizing the expected prefix plus text, and checks
that E5's prefixed IDs differ from raw input. The same role settings are passed
to the differentiable loss, exact search, corpus evaluator, and negative miner.
Distances remain Geoopt geodesic distances; the tests do not replace them with
cosine scores or assert that one geometry outperforms another.

For each family, the gate covers Poincare (16 coordinates) and a scaled
Lorentz-8 / Euclidean-8 product (17 packed coordinates):

* Bounded query/document encoding agrees with the full shared forward path.
* One AdamW step has finite, active encoder and projection gradients and changes
  both parameter groups.
* Chunked search agrees with a full geodesic distance matrix, evaluation returns
  valid metrics, and mining excludes known positives.
* Saving and restoring the local encoder preserves role embeddings, distances,
  ranking, prompt settings, and pinned source provenance. Restoration explicitly
  rejects socket connections.

Running the gate locally
------------------------

Unit tests use local fixtures and fake encoders and do not require model
downloads. Real-model tests are opt-in and need an initial internet connection
to retrieve the two public checkpoints. A CPU is sufficient; this gate does not
require CUDA or other accelerators.

.. code-block:: bash

   # Install either the minimum or current qualification profile.
   python -m pip install torch -c tests/compatibility/current.txt --index-url https://download.pytorch.org/whl/cpu
   python -m pip install -e ".[dev]" -c tests/compatibility/current.txt
   python -m pip check

   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q

   # This is the network-enabled stage; uses the normal Hugging Face cache.
   HF_HUB_DISABLE_XET=1 python scripts/preload_role_models.py

   # After preloading, the complete role gate runs offline.
   NEEMBED_REAL_STACK=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python -m pytest -vv -s tests/integration/test_real_stack_v11.py

Set ``HF_HUB_CACHE`` consistently during both preloading and offline execution
if using a custom cache. Missing pinned files fail the opt-in gate rather than
silently downloading another revision or skipping a failed encoder.

The pretrained qualification above covers Linux CPU with float32 encoder and
projection weights, float32 Poincare output, and float64 product geometry.
Existing local tests exercise a small equal-output-width query/document Router
and selected geometry dtype policies. They do not qualify arbitrary Router
layouts, every encoder family, CUDA/MPS training, Windows/macOS, quantization,
mixed precision, or all dtype/device combinations. Those configurations are not
claimed as tested by this matrix.
