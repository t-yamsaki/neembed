Public text API contracts
=========================

This guide describes the implemented text API; it is not an API freeze or a
promise of untested encoder/device compatibility. See :doc:`encoder_compatibility`
for the qualified dependency profiles and pinned MiniLM/E5 encoders.

Gradients, outputs, and mode
----------------------------

.. list-table:: Core calls
   :header-rows: 1
   :widths: 22 27 24 27

   * - Call
     - Gradients
     - Return
     - Model mode
   * - ``model(texts, ...)`` / ``forward``
     - Differentiable when grad is enabled
     - Geometry-valued Tensor batch
     - Follows caller's train/eval mode
   * - ``encode`` / ``encode_query`` / ``encode_document``
     - Always inference mode, no gradient graph
     - CPU NumPy array by default; Tensor with ``convert_to_tensor=True``
     - Switches to eval and stays there
   * - ``distance(a, b)``
     - Distance calculation uses no-grad
     - Tensor with broadcast batch shape; scalar for two points
     - Does not change mode
   * - ``rank`` / ``exact_corpus_search``
     - No-grad inference
     - Ranked Python dictionaries
     - Encoding leaves eval mode
   * - Evaluators / ``mine_hard_negatives``
     - No-grad inference
     - Metric dictionaries / ranked negatives
     - Restore previous mode, including on failure

``forward`` takes a batch of raw strings, such as ``["dog", "cat"]``. Use
``encode`` for a single string. For a differentiable training distance, use
``model.manifold.dist`` on forward outputs; the convenience ``distance`` helper
is an inference API. Losses and the composite objective follow the caller's
gradient context and mode, rather than entering inference mode themselves.

``encode`` returns a vector for string input (including an empty string) and a
matrix for sequence input. An empty sequence returns ``(0, output_width)``
without calling the encoder. ``batch_size`` defaults to 32 and must be a
positive Python integer, excluding booleans. It bounds each encoder call and
preserves input order; the full input and output still occupy memory. Invalid
batch size is rejected before changing mode; an encoder failure after encoding
starts leaves eval mode. Call ``model.train()`` explicitly when resuming training.

Geometry width and precision
----------------------------

``embedding_dim`` specifies the intrinsic projection width. Omitting it in a
single geometry preserves the encoder width. The model's ``embedding_dim``
property then reports that intrinsic width even when there is no projection.

.. list-table:: Coordinate contracts (``D`` = intrinsic dimension)
   :header-rows: 1
   :widths: 29 24 47

   * - Geometry
     - Output width
     - Mapping/output dtype policy
   * - Poincare / Euclidean
     - ``D``
     - Ordinary encoder/projection dtype
   * - Lorentz
     - ``D + 1``
     - Float64 mapping/output; first coordinate is time-like
   * - SphereProjection / Stereographic
     - ``D``
     - Float64 mapping/output
   * - Product
     - ``product_config.ambient_dim``
     - Shared float64 if any component is Lorentz, SphereProjection, or Stereographic;
       otherwise ordinary projection dtype

For a product, intrinsic width is the sum of component intrinsic dimensions.
Each Lorentz component adds one ambient coordinate. For example, Lorentz-2 plus
Euclidean-3 has projection width 5 and output width 6. Component scales affect
distances, not coordinate counts. Product construction accepts an omitted
``embedding_dim`` or one equal to the configured projection width.

Tensor outputs use the geometry device, which can differ from the encoder
device: SphereProjection/Stereographic and fixed-double products fall back to
CPU on MPS. NumPy outputs are moved to CPU. NumPy conversion must support the
output dtype; bfloat16 output requires Tensor conversion. The pretrained
qualification is Linux CPU with float32 encoder/projection weights. These
policies do not imply that all half-precision, CUDA, MPS, Router, or OS
combinations were validated. See :doc:`constant_curvature_semantics` and
:doc:`mixed_curvature` for the existing geometry policies.

Distance and ordering
---------------------

``distance`` accepts encoded points or real numeric array-like values with the
correct ambient width. Batch axes broadcast according to Geoopt/PyTorch rules.
It converts inputs to the configured geometry dtype/device; callers supply
valid manifold points. It does not turn raw encoder vectors into manifold
points or promise comprehensive validity checks for arbitrary inputs.

Ranking uses **ascending geodesic distance**: smaller means closer. Equal
distances preserve original candidate/corpus index order. Corpus search and
mining preserve query order; candidate text and caller-owned IDs remain unchanged.
These distances are not probability scores. :doc:`retrieval` describes exact
chunked search, relevance semantics, and exclusions; there is no ANN index here.

Roles and argument bundles
--------------------------

Role/prompt selection is shared by forward and encode: explicit ``prompt``
(including ``""``) or ``prompt_name`` takes precedence, then the role's saved
prompt, then the saved default. Providing both prompt and prompt name is invalid.
Document fallback names are ``document``, ``passage``, then ``corpus``.
No task/name/prompt means raw-text encoding, even with a saved default prompt.
No loss or evaluator automatically assigns a role to an omitted input option.

Configure a loss through its text argument names, such as ``anchors`` and
``positives``; configure search through ``queries`` and ``corpus``. Pair evaluators
use ``anchors``/``positives``, hierarchy evaluation uses ``texts``, and prototype
assignment uses ``sentences``. IDs, grades, depths, and prototype coordinates are
not text inputs. Binary relevance and mining exclusions accept non-string ID
iterables, including lists and sets; graded relevance maps IDs to finite,
non-negative real grades, with at least one positive grade per query.

The composite loss takes **outer positional argument bundles**, for example
``(anchors, positives)`` and ``(texts, depths)`` for ranking and depth components.
Their items may be text sequences or numeric tensors according to each component.
Configure each component's input options separately. With hierarchy weight zero,
``forward`` can omit hierarchy inputs and skips that component; ``component_losses``
always evaluates both unweighted components for diagnostics.

Runnable example and persistence
--------------------------------

The short example trains one step, performs role-aware exact search, and checks
a local save/restore using the same prefixes. Its English E5 texts are also used
in both READMEs; E5-small-v2 is not presented as a Japanese-language encoder.

.. code-block:: bash

   HF_HUB_DISABLE_XET=1 python scripts/preload_role_models.py
   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     python examples/v11_role_workflow.py --local-files-only

For a local text encoder, add ``--model-path /path/to/encoder``. The script uses
a temporary checkpoint directory. It is an API example, not a quality benchmark.

.. literalinclude:: ../../examples/v11_role_workflow.py
   :language: python
   :end-before: def main

``revision`` selects the remote encoder revision, preferably an immutable commit;
``local_files_only=True`` requires local/cached files. A checkpoint restored via
``from_pretrained`` uses its saved local encoder rather than refetching the base
model. New saves use ``format_version: 1`` independently of ProductConfig's
version, and store prompts, geometry, dtype, package version, and available source
provenance. Versionless v0.10-and-earlier configs remain loadable. Cache paths,
per-call input options/batch size, optimizer state, and external prototypes are
not implicit checkpoint defaults; float64 encoders include supplemental exact
tensor state. See :doc:`saving_loading` for the complete format and validation.
