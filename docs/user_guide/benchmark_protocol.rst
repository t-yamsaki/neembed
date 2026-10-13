Reproducible real-data benchmark protocol
=========================================

v0.11 defines the comparison and recording protocol for the first v0.12 real-data
experiments. It does **not** publish real-data scores or provide a general
experiment service. The existing ``experiments/compare_euclidean_poincare.py``
tiny regression remains unchanged. Its one seed and tiny held-out task are not
substitutes for the protocol below.

Comparison plan
---------------

Use one pinned Sentence Transformer encoder artifact, dataset preparation and
training schedule for all variants. The primary experiment fine-tunes the native
encoder as well as the projected variants. The original, unmodified pretrained
Sentence Transformer is an additional zero-shot reference, clearly labelled as
untrained; do not imply it received the fine-tuning budget.

.. list-table:: Planned trained comparisons
   :header-rows: 1
   :widths: 25 20 15 15 25

   * - Variant
     - Distance (ascending)
     - Intrinsic width
     - Output width
     - Embedding dtype
   * - Native Sentence Transformer, no projection
     - Cosine
     - Encoder width
     - Encoder width
     - Encoder dtype
   * - Euclidean projection
     - Cosine
     - D
     - D
     - float32 initially
   * - Euclidean projection
     - Euclidean L2
     - D
     - D
     - float32 initially
   * - Poincare projection
     - Geodesic
     - D
     - D
     - float32 initially
   * - Lorentz projection
     - Geodesic
     - D
     - D + 1
     - float64

The four projected variants use the same intrinsic D. Lorentz's time coordinate
is representation overhead, not an extra learned projection dimension. Record
native width separately; the native model is a useful baseline but has a
different output capacity. Initially use fixed curvature magnitude 1.0 for both
hyperbolic geometries; curvature learning and mixed-curvature products are
outside this first comparison. The dtype difference follows neembed's existing
numerical policy and must be disclosed. Native cosine and projected cosine use
``1 - cosine_similarity`` for ascending ranking; L2 uses the unnormalized
projected vectors. Do not silently substitute cosine for geodesic distance.

Before seeing test results, freeze a v1 configuration containing at least three
unique seeds, e.g. ``[0, 1, 2]``, the ordered trial lists and all partition IDs.
Every trained variant gets the same number of validation trials per seed, epochs,
max optimizer steps, batch size, AdamW learning rate and weight decay. Stop at the
first of the epoch or step limits, recording actual steps. Different distance
scales can require different temperature or margin values: use geometry-specific
predeclared grids with equal numbers of trials, rather than forcing one numerical
temperature/margin across geometries. v1 supports MNRL (``temperature``) or triplet
(``margin``) plans; the objective is shared within a comparison. A changed grid,
objective or training policy constitutes a new configuration and result set.

Restore identical initial encoder weights for **every** seed/variant/trial,
recording a SHA256 of the same canonical local snapshot artifact. Never reuse a
fine-tuned encoder from a preceding trial. Reset the seed immediately before model
construction, use the same projection initialization algorithm and RNG order for
projected variants, and use the same seeded batch order. The native architecture
has no projection. The primary plan uses ``encoder_policy=trainable`` for all
trained variants. A separate frozen-encoder experiment freezes the encoder for
all projected variants; its native model is an untrained reference with zero
optimizer steps and is not described as a tuned trainable baseline. Do not mix
frozen and trainable results in one comparison claim.

Data and input roles
--------------------

Record dataset source, revision, license, preprocessing/split policy, each
partition's ID, content SHA256, ordered example IDs and size. Hash the prepared
content (including text, relevance/labels and candidate IDs), not only filenames.
The v1 small benchmark format explicitly lists example IDs; validate actual bytes
and ordered IDs in the dataset adapter. Use separate train, validation and test
examples; validation/test query or hierarchy-entity leakage must be checked during
data preparation, including duplicates with different IDs. A shared retrieval
candidate corpus is permissible if it matches the task definition and is
recorded. ID disjointness alone does not prove text/entity disjointness.

``inputs.queries`` and ``inputs.documents`` each record ``task``, ``prompt_name``
and ``prompt``; use null values for the legacy raw-text path. Explicit prompts and
named prompts are exclusive. Use the same effective prompts for every variant,
including the native baseline, at training and evaluation. Resolve named prompts
from the pinned encoder snapshot, not an evolving remote default. Map these slots
to loss ``anchors``/``positives`` and retrieval ``queries``/``corpus`` as appropriate;
IDs and relevance labels are never prompt-bearing text. Do not add a prefix twice.
For differentiable native training use the encoder's preprocessing/forward path,
not its no-grad ``encode`` method. See :doc:`api_contracts` for role precedence.

Runner and result format
------------------------

``experiments/benchmark_protocol.py`` validates the versioned JSON plan and
implements a small two-stage adapter interface:

* ``fit_validate(context) -> TrialResult`` receives model/input/training settings,
  a seed, variant, trial parameters, train and validation partition metadata.
  The runner does not pass test partition metadata into this call.
* ``test(selected_artifact, test_split) -> metrics`` runs once per seed/variant
  after selecting the best finite validation primary score. Higher/lower is
  controlled by ``selection.direction``; ties select the first configured trial.

A real adapter must load and verify the pinned encoder/data bytes, apply the
recorded prompts, train a fresh model, and return an independent artifact or
checkpoint for each trial. Retaining one mutable model object across trials can
invalidate the selected state. The adapter must not access test data during
training/tuning. The runner controls its calls; it cannot prevent arbitrary code
inside an adapter from reading external data. No real-data adapter is implemented
by this protocol task.

Each trial reports ``validation`` metrics, optional finite/null raw ``diagnostics``
and ``observed`` execution facts. Observed fields are the initial
``model_snapshot_sha256``, ``encoder_trainable``, ``intrinsic_dim``, ``output_dim``,
``encoder_dtype``, ``embedding_dtype``, actual geometry ``device``,
``optimizer_steps``, ``hardware`` mapping and a ``note`` (string or null). Real
adapters must compute these facts, not copy them from the plan. Mismatches in
initial state, policy, dimensions, dtypes or device fail the trial; exceeding the
step budget also fails. Model ID/revision remain in the embedded configuration.
Frozen native references must run zero steps. Every other real variant must run
at least one optimizer step, including projected frozen models that train their
heads. Data hashes and license/preprocessing claims are adapter/caller
responsibilities, not automatically verified by the common runner.

``seed-N.json`` is self-contained, with these top-level fields:

.. list-table:: Result v1
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``format_version``
     - Integer 1; independent of the neembed saved-model version
   * - ``config``, ``config_sha256``
     - Full validated plan and hash of canonical JSON
   * - ``seed``, ``backend``
     - Seed, adapter name and explicit ``fixture`` or ``real`` kind
   * - ``environment``
     - Python, installed package versions (null if absent), CPU/platform and
       accelerator inventory, source commit/dirty flag and runner source hash
   * - ``runs``
     - One record per variant with every indexed trial, parameters, validation
       metrics, diagnostics, observations, selected trial index and final test

A trial is ``completed`` or ``partial`` after successful validation; ``partial``
means some requested metrics are unmeasured. A failed trial has an explicit
``error`` with stage, exception type and message; its scores remain null. A final
run is ``completed``, ``partial``, ``failed`` or ``unmeasured``. If all primary
validation scores are null, no trial wins and test is not called. A test failure
is recorded without switching to another trial. Missing metrics are JSON ``null``,
never zero; finite zero scores remain valid measurements. NaN, infinity and
boolean metrics fail validation. Existing seed result files are refused, and
ordinary adapter exceptions do not prevent recording the remaining seeds.
Process termination or a write failure can leave an incomplete result file; only
fully parseable, complete seed files are eligible for analysis.

Report every seed and failure/unmeasured counts. Mean and standard deviation, if
computed downstream, must disclose the number of measured seeds and use only
finite measurements. Never choose a seed or variant based on test scores. Compare
retrieval metrics such as nDCG@K, MRR and recall at shared K and relevance semantics;
raw training losses, mean distances and margins are geometry-specific diagnostics,
not comparable universal quality scores or selection criteria across geometries.

The runner resets Python, NumPy and PyTorch RNGs where installed and requests
PyTorch deterministic algorithms according to configuration. Record actual
hardware per trial; hardware/library changes need not produce bit-identical
results. CUDA may require ``CUBLAS_WORKSPACE_CONFIG`` before process startup;
unsupported deterministic operations fail explicitly. Installed versions and
hashes aid reproduction but are not a lockfile or proof of backend compliance.

Offline protocol fixture
------------------------

From the repository root, run:

.. code-block:: bash

   python experiments/run_benchmark.py validate experiments/configs/protocol_fixture.json
   python experiments/run_benchmark.py run-fixture experiments/configs/protocol_fixture.json \
       --output-dir /tmp/neembed-protocol-fixture

The checked-in JSON is a runnable **fixture plan**, with five declared variants,
three seeds and two trials each. It is not a real-model configuration. Its adapter
verifies local partition hashes and performs lexical-overlap ranking to exercise
selection and output. It executes no encoder, projection, geometry or optimizer:
observed widths/dtypes/model identity and raw loss/distance are null, while actual
optimizer steps are zero. ``fixture_recall_at_1`` is a fixture score; the explicit
``unmeasured_example`` metric stays null. Roles and numerical model settings in
this fixture are planned metadata, not measured execution facts.

For a v0.12 run, prepare real artifacts and their hashes, select real metrics and
audited split policies, create a ``kind=real`` plan, and supply a real adapter to
``run_protocol``. The fixture CLI deliberately cannot execute such a plan.
