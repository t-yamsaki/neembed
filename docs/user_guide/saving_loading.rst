Saving and loading
==================

``ManifoldSentenceTransformer`` saves and reloads the local state required to
reconstruct the configured sentence-model geometry. v0.4 keeps that helper
backward-compatible while making the boundary around external prototype state
explicit.

v0.10 products use this same directory layout and persist their normalized
ordered component metadata, including fixed scales. See :doc:`mixed_curvature`
for product save/load and hierarchy-selector reconstruction.

Encoder loading options
-----------------------

The constructor accepts four keyword-only options forwarded to Sentence
Transformers: ``revision``, ``local_files_only``, ``cache_folder``, and ``device``.
Use an encoder repository commit ID for ``revision`` when reproducible remote
loading matters; a branch or tag can change. For example, replace the revision
placeholder with the desired encoder commit ID:

.. code-block:: python

   from neembed import ManifoldSentenceTransformer

   model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       revision="<encoder-commit-id>",
       cache_folder="./encoder_cache",
       device="cpu",
       embedding_dim=32,
   )

Set ``local_files_only=True`` to load a local encoder directory or an already
cached encoder without downloading missing files. If the necessary files are
missing, Sentence Transformers reports the loading failure. Unspecified options
retain upstream defaults, including automatic encoder device selection and
normal download behavior. neembed does not enable ``trust_remote_code``.
Projection and geometry placement continue to follow the existing encoder
device and geometry dtype/device policies.

These options control the current load. New saves record available base-model
ID/revision provenance in ``neembed_config.json``, but never a dictionary of
loading options, cache paths, device selection, or credentials. The encoder
itself is still saved into ``encoder/``. Source provenance is descriptive and is
not used to fetch or substitute a remote encoder during restoration.

Save a sentence model
---------------------

.. code-block:: python

   model.save_pretrained("./saved_model")

The target directory contains:

.. code-block:: text

   saved_model/
   ├── encoder/
   ├── neembed_config.json
   └── projection.pt

``encoder/`` is written by Sentence Transformers. ``neembed_config.json``
stores the configured ``manifold`` name, current public ``curvature`` magnitude,
and projection dimension. ``projection.pt`` stores the projection state
dictionary.

Versionless configurations from v0.10 and earlier remain loadable. When
``learnable_curvature=True``, the configuration additionally records
that flag and saves the **current learned public curvature value**. Reloading
reconstructs the corresponding trainable geometry state from that value; no
separate curvature checkpoint is required.

The same layout is used for Poincare and Lorentz models. Lorentz does not add a
separate geometry checkpoint because the Geoopt manifold is reconstructed from
the saved public curvature. The public value remains the positive magnitude of
negative sectional curvature even though Geoopt internally uses ``k = 1 / c``
for Lorentz geometry.

Versioned metadata
------------------

New saves use ``format_version: 1`` for the complete ``neembed_config.json``.
This is independent of the nested ``product_config.version`` and the package's
``neembed_version``. Existing geometry field names and meanings are retained.
For example:

.. code-block:: json

   {
     "format_version": 1,
     "neembed_version": "0.10.0",
     "embedding_dim": 2,
     "manifold": "poincare",
     "curvature": 2.0,
     "input_config": {
       "prompts": {"query": "query: ", "passage": "passage: "},
       "default_prompt_name": null
     },
     "dtypes": {
       "encoder": "float32",
       "projection": "float32",
       "geometry": "float32"
     },
     "base_model": {"model_id": "org/model", "revision": "main"}
   }

``input_config`` snapshots the encoder's named prompts and default prompt name,
including explicitly empty prefixes. Restoration applies this snapshot to the
saved encoder. Query/document role selection continues to follow the shared
forward/encode rules; an unspecified role still takes the original raw-text
path. Per-call task/prompt arguments, loss/evaluator ``input_options``, and
``batch_size`` are not saved as implicit defaults.

``dtypes`` records the uniform floating-state dtype of the encoder, projection,
and geometry module. Supported values are ``float16``, ``bfloat16``, ``float32``,
and ``float64``; a module without floating state uses ``null``. The loader
restores these dtypes before copying projection weights, avoiding a cast through
the default projection dtype. Mixed floating dtypes within one module are
rejected at save time because this format cannot represent them. Geometry output
rules still apply: Lorentz adds its ambient coordinate and maps in ``float64``;
SphereProjection/Stereographic and products containing fixed-double components
retain their existing dtype/device policies. Geometry state dtype is distinct
from output width and output dtype. Device selection remains a load-time choice.

``base_model`` records a Hub-style ID and the available matching encoder revision
or the requested revision. Local paths and credential-bearing URLs are omitted;
unknown values are ``null``. The original provenance survives loading and
re-saving a checkpoint. A recorded branch/tag identifies the requested source;
only a known commit identifies an immutable revision. ``neembed_version`` records
the package that wrote the checkpoint, not a minimum required package version.

A config without ``format_version`` follows the legacy geometry-only path and
leaves encoder prompt/dtype restoration to its saved local files. Reading does
not rewrite that config; saving the restored model writes the new format.
Eight immutable fixture records produced by the v0.10.0 saver at commit
``7d4d55de113d7950ff05f3cb65f2806c2bbdbd85`` verify legacy embeddings and distances
for single, learnable-curvature, and product models.

Unknown format versions, missing fields, conflicting geometry fields, invalid
numeric values, unknown dtype names, malformed prompts, and duplicate JSON keys
raise clear ``ValueError`` exceptions. Metadata is checked before encoder loading;
projection checkpoint shape/dtype is checked during restoration. The nested
product schema continues to validate component ordering, dimensions, and scales.

Load a sentence model
---------------------

.. code-block:: python

   from neembed import ManifoldSentenceTransformer

   loaded = ManifoldSentenceTransformer.from_pretrained("./saved_model")

The same loading options can be supplied when restoring a local neembed model:

.. code-block:: python

   loaded = ManifoldSentenceTransformer.from_pretrained(
       "./saved_model",
       local_files_only=True,
       cache_folder="./encoder_cache",
       device="cpu",
   )

Here they apply to the saved ``encoder/`` directory. ``revision`` is also accepted
and forwarded, but is normally omitted for local restoration: it does not select
a revision of the neembed checkpoint or replace the saved encoder with a remote
model. The caller chooses the checkpoint directory through ``model_path``.

``from_pretrained()`` reconstructs the Sentence Transformer from the saved
``encoder/`` directory, rebuilds the configured Poincare or Lorentz manifold,
recreates the optional projection, and then restores the projection state.
Older configurations that do not contain ``learnable_curvature`` are treated as
fixed-curvature models.

The loaded model therefore preserves the saved manifold choice, current public
curvature magnitude, curvature trainability, and intrinsic projection
dimension. For Lorentz, the output contract also remains unchanged: intrinsic
dimension :math:`D` is represented with :math:`D + 1` ambient coordinates and
the geometry path uses ``float64``.

Save trainable prototypes separately
------------------------------------

``ManifoldPrototypes`` is an opt-in module external to
``ManifoldSentenceTransformer``. Its manifold-valued coordinates are therefore
**not** automatically included by ``model.save_pretrained()``. Save its
``state_dict()`` separately:

.. code-block:: python

   import torch

   model.save_pretrained("./saved_model")
   torch.save(prototypes.state_dict(), "./saved_model/prototypes.pt")

To restore the full structure, reload the sentence model first, construct a
compatible prototype module on that reconstructed manifold, and then load the
prototype state onto the reconstructed model device:

.. code-block:: python

   import torch

   from neembed import ManifoldPrototypes, ManifoldSentenceTransformer

   loaded_model = ManifoldSentenceTransformer.from_pretrained("./saved_model")
   loaded_prototypes = ManifoldPrototypes(
       loaded_model,
       num_prototypes=len(prototype_ids),
   )
   device = next(loaded_model.parameters()).device
   loaded_prototypes.load_state_dict(
       torch.load(
           "./saved_model/prototypes.pt",
           map_location=device,
           weights_only=True,
       )
   )

Using ``map_location`` keeps this example portable when a checkpoint saved on
one device (for example CUDA) is restored on another device (for example CPU).
The prototype count, intrinsic dimension, manifold type, and learned curvature
must be compatible with the saved coordinates. Reloading the model first is
especially important when curvature was learned jointly with prototypes,
because the prototype module must attach to the reconstructed current manifold.

Caller-owned structure is not serialized automatically
------------------------------------------------------

Prototype identifiers, ``parent_relations``, hierarchy-loss hyperparameters,
training batches, and optimizer state are caller-owned configuration. Recreate
those explicitly when resuming training. ``ManifoldPrototypeHierarchyLoss``
keeps those values as ordinary Python structure rather than introducing a graph
checkpoint format.

Likewise, ``ManifoldTrainer`` does not save optimizer state. If exact optimizer
resume semantics matter, persist the caller-owned optimizer state separately
using ordinary PyTorch/Geoopt mechanisms.

The save/load helpers remain local filesystem helpers. They do not add a model
registry or Hugging Face Hub integration. See :doc:`learnable_structure` for the
v0.4 parameter categories and joint-optimization contract.
