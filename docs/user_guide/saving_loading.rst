Saving and loading
==================

``ManifoldSentenceTransformer`` saves and reloads the local state required to
reconstruct either supported sentence-model geometry. v0.4 keeps that helper
backward-compatible while making the boundary around external prototype state
explicit.

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

For fixed-curvature models the configuration format remains compatible with
v0.3. When ``learnable_curvature=True``, the configuration additionally records
that flag and saves the **current learned public curvature value**. Reloading
reconstructs the corresponding trainable geometry state from that value; no
separate curvature checkpoint is required.

The same layout is used for Poincare and Lorentz models. Lorentz does not add a
separate geometry checkpoint because the Geoopt manifold is reconstructed from
the saved public curvature. The public value remains the positive magnitude of
negative sectional curvature even though Geoopt internally uses ``k = 1 / c``
for Lorentz geometry.

Load a sentence model
---------------------

.. code-block:: python

   from neembed import ManifoldSentenceTransformer

   loaded = ManifoldSentenceTransformer.from_pretrained("./saved_model")

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
prototype state:

.. code-block:: python

   import torch

   from neembed import ManifoldPrototypes, ManifoldSentenceTransformer

   loaded_model = ManifoldSentenceTransformer.from_pretrained("./saved_model")
   loaded_prototypes = ManifoldPrototypes(
       loaded_model,
       num_prototypes=len(prototype_ids),
   )
   loaded_prototypes.load_state_dict(
       torch.load("./saved_model/prototypes.pt", weights_only=True)
   )

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
