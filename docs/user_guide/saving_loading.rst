Saving and loading
==================

``ManifoldSentenceTransformer`` saves and reloads the local state required to
reconstruct either supported manifold configuration.

Save a model
------------

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
that flag and saves the current learned public curvature value. Reloading then
reconstructs the corresponding trainable Geoopt curvature state from that
value; no separate curvature checkpoint is required.

The same layout is used for Poincare and Lorentz models. Lorentz does not add a
separate geometry checkpoint because the Geoopt manifold is reconstructed from
the saved configuration rather than trained as an independent parameter set.

Load a model
------------

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
dimension. For Lorentz, the usual output contract also remains unchanged: an
intrinsic dimension :math:`D` is represented with :math:`D + 1` ambient
coordinates and the geometry path uses ``float64``.

The save/load helpers are local filesystem helpers. They do not add a separate
model registry or Hugging Face Hub integration.
