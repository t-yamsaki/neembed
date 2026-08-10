Saving and loading
==================

``ManifoldSentenceTransformer`` can save and reload the local state required by
neembed v0.1.

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
stores the manifold name, curvature, and configured projection dimension.
``projection.pt`` stores the projection state dictionary.

Load a model
------------

.. code-block:: python

   from neembed import ManifoldSentenceTransformer

   loaded = ManifoldSentenceTransformer.from_pretrained("./saved_model")

``from_pretrained()`` reconstructs the Sentence Transformer from the saved
``encoder/`` directory, rebuilds the configured Poincare manifold and optional
projection, and then restores the projection state.

The save/load helpers are local filesystem helpers in v0.1. They do not add a
separate model registry or Hugging Face Hub integration.
