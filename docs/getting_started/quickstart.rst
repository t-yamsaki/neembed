Quick start
===========

This example fine-tunes a pretrained Sentence Transformer on a tiny hierarchy
using Poincare-ball embeddings, then evaluates aligned retrieval pairs. The
same public workflow also supports ``manifold="lorentz"``.

.. code-block:: python

   from neembed import (
       ManifoldEmbeddingEvaluator,
       ManifoldSentenceTransformer,
       ManifoldMultipleNegativesRankingLoss,
       ManifoldTrainer,
   )

   model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       manifold="poincare",
       embedding_dim=64,
       curvature=1.0,
   )

   loss = ManifoldMultipleNegativesRankingLoss(
       model=model,
       temperature=0.1,
   )

   trainer = ManifoldTrainer(
       model=model,
       loss=loss,
   )

   train_batches = [
       (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
       (["dog", "cat"], ["mammal", "feline"]),
   ]

   evaluator = ManifoldEmbeddingEvaluator(
       model=model,
       anchors=["Shiba Inu", "Siamese cat"],
       positives=["dog", "cat"],
   )

   history = trainer.fit(
       train_batches,
       epochs=1,
       evaluator=evaluator,
   )
   print(history[-1]["validation"])

   embeddings = model.encode([
       "Shiba Inu",
       "dog",
       "mammal",
   ])

   distance = model.distance(embeddings[0], embeddings[1])
   print(float(distance))

Choosing Poincare or Lorentz
----------------------------

Select the geometry through the model constructor:

.. code-block:: python

   poincare_model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       manifold="poincare",
       embedding_dim=64,
       curvature=1.0,
   )

   lorentz_model = ManifoldSentenceTransformer(
       "sentence-transformers/all-MiniLM-L6-v2",
       manifold="lorentz",
       embedding_dim=64,
       curvature=1.0,
   )

``embedding_dim`` is the intrinsic projected dimension in both cases. A
Poincare embedding therefore has 64 coordinates, while the equivalent Lorentz
hyperboloid representation has 65 ambient coordinates because it includes one
additional time-like coordinate. Lorentz manifold operations and outputs use
``float64`` for numerical stability; the pretrained encoder and ordinary
Euclidean projection parameters keep their normal dtype.

The public ``curvature`` argument has the same meaning for both geometries: it
is the positive magnitude of the negative sectional curvature. See
:doc:`../user_guide/architecture` for the Geoopt parameter mapping and geometry
behavior.

For a complete runnable Lorentz train/evaluate/inference example:

.. code-block:: bash

   python examples/train_lorentz.py

Batch semantics
---------------

``ManifoldMultipleNegativesRankingLoss`` treats the positive at the same batch
index as the target for each anchor. Every off-diagonal positive candidate is
used as an in-batch negative.

For that reason, avoid duplicate positives within the same batch. A batch such
as ``(["dog", "cat"], ["mammal", "mammal"])`` would give contradictory
supervision because one anchor's positive would simultaneously be treated as
the other anchor's negative.

Next steps
----------

* Read :doc:`../user_guide/architecture` for manifold selection, curvature,
  dimensions, and Lorentz precision behavior.
* Read :doc:`../user_guide/training` for the geodesic ranking objective,
  optimizer behavior, and runnable Lorentz example.
* Read :doc:`../user_guide/evaluation` for metric definitions, epoch validation,
  DataLoader interoperability, and the Euclidean/Poincare/Lorentz benchmark.
* Read :doc:`../user_guide/inference` for ``encode()`` and ``distance()``.
* Read :doc:`../user_guide/saving_loading` for local model persistence across
  both supported manifolds.

The repository also contains complete runnable examples in
``examples/train_poincare.py``, ``examples/train_lorentz.py``, and
``examples/train_dataloader.py``.
