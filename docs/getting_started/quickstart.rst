Quick start
===========

This example fine-tunes a pretrained Sentence Transformer on a tiny hierarchy
using Poincare-ball embeddings, then evaluates aligned retrieval pairs.

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

* Read :doc:`../user_guide/architecture` for the Euclidean-to-Poincare mapping.
* Read :doc:`../user_guide/training` for the geodesic ranking objective and
  optimizer behavior.
* Read :doc:`../user_guide/evaluation` for metric definitions, epoch validation,
  DataLoader interoperability, and the Euclidean-vs-Poincare benchmark.
* Read :doc:`../user_guide/inference` for ``encode()`` and ``distance()``.
* Read :doc:`../user_guide/saving_loading` for local model persistence.

The repository also contains complete runnable examples in
``examples/train_poincare.py`` and ``examples/train_dataloader.py``.
