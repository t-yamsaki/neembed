Evaluation
==========

v0.2 adds a small evaluation workflow around the existing training path:

.. code-block:: text

   train
     ↓
   evaluate
     ↓
   compare

The goal is to make geometry-aware sentence embeddings measurable without
introducing a separate metrics framework. ``ManifoldEmbeddingEvaluator`` uses
the model's own ``encode()`` and ``distance()`` methods, so evaluation follows
the same manifold geometry used for inference.

Aligned retrieval task
----------------------

The evaluator receives aligned anchor-positive pairs. For each anchor, every
positive is treated as a retrieval candidate. The positive at the same index is
the correct target; all off-diagonal candidates are negatives.

At least two aligned pairs are required because the negative-distance metric is
undefined for a single pair.

.. code-block:: python

   from neembed import ManifoldEmbeddingEvaluator

   evaluator = ManifoldEmbeddingEvaluator(
       model=model,
       anchors=["Shiba Inu", "Siamese cat", "sparrow"],
       positives=["dog", "cat", "bird"],
   )

   metrics = evaluator()
   print(metrics)

The evaluator runs without gradient tracking and restores the model's original
training/evaluation mode before returning.

Metrics
-------

``retrieval_accuracy``
   Fraction of anchors whose nearest positive candidate is the aligned target.
   Higher is better. The value is between 0 and 1.

``mean_positive_distance``
   Mean distance of aligned anchor-positive pairs, i.e. the diagonal of the
   pairwise distance matrix. Lower means aligned pairs are closer under the
   model's geometry.

``mean_negative_distance``
   Mean distance across all off-diagonal anchor-candidate pairs. Higher means
   the non-matching candidates are farther away on average.

Distance magnitudes are geometry- and configuration-dependent. In particular,
curvature and the learned representation can change the scale of Poincare
geodesic distances. Compare distance values only when the evaluation setup is
meaningfully comparable; do not treat their absolute scale as a universal
quality score.

Validation during training
--------------------------

Pass an evaluator to ``ManifoldTrainer.fit()`` to run validation once after
each completed epoch:

.. code-block:: python

   history = trainer.fit(
       train_batches,
       epochs=3,
       evaluator=evaluator,
   )

With validation enabled, each history entry contains the epoch's mean
``train_loss`` and a nested ``validation`` mapping with the three evaluator
metrics. Without an evaluator, ``fit()`` keeps the original ``list[float]``
return shape.

See :doc:`training` for the training contract and optimizer behavior, and
:class:`neembed.ManifoldTrainer` for the generated API details.

PyTorch DataLoader interoperability
-----------------------------------

``ManifoldTrainer`` accepts any re-iterable input that yields
``(anchors, positives)`` batches. An ordinary ``torch.utils.data.DataLoader``
therefore works directly; neembed does not add a Dataset, DataModule, or custom
collator abstraction.

The runnable `DataLoader example
<https://github.com/t-yamsaki/neembed/blob/main/examples/train_dataloader.py>`_
shows default PyTorch collation, multi-epoch training, and epoch-end validation.
Keep positive candidates unique within each batch because the ranking loss uses
off-diagonal positives as in-batch negatives.

Euclidean vs Poincare benchmark
-------------------------------

The repository includes a deterministic, tiny hierarchy-retrieval benchmark:

.. code-block:: bash

   python experiments/compare_euclidean_poincare.py

The benchmark fine-tunes Euclidean and Poincare variants under matched data and
training conditions, then reports the shared v0.2 metrics plus
``final_training_loss`` for each variant. The JSON output also records the seed,
model name, task pairs, and key hyperparameters needed to reproduce the run.

See the `experiment documentation
<https://github.com/t-yamsaki/neembed/blob/main/experiments/README.md>`_ for the
fixed configuration and output contract.

This benchmark is an engineering and regression reference, not evidence that
one geometry is generally superior. It uses a tiny controlled task, does not
perform hyperparameter search, and should not be interpreted as a research
leaderboard result.

API reference
-------------

For constructor and return-value details generated directly from docstrings,
see :class:`neembed.ManifoldEmbeddingEvaluator`.
