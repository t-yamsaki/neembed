Evaluation
==========

v0.2 introduced a small evaluation workflow around the training path, and the
same evaluator now works with both supported hyperbolic manifolds:

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
       recall_at_k=(1, 2, 3),
   )

   metrics = evaluator()
   print(metrics)

The evaluator runs without gradient tracking and restores the model's original
training/evaluation mode before returning. Candidate ranks use ascending
manifold geodesic distance. The default ``recall_at_k=(1,)`` adds only
``recall_at_1``; pass other positive integer cutoffs to request additional
Recall@K values.

Metrics
-------

``retrieval_accuracy``
   Fraction of anchors whose nearest positive candidate is the aligned target.
   Higher is better. The value is between 0 and 1. Under the aligned retrieval
   contract this is the same quantity as Recall@1.

``recall_at_<K>``
   Fraction of anchors whose aligned target appears among the K nearest
   candidates. Higher is better. One key is returned for every configured
   ``recall_at_k`` cutoff. If K is greater than or equal to the number of
   candidates, Recall@K is 1 because every aligned target is present in the
   candidate pool.

``mrr``
   Mean reciprocal rank of the aligned target. For target ranks
   :math:`r_1, \ldots, r_N`, MRR is

   .. math::

      \mathrm{MRR} = \frac{1}{N}\sum_{i=1}^{N}\frac{1}{r_i}.

   Higher is better. MRR is 1 only when every aligned target is ranked first.

``mean_positive_distance``
   Mean distance of aligned anchor-positive pairs, i.e. the diagonal of the
   pairwise distance matrix. Lower means aligned pairs are closer under the
   model's geometry.

``mean_negative_distance``
   Mean distance across all off-diagonal anchor-candidate pairs. Higher means
   the non-matching candidates are farther away on average.

Distance magnitudes are geometry- and configuration-dependent. Curvature,
coordinate model, and the learned representation can all change the distance
scale. Compare raw distance values only when the evaluation setup is
meaningfully comparable; do not treat their absolute scale as a universal
quality score across cosine, Poincare, and Lorentz geometry.

Prototype assignment evaluation
-------------------------------

``ManifoldPrototypeAssignmentEvaluator`` measures whether sentence embeddings
are assigned to the intended learned prototype IDs by nearest geodesic distance.
Prototype identifiers remain ordinary caller-owned metadata aligned to prototype
row indices; they are not stored inside the Geoopt manifold parameter.

.. code-block:: python

   from neembed import ManifoldPrototypeAssignmentEvaluator

   evaluator = ManifoldPrototypeAssignmentEvaluator(
       model=model,
       prototypes=prototypes,
       prototype_ids=("animal", "dog", "cat"),
       sentences=("Shiba Inu", "Siamese cat", "living creature"),
       expected_prototype_ids=("dog", "cat", "animal"),
   )

   metrics = evaluator()
   print(metrics)

The evaluator returns two metrics:

``assignment_accuracy``
   Fraction of sentences whose nearest prototype has the expected caller-owned
   ID. Higher is better.

``mean_assigned_prototype_distance``
   Mean geodesic distance from each sentence embedding to the prototype selected
   by nearest-distance assignment. This is the distance to the predicted
   prototype, even when the predicted ID is incorrect.

Evaluation uses :class:`neembed.ManifoldPrototypes` directly, so Poincare and
Lorentz paths use their configured Geoopt manifold distance. The evaluator runs
without gradients, does not update model or prototype parameters, and restores
the original train/eval modes before returning. Prototype ID count and uniqueness
are validated explicitly, and unknown expected IDs fail before evaluation.

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
``train_loss`` and a nested ``validation`` mapping with the evaluator metrics.
Without an evaluator, ``fit()`` keeps the original ``list[float]`` return shape.

See :doc:`training` for the training contract, optimizer behavior, and runnable
Lorentz example, and :class:`neembed.ManifoldTrainer` for the generated API
details.

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

Euclidean vs Poincare vs Lorentz benchmark
------------------------------------------

The repository includes a deterministic, tiny hierarchy-retrieval benchmark:

.. code-block:: bash

   python experiments/compare_euclidean_poincare.py

The benchmark fine-tunes Euclidean, Poincare, and Lorentz variants under
matched data and training conditions. The two hyperbolic variants use the same
intrinsic projection dimension and the same public curvature magnitude. Each
variant reports the shared evaluator metrics plus ``final_training_loss``.

The JSON output records the seed, model name, task pairs, and key
hyperparameters needed to reproduce the run. It also records ambient dimensions
explicitly: for intrinsic dimension :math:`D`, Euclidean and Poincare use
:math:`D` output coordinates while Lorentz uses :math:`D + 1`. The extra
Lorentz coordinate is part of the hyperboloid representation, not additional
intrinsic model capacity.

See the `experiment documentation
<https://github.com/t-yamsaki/neembed/blob/main/experiments/README.md>`_ for the
fixed configuration and output contract.

This benchmark is an engineering and regression reference, not evidence that
one geometry is generally superior. It uses a tiny controlled task, does not
perform hyperparameter search, and should not be interpreted as a research
leaderboard result. Raw distance magnitudes also use different geometry-specific
metrics and should not be ranked directly across variants as universal quality
scores.

API reference
-------------

For constructor and return-value details generated directly from docstrings,
see :class:`neembed.ManifoldEmbeddingEvaluator` and
:class:`neembed.ManifoldPrototypeAssignmentEvaluator`.
