Training
========

neembed supports both the original multiple-negatives ranking objective and a
small prototype-based hierarchy objective. Both use manifold geodesic distance
from Geoopt rather than custom distance formulas.

Geodesic ranking objective
--------------------------

For a batch of aligned anchor-positive pairs, the model produces manifold
embeddings :math:`z_i` for anchors and :math:`z_j^+` for positive candidates.
The similarity logit is negative geodesic distance scaled by temperature:

.. math::

   s_{ij} = -\frac{d_{\mathcal M}(z_i, z_j^+)}{\tau}.

The target for anchor :math:`i` is the positive at the same batch index, so the
loss is

.. math::

   \mathcal L_i = -\log
   \frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}.

``ManifoldMultipleNegativesRankingLoss`` computes the full pairwise distance
matrix and applies cross entropy to these diagonal targets.

Explicit hard negatives are optional. Calling
``loss(anchors, positives, negatives)`` embeds the explicit negatives and
appends them after the positive candidate batch before computing the same
geodesic-distance logits. The diagonal targets still point to the leading
positive candidates, so the candidate pool for every anchor contains all
in-batch positive candidates plus all supplied explicit negatives.

In-batch and explicit negatives
-------------------------------

All off-diagonal positive candidates are treated as negatives. This makes the
batch itself part of the supervision signal. When explicit negatives are
supplied, they are additional candidates for every anchor in the batch rather
than one negative being paired only with the anchor at the same index.

Avoid duplicate positives within one batch. For example, pairing two different
anchors with the same ``"mammal"`` positive would make that item both a target
and an off-diagonal negative in the same loss matrix.

For the three-sequence form, ``anchors``, ``positives``, and ``negatives`` must
have the same non-zero batch length. Automatic hard-negative mining is not part
of this API; callers choose the explicit negative texts.

Prototype hierarchy objective
-----------------------------

``ManifoldPrototypeHierarchyLoss`` is the focused hierarchy-aware objective for
trainable ``ManifoldPrototypes``. It combines two interpretable terms:

- each sentence embedding is attracted to its assigned prototype;
- each child prototype should be at least ``margin`` closer to its declared
  parent than to every other prototype except that child and direct parent.

For child :math:`c`, direct parent :math:`p`, and any other prototype
:math:`n`, the hierarchy penalty is

.. math::

   \max(0, m + d_{\mathcal M}(c,p) - d_{\mathcal M}(c,n)).

The negative set excludes only the current child and its direct parent. In a
deeper hierarchy, ancestors, descendants, and other connected prototypes can
therefore participate as negatives.

Prototype identifiers and hierarchy edges stay as ordinary Python values; no
graph Dataset or DataModule is required. For example:

.. code-block:: python

   import geoopt

   from neembed import (
       ManifoldPrototypeHierarchyLoss,
       ManifoldPrototypes,
   )

   prototype_ids = ["animal", "dog", "cat"]
   parent_relations = [
       ("dog", "animal"),
       ("cat", "animal"),
   ]

   prototypes = ManifoldPrototypes(model, num_prototypes=len(prototype_ids))
   loss = ManifoldPrototypeHierarchyLoss(
       model,
       prototypes,
       prototype_ids=prototype_ids,
       parent_relations=parent_relations,
       margin=0.1,
   )

   parameters = [
       parameter for parameter in loss.parameters() if parameter.requires_grad
   ]
   optimizer = geoopt.optim.RiemannianAdam(
       parameters,
       lr=1e-3,
       stabilize=1,
   )
   trainer = ManifoldTrainer(model, loss, optimizer=optimizer)

   train_batches = [
       (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
   ]
   history = trainer.fit(train_batches, epochs=3)

The second sequence in each hierarchy batch contains prototype assignments, not
positive texts. ``prototype_ids`` must be unique non-empty strings, and
``parent_relations`` must reference those identifiers, give each child at most
one parent, and remain acyclic.

Optimizer behavior
------------------

By default, ``ManifoldTrainer`` preserves the ordinary Euclidean training path
and creates ``torch.optim.AdamW`` over ``model.parameters()``.

The trainable sentence-encoder weights and optional linear projection are
ordinary Euclidean PyTorch parameters. The forward output is manifold-valued,
but the output being on a manifold does not by itself require a Riemannian
optimizer. Gradients flow through Geoopt's differentiable map and geodesic
distance back to the Euclidean trainable parameters.

When trainable parameters themselves live on a manifold, such as
``ManifoldPrototypes``, pass a caller-owned Geoopt optimizer through the
``optimizer`` argument. ``ManifoldTrainer`` uses a supplied optimizer as-is,
so one ``geoopt.optim.RiemannianAdam`` can own both ordinary model parameters
and manifold parameters without neembed reimplementing Riemannian updates.
Callers are responsible for including every trainable parameter that
participates in the loss.

For example, a loss that depends on both the model and a prototype module can
use one mixed optimizer:

.. code-block:: python

   import geoopt

   parameters = [
       parameter
       for parameter in (*model.parameters(), *prototypes.parameters())
       if parameter.requires_grad
   ]
   optimizer = geoopt.optim.RiemannianAdam(
       parameters,
       lr=1e-3,
       stabilize=1,
   )
   trainer = ManifoldTrainer(model, loss, optimizer=optimizer)

The loss must actually depend on the prototype values for those parameters to
receive gradients. When curvature and manifold-valued prototypes are both
learnable, ``stabilize=1`` is part of the supported joint path: Geoopt projects
manifold parameters back onto the current manifold after each optimizer step.
neembed does not implement a separate retraction or projection rule.

v0.4 learnable-structure regression example
--------------------------------------------

Run the compact end-to-end v0.4 reference from the repository root with:

.. code-block:: bash

   python examples/v04_learnable_structure.py

The example uses a tiny ``animal -> {dog, cat}`` hierarchy and the public
``ManifoldSentenceTransformer``, ``ManifoldPrototypes``,
``ManifoldPrototypeHierarchyLoss``, and ``ManifoldTrainer`` APIs. It runs two
matched Poincare configurations with the same seed, data, intrinsic dimension,
and initial public curvature:

- ``fixed_structure`` keeps curvature and prototype coordinates fixed while the
  ordinary sentence-model parameters train;
- ``learnable_structure`` enables learnable curvature and trainable prototypes
  and supplies ``geoopt.optim.RiemannianAdam(..., stabilize=1)``.

The command reports final training loss, sentence-to-prototype assignment
accuracy on the tiny hierarchy, initial/final public curvature, curvature
change, prototype movement, manifold validity, and finite embedding/prototype
distance diagnostics. The learnable path also fails loudly if curvature does
not change, prototypes do not update, distances become non-finite, or prototype
points leave the manifold.

This is a deterministic engineering regression example, not evidence that
learnable curvature or trainable prototypes outperform fixed structure. Normal
CI exercises the same example logic with a tiny fake encoder, so CI does not
need to download a Sentence Transformer model repeatedly.

Minimal training loop
---------------------

``ManifoldTrainer.fit`` accepts an iterable yielding either two or three aligned
string sequences. Two-sequence batches preserve the existing contract; their
meaning is defined by the configured loss, such as anchor-positive pairs for
``ManifoldMultipleNegativesRankingLoss`` or sentence-assignment pairs for
``ManifoldPrototypeHierarchyLoss``. Three-sequence batches are forwarded as
``(anchors, positives, negatives)`` and are intended for losses such as
``ManifoldMultipleNegativesRankingLoss`` that accept explicit hard negatives.

For more than one epoch, the input must be re-iterable, such as a list or a
DataLoader; a one-shot iterator or generator is only suitable for a single
epoch. Without validation, the method keeps its original return value: one mean
loss value per completed epoch.

The original two-sequence form remains unchanged:

.. code-block:: python

   train_batches = [
       (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
       (["dog", "cat"], ["mammal", "feline"]),
   ]

   history = trainer.fit(train_batches, epochs=3)
   print(history)

For explicit hard negatives, supply a third aligned sequence. Every explicit
negative in that sequence is added to the candidate pool for every anchor in
the batch:

.. code-block:: python

   train_batches = [
       (
           ["Shiba Inu", "Siamese cat"],
           ["dog", "cat"],
           ["wolf", "tiger"],
       ),
   ]

   history = trainer.fit(train_batches, epochs=3)

Batches with any other arity are rejected rather than being interpreted by a
generic batch-dispatch framework.

Optional validation
-------------------

Pass a ``ManifoldEmbeddingEvaluator`` to run validation once after each
completed epoch. Validation runs without gradient tracking, and training mode
is restored before the next epoch.

.. code-block:: python

   evaluator = ManifoldEmbeddingEvaluator(
       model=model,
       anchors=["Shiba Inu", "Siamese cat"],
       positives=["dog", "cat"],
   )

   history = trainer.fit(
       train_batches,
       epochs=3,
       evaluator=evaluator,
   )

With an evaluator, each history entry has a mean ``train_loss`` and a nested
``validation`` dictionary containing the evaluator metrics. This keeps the
no-evaluator return behavior backward-compatible while making epoch validation
results explicit. See :doc:`evaluation` for metric definitions and
interpretation.

PyTorch DataLoader
------------------

An ordinary ``torch.utils.data.DataLoader`` can feed ``ManifoldTrainer``
directly; neembed does not require a custom Dataset or DataModule abstraction.
The runnable `DataLoader example
<https://github.com/t-yamsaki/neembed/blob/main/examples/train_dataloader.py>`_
uses a plain list of aligned text pairs, default PyTorch collation, multi-epoch
training, and optional validation. Its batches keep positive candidates unique
so the in-batch-negative objective remains well-defined. A DataLoader may also
yield three aligned string sequences for the explicit-hard-negative ranking
path; the same batch-length requirements apply.

Lorentz example
---------------

The same model, loss, trainer, and evaluator workflow works with
``manifold="lorentz"``. Run the small hierarchy-style example from the
repository root with:

.. code-block:: bash

   python examples/train_lorentz.py

The `Lorentz example
<https://github.com/t-yamsaki/neembed/blob/main/examples/train_lorentz.py>`_
prints finite training/evaluation results, a representative geodesic distance,
and the distinction between the configured intrinsic ``embedding_dim`` and the
Lorentz ambient output dimension, which has one additional time-like
coordinate. It is a usage example, not a claim that Lorentz is generally
better than Poincare geometry.

For argument and return-value details, see
:class:`neembed.ManifoldMultipleNegativesRankingLoss`,
:class:`neembed.ManifoldPrototypeHierarchyLoss`,
:class:`neembed.ManifoldEmbeddingEvaluator`, and
:class:`neembed.ManifoldTrainer`.