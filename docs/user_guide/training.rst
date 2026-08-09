Training
========

neembed v0.1 fine-tunes the pretrained encoder and optional projection with a
multiple-negatives ranking objective based on manifold geodesic distance.

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

In-batch negatives
------------------

All off-diagonal positive candidates are treated as negatives. This makes the
batch itself part of the supervision signal.

Avoid duplicate positives within one batch. For example, pairing two different
anchors with the same ``"mammal"`` positive would make that item both a target
and an off-diagonal negative in the same loss matrix.

Optimizer behavior
------------------

``ManifoldTrainer`` uses ordinary ``torch.optim.AdamW`` over
``model.parameters()``. This is intentional for v0.1.

The trainable sentence-encoder weights and optional linear projection are
ordinary Euclidean PyTorch parameters. The forward output is manifold-valued,
but the output being on a manifold does not by itself require a Riemannian
optimizer. Gradients flow through Geoopt's differentiable map and geodesic
distance back to the Euclidean trainable parameters.

A Riemannian optimizer becomes relevant when trainable parameters themselves
live on a manifold, for example trainable manifold prototypes or hierarchy
nodes. neembed v0.1 does not introduce such parameters.

Minimal training loop
---------------------

``ManifoldTrainer.fit`` accepts an iterable yielding ``(anchors, positives)``
batches. For more than one epoch, that input must be re-iterable, such as a
list or a DataLoader; a one-shot iterator or generator is only suitable for a
single epoch. The method returns one mean loss value per completed epoch:

.. code-block:: python

   train_batches = [
       (["Shiba Inu", "Siamese cat"], ["dog", "cat"]),
       (["dog", "cat"], ["mammal", "feline"]),
   ]

   history = trainer.fit(train_batches, epochs=3)
   print(history)

For argument and return-value details, see
:class:`neembed.ManifoldMultipleNegativesRankingLoss` and
:class:`neembed.ManifoldTrainer`.
