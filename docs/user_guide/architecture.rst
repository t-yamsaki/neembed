Architecture
============

neembed v0.1 is a small integration layer between a pretrained Sentence
Transformer and Geoopt's Poincare ball. It does not implement its own manifold
math.

Data flow
---------

The implemented forward path is:

.. code-block:: text

   Pretrained Sentence Encoder
           |
           v
   Euclidean sentence embedding
           |
           v
   Projection head (optional)
           |
           v
   Tangent-space representation
           |
           v
   Geoopt expmap0
           |
           v
   Poincare embedding

For an input sentence :math:`x`, the pretrained encoder produces a Euclidean
embedding

.. math::

   h = f_\theta(x).

When ``embedding_dim`` is provided, a learned linear projection maps the
encoder output to a tangent vector:

.. math::

   v = Wh.

When ``embedding_dim`` is omitted, the projection is ``torch.nn.Identity`` and
the encoder dimension is preserved.

The tangent vector is then mapped from the origin tangent space onto the
Poincare ball using Geoopt:

.. math::

   z = \mathrm{Exp}_0^c(v).

Here :math:`c` is the positive curvature parameter passed to
``geoopt.PoincareBall``.

Why map from the tangent space?
-------------------------------

Sentence Transformers already produce ordinary Euclidean vectors. neembed
keeps that pretrained encoder intact, optionally changes its output dimension
with a Euclidean linear layer, and then uses Geoopt's differentiable
``expmap0`` operation to obtain a manifold-valued representation.

This keeps the boundary between responsibilities small:

* Sentence Transformers owns text encoding.
* PyTorch owns the trainable Euclidean parameters.
* Geoopt owns the Poincare geometry.
* neembed connects those pieces for sentence-embedding fine-tuning.

v0.1 geometry scope
-------------------

Only ``manifold="poincare"`` is supported in v0.1. Lorentz, spherical, SPD,
product manifolds, learnable curvature, and trainable manifold parameters are
outside the current implementation and should not be assumed from the API.
