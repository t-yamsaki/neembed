Architecture
============

neembed is a small integration layer between a pretrained Sentence Transformer
and Geoopt manifolds. It does not implement its own manifold math.

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
   Manifold-valued embedding

For an input sentence :math:`x`, the pretrained encoder produces a Euclidean
embedding

.. math::

   h = f_\theta(x).

When ``embedding_dim`` is provided, a learned linear projection maps the
encoder output to an intrinsic tangent representation:

.. math::

   v = Wh.

When ``embedding_dim`` is omitted, the projection is ``torch.nn.Identity`` and
the encoder dimension is preserved.

For ``manifold="poincare"``, :math:`v` is mapped directly from the origin
tangent space onto the Poincare ball using Geoopt:

.. math::

   z = \mathrm{Exp}_0^c(v).

For ``manifold="lorentz"``, neembed prepends the required zero time-like
tangent coordinate before calling Geoopt's Lorentz ``expmap0``. If the intrinsic
``embedding_dim`` is :math:`D`, the resulting Lorentz point therefore has
ambient dimension :math:`D + 1`.

The public ``curvature`` argument is the positive magnitude of the negative
sectional curvature for both geometries. Poincare passes this value to
``geoopt.PoincareBall(c=...)``. Geoopt's Lorentz model uses the squared
hyperboloid radius ``k``, so neembed maps the same public value :math:`c` to
``k = 1 / c``.

Lorentz geometry is evaluated in ``float64`` while the pretrained encoder and
ordinary Euclidean projection parameters keep their normal dtype. Geoopt
recommends double precision for the Lorentz model because its Minkowski-space
operations are sensitive to floating-point error.

Why map from the tangent space?
-------------------------------

Sentence Transformers already produce ordinary Euclidean vectors. neembed
keeps that pretrained encoder intact, optionally changes its output dimension
with a Euclidean linear layer, and then uses Geoopt's differentiable
``expmap0`` operation to obtain a manifold-valued representation.

This keeps the boundary between responsibilities small:

* Sentence Transformers owns text encoding.
* PyTorch owns the trainable Euclidean parameters.
* Geoopt owns the Poincare and Lorentz geometry.
* neembed connects those pieces for sentence-embedding fine-tuning.

Geometry scope
--------------

The current development API supports ``manifold="poincare"`` and
``manifold="lorentz"``. The published v0.2.0 release is Poincare-only; Lorentz
support is targeted for v0.3.0. Spherical, SPD, product manifolds, learnable
curvature, and trainable manifold parameters remain outside the current scope.
