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

Output geometry vs parameter geometry
-------------------------------------

The forward output being manifold-valued does not mean that every trainable
parameter lives on a manifold. The pretrained encoder weights and optional
projection are ordinary Euclidean PyTorch parameters. ``learnable_curvature``
adds a trainable positive scalar geometry parameter, which is also not a
manifold-valued point.

``ManifoldPrototypes`` is different: its coordinates are true
``geoopt.ManifoldParameter`` values. That distinction determines optimizer
requirements. The ordinary model-only path can keep using AdamW, while updating
prototype points requires a Geoopt Riemannian optimizer. See
:doc:`learnable_structure` for the complete parameter and optimizer matrix.

Curvature semantics
-------------------

The public ``curvature`` argument is the positive magnitude of the negative
sectional curvature for both geometries. Poincare passes this value to
``geoopt.PoincareBall(c=...)``. Geoopt's Lorentz model uses the squared
hyperboloid radius ``k``, so neembed maps the same public value :math:`c` to
``k = 1 / c``.

Curvature remains fixed by default. Setting ``learnable_curvature=True`` makes
the same public curvature magnitude trainable. Poincare uses Geoopt's
constrained learnable ``c`` parameter. The Lorentz backend exposes an
unconstrained squared radius ``k``, so neembed adds a small positive PyTorch
parametrization around ``k`` while continuing to use Geoopt for all manifold
operations.

The model's ``curvature`` property always reports the current public curvature
magnitude, including after optimizer updates. Learnable curvature alone does not
require a Riemannian optimizer because the curvature state is a scalar geometry
parameter, not a manifold-valued coordinate.

Choosing a manifold
-------------------

Both supported manifolds use the same pretrained encoder, optional projection,
losses, trainer, evaluator, and sentence-model save/load API. The choice is a
coordinate-model choice rather than a different training framework.

``poincare``
   Produces :math:`D` coordinates for an intrinsic dimension :math:`D`. The
   geometry path keeps the model's ordinary parameter dtype. Poincare prototype
   points also use :math:`D` coordinates.

``lorentz``
   Represents the same intrinsic dimension :math:`D` with :math:`D + 1`
   ambient hyperboloid coordinates. The same ambient rule applies to Lorentz
   prototype points.

Lorentz geometry is evaluated in ``float64`` while the pretrained encoder and
ordinary Euclidean projection parameters keep their normal dtype. This keeps
the extra precision cost confined to the geometry path and follows Geoopt's
numerical guidance for the Lorentz model.

Use the same public curvature magnitude and intrinsic dimension when comparing
the two hyperbolic models. Do not infer that the extra Lorentz coordinate is
extra intrinsic model capacity, and do not assume either coordinate model is
generally superior. Choose based on the representation and numerical
trade-offs that matter for your application, then compare task metrics under
matched conditions. See :doc:`evaluation` for the repository benchmark and its
limits.

Why map from the tangent space?
-------------------------------

Sentence Transformers already produce ordinary Euclidean vectors. neembed
keeps that pretrained encoder intact, optionally changes its output dimension
with a Euclidean linear layer, and then uses Geoopt's differentiable
``expmap0`` operation to obtain a manifold-valued representation.

This keeps the boundary between responsibilities small:

* Sentence Transformers owns text encoding.
* PyTorch owns the ordinary Euclidean trainable parameters and the small
  positive Lorentz-curvature parametrization.
* Geoopt owns Poincare/Lorentz geometry and manifold-parameter optimization.
* neembed connects those pieces for sentence-embedding fine-tuning.

Geometry scope
--------------

The v0.4 development API supports ``manifold="poincare"`` and
``manifold="lorentz"`` with fixed or opt-in learnable curvature. It also
supports opt-in trainable manifold prototypes and the focused hierarchy-aware
objective described in :doc:`learnable_structure` and :doc:`training`.

Spherical, SPD, product manifolds, mixed-curvature manifold products, automatic
prototype discovery, and a generalized optimizer framework remain outside the
current scope.
