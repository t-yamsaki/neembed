Learnable manifold structure
============================

v0.4 adds opt-in trainable geometry while preserving the original v0.3
training path. The important distinction is **what is trainable**, not merely
whether the model output lies on a manifold.

Three parameter categories
--------------------------

neembed exposes three different kinds of trainable state:

.. list-table::
   :header-rows: 1
   :widths: 28 26 23 23

   * - State
     - Representation
     - Default optimizer guidance
     - Manifold-valued parameter?
   * - Sentence encoder and optional projection
     - Ordinary PyTorch parameters
     - ``AdamW``
     - No
   * - Learnable curvature
     - Positive scalar geometry parameter
     - ``AdamW`` is sufficient when no manifold-valued parameters participate
     - No
   * - ``ManifoldPrototypes``
     - ``geoopt.ManifoldParameter`` points
     - Geoopt Riemannian optimizer
     - Yes

The sentence embeddings returned by ``ManifoldSentenceTransformer`` are
manifold-valued **outputs**. That does not turn the encoder or projection
weights into manifold-valued parameters. Therefore the ordinary model-only
path, including model-only learnable curvature, does not require a Riemannian
optimizer.

Fixed and learnable curvature
-----------------------------

Curvature is fixed by default. Set ``learnable_curvature=True`` on
``ManifoldSentenceTransformer`` to make the geometry parameter trainable.

The public ``curvature`` value always means the positive magnitude :math:`c` of
negative sectional curvature for both supported coordinate models:

``manifold="poincare"``
   neembed uses Geoopt's Poincare-ball curvature ``c`` directly.

``manifold="lorentz"``
   Geoopt parameterizes the hyperboloid by squared radius ``k``. neembed maps
   the same public curvature magnitude to ``k = 1 / c`` and reports
   ``model.curvature = 1 / k``.

``model.curvature`` returns the current public value after optimizer updates.
Poincare uses Geoopt's positive curvature parameterization. Lorentz keeps
``k`` positive with a small PyTorch parametrization while all manifold
operations remain delegated to Geoopt.

Trainable manifold prototypes
-----------------------------

``ManifoldPrototypes`` stores prototype or hierarchy-node coordinates as true
``geoopt.ManifoldParameter`` values on the exact manifold owned by the model.
These parameters require Riemannian optimization when they are updated.

``embedding_dim`` is always the intrinsic dimension:

- Poincare prototypes have shape ``(num_prototypes, embedding_dim)``;
- Lorentz prototypes represent the same intrinsic dimension with one additional
  ambient time-like coordinate, so their shape is
  ``(num_prototypes, embedding_dim + 1)``.

This extra Lorentz coordinate is an ambient-coordinate requirement, not extra
intrinsic model capacity.

Mixed Euclidean and manifold optimization
-----------------------------------------

A caller-supplied Geoopt optimizer can own both ordinary PyTorch parameters and
``ManifoldParameter`` values. ``ManifoldTrainer`` uses the supplied optimizer
as-is and does not discover, merge, or rewrite parameter groups.

For a loss that depends on both the model and trainable prototypes, include all
participating trainable parameters. When curvature and prototype coordinates
are both learnable, the supported v0.4 path uses Geoopt stabilization after
every step, for example:

.. code-block:: python

   optimizer = geoopt.optim.RiemannianAdam(
       [parameter for parameter in loss.parameters() if parameter.requires_grad],
       lr=1e-3,
       stabilize=1,
   )

``stabilize=1`` lets Geoopt project manifold-valued parameters back onto the
current manifold after curvature changes. neembed does not implement a custom
retraction, geodesic update, or stabilization rule.

If no manifold-valued trainable parameters are present, omitting ``optimizer``
from ``ManifoldTrainer`` preserves the existing ``AdamW(model.parameters())``
behavior. See :doc:`training` for the complete optimizer and batching contract.

Hierarchy-aware objective
-------------------------

``ManifoldPrototypeHierarchyLoss`` combines sentence-to-prototype attraction
with a small hierarchy-ranking term. For each declared child prototype
:math:`c` and direct parent :math:`p`, every prototype :math:`n` other than that
child and direct parent is used as a negative:

.. math::

   \max(0, m + d_{\mathcal M}(c,p) - d_{\mathcal M}(c,n)).

This negative set is intentionally local to the declared edge. In a deeper
hierarchy, ancestors, descendants, or other connected prototypes are not
relation-aware exclusions; if they are neither the current child nor its direct
parent, they participate as negatives.

The first input sequence contains sentences. The second contains prototype
identifier assignments aligned by batch index. ``prototype_ids`` must be
unique non-empty strings. ``parent_relations`` contains explicit
``(child_id, parent_id)`` pairs, must reference those identifiers, may declare
at most one parent per child, and must remain acyclic.

These identifiers and relations are ordinary Python structure; neembed does
not introduce a graph Dataset or DataModule abstraction.

Persistence boundaries
----------------------

``ManifoldSentenceTransformer.save_pretrained()`` stores the encoder,
projection state, manifold name, current public curvature, intrinsic projection
dimension, and whether curvature is learnable. Reloading with
``from_pretrained()`` reconstructs the same curvature trainability and current
public curvature value. Older configurations without the learnable-curvature
flag remain fixed-curvature models.

``ManifoldPrototypes`` is an opt-in module external to the sentence model, so it
is **not** automatically included by ``model.save_pretrained()``. Save and load
its ``state_dict()`` separately after reconstructing a compatible model and
prototype module. Prototype identifiers, parent relations, loss hyperparameters,
and optimizer state are also caller-owned configuration rather than part of the
model save helper. See :doc:`saving_loading` for a concrete persistence pattern.

Runnable v0.4 regression example
--------------------------------

Run the compact end-to-end reference from the repository root:

.. code-block:: bash

   python examples/v04_learnable_structure.py

The example compares matched fixed-structure and learnable-structure Poincare
runs using the same tiny hierarchy, seed, intrinsic dimension, and initial
public curvature. It reports final training loss, assignment accuracy,
initial/final curvature, prototype movement, manifold validity, and finite
distance diagnostics.

This is an engineering regression check for usability and state behavior. It is
not evidence that learnable curvature or trainable prototypes are generally
superior to fixed structure, and it is not an external benchmark or
leaderboard.

Numerical caveats
-----------------

- Lorentz manifold operations and outputs use ``float64`` because hyperboloid
  calculations are more sensitive to numerical error. The encoder and ordinary
  Euclidean projection keep their normal dtype.
- Learnable curvature is kept positive, but positivity alone does not guarantee
  a numerically useful training trajectory. Monitor curvature and finite
  distances when using aggressive learning rates or long runs.
- When curvature changes jointly with manifold-valued prototypes, use the
  documented Geoopt stabilization path and continue checking prototype manifold
  validity.
- Poincare and Lorentz comparisons should match intrinsic dimension and public
  curvature. Lorentz's extra ambient coordinate must not be counted as extra
  intrinsic capacity.

The v0.4 scope remains intentionally narrow: Poincare and Lorentz geometry,
opt-in learnable curvature, trainable prototypes, and the focused hierarchy
objective. Product manifolds, mixed-curvature manifold products, automatic
prototype discovery, and a generalized optimizer framework are not part of
this API.
