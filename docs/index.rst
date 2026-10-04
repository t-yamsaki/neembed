neembed
=======

**neembed** fine-tunes pretrained sentence embedding models in non-Euclidean
spaces while delegating manifold geometry to Geoopt.

The public API is intentionally small: a sentence model, manifold-aware losses,
a minimal trainer, retrieval evaluators, exact corpus search, offline
hard-negative mining, explicit hierarchy supervision, and opt-in manifold
prototypes. The current development API supports Euclidean, Poincare, Lorentz,
SphereProjection, and generic signed-curvature Stereographic geometry. Learnable
curvature remains limited to the backward-compatible Poincare and Lorentz
positive-magnitude path. v0.10 supports flat mixed-curvature products
with fixed component scales, full-product retrieval, named distance diagnostics,
and component-targeted hierarchy supervision.

Start with :doc:`getting_started/quickstart` if you want the ordinary training
workflow. For v0.7 objective selection and graded relevance -- MNRL, symmetric
MNRL, Triplet, MarginMSE, DistanceMSE, and nDCG@K -- see
:doc:`user_guide/retrieval_objectives`. For the exact retrieval path --
small-list reranking, exact corpus search, corpus evaluation, and caller-invoked
hard-negative mining -- see :doc:`user_guide/retrieval`. For v0.8 explicit
hierarchy supervision -- radial order, depth, directed triplets, retrieval-plus-
hierarchy composition, and structure metrics -- see :doc:`user_guide/hierarchy`.
For v0.9 geometry selection, curvature naming, compatibility, persistence, and
float64/device guidance across negative, zero, and positive sectional curvature,
see :doc:`user_guide/constant_curvature_semantics`. Read
:doc:`user_guide/learnable_structure` to distinguish Euclidean trainable
parameters, learnable curvature, and manifold-valued prototypes. For v0.10
mixed-curvature configuration, training, diagnostics, compatibility, and the
end-to-end regression example, see :doc:`user_guide/mixed_curvature`. The schema
is documented in :doc:`api/product_config`, or jump to the :ref:`api-reference`
for the rest of the class and function details generated from public docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/architecture
   user_guide/constant_curvature_semantics
   user_guide/mixed_curvature
   user_guide/learnable_structure
   user_guide/training
   user_guide/retrieval_objectives
   user_guide/retrieval
   user_guide/hierarchy
   user_guide/evaluation
   user_guide/inference
   user_guide/saving_loading

.. _api-reference:

API reference
-------------

.. toctree::
   :maxdepth: 1

   api/model
   api/product_config
   api/losses
   api/trainer
   api/evaluator
   api/retrieval
   api/mining
   api/prototypes
