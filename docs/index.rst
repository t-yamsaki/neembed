neembed
=======

**neembed** fine-tunes pretrained sentence embedding models in non-Euclidean
spaces while delegating manifold geometry to Geoopt.

The public API is intentionally small: a sentence model, manifold-aware losses,
a minimal trainer, retrieval evaluators, exact corpus search, offline
hard-negative mining, explicit hierarchy supervision, and opt-in manifold
prototypes. The current development API supports Poincare and Lorentz
hyperbolic models, fixed or learnable curvature, and true manifold-valued
prototype parameters while preserving the original model-only AdamW path.

Start with :doc:`getting_started/quickstart` if you want the ordinary training
workflow. For v0.7 objective selection and graded relevance -- MNRL, symmetric
MNRL, Triplet, MarginMSE, DistanceMSE, and nDCG@K -- see
:doc:`user_guide/retrieval_objectives`. For the exact retrieval path --
small-list reranking, exact corpus search, corpus evaluation, and caller-invoked
hard-negative mining -- see :doc:`user_guide/retrieval`. For v0.8 explicit
hierarchy supervision -- radial order, depth, directed triplets, retrieval-plus-
hierarchy composition, and structure metrics -- see :doc:`user_guide/hierarchy`.
Read :doc:`user_guide/learnable_structure` to distinguish Euclidean trainable
parameters, learnable curvature, and manifold-valued prototypes, or jump to the
:ref:`api-reference` for class and function details generated from the public
docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/architecture
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
   api/losses
   api/trainer
   api/evaluator
   api/retrieval
   api/mining
   api/prototypes
