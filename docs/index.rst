neembed
=======

**neembed** fine-tunes pretrained sentence embedding models in non-Euclidean
spaces while delegating manifold geometry to Geoopt.

The public API is intentionally small: a sentence model, a manifold-aware
multiple-negatives ranking loss, a minimal trainer, an embedding evaluator, and
an opt-in manifold prototype module. The current API supports Poincare and Lorentz
hyperbolic models through the same sentence-model geometry semantics.

Start with :doc:`getting_started/quickstart` if you want to train a model. Read
:doc:`user_guide/architecture` for manifold selection, curvature, dimensions,
and Lorentz precision behavior, or jump to the :ref:`api-reference` for class
and method details generated from the public docstrings.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/architecture
   user_guide/training
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
   api/prototypes
