Loss
====

Retrieval objectives
--------------------

.. autoclass:: neembed.ManifoldMultipleNegativesRankingLoss
   :members:

.. autoclass:: neembed.ManifoldSymmetricMultipleNegativesRankingLoss
   :members:

.. autoclass:: neembed.ManifoldTripletLoss
   :members:

.. autoclass:: neembed.ManifoldMarginMSELoss
   :members:

.. autoclass:: neembed.ManifoldDistanceMSELoss
   :members:

Hierarchy objectives
--------------------

The hierarchy losses consume explicit caller-owned supervision and do not infer
or persist a graph. Radius always means geodesic distance from the configured
manifold origin via ``dist0``. See :doc:`../user_guide/hierarchy` for metadata,
origin, composition, and evaluation semantics.

.. autoclass:: neembed.ManifoldRadialOrderLoss
   :members:

.. autoclass:: neembed.ManifoldDepthLoss
   :members:

.. autoclass:: neembed.ManifoldHierarchyTripletLoss
   :members:

.. autoclass:: neembed.ManifoldRetrievalHierarchyLoss
   :members:
