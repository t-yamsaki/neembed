Hierarchy-native learning
=========================

v0.8 adds explicit hierarchy supervision without making neembed a graph or
ontology framework. The caller owns node identifiers, text labels, parent-child
edges, optional depth labels, and any unrelated-node negatives. neembed only
consumes the aligned supervision needed by each loss or evaluator.

This guide complements :doc:`retrieval_objectives` and :doc:`retrieval` rather
than replacing them. Retrieval quality and hierarchy structure are separate
concerns and should be evaluated separately.

Caller-owned hierarchy metadata
-------------------------------

A hierarchy starts with ordinary Python data, for example::

    node_ids = ("root", "animal", "dog", "shiba-inu")
    texts = ("root concept", "animal", "dog", "Shiba Inu")
    parent_child_edges = (
        ("root", "animal"),
        ("animal", "dog"),
        ("dog", "shiba-inu"),
    )
    depths = {
        "root": 0,
        "animal": 1,
        "dog": 2,
        "shiba-inu": 3,
    }

neembed does not build a graph object, infer edges from text, compute ontology
similarity, or persist caller metadata in model state. IDs are metadata owned by
the application; text sequences are the values encoded by the sentence model.

When an API consumes complete hierarchy metadata, the validation contract is:

* ``node_ids`` are non-empty, unique strings.
* Every edge is an explicit ``(parent_id, child_id)`` pair referencing known
  nodes.
* Self edges, duplicate edges, and directed cycles are rejected.
* ``contract="tree"`` means an acyclic directed forest: each node has at most
  one parent. ``contract="dag"`` permits multiple parents while remaining
  acyclic.
* Roots are the nodes with no incoming edge. If roots are supplied, they must
  match that inferred set exactly.
* Depths are optional caller-owned non-negative integers and may be partial.
  Supplied values must increase consistently along every directed edge and path;
  they are not required to equal a shortest-path depth.
* Normalization is deterministic and does not materialize a transitive closure.

:class:`neembed.ManifoldHierarchyEvaluator` applies this hierarchy validation
when it receives IDs, edges, and optional depths. The training losses use smaller
aligned input contracts, so they validate their own batch shape and scalar
parameters but do not silently infer or reconstruct the full graph.

Radius means geodesic distance from the origin
----------------------------------------------

All v0.8 radial semantics use the configured manifold's ``dist0`` operation.
Radius is therefore geodesic distance from the manifold origin, not an ambient
Euclidean norm.

For the Poincare ball, the origin is the zero vector. For the Lorentz /
hyperboloid model, the origin is ``(sqrt(k), 0, ..., 0)``, where Geoopt's
squared-radius parameter ``k`` is ``1 / curvature`` under neembed's public
curvature convention.

The same public interpretation is used by the radial loss, depth loss,
directed hierarchy loss, and hierarchy evaluator on both supported manifolds.
The v0.8 end-to-end regression example intentionally uses Poincare geometry to
keep one compact reference workflow; Lorentz behavior is covered by the
lower-level API tests.

Radial order
------------

:class:`neembed.ManifoldRadialOrderLoss` receives aligned parent and child
texts. For one pair it penalizes::

    relu(radius(parent) + margin - radius(child))

A positive margin therefore asks the child to be farther from the origin than
its parent by at least that geodesic amount. The loss imposes direction through
radius only; it does not assert that all nodes at similar radii are semantically
similar.

Depth supervision
-----------------

:class:`neembed.ManifoldDepthLoss` receives aligned texts and non-negative
integer depths. A caller-owned depth ``d`` is mapped to the target radius::

    target_radius = d * radial_scale

and the loss is mean squared error between observed geodesic radius and that
target. Depth ``0`` therefore targets the manifold origin. Nodes sharing a depth
share a radial target but receive no angular or adjacency constraint from this
loss alone.

Depth is supervision, not a graph inference result. If IDs and texts are kept in
separate structures, the caller is responsible for preserving their alignment
before constructing a training batch.

Directed hierarchy triplets
---------------------------

:class:`neembed.ManifoldHierarchyTripletLoss` consumes aligned
``(parent, child, unrelated)`` text triplets. It combines a geodesic ranking
term with a radial direction term::

    relu(d(parent, child) - d(parent, unrelated) + margin)
    + radial_weight * relu(
        radius(parent) + radial_margin - radius(child)
      )

The first term prefers the declared child over an unrelated node around the
parent. The second prevents the objective from becoming purely symmetric by
requiring the parent to remain radially inside the child.

Negative construction remains caller-owned. neembed does not automatically
mine hierarchy negatives or compute transitive-closure training pairs.

Composing retrieval and hierarchy objectives
--------------------------------------------

:class:`neembed.ManifoldRetrievalHierarchyLoss` is deliberately small. It
combines exactly one retrieval loss and one hierarchy loss::

    total = retrieval_loss + hierarchy_weight * hierarchy_loss

Each component keeps its native positional input contract. This lets an existing
retrieval objective remain unchanged while hierarchy supervision is added
explicitly. ``hierarchy_weight`` is non-negative; a value of ``0`` is a true
retrieval-only path and skips hierarchy evaluation in ``forward``.

The wrapper is not a general loss graph, scheduler, curriculum system, or
learned weighting mechanism. Existing losses remain directly usable, and no
trainer redesign is required. These losses only optimize ordinary model
parameters, so the existing AdamW model-only path remains valid. If a separate
workflow introduces manifold-valued trainable parameters such as prototypes,
see :doc:`learnable_structure` for optimizer requirements.

Hierarchy metrics are not retrieval metrics
-------------------------------------------

:class:`neembed.ManifoldHierarchyEvaluator` reports structure diagnostics from
caller-owned hierarchy metadata independently of retrieval quality.

``parent_child_radial_order_accuracy``
    Fraction of supplied parent-child edges satisfying the strict condition
    ``radius(parent) < radius(child)``. Equal-radius ties are not correct.

``mean_radial_order_violation``
    Mean ``max(radius(parent) - radius(child), 0)`` over supplied edges. An exact
    tie has zero violation magnitude even though it is not counted as correct by
    the strict accuracy metric.

``depth_radius_spearman``
    When depths are explicitly supplied, a tie-aware Spearman rank association
    between those supplied depths and geodesic radii. Partial depth mappings use
    only labeled nodes. Fewer than two labels or a zero-variance ranking reports
    ``0.0`` so the metric remains finite. This is a descriptive association and
    does not imply causality.

The evaluator is deterministic, runs without gradients, and restores the
model's previous train/eval mode. These metrics describe whether embeddings
respect the supplied hierarchy; they do not establish that semantic retrieval
improved. Continue to use the retrieval evaluators described in
:doc:`evaluation` and :doc:`retrieval` for retrieval quality.

End-to-end v0.8 regression example
----------------------------------

``examples/v08_hierarchy_learning.py`` provides a small deterministic reference
that starts retrieval-only and hierarchy-aware models from the same Poincare
initialization. The hierarchy-aware path composes retrieval supervision with
radial-order, depth, and directed-triplet phases, then reports both retrieval
and hierarchy metrics before and after training.

Run it from the repository root::

    python examples/v08_hierarchy_learning.py

The taxonomy, edges, depths, and directed negatives in the example are all
explicit caller-owned data. The example is an engineering regression reference,
not a benchmark and not evidence that hierarchy-aware training is universally
superior.

See the source at `examples/v08_hierarchy_learning.py <https://github.com/t-yamsaki/neembed/blob/main/examples/v08_hierarchy_learning.py>`_.

API reference
-------------

The public v0.8 hierarchy surface is documented in :doc:`../api/losses` and
:doc:`../api/evaluator`:

* :class:`neembed.ManifoldRadialOrderLoss`
* :class:`neembed.ManifoldDepthLoss`
* :class:`neembed.ManifoldHierarchyTripletLoss`
* :class:`neembed.ManifoldRetrievalHierarchyLoss`
* :class:`neembed.ManifoldHierarchyEvaluator`

This surface intentionally stops short of ontology parsing, graph-database
integration, automatic taxonomy completion, or graph reconstruction benchmarks.
