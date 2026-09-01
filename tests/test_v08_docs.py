"""Documentation contract checks for the v0.8 hierarchy-native workflow."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v08_hierarchy_guide_covers_supervision_geometry_and_metrics() -> None:
    hierarchy = _read("docs/user_guide/hierarchy.rst")

    for term in (
        "Caller-owned hierarchy metadata",
        'contract="tree"',
        'contract="dag"',
        "directed cycles",
        "Poincare ball",
        "Lorentz",
        "dist0",
        "ManifoldRadialOrderLoss",
        "ManifoldDepthLoss",
        "ManifoldHierarchyTripletLoss",
        "ManifoldRetrievalHierarchyLoss",
        "ManifoldHierarchyEvaluator",
        "parent_child_radial_order_accuracy",
        "mean_radial_order_violation",
        "depth_radius_spearman",
        "examples/v08_hierarchy_learning.py",
        "not a benchmark",
    ):
        assert term in hierarchy


def test_v08_public_hierarchy_api_is_referenced() -> None:
    losses = _read("docs/api/losses.rst")
    evaluator = _read("docs/api/evaluator.rst")

    for public_loss in (
        "neembed.ManifoldRadialOrderLoss",
        "neembed.ManifoldDepthLoss",
        "neembed.ManifoldHierarchyTripletLoss",
        "neembed.ManifoldRetrievalHierarchyLoss",
    ):
        assert public_loss in losses
    assert "neembed.ManifoldHierarchyEvaluator" in evaluator
    assert "../user_guide/hierarchy" in losses
    assert "../user_guide/hierarchy" in evaluator


def test_v08_hierarchy_guide_is_linked_without_replacing_retrieval_guidance() -> None:
    index = _read("docs/index.rst")
    readme = _read("README.md")

    assert "user_guide/hierarchy" in index
    assert "user_guide/retrieval_objectives" in index
    assert "user_guide/retrieval" in index
    assert "Hierarchy-native learning guide" in readme
    assert "examples/v08_hierarchy_learning.py" in readme
    assert "Retrieval workflow guide" in readme
    assert "Retrieval objectives guide" in readme
