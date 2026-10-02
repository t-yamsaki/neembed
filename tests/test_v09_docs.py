"""Documentation contract checks for v0.9 constant-curvature selection."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v09_geometry_guide_covers_selection_compatibility_and_numerics() -> None:
    guide = _read("docs/user_guide/constant_curvature_semantics.rst")

    for term in (
        "Choosing a geometry",
        'manifold="euclidean"',
        'manifold="poincare"',
        'manifold="lorentz"',
        'manifold="sphere_projection"',
        'manifold="stereographic"',
        "Compatibility with existing workflows",
        "Constant-curvature compatibility matrix",
        "Learnable curvature",
        "sectional_curvature < 0",
        "sectional_curvature == 0",
        "sectional_curvature > 0",
        "float64 geometry paths",
        "Apple MPS",
        "examples/v09_constant_curvature_comparison.py",
        "not a benchmark",
        "geometry-superiority claim",
    ):
        assert term in guide

    assert "twice the corresponding Euclidean" in guide
    assert "public name is deliberately not ``k``" in guide


def test_v09_geometry_guide_references_current_public_workflows() -> None:
    guide = _read("docs/user_guide/constant_curvature_semantics.rst")

    for public_api in (
        "neembed.ManifoldMultipleNegativesRankingLoss",
        "neembed.ManifoldSymmetricMultipleNegativesRankingLoss",
        "neembed.ManifoldTripletLoss",
        "neembed.ManifoldMarginMSELoss",
        "neembed.ManifoldDistanceMSELoss",
        "neembed.ManifoldCorpusRetrievalEvaluator",
        "neembed.ManifoldGradedCorpusRetrievalEvaluator",
        "exact_corpus_search()",
        "mine_hard_negatives()",
        "neembed.ManifoldRadialOrderLoss",
        "neembed.ManifoldDepthLoss",
        "neembed.ManifoldHierarchyTripletLoss",
        "neembed.ManifoldRetrievalHierarchyLoss",
        "neembed.ManifoldHierarchyEvaluator",
    ):
        assert public_api in guide


def test_v09_geometry_guide_is_linked_from_index_readme_and_model_api() -> None:
    index = _read("docs/index.rst")
    readme = _read("README.md")
    model_api = _read("docs/api/model.rst")

    assert "user_guide/constant_curvature_semantics" in index
    assert "Constant-curvature geometry guide" in readme
    assert "examples/v09_constant_curvature_comparison.py" in readme
    assert "../user_guide/constant_curvature_semantics" in model_api

    assert "geometry choice is task-dependent" in readme.lower()
    assert "not a benchmark or geometry-superiority claim" in readme.lower()
