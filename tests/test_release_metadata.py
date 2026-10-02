"""Release-readiness checks for public v0.9 metadata and contracts."""

from importlib.metadata import metadata
from inspect import signature
from pathlib import Path

import neembed
from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldDepthLoss,
    ManifoldDistanceMSELoss,
    ManifoldEmbeddingEvaluator,
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldHierarchyEvaluator,
    ManifoldHierarchyTripletLoss,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss,
    ManifoldTrainer,
    ManifoldTripletLoss,
    exact_corpus_search,
    mine_hard_negatives,
)


ROOT = Path(__file__).parents[1]
DOCUMENTATION_URL = "https://neembed.readthedocs.io/en/latest/"


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_pyproject_declares_v09_public_metadata() -> None:
    pyproject = _read("pyproject.toml")

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'name = "neembed-geoopt"' in pyproject
    assert 'version = "0.9.0"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '{ name = "taishi-yamasaki" }' in pyproject
    assert 'Homepage = "https://github.com/t-yamsaki/neembed"' in pyproject
    assert f'Documentation = "{DOCUMENTATION_URL}"' in pyproject
    assert 'Repository = "https://github.com/t-yamsaki/neembed"' in pyproject
    assert 'Issues = "https://github.com/t-yamsaki/neembed/issues"' in pyproject
    assert "License ::" not in pyproject

    for python_version in ("3.10", "3.11", "3.12"):
        assert f'"Programming Language :: Python :: {python_version}"' in pyproject
    for dependency in ("torch", "sentence-transformers", "geoopt"):
        assert f'    "{dependency}",' in pyproject


def test_installed_distribution_exposes_v09_metadata() -> None:
    package_metadata = metadata("neembed-geoopt")
    project_urls = package_metadata.get_all("Project-URL") or []

    assert package_metadata["Name"] == "neembed-geoopt"
    assert package_metadata["Version"] == "0.9.0"
    assert package_metadata["Requires-Python"] == ">=3.10"
    assert f"Documentation, {DOCUMENTATION_URL}" in project_urls
    assert neembed.__name__ == "neembed"


def test_readmes_publish_v09_and_preserve_prior_release_scope() -> None:
    english = _read("README.md")
    japanese = _read("docs/README_ja.md")
    installation = _read("docs/getting_started/installation.rst")

    for document in (english, japanese, installation):
        assert "pip install neembed-geoopt" in document

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        for release in ("v0.9.0", "v0.8", "v0.7", "v0.6", "v0.5", "v0.4", "v0.3"):
            assert release in readme
        for public_surface in (
            "Poincaré",
            "Lorentz",
            "Euclidean",
            "SphereProjection",
            "Stereographic",
            "ManifoldPrototypes",
            "ManifoldEmbeddingEvaluator",
            "model.rank()",
            "exact_corpus_search()",
            "ManifoldCorpusRetrievalEvaluator",
            "mine_hard_negatives()",
            "ManifoldTripletLoss",
            "ManifoldMarginMSELoss",
            "ManifoldDistanceMSELoss",
            "ManifoldSymmetricMultipleNegativesRankingLoss",
            "ManifoldGradedCorpusRetrievalEvaluator",
            "ManifoldRadialOrderLoss",
            "ManifoldDepthLoss",
            "ManifoldHierarchyTripletLoss",
            "ManifoldRetrievalHierarchyLoss",
            "ManifoldHierarchyEvaluator",
        ):
            assert public_surface in readme
        assert "Recall@K" in readme
        assert "MRR" in readme
        assert "nDCG@K" in readme
        assert "MIT License" in readme
        assert DOCUMENTATION_URL in readme
        assert "user_guide/retrieval_objectives.html" in readme
        assert "user_guide/hierarchy.html" in readme
        assert "user_guide/constant_curvature_semantics.html" in readme
        assert "examples/v08_hierarchy_learning.py" in readme
        assert "examples/v09_constant_curvature_comparison.py" in readme

    assert "Package version v0.9.0" in english
    assert "package version v0.9.0" in japanese
    assert "fixed-curvature v0.3 path backward-compatible" in english
    assert "fixed-curvature の v0.3 path と後方互換" in japanese
    assert "not a benchmark or geometry-superiority claim" in english
    assert "geometry-superiority claim" in japanese


def test_readmes_defer_detailed_geometry_guidance_to_read_the_docs() -> None:
    english = _read("README.md")
    japanese = _read("docs/README_ja.md")
    geometry = _read("docs/user_guide/constant_curvature_semantics.rst")

    for readme in (english, japanese):
        assert "$$" not in readme
        assert "user_guide/constant_curvature_semantics.html" in readme

    for term in (
        "Choosing a geometry",
        "Compatibility with existing workflows",
        "sectional_curvature",
        "SphereProjection",
        "Stereographic",
        "float64",
        "Apple MPS",
        "examples/v09_constant_curvature_comparison.py",
    ):
        assert term in geometry
    assert "does **not** by itself guarantee better embedding quality" in geometry


def test_public_api_preserves_prior_contracts_and_exposes_v09_constructor() -> None:
    required_public_names = {
        "ManifoldSentenceTransformer",
        "ManifoldMultipleNegativesRankingLoss",
        "ManifoldPrototypeHierarchyLoss",
        "ManifoldPrototypes",
        "ManifoldTrainer",
        "ManifoldEmbeddingEvaluator",
        "ManifoldPrototypeAssignmentEvaluator",
        "ManifoldCorpusRetrievalEvaluator",
        "exact_corpus_search",
        "mine_hard_negatives",
        "ManifoldTripletLoss",
        "ManifoldMarginMSELoss",
        "ManifoldDistanceMSELoss",
        "ManifoldSymmetricMultipleNegativesRankingLoss",
        "ManifoldGradedCorpusRetrievalEvaluator",
        "ManifoldRadialOrderLoss",
        "ManifoldDepthLoss",
        "ManifoldHierarchyTripletLoss",
        "ManifoldRetrievalHierarchyLoss",
        "ManifoldHierarchyEvaluator",
    }
    assert required_public_names.issubset(set(neembed.__all__))

    constructor = signature(ManifoldSentenceTransformer).parameters
    assert tuple(constructor) == (
        "model_name_or_path",
        "manifold",
        "embedding_dim",
        "curvature",
        "learnable_curvature",
        "sectional_curvature",
    )
    assert constructor["manifold"].default == "poincare"
    assert constructor["curvature"].default == 1.0
    assert constructor["learnable_curvature"].default is False
    assert constructor["sectional_curvature"].default is None

    mnrl = signature(ManifoldMultipleNegativesRankingLoss.forward).parameters
    assert tuple(mnrl) == ("self", "anchors", "positives", "negatives")
    assert mnrl["negatives"].default is None
    assert tuple(signature(ManifoldSentenceTransformer.rank).parameters) == (
        "self",
        "query",
        "candidates",
        "top_k",
    )
    assert tuple(signature(exact_corpus_search).parameters) == (
        "model",
        "queries",
        "corpus",
        "top_k",
        "query_chunk_size",
        "corpus_chunk_size",
    )
    assert tuple(signature(mine_hard_negatives).parameters) == (
        "model",
        "queries",
        "corpus",
        "query_ids",
        "corpus_ids",
        "positive_corpus_ids",
        "excluded_corpus_ids",
        "num_negatives",
        "query_chunk_size",
        "corpus_chunk_size",
    )
    assert tuple(signature(ManifoldTripletLoss.forward).parameters) == (
        "self",
        "anchors",
        "positives",
        "negatives",
    )
    assert tuple(signature(ManifoldMarginMSELoss.forward).parameters) == (
        "self",
        "anchors",
        "positives",
        "negatives",
        "target_margin",
    )
    assert tuple(signature(ManifoldDistanceMSELoss.forward).parameters) == (
        "self",
        "texts_a",
        "texts_b",
        "target_distance",
    )
    assert tuple(signature(ManifoldSymmetricMultipleNegativesRankingLoss.forward).parameters) == (
        "self",
        "anchors",
        "positives",
        "negatives",
    )
    assert tuple(signature(ManifoldRadialOrderLoss.forward).parameters) == (
        "self",
        "parents",
        "children",
    )
    assert tuple(signature(ManifoldDepthLoss.forward).parameters) == (
        "self",
        "texts",
        "depths",
    )
    assert tuple(signature(ManifoldHierarchyTripletLoss.forward).parameters) == (
        "self",
        "parents",
        "children",
        "unrelated",
    )
    assert tuple(signature(ManifoldRetrievalHierarchyLoss.forward).parameters) == (
        "self",
        "retrieval_inputs",
        "hierarchy_inputs",
    )

    for evaluator in (
        ManifoldEmbeddingEvaluator,
        ManifoldCorpusRetrievalEvaluator,
        ManifoldPrototypeAssignmentEvaluator,
        ManifoldGradedCorpusRetrievalEvaluator,
        ManifoldHierarchyEvaluator,
    ):
        assert "model" in signature(evaluator).parameters

    fit_doc = ManifoldTrainer.fit.__doc__ or ""
    assert "two- or three-sequence batches" in fit_doc
    assert "margin-regression batches" in fit_doc


def test_release_suite_keeps_prior_paths_and_covers_v09_regressions() -> None:
    required_tests = {
        "test_v04_learnable_structure.py",
        "test_v05_retrieval_example.py",
        "test_v06_exact_retrieval_example.py",
        "test_v07_objective_comparison_example.py",
        "test_v08_hierarchy_learning_example.py",
        "test_euclidean.py",
        "test_sphere_projection.py",
        "test_stereographic.py",
        "test_stereographic_dtype.py",
        "test_stereographic_mps_policy.py",
        "test_stereographic_prototype_initialization.py",
        "test_stereographic_recurse_policy.py",
        "test_v09_curvature_contract.py",
        "test_v09_constant_curvature_comparison_example.py",
        "test_v09_docs.py",
    }
    assert required_tests.issubset(
        {path.name for path in (ROOT / "tests").glob("test_*.py")}
    )

    comparison = _read("tests/test_v09_constant_curvature_comparison_example.py")
    for case_name in (
        "euclidean",
        "poincare",
        "lorentz",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    ):
        assert case_name in comparison
    assert "not a benchmark" in comparison
    assert "superiority" in comparison


def test_release_real_stack_covers_v04_through_v09_paths() -> None:
    workflow = _read(".github/workflows/release.yml")
    v09_stack = _read("tests/integration/test_real_stack_v09.py")

    assert 'HF_HUB_DISABLE_XET: "1"' in workflow
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert 'NEEMBED_REAL_STACK: "1"' in workflow
    assert "timeout-minutes: 5" in workflow
    assert "timeout-minutes: 10" in workflow
    assert "python -m pytest -vv -s --durations=20 tests/integration" in workflow

    for prior_stack in (
        "tests/integration/test_real_stack.py",
        "tests/integration/test_real_stack_learnable_curvature.py",
        "tests/integration/test_real_stack_v05.py",
        "tests/integration/test_real_stack_v06.py",
        "tests/integration/test_real_stack_v07.py",
        "tests/integration/test_real_stack_v08.py",
    ):
        assert (ROOT / prior_stack).is_file()

    for case_name in (
        "euclidean",
        "poincare",
        "lorentz",
        "sphere_projection",
        "stereographic_negative",
        "stereographic_zero",
        "stereographic_positive",
    ):
        assert case_name in v09_stack
    assert "ManifoldMultipleNegativesRankingLoss" in v09_stack
    assert "_assert_finite_backward" in v09_stack
    assert "model.sectional_curvature" in v09_stack
    assert "model.curvature" in v09_stack
    assert "torch.float64" in v09_stack


def test_release_workflow_builds_checks_and_smokes_exact_validated_packages() -> None:
    workflow = _read(".github/workflows/release.yml")
    ci = _read(".github/workflows/ci.yml")
    docs = _read(".github/workflows/docs.yml")

    for python_version in ("3.10", "3.11", "3.12"):
        assert f'- "{python_version}"' in ci
    assert "python -m build" in ci
    assert "python -m twine check dist/*" in ci
    assert "Install built wheel and verify import" in ci
    assert "sphinx" in docs.lower()

    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "python-package-distributions" in workflow
    assert "smoke-testpypi:" in workflow
    assert "smoke-pypi:" in workflow
    assert "--index-url https://test.pypi.org/simple/" in workflow
    assert "neembed-geoopt==${PACKAGE_VERSION}" in workflow
    assert "sha256sum" in workflow
    assert "Verify TestPyPI wheel matches validated source artifact" in workflow
    assert "Verify PyPI wheel matches validated source artifact" in workflow
    for regression in (
        "tests/test_v04_learnable_structure.py",
        "tests/test_v05_retrieval_example.py",
        "tests/test_v06_exact_retrieval_example.py",
        "tests/test_v07_objective_comparison_example.py",
        "tests/test_v08_hierarchy_learning_example.py",
        "tests/test_v09_constant_curvature_comparison_example.py",
    ):
        assert regression in workflow
    assert "from importlib.metadata import version; import neembed" in workflow
    assert "TestPyPI validated source commit ${GITHUB_SHA}" in workflow


def test_release_workflow_keeps_publish_tag_docs_and_release_boundaries() -> None:
    workflow = _read(".github/workflows/release.yml")

    assert (
        "if: github.event_name == 'push' && "
        "startsWith(github.ref, 'refs/tags/v')"
    ) in workflow
    assert (
        "if: github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main'"
    ) in workflow
    assert "Verify release tag points at current main" in workflow
    assert "git fetch --no-tags --depth=1 origin main" in workflow
    assert 'if [ "${GITHUB_SHA}" != "${MAIN_SHA}" ]; then' in workflow
    assert "Verify tag matches package version" in workflow
    assert "environment:\n      name: testpypi" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "https://test.pypi.org/p/neembed-geoopt" in workflow
    assert "https://pypi.org/p/neembed-geoopt" in workflow
    assert "verify-hosted-docs:" in workflow
    assert (
        "https://neembed.readthedocs.io/en/latest/"
        "user_guide/constant_curvature_semantics.html"
    ) in workflow
    assert 'grep -F "v0.9"' in workflow
    assert "create-github-release:" in workflow
    assert "needs: [smoke-pypi, verify-hosted-docs]" in workflow
    assert "contents: write" in workflow
    assert 'gh release create "${GITHUB_REF_NAME}"' in workflow
    assert "v0.9.0 adds backward-compatible constant-curvature geometry selection" in workflow
    assert "not a research benchmark or a claim of geometry superiority" in workflow
    assert "ProductManifold/Scaled mixed-curvature support" in workflow


def test_gitignore_protects_common_public_release_artifacts() -> None:
    gitignore = _read(".gitignore").splitlines()
    required_patterns = {
        ".env.*",
        "*.pem",
        "*.key",
        "*.p12",
        "*.pfx",
        "checkpoints/",
        "artifacts/",
        "outputs/",
    }
    assert required_patterns.issubset(set(gitignore))
