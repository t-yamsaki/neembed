"""Release-readiness checks for public v0.10 metadata and contracts."""

from importlib.metadata import metadata
from inspect import signature
from pathlib import Path

import neembed
from neembed import (
    ManifoldDepthLoss,
    ManifoldDistanceMSELoss,
    ManifoldHierarchyTripletLoss,
    ManifoldHierarchyEvaluator,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
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


def test_v10_public_package_metadata() -> None:
    pyproject = _read("pyproject.toml")
    package_metadata = metadata("neembed-geoopt")
    project_urls = package_metadata.get_all("Project-URL") or []

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'name = "neembed-geoopt"' in pyproject
    assert 'version = "0.10.0"' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '{ name = "taishi-yamasaki" }' in pyproject
    assert f'Documentation = "{DOCUMENTATION_URL}"' in pyproject
    for python_version in ("3.10", "3.11", "3.12"):
        assert f'"Programming Language :: Python :: {python_version}"' in pyproject
    for dependency in ("torch", "sentence-transformers", "geoopt"):
        assert f'    "{dependency}",' in pyproject

    assert package_metadata["Name"] == "neembed-geoopt"
    assert package_metadata["Version"] == "0.10.0"
    assert package_metadata["Requires-Python"] == ">=3.10"
    assert f"Documentation, {DOCUMENTATION_URL}" in project_urls
    assert neembed.__name__ == "neembed"


def test_v10_readmes_preserve_prior_scope_and_publish_product_surface() -> None:
    english = _read("README.md")
    japanese = _read("docs/README_ja.md")
    installation = _read("docs/getting_started/installation.rst")

    for document in (english, japanese, installation):
        assert "pip install neembed-geoopt" in document

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        for release in ("v0.10.0", "v0.9", "v0.8", "v0.7", "v0.6", "v0.5", "v0.4", "v0.3"):
            assert release in readme
        for term in (
            "Poincaré",
            "Lorentz",
            "Euclidean",
            "SphereProjection",
            "Stereographic",
            "ManifoldPrototypes",
            "ManifoldEmbeddingEvaluator",
            "model.rank()",
            "exact_corpus_search()",
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
            "Recall@K",
            "MRR",
            "nDCG@K",
            "MIT License",
            "user_guide/constant_curvature_semantics.html",
            "examples/v09_constant_curvature_comparison.py",
            "ProductManifold",
            "Scaled",
            "user_guide/mixed_curvature.html",
            "examples/v10_mixed_curvature_workflow.py",
        ):
            assert term in readme
        assert DOCUMENTATION_URL in readme

    assert "Package version v0.10.0" in english
    assert "package version v0.10.0" in japanese
    assert "fixed-curvature v0.3 path backward-compatible" in english
    assert "fixed-curvature の v0.3 path と後方互換" in japanese
    assert "not a benchmark or geometry-superiority claim" in english
    assert "geometry-superiority claim" in japanese


def test_v10_geometry_guide_is_the_detailed_release_reference() -> None:
    geometry = _read("docs/user_guide/constant_curvature_semantics.rst")
    normalized_geometry = " ".join(geometry.split())

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
    assert "does **not** by itself guarantee better embedding quality" in normalized_geometry


def test_v10_constructor_extends_without_replacing_prior_public_contracts() -> None:
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
        "ProductComponentConfig",
        "ProductConfig",
        "normalize_product_config",
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
        "product_config",
        "revision",
        "local_files_only",
        "cache_folder",
        "device",
    )
    assert constructor["manifold"].default == "poincare"
    assert constructor["curvature"].default == 1.0
    assert constructor["learnable_curvature"].default is False
    assert constructor["sectional_curvature"].default is None
    assert constructor["product_config"].default is None
    loading_defaults = {
        "revision": None,
        "local_files_only": False,
        "cache_folder": None,
        "device": None,
    }
    restored = signature(ManifoldSentenceTransformer.from_pretrained).parameters
    for name, default in loading_defaults.items():
        assert constructor[name].kind is constructor[name].KEYWORD_ONLY
        assert constructor[name].default is default
        assert restored[name].kind is restored[name].KEYWORD_ONLY
        assert restored[name].default is default
    for api in (
        ManifoldDepthLoss, ManifoldRadialOrderLoss, ManifoldHierarchyTripletLoss,
        ManifoldHierarchyEvaluator,
    ):
        component = signature(api).parameters["component"]
        assert component.kind is component.KEYWORD_ONLY
        assert component.default is None
    assert callable(ManifoldSentenceTransformer.product_distance_diagnostics)

    mnrl = signature(ManifoldMultipleNegativesRankingLoss.forward).parameters
    assert tuple(mnrl) == ("self", "anchors", "positives", "negatives")
    assert mnrl["negatives"].default is None
    assert tuple(signature(ManifoldSentenceTransformer.rank).parameters) == (
        "self",
        "query",
        "candidates",
        "top_k",
        "input_options",
    )
    rank_options = signature(ManifoldSentenceTransformer.rank).parameters["input_options"]
    assert rank_options.kind is rank_options.KEYWORD_ONLY
    assert rank_options.default is None
    assert "query_chunk_size" in signature(exact_corpus_search).parameters
    assert "num_negatives" in signature(mine_hard_negatives).parameters
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
    fit_doc = ManifoldTrainer.fit.__doc__ or ""
    assert "two- or three-sequence batches" in fit_doc
    assert "margin-regression batches" in fit_doc


def test_v10_release_suite_keeps_prior_regressions_and_geometry_matrix() -> None:
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
        "test_product_config.py",
        "test_product_manifold.py",
        "test_product_retrieval.py",
        "test_product_hierarchy.py",
        "test_v10_mixed_curvature_workflow_example.py",
    }
    assert required_tests.issubset({path.name for path in (ROOT / "tests").glob("test_*.py")})

    comparison = _read("tests/test_v09_constant_curvature_comparison_example.py")
    real_stack = _read("tests/integration/test_real_stack_v09.py")
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
        assert case_name in real_stack
    assert "ManifoldMultipleNegativesRankingLoss" in real_stack
    assert "_assert_finite_backward" in real_stack
    assert "model.sectional_curvature" in real_stack
    assert "model.curvature" in real_stack
    assert "torch.float64" in real_stack

    for prior_stack in (
        "tests/integration/test_real_stack.py",
        "tests/integration/test_real_stack_learnable_curvature.py",
        "tests/integration/test_real_stack_v05.py",
        "tests/integration/test_real_stack_v06.py",
        "tests/integration/test_real_stack_v07.py",
        "tests/integration/test_real_stack_v08.py",
    ):
        assert (ROOT / prior_stack).is_file()
    assert (ROOT / "tests/integration/test_real_stack_v10.py").is_file()


def test_v10_release_workflow_builds_and_smokes_validated_artifacts() -> None:
    workflow = _read(".github/workflows/release.yml")
    ci = _read(".github/workflows/ci.yml")
    docs = _read(".github/workflows/docs.yml")

    for python_version in ("3.10", "3.11", "3.12"):
        assert f'- "{python_version}"' in ci
    assert "python -m build" in ci
    assert "python -m twine check dist/*" in ci
    assert "Install built wheel and verify import" in ci
    assert "sphinx" in docs.lower()

    for term in (
        'HF_HUB_DISABLE_XET: "1"',
        'HF_HUB_OFFLINE: "1"',
        'TRANSFORMERS_OFFLINE: "1"',
        'NEEMBED_REAL_STACK: "1"',
        "timeout-minutes: 5",
        "timeout-minutes: 10",
        "python -m pytest -vv -s --durations=20 tests/integration",
        "python -m build",
        "python -m twine check dist/*",
        "python-package-distributions",
        "smoke-testpypi:",
        "smoke-pypi:",
        "--index-url https://test.pypi.org/simple/",
        "neembed-geoopt==${PACKAGE_VERSION}",
        "sha256sum",
        "Verify TestPyPI wheel matches validated source artifact",
        "Verify PyPI wheel matches validated source artifact",
        "tests/test_v09_constant_curvature_comparison_example.py",
        "tests/test_v10_mixed_curvature_workflow_example.py",
        "from importlib.metadata import version; import neembed",
    ):
        assert term in workflow


def test_v10_release_workflow_preserves_publish_and_docs_boundaries() -> None:
    workflow = _read(".github/workflows/release.yml")

    assert "Verify release tag points at current main" in workflow
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
    assert "https://neembed.readthedocs.io/en/latest/user_guide/mixed_curvature.html" in workflow
    assert 'grep -F "${EXPECTED_DOC_TITLE}" /tmp/neembed-mixed-curvature.html' in workflow
    assert 'grep -F "v0.10" /tmp/neembed-mixed-curvature.html' in workflow
    assert "create-github-release:" in workflow
    assert "needs: [smoke-pypi, verify-hosted-docs]" in workflow
    assert 'gh release create "${GITHUB_REF_NAME}"' in workflow
    assert "v0.10.0 adds ProductManifold/Scaled mixed-curvature support" in workflow
    assert "not a research benchmark or a claim of geometry superiority" in workflow
    assert "ProductManifold/Scaled mixed-curvature support" in workflow


def test_gitignore_protects_common_public_release_artifacts() -> None:
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
    assert required_patterns.issubset(set(_read(".gitignore").splitlines()))
