"""Release-readiness checks for public v0.5 metadata and contracts."""

from importlib.metadata import metadata
from inspect import signature
from pathlib import Path

import neembed
from neembed import (
    ManifoldEmbeddingEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldSentenceTransformer,
    ManifoldTrainer,
)


ROOT = Path(__file__).parents[1]
DOCUMENTATION_URL = "https://neembed.readthedocs.io/en/latest/"


def test_pyproject_declares_v05_public_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'name = "neembed-geoopt"' in pyproject
    assert 'version = "0.5.0"' in pyproject
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


def test_installed_distribution_exposes_v05_metadata() -> None:
    package_metadata = metadata("neembed-geoopt")
    project_urls = package_metadata.get_all("Project-URL") or []

    assert package_metadata["Name"] == "neembed-geoopt"
    assert package_metadata["Version"] == "0.5.0"
    assert package_metadata["Requires-Python"] == ">=3.10"
    assert f"Documentation, {DOCUMENTATION_URL}" in project_urls


def test_distribution_name_keeps_neembed_import_package() -> None:
    assert (ROOT / "src" / "neembed" / "__init__.py").is_file()
    assert neembed.__name__ == "neembed"

    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")
    installation = (
        ROOT / "docs" / "getting_started" / "installation.rst"
    ).read_text(encoding="utf-8")

    for document in (english, japanese, installation):
        assert "pip install neembed-geoopt" in document

    assert "from neembed import (" in english
    assert "from neembed import (" in japanese


def test_readmes_describe_v05_and_preserve_compatibility_scope() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        assert "v0.5.0" in readme
        assert "v0.4" in readme
        assert "v0.3" in readme
        assert "Lorentz" in readme
        assert "ManifoldEmbeddingEvaluator" in readme
        assert "ManifoldPrototypes" in readme
        assert "ManifoldPrototypeHierarchyLoss" in readme
        assert "Recall@K" in readme
        assert "MRR" in readme
        assert "model.rank()" in readme
        assert "prototype-assignment" in readme
        assert "Euclidean-vs-Poincaré-vs-Lorentz" in readme
        assert "MIT License" in readme
        assert DOCUMENTATION_URL in readme

    assert "Package version v0.5.0" in english
    assert "package version v0.5.0" in japanese
    assert "development branch toward v0.5" not in english
    assert "v0.5 に向けた development branch" not in japanese
    assert "fixed-curvature v0.3 path backward-compatible" in english
    assert "fixed-curvature の v0.3 path と後方互換" in japanese


def test_readmes_defer_detailed_guidance_to_read_the_docs() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")
    retrieval = (ROOT / "docs" / "user_guide" / "retrieval.rst").read_text(
        encoding="utf-8"
    )

    for readme in (english, japanese):
        assert DOCUMENTATION_URL in readme
        assert "$$" not in readme
        assert "user_guide/retrieval.html" in readme

    assert "## Training objective" not in english
    assert "## Numerical considerations" not in english
    assert "## 学習目的" not in japanese
    assert "## 数値安定性" not in japanese
    assert "Recall@K" in retrieval
    assert "MRR" in retrieval
    assert "in-memory" in retrieval
    assert "neembed does not mine hard negatives automatically" in retrieval


def test_public_api_signatures_cover_v05_retrieval_contracts() -> None:
    for public_name in (
        "ManifoldSentenceTransformer",
        "ManifoldMultipleNegativesRankingLoss",
        "ManifoldPrototypeHierarchyLoss",
        "ManifoldPrototypes",
        "ManifoldTrainer",
        "ManifoldEmbeddingEvaluator",
        "ManifoldPrototypeAssignmentEvaluator",
    ):
        assert public_name in neembed.__all__

    loss_parameters = signature(
        ManifoldMultipleNegativesRankingLoss.forward
    ).parameters
    evaluator_parameters = signature(ManifoldEmbeddingEvaluator).parameters
    prototype_evaluator_parameters = signature(
        ManifoldPrototypeAssignmentEvaluator
    ).parameters
    rank_parameters = signature(ManifoldSentenceTransformer.rank).parameters

    assert tuple(loss_parameters) == ("self", "anchors", "positives", "negatives")
    assert loss_parameters["negatives"].default is None
    assert "recall_at_k" in evaluator_parameters
    assert tuple(rank_parameters) == ("self", "query", "candidates", "top_k")
    assert rank_parameters["top_k"].default is None
    assert "prototype_ids" in prototype_evaluator_parameters
    assert "expected_prototype_ids" in prototype_evaluator_parameters

    fit_doc = ManifoldTrainer.fit.__doc__ or ""
    assert "two- or three-sequence batches" in fit_doc
    assert "(anchors, positives, negatives)" in fit_doc


def test_release_suite_keeps_v04_and_covers_v05_regressions() -> None:
    required_tests = {
        "test_hard_negatives.py",
        "test_evaluator.py",
        "test_ranking.py",
        "test_prototype_evaluator.py",
        "test_v04_learnable_structure.py",
        "test_v05_retrieval_example.py",
    }
    assert required_tests.issubset(
        {path.name for path in (ROOT / "tests").glob("test_*.py")}
    )

    v05_example = (ROOT / "tests" / "test_v05_retrieval_example.py").read_text(
        encoding="utf-8"
    )
    hard_negatives = (ROOT / "tests" / "test_hard_negatives.py").read_text(
        encoding="utf-8"
    )
    ranking = (ROOT / "tests" / "test_ranking.py").read_text(encoding="utf-8")
    prototype_evaluator = (
        ROOT / "tests" / "test_prototype_evaluator.py"
    ).read_text(encoding="utf-8")

    assert "initial_hard_negative_loss" in v05_example
    assert 'results["retrieval"]' in v05_example
    assert 'results["ranked_results"]' in v05_example
    assert 'results["prototype"]' in v05_example
    assert "test_two_sequence_loss_keeps_existing_formula" in hard_negatives
    assert "test_explicit_negatives_are_appended_to_geodesic_candidate_pool" in hard_negatives
    assert 'pytest.mark.parametrize("manifold", ["poincare", "lorentz"])' in ranking
    assert "test_prototype_assignment_evaluator_reports_perfect_assignment" in prototype_evaluator
    assert "test_prototype_assignment_evaluator_rejects_unknown_expected_id" in prototype_evaluator


def test_release_real_stack_covers_v04_and_v05_paths() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    real_stack = (
        ROOT / "tests" / "integration" / "test_real_stack.py"
    ).read_text(encoding="utf-8")
    learnable_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_learnable_curvature.py"
    ).read_text(encoding="utf-8")
    v05_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_v05.py"
    ).read_text(encoding="utf-8")

    assert 'NEEMBED_REAL_STACK: "1"' in workflow
    assert "python -m pytest -vv -s --durations=20 tests/integration" in workflow
    assert 'manifold: str = "poincare"' in real_stack
    assert 'manifold="lorentz"' in real_stack
    assert "torch.optim.AdamW" in real_stack
    assert 'learnable_curvature=True' in learnable_stack
    assert '["poincare", "lorentz"]' in learnable_stack
    assert 'pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])' in v05_stack
    assert "EXPLICIT_NEGATIVES" in v05_stack
    assert "ManifoldEmbeddingEvaluator" in v05_stack
    assert "model.rank(" in v05_stack
    assert "geoopt.optim.RiemannianAdam" in v05_stack
    assert "ManifoldPrototypeAssignmentEvaluator" in v05_stack


def test_release_workflow_builds_checks_and_smokes_published_packages() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m build" in workflow
    assert "python -m twine check dist/*" in workflow
    assert "python-package-distributions" in workflow
    assert "smoke-testpypi:" in workflow
    assert "smoke-pypi:" in workflow
    assert "--index-url https://test.pypi.org/simple/" in workflow
    assert "neembed-geoopt==${PACKAGE_VERSION}" in workflow
    assert "tests/test_v04_learnable_structure.py" in workflow
    assert "tests/test_v05_retrieval_example.py" in workflow
    assert "from importlib.metadata import version; import neembed" in workflow
    assert "TestPyPI validated source commit ${GITHUB_SHA}" in workflow


def test_release_workflow_keeps_trusted_publish_and_tag_boundaries() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert (
        "if: github.event_name == 'push' && "
        "startsWith(github.ref, 'refs/tags/v')"
    ) in workflow
    assert (
        "if: github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main'"
    ) in workflow
    assert "environment:\n      name: testpypi" in workflow
    assert "environment:\n      name: pypi" in workflow
    assert "id-token: write" in workflow
    assert "pypa/gh-action-pypi-publish@release/v1" in workflow
    assert "https://test.pypi.org/p/neembed-geoopt" in workflow
    assert "https://pypi.org/p/neembed-geoopt" in workflow
    assert "create-github-release:" in workflow
    assert "needs: smoke-pypi" in workflow
    assert "contents: write" in workflow
    assert 'gh release create "${GITHUB_REF_NAME}"' in workflow
    assert "v0.5.0 adds caller-supplied explicit hard negatives" in workflow
    assert "not a research benchmark or a claim of geometry superiority" in workflow


def test_gitignore_protects_common_public_release_artifacts() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()

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
