"""Release-readiness checks for public v0.7 metadata and contracts."""

from importlib.metadata import metadata
from inspect import signature
from pathlib import Path

import neembed
from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldDistanceMSELoss,
    ManifoldEmbeddingEvaluator,
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeAssignmentEvaluator,
    ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss,
    ManifoldTrainer,
    ManifoldTripletLoss,
    exact_corpus_search,
    mine_hard_negatives,
)


ROOT = Path(__file__).parents[1]
DOCUMENTATION_URL = "https://neembed.readthedocs.io/en/latest/"


def test_pyproject_declares_v07_public_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'name = "neembed-geoopt"' in pyproject
    assert 'version = "0.7.0"' in pyproject
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


def test_installed_distribution_exposes_v07_metadata() -> None:
    package_metadata = metadata("neembed-geoopt")
    project_urls = package_metadata.get_all("Project-URL") or []

    assert package_metadata["Name"] == "neembed-geoopt"
    assert package_metadata["Version"] == "0.7.0"
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


def test_readmes_describe_v07_and_preserve_prior_release_scope() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        assert "v0.7.0" in readme
        assert "v0.6" in readme
        assert "v0.5" in readme
        assert "v0.4" in readme
        assert "v0.3" in readme
        assert "Lorentz" in readme
        assert "ManifoldEmbeddingEvaluator" in readme
        assert "ManifoldPrototypes" in readme
        assert "Recall@K" in readme
        assert "MRR" in readme
        assert "model.rank()" in readme
        assert "exact_corpus_search()" in readme
        assert "ManifoldCorpusRetrievalEvaluator" in readme
        assert "mine_hard_negatives()" in readme
        assert "ManifoldTripletLoss" in readme
        assert "ManifoldMarginMSELoss" in readme
        assert "ManifoldDistanceMSELoss" in readme
        assert "ManifoldSymmetricMultipleNegativesRankingLoss" in readme
        assert "ManifoldGradedCorpusRetrievalEvaluator" in readme
        assert "nDCG@K" in readme
        assert "ANN" in readme
        assert "MIT License" in readme
        assert DOCUMENTATION_URL in readme
        assert "user_guide/retrieval_objectives.html" in readme

    assert "Package version v0.7.0" in english
    assert "package version v0.7.0" in japanese
    assert "fixed-curvature v0.3 path backward-compatible" in english
    assert "fixed-curvature の v0.3 path と後方互換" in japanese


def test_readmes_defer_detailed_guidance_to_read_the_docs() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")
    retrieval = (ROOT / "docs" / "user_guide" / "retrieval.rst").read_text(
        encoding="utf-8"
    )
    objectives = (
        ROOT / "docs" / "user_guide" / "retrieval_objectives.rst"
    ).read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert DOCUMENTATION_URL in readme
        assert "$$" not in readme
        assert "user_guide/retrieval.html" in readme
        assert "user_guide/retrieval_objectives.html" in readme

    assert "## Training objective" not in english
    assert "## Numerical considerations" not in english
    assert "## 学習目的" not in japanese
    assert "## 数値安定性" not in japanese
    assert "Recall@K" in retrieval
    assert "MRR" in retrieval
    assert "exact_corpus_search()" in retrieval
    assert "mine_hard_negatives()" in retrieval
    assert "v0.7" in objectives
    assert "nDCG" in objectives
    assert "ManifoldTripletLoss" in objectives
    assert "ManifoldMarginMSELoss" in objectives
    assert "ManifoldDistanceMSELoss" in objectives


def test_public_api_preserves_v04_v06_contracts_and_exposes_v07() -> None:
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
    }
    assert required_public_names.issubset(set(neembed.__all__))

    mnrl_parameters = signature(ManifoldMultipleNegativesRankingLoss.forward).parameters
    aligned_evaluator_parameters = signature(ManifoldEmbeddingEvaluator).parameters
    corpus_evaluator_parameters = signature(ManifoldCorpusRetrievalEvaluator).parameters
    prototype_evaluator_parameters = signature(
        ManifoldPrototypeAssignmentEvaluator
    ).parameters
    rank_parameters = signature(ManifoldSentenceTransformer.rank).parameters
    search_parameters = signature(exact_corpus_search).parameters
    mining_parameters = signature(mine_hard_negatives).parameters

    assert tuple(mnrl_parameters) == ("self", "anchors", "positives", "negatives")
    assert mnrl_parameters["negatives"].default is None
    assert "recall_at_k" in aligned_evaluator_parameters
    assert tuple(rank_parameters) == ("self", "query", "candidates", "top_k")
    assert rank_parameters["top_k"].default is None
    assert "prototype_ids" in prototype_evaluator_parameters
    assert "expected_prototype_ids" in prototype_evaluator_parameters

    for name in (
        "query_ids",
        "queries",
        "corpus_ids",
        "corpus",
        "relevance",
        "recall_at_k",
        "query_chunk_size",
        "corpus_chunk_size",
    ):
        assert name in corpus_evaluator_parameters

    assert tuple(search_parameters) == (
        "model",
        "queries",
        "corpus",
        "top_k",
        "query_chunk_size",
        "corpus_chunk_size",
    )
    assert search_parameters["top_k"].default is None
    assert tuple(mining_parameters) == (
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
    assert mining_parameters["excluded_corpus_ids"].default is None
    assert mining_parameters["num_negatives"].default == 1

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
    symmetric_parameters = signature(
        ManifoldSymmetricMultipleNegativesRankingLoss.forward
    ).parameters
    assert tuple(symmetric_parameters) == (
        "self",
        "anchors",
        "positives",
        "negatives",
    )
    assert symmetric_parameters["negatives"].default is None

    graded_parameters = signature(ManifoldGradedCorpusRetrievalEvaluator).parameters
    for name in (
        "model",
        "query_ids",
        "queries",
        "corpus_ids",
        "corpus",
        "graded_relevance",
        "recall_at_k",
        "ndcg_at_k",
        "query_chunk_size",
        "corpus_chunk_size",
    ):
        assert name in graded_parameters

    fit_doc = ManifoldTrainer.fit.__doc__ or ""
    assert "two- or three-sequence batches" in fit_doc
    assert "margin-regression batches" in fit_doc
    assert "(anchors, positives, negatives)" in fit_doc
    assert "(anchors, positives, negatives, target_margin)" in fit_doc


def test_release_suite_keeps_prior_paths_and_covers_v07_regressions() -> None:
    required_tests = {
        "test_hard_negatives.py",
        "test_evaluator.py",
        "test_ranking.py",
        "test_prototype_evaluator.py",
        "test_v04_learnable_structure.py",
        "test_v05_retrieval_example.py",
        "test_retrieval.py",
        "test_exact_corpus_search_batching.py",
        "test_corpus_evaluator.py",
        "test_corpus_evaluator_streaming.py",
        "test_mining.py",
        "test_v06_exact_retrieval_example.py",
        "test_triplet_loss.py",
        "test_margin_mse_loss.py",
        "test_distance_mse_loss.py",
        "test_symmetric_ranking_loss.py",
        "test_graded_corpus_evaluator.py",
        "test_graded_corpus_evaluator_large_grades.py",
        "test_v07_objective_comparison_example.py",
    }
    assert required_tests.issubset(
        {path.name for path in (ROOT / "tests").glob("test_*.py")}
    )

    v07_example = (
        ROOT / "tests" / "test_v07_objective_comparison_example.py"
    ).read_text(encoding="utf-8")
    graded = (ROOT / "tests" / "test_graded_corpus_evaluator.py").read_text(
        encoding="utf-8"
    )
    symmetric = (ROOT / "tests" / "test_symmetric_ranking_loss.py").read_text(
        encoding="utf-8"
    )

    assert "test_v07_objective_comparison_is_deterministic_and_finite" in v07_example
    assert '"ndcg_at_3"' in v07_example
    assert '"triplet"' in v07_example
    assert '"margin_mse"' in v07_example
    assert '"distance_mse"' in v07_example
    assert "ndcg" in graded.lower()
    assert "explicit" in symmetric.lower()
    assert "reverse" in symmetric.lower()


def test_release_real_stack_covers_v04_through_v07_paths() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    real_stack = (ROOT / "tests" / "integration" / "test_real_stack.py").read_text(
        encoding="utf-8"
    )
    learnable_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_learnable_curvature.py"
    ).read_text(encoding="utf-8")
    v05_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_v05.py"
    ).read_text(encoding="utf-8")
    v06_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_v06.py"
    ).read_text(encoding="utf-8")
    v07_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_v07.py"
    ).read_text(encoding="utf-8")

    assert 'HF_HUB_DISABLE_XET: "1"' in workflow
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert 'NEEMBED_REAL_STACK: "1"' in workflow
    assert "timeout-minutes: 5" in workflow
    assert "timeout-minutes: 10" in workflow
    assert "python -m pytest -vv -s --durations=20 tests/integration" in workflow

    assert 'manifold: str = "poincare"' in real_stack
    assert 'manifold="lorentz"' in real_stack
    assert "torch.optim.AdamW" in real_stack
    assert 'learnable_curvature=True' in learnable_stack
    assert '["poincare", "lorentz"]' in learnable_stack
    assert 'pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])' in v05_stack
    assert "ManifoldEmbeddingEvaluator" in v05_stack
    assert "ManifoldPrototypeAssignmentEvaluator" in v05_stack
    assert 'pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])' in v06_stack
    assert "exact_corpus_search" in v06_stack
    assert "ManifoldCorpusRetrievalEvaluator" in v06_stack
    assert "mine_hard_negatives" in v06_stack

    assert 'pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])' in v07_stack
    assert "ManifoldTripletLoss" in v07_stack
    assert "ManifoldMarginMSELoss" in v07_stack
    assert "ManifoldDistanceMSELoss" in v07_stack
    assert "ManifoldSymmetricMultipleNegativesRankingLoss" in v07_stack
    assert "ManifoldGradedCorpusRetrievalEvaluator" in v07_stack
    assert '"ndcg_at_3"' in v07_stack


def test_release_workflow_builds_checks_and_smokes_exact_validated_packages() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    docs = (ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")

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
    assert "tests/test_v04_learnable_structure.py" in workflow
    assert "tests/test_v05_retrieval_example.py" in workflow
    assert "tests/test_v06_exact_retrieval_example.py" in workflow
    assert "tests/test_v07_objective_comparison_example.py" in workflow
    assert "from importlib.metadata import version; import neembed" in workflow
    assert "TestPyPI validated source commit ${GITHUB_SHA}" in workflow


def test_release_workflow_keeps_trusted_publish_tag_and_docs_boundaries() -> None:
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
    assert "https://neembed.readthedocs.io/en/latest/user_guide/retrieval_objectives.html" in workflow
    assert 'grep -F "v0.7"' in workflow
    assert "create-github-release:" in workflow
    assert "needs: [smoke-pypi, verify-hosted-docs]" in workflow
    assert "contents: write" in workflow
    assert 'gh release create "${GITHUB_REF_NAME}"' in workflow
    assert "v0.7.0 adds manifold retrieval objectives" in workflow
    assert "not a research benchmark or a claim of objective superiority" in workflow


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
