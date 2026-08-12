"""Release-readiness checks for public v0.4 metadata and training contracts."""

from importlib.metadata import metadata
from pathlib import Path

import neembed


ROOT = Path(__file__).parents[1]
DOCUMENTATION_URL = "https://neembed.readthedocs.io/en/latest/"


def test_pyproject_declares_v04_public_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'name = "neembed-geoopt"' in pyproject
    assert 'version = "0.4.0"' in pyproject
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


def test_installed_distribution_exposes_release_metadata() -> None:
    package_metadata = metadata("neembed-geoopt")
    project_urls = package_metadata.get_all("Project-URL") or []

    assert package_metadata["Name"] == "neembed-geoopt"
    assert package_metadata["Version"] == "0.4.0"
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


def test_readmes_describe_v04_release_and_v03_compatibility() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        assert "v0.4.0" in readme
        assert "v0.3" in readme
        assert "Lorentz" in readme
        assert "ManifoldEmbeddingEvaluator" in readme
        assert "ManifoldPrototypes" in readme
        assert "ManifoldPrototypeHierarchyLoss" in readme
        assert "Euclidean-vs-Poincaré-vs-Lorentz" in readme
        assert "MIT License" in readme
        assert DOCUMENTATION_URL in readme

    assert "latest tagged / PyPI release is v0.4.0" in english
    assert "最新の tag / PyPI release は v0.4.0" in japanese
    assert "fixed-curvature v0.3 path backward-compatible" in english
    assert "fixed-curvature の v0.3 path と後方互換" in japanese
    assert "https://github.com/t-yamsaki/neembed.git" in english
    assert "https://github.com/t-yamsaki/neembed.git" in japanese


def test_readmes_defer_detailed_guidance_to_read_the_docs() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert DOCUMENTATION_URL in readme
        assert "$$" not in readme

    assert "## Training objective" not in english
    assert "## Numerical considerations" not in english
    assert "## 学習目的" not in japanese
    assert "## 数値安定性" not in japanese


def test_quick_start_avoids_duplicate_in_batch_positives() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    assert '["mammal", "mammal"]' not in english
    assert '["哺乳類", "哺乳類"]' not in japanese
    assert '(["dog", "cat"], ["mammal", "feline"])' in english
    assert '(["犬", "猫"], ["哺乳類", "ネコ科"])' in japanese


def test_public_api_and_persistence_cover_v04_geometry_contracts() -> None:
    public_api = (ROOT / "src" / "neembed" / "__init__.py").read_text(encoding="utf-8")
    model = (ROOT / "src" / "neembed" / "model.py").read_text(encoding="utf-8")
    losses = (ROOT / "src" / "neembed" / "losses.py").read_text(encoding="utf-8")
    trainer = (ROOT / "src" / "neembed" / "trainer.py").read_text(encoding="utf-8")
    evaluator = (ROOT / "src" / "neembed" / "evaluator.py").read_text(encoding="utf-8")

    normalized_losses = " ".join(losses.split())

    for public_name in (
        "ManifoldSentenceTransformer",
        "ManifoldMultipleNegativesRankingLoss",
        "ManifoldPrototypeHierarchyLoss",
        "ManifoldPrototypes",
        "ManifoldTrainer",
        "ManifoldEmbeddingEvaluator",
    ):
        assert public_name in neembed.__all__

    assert "LorentzTrainer" not in public_api
    assert "LorentzEvaluator" not in public_api

    assert 'Supports ``"poincare"`` and ``"lorentz"``' in model
    assert "Lorentz embeddings use one additional ambient coordinate" in model
    assert "Positive, finite magnitude of the negative sectional curvature" in model
    assert "learnable_curvature" in model
    assert "Lorentz outputs" in model and "float64" in model
    assert '"manifold": self.manifold_name' in model
    assert '"curvature": self.curvature' in model
    assert 'config["learnable_curvature"] = True' in model
    assert 'manifold=config["manifold"]' in model
    assert 'curvature=config["curvature"]' in model
    assert 'config.get("learnable_curvature", False)' in model

    assert "NumPy arrays are returned by default" in model
    assert "torch.inference_mode()" in model
    assert "torch.no_grad()" in model
    assert "duplicate positives within one batch should be avoided" in normalized_losses
    assert "every other prototype" in normalized_losses
    assert "Iterable yielding ``(anchors, positives)`` batches" in trainer
    assert "caller-owned optimizer" in trainer

    for metric in (
        "retrieval_accuracy",
        "mean_positive_distance",
        "mean_negative_distance",
    ):
        assert metric in evaluator


def test_release_suite_covers_v04_learnable_structure_contracts() -> None:
    prototypes = (ROOT / "tests" / "test_prototypes.py").read_text(encoding="utf-8")
    riemannian = (ROOT / "tests" / "test_riemannian_training.py").read_text(
        encoding="utf-8"
    )
    hierarchy = (ROOT / "tests" / "test_hierarchy_loss.py").read_text(encoding="utf-8")
    v04_example = (ROOT / "tests" / "test_v04_learnable_structure.py").read_text(
        encoding="utf-8"
    )

    assert "test_state_dict_round_trip_preserves_prototype_points" in prototypes
    assert "test_joint_learnable_curvature_and_prototypes_stay_valid_with_stabilization" in prototypes
    assert "geoopt.optim.RiemannianAdam" in riemannian
    assert "optimized_parameters == expected_parameters" in riemannian
    assert 'pytest.mark.parametrize("manifold_name", ["poincare", "lorentz"])' in hierarchy
    assert "correct" in hierarchy and "corrupt" in hierarchy
    assert "test_v04_example_reports_deterministic_learnable_structure_diagnostics" in v04_example
    assert "learnable[\"prototype_shift\"] > 0.0" in v04_example


def test_release_real_stack_covers_fixed_and_learnable_geometries() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    real_stack = (
        ROOT / "tests" / "integration" / "test_real_stack.py"
    ).read_text(encoding="utf-8")
    learnable_stack = (
        ROOT / "tests" / "integration" / "test_real_stack_learnable_curvature.py"
    ).read_text(encoding="utf-8")

    assert 'NEEMBED_REAL_STACK: "1"' in workflow
    assert "python -m pytest -q tests/integration" in workflow
    assert 'manifold: str = "poincare"' in real_stack
    assert 'manifold="lorentz"' in real_stack
    assert "geoopt.Lorentz" in real_stack
    assert "torch.optim.AdamW" in real_stack
    assert 'learnable_curvature=True' in learnable_stack
    assert '["poincare", "lorentz"]' in learnable_stack
    assert "trainer.optimizer.step()" in learnable_stack


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
    assert "tests/test_import.py tests/test_v04_learnable_structure.py" in workflow
    assert "from importlib.metadata import version; import neembed" in workflow


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
    assert "not a claim that learnable structure is generally superior" in workflow


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
