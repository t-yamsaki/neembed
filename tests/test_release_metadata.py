"""Release-readiness checks for public v0.1 metadata."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_pyproject_declares_v01_public_metadata() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools>=77.0.3"]' in pyproject
    assert 'version = "0.1.0"' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert '{ name = "taishi-yamasaki" }' in pyproject
    assert 'Repository = "https://github.com/t-yamsaki/neembed"' in pyproject
    assert 'Issues = "https://github.com/t-yamsaki/neembed/issues"' in pyproject
    assert "License ::" not in pyproject

    for dependency in ("torch", "sentence-transformers", "geoopt"):
        assert f'    "{dependency}",' in pyproject


def test_readmes_describe_the_implemented_v01_release() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    for readme in (english, japanese):
        assert "TBD" not in readme
        assert "<YOUR_USERNAME>" not in readme
        assert "0.1.0" in readme
        assert "MIT License" in readme
        assert "- [x]" in readme

    assert "https://github.com/t-yamsaki/neembed.git" in english
    assert "https://github.com/t-yamsaki/neembed.git" in japanese


def test_quick_start_avoids_duplicate_in_batch_positives() -> None:
    english = (ROOT / "README.md").read_text(encoding="utf-8")
    japanese = (ROOT / "docs" / "README_ja.md").read_text(encoding="utf-8")

    assert '["mammal", "mammal"]' not in english
    assert '["哺乳類", "哺乳類"]' not in japanese
    assert '(["dog", "cat"], ["mammal", "feline"])' in english
    assert '(["犬", "猫"], ["哺乳類", "ネコ科"])' in japanese


def test_public_api_docstrings_cover_v01_usage_constraints() -> None:
    model = (ROOT / "src" / "neembed" / "model.py").read_text(encoding="utf-8")
    losses = (ROOT / "src" / "neembed" / "losses.py").read_text(encoding="utf-8")
    trainer = (ROOT / "src" / "neembed" / "trainer.py").read_text(encoding="utf-8")

    normalized_losses = " ".join(losses.split())

    assert "NumPy arrays are returned by default" in model
    assert "torch.inference_mode()" in model
    assert "torch.no_grad()" in model
    assert "duplicate positives within one batch should be avoided" in normalized_losses
    assert "Iterable yielding ``(anchors, positives)`` batches" in trainer


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
