"""Documentation contract checks for the v0.3 manifold workflow."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_v03_docs_cover_manifold_selection_and_dimensions() -> None:
    index = _read("docs/index.rst")
    quickstart = _read("docs/getting_started/quickstart.rst")
    architecture = _read("docs/user_guide/architecture.rst")

    assert "Poincare and Lorentz" in index
    assert 'manifold="poincare"' in quickstart
    assert 'manifold="lorentz"' in quickstart
    assert "examples/train_lorentz.py" in quickstart
    assert "embedding_dim" in architecture
    assert "D + 1" in architecture
    assert "k = 1 / c" in architecture
    assert "float64" in architecture


def test_v03_docs_cover_save_load_and_three_way_benchmark() -> None:
    saving = _read("docs/user_guide/saving_loading.rst")
    evaluation = _read("docs/user_guide/evaluation.rst")

    assert "Poincare and Lorentz" in saving
    assert "neembed_config.json" in saving
    assert "manifold" in saving
    assert "curvature" in saving
    assert "Euclidean vs Poincare vs Lorentz" in evaluation
    assert "python experiments/compare_euclidean_poincare.py" in evaluation
    assert "engineering and regression reference" in evaluation
    assert "generally superior" in evaluation


def test_english_and_japanese_readmes_surface_v03_examples() -> None:
    english = _read("README.md")
    japanese = _read("docs/README_ja.md")

    for readme in (english, japanese):
        assert "examples/train_lorentz.py" in readme
        assert "Euclidean-vs-Poincaré-vs-Lorentz" in readme
