"""Release metadata and public API contract tests."""

from pathlib import Path
import tomllib

import neembed


ROOT = Path(__file__).parents[1]


def _project() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as file:
        return tomllib.load(file)["project"]


def test_distribution_and_import_names_remain_distinct() -> None:
    project = _project()

    assert project["name"] == "neembed-geoopt"
    assert neembed.__name__ == "neembed"


def test_python_support_and_documentation_metadata_are_stable() -> None:
    project = _project()

    assert project["requires-python"] == ">=3.10"
    assert project["urls"]["Documentation"] == "https://neembed.readthedocs.io/en/latest/"
    assert {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }.issubset(project["classifiers"])


def test_v02_evaluator_is_part_of_the_public_api() -> None:
    assert "ManifoldEmbeddingEvaluator" in neembed.__all__
    assert neembed.ManifoldEmbeddingEvaluator.__module__ == "neembed.evaluator"
