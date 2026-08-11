"""Release metadata and public API contract tests."""

from importlib.metadata import metadata

import neembed


def _distribution_metadata():
    return metadata("neembed-geoopt")


def test_distribution_and_import_names_remain_distinct() -> None:
    package_metadata = _distribution_metadata()

    assert package_metadata["Name"] == "neembed-geoopt"
    assert neembed.__name__ == "neembed"


def test_python_support_and_documentation_metadata_are_stable() -> None:
    package_metadata = _distribution_metadata()
    project_urls = package_metadata.get_all("Project-URL") or []
    classifiers = package_metadata.get_all("Classifier") or []

    assert package_metadata["Requires-Python"] == ">=3.10"
    assert "Documentation, https://neembed.readthedocs.io/en/latest/" in project_urls
    assert {
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }.issubset(classifiers)


def test_v02_evaluator_is_part_of_the_public_api() -> None:
    assert "ManifoldEmbeddingEvaluator" in neembed.__all__
    assert neembed.ManifoldEmbeddingEvaluator.__module__ == "neembed.evaluator"
