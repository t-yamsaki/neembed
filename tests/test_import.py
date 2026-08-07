"""Smoke tests for the minimal neembed package scaffold."""

import importlib


def test_package_and_scaffold_modules_import() -> None:
    modules = (
        "neembed",
        "neembed.model",
        "neembed.manifolds",
        "neembed.losses",
        "neembed.trainer",
    )

    for module_name in modules:
        assert importlib.import_module(module_name) is not None
