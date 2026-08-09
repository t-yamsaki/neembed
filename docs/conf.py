"""Sphinx configuration for the neembed documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "neembed"
author = "taishi-yamasaki"
copyright = "2026, taishi-yamasaki"

_version_match = re.search(
    r'^version\s*=\s*"([^"]+)"',
    (ROOT / "pyproject.toml").read_text(encoding="utf-8"),
    flags=re.MULTILINE,
)
if _version_match is None:
    raise RuntimeError("Could not read project version from pyproject.toml")
version = _version_match.group(1)
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_title = f"neembed {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/t-yamsaki/neembed/",
    "source_branch": "main",
    "source_directory": "docs/",
}

autodoc_mock_imports = [
    "geoopt",
    "sentence_transformers",
    "torch",
]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

language = "en"
pygments_style = "sphinx"
pygments_dark_style = "monokai"
