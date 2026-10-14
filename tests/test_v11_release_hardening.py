"""Executable checks of the release's hosted-doc gate and installed-wheel scope."""
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).parents[1]


def _hosted_script():
    workflow = (ROOT / ".github/workflows/release.yml").read_text()
    block = workflow.split("      - name: Verify Read the Docs deployed the release-specific v0.11 build\n", 1)[1]
    block = block.split("\n  create-github-release:", 1)[0]
    return textwrap.dedent(block.split("        run: |\n", 1)[1])


@pytest.mark.parametrize("broken", [None, "api_contracts", "encoder_compatibility", "benchmark_protocol", "missing"])
@pytest.mark.skipif(os.name != "posix", reason="release workflow uses Ubuntu/Bash")
def test_hosted_docs_gate_rejects_stale_missing_or_incomplete_v11_pages(tmp_path, broken):
    pages = {
        "constant_curvature_semantics": "<title>neembed 0.11.0</title> v0.9",
        "mixed_curvature": "<title>neembed 0.11.0</title> v0.10",
        "api_contracts": "<title>neembed 0.11.0</title> encode_query batch_size",
        "encoder_compatibility": "<title>neembed 0.11.0</title> 5.4.1",
        "benchmark_protocol": "<title>neembed 0.11.0</title> optimizer_steps",
    }
    if broken == "missing":
        pages.pop("api_contracts")
    elif broken == "encoder_compatibility":
        pages[broken] = pages[broken].replace("0.11.0", "0.10.0")
    elif broken:
        pages[broken] = "<title>neembed 0.11.0</title> unrelated content"
    fixtures = tmp_path / "pages.json"
    fixtures.write_text(json.dumps(pages))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text(f"#!{sys.executable}\n" + textwrap.dedent('''\
        import json, os, pathlib, sys
        args = sys.argv[1:]
        url = next(arg for arg in args if arg.startswith('https://'))
        page = url.rsplit('/', 1)[-1].removesuffix('.html')
        pages = json.loads(pathlib.Path(os.environ['NEEMBED_DOC_FIXTURES']).read_text())
        if page not in pages:
            sys.exit(22)
        pathlib.Path(args[args.index('-o') + 1]).write_text(pages[page])
    '''))
    curl.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               GITHUB_REF_NAME="v0.11.0", NEEMBED_DOC_FIXTURES=str(fixtures))
    script = _hosted_script().replace("/tmp/neembed-", str(tmp_path / "neembed-"))
    result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
    assert result.returncode == (0 if broken is None else 1), result.stderr
    assert ("matches neembed 0.11.0" if broken is None else "did not expose") in result.stdout


def test_installed_wheel_and_published_smokes_cover_v11_and_legacy_state():
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    release = (ROOT / ".github/workflows/release.yml").read_text()
    installed_step = ci.split("      - name: Install built wheel and verify import", 1)[1]
    assert 'Path(neembed.__file__).resolve().is_relative_to(Path("src").resolve())' in installed_step
    assert 'HF_HUB_OFFLINE: "1"' in installed_step
    for smoke in (installed_step, *release.split("      - name: Verify installed version and v0.4-v0.11 smoke")[1:]):
        for filename in ("test_v11_role_example.py", "test_role_prompts.py", "test_encode_batching.py", "test_model_metadata.py"):
            assert "tests/" + filename in smoke
    assert release.count("      - name: Verify installed version and v0.4-v0.11 smoke") == 2
    notes = (ROOT / ".github/RELEASE_NOTES.md").read_text()
    assert notes.startswith("# v0.11.0\n")
    for term in ("Versionless v0.10", "batch_size=32", "format 1", "no real-data performance conclusion"):
        assert term in notes
