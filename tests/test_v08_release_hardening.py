"""Regression checks for the v0.8 release hardening gates."""

import os
from pathlib import Path
import runpy
import subprocess

import pytest
import torch
from torch import nn


ROOT = Path(__file__).parents[1]


class _TinyProjectionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(1.0)


def _load_backward_helper():
    namespace = runpy.run_path(
        str(ROOT / "tests" / "integration" / "test_real_stack_v08.py")
    )
    return namespace["_assert_finite_backward"]


def test_v08_backward_gate_rejects_connected_zero_projection_gradient() -> None:
    assert_finite_backward = _load_backward_helper()
    model = _TinyProjectionModel()
    zero_loss = (model.projection.weight * 0.0).sum()

    with pytest.raises(AssertionError, match="non-zero gradient"):
        assert_finite_backward(model, zero_loss)


def test_v08_backward_gate_accepts_active_finite_projection_gradient() -> None:
    assert_finite_backward = _load_backward_helper()
    model = _TinyProjectionModel()
    active_loss = model.projection.weight.square().sum()

    assert_finite_backward(model, active_loss)


def test_v08_hosted_docs_gate_requires_release_specific_version_marker() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert 'EXPECTED_DOC_TITLE="neembed ${PACKAGE_VERSION}"' in workflow
    assert 'grep -F "${EXPECTED_DOC_TITLE}" /tmp/neembed-hierarchy.html' in workflow

    stale_v07_html = "<title>neembed 0.7.0</title><p>v0.8 hierarchy guidance</p>"
    current_v08_html = "<title>neembed 0.8.0</title><p>v0.8 hierarchy guidance</p>"
    expected_marker = "neembed 0.8.0"

    assert "v0.8" in stale_v07_html
    assert expected_marker not in stale_v07_html
    assert expected_marker in current_v08_html


def test_v08_hosted_docs_gate_derives_version_from_tag_without_checkout(
    tmp_path: Path,
) -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )
    hosted_docs_job = workflow.split("  verify-hosted-docs:\n", 1)[1].split(
        "  create-github-release:\n", 1
    )[0]

    assert 'PACKAGE_VERSION="${GITHUB_REF_NAME#v}"' in hosted_docs_job
    assert "pyproject.toml" not in hosted_docs_job
    assert "actions/checkout" not in hosted_docs_job

    env = dict(os.environ)
    env["GITHUB_REF_NAME"] = "v0.8.0"
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'PACKAGE_VERSION="${GITHUB_REF_NAME#v}"; printf "%s" "${PACKAGE_VERSION}"',
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout == "0.8.0"
