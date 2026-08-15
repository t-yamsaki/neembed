"""Regression checks for bounded, observable real-stack release verification."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_release_real_stack_preloads_model_and_runs_offline_with_timeouts() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "- name: Preload real-stack sentence model" in workflow
    assert 'SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")' in workflow
    assert 'HF_HUB_DISABLE_XET: "1"' in workflow
    assert "timeout-minutes: 5" in workflow

    assert "- name: Run real-stack acceptance tests" in workflow
    assert 'NEEMBED_REAL_STACK: "1"' in workflow
    assert 'HF_HUB_OFFLINE: "1"' in workflow
    assert 'TRANSFORMERS_OFFLINE: "1"' in workflow
    assert "timeout-minutes: 10" in workflow
    assert "python -m pytest -vv -s --durations=20 tests/integration" in workflow
