"""Offline checks of the real-data protocol, including failure preservation."""
from copy import deepcopy
import json
from pathlib import Path
import random
import socket
import subprocess
import sys

import pytest

from experiments.benchmark_protocol import canonical_hash, load_config, run_protocol, validate_config
from experiments.run_benchmark import FixtureBackend, main

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/configs/protocol_fixture.json"


@pytest.fixture
def config():
    return load_config(CONFIG)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("protocol fixtures must not use the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def records(paths):
    return [json.loads(path.read_text()) for path in paths]


@pytest.mark.parametrize("path,value", [
    (("format_version",), 2), (("format_version",), True),
    (("model", "revision"), "main"), (("model", "snapshot_sha256"), "unknown"),
    (("seeds",), [0, 1]), (("seeds",), [0, 0, 2]), (("seeds",), [True, 1, 2]),
    (("seeds",), [0, 1, 2**32]),
    (("dataset", "splits", "test", "ids"), ["val-dog", "test-bird"]),
    (("dataset", "splits", "test", "size"), 3),
    (("dataset", "splits", "test", "sha256"), "bad"),
    (("inputs", "queries", "task"), "corpus"),
    (("inputs", "queries", "prompt_name"), "query"),
    (("variants", 4, "output_dim"), 2), (("variants", 4, "embedding_dtype"), "float32"),
    (("variants", 3, "curvature"), 0.5), (("variants", 4, "curvature"), 2.0),
    (("variants", 1, "intrinsic_dim"), 3),
    (("variants", 0, "intrinsic_dim"), 2),
    (("variants", 0, "trials"), [{"temperature": 0.1}]),
    (("variants", 0, "trials"), [{"temperature": 0.1}, {"temperature": 0.1}]),
    (("variants", 0, "trials", 0, "temperature"), float("nan")),
    (("training", "batch_size"), True), (("training", "weight_decay"), -1),
    (("selection", "metric"), "not-listed"), (("execution", "deterministic"), 1),
])
def test_invalid_plans_are_rejected(config, path, value):
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_config_copy_and_triplet_budget(config):
    validated = validate_config(config)
    validated["model"]["id"] = "changed"
    assert config["model"]["id"] == "lexical-overlap-fixture"
    config["training"]["objective"] = "triplet"
    for variant in config["variants"]:
        variant["trials"] = [{"margin": 0.1}, {"margin": 1.0}]
    validate_config(config)
    config["unexpected"] = True
    with pytest.raises(ValueError):
        validate_config(config)


def test_fixture_seed_files_preserve_provenance_and_unmeasured_values(config, tmp_path):
    first = records(run_protocol(config, FixtureBackend(config), tmp_path / "first"))
    second = records(run_protocol(config, FixtureBackend(config), tmp_path / "second"))
    for seed, left, right in zip(config["seeds"], first, second):
        assert left["seed"] == seed
        assert left["config"] == config
        assert left["config_sha256"] == canonical_hash(config)
        assert left["backend"]["kind"] == "fixture"
        assert left["runs"] == right["runs"]
        assert left["environment"]["dependencies"]["torch"]
        assert left["environment"]["hardware"]["cpu_count"]
        assert len(left["runs"]) == 5
        for run in left["runs"]:
            assert len(run["trials"]) == 2
            assert run["selected_trial"] == 0  # stable validation tie ordering
            assert run["status"] == "partial"
            assert run["test"]["fixture_recall_at_1"] == 1
            assert run["test"]["unmeasured_example"] is None
            for trial in run["trials"]:
                assert trial["diagnostics"]["final_training_loss"] is None
                assert trial["observed"]["output_dim"] is None
                assert trial["observed"]["optimizer_steps"] == 0
    with pytest.raises(FileExistsError):
        run_protocol(config, FixtureBackend(config), tmp_path / "first")


class TrackingBackend(FixtureBackend):
    def __init__(self, config):
        super().__init__(config)
        self.fits = []
        self.tests = []

    def fit_validate(self, context):
        assert "test" not in context
        assert "splits" not in context["dataset"]
        assert "test-dog" not in json.dumps(context)
        self.fits.append(deepcopy(context))
        result = super().fit_validate(context)
        result.artifact = {"seed": context["seed"], "temperature": context["trial"]["temperature"],
                           "rng": random.random()}
        result.validation["fixture_recall_at_1"] = context["trial"]["temperature"]
        return result

    def test(self, artifact, split):
        self.tests.append((deepcopy(artifact), deepcopy(split)))
        return {"fixture_recall_at_1": 0.0, "unmeasured_example": None}


def test_validation_selection_equal_budget_and_rng_reset(config, tmp_path):
    backend = TrackingBackend(config)
    result = records(run_protocol(config, backend, tmp_path))
    assert len(backend.fits) == 3 * 5 * 2
    assert len(backend.tests) == 3 * 5
    assert all(artifact["temperature"] == 0.1 for artifact, _ in backend.tests)
    assert all(split == config["dataset"]["splits"]["test"] for _, split in backend.tests)
    assert all(run["selected_trial"] == 1 for seed in result for run in seed["runs"])
    assert all(run["test"]["fixture_recall_at_1"] == 0 for seed in result for run in seed["runs"])
    for seed in config["seeds"]:
        assert len({artifact["rng"] for artifact, _ in backend.tests if artifact["seed"] == seed}) == 1


class FailureBackend(TrackingBackend):
    def fit_validate(self, context):
        if context["seed"] == 0:
            raise RuntimeError("training failed")
        result = super().fit_validate(context)
        if context["seed"] == 1:
            result.validation["fixture_recall_at_1"] = None
        return result

    def test(self, artifact, split):
        self.tests.append((artifact, split))
        raise RuntimeError("test failed")


def test_failure_and_missing_scores_never_become_success_or_zero(config, tmp_path):
    config["variants"] = config["variants"][:1]
    backend = FailureBackend(config)
    failed, missing, test_failed = records(run_protocol(config, backend, tmp_path))
    run = failed["runs"][0]
    assert run["status"] == "failed" and run["selected_trial"] is None
    assert all(t["error"]["message"] == "training failed" for t in run["trials"])
    assert all(v is None for v in run["test"].values())
    assert missing["runs"][0]["status"] == "unmeasured"
    assert missing["runs"][0]["selected_trial"] is None
    run = test_failed["runs"][0]
    assert run["status"] == "failed" and run["selected_trial"] == 1
    assert run["error"] == {"stage": "test", "type": "RuntimeError", "message": "test failed"}
    assert all(v is None for v in run["test"].values())
    assert len(backend.tests) == 1  # no alternate trial fallback after seeing test


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True])
def test_nonfinite_or_boolean_metrics_fail_explicitly(config, tmp_path, bad_value):
    backend = TrackingBackend(config)
    config["variants"] = config["variants"][:1]
    original = backend.fit_validate
    def invalid(context):
        fitted = original(context)
        fitted.validation["fixture_recall_at_1"] = bad_value
        return fitted
    backend.fit_validate = invalid
    outputs = records(run_protocol(config, backend, tmp_path))
    assert all(s["runs"][0]["status"] == "failed" for s in outputs)
    assert backend.tests == []


def test_fixture_identity_and_real_backend_mismatch_rejected(config, tmp_path):
    config["dataset"]["splits"]["train"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact"):
        FixtureBackend(config)
    config = load_config(CONFIG)
    backend = FixtureBackend(config)
    config["kind"] = "real"
    with pytest.raises(ValueError, match="backend.kind"):
        run_protocol(config, backend, tmp_path)


def test_real_adapter_must_report_matching_observed_state(config, tmp_path):
    backend = TrackingBackend(config)
    config["kind"] = backend.kind = "real"
    config["variants"] = config["variants"][:1]
    result = records(run_protocol(config, backend, tmp_path))
    assert all(s["runs"][0]["status"] == "failed" for s in result)
    assert all("observed" in s["runs"][0]["trials"][0]["error"]["message"] for s in result)
    assert backend.tests == []


class ObservedStubBackend(TrackingBackend):
    """Contract stub, not a trained real-model implementation."""
    kind = "real"

    def fit_validate(self, context):
        fitted = super().fit_validate(context)
        fitted.observed = {"model_snapshot_sha256": context["model"]["snapshot_sha256"],
                           "encoder_trainable": context["training"]["encoder_policy"] == "trainable",
                           "intrinsic_dim": context["variant"]["intrinsic_dim"],
                           "output_dim": context["variant"]["output_dim"],
                           "encoder_dtype": context["model"]["encoder_dtype"],
                           "embedding_dtype": context["variant"]["embedding_dtype"],
                           "device": "cpu", "optimizer_steps": 0,
                           "hardware": {"implementation": "contract test stub"}, "note": None}
        return fitted


def test_real_contract_and_minimize_selection(config, tmp_path):
    config["kind"] = "real"
    config["selection"]["direction"] = "minimize"
    backend = ObservedStubBackend(config)
    outputs = records(run_protocol(config, backend, tmp_path))
    assert all(run["selected_trial"] == 0 for seed in outputs for run in seed["runs"])
    assert all(run["status"] == "partial" for seed in outputs for run in seed["runs"])
    assert all(artifact["temperature"] == 0.05 for artifact, _ in backend.tests)
    assert outputs[0]["runs"][4]["trials"][0]["observed"]["output_dim"] == 3


@pytest.mark.parametrize("key,value", [("model_snapshot_sha256", "f" * 64), ("encoder_trainable", False),
                                      ("output_dim", 99), ("embedding_dtype", "float64"),
                                      ("device", "cuda:0"), ("optimizer_steps", 2)])
def test_real_observation_mismatches_fail_trials(config, tmp_path, key, value):
    config["kind"] = "real"
    config["variants"] = config["variants"][:1]
    backend = ObservedStubBackend(config)
    original = backend.fit_validate
    def mismatched(context):
        fitted = original(context)
        fitted.observed[key] = value
        return fitted
    backend.fit_validate = mismatched
    outputs = records(run_protocol(config, backend, tmp_path))
    assert all(seed["runs"][0]["status"] == "failed" for seed in outputs)
    assert backend.tests == []


def test_adapter_mutation_cannot_change_the_expected_plan(config, tmp_path):
    config["kind"] = "real"
    config["variants"] = config["variants"][:1]
    backend = ObservedStubBackend(config)
    original = backend.fit_validate
    def mutated(context):
        context["variant"]["output_dim"] = 99
        return original(context)
    backend.fit_validate = mutated
    result = records(run_protocol(config, backend, tmp_path))
    assert config["variants"][0]["output_dim"] == 4
    assert all(seed["runs"][0]["status"] == "failed" for seed in result)


def test_cli_validate_and_fixture_outputs(config, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "validate", str(CONFIG)])
    main()
    assert json.loads(capsys.readouterr().out)["valid"] is True
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "run-fixture", str(CONFIG), "--output-dir", str(tmp_path)])
    main()
    cli = json.loads(capsys.readouterr().out)
    assert cli["kind"] == "fixture" and len(cli["seed_files"]) == 3
    # Check direct-file CLI import resolution independently of pytest sys.path.
    result = subprocess.run([sys.executable, str(ROOT / "experiments/run_benchmark.py"), "validate", str(CONFIG)],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["valid"] is True
