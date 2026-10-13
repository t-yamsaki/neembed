"""Small, versioned experiment protocol; dataset/model adapters stay external.

The runner controls seed resets, validation-only selection and one final test
call. It does not implement a real-data/model adapter or assert model quality.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import subprocess
from typing import Any, Protocol

VERSION = 1


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _keys(value: Any, required: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(f"{label} must have exactly these keys: {sorted(required)}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")


def _integer(value: Any, label: str, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _number(value: Any, label: str, *, positive: bool = False) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    if positive and value <= 0:
        raise ValueError(f"{label} must be positive")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate format v1 without downloads; return an independent copy.

    Hashes describe caller-supplied artifacts, not proof that a real adapter
    loaded them. A real adapter must verify bytes and report observed state.
    """
    _keys(config, {"format_version", "benchmark", "kind", "model", "inputs",
                   "dataset", "seeds", "variants", "training", "selection",
                   "execution"}, "config")
    if type(config["format_version"]) is not int or config["format_version"] != VERSION:
        raise ValueError("unsupported config format_version")
    _text(config["benchmark"], "benchmark")
    if config["kind"] not in ("fixture", "real"):
        raise ValueError("kind must be fixture or real")
    model = config["model"]
    _keys(model, {"id", "revision", "snapshot_sha256", "native_dim", "encoder_dtype"}, "model")
    _text(model["id"], "model.id")
    if not isinstance(model["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model["revision"]):
        raise ValueError("model.revision must be an immutable 40/64-character hash")
    if not isinstance(model["snapshot_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", model["snapshot_sha256"]):
        raise ValueError("model.snapshot_sha256 must fingerprint the initial encoder artifact")
    _integer(model["native_dim"], "model.native_dim")
    if model["encoder_dtype"] not in ("float32", "float64"):
        raise ValueError("unsupported encoder_dtype")
    _keys(config["inputs"], {"queries", "documents"}, "inputs")
    for name, options in config["inputs"].items():
        _keys(options, {"task", "prompt_name", "prompt"}, f"inputs.{name}")
        for key, value in options.items():
            if value is not None and not isinstance(value, str):
                raise ValueError(f"inputs.{name}.{key} must be string or null")
        if options["task"] not in (None, "query", "document"):
            raise ValueError("task must be null, query or document")
        if options["prompt"] is not None and options["prompt_name"] is not None:
            raise ValueError("prompt and prompt_name are mutually exclusive")
    data = config["dataset"]
    _keys(data, {"id", "revision", "source", "license", "split_policy", "splits"}, "dataset")
    for key in ("id", "revision", "source", "license", "split_policy"):
        _text(data[key], f"dataset.{key}")
    _keys(data["splits"], {"train", "validation", "test"}, "dataset.splits")
    for name, split in data["splits"].items():
        _keys(split, {"id", "sha256", "size", "ids"}, f"split.{name}")
        _text(split["id"], f"split.{name}.id")
        if not isinstance(split["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", split["sha256"]):
            raise ValueError("split.sha256 must be a SHA256 digest")
        _integer(split["size"], "split.size")
        ids = split["ids"]
        if not isinstance(ids, list) or len(ids) != split["size"]:
            raise ValueError("split.ids must list all example IDs in artifact order")
        for item in ids:
            _text(item, "example ID")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate example IDs")
    splits = list(data["splits"].values())
    if len({s["id"] for s in splits}) != 3:
        raise ValueError("train, validation and test split IDs must differ")
    for i, split in enumerate(splits):
        for other in splits[i + 1:]:
            if set(split["ids"]) & set(other["ids"]):
                raise ValueError("train, validation and test example IDs must be disjoint")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or len(seeds) < 3:
        raise ValueError("at least three seeds are required")
    for seed in seeds:
        _integer(seed, "seed", 0)
        if seed >= 2**32:
            raise ValueError("seed must fit NumPy's uint32 seed range")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    training = config["training"]
    _keys(training, {"encoder_policy", "objective", "optimizer", "learning_rate",
                     "weight_decay", "epochs", "batch_size", "max_steps", "trials_per_variant"}, "training")
    if training["encoder_policy"] not in ("trainable", "frozen"):
        raise ValueError("encoder_policy must be trainable or frozen")
    if training["objective"] not in ("mnrl", "triplet") or training["optimizer"] != "AdamW":
        raise ValueError("supported plan: mnrl/triplet with AdamW")
    _number(training["learning_rate"], "learning_rate", positive=True)
    _number(training["weight_decay"], "weight_decay")
    if training["weight_decay"] < 0:
        raise ValueError("weight_decay must be nonnegative")
    for key in ("epochs", "batch_size", "max_steps", "trials_per_variant"):
        _integer(training[key], key)
    selection = config["selection"]
    _keys(selection, {"metric", "direction", "metrics"}, "selection")
    _text(selection["metric"], "selection.metric")
    if selection["direction"] not in ("maximize", "minimize"):
        raise ValueError("selection.direction must be maximize or minimize")
    metrics = selection["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("selection.metrics must be a nonempty list")
    for metric in metrics:
        _text(metric, "metric name")
    if len(set(metrics)) != len(metrics) or selection["metric"] not in metrics:
        raise ValueError("metrics must be unique and include selection.metric")
    _keys(config["execution"], {"device", "deterministic"}, "execution")
    if not isinstance(config["execution"]["device"], str) or not re.fullmatch(r"cpu|mps|cuda(?::\d+)?", config["execution"]["device"]):
        raise ValueError("execution.device must be cpu, mps or cuda[:index]")
    if type(config["execution"]["deterministic"]) is not bool:
        raise ValueError("execution.deterministic must be boolean")
    variants = config["variants"]
    if not isinstance(variants, list) or not variants:
        raise ValueError("variants must be nonempty")
    names = []
    dims = []
    for variant in variants:
        _keys(variant, {"name", "geometry", "distance", "intrinsic_dim", "output_dim",
                        "embedding_dtype", "curvature", "trials"}, "variant")
        _text(variant["name"], "variant.name")
        names.append(variant["name"])
        geometry = variant["geometry"]
        allowed = {"native": {"cosine"}, "euclidean": {"cosine", "euclidean"},
                   "poincare": {"geodesic"}, "lorentz": {"geodesic"}}
        if not isinstance(geometry, str) or geometry not in allowed or not isinstance(variant["distance"], str) or variant["distance"] not in allowed[geometry]:
            raise ValueError("invalid geometry/distance combination")
        _integer(variant["intrinsic_dim"], "intrinsic_dim")
        _integer(variant["output_dim"], "output_dim")
        expected = variant["intrinsic_dim"] + (geometry == "lorentz")
        if variant["output_dim"] != expected:
            raise ValueError("output_dim must account for Lorentz's time coordinate")
        if geometry == "native":
            if variant["intrinsic_dim"] != model["native_dim"]:
                raise ValueError("native variant must use the encoder's native width")
        else:
            dims.append(variant["intrinsic_dim"])
        expected_dtype = model["encoder_dtype"] if geometry == "native" else (
            "float64" if geometry == "lorentz" else "float32"
        )
        if variant["embedding_dtype"] != expected_dtype:
            raise ValueError(f"v1 {geometry} embedding_dtype must be {expected_dtype}")
        if geometry in ("poincare", "lorentz"):
            _number(variant["curvature"], "curvature", positive=True)
            if variant["curvature"] != 1.0:
                raise ValueError("v1 fixes Poincare and Lorentz curvature magnitude at 1.0")
        elif variant["curvature"] is not None:
            raise ValueError("native/Euclidean curvature must be null")
        trials = variant["trials"]
        if not isinstance(trials, list) or len(trials) != training["trials_per_variant"]:
            raise ValueError("each variant must receive the same tuning trial budget")
        tuning_key = "temperature" if training["objective"] == "mnrl" else "margin"
        for trial in trials:
            _keys(trial, {tuning_key}, "trial")
            _number(trial[tuning_key], tuning_key, positive=True)
        if len({canonical_hash(t) for t in trials}) != len(trials):
            raise ValueError("duplicate tuning trials")
    if len(names) != len(set(names)) or len(set(dims)) > 1:
        raise ValueError("variant names must be unique; projected intrinsic dimensions must match")
    json.dumps(config, allow_nan=False)
    return deepcopy(config)


def load_config(path: str | Path) -> dict[str, Any]:
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def environment() -> dict[str, Any]:
    """Report installed versions and machine; requested device is in config."""
    versions = {}
    for name in ("neembed-geoopt", "torch", "sentence-transformers", "transformers", "geoopt", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    hardware = {"platform": platform.platform(), "machine": platform.machine(),
                "processor": platform.processor() or None, "cpu_count": os.cpu_count()}
    try:
        import torch
    except ImportError:
        hardware["accelerators"] = None
    else:
        hardware["accelerators"] = {"cuda_version": torch.version.cuda,
                                    "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                                    "mps_available": torch.backends.mps.is_available()}
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": sys.version, "dependencies": versions, "hardware": hardware,
            "source_commit": commit, "source_dirty": dirty,
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def set_seed(seed: int, deterministic: bool) -> None:
    """Reset supported RNGs before every trial and final test evaluation."""
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(deterministic)
        torch.backends.cudnn.benchmark = False


@dataclass
class TrialResult:
    """Adapter output; metrics are finite numbers or explicitly unmeasured null.

    Observed fields must describe actual execution, not echo a planned model.
    artifact is retained only in memory for the selected trial's test call.
    """
    artifact: Any
    validation: dict[str, float | None]
    diagnostics: dict[str, float | None]
    observed: dict[str, Any]


class Backend(Protocol):
    kind: str
    name: str

    def fit_validate(self, context: dict[str, Any]) -> TrialResult: ...
    def test(self, artifact: Any, split: dict[str, Any]) -> dict[str, float | None]: ...


def _metrics(values: Any, names: list[str]) -> dict[str, float | None]:
    _keys(values, set(names), "metrics")
    for name, value in values.items():
        if value is not None:
            _number(value, name)
    return deepcopy(values)


def _error(error: Exception, stage: str) -> dict[str, str]:
    return {"stage": stage, "type": type(error).__name__, "message": str(error)}


def _observed(value: Any, context: dict[str, Any], kind: str) -> dict[str, Any]:
    _keys(value, {"model_snapshot_sha256", "encoder_trainable", "intrinsic_dim", "output_dim",
                  "encoder_dtype", "embedding_dtype", "device", "optimizer_steps", "hardware", "note"}, "observed")
    _integer(value["optimizer_steps"], "observed.optimizer_steps", 0)
    if value["optimizer_steps"] > context["training"]["max_steps"]:
        raise ValueError("adapter exceeded max_steps")
    _text(value["device"], "observed.device")
    if not isinstance(value["hardware"], dict):
        raise ValueError("observed.hardware must describe actual execution hardware")
    if value["note"] is not None and not isinstance(value["note"], str):
        raise ValueError("observed.note must be string or null")
    if kind == "real":
        _integer(value["intrinsic_dim"], "observed.intrinsic_dim")
        _integer(value["output_dim"], "observed.output_dim")
        expected = {"model_snapshot_sha256": context["model"]["snapshot_sha256"],
                    "encoder_trainable": context["training"]["encoder_policy"] == "trainable",
                    "intrinsic_dim": context["variant"]["intrinsic_dim"],
                    "output_dim": context["variant"]["output_dim"],
                    "encoder_dtype": context["model"]["encoder_dtype"],
                    "embedding_dtype": context["variant"]["embedding_dtype"]}
        if type(value["encoder_trainable"]) is not bool or any(value[k] != v for k, v in expected.items()):
            raise ValueError("observed model state/dimensions/dtypes/policy differ from plan")
        frozen_native = context["variant"]["geometry"] == "native" and not value["encoder_trainable"]
        if frozen_native:
            if value["optimizer_steps"] != 0:
                raise ValueError("frozen native references must execute zero optimizer steps")
        elif value["optimizer_steps"] == 0:
            raise ValueError("trained variants must execute at least one optimizer step")
        if context["execution"]["device"] == "cuda":
            matches = re.fullmatch(r"cuda(?::\d+)?", value["device"])
        else:
            matches = value["device"] == context["execution"]["device"]
        if not matches:
            raise ValueError("observed geometry device differs from requested device")
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def run_protocol(config: dict[str, Any], backend: Backend, output_dir: str | Path) -> list[Path]:
    """Run equal trial budgets, select on validation, test once per seed/variant.

    Normal adapter errors are recorded and other trials/seeds continue. Null
    primary scores cannot win; ties select the first configured trial. Failed
    final tests are not retried with other trials (which would leak test data).
    Existing seed files are refused. Each seed JSON stands alone.
    """
    config = validate_config(config)
    if backend.kind != config["kind"]:
        raise ValueError("backend.kind must match config.kind")
    output_dir = Path(output_dir)
    paths = [output_dir / f"seed-{seed}.json" for seed in config["seeds"]]
    if any(path.exists() for path in paths):
        raise FileExistsError("seed result already exists; use a new output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = environment()
    names = config["selection"]["metrics"]
    primary = config["selection"]["metric"]
    for seed, path in zip(config["seeds"], paths):
        result = {"format_version": VERSION, "config": config, "config_sha256": canonical_hash(config),
                  "seed": seed, "backend": {"name": backend.name, "kind": backend.kind},
                  "environment": metadata, "runs": []}
        for variant in config["variants"]:
            run = {"variant": variant["name"], "status": "unmeasured", "trials": [],
                   "selected_trial": None, "test": {name: None for name in names}, "error": None}
            best_score = None
            best: TrialResult | None = None
            for index, trial in enumerate(variant["trials"]):
                record = {"index": index, "parameters": trial, "status": "failed",
                          "validation": {name: None for name in names}, "diagnostics": None,
                          "observed": None, "error": None}
                context = deepcopy({"model": config["model"], "inputs": config["inputs"],
                                    "dataset": {k: v for k, v in config["dataset"].items() if k != "splits"},
                                    "train": config["dataset"]["splits"]["train"],
                                    "validation": config["dataset"]["splits"]["validation"],
                                    "seed": seed, "variant": variant, "trial": trial,
                                    "training": config["training"], "execution": config["execution"]})
                try:
                    set_seed(seed, config["execution"]["deterministic"])
                    expected_context = deepcopy(context)
                    fitted = backend.fit_validate(context)
                    scores = _metrics(fitted.validation, names)
                    for value in fitted.diagnostics.values():
                        if value is not None:
                            _number(value, "diagnostic")
                    observed = _observed(fitted.observed, expected_context, backend.kind)
                    record.update(status="completed" if all(v is not None for v in scores.values()) else "partial",
                                  validation=scores, diagnostics=deepcopy(fitted.diagnostics), observed=observed)
                    score = scores[primary]
                    if score is not None and (best_score is None or
                            (score > best_score if config["selection"]["direction"] == "maximize" else score < best_score)):
                        best, best_score = fitted, score
                        run["selected_trial"] = index
                except Exception as error:
                    record["error"] = _error(error, "fit_validate")
                run["trials"].append(record)
            if best is not None:
                try:
                    set_seed(seed, config["execution"]["deterministic"])
                    run["test"] = _metrics(backend.test(best.artifact, deepcopy(config["dataset"]["splits"]["test"])), names)
                    run["status"] = "completed" if all(v is not None for v in run["test"].values()) else "partial"
                except Exception as error:
                    run["status"] = "failed"
                    run["error"] = _error(error, "test")
            elif all(t["status"] == "failed" for t in run["trials"]):
                run["status"] = "failed"
            result["runs"].append(run)
        # Exclusive write prevents accidental overwrites of previous measurements.
        with path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    return paths
