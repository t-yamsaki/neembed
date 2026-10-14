"""Validate a protocol configuration or exercise it with an offline fixture.

The lexical fixture deliberately does not train encoders or execute manifold
geometry. Its metrics validate orchestration only, never model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

try:
    from .benchmark_protocol import TrialResult, canonical_hash, load_config, run_protocol
except ImportError:  # Direct `python experiments/run_benchmark.py` invocation.
    from benchmark_protocol import TrialResult, canonical_hash, load_config, run_protocol

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "fixtures" / "protocol_retrieval.json"
MODEL_HASH = hashlib.sha256(b"lexical-overlap-fixture-v1").hexdigest()


class FixtureBackend:
    """Non-model fixture adapter with byte-verified splits and real ID ranking."""
    kind = "fixture"
    name = "lexical-overlap-v1"

    def __init__(self, config: dict) -> None:
        expected_model = {"id": "lexical-overlap-fixture", "revision": MODEL_HASH,
                          "snapshot_sha256": MODEL_HASH, "native_dim": 4, "encoder_dtype": "float32"}
        if config["model"] != expected_model or config["dataset"]["id"] != "protocol-retrieval-fixture":
            raise ValueError("fixture backend requires the checked-in fixture model/data identities")
        self.splits = json.loads(FIXTURE.read_text(encoding="utf-8"))
        if config["dataset"]["revision"] != canonical_hash(self.splits):
            raise ValueError("fixture dataset revision does not match artifact")
        self.identities = config["dataset"]["splits"]
        for name, rows in self.splits.items():
            identity = self.identities[name]
            if identity["sha256"] != canonical_hash(rows) or identity["ids"] != [r["id"] for r in rows] or identity["size"] != len(rows):
                raise ValueError(f"fixture {name} artifact does not match configuration")

    def _rows(self, split: dict, name: str) -> list[dict]:
        if split != self.identities[name]:
            raise ValueError(f"unexpected {name} partition")
        return self.splits[name]

    @staticmethod
    def _metrics(rows: list[dict], bias: float) -> dict[str, float | None]:
        correct = 0
        for row in rows:
            words = set(row["query"].split())
            scores = [len(words & set(doc.split())) + bias * i for i, doc in enumerate(row["documents"])]
            best = max(range(len(scores)), key=lambda i: scores[i])
            correct += best == row["positive_index"]
        return {"fixture_recall_at_1": correct / len(rows), "unmeasured_example": None}

    def fit_validate(self, context: dict) -> TrialResult:
        self._rows(context["train"], "train")
        rows = self._rows(context["validation"], "validation")
        # Seeded nuisance value exercises RNG resets; no optimizer or model runs.
        bias = random.uniform(-0.01, 0.01)
        observed = {"model_snapshot_sha256": None, "encoder_trainable": None,
                    "intrinsic_dim": None, "output_dim": None, "encoder_dtype": None,
                    "embedding_dtype": None, "device": "cpu", "optimizer_steps": 0,
                    "hardware": {"implementation": "Python lexical overlap"},
                    "note": "No encoder, projection, loss or geometry was executed."}
        return TrialResult(artifact=bias, validation=self._metrics(rows, bias),
                           diagnostics={"final_training_loss": None, "mean_positive_distance": None},
                           observed=observed)

    def test(self, artifact: float, split: dict) -> dict[str, float | None]:
        return self._metrics(self._rows(split, "test"), artifact)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate any v1 plan without running models")
    validate.add_argument("config", type=Path)
    fixture = subparsers.add_parser("run-fixture", help="exercise the runner; NOT a real benchmark")
    fixture.add_argument("config", type=Path)
    fixture.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "validate":
        print(json.dumps({"valid": True, "config_sha256": canonical_hash(config)}))
    else:
        paths = run_protocol(config, FixtureBackend(config), args.output_dir)
        print(json.dumps({"kind": "fixture", "seed_files": [str(path) for path in paths]}))


if __name__ == "__main__":
    main()
