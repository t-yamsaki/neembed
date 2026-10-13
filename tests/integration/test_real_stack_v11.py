"""Pinned MiniLM/E5 qualification for the shared role-aware text APIs."""

from __future__ import annotations

from importlib.metadata import version
import json
import os
from pathlib import Path
import socket

import pytest
import torch

from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    exact_corpus_search,
    mine_hard_negatives,
)


pytestmark = [
    pytest.mark.real_stack,
    pytest.mark.skipif(os.environ.get("NEEMBED_REAL_STACK") != "1", reason="set NEEMBED_REAL_STACK=1 and preload the pinned encoders"),
]
MODELS = json.loads((Path(__file__).with_name("role_models.json")).read_text())
QUERIES = ["What is a dog?", "Which animals can fly?"]
CORPUS = ["A dog is a domesticated mammal.", "Birds can fly using their wings.", "Trees are plants."]
INPUT_OPTIONS = {"queries": {"task": "query"}, "corpus": {"task": "document"}}
GEOMETRIES = [
    pytest.param({"manifold": "poincare", "embedding_dim": 16}, id="poincare"),
    pytest.param({"product_config": [
        {"manifold": "lorentz", "intrinsic_dim": 8, "scale": 2.0},
        {"manifold": "euclidean", "intrinsic_dim": 8},
    ]}, id="product"),
]


def _search(model):
    return exact_corpus_search(model, QUERIES, CORPUS, top_k=3, query_chunk_size=1, corpus_chunk_size=2, input_options=INPUT_OPTIONS)


def _evaluate(model):
    return ManifoldCorpusRetrievalEvaluator(
        model=model, query_ids=["q0", "q1"], queries=QUERIES,
        corpus_ids=["d0", "d1", "d2"], corpus=CORPUS,
        relevance={"q0": {"d0"}, "q1": {"d1"}}, recall_at_k=(1, 3),
        query_chunk_size=1, corpus_chunk_size=2, input_options=INPUT_OPTIONS,
    )()


@pytest.fixture
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("spec", MODELS, ids=[m["name"].split("/")[-1] for m in MODELS])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_pinned_role_training_retrieval_and_offline_checkpoint(spec, geometry, tmp_path, monkeypatch, cpu_threads):
    torch.manual_seed(0)
    print({package: version(package) for package in ("torch", "sentence-transformers", "transformers", "huggingface-hub", "numpy", "geoopt")})
    print(f"encoder={spec['name']} revision={spec['revision']} geometry={geometry}")
    model = ManifoldSentenceTransformer(
        spec["name"], revision=spec["revision"], local_files_only=True, device="cpu", **geometry,
    )
    model.encoder.prompts = dict(spec["prompts"])
    model.encoder.default_prompt_name = None

    # Inspect real token IDs: a forwarded keyword alone would not prove E5's
    # prefix reached the tokenizer, or that it was inserted exactly once.
    original = model.encoder.preprocess
    seen = []
    tokenizer = model.encoder[0].tokenizer

    def verify_prefix(texts, *, prompt=None, task=None):
        expected_prompt = spec["prompts"][task]
        assert prompt == expected_prompt
        features = original(texts, prompt=prompt, task=task)
        expected_ids = tokenizer([expected_prompt + text for text in texts], padding=True, truncation=True,
                                 max_length=model.encoder.max_seq_length, return_tensors="pt")["input_ids"]
        assert torch.equal(features["input_ids"].cpu(), expected_ids)
        if expected_prompt:
            raw_ids = tokenizer(texts, padding=True, truncation=True, max_length=model.encoder.max_seq_length, return_tensors="pt")["input_ids"]
            assert not torch.equal(expected_ids, raw_ids)
        seen.append((task, prompt, len(texts)))
        return features

    monkeypatch.setattr(model.encoder, "preprocess", verify_prefix)
    # Public wrappers must agree with the differentiable shared forward path.
    model.eval()
    for task, texts in (("query", QUERIES), ("document", CORPUS)):
        actual = getattr(model, "encode_" + task)(texts, batch_size=1, convert_to_tensor=True)
        with torch.no_grad():
            expected = model(texts, task=task)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    model.train()
    loss_fn = ManifoldMultipleNegativesRankingLoss(model, temperature=.1, input_options={
        "anchors": {"task": "query"}, "positives": {"task": "document"},
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    encoder_parameter = next(p for p in model.encoder.parameters() if p.requires_grad)
    encoder_before = encoder_parameter.detach().clone()
    projection_before = model.projection.weight.detach().clone()
    loss = loss_fn(QUERIES, CORPUS[:2])
    assert torch.isfinite(loss)
    loss.backward()
    for module in (model.encoder, model.projection):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(torch.count_nonzero(g) > 0 for g in grads)
    optimizer.step()
    assert not torch.equal(encoder_parameter, encoder_before)
    assert not torch.equal(model.projection.weight, projection_before)

    queries = model.encode_query(QUERIES, batch_size=1, convert_to_tensor=True)
    corpus = model.encode_document(CORPUS, batch_size=2, convert_to_tensor=True)
    expected_distances = model.distance(queries[:, None, :], corpus[None, :, :])
    assert torch.isfinite(expected_distances).all()
    results = _search(model)
    for i, rows in enumerate(results):
        expected_order = sorted(range(len(CORPUS)), key=lambda j: (float(expected_distances[i, j]), j))
        assert [r["index"] for r in rows] == expected_order
        for row in rows:
            assert row["distance"] == pytest.approx(float(expected_distances[i, row["index"]]), rel=1e-5, abs=1e-6)
    metrics = _evaluate(model)
    assert all(0 <= value <= 1 for value in metrics.values())
    negatives = mine_hard_negatives(
        model, QUERIES, CORPUS, query_ids=["q0", "q1"], corpus_ids=["d0", "d1", "d2"],
        positive_corpus_ids={"q0": {"d0"}, "q1": {"d1"}}, num_negatives=1, input_options=INPUT_OPTIONS,
    )
    assert negatives[0][0]["corpus_id"] != "d0" and negatives[1][0]["corpus_id"] != "d1"
    assert {task for task, _, _ in seen} == {"query", "document"}

    # Restore the actual loader while retaining evidence of all prior prefixes.
    monkeypatch.setattr(model.encoder, "preprocess", original)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "neembed_config.json").read_text())
    assert config["base_model"] == {"model_id": spec["name"], "revision": spec["revision"]}

    def reject_network(*args, **kwargs):
        pytest.fail("saved encoder restoration attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    restored = ManifoldSentenceTransformer.from_pretrained(tmp_path, local_files_only=True, device="cpu")
    assert restored.encoder.prompts == spec["prompts"]
    for task, texts, expected in (("query", QUERIES, queries), ("document", CORPUS, corpus)):
        actual = getattr(restored, "encode_" + task)(texts, batch_size=1, convert_to_tensor=True)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    after = _search(restored)
    assert [[r["index"] for r in rows] for rows in after] == [[r["index"] for r in rows] for rows in results]
    for old_rows, new_rows in zip(results, after):
        for old, new in zip(old_rows, new_rows):
            assert new["distance"] == pytest.approx(old["distance"], rel=1e-5, abs=1e-6)
    assert _evaluate(restored) == pytest.approx(metrics)
    print(f"loss={float(loss.detach()):.8f} finite_gradients=True checkpoint_agreement=True metrics={metrics}")
