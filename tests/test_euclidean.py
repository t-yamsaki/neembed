"""Regression coverage for the v0.9 Euclidean baseline."""

import json
from pathlib import Path

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldDistanceMSELoss,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss,
    ManifoldTripletLoss,
    exact_corpus_search,
    mine_hard_negatives,
)
from neembed.manifolds import get_manifold


class FakeSentenceTransformer(nn.Module):
    """Tiny local encoder with deterministic text-length features."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

        state_path = Path(model_name_or_path) / "encoder.pt"
        if state_path.exists():
            self.load_state_dict(torch.load(state_path, weights_only=True))

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [[float(len(sentence)), 1.0, -1.0] for sentence in sentences]
        return {"input_features": torch.tensor(rows)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}

    def save_pretrained(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "encoder.pt")


def _patch_encoder(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)


def _make_model(monkeypatch) -> ManifoldSentenceTransformer:
    _patch_encoder(monkeypatch)
    torch.manual_seed(7)
    return ManifoldSentenceTransformer(
        "fake-model",
        manifold="euclidean",
        embedding_dim=2,
        sectional_curvature=0.0,
    )


def _assert_active_backward(model: ManifoldSentenceTransformer, loss: torch.Tensor) -> None:
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert isinstance(model.projection, nn.Linear)
    gradient = model.projection.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_get_manifold_constructs_vector_euclidean_geometry() -> None:
    manifold = get_manifold("euclidean", sectional_curvature=0.0)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[4.0, 6.0], [3.0, 8.0]])

    assert isinstance(manifold, geoopt.Euclidean)
    assert manifold.ndim == 1
    assert torch.allclose(
        manifold.dist(x, y),
        torch.linalg.vector_norm(x - y, dim=-1),
    )


@pytest.mark.parametrize("sectional_curvature", [-1.0, 1.0, float("inf"), float("nan")])
def test_euclidean_rejects_nonzero_or_nonfinite_sectional_curvature(
    sectional_curvature: float,
) -> None:
    with pytest.raises(ValueError, match="sectional_curvature must be exactly 0.0"):
        get_manifold("euclidean", sectional_curvature=sectional_curvature)


def test_euclidean_rejects_legacy_or_learnable_curvature_inputs() -> None:
    with pytest.raises(ValueError, match="legacy poincare/lorentz"):
        get_manifold("euclidean", curvature=2.0)
    with pytest.raises(ValueError, match="not supported for euclidean"):
        get_manifold("euclidean", learnable=True)
    with pytest.raises(ValueError, match="must be None"):
        get_manifold("poincare", sectional_curvature=-1.0)


def test_euclidean_forward_is_projection_output_without_manifold_mapping(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    texts = ["a", "abcd"]
    features = model.encoder.preprocess(texts)
    encoder_output = model.encoder(features)["sentence_embedding"]
    expected = model.projection(encoder_output)

    actual = model(texts)

    assert actual.shape == (2, 2)
    assert torch.allclose(actual, expected)
    assert model.sectional_curvature == 0.0
    with pytest.raises(AttributeError, match="only defined for poincare and lorentz"):
        _ = model.curvature


def test_euclidean_model_distance_matches_torch_l2_norm(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = torch.tensor([[4.0, 6.0], [3.0, 8.0]])

    actual = model.distance(x, y)
    expected = torch.linalg.vector_norm(x - y, dim=-1)

    assert torch.allclose(actual, expected)


def test_euclidean_v07_retrieval_objectives_forward_and_backward(monkeypatch) -> None:
    cases = [
        lambda model: ManifoldMultipleNegativesRankingLoss(model)(
            ["a", "aaaa"], ["aa", "aaaaaaaa"]
        ),
        lambda model: ManifoldSymmetricMultipleNegativesRankingLoss(model)(
            ["a", "aaaa"], ["aa", "aaaaaaaa"]
        ),
        lambda model: ManifoldTripletLoss(model, margin=2.0)(
            ["a", "aaaa"],
            ["aaaaaaaa", "aaaaaaaaaa"],
            ["aa", "aaaaa"],
        ),
        lambda model: ManifoldMarginMSELoss(model)(
            ["a", "aaaa"],
            ["aa", "aaaaa"],
            ["aaaaaaaa", "aaaaaaaaaa"],
            target_margin=10.0,
        ),
        lambda model: ManifoldDistanceMSELoss(model)(
            ["a", "aaaa"],
            ["aaaaaaaa", "aaaaaaaaaa"],
            target_distance=10.0,
        ),
    ]

    for make_loss in cases:
        model = _make_model(monkeypatch)
        _assert_active_backward(model, make_loss(model))


def test_euclidean_exact_search_evaluator_and_mining_use_public_api(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    corpus = ["a", "cc", "bbb", "dddd"]
    corpus_ids = ["c0", "c1", "c2", "c3"]

    search_results = exact_corpus_search(
        model,
        ["a"],
        corpus,
        top_k=2,
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    assert [result["index"] for result in search_results[0]] == [0, 1]
    assert all(result["distance"] >= 0.0 for result in search_results[0])

    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model,
        query_ids=["q0"],
        queries=["a"],
        corpus_ids=corpus_ids,
        corpus=corpus,
        relevance={"q0": ["c0"]},
        recall_at_k=(1, 2),
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    metrics = evaluator()
    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert metrics["mrr"] == pytest.approx(1.0)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())

    mined = mine_hard_negatives(
        model,
        ["a"],
        corpus,
        query_ids=["q0"],
        corpus_ids=corpus_ids,
        positive_corpus_ids={"q0": ["c0"]},
        num_negatives=1,
        query_chunk_size=1,
        corpus_chunk_size=2,
    )
    assert mined[0][0]["corpus_id"] == "c1"
    assert mined[0][0]["distance"] >= 0.0


def test_euclidean_save_load_round_trip_uses_sectional_curvature_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _make_model(monkeypatch)
    before = model.encode(["a", "abcd"], convert_to_tensor=True)
    save_path = tmp_path / "euclidean-model"

    model.save_pretrained(save_path)
    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["a", "abcd"], convert_to_tensor=True)

    assert config == {
        "embedding_dim": 2,
        "manifold": "euclidean",
        "sectional_curvature": 0.0,
    }
    assert loaded.manifold_name == "euclidean"
    assert loaded.sectional_curvature == 0.0
    assert torch.allclose(before, after)
