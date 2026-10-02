"""Regression coverage for the v0.9 SphereProjection backend."""

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
    """Tiny deterministic encoder that keeps test points near the origin."""

    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [
                        [0.01, 0.0, 0.0],
                        [0.02, 0.0, 0.0],
                        [-0.01, 0.0, 0.0],
                        [0.005, 0.0, 0.0],
                    ]
                )
            )

        state_path = Path(model_name_or_path) / "encoder.pt"
        if state_path.exists():
            self.load_state_dict(torch.load(state_path, weights_only=True))

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4

    def preprocess(self, sentences: list[str]) -> dict[str, torch.Tensor]:
        rows = [[float(len(sentence)), 0.0, 0.0] for sentence in sentences]
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
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold="sphere_projection",
        embedding_dim=2,
        sectional_curvature=0.5,
    )
    assert isinstance(model.projection, nn.Linear)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.weight[0, 0] = 0.5
        model.projection.weight[1, 1] = 0.25
        model.projection.bias.zero_()
    return model


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


def test_get_manifold_constructs_fixed_positive_sphere_projection() -> None:
    manifold = get_manifold(
        "sphere_projection",
        sectional_curvature=0.5,
    )

    assert isinstance(manifold, geoopt.SphereProjection)
    assert manifold.ndim == 1
    assert float(manifold.k.detach()) == pytest.approx(0.5)
    assert not manifold.k.requires_grad


@pytest.mark.parametrize(
    "sectional_curvature",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_sphere_projection_rejects_invalid_sectional_curvature(
    sectional_curvature: float,
) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        get_manifold(
            "sphere_projection",
            sectional_curvature=sectional_curvature,
        )


def test_sphere_projection_requires_new_fixed_curvature_contract() -> None:
    with pytest.raises(ValueError, match="requires sectional_curvature"):
        get_manifold("sphere_projection")
    with pytest.raises(ValueError, match="legacy poincare/lorentz"):
        get_manifold(
            "sphere_projection",
            curvature=2.0,
            sectional_curvature=0.5,
        )
    with pytest.raises(ValueError, match="not supported for sphere_projection"):
        get_manifold(
            "sphere_projection",
            learnable=True,
            sectional_curvature=0.5,
        )


def test_sphere_projection_forward_uses_origin_expmap_and_returns_valid_points(
    monkeypatch,
) -> None:
    model = _make_model(monkeypatch)
    texts = ["a", "abcd"]
    features = model.encoder.preprocess(texts)
    encoder_output = model.encoder(features)["sentence_embedding"]
    tangent = model.projection(encoder_output)
    expected = model.manifold.expmap0(tangent.to(dtype=torch.float64))

    actual = model(texts)

    assert actual.shape == (2, 2)
    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, expected)
    assert model.manifold.check_point_on_manifold(actual)
    assert model.sectional_curvature == pytest.approx(0.5)
    with pytest.raises(AttributeError):
        _ = model.curvature


def test_sphere_projection_distance_is_finite_and_nonnegative(monkeypatch) -> None:
    model = _make_model(monkeypatch)
    embeddings = model(["a", "abcd", "abcdefgh"])

    distances = model.manifold.dist(
        embeddings[:1],
        embeddings[1:],
    )

    assert distances.shape == (2,)
    assert torch.isfinite(distances).all()
    assert torch.all(distances >= 0.0)
    assert torch.all(distances > 0.0)


def test_sphere_projection_v07_retrieval_objectives_forward_and_backward(
    monkeypatch,
) -> None:
    cases = [
        lambda model: ManifoldMultipleNegativesRankingLoss(model)(
            ["a", "aaaa"], ["aa", "aaaaaaaa"]
        ),
        lambda model: ManifoldSymmetricMultipleNegativesRankingLoss(model)(
            ["a", "aaaa"], ["aa", "aaaaaaaa"]
        ),
        lambda model: ManifoldTripletLoss(model, margin=10.0)(
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


def test_sphere_projection_exact_search_evaluator_and_mining(monkeypatch) -> None:
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


def test_sphere_projection_save_load_round_trip_preserves_signed_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _make_model(monkeypatch)
    before = model.encode(["a", "abcd"], convert_to_tensor=True)
    save_path = tmp_path / "sphere-projection-model"

    model.save_pretrained(save_path)
    config = json.loads((save_path / "neembed_config.json").read_text(encoding="utf-8"))
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["a", "abcd"], convert_to_tensor=True)

    assert config == {
        "embedding_dim": 2,
        "manifold": "sphere_projection",
        "sectional_curvature": pytest.approx(0.5),
    }
    assert loaded.manifold_name == "sphere_projection"
    assert loaded.sectional_curvature == pytest.approx(0.5)
    assert torch.allclose(before, after)
