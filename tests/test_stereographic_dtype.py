"""Numerical dtype and stability coverage for v0.9 stereographic geometry."""

from pathlib import Path

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldSentenceTransformer,
    exact_corpus_search,
    mine_hard_negatives,
)
from neembed.manifolds import get_manifold


class FakeSentenceTransformer(nn.Module):
    """Tiny deterministic float32 encoder with bounded outputs."""

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
                    ],
                    dtype=torch.float32,
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
        return {"input_features": torch.tensor(rows, dtype=torch.float32)}

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"sentence_embedding": self.linear(features["input_features"])}

    def save_pretrained(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), output_path / "encoder.pt")


def _patch_encoder(monkeypatch) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", FakeSentenceTransformer)


def _make_model(
    monkeypatch,
    manifold: str,
    sectional_curvature: float | None = None,
) -> ManifoldSentenceTransformer:
    _patch_encoder(monkeypatch)
    kwargs = {"manifold": manifold, "embedding_dim": 2}
    if sectional_curvature is not None:
        kwargs["sectional_curvature"] = sectional_curvature
    model = ManifoldSentenceTransformer("fake-model", **kwargs)
    assert isinstance(model.projection, nn.Linear)
    with torch.no_grad():
        model.projection.weight.zero_()
        model.projection.weight[0, 0] = 0.5
        model.projection.weight[1, 1] = 0.25
        model.projection.bias.zero_()
    return model


def test_stereographic_curvature_state_is_float64() -> None:
    sphere = get_manifold("sphere_projection", sectional_curvature=0.5)
    negative = get_manifold("stereographic", sectional_curvature=-0.5)
    zero = get_manifold("stereographic", sectional_curvature=0.0)
    positive = get_manifold("stereographic", sectional_curvature=0.5)

    assert sphere.k.dtype == torch.float64
    assert negative.k.dtype == torch.float64
    assert zero.k.dtype == torch.float64
    assert positive.k.dtype == torch.float64


@pytest.mark.parametrize(
    ("manifold", "sectional_curvature", "expected_dtype"),
    [
        ("poincare", None, torch.float32),
        ("euclidean", 0.0, torch.float32),
        ("lorentz", None, torch.float64),
        ("sphere_projection", 0.5, torch.float64),
        ("stereographic", -0.5, torch.float64),
        ("stereographic", 0.0, torch.float64),
        ("stereographic", 0.5, torch.float64),
    ],
)
def test_model_geometry_dtype_policy_preserves_legacy_paths(
    monkeypatch,
    manifold: str,
    sectional_curvature: float | None,
    expected_dtype: torch.dtype,
) -> None:
    model = _make_model(monkeypatch, manifold, sectional_curvature)

    assert model.encoder.linear.weight.dtype == torch.float32
    assert model.projection.weight.dtype == torch.float32

    embeddings = model(["a", "abcd"])
    assert embeddings.dtype == expected_dtype
    assert torch.isfinite(embeddings).all()


@pytest.mark.parametrize(
    ("manifold", "sectional_curvature"),
    [
        ("sphere_projection", 0.25),
        ("sphere_projection", 4.0),
        ("stereographic", -4.0),
        ("stereographic", -0.25),
        ("stereographic", 0.0),
        ("stereographic", 0.25),
        ("stereographic", 4.0),
    ],
)
def test_stereographic_representative_regimes_have_finite_points_distances_and_gradients(
    monkeypatch,
    manifold: str,
    sectional_curvature: float,
) -> None:
    model = _make_model(monkeypatch, manifold, sectional_curvature)

    embeddings = model(["a", "aaaa", "aaaaaaaa"])
    assert embeddings.dtype == torch.float64
    assert torch.isfinite(embeddings).all()
    assert model.manifold.check_point_on_manifold(embeddings)

    distance = model.manifold.dist(embeddings[0], embeddings[-1])
    assert distance.dtype == torch.float64
    assert torch.isfinite(distance)
    assert distance > 0.0

    model.zero_grad(set_to_none=True)
    distance.backward()
    gradient = model.projection.weight.grad
    assert gradient is not None
    assert gradient.dtype == torch.float32
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize(
    ("manifold", "sectional_curvature"),
    [
        ("sphere_projection", 0.5),
        ("stereographic", -0.5),
        ("stereographic", 0.0),
        ("stereographic", 0.5),
    ],
)
def test_encode_and_distance_keep_stereographic_geometry_in_float64(
    monkeypatch,
    manifold: str,
    sectional_curvature: float,
) -> None:
    model = _make_model(monkeypatch, manifold, sectional_curvature)

    tensor_embeddings = model.encode(
        ["a", "abcd"],
        convert_to_tensor=True,
    )
    array_embeddings = model.encode(["a", "abcd"])
    distance = model.distance(
        torch.tensor([0.01, 0.02], dtype=torch.float32),
        torch.tensor([0.03, -0.01], dtype=torch.float32),
    )

    assert tensor_embeddings.dtype == torch.float64
    assert torch.as_tensor(array_embeddings).dtype == torch.float64
    assert distance.dtype == torch.float64
    assert torch.isfinite(distance)


@pytest.mark.parametrize(
    ("manifold", "sectional_curvature"),
    [
        ("sphere_projection", 0.5),
        ("stereographic", -0.5),
        ("stereographic", 0.0),
        ("stereographic", 0.5),
    ],
)
def test_retrieval_evaluation_and_mining_do_not_downcast_stereographic_embeddings(
    monkeypatch,
    manifold: str,
    sectional_curvature: float,
) -> None:
    model = _make_model(monkeypatch, manifold, sectional_curvature)
    seen_dtypes: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []
    original_distance = model.distance

    def recording_distance(a, b):
        a_dtype = torch.as_tensor(a).dtype
        b_dtype = torch.as_tensor(b).dtype
        result = original_distance(a, b)
        seen_dtypes.append((a_dtype, b_dtype, result.dtype))
        return result

    monkeypatch.setattr(model, "distance", recording_distance)

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

    assert seen_dtypes
    assert all(
        a_dtype == torch.float64
        and b_dtype == torch.float64
        and result_dtype == torch.float64
        for a_dtype, b_dtype, result_dtype in seen_dtypes
    )
