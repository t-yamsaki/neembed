"""Regression tests for v0.10 ProductManifold sentence embeddings."""

import json
from pathlib import Path

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer
from neembed.product_runtime import product_geometry_device, product_geometry_dtype


class FakeSentenceTransformer(nn.Module):
    """Tiny trainable encoder with a local save/load surface."""

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


def _two_component_config():
    return [
        {
            "name": "negative",
            "manifold": "poincare",
            "intrinsic_dim": 2,
            "curvature": 0.5,
        },
        {
            "name": "flat",
            "manifold": "euclidean",
            "intrinsic_dim": 2,
        },
    ]


def _three_component_config():
    return [
        {
            "name": "hyperboloid",
            "manifold": "lorentz",
            "intrinsic_dim": 2,
            "curvature": 0.5,
        },
        {
            "name": "spherical",
            "manifold": "sphere_projection",
            "intrinsic_dim": 2,
            "sectional_curvature": 0.25,
        },
        {
            "name": "flat",
            "manifold": "euclidean",
            "intrinsic_dim": 1,
            "scale": 2.0,
        },
    ]


@pytest.mark.parametrize(
    ("config", "expected_shape", "expected_dtype"),
    [
        (_two_component_config(), (2, 4), torch.float32),
        (_three_component_config(), (2, 6), torch.float64),
    ],
)
def test_product_forward_returns_valid_geoopt_points(
    monkeypatch,
    config,
    expected_shape,
    expected_dtype,
) -> None:
    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer("fake-model", product_config=config)

    embeddings = model(["dog", "mammal"])
    valid, reason = model.manifold.check_point_on_manifold(
        embeddings,
        explain=True,
    )

    assert isinstance(model.manifold, geoopt.ProductManifold)
    assert embeddings.shape == expected_shape
    assert embeddings.dtype == expected_dtype
    assert torch.isfinite(embeddings).all()
    assert valid, reason
    assert model.embedding_dim == model.product_config.projection_dim
    assert embeddings.shape[-1] == model.product_config.ambient_dim


def test_product_distance_matches_geoopt_directly(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        product_config=_three_component_config(),
    )
    embeddings = model.encode(["a", "longer"], convert_to_tensor=True)

    expected = model.manifold.dist(embeddings[0], embeddings[1])
    actual = model.distance(embeddings[0], embeddings[1])

    assert actual.dtype == torch.float64
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_product_forward_backward_has_finite_nonzero_gradients(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        product_config=_three_component_config(),
    )

    loss = model(["dog", "mammal", "animal"]).square().mean()
    loss.backward()

    assert model.projection.weight.grad is not None
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
    assert torch.count_nonzero(model.projection.weight.grad) > 0
    assert torch.count_nonzero(model.encoder.linear.weight.grad) > 0


def test_product_dtype_and_mps_device_policy(monkeypatch) -> None:
    _patch_encoder(monkeypatch)
    simple = ManifoldSentenceTransformer(
        "fake-model",
        product_config=_two_component_config(),
    )
    double = ManifoldSentenceTransformer(
        "fake-model",
        product_config=_three_component_config(),
    )

    assert product_geometry_dtype(simple.product_config, torch.float32) == torch.float32
    assert product_geometry_dtype(double.product_config, torch.float32) == torch.float64
    assert product_geometry_device(simple.product_config, "mps") == torch.device("mps")
    assert product_geometry_device(double.product_config, "mps") == torch.device("cpu")


def test_product_constructor_rejects_conflicting_single_geometry_arguments(
    monkeypatch,
) -> None:
    _patch_encoder(monkeypatch)
    config = _two_component_config()

    with pytest.raises(ValueError, match="embedding_dim must match"):
        ManifoldSentenceTransformer(
            "fake-model",
            embedding_dim=3,
            product_config=config,
        )
    with pytest.raises(ValueError, match="top-level curvature"):
        ManifoldSentenceTransformer(
            "fake-model",
            curvature=2.0,
            product_config=config,
        )
    with pytest.raises(ValueError, match="learnable_curvature"):
        ManifoldSentenceTransformer(
            "fake-model",
            learnable_curvature=True,
            product_config=config,
        )
    with pytest.raises(ValueError, match="requires product_config"):
        ManifoldSentenceTransformer("fake-model", manifold="product")


def test_product_save_load_round_trip_preserves_order_parameters_and_embeddings(
    monkeypatch,
    tmp_path,
) -> None:
    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        "fake-model",
        product_config=_three_component_config(),
    )
    before = model.encode(["dog", "mammal"], convert_to_tensor=True)
    projection_before = {
        name: tensor.detach().clone()
        for name, tensor in model.projection.state_dict().items()
    }
    save_path = tmp_path / "saved-product"

    model.save_pretrained(save_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(save_path)
    after = loaded.encode(["dog", "mammal"], convert_to_tensor=True)
    saved_config = json.loads(
        (save_path / "neembed_config.json").read_text(encoding="utf-8")
    )

    assert saved_config["manifold"] == "product"
    assert saved_config["embedding_dim"] == 5
    assert saved_config["product_config"] == model.product_config.to_dict()
    assert loaded.manifold_name == "product"
    assert loaded.product_config == model.product_config
    assert loaded.product_config.component_names == (
        "hyperboloid",
        "spherical",
        "flat",
    )
    for name, tensor in loaded.projection.state_dict().items():
        assert torch.equal(tensor, projection_before[name])
    assert before.shape == after.shape == (2, 6)
    assert torch.allclose(before, after, atol=1e-10, rtol=1e-10)
