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


@pytest.mark.parametrize("sectional", [-0.5, 0.0, 0.25])
def test_stereographic_product_split_map_pack_and_distance(monkeypatch, sectional):
    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    config = _two_component_config() + [{
        "name": "signed", "manifold": "stereographic",
        "intrinsic_dim": 3, "sectional_curvature": sectional,
    }]
    model = ManifoldSentenceTransformer("fake-model", product_config=config)
    embeddings = model(["a", "bb", "ccc"])
    features = model.encoder.preprocess(["a", "bb", "ccc"])
    tangent = model.projection(model.encoder(features)["sentence_embedding"]).double()
    chunks = tangent.split([2, 2, 3], dim=-1)
    expected = model.manifold.pack_point(
        model.manifold.manifolds[0].expmap0(chunks[0]),
        chunks[1],
        model.manifold.manifolds[2].expmap0(chunks[2]),
    )
    assert embeddings.dtype == torch.float64
    assert model.manifold.check_point_on_manifold(embeddings)
    torch.testing.assert_close(embeddings, expected)
    a, b = embeddings[:2, None, :], embeddings[None, :, :]
    distances = model.distance(a.detach().numpy(), b.detach().numpy())
    assert distances.shape == (2, 3)
    torch.testing.assert_close(distances, model.manifold.dist(a, b))
    # Exercise autograd through product geodesics, including the promoted Poincare.
    loss = model.manifold.dist(embeddings[0], embeddings[1:]).sum()
    loss.backward()
    for parameter in (model.encoder.linear.weight, model.projection.weight):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_product_scales_are_metadata_only(monkeypatch):
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", product_config=_three_component_config())
    config = [dict(component, scale=7.0) for component in _three_component_config()]
    scaled = ManifoldSentenceTransformer("fake-model", product_config=config)
    scaled.load_state_dict(model.state_dict())
    a = model.encode(["a", "bbb"], convert_to_tensor=True)
    b = scaled.encode(["a", "bbb"], convert_to_tensor=True)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(model.distance(a[0], a[1]), scaled.distance(b[0], b[1]))


@pytest.mark.parametrize("damage", ["list", "version", "missing_scale", "unknown", "width"])
def test_product_load_rejects_corrupted_metadata(monkeypatch, tmp_path, damage):
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", product_config=_three_component_config())
    model.save_pretrained(tmp_path)
    path = tmp_path / "neembed_config.json"
    saved = json.loads(path.read_text())
    if damage == "list":
        saved["product_config"] = _three_component_config()
    elif damage == "version":
        saved["product_config"]["version"] = 2
    elif damage == "missing_scale":
        del saved["product_config"]["components"][0]["scale"]
    elif damage == "unknown":
        saved["product_config"]["unexpected"] = True
    else:
        saved["embedding_dim"] += 1
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        ManifoldSentenceTransformer.from_pretrained(tmp_path)


def test_product_transforms_preserve_common_double_geometry(monkeypatch):
    _patch_encoder(monkeypatch)
    config = _three_component_config() + [_two_component_config()[0]]
    model = ManifoldSentenceTransformer("fake-model", product_config=config)
    original = {key: value.clone() for key, value in model.manifold.state_dict().items()}
    for method, dtype in [("double", torch.float64), ("float", torch.float32),
                          ("half", torch.float16), ("bfloat16", torch.bfloat16)]:
        getattr(model, method)()
        assert model.projection.weight.dtype == dtype
        for key, value in model.manifold.state_dict().items():
            assert value.dtype == torch.float64
            assert value.device.type == "cpu"
            assert torch.equal(value, original[key])
    model.to(device="cpu", dtype=torch.float32).cpu()
    pointers = {key: value.data_ptr() for key, value in model.state_dict().items()}
    model.to_empty(device="meta", recurse=False)
    assert pointers == {key: value.data_ptr() for key, value in model.state_dict().items()}
    model.to("meta")
    assert all(value.device.type == "meta" for value in model.state_dict().values())
    model.to_empty(device="cpu")
    assert all(value.device.type == "cpu" for value in model.state_dict().values())
    assert all(value.dtype == torch.float64 for value in model.manifold.state_dict().values())


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_product_accelerator_forward_backward_and_cpu_return(monkeypatch, device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} hardware unavailable")
    _patch_encoder(monkeypatch)
    model = ManifoldSentenceTransformer("fake-model", product_config=_three_component_config())
    model = model.cuda() if device == "cuda" else model.to("mps")
    embeddings = model(["a", "bb"])
    geometry_device = "cuda" if device == "cuda" else "cpu"
    assert embeddings.device.type == geometry_device
    assert embeddings.dtype == torch.float64
    assert all(value.device.type == geometry_device for value in model.manifold.state_dict().values())
    embeddings.square().mean().backward()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
    assert torch.isfinite(model.projection.weight.grad).all()
    model.cpu()
    assert model(["a"]).device.type == "cpu"


def test_product_ranking_loss_and_exact_retrieval(monkeypatch):
    from neembed.losses import ManifoldMultipleNegativesRankingLoss
    from neembed.retrieval import exact_corpus_search

    _patch_encoder(monkeypatch)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer("fake-model", product_config=_three_component_config())
    loss = ManifoldMultipleNegativesRankingLoss(model)(["a", "bbb"], ["aa", "bbbb"])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
    assert torch.isfinite(model.projection.weight.grad).all()
    corpus = ["aa", "bbbb", "ccccc"]
    result = exact_corpus_search(model, ["a", "bbb"], corpus, query_chunk_size=1, corpus_chunk_size=2)
    assert result == [model.rank(query, corpus) for query in ["a", "bbb"]]


def test_product_mps_transfer_policy_without_accelerator(monkeypatch):
    class SimulatedMpsEncoder(FakeSentenceTransformer):
        @property
        def device(self):
            return torch.device(getattr(self, "target", "cpu"))

        def _apply(self, fn, recurse=True):
            self.target = "mps"
            return self

    monkeypatch.setattr(model_module, "SentenceTransformer", SimulatedMpsEncoder)
    model = ManifoldSentenceTransformer("fake-model", product_config=_three_component_config())
    # Simulate ordinary child transfers; exercise the real product _apply path.
    monkeypatch.setattr(model.projection, "_apply", lambda fn, recurse=True: model.projection)
    original = {key: value.clone() for key, value in model.manifold.state_dict().items()}
    nn.Sequential(model).to("mps")
    for key, value in model.manifold.state_dict().items():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
        assert torch.equal(value, original[key])
    model.manifold.to("meta")
    model.to_empty(device="mps")
    for value in model.manifold.state_dict().values():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float64
