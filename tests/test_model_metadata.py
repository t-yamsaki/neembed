"""Versioned metadata and immutable fixtures produced by the v0.10.0 saver."""

from importlib.metadata import version
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldSentenceTransformer


FIXTURE = json.loads((Path(__file__).parent / "fixtures/v010_models.json").read_text())


class CheckpointEncoder(nn.Module):
    def __init__(self, name, **kwargs):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        self.prompts = {"query": "query: ", "passage": "passage: ", "empty": ""}
        self.default_prompt_name = "passage"
        path = Path(name) / "encoder.pt"
        if path.exists():
            state = torch.load(path, weights_only=True)
            self.linear.to(dtype=state["linear.weight"].dtype)
            self.load_state_dict(state)
        self.to(kwargs.get("device", "cpu"))

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts, *, prompt=None, task=None):
        return {"features": torch.tensor([
            [len((prompt or "") + text)/100, .1, -.1] for text in texts
        ], dtype=self.linear.weight.dtype)}

    def forward(self, features, *, task=None):
        return {"sentence_embedding": self.linear(features["features"])}

    def save_pretrained(self, path):
        # Intentionally only persist weights: neembed must restore prompt metadata.
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "encoder.pt")


@pytest.fixture
def saved(monkeypatch, tmp_path):
    monkeypatch.setattr(model_module, "SentenceTransformer", CheckpointEncoder)
    model = ManifoldSentenceTransformer("org/base-model", revision="fixed-revision", embedding_dim=2)
    model.save_pretrained(tmp_path)
    return model, tmp_path


@pytest.mark.parametrize("name", list(FIXTURE["models"]))
def test_load_fixed_v010_fixtures_and_upgrade_on_save(monkeypatch, tmp_path, name):
    monkeypatch.setattr(model_module, "SentenceTransformer", CheckpointEncoder)
    record = FIXTURE["models"][name]
    (tmp_path / "encoder").mkdir()
    (tmp_path / "neembed_config.json").write_text(json.dumps(record["config"]))
    torch.save({k: torch.tensor(v) for k, v in FIXTURE["encoder_state"].items()}, tmp_path / "encoder/encoder.pt")
    torch.save({k: torch.tensor(v) for k, v in record["projection_state"].items()}, tmp_path / "projection.pt")
    model = ManifoldSentenceTransformer.from_pretrained(tmp_path, local_files_only=True, device="cpu")
    actual = model.encode(FIXTURE["texts"], convert_to_tensor=True)
    torch.testing.assert_close(actual, torch.tensor(record["embeddings"], dtype=actual.dtype), rtol=1e-6, atol=1e-8)
    assert float(model.distance(actual[0], actual[1])) == pytest.approx(record["distance"], rel=1e-5, abs=1e-8)
    assert model.learnable_curvature == record["config"].get("learnable_curvature", False)
    if model.learnable_curvature:
        model(FIXTURE["texts"]).square().sum().backward()
        assert all(p.grad is not None for p in model.manifold.parameters() if p.requires_grad)
    # A read does not rewrite the original fixture/checkpoint.
    assert json.loads((tmp_path / "neembed_config.json").read_text()) == record["config"]
    upgraded = tmp_path / "upgraded"
    model.save_pretrained(upgraded)
    assert json.loads((upgraded / "neembed_config.json").read_text())["format_version"] == 1


@pytest.mark.parametrize("geometry", [
    {"manifold": "poincare", "embedding_dim": 2, "learnable_curvature": True},
    {"manifold": "lorentz", "embedding_dim": 2, "learnable_curvature": True},
    {"manifold": "euclidean"},
    {"manifold": "sphere_projection", "embedding_dim": 2, "sectional_curvature": .5},
    {"manifold": "stereographic", "embedding_dim": 2, "sectional_curvature": -.5},
    {"product_config": [
        {"manifold": "lorentz", "intrinsic_dim": 2, "scale": 2.0},
        {"manifold": "euclidean", "intrinsic_dim": 1},
    ]},
    {"product_config": [
        {"manifold": "poincare", "intrinsic_dim": 2, "scale": 2.0},
        {"manifold": "euclidean", "intrinsic_dim": 1},
    ]},
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_versioned_round_trip_preserves_roles_dtypes_and_distances(monkeypatch, tmp_path, geometry, dtype):
    monkeypatch.setattr(model_module, "SentenceTransformer", CheckpointEncoder)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer("org/base-model", revision="fixed-revision", **geometry).to(dtype=dtype)
    model.encoder.prompts["query"] = "custom query instructions: "
    model.encoder.default_prompt_name = "empty"
    if isinstance(model.projection, nn.Linear) and dtype == torch.float64:
        with torch.no_grad():
            model.projection.weight.add_(1e-9)
    texts = ["dog", "mammal"]
    before = [model.encode(texts, task=task, convert_to_tensor=True) for task in (None, "query", "document")]
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "neembed_config.json").read_text())
    assert config["format_version"] == 1
    assert config["neembed_version"] == version("neembed-geoopt")
    assert config["base_model"] == {"model_id": "org/base-model", "revision": "fixed-revision"}
    assert config["input_config"] == {"prompts": model.encoder.prompts, "default_prompt_name": "empty"}
    restored = ManifoldSentenceTransformer.from_pretrained(tmp_path, local_files_only=True, device="cpu", revision="load-only")
    assert restored.encoder.linear.weight.dtype == dtype
    if isinstance(model.projection, nn.Linear):
        assert restored.projection.weight.dtype == dtype
        assert torch.equal(restored.projection.weight, model.projection.weight)
    for task, expected in zip((None, "query", "document"), before):
        actual = restored.encode(texts, task=task, convert_to_tensor=True)
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(restored.distance(*actual), model.distance(*expected), rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(restored.encode_query(texts, prompt_name="empty", convert_to_tensor=True), before[0], rtol=1e-6, atol=1e-8)
    if model.product_config is not None:
        assert config["product_config"]["version"] == 1
        assert restored.product_config == model.product_config
    resaved = tmp_path / "resaved"
    restored.save_pretrained(resaved)
    assert json.loads((resaved / "neembed_config.json").read_text())["base_model"] == config["base_model"]


@pytest.mark.parametrize("field", ["embedding_dim", "manifold", "neembed_version", "input_config", "dtypes", "base_model", "curvature"])
def test_required_fields_are_rejected_before_loading_encoder(saved, monkeypatch, field):
    _, path = saved
    config = json.loads((path / "neembed_config.json").read_text())
    del config[field]
    (path / "neembed_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(model_module, "SentenceTransformer", lambda *a, **k: pytest.fail("encoder must not load"))
    with pytest.raises(ValueError, match=field):
        ManifoldSentenceTransformer.from_pretrained(path)


@pytest.mark.parametrize("keys,value", [
    (("format_version",), 2), (("format_version",), True), (("format_version",), None),
    (("format_version",), "1"), (("format_version",), 1.0),
    (("embedding_dim",), 0), (("embedding_dim",), True), (("embedding_dim",), 2.5),
    (("manifold",), "future"), (("manifold",), []),
    (("curvature",), None), (("curvature",), False), (("curvature",), float("nan")),
    (("curvature",), float("inf")), (("curvature",), -1), (("learnable_curvature",), "false"),
    (("sectional_curvature",), 0), (("unexpected",), True), (("neembed_version",), ""),
    (("input_config",), []), (("input_config", "prompts"), {"query": None}),
    (("input_config", "default_prompt_name"), "missing"), (("input_config", "default_prompt_name"), []),
    (("dtypes", "encoder"), "int64"), (("dtypes", "projection"), None),
    (("dtypes", "geometry"), None), (("dtypes", "geometry"), []),
    (("base_model", "model_id"), "https://user:secret@example.test/model"),
    (("base_model", "revision"), "https://secret@example.test"), (("base_model",), {}),
])
def test_broken_values_are_rejected_before_loading_encoder(saved, monkeypatch, keys, value):
    _, path = saved
    config = json.loads((path / "neembed_config.json").read_text())
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    (path / "neembed_config.json").write_text(json.dumps(config))
    monkeypatch.setattr(model_module, "SentenceTransformer", lambda *a, **k: pytest.fail("encoder must not load"))
    with pytest.raises(ValueError):
        ManifoldSentenceTransformer.from_pretrained(path)


@pytest.mark.parametrize("text", ["[]", "null", "{}", "{broken", '{"manifold":"poincare","manifold":"lorentz"}'])
def test_malformed_json_and_duplicate_keys_fail_without_loading_encoder(tmp_path, monkeypatch, text):
    (tmp_path / "neembed_config.json").write_text(text)
    monkeypatch.setattr(model_module, "SentenceTransformer", lambda *a, **k: pytest.fail("encoder must not load"))
    with pytest.raises(ValueError):
        ManifoldSentenceTransformer.from_pretrained(tmp_path)


def test_provenance_uses_available_encoder_revision_and_excludes_private_paths(monkeypatch, tmp_path):
    from neembed._model_metadata import source_metadata

    encoder = SimpleNamespace(model_card_data=SimpleNamespace(base_model="org/base", base_model_revision="a"*40))
    assert source_metadata(encoder, "org/base", "main") == {"model_id": "org/base", "revision": "a"*40}
    encoder.model_card_data.base_model = "org/other"
    assert source_metadata(encoder, "org/base", "release/v1") == {"model_id": "org/base", "revision": "release/v1"}
    monkeypatch.setattr(model_module, "SentenceTransformer", CheckpointEncoder)
    for i, source in enumerate((str(tmp_path), "https://user:secret@example.test/model", "/private/cache/secret")):
        model = ManifoldSentenceTransformer(source, revision="sensitive/revision", cache_folder="/private/cache/secret")
        path = tmp_path / str(i)
        model.save_pretrained(path)
        config = json.loads((path / "neembed_config.json").read_text())
        assert config["base_model"] == {"model_id": None, "revision": None}
        assert "secret" not in json.dumps(config) and "sensitive" not in json.dumps(config)


def test_unrepresentable_mixed_encoder_dtype_is_not_silently_saved(saved):
    model, path = saved
    model.encoder.register_buffer("mixed", torch.ones(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="uniform floating dtype"):
        model.save_pretrained(path / "mixed")


def test_projection_checkpoint_dtype_must_agree_with_metadata(saved):
    _, path = saved
    state = torch.load(path / "projection.pt", weights_only=True)
    torch.save({key: value.double() for key, value in state.items()}, path / "projection.pt")
    with pytest.raises(ValueError, match="projection.pt.*dtype"):
        ManifoldSentenceTransformer.from_pretrained(path)
