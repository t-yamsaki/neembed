"""Bounded public encoding preserves text order and geometry output contracts."""

import numpy as np
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldSentenceTransformer


class CountingEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        self.prompts = {"query": "query: ", "passage": "passage: ", "custom": "custom: "}
        self.default_prompt_name = "custom"
        self.calls = []
        self.forward_calls = []
        self.fail_on_call = None

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts, *, task=None, prompt=None):
        self.calls.append((list(texts), task, prompt))
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError("encoder failed")
        prompted = [(prompt or "") + text for text in texts]
        rows = [
            [len(text) / 100, sum(map(ord, text)) / 10000, 0.1]
            for text in prompted
        ]
        return {
            "features": torch.tensor(rows, dtype=self.linear.weight.dtype),
            "task": task,
        }

    def forward(self, features, *, task=None):
        assert features["task"] == task
        self.forward_calls.append((len(features["features"]), self.training, torch.is_inference_mode_enabled()))
        return {"sentence_embedding": self.linear(features["features"])}

    def encode(self, *args, **kwargs):
        raise AssertionError("Use neembed's shared forward path")


GEOMETRIES = [
    pytest.param({"manifold": "poincare", "embedding_dim": 2}, 2, False, id="poincare"),
    pytest.param({"manifold": "lorentz", "embedding_dim": 2}, 3, True, id="lorentz"),
    pytest.param({"manifold": "euclidean"}, 4, False, id="euclidean-identity"),
    pytest.param({"manifold": "sphere_projection", "embedding_dim": 2,
                  "sectional_curvature": 0.5}, 2, True, id="sphere-projection"),
    pytest.param({"manifold": "stereographic", "embedding_dim": 2,
                  "sectional_curvature": -0.5}, 2, True, id="stereographic"),
    pytest.param({"product_config": [
        {"manifold": "poincare", "intrinsic_dim": 2, "scale": 2.0},
        {"manifold": "euclidean", "intrinsic_dim": 1},
    ]}, 3, False, id="product-float"),
    pytest.param({"product_config": [
        {"manifold": "lorentz", "intrinsic_dim": 2, "scale": 2.0},
        {"manifold": "sphere_projection", "intrinsic_dim": 1, "sectional_curvature": 0.5},
        {"manifold": "euclidean", "intrinsic_dim": 2},
    ]}, 6, True, id="product-double"),
]
TEXTS = ["cat", "", "dog mammal", "a", "dog", "z", "cat"]


def make_model(monkeypatch, geometry=None, dtype=torch.float32):
    monkeypatch.setattr(model_module, "SentenceTransformer", CountingEncoder)
    torch.manual_seed(0)
    return ManifoldSentenceTransformer("fake", **(geometry or {})).to(dtype=dtype)


@pytest.mark.parametrize("geometry,width,fixed_double", GEOMETRIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("convert_to_tensor", [False, True])
def test_batches_match_full_forward_and_preserve_order(
    monkeypatch, geometry, width, fixed_double, dtype, convert_to_tensor,
):
    model = make_model(monkeypatch, geometry, dtype)
    model.eval()
    with torch.no_grad():
        expected = model(TEXTS)
    for batch_size in (1, 2, len(TEXTS), 20):
        model.encoder.calls.clear()
        model.encoder.forward_calls.clear()
        model.train()
        actual = model.encode(tuple(TEXTS), batch_size=batch_size, convert_to_tensor=convert_to_tensor)
        assert actual.shape == (len(TEXTS), width)
        assert isinstance(actual, torch.Tensor if convert_to_tensor else np.ndarray)
        actual_tensor = actual if convert_to_tensor else torch.from_numpy(actual)
        assert actual_tensor.dtype == (torch.float64 if fixed_double else dtype)
        assert actual_tensor.device == expected.device
        torch.testing.assert_close(actual_tensor, expected, rtol=1e-5, atol=1e-7)
        assert torch.isfinite(actual_tensor).all()
        assert [text for texts, _, _ in model.encoder.calls for text in texts] == TEXTS
        assert all(0 < len(texts) <= batch_size for texts, _, _ in model.encoder.calls)
        assert all(task is None and prompt is None for _, task, prompt in model.encoder.calls)
        assert all(not training and inference for _, training, inference in model.encoder.forward_calls)
        assert not model.training
        if convert_to_tensor:
            assert torch.is_inference(actual) and not actual.requires_grad


@pytest.mark.parametrize("api,options,task,prompt", [
    ("encode", {}, None, None),
    ("encode", {"task": "query"}, "query", "query: "),
    ("encode", {"task": "document"}, "document", "passage: "),
    ("encode", {"prompt_name": "custom"}, None, "custom: "),
    ("encode_query", {}, "query", "query: "),
    ("encode_document", {}, "document", "passage: "),
    ("encode_query", {"prompt_name": "custom"}, "query", "custom: "),
    ("encode_document", {"prompt": "explicit: "}, "document", "explicit: "),
    ("encode_query", {"prompt": ""}, "query", ""),
])
def test_every_batch_uses_the_same_role_and_prompt(monkeypatch, api, options, task, prompt):
    model = make_model(monkeypatch)
    model.eval()
    with torch.no_grad():
        expected = model(TEXTS, task=task, prompt=prompt)
    model.encoder.calls.clear()
    actual = getattr(model, api)(TEXTS, batch_size=3, convert_to_tensor=True, **options)
    torch.testing.assert_close(actual, expected)
    assert model.encoder.calls == [
        (TEXTS[:3], task, prompt), (TEXTS[3:6], task, prompt), (TEXTS[6:], task, prompt),
    ]


@pytest.mark.parametrize("api", ["encode", "encode_query", "encode_document"])
def test_default_batch_size_is_bounded_and_single_string_is_not_split(monkeypatch, api):
    model = make_model(monkeypatch)
    encode = getattr(model, api)
    texts = [str(i) for i in range(70)]
    assert encode(texts).shape == (70, 4)
    assert [len(texts) for texts, _, _ in model.encoder.calls] == [32, 32, 6]
    model.encoder.calls.clear()
    for text in ("dog mammal", ""):
        actual = encode(text, batch_size=1, convert_to_tensor=True)
        expected = encode([text], batch_size=1, convert_to_tensor=True)[0]
        assert actual.shape == (4,)
        torch.testing.assert_close(actual, expected)
    assert [texts for texts, _, _ in model.encoder.calls] == [["dog mammal"], ["dog mammal"], [""], [""]]


@pytest.mark.parametrize("geometry,width,fixed_double", GEOMETRIES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_inputs_skip_encoder_and_keep_geometry_contract(monkeypatch, geometry, width, fixed_double, dtype):
    model = make_model(monkeypatch, geometry, dtype)
    for api in ("encode", "encode_query", "encode_document"):
        for tensor_output in (False, True):
            model.train()
            result = getattr(model, api)([], batch_size=2, convert_to_tensor=tensor_output)
            assert result.shape == (0, width)
            if tensor_output:
                assert result.dtype == (torch.float64 if fixed_double else dtype)
                assert result.device.type == "cpu"
                assert torch.is_inference(result) and not result.requires_grad
            else:
                assert isinstance(result, np.ndarray)
                assert result.dtype == (np.float64 if fixed_double or dtype == torch.float64 else np.float32)
            assert not model.training
    assert not model.encoder.calls and not model.encoder.forward_calls


@pytest.mark.parametrize("api", ["encode", "encode_query", "encode_document"])
@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.0, "2", None])
def test_invalid_batch_sizes_fail_before_encoding_or_changing_mode(monkeypatch, api, batch_size):
    model = make_model(monkeypatch)
    for texts in ([], ["dog"], "dog"):
        model.train()
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            getattr(model, api)(texts, batch_size=batch_size)
        assert model.training and not model.encoder.calls


@pytest.mark.parametrize("api,options,error", [
    ("encode", {"task": "invalid"}, ValueError),
    ("encode_query", {"prompt_name": "missing"}, ValueError),
    ("encode_document", {"prompt": "", "prompt_name": "custom"}, ValueError),
    ("encode", {"prompt": 123}, TypeError),
])
def test_empty_input_does_not_hide_invalid_prompt_options(monkeypatch, api, options, error):
    model = make_model(monkeypatch)
    with pytest.raises(error):
        getattr(model, api)([], **options)
    assert not model.encoder.calls


def test_later_batch_failure_keeps_existing_eval_mode_contract(monkeypatch):
    model = make_model(monkeypatch)
    model.encoder.fail_on_call = 2
    model.train()
    with pytest.raises(RuntimeError, match="encoder failed"):
        model.encode(TEXTS, batch_size=2)
    assert not model.training
    assert [len(texts) for texts, _, _ in model.encoder.calls] == [2, 2]


@pytest.mark.parametrize("device", ["cuda", "mps"])
@pytest.mark.parametrize("geometry,width,fixed_double", [GEOMETRIES[i] for i in (0, 3, 5, 6)])
def test_device_policy_is_the_same_for_batched_and_empty_output(monkeypatch, device, geometry, width, fixed_double):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable")
    model = make_model(monkeypatch, geometry).to(device)
    expected_device = torch.device("cpu" if device == "mps" and fixed_double else device)
    if device == "cuda":
        expected_device = torch.device("cuda", torch.cuda.current_device())
    full = model.encode_query(TEXTS, batch_size=20, convert_to_tensor=True)
    batched = model.encode_query(TEXTS, batch_size=2, convert_to_tensor=True)
    empty = model.encode_query([], convert_to_tensor=True)
    assert full.device == batched.device == empty.device == expected_device
    assert full.dtype == batched.dtype == empty.dtype
    assert empty.shape == (0, width)
    torch.testing.assert_close(full, batched, rtol=1e-5, atol=1e-7)
    assert isinstance(model.encode_query(TEXTS, batch_size=2), np.ndarray)
