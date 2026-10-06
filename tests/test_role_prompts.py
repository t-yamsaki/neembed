"""Shared, differentiable prompt selection and query/document routing."""

import json

import numpy as np
import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import ManifoldSentenceTransformer


class RoleEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        self.prompts = {
            "query": "query: ", "document": "document: ",
            "passage": "passage: ", "corpus": "corpus: ",
            "custom": "custom: ", "default": "default: ",
        }
        self.default_prompt_name = "default"
        self.preprocess_calls = []
        self.forward_tasks = []

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts, *, prompt=None, task=None):
        prompted = [(prompt or "") + text for text in texts]
        self.preprocess_calls.append((prompted, prompt, task))
        return {
            "input_features": torch.tensor([
                [len(text) / 100.0, 0.1, -0.1] for text in prompted
            ]),
            "prompt_length": 0 if not prompt else len(prompt),
            "preprocess_task": task,
        }

    def forward(self, features, *, task=None):
        assert features["preprocess_task"] == task
        self.forward_tasks.append(task)
        return {"sentence_embedding": self.linear(features["input_features"])}

    def encode(self, *args, **kwargs):
        raise AssertionError("neembed must not call encoder.encode")


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(model_module, "SentenceTransformer", RoleEncoder)
    torch.manual_seed(0)
    return ManifoldSentenceTransformer("fake-model", embedding_dim=2)


def test_unspecified_role_keeps_raw_text_even_with_saved_default(model):
    model.eval()
    before = model(["dog"])
    after = model.encode(["dog"], convert_to_tensor=True)
    assert torch.allclose(before, after)
    assert model.encoder.preprocess_calls == [
        (["dog"], None, None), (["dog"], None, None),
    ]


@pytest.mark.parametrize("options,expected", [
    ({"task": "query"}, "query: "),
    ({"task": "document"}, "document: "),
    ({"task": "query", "prompt_name": "custom"}, "custom: "),
    ({"task": "document", "prompt": "explicit: "}, "explicit: "),
    ({"task": "query", "prompt": ""}, ""),
    ({"prompt_name": "custom"}, "custom: "),
    ({"prompt": "explicit: "}, "explicit: "),
])
def test_prompt_is_selected_once_and_task_reaches_both_stages(model, options, expected):
    model(["dog"], **options)
    assert model.encoder.preprocess_calls[-1] == (
        [expected + "dog"], expected, options.get("task"),
    )
    assert model.encoder.forward_tasks[-1] == options.get("task")


@pytest.mark.parametrize("remaining,task,expected", [
    (["passage", "corpus", "default"], "document", "passage: "),
    (["corpus", "default"], "document", "corpus: "),
    (["default"], "document", "default: "),
    (["default"], "query", "default: "),
    ([], "query", None),
    ([], "document", None),
])
def test_role_fallback_and_no_prompt_models(model, remaining, task, expected):
    model.encoder.prompts = {k: v for k, v in model.encoder.prompts.items() if k in remaining}
    if not remaining:
        model.encoder.default_prompt_name = None
    model(["dog"], task=task)
    assert model.encoder.preprocess_calls[-1] == (
        [(expected or "") + "dog"], expected, task,
    )


def test_empty_saved_role_prompt_disables_default(model):
    model.encoder.prompts["query"] = ""
    model(["dog"], task="query")
    assert model.encoder.preprocess_calls[-1] == (["dog"], "", "query")


@pytest.mark.parametrize("options,error,match", [
    ({"task": "passage"}, ValueError, "task"),
    ({"prompt_name": "missing"}, ValueError, "unknown"),
    ({"task": "query", "prompt_name": "missing"}, ValueError, "unknown"),
    ({"prompt": "", "prompt_name": "query"}, ValueError, "not both"),
    ({"prompt": 123}, TypeError, "prompt"),
    ({"prompt_name": 123}, TypeError, "prompt_name"),
])
def test_invalid_prompt_configuration_fails_before_encoder(model, options, error, match):
    with pytest.raises(error, match=match):
        model(["dog"], **options)
    assert not model.encoder.preprocess_calls


@pytest.mark.parametrize("task", ["query", "document"])
@pytest.mark.parametrize("single", [False, True])
def test_inference_helpers_match_differentiable_forward(model, task, single):
    texts = ["dog", "mammal"]
    model.eval()
    expected = model(texts[:1] if single else texts, task=task, prompt_name="custom")
    helper = getattr(model, "encode_" + task)
    actual = helper(texts[0] if single else texts, prompt_name="custom", convert_to_tensor=True)
    assert torch.allclose(expected[0] if single else expected, actual)
    assert not actual.requires_grad and torch.is_inference(actual)
    assert not model.training
    array = helper(texts[0] if single else texts, prompt="")
    assert isinstance(array, np.ndarray)
    assert array.shape == actual.shape


@pytest.mark.parametrize("geometry", [
    {"manifold": "poincare", "embedding_dim": 2, "learnable_curvature": True},
    {"manifold": "lorentz", "embedding_dim": 2, "learnable_curvature": True},
    {"product_config": [
        {"manifold": "lorentz", "intrinsic_dim": 2},
        {"manifold": "euclidean", "intrinsic_dim": 2},
    ]},
])
def test_role_forward_keeps_encoder_projection_and_curvature_gradients(monkeypatch, geometry):
    monkeypatch.setattr(model_module, "SentenceTransformer", RoleEncoder)
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer("fake-model", **geometry)
    embeddings = model(["dog", "cat"], task="query")
    embeddings.square().sum().backward()
    parameters = [model.encoder.linear.weight, model.projection.weight]
    if model.learnable_curvature:
        parameters.extend(p for p in model.manifold.parameters() if p.requires_grad)
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_no_role_still_supports_legacy_encoder_signatures(monkeypatch):
    class LegacyEncoder(RoleEncoder):
        def preprocess(self, texts):
            return super().preprocess(texts)

        def forward(self, features):
            return super().forward(features)

    monkeypatch.setattr(model_module, "SentenceTransformer", LegacyEncoder)
    model = ManifoldSentenceTransformer("fake-model", embedding_dim=2)
    assert torch.isfinite(model.encode(["dog"], convert_to_tensor=True)).all()
    # A role-aware call must not silently discard task for unsupported encoders.
    model.encoder.prompts = {}
    model.encoder.default_prompt_name = None
    with pytest.raises(TypeError, match="task"):
        model(["dog"], task="query")


@pytest.fixture
def local_prompt_encoder(tmp_path):
    """A real local Transformer with prompt-excluding pooling, no downloads."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Transformer
    from sentence_transformers.sentence_transformer.modules import Pooling
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.manual_seed(0)
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
              "query", "passage", "default", ":", "dog", "cat", "mammal"]
    vocab = {word: index for index, word in enumerate(tokens)}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
        cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
        model_max_length=32,
    )
    backbone = tmp_path / "backbone"
    BertModel(BertConfig(
        vocab_size=len(tokens), hidden_size=8, intermediate_size=16,
        num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=32,
        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
    )).save_pretrained(backbone)
    fast_tokenizer.save_pretrained(backbone)
    encoder = SentenceTransformer(
        modules=[
            Transformer(str(backbone), model_args={"local_files_only": True},
                        tokenizer_args={"local_files_only": True}),
            Pooling(8, include_prompt=False),
        ],
        prompts={"query": "query: ", "document": "passage: ",
                 "passage": "passage: ", "fallback": "default: "},
        default_prompt_name="fallback", device="cpu",
    )
    path = tmp_path / "encoder"
    encoder.save_pretrained(str(path))
    return path


@pytest.mark.parametrize("task", ["query", "document"])
def test_batched_real_prompt_pooling_matches_full_forward(local_prompt_encoder, monkeypatch, task):
    model = ManifoldSentenceTransformer(
        str(local_prompt_encoder), manifold="euclidean", device="cpu", local_files_only=True,
    )
    texts = ["dog mammal", "cat", "dog cat mammal", "mammal", "dog"]
    model.eval()
    with torch.no_grad():
        expected = model(texts, task=task)
    original = model.encoder.preprocess
    calls = []

    def record_preprocess(batch, **kwargs):
        features = original(batch, **kwargs)
        calls.append((len(batch), kwargs["task"], features["prompt_length"]))
        return features

    monkeypatch.setattr(model.encoder, "preprocess", record_preprocess)
    actual = getattr(model, "encode_" + task)(texts, batch_size=2, convert_to_tensor=True)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert [size for size, _, _ in calls] == [2, 2, 1]
    assert all(role == task and prompt_length > 0 for _, role, prompt_length in calls)
    assert not model.encoder[1].include_prompt


@pytest.mark.parametrize("task", ["query", "document"])
def test_real_prompt_pooling_matches_native_encoder_and_local_round_trip(
    local_prompt_encoder, tmp_path, monkeypatch, task,
):
    model = ManifoldSentenceTransformer(
        str(local_prompt_encoder), manifold="euclidean", device="cpu", local_files_only=True,
    )
    texts = ["dog mammal", "cat"]
    expected = getattr(model.encoder, "encode_" + task)(texts, convert_to_tensor=True)
    seen_features = []
    original = model.encoder.preprocess

    def record_preprocess(*args, **kwargs):
        features = original(*args, **kwargs)
        seen_features.append(features)
        return features

    monkeypatch.setattr(model.encoder, "preprocess", record_preprocess)
    model.eval()
    differentiable = model(texts, task=task)
    actual = getattr(model, "encode_" + task)(texts, convert_to_tensor=True)
    assert torch.allclose(expected, differentiable, atol=1e-6)
    assert torch.allclose(expected, actual, atol=1e-6)
    assert all(features["prompt_length"] > 0 for features in seen_features)
    assert not model.encoder[1].include_prompt
    # Saved encoder prompts already round-trip via Sentence Transformers.
    saved = tmp_path / "saved-neembed"
    model.save_pretrained(saved)
    restored = ManifoldSentenceTransformer.from_pretrained(
        saved, local_files_only=True, device="cpu",
    )
    assert restored.encoder.prompts == model.encoder.prompts
    assert torch.allclose(
        actual, getattr(restored, "encode_" + task)(texts, convert_to_tensor=True), atol=1e-6,
    )


@pytest.mark.parametrize("prompts,task,expected_prompt", [
    ({"fallback": "default: "}, "query", "default: "),
    ({"fallback": "default: "}, "document", "default: "),
    ({"passage": "passage: ", "fallback": "default: "}, "document", "passage: "),
    ({"corpus": "passage: ", "fallback": "default: "}, "document", "passage: "),
    ({"query": "", "fallback": "default: "}, "query", ""),
    ({"document": "", "passage": "passage: ", "fallback": "default: "}, "document", ""),
])
def test_real_encoder_preserves_configured_role_keys_and_fallback_on_round_trip(
    local_prompt_encoder, tmp_path, prompts, task, expected_prompt,
):
    # Model configs commonly contain only a default or a passage prompt. ST 6.1
    # synthesizes absent query/document keys as "", losing that distinction.
    config_path = local_prompt_encoder / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config["prompts"] = prompts
    config_path.write_text(json.dumps(config))
    model = ManifoldSentenceTransformer(
        str(local_prompt_encoder), manifold="euclidean", device="cpu", local_files_only=True,
    )
    texts = ["dog mammal", "cat"]
    expected = model.encoder.encode(texts, prompt=expected_prompt, convert_to_tensor=True)
    raw = model.encoder.encode(texts, prompt="", convert_to_tensor=True)
    model.eval()
    assert model.encoder.prompts == prompts
    assert torch.allclose(model(texts, task=task), expected, atol=1e-6)
    assert torch.allclose(
        getattr(model, "encode_" + task)(texts, convert_to_tensor=True), expected, atol=1e-6,
    )
    assert torch.allclose(model.encode(texts, convert_to_tensor=True), raw, atol=1e-6)

    saved = tmp_path / "saved-neembed"
    model.save_pretrained(saved)
    saved_config = json.loads((saved / "encoder" / config_path.name).read_text())
    assert saved_config["prompts"] == prompts
    assert saved_config["model_type"] == "SentenceTransformer"
    restored = ManifoldSentenceTransformer.from_pretrained(
        saved, local_files_only=True, device="cpu",
    )
    assert restored.encoder.prompts == prompts
    assert torch.allclose(
        getattr(restored, "encode_" + task)(texts, convert_to_tensor=True), expected, atol=1e-6,
    )


def test_real_router_uses_matching_query_document_routes(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Router
    from sentence_transformers.sentence_transformer.modules import BoW, Dense

    torch.manual_seed(0)
    encoder = SentenceTransformer(modules=[Router.for_query_document(
        query_modules=[BoW(["query", "dog", "cat"]), Dense(3, 4)],
        document_modules=[BoW(["passage", "dog", "cat"]), Dense(3, 4)],
    )], prompts={"query": "query: ", "passage": "passage: "}, device="cpu")
    path = tmp_path / "router-encoder"
    encoder.save_pretrained(str(path))
    model = ManifoldSentenceTransformer(
        str(path), manifold="euclidean", local_files_only=True, device="cpu",
    )
    outputs = []
    for task in ("query", "document"):
        expected = getattr(model.encoder, "encode_" + task)(["dog"], convert_to_tensor=True)
        actual = getattr(model, "encode_" + task)(["dog"], convert_to_tensor=True)
        assert torch.allclose(expected, actual)
        outputs.append(actual)
    assert not torch.allclose(*outputs)
    model.train()
    (model(["dog"], task="query").square().sum()
     + model(["dog"], task="document").square().sum()).backward()
    for route in model.encoder[0].sub_modules.values():
        grad = route[1].linear.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0
