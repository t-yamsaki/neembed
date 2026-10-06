"""Input roles/prompts reach every differentiable objective independently."""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldDepthLoss, ManifoldDistanceMSELoss, ManifoldHierarchyTripletLoss,
    ManifoldMarginMSELoss, ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeHierarchyLoss, ManifoldPrototypes, ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss, ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss, ManifoldTrainer, ManifoldTripletLoss,
)


class PromptEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        self.prompts = {
            "query": "query: ", "document": "passage: ",
            "custom": "custom: ", "default": "default: ",
        }
        self.default_prompt_name = "default"
        self.calls = []
        self.tasks = []

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts, *, prompt=None, task=None):
        self.calls.append((tuple(texts), prompt, task))
        prefixed = [(prompt or "") + text for text in texts]
        return {
            "features": torch.tensor([
                [len(text) / 100, sum(map(ord, text)) % 997 / 1000, len(text.split()) / 10]
                for text in prefixed
            ]),
            "task": task,
            "prompt_length": len(prompt or ""),
        }

    def forward(self, features, *, task=None):
        assert features["task"] == task
        self.tasks.append(task)
        return {"sentence_embedding": self.linear(features["features"])}

    def encode(self, *args, **kwargs):
        raise AssertionError("losses must use differentiable forward")


TRIPLET_NAMES = ("anchors", "positives", "negatives")
TRIPLETS = (["dog", "cat"], ["mammal", "animal"], ["car", "boat"])
CASES = {
    "mnrl": (TRIPLET_NAMES, TRIPLETS),
    "symmetric": (TRIPLET_NAMES, TRIPLETS),
    "triplet": (TRIPLET_NAMES, TRIPLETS),
    "margin": (TRIPLET_NAMES, (*TRIPLETS, [0.5, 0.8])),
    "distance": (("texts_a", "texts_b"), (*TRIPLETS[:2], [0.5, 0.8])),
    "radial": (("parents", "children"), (TRIPLETS[1], TRIPLETS[0])),
    "depth": (("texts",), (TRIPLETS[0], [1, 2])),
    "hierarchy": (
        ("parents", "children", "unrelated"), (TRIPLETS[1], TRIPLETS[0], TRIPLETS[2]),
    ),
    "prototype": (("sentences",), (TRIPLETS[0], ["leaf", "root"])),
}


def make_model(monkeypatch, manifold):
    monkeypatch.setattr(model_module, "SentenceTransformer", PromptEncoder)
    torch.manual_seed(0)
    kwargs = {"manifold": manifold, "embedding_dim": 2, "learnable_curvature": True}
    if manifold == "product":
        kwargs = {"product_config": [
            {"name": "tree", "manifold": "poincare", "intrinsic_dim": 2},
            {"name": "flat", "manifold": "euclidean", "intrinsic_dim": 2},
        ]}
    return ManifoldSentenceTransformer("fake-model", **kwargs)


def make_loss(model, kind, input_options=None):
    kwargs = {"input_options": input_options}
    component = {"component": "tree"} if model.manifold_name == "product" else {}
    if kind == "mnrl":
        return ManifoldMultipleNegativesRankingLoss(model, 0.5, **kwargs)
    if kind == "symmetric":
        return ManifoldSymmetricMultipleNegativesRankingLoss(model, 0.5, **kwargs)
    if kind == "triplet":
        return ManifoldTripletLoss(model, 10.0, **kwargs)
    if kind == "margin":
        return ManifoldMarginMSELoss(model, **kwargs)
    if kind == "distance":
        return ManifoldDistanceMSELoss(model, **kwargs)
    if kind == "radial":
        return ManifoldRadialOrderLoss(model, 10.0, **component, **kwargs)
    if kind == "depth":
        return ManifoldDepthLoss(model, **component, **kwargs)
    if kind == "hierarchy":
        return ManifoldHierarchyTripletLoss(
            model, 10.0, radial_margin=10.0, **component, **kwargs,
        )
    prototypes = ManifoldPrototypes(model, num_prototypes=3, init_std=0.05)
    return ManifoldPrototypeHierarchyLoss(
        model, prototypes, ["root", "leaf", "other"], [("leaf", "root")], **kwargs,
    )


def options_for(names, mode="role"):
    options = {}
    for index, name in enumerate(names):
        task = "query" if index == 0 and len(names) > 1 else "document"
        options[name] = {"task": task}
        if mode == "named":
            options[name]["prompt_name"] = "custom"
        elif mode == "explicit":
            options[name]["prompt"] = name + ": "
        elif mode == "empty":
            options[name]["prompt"] = ""
    return options


def expected_calls(names, inputs, options, prompts):
    calls = []
    for name, texts in zip(names, inputs):
        settings = options.get(name, {})
        task = settings.get("task")
        prefix = settings.get("prompt")
        if prefix is None and settings.get("prompt_name") is not None:
            prefix = prompts[settings["prompt_name"]]
        if prefix is None and task is not None:
            prefix = prompts[task]
        calls.append((tuple(texts), prefix, task))
    return calls


@pytest.mark.parametrize("kind,mode,manifold", [
    (kind, mode, manifold)
    for kind in CASES
    for mode in ("role", "named", "explicit", "empty")
    for manifold in (("poincare", "lorentz", "product") if mode == "role" else ("poincare",))
    if not (kind == "prototype" and manifold == "product")
])
def test_every_objective_applies_each_inputs_options_once_and_has_gradients(
    monkeypatch, kind, mode, manifold,
):
    model = make_model(monkeypatch, manifold)
    names, inputs = CASES[kind]
    options = options_for(names, mode)
    loss = make_loss(model, kind, options)
    value = loss(*inputs)
    assert model.encoder.calls == expected_calls(names, inputs, options, model.encoder.prompts)
    assert model.encoder.tasks == [options[name]["task"] for name in names]
    assert value.ndim == 0 and torch.isfinite(value)
    value.backward()
    for parameter in (model.encoder.linear.weight, model.projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    for parameter in loss.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    if kind == "prototype":
        assert loss.prototypes.prototypes.grad is not None
        assert torch.count_nonzero(loss.prototypes.prototypes.grad) > 0


def retrieval_formula(model, kind, embeddings, inputs):
    a, b = embeddings[:2]
    if kind in ("mnrl", "symmetric"):
        candidates = torch.cat((b, embeddings[2]), dim=0)
        targets = torch.arange(len(a))
        logits = -model.manifold.dist(a[:, None], candidates[None]) / 0.5
        expected = F.cross_entropy(logits, targets)
        if kind == "symmetric":
            reverse_logits = -model.manifold.dist(b[:, None], a[None]) / 0.5
            reverse = F.cross_entropy(reverse_logits, targets)
            expected = (expected + reverse) / 2
    elif kind == "triplet":
        expected = F.relu(
            model.manifold.dist(a, b) - model.manifold.dist(a, embeddings[2]) + 10,
        ).mean()
    elif kind == "margin":
        predicted = model.manifold.dist(a, embeddings[2]) - model.manifold.dist(a, b)
        expected = F.mse_loss(predicted, predicted.new_tensor(inputs[-1]))
    else:
        predicted = model.manifold.dist(a, b)
        expected = F.mse_loss(predicted, predicted.new_tensor(inputs[-1]))
    return expected


@pytest.mark.parametrize("kind", ["mnrl", "symmetric", "triplet", "margin", "distance"])
def test_retrieval_values_match_manually_prompted_embedding_formula(monkeypatch, kind):
    model = make_model(monkeypatch, "poincare")
    names, inputs = CASES[kind]
    options = options_for(names, "explicit")
    loss = make_loss(model, kind, options)
    actual = loss(*inputs)
    # The reverse term must reuse original embeddings without re-encoding.
    assert len(model.encoder.calls) == len(names)
    embeddings = [model(texts, **options[name]) for name, texts in zip(names, inputs)]
    expected = retrieval_formula(model, kind, embeddings, inputs)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("kind", CASES)
@pytest.mark.parametrize("empty", [None, {}, "none_values"])
def test_unconfigured_inputs_keep_raw_text_with_default_prompt_and_legacy_signature(
    monkeypatch, kind, empty,
):
    model = make_model(monkeypatch, "poincare")
    names, inputs = CASES[kind]
    original_forward = model.forward

    def legacy_forward(texts):
        return original_forward(texts)

    monkeypatch.setattr(model, "forward", legacy_forward)
    if empty == "none_values":
        empty = {name: {"task": None, "prompt_name": None, "prompt": None} for name in names}
    loss = make_loss(model, kind, empty)
    assert torch.isfinite(loss(*inputs))
    assert model.encoder.calls == [(tuple(texts), None, None) for texts in inputs[:len(names)]]


def test_partial_options_do_not_infer_positive_or_negative_roles(monkeypatch):
    model = make_model(monkeypatch, "poincare")
    loss = make_loss(model, "mnrl", {"anchors": {"task": "query"}})
    loss(*CASES["mnrl"][1])
    assert [call[1:] for call in model.encoder.calls] == [
        ("query: ", "query"), (None, None), (None, None),
    ]


@pytest.mark.parametrize("options,prefix", [
    ({"prompt": "manual: "}, "manual: "),
    ({"prompt_name": "custom"}, "custom: "),
])
def test_prompt_can_be_selected_without_a_task(monkeypatch, options, prefix):
    model = make_model(monkeypatch, "poincare")
    loss = make_loss(model, "triplet", {"anchors": options})
    loss(*CASES["triplet"][1])
    assert [call[1:] for call in model.encoder.calls] == [
        (prefix, None), (None, None), (None, None),
    ]


@pytest.mark.parametrize("kind", ["mnrl", "symmetric"])
def test_in_batch_only_ranking_keeps_original_roles_and_skips_negative_options(monkeypatch, kind):
    model = make_model(monkeypatch, "poincare")
    names, inputs = CASES[kind]
    options = options_for(names)
    options["negatives"] = {"prompt_name": "not-needed"}
    loss = make_loss(model, kind, options)
    actual = loss(*inputs[:2])
    assert model.encoder.calls == expected_calls(names[:2], inputs, options, model.encoder.prompts)
    a, b = [model(texts, **options[name]) for name, texts in zip(names[:2], inputs)]
    logits = -model.manifold.dist(a[:, None], b[None]) / 0.5
    targets = torch.arange(len(a))
    expected = F.cross_entropy(logits, targets)
    if kind == "symmetric":
        expected = (expected + F.cross_entropy(logits.T, targets)) / 2
    assert torch.allclose(actual, expected)


def test_options_are_copied_and_shared_document_dict_does_not_mix_inputs(monkeypatch):
    model = make_model(monkeypatch, "poincare")
    document = {"task": "document", "prompt": "passage: "}
    options = {"anchors": {"task": "query"}, "positives": document, "negatives": document}
    loss = make_loss(model, "triplet", options)
    document["prompt"] = "changed: "
    options["anchors"] = {}
    loss(*CASES["triplet"][1])
    assert [call[1:] for call in model.encoder.calls] == [
        ("query: ", "query"), ("passage: ", "document"), ("passage: ", "document"),
    ]


@pytest.mark.parametrize("options,error,match", [
    ([], TypeError, "input_options must be a mapping"),
    ({"anchors": "query"}, TypeError, "must be a mapping"),
    ({"anchor": {"task": "query"}}, ValueError, "unknown text input"),
    ({"target_margin": {"task": "query"}}, ValueError, "unknown text input"),
    ({"anchors": {"role": "query"}}, ValueError, "unknown forward option"),
    ({"anchors": {"convert_to_tensor": True}}, ValueError, "unknown forward option"),
])
def test_invalid_option_structure_is_rejected_at_construction(monkeypatch, options, error, match):
    model = make_model(monkeypatch, "poincare")
    with pytest.raises(error, match=match):
        make_loss(model, "margin", options)
    assert model.encoder.calls == []


@pytest.mark.parametrize("options,error,match", [
    ({"task": "invalid"}, ValueError, "task must be"),
    ({"prompt_name": "unknown"}, ValueError, "unknown encoder prompt"),
    ({"prompt": "", "prompt_name": "custom"}, ValueError, "either prompt or prompt_name"),
    ({"prompt": 123}, TypeError, "prompt must be a string"),
])
def test_option_values_use_model_validation(monkeypatch, options, error, match):
    model = make_model(monkeypatch, "poincare")
    loss = make_loss(model, "triplet", {"anchors": options})
    with pytest.raises(error, match=match):
        loss(*CASES["triplet"][1])
    assert model.encoder.calls == []


@pytest.mark.parametrize("manifold", ["poincare", "lorentz", "product"])
def test_composite_keeps_retrieval_and_hierarchy_options_independent(monkeypatch, manifold):
    model = make_model(monkeypatch, manifold)
    retrieval_names, retrieval_inputs = CASES["triplet"]
    hierarchy_names, hierarchy_inputs = CASES["hierarchy"]
    retrieval_options = options_for(retrieval_names)
    hierarchy_options = {name: {"task": "document", "prompt_name": "custom"} for name in hierarchy_names}
    retrieval = make_loss(model, "triplet", retrieval_options)
    hierarchy = make_loss(model, "hierarchy", hierarchy_options)
    composite = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy, hierarchy_weight=0.5)
    rv, hv = composite.component_losses(retrieval_inputs, hierarchy_inputs)
    model.encoder.calls.clear()
    value = composite(retrieval_inputs, hierarchy_inputs)
    assert torch.allclose(value, rv + 0.5 * hv)
    assert model.encoder.calls == (
        expected_calls(retrieval_names, retrieval_inputs, retrieval_options, model.encoder.prompts)
        + expected_calls(hierarchy_names, hierarchy_inputs, hierarchy_options, model.encoder.prompts)
    )
    value.backward()
    assert model.encoder.linear.weight.grad is not None
    assert torch.isfinite(model.encoder.linear.weight.grad).all()
    assert torch.count_nonzero(model.encoder.linear.weight.grad) > 0


def test_zero_weight_composite_skips_hierarchy_input_prompts(monkeypatch):
    model = make_model(monkeypatch, "poincare")
    retrieval = make_loss(model, "triplet", options_for(CASES["triplet"][0]))
    hierarchy = make_loss(model, "hierarchy", {"parents": {"prompt_name": "unknown"}})
    composite = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy, hierarchy_weight=0)
    assert torch.isfinite(composite(CASES["triplet"][1], None))
    assert len(model.encoder.calls) == 3


@pytest.mark.parametrize("kind", ["mnrl", "symmetric", "triplet", "margin"])
def test_existing_trainer_uses_configured_roles_without_batch_changes(monkeypatch, kind):
    model = make_model(monkeypatch, "poincare")
    names, inputs = CASES[kind]
    options = options_for(names)
    loss = make_loss(model, kind, options)
    before = model.encoder.linear.weight.detach().clone()
    trainer = ManifoldTrainer(model, loss, verbose=False)
    history = trainer.fit([inputs])
    assert len(history) == 1 and math.isfinite(history[0])
    assert model.encoder.calls == expected_calls(names, inputs, options, model.encoder.prompts)
    assert not torch.equal(model.encoder.linear.weight, before)


@pytest.mark.parametrize("kind", ["mnrl", "symmetric", "triplet", "margin"])
def test_real_query_document_router_receives_loss_roles_and_gradients(tmp_path, kind):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Router
    from sentence_transformers.sentence_transformer.modules import BoW, Dense

    torch.manual_seed(0)
    encoder = SentenceTransformer(modules=[Router.for_query_document(
        query_modules=[BoW(["query", "dog", "cat"]), Dense(3, 4)],
        document_modules=[BoW(["passage", "dog", "cat"]), Dense(3, 4)],
    )], prompts={"query": "query: ", "document": "passage: "}, device="cpu")
    path = tmp_path / "router"
    encoder.save_pretrained(str(path))
    model = ManifoldSentenceTransformer(
        str(path), manifold="euclidean", local_files_only=True, device="cpu",
    )
    options = options_for(("anchors", "positives", "negatives"))
    inputs = (["dog", "cat"], ["dog", "cat"], ["cat", "dog"])
    if kind == "margin":
        inputs += ([0.5, 0.8],)
    loss = make_loss(model, kind, options)
    value = loss(*inputs)
    assert torch.isfinite(value)
    # Check that the loss sees the same embeddings as role-aware inference.
    embeddings = [
        model.encode_query(inputs[0], convert_to_tensor=True),
        model.encode_document(inputs[1], convert_to_tensor=True),
        model.encode_document(inputs[2], convert_to_tensor=True),
    ]
    expected = retrieval_formula(model, kind, embeddings, inputs)
    assert torch.allclose(value, expected)
    value.backward()
    for route in model.encoder[0].sub_modules.values():
        grad = route[1].linear.weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0
