"""Component-targeted hierarchy supervision leaves full-product retrieval intact."""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldDepthLoss,
    ManifoldHierarchyEvaluator,
    ManifoldHierarchyTripletLoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldRadialOrderLoss,
    ManifoldRetrievalHierarchyLoss,
    ManifoldSentenceTransformer,
)


POINTS = {
    "root": [0.05, 0.02, 0.04, 0.01],
    "parent": [0.4, 0.1, 0.2, 0.15],
    "child": [0.1, 0.05, 0.5, 0.2],
    "other": [-0.2, 0.25, 0.1, -0.2],
}


class TinyEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))
        path = Path(model_name_or_path) / "encoder.pt"
        if path.exists():
            self.load_state_dict(torch.load(path, weights_only=True))

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts):
        return {"features": torch.tensor([POINTS[t] for t in texts])}

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["features"])}

    def save_pretrained(self, output_path):
        Path(output_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(output_path) / "encoder.pt")


def _make_model(monkeypatch, config):
    monkeypatch.setattr(model_module, "SentenceTransformer", TinyEncoder)
    model = ManifoldSentenceTransformer("tiny", product_config=config)
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(4))
        model.projection.bias.zero_()
    return model


@pytest.fixture(params=["poincare-euclidean", "poincare-sphere", "euclidean-lorentz", "sphere-lorentz"])
def model_and_index(monkeypatch, request):
    kinds = request.param.split("-")
    config = []
    for kind in kinds:
        if kind in {"poincare", "lorentz"}:
            config.append({"name": "hierarchy", "manifold": kind,
                           "intrinsic_dim": 2, "curvature": 0.5, "scale": 1.75})
        else:
            component = {"name": "content", "manifold": "euclidean",
                         "intrinsic_dim": 2, "scale": 0.75}
            if kind == "sphere":
                component.update(manifold="sphere_projection", sectional_curvature=0.25)
            config.append(component)
    index = 0 if kinds[0] in {"poincare", "lorentz"} else 1
    return _make_model(monkeypatch, config), index


def _selected_points(model, index, texts):
    widths = [c.ambient_dim for c in model.product_config.components]
    return torch.split(model(texts), widths, dim=-1)[index]


@pytest.mark.parametrize("selection", ["name", "index"])
@pytest.mark.parametrize("objective", ["depth", "radial", "directed"])
def test_hierarchy_matches_selected_scaled_geometry_and_only_its_projection_rows(
    model_and_index, selection, objective,
):
    model, index = model_and_index
    component = "hierarchy" if selection == "name" else index
    geometry = model.manifold.manifolds[index]
    parents, children, unrelated = ["parent", "root"], ["child", "parent"], ["other", "child"]
    p = _selected_points(model, index, parents)
    c = _selected_points(model, index, children)
    n = _selected_points(model, index, unrelated)
    if objective == "depth":
        loss = ManifoldDepthLoss(model, radial_scale=0.6, component=component)
        expected = F.mse_loss(geometry.dist0(p), p.new_tensor([0.0, 0.6]))
        actual = loss(parents, [0, 1])
    elif objective == "radial":
        loss = ManifoldRadialOrderLoss(model, margin=2.0, component=component)
        expected = F.relu(geometry.dist0(p) + 2.0 - geometry.dist0(c)).mean()
        actual = loss(parents, children)
    else:
        loss = ManifoldHierarchyTripletLoss(
            model, margin=2.0, radial_margin=2.0, radial_weight=0.4, component=component,
        )
        expected = (F.relu(geometry.dist(p, c) - geometry.dist(p, n) + 2.0)
                    + 0.4 * F.relu(geometry.dist0(p) + 2.0 - geometry.dist0(c))).mean()
        actual = loss(parents, children, unrelated)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual) and actual.requires_grad
    actual.backward()
    selected = slice(2 * index, 2 * index + 2)
    other = slice(2 * (1 - index), 2 * (1 - index) + 2)
    for parameter in (model.projection.weight, model.encoder.linear.weight):
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad[selected].abs().sum() > 0
        assert torch.count_nonzero(parameter.grad[other]) == 0

    model.zero_grad()
    retrieval = ManifoldMultipleNegativesRankingLoss(model, temperature=0.5)
    retrieval(parents, children, unrelated).backward()
    assert model.projection.weight.grad[selected].abs().sum() > 0
    assert model.projection.weight.grad[other].abs().sum() > 0


@pytest.mark.parametrize("training", [True, False])
def test_selected_hierarchy_evaluator_matches_radii_and_restores_mode(model_and_index, training):
    model, index = model_and_index
    texts = ["root", "parent", "child"]
    geometry = model.manifold.manifolds[index]
    radii = geometry.dist0(_selected_points(model, index, texts)).detach()
    differences = radii[:-1] - radii[1:]
    model.train(training)
    for selection in ("hierarchy", index):
        metrics = ManifoldHierarchyEvaluator(
            model=model, node_ids=texts, texts=texts,
            parent_child_edges=[("root", "parent"), ("parent", "child")],
            depths={"root": 0, "parent": 1, "child": 2}, component=selection,
        )()
        assert model.training == training
        assert metrics["parent_child_radial_order_accuracy"] == pytest.approx(
            float((differences < 0).float().mean()))
        assert metrics["mean_radial_order_violation"] == pytest.approx(
            float(differences.clamp_min(0).mean()))
        assert metrics["depth_radius_spearman"] == pytest.approx(0.5 if index == 0 else 1.0)
    assert all(p.grad is None for p in model.parameters())


def test_product_joint_loss_uses_full_retrieval_and_selected_hierarchy(model_and_index):
    model, index = model_and_index
    retrieval = ManifoldMultipleNegativesRankingLoss(model, temperature=0.5)
    hierarchy = ManifoldDepthLoss(model, component="hierarchy")
    combined = ManifoldRetrievalHierarchyLoss(retrieval, hierarchy, hierarchy_weight=0.3)
    retrieval_inputs = (["parent", "root"], ["child", "other"])
    hierarchy_inputs = (["parent", "child"], [0, 1])
    expected = retrieval(*retrieval_inputs) + 0.3 * hierarchy(*hierarchy_inputs)
    actual = combined(retrieval_inputs, hierarchy_inputs)
    torch.testing.assert_close(actual, expected)
    expected_grad = torch.autograd.grad(expected, model.projection.weight)[0]
    actual.backward()
    torch.testing.assert_close(model.projection.weight.grad, expected_grad)
    assert model.projection.weight.grad[2 * index:2 * index + 2].abs().sum() > 0
    assert model.projection.weight.grad[2 * (1 - index):2 * (1 - index) + 2].abs().sum() > 0


def test_component_identity_and_hierarchy_distance_survive_save_load(model_and_index, tmp_path):
    model, index = model_and_index
    texts = ["root", "parent", "child"]
    before_embeddings = model.encode(texts, convert_to_tensor=True)
    before_distance = model.distance(before_embeddings[:, None], before_embeddings[None])
    before_loss = ManifoldDepthLoss(model, component="hierarchy")(texts, [0, 1, 2]).detach()
    model.save_pretrained(tmp_path)
    loaded = ManifoldSentenceTransformer.from_pretrained(tmp_path)
    assert loaded.product_config == model.product_config
    assert loaded.product_config.components[index].name == "hierarchy"
    after_embeddings = loaded.encode(texts, convert_to_tensor=True)
    torch.testing.assert_close(after_embeddings, before_embeddings)
    torch.testing.assert_close(loaded.distance(after_embeddings[:, None], after_embeddings[None]),
                               before_distance)
    for selection in ("hierarchy", index):
        torch.testing.assert_close(ManifoldDepthLoss(loaded, component=selection)(texts, [0, 1, 2]),
                                   before_loss)


def _construct_api(api, model, component):
    if api is ManifoldHierarchyEvaluator:
        return api(model=model, node_ids=["root", "child"], texts=["root", "child"],
                   parent_child_edges=[("root", "child")], component=component)
    return api(model, component=component)


@pytest.mark.parametrize("api", [ManifoldDepthLoss, ManifoldRadialOrderLoss,
                                 ManifoldHierarchyTripletLoss, ManifoldHierarchyEvaluator])
@pytest.mark.parametrize("selection", [None, "missing", -1, 2, True, 0.5, [], "content", 1])
def test_invalid_or_incompatible_selections_fail_before_encoding(monkeypatch, api, selection):
    model = _make_model(monkeypatch, [
        {"name": "hierarchy", "manifold": "poincare", "intrinsic_dim": 2},
        {"name": "content", "manifold": "euclidean", "intrinsic_dim": 2},
    ])

    def unexpected_encoding(*args, **kwargs):
        pytest.fail("invalid component must be rejected before encoding")

    monkeypatch.setattr(model, "forward", unexpected_encoding)
    monkeypatch.setattr(model, "encode", unexpected_encoding)
    with pytest.raises(ValueError, match="component"):
        _construct_api(api, model, selection)


@pytest.mark.parametrize("api", [ManifoldDepthLoss, ManifoldRadialOrderLoss,
                                 ManifoldHierarchyTripletLoss, ManifoldHierarchyEvaluator])
def test_single_manifold_rejects_component_selector(monkeypatch, api):
    monkeypatch.setattr(model_module, "SentenceTransformer", TinyEncoder)
    model = ManifoldSentenceTransformer("tiny", manifold="poincare", embedding_dim=2)
    with pytest.raises(ValueError, match="component selection requires a product model"):
        _construct_api(api, model, 0)


@pytest.mark.parametrize("curvature", [-0.5, 0.0, 0.25])
def test_generic_stereographic_radial_component_is_not_implicitly_enabled(monkeypatch, curvature):
    model = _make_model(monkeypatch, [
        {"name": "generic", "manifold": "stereographic", "intrinsic_dim": 2,
         "sectional_curvature": curvature},
        {"name": "content", "manifold": "euclidean", "intrinsic_dim": 2},
    ])
    with pytest.raises(ValueError, match="must use poincare or lorentz"):
        ManifoldDepthLoss(model, component="generic")


def test_multiple_hyperbolic_components_require_explicit_identity(monkeypatch):
    model = _make_model(monkeypatch, [
        {"name": "0", "manifold": "poincare", "intrinsic_dim": 2},
        {"name": "1", "manifold": "lorentz", "intrinsic_dim": 2},
    ])
    with pytest.raises(ValueError, match="requires an explicit component"):
        ManifoldDepthLoss(model)
    # Numeric strings are names; integers are zero-based positions.
    by_name = ManifoldDepthLoss(model, component="1")(["parent", "child"], [0, 1])
    by_index = ManifoldDepthLoss(model, component=1)(["parent", "child"], [0, 1])
    torch.testing.assert_close(by_name, by_index)
