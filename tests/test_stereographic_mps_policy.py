"""Regression coverage for stereographic float64 device selection on Apple MPS."""

import geoopt
import pytest
import torch
from torch import nn

import neembed.model as model_module
import neembed.prototypes as prototypes_module
from neembed import ManifoldPrototypeHierarchyLoss, ManifoldPrototypes
from neembed.manifolds import get_manifold
from neembed.model import ManifoldSentenceTransformer, _select_geometry_device


def test_stereographic_float64_geometry_falls_back_to_cpu_on_mps() -> None:
    assert _select_geometry_device("sphere_projection", "mps") == torch.device("cpu")
    assert _select_geometry_device("stereographic", "mps") == torch.device("cpu")


def test_non_stereographic_device_behavior_is_unchanged_on_mps() -> None:
    assert _select_geometry_device("poincare", "mps") == torch.device("mps")
    assert _select_geometry_device("euclidean", "mps") == torch.device("mps")
    assert _select_geometry_device("lorentz", "mps") == torch.device("mps")


def test_stereographic_geometry_keeps_supported_devices() -> None:
    assert _select_geometry_device("sphere_projection", "cpu") == torch.device("cpu")
    assert _select_geometry_device("stereographic", "cuda") == torch.device("cuda")


class _FakeMpsEncoder(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return torch.device("mps")

    def get_embedding_dimension(self) -> int:
        return 4


class _FakeCpuEncoder(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4


class _RecordingManifold(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("k", torch.tensor(0.5, dtype=torch.float64))
        self.requested_device: torch.device | None = None
        self.requested_dtype: torch.dtype | None = None

    def to(self, device=None, *args, **kwargs):
        if device is not None:
            self.requested_device = torch.device(device)
        self.requested_dtype = kwargs.get("dtype")
        return self

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        self.requested_device = self.k.device
        self.requested_dtype = self.k.dtype
        return result


class _SimulatedTransferModule(nn.Module):
    """Pretend an ordinary child module followed ``.to('mps')`` without MPS CI."""

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def _apply(self, fn, recurse: bool = True):
        self._device = torch.device("mps")
        return self


class _SimulatedProjection(_SimulatedTransferModule):
    """Keep CPU storage for CI while exposing simulated MPS placement."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.02, -0.01], dtype=torch.float32))


class _SimulatedPrototypeModel(ManifoldSentenceTransformer):
    """Minimal sentence model that exercises real manifold/prototype autograd."""

    def __init__(self, manifold_name: str) -> None:
        nn.Module.__init__(self)
        self.encoder = _SimulatedTransferModule()
        self.projection = _SimulatedProjection()
        self.manifold_name = manifold_name
        self.embedding_dim = 2
        self.learnable_curvature = False
        self.manifold = get_manifold(
            manifold_name,
            sectional_curvature=0.5,
        )

    def forward(self, sentences) -> torch.Tensor:
        lengths = torch.tensor(
            [float(len(sentence)) for sentence in sentences],
            dtype=torch.float32,
        )
        tangent = lengths[:, None] * self.projection.weight[None, :]
        tangent = tangent.to(device="cpu", dtype=torch.float64)
        return self.manifold.expmap0(tangent)


def _make_transfer_model(
    manifold_name: str,
) -> tuple[ManifoldSentenceTransformer, _RecordingManifold]:
    model = ManifoldSentenceTransformer.__new__(ManifoldSentenceTransformer)
    nn.Module.__init__(model)
    model.encoder = _SimulatedTransferModule()
    model.projection = _SimulatedTransferModule()
    model.manifold_name = manifold_name
    manifold = _RecordingManifold()
    model.manifold = manifold
    return model, manifold


def test_constructor_does_not_move_float64_stereographic_state_to_mps(
    monkeypatch,
) -> None:
    manifold = _RecordingManifold()
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeMpsEncoder)
    monkeypatch.setattr(model_module, "get_manifold", lambda *args, **kwargs: manifold)

    original_linear_to = nn.Linear.to

    def keep_fake_projection_on_cpu(self, device=None, *args, **kwargs):
        if device is not None and torch.device(device).type == "mps":
            return self
        return original_linear_to(self, device, *args, **kwargs)

    monkeypatch.setattr(nn.Linear, "to", keep_fake_projection_on_cpu)

    ManifoldSentenceTransformer(
        "fake-model",
        manifold="sphere_projection",
        embedding_dim=2,
        sectional_curvature=0.5,
    )

    assert manifold.requested_device == torch.device("cpu")


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_model_to_mps_preserves_cpu_float64_geometry_fallback(
    manifold_name: str,
) -> None:
    model, manifold = _make_transfer_model(manifold_name)

    model.to("mps")

    assert model.encoder.device == torch.device("mps")
    assert manifold.requested_device == torch.device("cpu")
    assert manifold.requested_dtype == torch.float64
    assert model.manifold is manifold


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_parent_module_to_mps_preserves_cpu_float64_geometry_fallback(
    manifold_name: str,
) -> None:
    model, manifold = _make_transfer_model(manifold_name)
    parent = nn.Module()
    parent.add_module("model", model)

    parent.to("mps")

    assert model.encoder.device == torch.device("mps")
    assert manifold.requested_device == torch.device("cpu")
    assert manifold.requested_dtype == torch.float64
    assert model.manifold is manifold


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_prototype_hierarchy_loss_to_mps_keeps_prototypes_on_cpu_and_backpropagates(
    monkeypatch,
    manifold_name: str,
) -> None:
    model = _SimulatedPrototypeModel(manifold_name)
    prototypes = ManifoldPrototypes(model, 3, init_std=0.01)
    loss_module = ManifoldPrototypeHierarchyLoss(
        model,
        prototypes,
        prototype_ids=("root", "child", "other"),
        parent_relations=(("child", "root"),),
    )
    monkeypatch.setattr(
        prototypes_module,
        "_infer_apply_device",
        lambda fn, source_device: torch.device("mps"),
    )

    loss_module.to("mps")

    assert model.encoder.device == torch.device("mps")
    assert model.manifold.k.device == torch.device("cpu")
    assert model.manifold.k.dtype == torch.float64
    assert prototypes.prototypes.device == torch.device("cpu")
    assert prototypes.prototypes.dtype == torch.float64

    loss = loss_module(
        ("a", "aaaa", "aaaaaaaa"),
        ("root", "child", "other"),
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert torch.count_nonzero(model.projection.weight.grad) > 0
    assert prototypes.prototypes.grad is not None
    assert torch.isfinite(prototypes.prototypes.grad).all()


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_model_to_empty_materializes_protected_manifold(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeCpuEncoder)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=0.5,
    )

    model.to("meta")
    assert model.manifold.k.device.type == "meta"

    model.to_empty(device="cpu")

    assert model.encoder.device == torch.device("cpu")
    assert model.manifold.k.device == torch.device("cpu")
    assert model.manifold.k.dtype == torch.float64


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_parent_transfer_is_independent_of_prototype_child_order(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeCpuEncoder)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=0.5,
    )
    prototypes = ManifoldPrototypes(model, 3, init_std=0.01)
    parent = nn.Module()
    parent.add_module("prototypes", prototypes)
    parent.add_module("model", model)

    parent.to("meta")

    assert prototypes.prototypes.device.type == "meta"
    assert prototypes.prototypes.dtype == torch.float64
    assert isinstance(prototypes.prototypes, geoopt.ManifoldParameter)
    assert model.manifold.k.device.type == "meta"
    assert prototypes.manifold is model.manifold

    parent.to_empty(device="cpu")

    assert prototypes.prototypes.device == torch.device("cpu")
    assert prototypes.prototypes.dtype == torch.float64
    assert isinstance(prototypes.prototypes, geoopt.ManifoldParameter)
    assert model.manifold.k.device == torch.device("cpu")
    assert model.manifold.k.dtype == torch.float64
    assert prototypes.manifold is model.manifold


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_model_dtype_casts_never_quantize_fixed_double_curvature(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeCpuEncoder)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=0.1234567890123,
    )
    original_curvature = model.manifold.k.detach().clone()

    model.double()
    assert model.encoder.linear.weight.dtype == torch.float64
    assert model.projection.weight.dtype == torch.float64
    assert torch.equal(model.manifold.k, original_curvature)

    model.float()
    assert model.encoder.linear.weight.dtype == torch.float32
    assert model.projection.weight.dtype == torch.float32
    assert model.manifold.k.dtype == torch.float64
    assert torch.equal(model.manifold.k, original_curvature)

    model.half()
    assert model.encoder.linear.weight.dtype == torch.float16
    assert model.projection.weight.dtype == torch.float16
    assert model.manifold.k.dtype == torch.float64
    assert torch.equal(model.manifold.k, original_curvature)

    model.to(dtype=torch.float32)
    assert model.encoder.linear.weight.dtype == torch.float32
    assert model.projection.weight.dtype == torch.float32
    assert torch.equal(model.manifold.k, original_curvature)


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_prototype_dtype_casts_preserve_values_gradients_metadata_and_identity(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeCpuEncoder)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=0.5,
    )
    prototypes = ManifoldPrototypes(model, 3, init_std=0.01)
    parameter = prototypes.prototypes
    manifold = parameter.manifold
    values = torch.tensor(
        [
            [0.1234567890123, -0.2345678901234],
            [0.3456789012345, -0.4567890123456],
            [0.5678901234567, -0.6789012345678],
        ],
        dtype=torch.float64,
    )
    gradient = torch.tensor(
        [
            [0.0123456789012, -0.0234567890123],
            [0.0345678901234, -0.0456789012345],
            [0.0567890123456, -0.0678901234567],
        ],
        dtype=torch.float64,
    )
    with torch.no_grad():
        parameter.copy_(values)
    parameter.grad = gradient.clone()

    for convert in (
        lambda module: module.float(),
        lambda module: module.half(),
        lambda module: module.to(device="cpu", dtype=torch.float32),
    ):
        convert(prototypes)
        assert prototypes.prototypes is parameter
        assert isinstance(parameter, geoopt.ManifoldParameter)
        assert parameter.manifold is manifold
        assert parameter.dtype == torch.float64
        assert torch.equal(parameter, values)
        assert parameter.grad is not None
        assert parameter.grad.dtype == torch.float64
        assert torch.equal(parameter.grad, gradient)


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_to_empty_mps_empties_geometry_on_cpu_fallback(
    manifold_name: str,
) -> None:
    model = _SimulatedPrototypeModel(manifold_name)
    prototypes = ManifoldPrototypes(model, 3, init_std=0.01)
    old_k = model.manifold.k.detach()
    old_prototypes = prototypes.prototypes.detach()

    model.to_empty(device="mps")
    prototypes.to_empty(device="mps")

    assert model.encoder.device == torch.device("mps")
    assert model.manifold.k.device == torch.device("cpu")
    assert model.manifold.k.dtype == torch.float64
    assert model.manifold.k.data_ptr() != old_k.data_ptr()
    assert prototypes.prototypes.device == torch.device("cpu")
    assert prototypes.prototypes.dtype == torch.float64
    assert prototypes.prototypes.data_ptr() != old_prototypes.data_ptr()


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_meta_to_empty_mps_preserves_prototype_gradient_and_optimizer_reference(
    monkeypatch,
    manifold_name: str,
) -> None:
    monkeypatch.setattr(model_module, "SentenceTransformer", _FakeCpuEncoder)
    model = ManifoldSentenceTransformer(
        "fake-model",
        manifold=manifold_name,
        embedding_dim=2,
        sectional_curvature=0.5,
    )
    prototypes = ManifoldPrototypes(model, 3, init_std=0.01)
    parameter = prototypes.prototypes
    parameter.grad = torch.full_like(parameter, 0.125)
    optimizer = geoopt.optim.RiemannianAdam([parameter], lr=1e-3)

    prototypes.to("meta")
    assert prototypes.prototypes is parameter
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert parameter.device.type == "meta"
    assert parameter.grad is not None
    assert parameter.grad.device.type == "meta"

    prototypes.to_empty(device="mps")

    assert prototypes.prototypes is parameter
    assert optimizer.param_groups[0]["params"][0] is parameter
    assert isinstance(parameter, geoopt.ManifoldParameter)
    assert parameter.manifold is model.manifold
    assert parameter.device == torch.device("cpu")
    assert parameter.dtype == torch.float64
    assert parameter.grad is not None
    assert parameter.grad.device == torch.device("cpu")
    assert parameter.grad.dtype == torch.float64
