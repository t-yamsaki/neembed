"""Regression coverage for non-recursive module transforms."""

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed.model import ManifoldSentenceTransformer


class _FakeCpuEncoder(nn.Module):
    def __init__(self, model_name_or_path: str) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)

    @property
    def device(self) -> torch.device:
        return self.linear.weight.device

    def get_embedding_dimension(self) -> int:
        return 4


@pytest.mark.parametrize("manifold_name", ["sphere_projection", "stereographic"])
def test_to_empty_recurse_false_leaves_protected_children_untouched(
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

    curvature = model.manifold.k.detach().clone()
    curvature_ptr = model.manifold.k.data_ptr()
    encoder_ptr = model.encoder.linear.weight.data_ptr()
    projection_ptr = model.projection.weight.data_ptr()

    model.to_empty(device="meta", recurse=False)

    assert model.manifold.k.device == torch.device("cpu")
    assert model.manifold.k.dtype == torch.float64
    assert model.manifold.k.data_ptr() == curvature_ptr
    assert torch.equal(model.manifold.k, curvature)
    assert model.encoder.linear.weight.device == torch.device("cpu")
    assert model.encoder.linear.weight.data_ptr() == encoder_ptr
    assert model.projection.weight.device == torch.device("cpu")
    assert model.projection.weight.data_ptr() == projection_ptr
