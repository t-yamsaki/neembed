"""Small composite objective for joint retrieval and hierarchy supervision."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
from torch import nn


class ManifoldRetrievalHierarchyLoss(nn.Module):
    """Combine one retrieval loss with one hierarchy loss.

    The objective is

    ``retrieval_loss + hierarchy_weight * hierarchy_loss``.

    Each component keeps its native positional input contract. Callers pass one
    argument bundle for the retrieval loss and one argument bundle for the
    hierarchy loss; the composite simply unpacks each bundle into its component.
    For example, a two-input retrieval loss and a three-input hierarchy loss can
    be combined with ``(anchors, positives)`` and
    ``(parents, children, unrelated)`` respectively.

    ``hierarchy_weight=0`` is a true retrieval-only path: the hierarchy component
    is not evaluated, so the value and gradients match direct retrieval-loss use
    even when hierarchy inputs are unavailable or expensive to compute.

    Args:
        retrieval_loss: Existing retrieval objective returning a scalar tensor.
        hierarchy_loss: Existing hierarchy objective returning a scalar tensor.
        hierarchy_weight: Non-negative, finite multiplier for the hierarchy term.

    Notes:
        The component modules are registered as ordinary PyTorch submodules, so
        ``parameters()`` and caller-owned optimizers behave normally. Existing
        losses remain directly usable without this wrapper.
    """

    def __init__(
        self,
        retrieval_loss: nn.Module,
        hierarchy_loss: nn.Module,
        *,
        hierarchy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(retrieval_loss, nn.Module):
            raise TypeError("retrieval_loss must be a torch.nn.Module")
        if not isinstance(hierarchy_loss, nn.Module):
            raise TypeError("hierarchy_loss must be a torch.nn.Module")
        if (
            isinstance(hierarchy_weight, bool)
            or hierarchy_weight < 0
            or not math.isfinite(hierarchy_weight)
        ):
            raise ValueError("hierarchy_weight must be non-negative and finite")

        self.retrieval_loss = retrieval_loss
        self.hierarchy_loss = hierarchy_loss
        self.hierarchy_weight = float(hierarchy_weight)

    @staticmethod
    def _normalize_inputs(name: str, inputs: Sequence[Any]) -> tuple[Any, ...]:
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ValueError(f"{name} must be a positional input sequence")
        if len(inputs) == 0:
            raise ValueError(f"{name} must contain at least one positional input")
        return tuple(inputs)

    def component_losses(
        self,
        retrieval_inputs: Sequence[Any],
        hierarchy_inputs: Sequence[Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the two unweighted component losses for diagnostics.

        Unlike :meth:`forward`, this method always evaluates both components,
        including when ``hierarchy_weight`` is zero.
        """
        retrieval_args = self._normalize_inputs("retrieval_inputs", retrieval_inputs)
        hierarchy_args = self._normalize_inputs("hierarchy_inputs", hierarchy_inputs)
        retrieval_value = self.retrieval_loss(*retrieval_args)
        hierarchy_value = self.hierarchy_loss(*hierarchy_args)
        return retrieval_value, hierarchy_value

    def forward(
        self,
        retrieval_inputs: Sequence[Any],
        hierarchy_inputs: Sequence[Any] | None = None,
    ) -> torch.Tensor:
        """Return ``retrieval + hierarchy_weight * hierarchy``.

        Args:
            retrieval_inputs: Positional arguments for ``retrieval_loss``.
            hierarchy_inputs: Positional arguments for ``hierarchy_loss``. This
                may be ``None`` only when ``hierarchy_weight`` is zero.

        Returns:
            Scalar composite loss returned by the component objectives.
        """
        retrieval_args = self._normalize_inputs("retrieval_inputs", retrieval_inputs)
        retrieval_value = self.retrieval_loss(*retrieval_args)

        if self.hierarchy_weight == 0.0:
            return retrieval_value
        if hierarchy_inputs is None:
            raise ValueError(
                "hierarchy_inputs are required when hierarchy_weight is positive"
            )

        hierarchy_args = self._normalize_inputs("hierarchy_inputs", hierarchy_inputs)
        hierarchy_value = self.hierarchy_loss(*hierarchy_args)
        return retrieval_value + self.hierarchy_weight * hierarchy_value
