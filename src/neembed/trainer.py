"""Minimal Euclidean training loop for neembed."""

from collections.abc import Iterable, Sequence

import torch

from neembed.evaluator import ManifoldEmbeddingEvaluator
from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.model import ManifoldSentenceTransformer


class ManifoldTrainer:
    """Fine-tune a manifold sentence model with ordinary AdamW optimization.

    Args:
        model: Model whose Euclidean parameters are optimized.
        loss: Manifold-aware ranking loss used for each training batch.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight decay.
        verbose: Print one mean-loss line after each epoch when ``True``.
    """

    def __init__(
        self,
        model: ManifoldSentenceTransformer,
        loss: ManifoldMultipleNegativesRankingLoss,
        *,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        verbose: bool = True,
    ) -> None:
        self.model = model
        self.loss = loss
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.verbose = verbose

    def fit(
        self,
        train_dataloader: Iterable[tuple[Sequence[str], Sequence[str]]],
        *,
        epochs: int = 1,
        evaluator: ManifoldEmbeddingEvaluator | None = None,
    ) -> list[float] | list[dict[str, object]]:
        """Train on anchor-positive batches with optional epoch validation.

        Args:
            train_dataloader: Iterable yielding ``(anchors, positives)`` batches.
                Each element is a sequence of texts consumed by the configured
                ranking loss. For ``epochs > 1``, the iterable must be
                re-iterable; a one-shot iterator or generator is suitable only
                for a single epoch.
            epochs: Number of full passes over ``train_dataloader``.
            evaluator: Optional manifold embedding evaluator called once after
                each completed epoch.

        Returns:
            Without an evaluator, the existing list of mean training losses is
            returned unchanged. With an evaluator, each epoch returns a mapping
            with ``train_loss`` and a nested ``validation`` metrics dictionary.
        """
        loss_history: list[float] = []
        evaluation_history: list[dict[str, object]] = []

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            steps = 0

            for anchors, positives in train_dataloader:
                self.optimizer.zero_grad()
                batch_loss = self.loss(anchors, positives)
                batch_loss.backward()
                self.optimizer.step()

                total_loss += float(batch_loss.detach())
                steps += 1

            epoch_loss = total_loss / steps
            if evaluator is None:
                loss_history.append(epoch_loss)
            else:
                try:
                    with torch.no_grad():
                        validation_metrics = evaluator()
                finally:
                    self.model.train()
                evaluation_history.append(
                    {
                        "train_loss": epoch_loss,
                        "validation": validation_metrics,
                    }
                )

            if self.verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        if evaluator is None:
            return loss_history
        return evaluation_history
