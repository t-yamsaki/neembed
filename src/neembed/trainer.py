"""Minimal Euclidean training loop for neembed."""

from collections.abc import Iterable, Sequence

import torch

from neembed.losses import ManifoldMultipleNegativesRankingLoss
from neembed.model import ManifoldSentenceTransformer


class ManifoldTrainer:
    """Fine-tune a manifold sentence model with ordinary Euclidean optimization."""

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
    ) -> list[float]:
        """Train for the requested epochs and return mean loss per epoch."""
        history: list[float] = []
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
            history.append(epoch_loss)
            if self.verbose:
                print(f"Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f}")

        return history
