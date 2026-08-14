"""neembed package."""

from neembed.evaluator import (
    ManifoldEmbeddingEvaluator,
    ManifoldPrototypeAssignmentEvaluator,
)
from neembed.losses import (
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeHierarchyLoss,
)
from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes
from neembed.trainer import ManifoldTrainer

__all__ = [
    "ManifoldEmbeddingEvaluator",
    "ManifoldMultipleNegativesRankingLoss",
    "ManifoldPrototypeAssignmentEvaluator",
    "ManifoldPrototypeHierarchyLoss",
    "ManifoldPrototypes",
    "ManifoldSentenceTransformer",
    "ManifoldTrainer",
]
