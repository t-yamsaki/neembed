"""neembed package."""

from neembed.evaluator import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldEmbeddingEvaluator,
    ManifoldPrototypeAssignmentEvaluator,
)
from neembed.losses import (
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeHierarchyLoss,
)
from neembed.mining import mine_hard_negatives
from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes
from neembed.retrieval import exact_corpus_search
from neembed.trainer import ManifoldTrainer

__all__ = [
    "ManifoldCorpusRetrievalEvaluator",
    "ManifoldEmbeddingEvaluator",
    "ManifoldMultipleNegativesRankingLoss",
    "ManifoldPrototypeAssignmentEvaluator",
    "ManifoldPrototypeHierarchyLoss",
    "ManifoldPrototypes",
    "ManifoldSentenceTransformer",
    "ManifoldTrainer",
    "exact_corpus_search",
    "mine_hard_negatives",
]
