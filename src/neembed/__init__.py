"""neembed package."""

from neembed.evaluator import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldEmbeddingEvaluator,
    ManifoldPrototypeAssignmentEvaluator,
)
from neembed.losses import (
    ManifoldDistanceMSELoss,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldPrototypeHierarchyLoss,
    ManifoldTripletLoss,
)
from neembed.mining import mine_hard_negatives
from neembed.model import ManifoldSentenceTransformer
from neembed.prototypes import ManifoldPrototypes
from neembed.retrieval import exact_corpus_search
from neembed.symmetric_loss import ManifoldSymmetricMultipleNegativesRankingLoss
from neembed.trainer import ManifoldTrainer

__all__ = [
    "ManifoldCorpusRetrievalEvaluator",
    "ManifoldDistanceMSELoss",
    "ManifoldEmbeddingEvaluator",
    "ManifoldMarginMSELoss",
    "ManifoldMultipleNegativesRankingLoss",
    "ManifoldPrototypeAssignmentEvaluator",
    "ManifoldPrototypeHierarchyLoss",
    "ManifoldPrototypes",
    "ManifoldSentenceTransformer",
    "ManifoldSymmetricMultipleNegativesRankingLoss",
    "ManifoldTrainer",
    "ManifoldTripletLoss",
    "exact_corpus_search",
    "mine_hard_negatives",
]
