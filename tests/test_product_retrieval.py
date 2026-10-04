"""Product embeddings use the existing objectives and exact retrieval workflow."""

import copy
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import neembed.evaluator as evaluator_module
import neembed.model as model_module
from neembed import (
    ManifoldCorpusRetrievalEvaluator,
    ManifoldDistanceMSELoss,
    ManifoldGradedCorpusRetrievalEvaluator,
    ManifoldMarginMSELoss,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldSentenceTransformer,
    ManifoldSymmetricMultipleNegativesRankingLoss,
    ManifoldTrainer,
    ManifoldTripletLoss,
    exact_corpus_search,
    mine_hard_negatives,
)


POINTS = {
    "q0": [0.0, 0.0, 0.0, 0.0],
    "q1": [0.2, -0.1, 0.3, 0.05],
    "tie": [0.1, 0.1, -0.1, 0.2],
    "far": [-0.3, 0.4, -0.2, 0.3],
}
QUERIES = ["q0", "q1"]
CORPUS = ["tie", "tie", "q0", "q1", "far"]
QUERY_IDS = ["query-zero", "query-one"]
CORPUS_IDS = ["duplicate-a", "duplicate-b", "query-zero", "positive-one", "other"]
RELEVANCE = {
    "query-zero": ["duplicate-a", "other"],
    "query-one": ["positive-one"],
}


class TinyEncoder(nn.Module):
    """Distinct controlled features plus trainable parameters, without downloads."""

    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(4))

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts):
        return {"features": torch.tensor([POINTS[t] for t in texts])}

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["features"])}


@pytest.fixture(params=["poincare-euclidean", "poincare-sphere", "lorentz-sphere"])
def product_model(monkeypatch, request):
    monkeypatch.setattr(model_module, "SentenceTransformer", TinyEncoder)
    negative, other = request.param.split("-")
    second = {"manifold": "euclidean", "intrinsic_dim": 2, "scale": 1.6}
    if other == "sphere":
        second.update(manifold="sphere_projection", sectional_curvature=0.25)
    model = ManifoldSentenceTransformer("tiny", product_config=[
        {"manifold": negative, "intrinsic_dim": 2, "curvature": 0.5, "scale": 0.7},
        second,
    ])
    with torch.no_grad():
        model.projection.weight.copy_(torch.eye(4))
        model.projection.bias.zero_()
    return model


@pytest.mark.parametrize("objective", ["mnrl", "mnrl-hard", "symmetric", "triplet",
                                       "margin", "distance"])
def test_product_objectives_match_direct_metric_and_backprop(product_model, objective):
    model = product_model
    anchors, positives, negatives = QUERIES, ["tie", "far"], ["q1", "tie"]
    a, p, n = model(anchors), model(positives), model(negatives)
    d = model.manifold.dist
    targets = torch.arange(2)
    temperature = 0.5
    if objective in {"mnrl", "mnrl-hard", "symmetric"}:
        loss_type = (ManifoldSymmetricMultipleNegativesRankingLoss
                     if objective == "symmetric" else ManifoldMultipleNegativesRankingLoss)
        loss = loss_type(model, temperature=temperature)
        candidates = torch.cat((p, n)) if objective != "mnrl" else p
        expected = F.cross_entropy(-d(a[:, None], candidates[None]) / temperature, targets)
        if objective == "symmetric":
            expected = 0.5 * (expected + F.cross_entropy(
                -d(p[:, None], a[None]) / temperature, targets))
        actual = loss(anchors, positives) if objective == "mnrl" else loss(
            anchors, positives, negatives)
    elif objective == "triplet":
        expected = F.relu(d(a, p) - d(a, n) + 2.0).mean()
        actual = ManifoldTripletLoss(model, margin=2.0)(anchors, positives, negatives)
    elif objective == "margin":
        target = a.new_tensor([0.2, -0.1])
        expected = F.mse_loss(d(a, n) - d(a, p), target)
        actual = ManifoldMarginMSELoss(model)(anchors, positives, negatives, target)
    else:
        target = a.new_tensor([0.1, 0.2])
        expected = F.mse_loss(d(a, p), target)
        actual = ManifoldDistanceMSELoss(model)(anchors, positives, target)
    assert actual.ndim == 0 and torch.isfinite(actual)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    for parameter in (model.encoder.linear.weight, model.projection.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    # Both component projection slices participate in the objective.
    assert (model.projection.weight.grad.abs().sum(dim=1) > 0).all()


def _direct_ranking(model):
    q = model.encode(QUERIES, convert_to_tensor=True)
    c = model.encode(CORPUS, convert_to_tensor=True)
    distances = model.manifold.dist(q[:, None], c[None]).detach().cpu().tolist()
    rankings = [sorted(range(len(CORPUS)), key=lambda i: (row[i], i)) for row in distances]
    return distances, rankings


def _record_blocks(monkeypatch, model, query_chunk, corpus_chunk):
    encode, distance = model.encode, model.distance
    batches, blocks = [], []

    def recorded_encode(texts, **kwargs):
        batches.append(len(texts))
        return encode(texts, **kwargs)

    def recorded_distance(a, b):
        assert a.shape[0] <= query_chunk and b.shape[1] <= corpus_chunk
        blocks.append((a.shape[0], b.shape[1]))
        result = distance(a, b)
        assert not result.requires_grad
        return result

    monkeypatch.setattr(model, "encode", recorded_encode)
    monkeypatch.setattr(model, "distance", recorded_distance)
    return batches, blocks


@pytest.mark.parametrize("chunks", [(1, 1), (2, 2), (3, 4)])
@pytest.mark.parametrize("top_k", [None, 1, 3])
def test_product_search_and_rank_match_full_distance(product_model, chunks, top_k, monkeypatch):
    model = product_model
    distances, rankings = _direct_ranking(model)
    for query_index, query in enumerate(QUERIES):
        actual = model.rank(query, CORPUS, top_k=top_k)
        expected = rankings[query_index][:top_k]
        assert [r["index"] for r in actual] == expected
        assert [r["distance"] for r in actual] == pytest.approx(
            [distances[query_index][i] for i in expected], rel=1e-6, abs=1e-7)
    batches, blocks = _record_blocks(monkeypatch, model, *chunks)
    actual = exact_corpus_search(model, QUERIES, CORPUS, top_k=top_k,
                                query_chunk_size=chunks[0], corpus_chunk_size=chunks[1])
    assert batches == ([min(chunks[0], len(QUERIES) - i)
                        for i in range(0, len(QUERIES), chunks[0])]
                       + [min(chunks[1], len(CORPUS) - i)
                          for i in range(0, len(CORPUS), chunks[1])])
    assert len(blocks) == math.ceil(2 / chunks[0]) * math.ceil(5 / chunks[1])
    for query_index, rows in enumerate(actual):
        expected = rankings[query_index][:top_k]
        assert [r["index"] for r in rows] == expected
        assert [r["candidate"] for r in rows] == [CORPUS[i] for i in expected]
        assert [r["distance"] for r in rows] == pytest.approx(
            [distances[query_index][i] for i in expected], rel=1e-6, abs=1e-7)
    # Identical text under different corpus IDs retains input order across blocks.
    assert all(r.index(0) < r.index(1) for r in rankings)


@pytest.mark.parametrize("training", [True, False])
def test_product_binary_evaluator_streams_exact_ranks(product_model, monkeypatch, training):
    model = product_model
    _, rankings = _direct_ranking(model)
    expected_ranks = [
        [ranking.index(CORPUS_IDS.index(cid)) + 1 for cid in RELEVANCE[qid]]
        for qid, ranking in zip(QUERY_IDS, rankings)
    ]

    def no_full_ranking(*args, **kwargs):
        raise AssertionError("binary evaluation must stream rank counts")

    monkeypatch.setattr(evaluator_module, "exact_corpus_search", no_full_ranking, raising=False)
    batches, blocks = _record_blocks(monkeypatch, model, 1, 2)
    relevance_before = copy.deepcopy(RELEVANCE)
    model.train(training)
    metrics = ManifoldCorpusRetrievalEvaluator(
        model=model, queries=QUERIES, query_ids=QUERY_IDS,
        corpus=CORPUS, corpus_ids=CORPUS_IDS, relevance=RELEVANCE,
        recall_at_k=(1, 3, 10), query_chunk_size=1, corpus_chunk_size=2,
    )()
    assert model.training == training
    assert RELEVANCE == relevance_before
    assert len(blocks) == 12  # Two streaming passes over 2 x 3 blocks.
    assert max(batches) <= 2
    assert metrics["mrr"] == pytest.approx(sum(1 / min(r) for r in expected_ranks) / 2)
    for k in (1, 3, 10):
        expected = sum(sum(rank <= k for rank in ranks) / len(ranks)
                       for ranks in expected_ranks) / 2
        assert metrics[f"recall_at_{k}"] == pytest.approx(expected)


def test_product_graded_evaluator_matches_direct_ndcg(product_model):
    model = product_model
    _, rankings = _direct_ranking(model)
    grades = {"query-zero": {"duplicate-b": 3.0, "other": 1.0},
              "query-one": {"positive-one": 2.0}}
    model.train()
    metrics = ManifoldGradedCorpusRetrievalEvaluator(
        model=model, queries=QUERIES, query_ids=QUERY_IDS,
        corpus=CORPUS, corpus_ids=CORPUS_IDS, graded_relevance=grades,
        recall_at_k=(1, 3), ndcg_at_k=(1, 3, 10),
        query_chunk_size=1, corpus_chunk_size=2,
    )()
    assert model.training
    for k in (1, 3, 10):
        expected = []
        for qid, ranking in zip(QUERY_IDS, rankings):
            gains = {cid: 2 ** grade - 1 for cid, grade in grades[qid].items()}
            dcg = sum(gains.get(CORPUS_IDS[i], 0) / math.log2(rank + 2)
                      for rank, i in enumerate(ranking[:k]))
            idcg = sum(gain / math.log2(rank + 2)
                       for rank, gain in enumerate(sorted(gains.values(), reverse=True)[:k]))
            expected.append(dcg / idcg)
        assert metrics[f"ndcg_at_{k}"] == pytest.approx(sum(expected) / 2)


@pytest.mark.parametrize("training", [True, False])
def test_product_miner_preserves_ids_exclusions_ties_and_mode(product_model, monkeypatch, training):
    model = product_model
    distances, rankings = _direct_ranking(model)
    positives = {"query-zero": ["duplicate-a"], "query-one": ["positive-one"]}
    exclusions = {"query-zero": ["duplicate-b"], "query-one": ["other"]}
    snapshot = copy.deepcopy((positives, exclusions, QUERY_IDS, CORPUS_IDS))
    batches, blocks = _record_blocks(monkeypatch, model, 1, 1)
    model.train(training)
    actual = mine_hard_negatives(
        model, QUERIES, CORPUS, query_ids=QUERY_IDS, corpus_ids=CORPUS_IDS,
        positive_corpus_ids=positives, excluded_corpus_ids=exclusions,
        num_negatives=2, query_chunk_size=1, corpus_chunk_size=1,
    )
    assert model.training == training
    assert snapshot == (positives, exclusions, QUERY_IDS, CORPUS_IDS)
    assert max(batches) == 1 and len(blocks) == 10
    for qi, (qid, rows) in enumerate(zip(QUERY_IDS, actual)):
        blocked = set(positives[qid]) | set(exclusions[qid]) | {qid}
        expected = [i for i in rankings[qi] if CORPUS_IDS[i] not in blocked][:2]
        assert [r["index"] for r in rows] == expected
        assert [r["corpus_id"] for r in rows] == [CORPUS_IDS[i] for i in expected]
        assert [r["candidate"] for r in rows] == [CORPUS[i] for i in expected]
        assert [r["distance"] for r in rows] == pytest.approx(
            [distances[qi][i] for i in expected], rel=1e-6, abs=1e-7)


def test_product_mined_negatives_train_with_existing_trainer(product_model):
    model = product_model
    negatives = mine_hard_negatives(
        model, QUERIES, CORPUS, query_ids=QUERY_IDS, corpus_ids=CORPUS_IDS,
        positive_corpus_ids=RELEVANCE, query_chunk_size=1, corpus_chunk_size=2,
    )
    evaluator = ManifoldCorpusRetrievalEvaluator(
        model=model, queries=QUERIES, query_ids=QUERY_IDS,
        corpus=CORPUS, corpus_ids=CORPUS_IDS, relevance=RELEVANCE,
        recall_at_k=(1, 3), query_chunk_size=1, corpus_chunk_size=2,
    )
    before = model.projection.weight.detach().clone()
    trainer = ManifoldTrainer(model, ManifoldMultipleNegativesRankingLoss(model, temperature=0.5),
                              learning_rate=1e-3, verbose=False)
    history = trainer.fit([(QUERIES, ["tie", "q1"],
                            [row[0]["candidate"] for row in negatives])],
                          epochs=2, evaluator=evaluator)
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert len(history) == 2 and model.training
    assert all(math.isfinite(row["train_loss"]) for row in history)
    assert all(math.isfinite(value) for row in history for value in row["validation"].values())
    assert not torch.equal(before, model.projection.weight)
