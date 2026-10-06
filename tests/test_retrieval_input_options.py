"""Role-aware retrieval, evaluation and mining share training preprocessing."""

import math

import pytest
import torch
from torch import nn

import neembed.model as model_module
from neembed import (
    ManifoldCorpusRetrievalEvaluator, ManifoldEmbeddingEvaluator,
    ManifoldGradedCorpusRetrievalEvaluator, ManifoldHierarchyEvaluator,
    ManifoldMultipleNegativesRankingLoss, ManifoldPrototypeAssignmentEvaluator,
    ManifoldPrototypes, ManifoldSentenceTransformer, exact_corpus_search,
    mine_hard_negatives,
)


QUERIES = ["q1", "q0"]
CORPUS = ["d0", "d0dup", "d1", "d2", "d3"]
QUERY_IDS = ["query-B", "query-A"]
CORPUS_IDS = ["positive-A", "duplicate-A", "positive-B", "negative-C", "negative-D"]
RELEVANCE = {"query-B": ["positive-B"], "query-A": ["positive-A", "duplicate-A"]}
APIS = ("rank", "search", "binary", "graded", "mining", "pair", "hierarchy", "prototype")


class PrefixEncoder(nn.Module):
    """Different input types require different prefixes and routes when opted in."""

    def __init__(self, model_name_or_path):
        super().__init__()
        self.linear = nn.Linear(3, 4, bias=False)
        self.prompts = {
            "query": "query: ", "document": "passage: ",
            "named_query": "Q> ", "named_document": "D> ", "default": "default: ",
        }
        self.default_prompt_name = "default"
        self.require_roles = False
        self.calls = []
        self.rows = {
            "q0": [0.2, 0.1, 0.3], "q1": [-0.2, 0.1, 0.3],
            "d0": [0.2, 0.1, 0.3], "d0dup": [0.2, 0.1, 0.3],
            "d1": [-0.2, 0.1, 0.3], "d2": [0.1, -0.3, 0.2], "d3": [0.0, 0.2, -0.3],
        }

    @property
    def device(self):
        return self.linear.weight.device

    def get_embedding_dimension(self):
        return 4

    def preprocess(self, texts, *, prompt=None, task=None):
        if self.require_roles:
            for text in texts:
                expected_task = "query" if text.startswith("q") else "document"
                assert task == expected_task
                allowed = {
                    "query": ("query: ", "Q> ", "query explicit: ", ""),
                    "document": ("passage: ", "D> ", "document explicit: ", ""),
                }
                assert prompt in allowed[task]
        self.calls.append((tuple(texts), prompt, task, torch.is_grad_enabled()))
        return {
            "features": torch.tensor([
                [self.rows[text][0] + len(prompt or "") / 100, *self.rows[text][1:]]
                for text in texts
            ]),
            "task": task,
            "prompt_length": len(prompt or ""),
        }

    def forward(self, features, *, task=None):
        assert features["task"] == task
        assert "prompt_length" in features
        return {"sentence_embedding": self.linear(features["features"])}

    def encode(self, *args, **kwargs):
        raise AssertionError("neembed must use its shared model forward path")


def make_model(monkeypatch, manifold="poincare"):
    monkeypatch.setattr(model_module, "SentenceTransformer", PrefixEncoder)
    torch.manual_seed(0)
    kwargs = {"manifold": manifold, "embedding_dim": 2}
    if manifold == "product":
        kwargs = {"product_config": [
            {"name": "tree", "manifold": "lorentz", "intrinsic_dim": 2, "scale": 1.5},
            {"name": "flat", "manifold": "euclidean", "intrinsic_dim": 2},
        ]}
    return ManifoldSentenceTransformer("fake-model", **kwargs)


def role_options(mode="role"):
    query, document = {"task": "query"}, {"task": "document"}
    if mode == "named":
        query["prompt_name"], document["prompt_name"] = "named_query", "named_document"
    elif mode == "explicit":
        query["prompt"], document["prompt"] = "query explicit: ", "document explicit: "
    elif mode == "empty":
        query["prompt"], document["prompt"] = "", ""
    return query, document


def make_call(api, model, query, document, *, chunk=2):
    options = {"queries": query, "corpus": document}
    if api == "rank":
        return lambda: model.rank(
            QUERIES[0], CORPUS, input_options={"query": query, "candidates": document},
        )
    if api == "search":
        return lambda: exact_corpus_search(
            model, QUERIES, CORPUS, query_chunk_size=chunk, corpus_chunk_size=chunk,
            input_options=options,
        )
    common = dict(model=model, queries=QUERIES, corpus=CORPUS,
                  query_ids=QUERY_IDS, corpus_ids=CORPUS_IDS,
                  query_chunk_size=chunk, corpus_chunk_size=chunk, input_options=options)
    if api == "mining":
        return lambda: mine_hard_negatives(
            **common, positive_corpus_ids=RELEVANCE, num_negatives=2,
            excluded_corpus_ids={"query-B": ["negative-D"], "query-A": ["positive-B"]},
        )
    if api == "binary":
        return ManifoldCorpusRetrievalEvaluator(**common, relevance=RELEVANCE, recall_at_k=(1, 3))
    if api == "graded":
        return ManifoldGradedCorpusRetrievalEvaluator(
            **common, graded_relevance={
                "query-B": {"positive-B": 2.0},
                "query-A": {"positive-A": 2.0, "duplicate-A": 1.0},
            }, recall_at_k=(1, 3), ndcg_at_k=(1, 3),
        )
    if api == "pair":
        return ManifoldEmbeddingEvaluator(
            model=model, anchors=QUERIES, positives=["d1", "d0"],
            input_options={"anchors": query, "positives": document},
        )
    if api == "hierarchy":
        return ManifoldHierarchyEvaluator(
            model=model, node_ids=["root", "leaf"], texts=["d0", "d2"],
            parent_child_edges=[("root", "leaf")], depths={"root": 0, "leaf": 1},
            input_options={"texts": document},
        )
    prototypes = ManifoldPrototypes(model, num_prototypes=2, init_std=0.05)
    return ManifoldPrototypeAssignmentEvaluator(
        model=model, prototypes=prototypes, prototype_ids=["root", "leaf"],
        sentences=["d0", "d1"], expected_prototype_ids=["root", "leaf"],
        input_options={"sentences": document},
    )


@pytest.mark.parametrize("api", APIS)
@pytest.mark.parametrize("mode", ["role", "named", "explicit", "empty"])
def test_every_api_applies_matching_roles_and_prefixes_without_gradients(monkeypatch, api, mode):
    model = make_model(monkeypatch)
    model.encoder.require_roles = True
    query, document = role_options(mode)
    call = make_call(api, model, query, document)
    before = [p.detach().clone() for p in model.parameters()]
    result = call()
    assert result
    if isinstance(result, dict):
        assert all(math.isfinite(value) for value in result.values())
    for texts, prefix, task, grad_enabled in model.encoder.calls:
        options = query if task == "query" else document
        expected_prefix = options.get("prompt", model.encoder.prompts[
            options.get("prompt_name", task)
        ])
        assert prefix == expected_prefix
        assert not grad_enabled
        assert all(text in QUERIES + CORPUS for text in texts)
    # nDCG and binary rank counting must both use the selected roles.
    multiplier = 2 if api == "graded" else 1
    if api in ("search", "binary", "graded", "mining"):
        assert [text for call in model.encoder.calls for text in call[0]] == (
            QUERIES + CORPUS
        ) * multiplier
        assert max(len(call[0]) for call in model.encoder.calls) <= 2
    for parameter, original in zip(model.parameters(), before):
        assert parameter.grad is None
        assert torch.equal(parameter, original)


@pytest.mark.parametrize("api", APIS)
def test_missing_settings_keep_raw_text_even_with_saved_default(monkeypatch, api):
    model = make_model(monkeypatch)
    call = make_call(api, model, {}, {})
    call()
    assert model.encoder.calls
    assert all(prefix is None and task is None for _, prefix, task, _ in model.encoder.calls)


def full_reference(model, query, document):
    q = model.encode(QUERIES, convert_to_tensor=True, **query)
    c = model.encode(CORPUS, convert_to_tensor=True, **document)
    distances = model.distance(q[:, None], c[None]).tolist()
    return [[{
        "index": index, "candidate": CORPUS[index], "distance": row[index],
    } for index in sorted(range(len(row)), key=lambda index: (row[index], index))]
        for row in distances]


@pytest.mark.parametrize("manifold", ["poincare", "lorentz", "product"])
@pytest.mark.parametrize("query_chunk,corpus_chunk", [(1, 1), (2, 3), (99, 99)])
def test_chunked_search_matches_full_matrix_query_order_and_stable_ties(
    monkeypatch, manifold, query_chunk, corpus_chunk,
):
    model = make_model(monkeypatch, manifold)
    query, document = role_options()
    expected = full_reference(model, query, document)
    model.encoder.calls.clear()
    actual = exact_corpus_search(
        model, QUERIES, CORPUS, query_chunk_size=query_chunk, corpus_chunk_size=corpus_chunk,
        input_options={"queries": query, "corpus": document},
    )
    for results, reference in zip(actual, expected):
        assert [r["index"] for r in results] == [r["index"] for r in reference]
        # Float32 encoder kernels can round differently at different batch sizes.
        assert [r["distance"] for r in results] == pytest.approx(
            [r["distance"] for r in reference], abs=1e-6,
        )
        indices = [r["index"] for r in results]
        assert indices.index(0) < indices.index(1)  # Identical points, original corpus order.
    top = exact_corpus_search(
        model, QUERIES, CORPUS, top_k=3, query_chunk_size=query_chunk,
        corpus_chunk_size=corpus_chunk, input_options={"queries": query, "corpus": document},
    )
    assert top == [row[:3] for row in actual]
    for texts, _, task, _ in model.encoder.calls:
        assert len(texts) <= (query_chunk if task == "query" else corpus_chunk)


@pytest.mark.parametrize("api", ["binary", "graded", "mining"])
def test_evaluation_and_mining_match_full_chunks_and_filter_caller_ids(monkeypatch, api):
    model = make_model(monkeypatch)
    query, document = role_options()
    chunked = make_call(api, model, query, document, chunk=1)()
    full = make_call(api, model, query, document, chunk=99)()
    if api == "mining":
        rankings = full_reference(model, query, document)
        blocked = ({"positive-B", "negative-D"}, {"positive-A", "duplicate-A", "positive-B"})
        for result, reference, exclusions in zip(chunked, rankings, blocked):
            expected = [r for r in reference if CORPUS_IDS[r["index"]] not in exclusions][:2]
            assert [r["index"] for r in result] == [r["index"] for r in expected]
            assert [r["corpus_id"] for r in result] == [CORPUS_IDS[r["index"]] for r in expected]
            assert [r["distance"] for r in result] == pytest.approx(
                [r["distance"] for r in expected], abs=1e-6,
            )
        for left, right in zip(chunked, full):
            assert [(r["index"], r["corpus_id"], r["candidate"]) for r in left] == [
                (r["index"], r["corpus_id"], r["candidate"]) for r in right
            ]
            assert [r["distance"] for r in left] == pytest.approx(
                [r["distance"] for r in right], abs=1e-6,
            )
    else:
        assert chunked == pytest.approx(full)


@pytest.mark.parametrize("api", ["binary", "graded", "mining", "pair", "hierarchy", "prototype"])
@pytest.mark.parametrize("training", [False, True])
def test_evaluators_and_miner_restore_mode_on_success_and_prompt_error(monkeypatch, api, training):
    model = make_model(monkeypatch)
    query, document = role_options()
    model.train(training)
    make_call(api, model, query, document)()
    assert model.training is training
    document = {"task": "document", "prompt_name": "unknown"}
    with pytest.raises(ValueError, match="unknown encoder prompt"):
        make_call(api, model, query, document)()
    assert model.training is training


@pytest.mark.parametrize("manifold", ["poincare", "lorentz", "product"])
def test_loss_and_retrieval_encode_the_same_inputs_with_the_same_settings(monkeypatch, manifold):
    model = make_model(monkeypatch, manifold)
    query, document = role_options("named")
    captured = []

    def capture(module, args, output):
        captured.append(output.detach().clone())

    handle = model.register_forward_hook(capture)
    loss = ManifoldMultipleNegativesRankingLoss(model, input_options={
        "anchors": query, "positives": document, "negatives": document,
    })
    positives, negatives = ["d1", "d0"], ["d2", "d3"]
    value = loss(QUERIES, positives, negatives)
    value.backward()
    handle.remove()
    training_calls = [(texts, prefix, task) for texts, prefix, task, _ in model.encoder.calls]
    model.encoder.calls.clear()
    exact_corpus_search(model, QUERIES, positives + negatives, query_chunk_size=2, corpus_chunk_size=2,
                        input_options={"queries": query, "corpus": document})
    assert [(texts, prefix, task) for texts, prefix, task, _ in model.encoder.calls] == training_calls
    q = model.encode(QUERIES, convert_to_tensor=True, **query)
    p = model.encode(positives, convert_to_tensor=True, **document)
    n = model.encode(negatives, convert_to_tensor=True, **document)
    for actual, expected in zip((q, p, n), captured):
        assert torch.allclose(actual, expected)
    assert torch.isfinite(model.encoder.linear.weight.grad).all()


@pytest.mark.parametrize("api", ["binary", "graded", "pair", "hierarchy", "prototype"])
def test_evaluator_options_are_snapshots_of_callers_dictionaries(monkeypatch, api):
    model = make_model(monkeypatch)
    model.encoder.require_roles = True
    query, document = role_options()
    evaluator = make_call(api, model, query, document)
    query["prompt"] = "changed query"
    document["task"], document["prompt"] = "query", "changed document"
    evaluator()
    assert all(prefix in ("query: ", "passage: ") for _, prefix, _, _ in model.encoder.calls)


@pytest.mark.parametrize("api", ["search", "binary", "graded", "mining"])
def test_partial_options_do_not_infer_corpus_settings(monkeypatch, api):
    model = make_model(monkeypatch)
    query, _ = role_options()
    make_call(api, model, query, {})()
    for texts, prefix, task, _ in model.encoder.calls:
        if texts[0].startswith("q"):
            assert (prefix, task) == ("query: ", "query")
        else:
            assert (prefix, task) == (None, None)


def test_input_and_option_typos_are_rejected_before_encoding(monkeypatch):
    model = make_model(monkeypatch)
    with pytest.raises(ValueError, match="unknown text input"):
        model.rank("q0", CORPUS, input_options={"queries": {"task": "query"}})
    with pytest.raises(ValueError, match="unknown text input"):
        exact_corpus_search(model, QUERIES, CORPUS, input_options={"query": {}})
    with pytest.raises(ValueError, match="unknown forward option"):
        exact_corpus_search(model, QUERIES, CORPUS, input_options={"queries": {"role": "query"}})
    with pytest.raises(ValueError, match="unknown text input"):
        ManifoldHierarchyEvaluator(
            model=model, node_ids=["root", "leaf"], texts=["d0", "d2"],
            parent_child_edges=[("root", "leaf")], input_options={"node_ids": {}},
        )
    assert model.encoder.calls == []


def test_real_router_search_matches_native_query_document_encoding(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.base.modules import Router
    from sentence_transformers.sentence_transformer.modules import BoW, Dense

    torch.manual_seed(0)
    encoder = SentenceTransformer(modules=[Router.for_query_document(
        query_modules=[BoW(["query", "q0", "q1"]), Dense(3, 4)],
        document_modules=[BoW(["passage", *CORPUS]), Dense(6, 4)],
    )], prompts={"query": "query: ", "document": "passage: "}, device="cpu")
    path = tmp_path / "router"
    encoder.save_pretrained(str(path))
    model = ManifoldSentenceTransformer(str(path), manifold="euclidean",
                                        local_files_only=True, device="cpu")
    query, document = role_options()
    q = model.encoder.encode_query(QUERIES, convert_to_tensor=True)
    c = model.encoder.encode_document(CORPUS, convert_to_tensor=True)
    native_distances = model.distance(q[:, None], c[None]).tolist()
    results = make_call("search", model, query, document, chunk=1)()
    for row, distances in zip(results, native_distances):
        expected = sorted(range(len(CORPUS)), key=lambda index: (distances[index], index))
        assert [r["index"] for r in row] == expected
        assert [r["distance"] for r in row] == pytest.approx(
            [distances[i] for i in expected], abs=1e-6,
        )
    reranked = make_call("rank", model, query, document)()
    assert [r["index"] for r in reranked] == [r["index"] for r in results[0]]
    assert [r["distance"] for r in reranked] == pytest.approx(
        [r["distance"] for r in results[0]], abs=1e-6,
    )
    for api in ("binary", "graded", "mining"):
        assert make_call(api, model, query, document)()
