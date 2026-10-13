"""One role-aware training step, exact search, and an offline local round trip.

Run with --local-files-only after scripts/preload_role_models.py, or use
--model-path for a local text encoder. The example is not a retrieval benchmark.
"""

import argparse
import json
from tempfile import TemporaryDirectory

import torch

from neembed import ManifoldMultipleNegativesRankingLoss, ManifoldSentenceTransformer, exact_corpus_search


MODEL = "intfloat/e5-small-v2"
REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"


def run_example(model_name_or_path=MODEL, *, revision=REVISION, local_files_only=False):
    torch.manual_seed(0)
    model = ManifoldSentenceTransformer(
        model_name_or_path, revision=revision, local_files_only=local_files_only,
        device="cpu", embedding_dim=16,
    )
    model.encoder.prompts.update({"query": "query: ", "document": "passage: "})
    queries = ["What is a dog?", "Which animals can fly?"]
    documents = ["A dog is a mammal.", "Birds can fly.", "Trees are plants."]
    loss_fn = ManifoldMultipleNegativesRankingLoss(model, input_options={
        "anchors": {"task": "query"}, "positives": {"task": "document"},
    })
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()  # forward/loss follow the caller's mode; encode leaves the model in eval.
    optimizer.zero_grad()
    loss = loss_fn(queries, documents[:2])
    loss.backward()
    optimizer.step()

    embeddings = model.encode_query(queries, batch_size=1, convert_to_tensor=True)
    results = exact_corpus_search(model, queries, documents, top_k=2, input_options={
        "queries": {"task": "query"}, "corpus": {"task": "document"},
    })  # smallest geodesic distance first; result indices refer to original texts.
    with TemporaryDirectory() as checkpoint:
        model.save_pretrained(checkpoint)
        restored = ManifoldSentenceTransformer.from_pretrained(checkpoint, local_files_only=True, device="cpu")
        after = restored.encode_query(queries, batch_size=1, convert_to_tensor=True)
        torch.testing.assert_close(after, embeddings, rtol=1e-5, atol=1e-6)
    return {"loss": float(loss.detach()), "embedding_shape": list(embeddings.shape), "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--model-path", help="Use a saved local text encoder instead of pinned E5.")
    args = parser.parse_args()
    print(json.dumps(run_example(
        args.model_path or MODEL, revision=None if args.model_path else REVISION,
        local_files_only=args.local_files_only,
    ), indent=2))


if __name__ == "__main__":
    main()
