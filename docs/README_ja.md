# neembed

**Geoopt を使って、既存の sentence embedding モデルを非ユークリッド空間で fine-tuning するための軽量ライブラリ。**

[Documentation](https://neembed.readthedocs.io/en/latest/) · [English README](../README.md)

> **Status:** package version v0.8.0 では、caller-owned hierarchy supervision、radial / depth / directed hierarchy objective、retrieval-plus-hierarchy composition、structure evaluation、deterministic hierarchy regression example を追加しつつ、v0.4-v0.7 の公開 contract を維持しています。公開 API は意図的に小さく保っていますが、安定版 1.0 までは変更される可能性があります。

`neembed` は、pretrained Sentence Transformer と manifold-valued representation をつなぐ軽量な integration layer です。pretrained encoder はそのまま利用し、必要に応じて Euclidean embedding を projection したうえで、双曲幾何の演算を Geoopt に委譲します。

```text
Pretrained Sentence Encoder
        ↓
Euclidean sentence embedding
        ↓
Projection head (optional)
        ↓
Tangent-space representation
        ↓
Geoopt manifold map
        ↓
Non-Euclidean embedding
```

## なぜ非ユークリッド埋め込みなのか

階層的・木構造的な関係は、平坦なユークリッド空間では表現しづらい場合があります。双曲空間は、急速に分岐する次のような構造と相性のよい候補です。

- taxonomy・概念階層
- knowledge graph
- 階層ラベル
- 木構造に近い意味関係

現在の API は、Poincaré ball と Lorentz / Hyperboloid を同じ model・loss・trainer・evaluator・sentence-model save/load workflow で扱えます。

## 現在のスコープ

v0.4 は fixed-curvature の v0.3 path と後方互換です。

- pretrained encoder として Sentence Transformers を利用
- Geoopt による Poincaré-ball と Lorentz / Hyperboloid embedding
- 任意の低次元 tangent-space projection
- 2つの双曲モデルで統一した public curvature semantics
- geodesic distance と manifold-aware multiple-negatives ranking loss
- model-only path での通常の `AdamW` fine-tuning
- `encode()` / `distance()`、`ManifoldEmbeddingEvaluator`、DataLoader 連携、ローカル save/load
- geometry consistency regression と matched Euclidean-vs-Poincaré-vs-Lorentz engineering benchmark

v0.4 ではさらに以下を追加しています。

- Poincaré / Lorentz の fixed curvature と opt-in learnable curvature
- 真の manifold-valued trainable parameter である `ManifoldPrototypes`
- sentence assignment と parent-child structure を扱う `ManifoldPrototypeHierarchyLoss`
- manifold parameter 用の明示的な caller-supplied Geoopt Riemannian optimizer path
- Geoopt stabilization を使った learnable curvature + prototype の共同学習
- fixed-vs-learnable structure の compact regression example

v0.5 では、retrieval framework を追加せず、次の focused retrieval workflow を追加しています。

- 従来の `(anchors, positives)` contract を維持したまま使える caller-supplied explicit hard negatives
- aligned retrieval evaluator の Recall@K と MRR
- 小規模な in-memory geodesic reranking 用の `model.rank()`
- learned manifold structure 向け nearest-prototype assignment evaluation
- 1本の再現可能な v0.5 retrieval regression example

v0.6 では ANN infrastructure を追加せず、caller-owned corpus に対する exact retrieval workflow まで拡張します。

- full query-by-corpus distance matrix を materialize しない chunked exact geodesic distance evaluation
- exact text-corpus top-k retrieval 用の `exact_corpus_search()`
- explicit query/corpus ID と multi-positive Recall@K / MRR を扱う `ManifoldCorpusRetrievalEvaluator`
- positive / self / additional exclusion を明示できる deterministic caller-invoked `mine_hard_negatives()`
- exact search → evaluation → mining → explicit-negative training をつなぐ再現可能な v0.6 regression example

v0.7 では v0.4-v0.6 contract を変えずに、retrieval objective と graded evaluation の選択肢を追加します。

- aligned geodesic margin triplet 用の `ManifoldTripletLoss`
- teacher の positive-minus-negative margin を回帰する `ManifoldMarginMSELoss`
- geodesic distance を直接回帰する `ManifoldDistanceMSELoss`
- 既存 one-directional MNRL を変えずに追加する opt-in `ManifoldSymmetricMultipleNegativesRankingLoss`
- 既存 binary/multi-positive evaluator の出力を維持したまま nDCG@K を提供する `ManifoldGradedCorpusRetrievalEvaluator`
- MNRL / Triplet / MarginMSE / DistanceMSE と MRR / Recall@K / nDCG@K を固定条件で比較する deterministic Poincaré example

v0.8 では graph framework を導入せず、明示的な hierarchy-native learning を追加します。

- caller-owned node ID、parent-child edge、optional depth、directed negative
- `ManifoldRadialOrderLoss`、`ManifoldDepthLoss`、`ManifoldHierarchyTripletLoss`
- retrieval objective 1つと hierarchy objective 1つを組み合わせる `ManifoldRetrievalHierarchyLoss`
- radial-order と depth-vs-radius structure diagnostic を返す `ManifoldHierarchyEvaluator`
- retrieval-only と hierarchy-aware を比較する deterministic Poincaré regression example

新しい manifold family、ontology parsing、graph-database integration、ANN / vector database integration、distributed retrieval はこの scope の対象外です。

manifold-valued な **出力** を返すだけでは Riemannian optimization は必要ありません。encoder / projection parameter と learnable curvature は manifold 上の点ではありません。parameter・optimizer・persistence・numerical behavior の詳細は [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html) を参照してください。小規模 in-memory reranking、exact corpus search、外部 ANN system の境界を含む end-to-end retrieval workflow は [Retrieval workflow guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html) にまとめています。v0.7 の objective と graded evaluation の選び方は [Retrieval objectives guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval_objectives.html)、v0.8 の explicit hierarchy supervision・origin/radius semantics・composition・structure metrics は [Hierarchy-native learning guide](https://neembed.readthedocs.io/en/latest/user_guide/hierarchy.html) を参照してください。

## インストール

```bash
pip install neembed-geoopt
```

PyPI 上の distribution 名は `neembed-geoopt` ですが、Python の import package 名は `neembed` のままです。

開発版：

```bash
git clone https://github.com/t-yamsaki/neembed.git
cd neembed
pip install -e ".[dev]"
```

## Quick Start

```python
from neembed import (
    ManifoldSentenceTransformer,
    ManifoldMultipleNegativesRankingLoss,
    ManifoldTrainer,
)

model = ManifoldSentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    manifold="poincare",
    embedding_dim=64,
    curvature=1.0,
)

loss = ManifoldMultipleNegativesRankingLoss(
    model=model,
    temperature=0.1,
)

trainer = ManifoldTrainer(model=model, loss=loss)

train_batches = [
    (["柴犬", "シャム猫"], ["犬", "猫"]),
    (["犬", "猫"], ["哺乳類", "ネコ科"]),
]

trainer.fit(train_batches, epochs=1)

embeddings = model.encode(["柴犬", "犬", "哺乳類"])
distance = model.distance(embeddings[0], embeddings[1])
print(float(distance))
```

各 anchor は同じ batch index の positive と対応します。off-diagonal candidate は in-batch negative になるため、同じ batch 内で positive を重複させないでください。この model-only path は出力が manifold-valued でも通常の AdamW behavior のままです。目的関数と batching の詳細は [Training guide](https://neembed.readthedocs.io/en/latest/user_guide/training.html)、optional explicit negatives と retrieval evaluation は [Retrieval workflow guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html)、v0.7 の objective / metric 選択は [Retrieval objectives guide](https://neembed.readthedocs.io/en/latest/user_guide/retrieval_objectives.html)、v0.8 の explicit hierarchy supervision は [Hierarchy-native learning guide](https://neembed.readthedocs.io/en/latest/user_guide/hierarchy.html)、trainable manifold prototype を追加する前には [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html) を参照してください。

## ドキュメント

詳細ガイドは Read the Docs にあります。

- [Installation](https://neembed.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://neembed.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Architecture](https://neembed.readthedocs.io/en/latest/user_guide/architecture.html)
- [Learnable structure](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html)
- [Training](https://neembed.readthedocs.io/en/latest/user_guide/training.html)
- [Retrieval workflow](https://neembed.readthedocs.io/en/latest/user_guide/retrieval.html)
- [Retrieval objectives](https://neembed.readthedocs.io/en/latest/user_guide/retrieval_objectives.html)
- [Hierarchy-native learning](https://neembed.readthedocs.io/en/latest/user_guide/hierarchy.html)
- [Evaluation](https://neembed.readthedocs.io/en/latest/user_guide/evaluation.html)
- [Inference](https://neembed.readthedocs.io/en/latest/user_guide/inference.html)
- [Saving and Loading](https://neembed.readthedocs.io/en/latest/user_guide/saving_loading.html)
- [API Reference](https://neembed.readthedocs.io/en/latest/#api-reference)

## 実行例と検証

repository root から主な reference を実行できます。

```bash
python examples/train_poincare.py
python examples/train_lorentz.py
python examples/v04_learnable_structure.py
python examples/v05_retrieval_workflow.py
python examples/v06_exact_retrieval_workflow.py
python examples/v07_objective_comparison.py
python examples/v08_hierarchy_learning.py
```

- [examples/train_poincare.py](../examples/train_poincare.py) — 最小の Poincaré workflow
- [examples/train_lorentz.py](../examples/train_lorentz.py) — Lorentz の train / evaluate / inference と intrinsic-vs-ambient dimension
- [examples/train_dataloader.py](../examples/train_dataloader.py) — 通常の PyTorch `DataLoader` と epoch validation
- [examples/train_hierarchy.py](../examples/train_hierarchy.py) — 最小の hierarchy-aware prototype objective
- [examples/v04_learnable_structure.py](../examples/v04_learnable_structure.py) — fixed-vs-learnable structure の regression diagnostics。性能優位性を示す benchmark ではありません
- [examples/v05_retrieval_workflow.py](../examples/v05_retrieval_workflow.py) — explicit hard negatives、Recall@K / MRR、in-memory geodesic reranking、prototype assignment を1つにまとめた Poincaré regression workflow。research benchmark ではありません
- [examples/v06_exact_retrieval_workflow.py](../examples/v06_exact_retrieval_workflow.py) — exact corpus search、explicit-ID corpus evaluation、offline hard-negative mining、既存 three-sequence trainer を1つにまとめた Poincaré regression workflow。research benchmark ではありません
- [examples/v07_objective_comparison.py](../examples/v07_objective_comparison.py) — fixed data / initialization で MNRL、Triplet、MarginMSE、DistanceMSE と MRR、Recall@K、nDCG@K を比較する deterministic workflow。research benchmark や superiority claim ではありません
- [examples/v08_hierarchy_learning.py](../examples/v08_hierarchy_learning.py) — explicit caller-owned hierarchy supervision を使った retrieval-only vs hierarchy-aware の deterministic Poincaré regression。benchmark や superiority claim ではありません
- [experiments/README.md](../experiments/README.md) — 再現可能な Euclidean-vs-Poincaré-vs-Lorentz engineering benchmark と解釈上の注意

## License

`neembed` は [MIT License](../LICENSE) で公開します。third-party dependency はそれぞれのライセンスに従います。
