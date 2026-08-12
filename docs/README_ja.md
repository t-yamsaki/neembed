# neembed

**Geoopt を使って、既存の sentence embedding モデルを非ユークリッド空間で fine-tuning するための軽量ライブラリ。**

[Documentation](https://neembed.readthedocs.io/en/latest/) · [English README](../README.md)

> **Status:** 最新の tag / PyPI release は v0.3.0 です。`main` には v0.4 開発機能として、opt-in の learnable curvature、trainable manifold prototype、hierarchy-aware objective、manifold-valued parameter 用の caller-supplied Riemannian optimization、learnable-structure regression example も含まれています。公開 API は意図的に小さく保っていますが、安定版 1.0 までは変更される可能性があります。

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

現在の development API は、Poincaré ball と Lorentz / Hyperboloid を同じ model・loss・trainer・evaluator・sentence-model save/load workflow で扱えます。

## 現在のスコープ

release 済みの v0.3 path には以下を含みます。

- pretrained encoder として Sentence Transformers を利用
- Geoopt による Poincaré-ball と Lorentz / Hyperboloid embedding
- 任意の低次元 tangent-space projection
- 2つの双曲モデルで統一した public curvature semantics
- geodesic distance と manifold-aware multiple-negatives ranking loss
- model-only path での通常の `AdamW` fine-tuning
- `encode()` / `distance()`、`ManifoldEmbeddingEvaluator`、DataLoader 連携、ローカル save/load
- geometry consistency regression と matched Euclidean/Poincaré/Lorentz engineering benchmark

v0.4 development API ではさらに以下を追加しています。

- Poincaré / Lorentz の fixed curvature と opt-in learnable curvature
- 真の manifold-valued trainable parameter である `ManifoldPrototypes`
- sentence assignment と parent-child structure を扱う `ManifoldPrototypeHierarchyLoss`
- manifold parameter 用の明示的な caller-supplied Geoopt Riemannian optimizer path
- Geoopt stabilization を使った learnable curvature + prototype の共同学習
- fixed-vs-learnable structure の compact regression example

manifold-valued な **出力** を返すだけでは Riemannian optimization は必要ありません。encoder / projection parameter と learnable curvature は manifold 上の点ではありません。parameter・optimizer・persistence・numerical behavior の詳細は [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html) を参照してください。

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

各 anchor は同じ batch index の positive と対応します。off-diagonal candidate は in-batch negative になるため、同じ batch 内で positive を重複させないでください。この model-only path は出力が manifold-valued でも通常の AdamW behavior のままです。目的関数と batching の詳細は [Training guide](https://neembed.readthedocs.io/en/latest/user_guide/training.html)、trainable manifold prototype を追加する前には [Learnable structure guide](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html) を参照してください。

## ドキュメント

詳細ガイドは Read the Docs にあります。

- [Installation](https://neembed.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://neembed.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Architecture](https://neembed.readthedocs.io/en/latest/user_guide/architecture.html)
- [Learnable structure](https://neembed.readthedocs.io/en/latest/user_guide/learnable_structure.html)
- [Training](https://neembed.readthedocs.io/en/latest/user_guide/training.html)
- [Evaluation](https://neembed.readthedocs.io/en/latest/user_guide/evaluation.html)
- [Inference](https://neembed.readthedocs.io/en/latest/user_guide/inference.html)
- [Saving and Loading](https://neembed.readthedocs.io/en/latest/user_guide/saving_loading.html)
- [API Reference](https://neembed.readthedocs.io/en/latest/#api-reference)

## 実行例と検証

主な runnable reference は以下です。

- [examples/train_poincare.py](../examples/train_poincare.py) — 最小の Poincaré workflow
- [examples/train_lorentz.py](../examples/train_lorentz.py) — Lorentz の train / evaluate / inference と intrinsic-vs-ambient dimension
- [examples/train_dataloader.py](../examples/train_dataloader.py) — 通常の PyTorch `DataLoader` と epoch validation
- [examples/train_hierarchy.py](../examples/train_hierarchy.py) — 最小の hierarchy-aware prototype objective
- [examples/v04_learnable_structure.py](../examples/v04_learnable_structure.py) — fixed-vs-learnable structure の regression diagnostics。性能優位性を示す benchmark ではありません
- [experiments/README.md](../experiments/README.md) — 再現可能な Euclidean-vs-Poincaré-vs-Lorentz engineering benchmark と解釈上の注意

## License

`neembed` は [MIT License](../LICENSE) で公開します。third-party dependency はそれぞれのライセンスに従います。
