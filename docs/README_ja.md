# neembed

**Geoopt を使って、既存の sentence embedding モデルを非ユークリッド空間で fine-tuning するための軽量ライブラリ。**

[Documentation](https://neembed.readthedocs.io/en/latest/) · [English README](../README.md)

> **Status:** v0.1 の実装は完了しており、最初の PyPI 公開に向けて準備中です。公開 API は意図的に小さく保っていますが、安定版 1.0 までは変更される可能性があります。

`neembed` は、pretrained Sentence Transformer と manifold-valued representation をつなぐ軽量な integration layer です。pretrained encoder はそのまま利用し、必要に応じて Euclidean embedding を projection したうえで、Poincaré-ball の幾何演算を Geoopt に委譲します。

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

v0.1 は対象を広げすぎず、Poincaré ball のみに集中します。

## v0.1

最初のリリースには以下を含みます。

- pretrained encoder として Sentence Transformers を利用
- Geoopt による Poincaré-ball embedding
- 任意の低次元 tangent-space projection
- geodesic distance
- manifold-aware な multiple-negatives ranking / InfoNCE-style loss
- 通常の `AdamW` による fine-tuning
- `encode()` / `distance()` inference helper
- `save_pretrained()` / `from_pretrained()` によるローカル保存・復元
- 数値安定性テスト、実行例、Euclidean baseline experiment

詳細な挙動、前提、API signature は [Documentation](https://neembed.readthedocs.io/en/latest/) にまとめています。

## インストール

`v0.1.0` を PyPI に公開後：

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

各 anchor は同じ batch index の positive と対応します。off-diagonal candidate は in-batch negative になるため、同じ batch 内で positive を重複させないでください。目的関数とbatchingの詳細は [Training guide](https://neembed.readthedocs.io/en/latest/user_guide/training.html) を参照してください。

## ドキュメント

詳細ガイドは Read the Docs にあります。

- [Installation](https://neembed.readthedocs.io/en/latest/getting_started/installation.html)
- [Quick Start](https://neembed.readthedocs.io/en/latest/getting_started/quickstart.html)
- [Architecture](https://neembed.readthedocs.io/en/latest/user_guide/architecture.html)
- [Training](https://neembed.readthedocs.io/en/latest/user_guide/training.html)
- [Inference](https://neembed.readthedocs.io/en/latest/user_guide/inference.html)
- [Saving and Loading](https://neembed.readthedocs.io/en/latest/user_guide/saving_loading.html)
- [API Reference](https://neembed.readthedocs.io/en/latest/#api-reference)

## 実行例と検証

- [examples/train_poincare.py](../examples/train_poincare.py) に end-to-end の実行例があります。
- [experiments/README.md](../experiments/README.md) に Euclidean-vs-Poincaré の固定比較実験と結果の解釈上の注意をまとめています。

## License

`neembed` は [MIT License](../LICENSE) で公開します。third-party dependency はそれぞれのライセンスに従います。
