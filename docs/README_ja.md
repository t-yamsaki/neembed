# neembed

**Geoopt を使って、既存の sentence embedding モデルを非ユークリッド空間で fine-tuning するための軽量ライブラリ。**

[English README](../README.md)

> **Status:** v0.1 の実装は完了しており、最初の PyPI 公開に向けて準備中です。公開 API は意図的に小さく保っていますが、安定版 1.0 までは変更される可能性があります。

`neembed` は、既存のテキスト埋め込みモデルと manifold-valued representation をつなぐ薄い integration layer です。

```text
Pretrained Sentence Encoder
        │
        ▼
Euclidean sentence embedding
        │
        ▼
Projection head (optional)
        │
        ▼
Tangent-space representation
        │
        ▼
Geoopt expmap₀
        │
        ▼
Non-Euclidean embedding
        │
        ▼
Geodesic-distance loss
```

リーマン幾何の演算を独自実装するのではなく、Poincaré ball や geodesic distance などの多様体演算は [Geoopt](https://geoopt.readthedocs.io/) に任せ、`neembed` は **既存の Sentence Transformer を非ユークリッド空間で学習しやすくすること**に集中します。

## なぜ neembed なのか

一般的な pretrained sentence embedding model はテキストをユークリッド空間のベクトルとして表現します。多くの意味類似度・検索タスクでは有効ですが、階層的・木構造的なデータは平坦な空間では表現しづらい場合があります。

代表例：

- taxonomy・概念階層
- knowledge graph
- 階層ラベル
- 木構造に近い意味関係

双曲空間は、急速に分岐する木構造を比較的コンパクトに表現できるため、階層データとの相性が良いと考えられています。

## 設計方針

1. **既存の pretrained embedding model を再利用する。** Sentence Transformers / Hugging Face が提供する encoder を作り直さない。
2. **幾何演算は Geoopt を再利用する。** exponential map、logarithmic map、geodesic distance などを理由なく独自実装しない。
3. **公開 API を小さく保つ。** 基本概念は `Model`、`Loss`、`Trainer` の3つ。
4. **最初は1つの幾何に集中する。** v0.1 は Poincaré ball のみをサポート。
5. **manifold-valued output と manifold-valued parameter を区別する。** 出力が多様体上にあるだけなら Riemannian optimizer は必須ではない。

## v0.1 のスコープ

最初のリリースには以下を含みます。

- pretrained encoder として Sentence Transformers を利用
- Geoopt による Poincaré ball embedding
- 接空間への任意の低次元 projection
- geodesic distance
- in-batch multiple-negatives ranking / InfoNCE-style loss
- `AdamW` による通常の fine-tuning
- `encode()` / `distance()`
- `save_pretrained()` / `from_pretrained()` によるローカル保存・復元
- Poincaré path の数値安定性テスト
- end-to-end の実行例
- Euclidean と Poincaré を比較する再現可能な baseline experiment

v0.1 では扱いません。

- 多様体演算の独自実装
- 汎用 manifold registry
- Lorentz / sphere / SPD / product manifold
- learnable curvature
- manifold-valued な trainable prototype / classifier
- distributed contrastive training
- vector database / ANN
- Sentence Transformers の置き換え

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

runtime dependency は意図的に次の3つへ限定しています。

```text
torch
sentence-transformers
geoopt
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

trainer = ManifoldTrainer(
    model=model,
    loss=loss,
)

train_batches = [
    (["柴犬", "シャム猫"], ["犬", "猫"]),
    (["犬", "猫"], ["哺乳類", "ネコ科"]),
]

trainer.fit(train_batches, epochs=1)

embeddings = model.encode([
    "柴犬",
    "犬",
    "哺乳類",
])

distance = model.distance(embeddings[0], embeddings[1])
print(float(distance))
```

各 anchor は同じ batch index の positive と対応します。off-diagonal の positive candidate は in-batch negative として扱われるため、同じ batch 内で positive を重複させないでください。

完全な実行例は [examples/train_poincare.py](../examples/train_poincare.py) を参照してください。

```bash
python examples/train_poincare.py
```

## モデルアーキテクチャ

v0.1 の Poincaré 実装は次の構成です。

```text
SentenceTransformer
        │
        │ pretrained sentence embedding h ∈ R^d
        ▼
Linear projection (optional)
        │
        │ tangent vector v ∈ T₀M
        ▼
Geoopt expmap₀
        │
        │ z ∈ M
        ▼
Poincaré embedding
```

概念的には、

$$
h = f_\theta(x),
$$

$$
v = Wh,
$$

$$
z = \mathrm{Exp}_0^c(v),
$$

です。$f_\theta$ は pretrained sentence encoder、$W$ は任意の projection layer、$z$ は最終的な manifold-valued embedding です。

## 学習目的

anchor-positive pair のバッチに対して、Euclidean / cosine similarity の代わりに負の geodesic distance を利用します。

$$
s_{ij} = -\frac{d_{\mathcal M}(z_i, z_j^+)}{\tau}.
$$

in-batch objective は、

$$
\mathcal L_i = -\log\frac{\exp(s_{ii})}{\sum_j \exp(s_{ij})}.
$$

となり、multiple-negatives ranking / InfoNCE の manifold-aware な形として扱います。

## v0.1 で AdamW を使う理由

encoder と optional projection は通常の Euclidean PyTorch parameter です。出力だけを Geoopt を通して多様体上へ写すため、gradient は manifold map / geodesic distance を通じて逆伝播でき、trainable parameter は通常の `AdamW` で最適化できます。

trainable hierarchy node や manifold prototype のように、学習パラメータそのものが多様体上にある場合は Riemannian optimizer が必要になりますが、v0.1 の対象外です。

## 保存と読み込み

```python
model.save_pretrained("./saved_model")

loaded = ManifoldSentenceTransformer.from_pretrained("./saved_model")
```

保存ディレクトリには underlying Sentence Transformer の状態と、neembed の projection / manifold configuration が保存されます。

## Validation experiment

最初の比較実験では、fine-tuning 前の Euclidean encoder と Poincaré fine-tuning 後の同じ encoder を、同じ held-out hierarchy retrieval pair で比較します。

```bash
python experiments/compare_euclidean_poincare.py
```

固定設定と結果の読み方は [experiments/README.md](../experiments/README.md) を参照してください。この小規模実験は、双曲埋め込みが常に優れていると主張する benchmark ではありません。

## 数値安定性

Poincaré embedding は ball の境界付近で数値的に不安定になり得ます。v0.1 では projection / geometry を Geoopt に委譲し、以下をテストしています。

- representative / large tangent vector から得られる embedding が finite
- embedding が有効な Poincaré ball domain 内にある
- near-boundary geodesic distance が finite
- forward / loss path の gradient が finite
- curvature / temperature の edge case

## Roadmap

### v0.1 — Minimal hyperbolic fine-tuning

- [x] Sentence Transformer backbone
- [x] Geoopt Poincaré ball
- [x] optional projection head
- [x] geodesic distance
- [x] manifold multiple-negatives ranking loss
- [x] minimal trainer
- [x] `encode()`
- [x] save / load
- [x] numerical stability tests
- [x] minimal end-to-end example
- [x] Euclidean baseline experiment

### v0.2 — More objectives and geometries

- [ ] triplet loss
- [ ] explicit hard negatives
- [ ] Lorentz model
- [ ] evaluation utilities
- [ ] Hugging Face Dataset examples

### v0.3 — Learnable manifold structure

- [ ] learnable curvature
- [ ] `geoopt.ManifoldParameter`
- [ ] RiemannianAdam
- [ ] manifold prototypes
- [ ] hierarchical objectives

### Later

- [ ] spherical embeddings
- [ ] product manifolds
- [ ] mixed-curvature representations
- [ ] distributed contrastive learning
- [ ] retrieval / reranking integrations

## neembed が目指さないもの

`neembed` は以下を目指しません。

- 新しい Riemannian geometry framework
- Geoopt の代替
- Sentence Transformers の代替
- 汎用 hyperbolic neural-network library
- vector database

役割は意図的に小さく定義します。

> **Geoopt を使って、pretrained sentence embedding model を manifold-valued embedding model に変換する。**

## Contributing

大きな abstraction や subsystem を追加する前に、次の3点を確認します。

1. 対象となる manifold fine-tuning workflow に本当に必要か？
2. Geoopt や Sentence Transformers がすでに提供していないか？
3. core API を壊さず後から追加できないか？

大きな framework 化より、小さくテスト可能な追加を優先します。

## References

- [Geoopt documentation](https://geoopt.readthedocs.io/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [Creating Custom Sentence Transformer Models](https://www.sbert.net/docs/sentence_transformer/usage/custom_models.html)

## License

`neembed` は [MIT License](../LICENSE) で公開します。

Geoopt を含む third-party dependency は、それぞれのライセンスに従います。neembed は Geoopt の source code を vendoring / relicensing せず、依存ライブラリとして利用します。
