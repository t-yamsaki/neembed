# neembed

**Geoopt を使って、既存の sentence embedding モデルを非ユークリッド空間で fine-tuning するための軽量ライブラリ。**

[English README](../README.md)

> **Status:** Early development. 以下の API は v0.1 で目指す最小インターフェースであり、最初の安定版リリースまでは変更される可能性があります。

`neembed` は、既存のテキスト埋め込みモデルと多様体上の最適化をつなぐ薄い integration layer です。

基本アイデアはシンプルです。

```text
Pretrained Sentence Encoder
        │
        ▼
Euclidean sentence embedding
        │
        ▼
Projection head
        │
        ▼
Tangent-space representation
        │
        ▼
Geoopt manifold map
        │
        ▼
Non-Euclidean embedding
        │
        ▼
Geodesic-distance loss
```

リーマン幾何の演算を独自実装するのではなく、Poincaré ball や測地距離などの多様体演算は [Geoopt](https://geoopt.readthedocs.io/) に任せ、`neembed` は **既存の埋め込みモデルを非ユークリッド空間で学習しやすくすること**に集中します。

## なぜ neembed なのか

一般的な pretrained sentence embedding model は、テキストをユークリッド空間のベクトルとして表現します。

これは多くの意味類似度・検索タスクで有効ですが、データによっては平坦な空間では表現しづらい構造を持ちます。

例えば：

- taxonomy・概念階層
- knowledge graph
- 階層ラベル
- 木構造に近い意味関係
- 球面的・混合曲率的な潜在構造を持つデータ

特に双曲空間は、急速に分岐する木構造を比較的コンパクトに表現できるため、階層データとの相性が良いと考えられています。

`neembed` の目標は、このような非ユークリッド埋め込みの実験を、通常の Sentence Transformers の fine-tuning に近い感覚で実行できるようにすることです。

## 設計方針

`neembed` では、意図的に以下の原則を採用します。

1. **既存の pretrained embedding model を再利用する。**  
   Hugging Face や Sentence Transformers が提供している encoder を作り直さない。

2. **幾何演算は Geoopt を再利用する。**  
   exponential map、logarithmic map、geodesic distance、Riemannian optimizer などを理由なく独自実装しない。

3. **公開 API を小さく保つ。**  
   基本概念は `Model`、`Loss`、`Trainer` の3つにする。

4. **最初は1つの幾何に集中する。**  
   v0.1 は Poincaré ball に限定し、Lorentz、sphere、product manifold、mixed curvature は後から追加する。

5. **manifold-valued output と manifold-valued parameter を区別する。**  
   出力が多様体上にあることと、Riemannian optimizer が必要であることは同義ではない。

## スコープ

### v0.1

最初のリリースは意図的に狭くします。

- pretrained encoder の第一選択肢として Sentence Transformers を利用
- Geoopt による Poincaré ball embedding
- 接空間への任意の projection
- geodesic distance
- in-batch contrastive / multiple-negatives ranking objective
- `AdamW` による通常の fine-tuning
- embedding の生成と距離計算

### v0.1 では扱わないもの

以下は明示的に後回しにします。

- 多様体演算の独自実装
- 汎用 manifold registry
- Lorentz / sphere / SPD / product manifold
- learnable curvature
- manifold prototype
- Riemannian classifier
- distributed contrastive training
- 独自 vector database / ANN index
- 大規模な設定フレームワーク
- Sentence Transformers の代替となる汎用 training framework

## インストール

リリース後：

```bash
pip install neembed
```

開発版：

```bash
git clone https://github.com/<YOUR_USERNAME>/neembed.git
cd neembed
pip install -e ".[dev]"
```

主要な依存ライブラリは以下を想定しています。

```text
torch
sentence-transformers
geoopt
```

## Quick Start

v0.1 で目指す API は次の形です。

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

trainer.fit(train_dataset)

embeddings = model.encode([
    "柴犬",
    "犬",
    "哺乳類",
    "動物",
])

distance = model.distance(
    embeddings[0],
    embeddings[1],
)
```

通常の sentence embedding の fine-tuning から、できるだけ小さな概念変更だけで非ユークリッド埋め込みへ移行できることを目指します。

## モデルアーキテクチャ

初期の Poincaré 実装では次の構成を想定します。

```text
SentenceTransformer
        │
        │ pretrained sentence embedding h ∈ R^d
        ▼
Linear projection (optional)
        │
        │ tangent vector v ∈ T₀M
        ▼
Scale / stabilization
        │
        ▼
expmap₀
        │
        │ z ∈ M
        ▼
Poincaré embedding
```

概念的には、

\[
h = f_\theta(x),
\]

\[
v = Wh,
\]

\[
z = \operatorname{Exp}_0^c(v),
\]

とします。

ここで、

- \(f_\theta\)：pretrained sentence encoder
- \(W\)：任意の projection layer
- \(v\)：接空間上の表現
- \(z\)：最終的な manifold-valued embedding

です。

Transformer 本体まで双曲ニューラルネットワーク化するのではなく、Transformer は通常の PyTorch model のまま利用し、最終表現だけを多様体上へ写します。

## 学習目的

初期版の contrastive objective では、cosine similarity や Euclidean distance の代わりに geodesic distance を利用します。

anchor-positive pair のバッチに対して、

\[
s_{ij}
=
-\frac{d_{\mathcal M}(z_i, z_j^+)}{\tau}
\]

とします。

ここで \(d_{\mathcal M}\) は多様体上の geodesic distance、\(\tau\) は temperature です。

損失は、

\[
\mathcal L_i
=
-\log
\frac{\exp(s_{ii})}
{\sum_j \exp(s_{ij})}
\]

とし、in-batch multiple-negatives ranking / InfoNCE の manifold-aware な拡張として扱います。

## Optimizer

**非ユークリッド空間へ出力するだけなら、Riemannian optimizer は必須ではありません。**

最小モデルでは、

```text
Transformer parameters   ┐
Projection parameters    ├─ Euclidean parameters → AdamW
                         │
Output embeddings        └─ manifold 上へ map
```

という構造です。

manifold map と geodesic loss を通じて gradient が通常のモデルパラメータまで逆伝播するため、v0.1 では `AdamW` で十分です。

Geoopt の `RiemannianAdam` のような optimizer が必要になるのは、例えば次のように**学習パラメータそのものが多様体上に存在する場合**です。

- manifold prototype
- trainable hierarchy node
- class centroid
- manifold-valued entity embedding

これらは v0.1 の対象外とします。

## 想定パッケージ構成

```text
neembed/
├── pyproject.toml
├── README.md
├── docs/
│   └── README_ja.md
├── src/
│   └── neembed/
│       ├── __init__.py
│       ├── model.py
│       ├── manifolds.py
│       ├── losses.py
│       └── trainer.py
├── examples/
│   └── train_poincare.py
└── tests/
    ├── test_model.py
    ├── test_losses.py
    └── test_training.py
```

### `model.py`

pretrained sentence encoder と manifold-valued output の接続を担当します。

```python
model = ManifoldSentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    manifold="poincare",
    embedding_dim=64,
    curvature=1.0,
)
```

### `manifolds.py`

Geoopt に対する薄い interface のみを提供します。

v0.1 で必要なのは Poincaré ball だけです。

```python
import geoopt

def get_manifold(name: str, curvature: float = 1.0):
    if name == "poincare":
        return geoopt.PoincareBall(c=curvature)

    raise ValueError(f"Unsupported manifold: {name}")
```

実際の利用例が要求するまでは、独自の汎用 `Manifold` 抽象クラスは作りません。

### `losses.py`

manifold geometry に基づく loss を実装します。

v0.1 の中心：

```text
ManifoldMultipleNegativesRankingLoss
```

将来候補：

```text
ManifoldTripletLoss
ManifoldContrastiveLoss
HierarchyLoss
```

### `trainer.py`

Hugging Face や Sentence Transformers の training framework 全体を置き換えるのではなく、最小限の PyTorch training loop を提供します。

責務は以下に限定します。

- forward
- loss calculation
- backward
- optimizer step
- validation hook
- checkpoint

## データ

ライブラリ独自の dataset format は必須にしません。

pair data を使う場合、概念的には例えば次のようになります。

```json
{"anchor": "柴犬", "positive": "犬"}
{"anchor": "犬", "positive": "哺乳類"}
{"anchor": "哺乳類", "positive": "脊椎動物"}
```

ただし、利用者は通常の PyTorch Dataset や Hugging Face Dataset をそのまま使えることを目指します。

正例関係の例：

- 子ノード → 親ノード
- 文書 → 正しいカテゴリ
- query → relevant document
- paraphrase
- synonym
- click された query-document pair

## 評価

適切な評価指標はタスクによって異なります。

検索：

- Recall@K
- MRR
- nDCG

階層埋め込み：

- parent retrieval accuracy
- ancestor retrieval accuracy
- hierarchy reconstruction
- manifold radius と既知の hierarchy depth の Spearman 相関

一般的な representation learning：

- downstream classification
- clustering
- 元の Euclidean encoder との比較

実験では、fine-tuning 前の pretrained Euclidean model を baseline として必ず比較することを推奨します。

## 数値安定性

Poincaré ball では、embedding が境界付近へ近づくと数値的に不安定になる可能性があります。

実用上は次のような制御を検討します。

- tangent vector scaling
- Geoopt による projection / stabilization
- 保守的な learning rate
- curvature sweep
- temperature sweep
- gradient clipping

深い階層や最適化が難しい条件では、将来的に Lorentz model backend を追加する価値があります。

## Roadmap

### v0.1 — Minimal hyperbolic fine-tuning

- [ ] Sentence Transformer backbone
- [ ] Geoopt Poincaré ball
- [ ] projection head
- [ ] geodesic distance
- [ ] manifold multiple-negatives ranking loss
- [ ] minimal trainer
- [ ] `encode()`
- [ ] save / load
- [ ] basic tests
- [ ] minimal example

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

役割はもっと小さく定義します。

> **Geoopt を使って、pretrained sentence embedding model を manifold-valued embedding model に変換する。**

## Contributing

現在は v0.1 の surface area を小さく保つことを優先します。

大きな abstraction や subsystem を追加する前に、次の3点を確認します。

1. Poincaré fine-tuning に本当に必要か？
2. Geoopt や Sentence Transformers がすでに提供していないか？
3. core API を壊さず後から追加できないか？

大きな framework 化より、小さくテスト可能な追加を優先します。

## References

- [Geoopt documentation](https://geoopt.readthedocs.io/)
- [Sentence Transformers documentation](https://www.sbert.net/)
- [Creating Custom Sentence Transformer Models](https://www.sbert.net/docs/sentence_transformer/usage/custom_models.html)

## License

TBD.
