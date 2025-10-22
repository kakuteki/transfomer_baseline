# 独英翻訳Transformerモデル

PyTorchで実装された6層Transformerモデルによる独→英翻訳システムです。Multi30kデータセットを使用して訓練されます。

## 特徴

- **本格的な6層Transformer**: 「Attention Is All You Need」論文に準拠したエンコーダー・デコーダーアーキテクチャ
- **Multi30kデータセット**: 約29,000の独英翻訳ペアを使用
- **マルチヘッドアテンション**: 8ヘッドのセルフアテンション機構
- **ビームサーチ対応**: グリーディサーチに加えてビームサーチにも対応
- **インタラクティブ翻訳**: 訓練後に対話的に翻訳を実行可能
- **チェックポイント機能**: 学習中断時の自動再開に対応
- **BLEU監視**: 初期エポックでの学習品質チェック

## 学習結果（Multi30k-pretrainingブランチ）

100エポックの学習を完了し、以下の結果を達成しました：

### 最終スコア（Epoch 100）
- **BLEUスコア**: 16.91
- **訓練損失**: 0.592（初期値: 5.52）
- **検証損失**: 2.597（初期値: 4.11）
- **総パラメータ数**: 16,286,767

### BLEUスコア推移

| エポック | BLEUスコア | 訓練損失 | 検証損失 | 備考 |
|---------|-----------|---------|---------|------|
| 1       | 0.13      | 5.52    | 4.11    | 学習開始 |
| 5       | 10.89     | 2.61    | 2.50    | 急速な改善 |
| 10      | 14.61     | 2.11    | 2.19    | 安定した向上 |
| 15      | 15.76     | 1.75    | 2.10    | 収束開始 |
| 20      | 16.97     | 1.53    | 2.09    | ピーク付近 |
| 25      | 17.73     | 1.36    | 2.13    | 最高スコア |
| 100     | 16.91     | 0.59    | 2.60    | 最終結果 |

### 学習環境
- **GPU**: NVIDIA A100-SXM4-80GB
- **学習時間**: 約2時間（100エポック）
- **フレームワーク**: PyTorch + CUDA 12.1

## 必要な依存関係

```bash
pip install -r requirements.txt
```

### spaCyモデルのインストール

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## 使用方法

### Docker使用（推奨）

#### 1. イメージのビルド

```bash
docker-compose build
```

#### 2. データセットのダウンロード

```bash
docker-compose run --rm transformer python download_data.py
```

#### 3. モデルの訓練

```bash
docker-compose run --rm transformer
```

#### 4. インタラクティブ翻訳

```bash
docker-compose run --rm transformer python app.py interactive
```

**注意**: GPU非対応環境の場合は、`docker-compose.yml`の`deploy`セクションを削除してください。

### ローカル環境での実行

#### 1. データセットのダウンロード

```bash
python download_data.py
```

#### 2. モデルの訓練

```bash
python app.py
```

#### 3. インタラクティブ翻訳

```bash
python app.py interactive
```

## ファイル構成

- `app.py`: メインの訓練・評価スクリプト
- `download_data.py`: Multi30kデータセットのダウンロード
- `data/`: データセットの保存ディレクトリ
- `best_model.pt`: 訓練された最良モデル
- `requirements.txt`: 必要なライブラリ一覧

## モデル構成

### ハイパーパラメータ

- **d_model**: 256 (隠れ層の次元数)
- **n_heads**: 8 (マルチヘッドアテンションのヘッド数)
- **n_encoder_layers**: 6 (エンコーダー層数)
- **n_decoder_layers**: 6 (デコーダー層数)
- **d_ff**: 1024 (フィードフォワード層の次元数)
- **dropout**: 0.1
- **batch_size**: 64
- **num_epochs**: 100 (app_enhanced.py)
- **learning_rate**: 1e-3 (warmupスケジュール付き)
- **warmup_steps**: 4000

### アーキテクチャ

1. **エンベディング層**: 語彙を256次元ベクトルに変換
2. **位置エンコーディング**: 正弦波による位置情報付加
3. **エンコーダー**: 6層のマルチヘッドアテンション + フィードフォワード
4. **デコーダー**: 6層のマスクアテンション + クロスアテンション + フィードフォワード
5. **出力層**: 語彙サイズの線形層（Label Smoothing適用）

## 評価指標

- **Loss**: クロスエントロピー損失
- **BLEU**: sacrebleuによるBLEUスコア計算
- グリーディサーチとビームサーチ（beam_size=3）の両方で評価

## 注意事項

- GPUが利用可能な場合は自動的にCUDAを使用します
- 本格的な6層モデルのため、GPU推奨（A100での学習を想定）
- `app.py`: 30エポック版（軽量学習用）
- `app_enhanced.py`: 100エポック版（本格学習用、チェックポイント機能付き）
- メイン関数のconfigで層数などのハイパーパラメータを設定可能です

## 学習済みモデルの利用

Multi30k-pretrainingブランチには、100エポック学習済みのモデルが含まれています：
- `models/best_model.pt`: ベストモデル（検証損失最小）
- `checkpoints/checkpoint_epoch_99.pt`: Epoch 99のチェックポイント
- `logs/training_log.csv`: 全エポックの詳細ログ

## ライセンス
Apache License 2.0
