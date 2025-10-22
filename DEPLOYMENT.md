# GCP VM デプロイガイド

このガイドでは、GCP VM上でTransformerの学習を実行する方法を説明します。

## システム要件

- GCP VM: tts-train-a100 (us-central1-a)
- GPU: NVIDIA A100
- OS: Ubuntu 22.04
- Docker & NVIDIA Container Toolkit

## 改善点

このバージョンでは以下の機能が追加されています：

### 1. 本格的なTransformer実装
- **6層エンコーダー + 6層デコーダー**（論文 "Attention Is All You Need" 準拠）
- Label Smoothing
- Warmupスケジューラー
- 正しいAdam設定 (betas=(0.9, 0.98))

### 2. チェックポイント機能
- 毎エポック自動保存
- 学習中断時の自動再開
- 最新3つのチェックポイントを保持

### 3. BLEU スコア監視
- **初期5エポックは毎エポック計算**（学習が正しく進んでいるか確認）
- その後は5エポックごとに計算
- BLEUスコアが極端に低い場合は警告表示

### 4. 継続的な学習
- Dockerコンテナによる安定した実行環境
- 自動再起動設定
- ログファイルへの記録

## クイックスタート

### 方法1: 自動デプロイスクリプト（推奨）

```bash
# Windows (Git Bash または WSL)
cd transfomer_baseline
bash deploy_to_vm.sh
```

### 方法2: 手動デプロイ

#### 1. VMにファイルをアップロード

```bash
gcloud compute scp --recurse \
    --zone=us-central1-a \
    --project=graphic-matrix-464010-k0 \
    ./* tts-train-a100:~/transformer_baseline/
```

#### 2. VMに接続

```bash
gcloud compute ssh tts-train-a100 \
    --zone=us-central1-a \
    --project=graphic-matrix-464010-k0
```

#### 3. Dockerのセットアップ（初回のみ）

```bash
# Dockerのインストール
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit のインストール
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Docker Compose のインストール
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
    -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# ログアウトして再ログイン（Docker グループに追加するため）
exit
```

#### 4. Dockerイメージのビルド

```bash
cd ~/transformer_baseline
sudo docker-compose build
```

#### 5. 学習の開始

```bash
# バックグラウンドで実行
sudo docker-compose up -d

# ログを確認
sudo docker-compose logs -f
```

## 学習の管理

### ログの確認

```bash
# Docker コンテナのログ
sudo docker-compose logs -f

# 学習ログファイル（CSV形式）
tail -f ~/transformer_baseline/logs/training_log_*.csv

# チェックポイントの確認
ls -lh ~/transformer_baseline/checkpoints/
```

### 学習の停止

```bash
sudo docker-compose down
```

### チェックポイントから再開

学習は自動的に最新のチェックポイントから再開されます（`AUTO_RESUME=true`）。
手動で制御したい場合は、`docker-compose.yml`の環境変数を変更してください：

```yaml
environment:
  - AUTO_RESUME=false  # 手動で再開を選択
```

## BLEU スコア監視

### 正常な学習の目安

- **Epoch 1-2**: BLEU 1.0-5.0 程度
- **Epoch 3-5**: BLEU 5.0-15.0 程度
- **Epoch 10以降**: BLEU 20.0以上

⚠️ **警告**: 初期5エポックでBLEUスコアが1.0未満の場合、学習が正しく進んでいない可能性があります。

### BLEUスコアが上がらない場合の対処

1. **損失が下がっているか確認**
   ```bash
   tail ~/transformer_baseline/logs/training_log_*.csv
   ```

2. **GPU使用率の確認**（VMに接続して）
   ```bash
   nvidia-smi
   ```

3. **学習率の確認** - Warmupスケジューラーが正しく機能しているか

4. **データセットの確認** - 正しくロードされているか

## ファイル構成

```
transformer_baseline/
├── app_enhanced.py          # 強化版学習スクリプト
├── Dockerfile               # GPU対応Dockerイメージ
├── docker-compose.yml       # Docker Compose設定
├── data/                    # データセット
├── models/                  # ベストモデル
├── checkpoints/             # チェックポイント（最新3つ）
└── logs/                    # 学習ログ（CSV形式）
```

## トラブルシューティング

### GPU が認識されない

```bash
# NVIDIA ドライバーの確認
nvidia-smi

# Docker で GPU が使えるか確認
sudo docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### メモリ不足エラー

`docker-compose.yml`でバッチサイズを調整：

```yaml
environment:
  - BATCH_SIZE=32  # デフォルトは64
```

または `app_enhanced.py` の config を直接編集。

### チェックポイントが破損

```bash
# チェックポイントを削除して最初から
rm -rf ~/transformer_baseline/checkpoints/*
```

## モニタリング

### リアルタイム監視（別ターミナル）

```bash
# GPU使用率
watch -n 1 nvidia-smi

# ログの監視
tail -f ~/transformer_baseline/logs/training_log_*.csv
```

### CSV ログの分析

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('logs/training_log_XXXXXX.csv')
plt.plot(df['epoch'], df['bleu_score'])
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')
plt.savefig('bleu_progress.png')
```

## 参考

- 論文: "Attention Is All You Need" (Vaswani et al., 2017)
- データセット: Multi30k (約29,000の独英翻訳ペア)
- アーキテクチャ: 6層エンコーダー + 6層デコーダー, 8ヘッドアテンション
