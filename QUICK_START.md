# クイックスタートガイド

## 完了した改善点

✅ **6層Transformerアーキテクチャ**（論文準拠）
✅ **チェックポイント機能**（自動保存・再開）
✅ **BLEU監視強化**（初期5エポックは毎回計算）
✅ **GPU対応Dockerfile**
✅ **継続学習対応Docker Compose**

## VMでの学習開始方法

### オプション1: GitHubから直接セットアップ（推奨）

1. **VMに接続**
```bash
gcloud compute ssh tts-train-a100 --zone=us-central1-a
```

2. **セットアップスクリプトを実行**
```bash
# セットアップスクリプトをダウンロード
curl -O https://raw.githubusercontent.com/kakuteki/transfomer_baseline/main/vm_setup.sh
chmod +x vm_setup.sh

# 実行
./vm_setup.sh
```

3. **学習開始**
```bash
cd ~/transformer_baseline
sudo docker-compose up -d

# ログ確認
sudo docker-compose logs -f
```

### オプション2: ローカルの変更をGitHubにpush

ローカルで改善したコードをGitHubにpushする場合：

```bash
cd transfomer_baseline

# 変更をステージング
git add .

# コミット
git commit -m "Add enhanced training with checkpoints and BLEU monitoring"

# Push（リポジトリに書き込み権限が必要）
git push origin main
```

その後、VM上で：
```bash
gcloud compute ssh tts-train-a100 --zone=us-central1-a

cd ~/transformer_baseline
git pull

# Dockerイメージを再ビルド
sudo docker-compose build

# 学習開始
sudo docker-compose up -d
```

### オプション3: ローカルから直接ファイルをコピー（Windows）

注意: SSH設定の問題により、この方法は現在利用できません。GitHubを経由する方法をお勧めします。

## 主な改善内容

### 1. app_enhanced.py
- ✅ チェックポイントからの自動再開
- ✅ 初期5エポックは毎エポックBLEU計算
- ✅ BLEUスコアが1.0未満の場合に警告
- ✅ CSVログファイル出力
- ✅ GPU情報表示
- ✅ 100エポックまでの長期学習対応

### 2. Dockerfile
- ✅ NVIDIA CUDAベースイメージ（12.1.0）
- ✅ GPU対応
- ✅ checkpoints, logsディレクトリ作成

### 3. docker-compose.yml
- ✅ checkpoints, logsボリュームマウント
- ✅ NVIDIA runtime設定
- ✅ 自動再起動（restart: unless-stopped）
- ✅ 共有メモリ8GB設定
- ✅ AUTO_RESUME環境変数

## BLEU監視について

### 正常な学習の目安

| エポック | 期待されるBLEUスコア |
|---------|-------------------|
| 1-2     | 1.0 - 5.0         |
| 3-5     | 5.0 - 15.0        |
| 10+     | 20.0+             |

⚠️ **初期5エポックでBLEU < 1.0の場合、学習が正しく進んでいません**

## ログとチェックポイント

```
transformer_baseline/
├── logs/
│   └── training_log_YYYYMMDD_HHMMSS.csv  # 学習ログ
├── checkpoints/
│   ├── checkpoint_epoch_N.pt              # 最新3つ保持
│   └── ...
└── models/
    └── best_model.pt                      # ベストモデル
```

## トラブルシューティング

### BLEUスコアが上がらない
1. ログで損失が下がっているか確認
2. `nvidia-smi`でGPU使用率確認
3. データセットが正しくロードされているか確認

### メモリ不足
`app_enhanced.py`のバッチサイズを32に変更

### チェックポイントから再開したくない
`docker-compose.yml`で `AUTO_RESUME=false` に設定

## 次のステップ

学習が開始されたら：

1. **ログ監視**
```bash
tail -f logs/training_log_*.csv
```

2. **GPU使用率確認**
```bash
watch -n 1 nvidia-smi
```

3. **BLEUスコアの推移確認**
- 初期5エポックで1.0以上になるか
- Epoch 10で20.0以上を目指す

## 参考資料

詳細は `DEPLOYMENT.md` を参照してください。
