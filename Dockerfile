FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# タイムゾーンの設定（インタラクティブプロンプトを避ける）
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    gcc \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# spaCyモデルのダウンロード
RUN python3 -m spacy download de_core_news_sm && \
    python3 -m spacy download en_core_web_sm

# アプリケーションファイルのコピー
COPY app.py .
COPY app_enhanced.py .
COPY download_data.py .

# ディレクトリの作成
RUN mkdir -p data models checkpoints logs

# 環境変数
ENV PYTHONUNBUFFERED=1

# デフォルトコマンド
CMD ["python3", "app_enhanced.py"]
