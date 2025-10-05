FROM python:3.10-slim

WORKDIR /app

# システムパッケージのインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Pythonパッケージのインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# spaCyモデルのダウンロード
RUN python -m spacy download de_core_news_sm && \
    python -m spacy download en_core_web_sm

# アプリケーションファイルのコピー
COPY app.py .
COPY download_data.py .

# dataディレクトリの作成
RUN mkdir -p data

# デフォルトコマンド
CMD ["python", "app.py"]
