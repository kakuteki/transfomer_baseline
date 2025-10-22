#!/bin/bash

# VM内で実行するセットアップスクリプト
# このスクリプトをVMにコピーして実行してください

set -e

echo "========================================="
echo "Transformer Baseline - VM セットアップ"
echo "========================================="

# ディレクトリ作成
mkdir -p ~/transformer_baseline
cd ~/transformer_baseline

# GitHubからクローン（最新版を取得）
echo "GitHubからコードを取得中..."
if [ -d ".git" ]; then
    git pull
else
    git clone https://github.com/kakuteki/transfomer_baseline.git .
fi

# 必要なディレクトリを作成
mkdir -p data models checkpoints logs

echo ""
echo "========================================="
echo "Docker のセットアップ"
echo "========================================="

# Dockerのインストール確認
if ! command -v docker &> /dev/null; then
    echo "Dockerをインストール中..."
    sudo apt-get update
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo systemctl enable docker
    sudo usermod -aG docker $USER
    echo "✓ Docker インストール完了"
else
    echo "✓ Docker は既にインストールされています"
fi

# NVIDIA Container Toolkit のインストール
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "NVIDIA Container Toolkitをインストール中..."
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "✓ NVIDIA Container Toolkit インストール完了"
else
    echo "✓ NVIDIA Container Toolkit は既にインストールされています"
fi

# Docker Compose のインストール
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Composeをインストール中..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo "✓ Docker Compose インストール完了"
else
    echo "✓ Docker Compose は既にインストールされています"
fi

# GPUの確認
echo ""
echo "========================================="
echo "GPU 確認"
echo "========================================="
nvidia-smi

# Dockerイメージのビルド
echo ""
echo "========================================="
echo "Docker イメージのビルド"
echo "========================================="
sudo docker-compose build

echo ""
echo "========================================="
echo "セットアップ完了！"
echo "========================================="
echo ""
echo "学習を開始するには："
echo "  sudo docker-compose up -d"
echo ""
echo "ログを確認："
echo "  sudo docker-compose logs -f"
echo ""
echo "学習を停止："
echo "  sudo docker-compose down"
echo ""
echo "注意: 'docker' グループに追加されたため、一度ログアウトして再ログインすると"
echo "      'sudo' なしで docker コマンドを使用できます。"
echo ""
