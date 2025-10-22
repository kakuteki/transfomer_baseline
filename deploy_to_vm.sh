#!/bin/bash

# デプロイスクリプト for GCP VM
# 使用方法: ./deploy_to_vm.sh

set -e

VM_NAME="tts-train-a100"
ZONE="us-central1-a"
PROJECT="graphic-matrix-464010-k0"

echo "========================================="
echo "Transformer Baseline - VM デプロイスクリプト"
echo "========================================="

# VMの状態を確認
echo "VMの状態を確認中..."
VM_STATUS=$(gcloud compute instances describe $VM_NAME --zone=$ZONE --project=$PROJECT --format="get(status)")
echo "VM Status: $VM_STATUS"

if [ "$VM_STATUS" != "RUNNING" ]; then
    echo "VMを起動中..."
    gcloud compute instances start $VM_NAME --zone=$ZONE --project=$PROJECT
    echo "VMの起動を待機中..."
    sleep 30
fi

# 作業ディレクトリを作成
echo "VMに作業ディレクトリを作成中..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT --command="mkdir -p ~/transformer_baseline"

# ファイルをアップロード
echo "ファイルをVMにアップロード中..."
gcloud compute scp --recurse \
    --zone=$ZONE \
    --project=$PROJECT \
    ./* $VM_NAME:~/transformer_baseline/

# Docker と NVIDIA Container Toolkit のインストール
echo "VMでDockerとNVIDIA Container Toolkitをセットアップ中..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT << 'EOF'
    # Docker のインストール確認
    if ! command -v docker &> /dev/null; then
        echo "Dockerをインストール中..."
        sudo apt-get update
        sudo apt-get install -y docker.io
        sudo systemctl start docker
        sudo systemctl enable docker
        sudo usermod -aG docker $USER
    fi

    # NVIDIA Container Toolkit のインストール
    if ! command -v nvidia-container-toolkit &> /dev/null; then
        echo "NVIDIA Container Toolkitをインストール中..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
            sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi

    # Docker Compose のインストール
    if ! command -v docker-compose &> /dev/null; then
        echo "Docker Composeをインストール中..."
        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
            -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    fi

    echo "セットアップ完了！"
EOF

# Dockerイメージのビルド
echo "Dockerイメージをビルド中..."
gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT << 'EOF'
    cd ~/transformer_baseline
    sudo docker-compose build
EOF

# 学習開始のオプション
echo ""
echo "========================================="
echo "デプロイ完了！"
echo "========================================="
echo ""
echo "次のコマンドでVMに接続できます："
echo "  gcloud compute ssh $VM_NAME --zone=$ZONE --project=$PROJECT"
echo ""
echo "学習を開始するには、VM内で以下を実行："
echo "  cd ~/transformer_baseline"
echo "  sudo docker-compose up -d"
echo ""
echo "ログを確認："
echo "  sudo docker-compose logs -f"
echo ""
echo "学習を停止："
echo "  sudo docker-compose down"
echo ""
