#Hugging Faceのdatasetsを使用したバージョン

import torch
from datasets import load_dataset
import pickle
import os

# Multi30kデータセットをダウンロード
print("Multi30kデータセットをダウンロード中...")
dataset = load_dataset("bentrevett/multi30k")

# データを保存
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 各スプリットのデータを保存
splits = {
    'train': dataset['train'],
    'validation': dataset['validation'],
    'test': dataset['test']
}

for split_name, split_data in splits.items():
    # ドイツ語と英語のテキストを抽出
    de_texts = [item['de'] for item in split_data]
    en_texts = [item['en'] for item in split_data]
    
    # 保存
    with open(f"{data_dir}/{split_name}_de.pkl", 'wb') as f:
        pickle.dump(de_texts, f)
    with open(f"{data_dir}/{split_name}_en.pkl", 'wb') as f:
        pickle.dump(en_texts, f)
    
    print(f"{split_name}: {len(de_texts)} サンプルを保存しました")

print("データのダウンロードと保存が完了しました！")
