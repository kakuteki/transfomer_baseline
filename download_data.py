#Hugging Faceのdatasetsを使用したバージョン

from datasets import load_dataset
import pickle
import os

# Multi30kデータセットをダウンロード
# （data/ に同梱している pkl はこのデータと全文一致する。
#   以前ここで落としていた "wmt14" は今の datasets では ID が通らず失敗し、
#   通したとしても 450 万文対で位置エンコーディングの上限 100 トークンを超える文を含む）
print("Multi30kデータセットをダウンロード中...")
dataset = load_dataset("bentrevett/multi30k")

# データを保存
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 各スプリット(train/validation/test)のデータを保存
for split_name, split_data in dataset.items():
    # ドイツ語と英語のテキストを抽出
    de_texts = [item['de'] for item in split_data]
    en_texts = [item['en'] for item in split_data]

    # pickle形式で保存
    with open(f"{data_dir}/{split_name}_de.pkl", 'wb') as f:
        pickle.dump(de_texts, f)
    with open(f"{data_dir}/{split_name}_en.pkl", 'wb') as f:
        pickle.dump(en_texts, f)

    print(f"{split_name}: {len(de_texts)} サンプルを保存しました")

print("データのダウンロードと保存が完了しました！")
