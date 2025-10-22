import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
from tqdm import tqdm
import random
import sacrebleu
import spacy
from torch.nn.utils.rnn import pad_sequence

# ================== Transformerモデル部分 ==================

class MultiHeadAttention(nn.Module):
    """マルチヘッドアテンション層"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        output = self.W_o(context)
        return output

class EncoderLayer(nn.Module):
    """エンコーダー層（1層分）"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

class DecoderLayer(nn.Module):
    """デコーダー層（1層分）"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        self_attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attn(x, enc_output, enc_output, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TranslationTransformer(nn.Module):
    """翻訳用Transformer（1層版）"""
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 1,
        n_decoder_layers: int = 1,
        d_ff: int = 1024,
        max_seq_len: int = 100,
        dropout: float = 0.1,
        pad_idx: int = 0
    ):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx

        # エンベディング層
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # エンコーダー層
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # デコーダー層
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # 出力層
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 重み初期化
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """パディングマスクの生成"""
        return (seq != self.pad_idx).unsqueeze(1).unsqueeze(2)

    def create_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        """未来を見ないマスクの生成"""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        return mask == 0

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """エンコーダー部分"""
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, src_mask)

        return x

    def decode(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """デコーダー部分"""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, enc_output, tgt_mask, src_mask)

        return x

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, src_seq_len]
            tgt: [batch_size, tgt_seq_len]
        Returns:
            output: [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # パディングマスクの生成
        src_mask = self.create_padding_mask(src)

        # ターゲットマスクの生成（パディング + 未来を見ない）
        tgt_seq_len = tgt.size(1)
        tgt_pad_mask = self.create_padding_mask(tgt)
        tgt_sub_mask = self.create_subsequent_mask(tgt_seq_len, tgt.device)
        tgt_mask = tgt_pad_mask & tgt_sub_mask

        # エンコード
        enc_output = self.encode(src, src_mask)

        # デコード
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)

        # 出力投影
        output = self.output_projection(dec_output)

        return output

    @torch.no_grad()
    def translate(
        self,
        src: torch.Tensor,
        max_len: int = 50,
        sos_idx: int = 2,
        eos_idx: int = 3,
        beam_size: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """推論時の翻訳（グリーディまたはビームサーチ）"""
        self.eval()
        device = src.device
        batch_size = src.size(0)

        # エンコード
        src_mask = self.create_padding_mask(src)
        enc_output = self.encode(src, src_mask)

        if beam_size == 1:
            # グリーディサーチ（温度パラメータ付き）
            tgt = torch.full((batch_size, 1), sos_idx, device=device)

            for _ in range(max_len - 1):
                tgt_mask = self.create_subsequent_mask(tgt.size(1), device)
                dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)
                logits = self.output_projection(dec_output[:, -1:, :]) / temperature
                next_token = logits.argmax(dim=-1)
                tgt = torch.cat([tgt, next_token], dim=1)

                if (next_token == eos_idx).all():
                    break

            return tgt
        else:
            # ビームサーチ（簡易版）
            return self._beam_search(enc_output, src_mask, beam_size, max_len, sos_idx, eos_idx, temperature)

    def _beam_search(self, enc_output, src_mask, beam_size, max_len, sos_idx, eos_idx, temperature=1.0):
        """ビームサーチ実装"""
        device = enc_output.device
        batch_size = enc_output.size(0)

        # バッチサイズ1のみサポート（簡易版）
        assert batch_size == 1, "Beam search only supports batch_size=1"

        # 初期化
        beams = [(torch.tensor([[sos_idx]], device=device), 0)]
        finished = []

        for _ in range(max_len - 1):
            candidates = []

            for seq, score in beams:
                if seq[0, -1].item() == eos_idx:
                    finished.append((seq, score))
                    continue

                tgt_mask = self.create_subsequent_mask(seq.size(1), device)
                dec_output = self.decode(seq, enc_output, tgt_mask, src_mask)
                logits = self.output_projection(dec_output[:, -1:, :]) / temperature
                log_probs = F.log_softmax(logits, dim=-1)

                topk_log_probs, topk_indices = log_probs[0, 0].topk(beam_size)

                for k in range(beam_size):
                    new_seq = torch.cat([seq, topk_indices[k].unsqueeze(0).unsqueeze(0)], dim=1)
                    new_score = score + topk_log_probs[k].item()
                    candidates.append((new_seq, new_score))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            if not beams:
                break

        finished.extend(beams)
        finished = sorted(finished, key=lambda x: x[1] / x[0].size(1), reverse=True)

        return finished[0][0] if finished else torch.tensor([[sos_idx, eos_idx]], device=device)


# ================== トークナイザー ==================

class Vocabulary:
    """語彙管理クラス"""
    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold
        self.token2idx = {}
        self.idx2token = {}

        # 特殊トークン
        self.pad_idx = 0
        self.unk_idx = 1
        self.sos_idx = 2
        self.eos_idx = 3

        self.token2idx['<pad>'] = self.pad_idx
        self.token2idx['<unk>'] = self.unk_idx
        self.token2idx['<sos>'] = self.sos_idx
        self.token2idx['<eos>'] = self.eos_idx

        for token, idx in self.token2idx.items():
            self.idx2token[idx] = token

    def build_vocab(self, sentences, tokenizer):
        """語彙を構築"""
        counter = Counter()
        for sentence in sentences:
            tokens = tokenizer(sentence)
            counter.update(tokens)

        idx = len(self.token2idx)
        for token, freq in counter.items():
            if freq >= self.freq_threshold:
                if token not in self.token2idx:
                    self.token2idx[token] = idx
                    self.idx2token[idx] = token
                    idx += 1

    def encode(self, sentence, tokenizer):
        """文をトークンIDに変換"""
        tokens = tokenizer(sentence)
        return [self.token2idx.get(token, self.unk_idx) for token in tokens]

    def decode(self, indices):
        """トークンIDを文に変換"""
        tokens = []
        for idx in indices:
            if idx == self.eos_idx:
                break
            if idx not in [self.pad_idx, self.sos_idx]:
                tokens.append(self.idx2token.get(idx, '<unk>'))
        return tokens

    def __len__(self):
        return len(self.token2idx)


# ================== データセット ==================

class TranslationDataset(Dataset):
    """翻訳データセット"""
    def __init__(self, src_texts, tgt_texts, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        # エンコード
        src_indices = self.src_vocab.encode(src_text, self.src_tokenizer)
        tgt_indices = [self.tgt_vocab.sos_idx] + \
                     self.tgt_vocab.encode(tgt_text, self.tgt_tokenizer) + \
                     [self.tgt_vocab.eos_idx]

        return {
            'src': torch.tensor(src_indices),
            'tgt': torch.tensor(tgt_indices),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


def collate_fn(batch):
    """バッチ処理用のcollate関数"""
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    # パディング
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)

    return {
        'src': src_batch,
        'tgt': tgt_batch,
        'src_texts': [item['src_text'] for item in batch],
        'tgt_texts': [item['tgt_text'] for item in batch]
    }


# ================== データ準備 ==================

def prepare_data_wmt14():
    """WMT14 DE-ENデータセットの準備（Hugging Face Datasets使用）"""
    print("=" * 60)
    print("WMT14 DE-EN Dataset Preparation")
    print("Using Hugging Face Datasets")
    print("=" * 60)

    # トークナイザー
    print("\nLoading tokenizers...")
    nlp_de = spacy.load('de_core_news_sm')
    nlp_en = spacy.load('en_core_web_sm')
    spacy_de = lambda text: [token.text for token in nlp_de.tokenizer(text)]
    spacy_en = lambda text: [token.text for token in nlp_en.tokenizer(text)]

    # データディレクトリ
    data_dir = "data/wmt14"
    os.makedirs(data_dir, exist_ok=True)

    # Pickleファイルから読み込み（既に準備されている場合）
    vocab_file = f"{data_dir}/vocab.pkl"
    data_file = f"{data_dir}/processed_data.pkl"

    if os.path.exists(vocab_file) and os.path.exists(data_file):
        print("\nLoading preprocessed data...")
        with open(vocab_file, 'rb') as f:
            src_vocab, tgt_vocab = pickle.load(f)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            train_src, train_tgt = data['train']
            val_src, val_tgt = data['val']
            test_src, test_tgt = data['test']
        print("✓ Preprocessed data loaded")
    else:
        print("\nDownloading WMT14 dataset from Hugging Face...")
        print("This may take several minutes (downloading ~1.6GB)...")

        from datasets import load_dataset

        # WMT14 DE-ENデータセットをダウンロード
        print("\nLoading dataset...")
        dataset = load_dataset("wmt14", "de-en", cache_dir=data_dir)

        print(f"✓ Dataset loaded")
        print(f"  Training samples: {len(dataset['train'])}")
        print(f"  Validation samples: {len(dataset['validation'])}")
        print(f"  Test samples: {len(dataset['test'])}")

        # データサイズ制限（メモリ節約のため）
        max_train_size = int(os.environ.get('MAX_TRAIN_SIZE', 500000))  # デフォルト50万ペア

        # データを抽出
        print("\nExtracting training data...")
        train_data = dataset['train']

        # ランダムサンプリング
        if len(train_data) > max_train_size:
            print(f"Sampling {max_train_size} from {len(train_data)} training samples...")
            indices = random.sample(range(len(train_data)), max_train_size)
            train_data = train_data.select(indices)

        train_src = [item['translation']['de'] for item in train_data]
        train_tgt = [item['translation']['en'] for item in train_data]

        print("Extracting validation data...")
        val_data = dataset['validation']
        val_src = [item['translation']['de'] for item in val_data]
        val_tgt = [item['translation']['en'] for item in val_data]

        print("Extracting test data...")
        test_data = dataset['test']
        test_src = [item['translation']['de'] for item in test_data]
        test_tgt = [item['translation']['en'] for item in test_data]

        print(f"\nFinal dataset sizes:")
        print(f"  Training: {len(train_src)}")
        print(f"  Validation: {len(val_src)}")
        print(f"  Test: {len(test_src)}")

        # 語彙を構築
        print("\nBuilding vocabularies...")
        print("This may take several minutes for large datasets...")

        src_vocab = Vocabulary(freq_threshold=5)  # WMT14は大規模なので閾値を上げる
        tgt_vocab = Vocabulary(freq_threshold=5)

        print("Building source vocabulary (DE)...")
        src_vocab.build_vocab(train_src, spacy_de)

        print("Building target vocabulary (EN)...")
        tgt_vocab.build_vocab(train_tgt, spacy_en)

        print(f"✓ Source vocabulary size (DE): {len(src_vocab)}")
        print(f"✓ Target vocabulary size (EN): {len(tgt_vocab)}")

        # データを保存
        print("\nSaving preprocessed data...")
        with open(vocab_file, 'wb') as f:
            pickle.dump((src_vocab, tgt_vocab), f)
        with open(data_file, 'wb') as f:
            pickle.dump({
                'train': (train_src, train_tgt),
                'val': (val_src, val_tgt),
                'test': (test_src, test_tgt)
            }, f)
        print("✓ Data saved to disk")

    # データセット作成
    print("\nCreating PyTorch datasets...")
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, spacy_de, spacy_en)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, spacy_de, spacy_en)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, spacy_de, spacy_en)

    print("\n" + "=" * 60)
    print(f"Dataset: WMT14 DE→EN")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Source vocab (DE): {len(src_vocab)}")
    print(f"Target vocab (EN): {len(tgt_vocab)}")
    print("=" * 60)

    return train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab, spacy_de, spacy_en


# ================== 訓練関数 ==================

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    """1エポックの訓練"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        # Teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input)

        # Loss計算
        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)
        loss = criterion(output, tgt_output)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # バッチごとにステップ

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """評価"""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output = model(src, tgt_input)

            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)
            loss = criterion(output, tgt_output)

            total_loss += loss.item()

    return total_loss / len(dataloader)


def calculate_bleu(model, dataloader, tgt_vocab, device, beam_size=1):
    """BLEUスコアの計算"""
    model.eval()

    hypotheses = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating BLEU"):
            src = batch['src'].to(device)

            # 翻訳生成
            translations = model.translate(
                src,
                max_len=50,
                sos_idx=tgt_vocab.sos_idx,
                eos_idx=tgt_vocab.eos_idx,
                beam_size=beam_size
            )

            # デコード
            for i in range(len(translations)):
                # 生成文
                hyp_indices = translations[i].cpu().tolist()
                hyp_tokens = tgt_vocab.decode(hyp_indices)
                hypotheses.append(' '.join(hyp_tokens))

                # 参照文
                references.append(batch['tgt_texts'][i])

    # BLEUスコア計算
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return bleu.score


# ================== メイン関数 ==================

def main():
    import csv
    import time
    from datetime import datetime

    # 設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # GPU情報を表示
    if torch.cuda.is_available():
        print(f"GPU名: {torch.cuda.get_device_name(0)}")
        print(f"利用可能メモリ: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # ハイパーパラメータ（WMT14専用）
    config = {
        'd_model': 512,  # より大きなモデル
        'n_heads': 8,
        'n_encoder_layers': 6,  # エンコーダー層数
        'n_decoder_layers': 6,  # デコーダー層数
        'd_ff': 2048,  # より大きなFFN
        'dropout': 0.1,
        'batch_size': 32,  # より大規模なデータのため小さめ
        'num_epochs': 50,  # WMT14用
        'learning_rate': 1e-3,
        'warmup_steps': 8000,  # より長いwarmup
        'checkpoint_dir': 'checkpoints_wmt14',
        'log_dir': 'logs_wmt14',
        'model_dir': 'models_wmt14',
        'resume_from': None  # チェックポイントパスを指定すると再開
    }

    # ディレクトリ作成
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['model_dir'], exist_ok=True)

    # チェックポイントの自動検出と再開
    auto_resume = os.environ.get('AUTO_RESUME', 'true').lower() == 'true'
    checkpoint_files = sorted([f for f in os.listdir(config['checkpoint_dir']) if f.startswith('checkpoint_epoch_')])
    if checkpoint_files and config['resume_from'] is None:
        latest_checkpoint = os.path.join(config['checkpoint_dir'], checkpoint_files[-1])
        print(f"\n最新のチェックポイントを検出: {latest_checkpoint}")
        if auto_resume:
            config['resume_from'] = latest_checkpoint
            print("自動的にチェックポイントから再開します")
        else:
            response = input("このチェックポイントから再開しますか? (y/n): ").strip().lower()
            if response == 'y':
                config['resume_from'] = latest_checkpoint

    print("\n設定:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # データ準備
    try:
        # WMT14データセットを準備
        train_dataset, val_dataset, test_dataset, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer = prepare_data_wmt14()
    except Exception as e:
        print(f"\nエラー: {e}")
        print("\nspaCyのモデルがインストールされていない可能性があります。")
        print("以下のコマンドを実行してください:")
        print("  python -m spacy download de_core_news_sm")
        print("  python -m spacy download en_core_web_sm")
        import traceback
        traceback.print_exc()
        return

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # 推論時は1文ずつ
        shuffle=False,
        collate_fn=collate_fn
    )

    # モデル作成
    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=src_vocab.pad_idx
    ).to(device)

    print(f"\nモデルパラメータ数: {sum(p.numel() for p in model.parameters()):,}")

    # 最適化
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # 学習率スケジューラー（Transformer論文のwarmup - ステップベース）
    class TransformerScheduler:
        def __init__(self, optimizer, d_model, warmup_steps):
            self.optimizer = optimizer
            self.d_model = d_model
            self.warmup_steps = warmup_steps
            self.step_num = 0

        def step(self):
            self.step_num += 1
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        def _get_lr(self):
            step = max(1, self.step_num)
            return (self.d_model ** -0.5) * min(step ** -0.5, step * self.warmup_steps ** -1.5)

        def get_last_lr(self):
            return [self._get_lr()]

    scheduler = TransformerScheduler(optimizer, config['d_model'], config['warmup_steps'])

    # 損失関数（Label smoothing追加）
    class LabelSmoothingLoss(nn.Module):
        def __init__(self, smoothing=0.1, pad_idx=0):
            super().__init__()
            self.smoothing = smoothing
            self.pad_idx = pad_idx
            self.confidence = 1.0 - smoothing

        def forward(self, pred, target):
            # pred: [batch_size * seq_len, vocab_size]
            # target: [batch_size * seq_len]
            vocab_size = pred.size(-1)

            # マスク作成
            mask = target != self.pad_idx

            # One-hot encoding
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (vocab_size - 2))  # <pad>と正解を除く
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0

            # パディング部分をマスク
            true_dist = true_dist * mask.unsqueeze(1)

            # KL divergence
            loss = F.kl_div(
                F.log_softmax(pred, dim=-1),
                true_dist,
                reduction='none'
            ).sum(dim=-1)

            return (loss * mask).sum() / mask.sum()

    criterion = LabelSmoothingLoss(smoothing=0.1, pad_idx=src_vocab.pad_idx)

    # チェックポイントから再開
    start_epoch = 0
    best_val_loss = float('inf')
    bleu_history = []

    if config['resume_from'] and os.path.exists(config['resume_from']):
        print(f"\nチェックポイントから再開: {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.step_num = checkpoint['scheduler_step']
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        bleu_history = checkpoint.get('bleu_history', [])
        print(f"エポック {start_epoch} から再開します")

    # ログファイルの作成
    log_file = os.path.join(config['log_dir'], f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'learning_rate', 'bleu_score', 'time'])

    # 訓練
    print("\n訓練開始...")

    for epoch in range(start_epoch, config['num_epochs']):
        epoch_start_time = time.time()
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # 訓練
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        # 検証
        val_loss = evaluate(model, val_loader, criterion, device)

        print(f"  訓練損失: {train_loss:.4f}")
        print(f"  検証損失: {val_loss:.4f}")
        print(f"  学習率: {scheduler.get_last_lr()[0]:.6f}")

        # BLEU スコアの計算（初期5エポックは毎回、その後は5エポックごと）
        bleu_score = None
        if epoch < 5 or (epoch + 1) % 5 == 0:
            print("  BLEUスコアを計算中...")
            bleu_score = calculate_bleu(model, test_loader, tgt_vocab, device, beam_size=1)
            print(f"  BLEUスコア: {bleu_score:.2f}")
            bleu_history.append({'epoch': epoch + 1, 'bleu': bleu_score})

            # BLEU スコアが極端に低い場合は警告
            if epoch < 5 and bleu_score < 1.0:
                print("  ⚠️  警告: BLEUスコアが非常に低いです。学習が正しく進んでいない可能性があります。")

        # エポック時間
        epoch_time = time.time() - epoch_start_time
        print(f"  エポック時間: {epoch_time:.2f}秒")

        # ログに記録
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_loss,
                val_loss,
                scheduler.get_last_lr()[0],
                bleu_score if bleu_score else '',
                epoch_time
            ])

        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_step': scheduler.step_num,
                'config': config,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'bleu_history': bleu_history
            }, f"{config['model_dir']}/best_model_wmt14.pt")
            print("  ✓ ベストモデルを保存しました")

        # チェックポイント保存（毎エポック）
        checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch + 1}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_step': scheduler.step_num,
            'config': config,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'epoch': epoch,
            'bleu_history': bleu_history
        }, checkpoint_path)

        # 古いチェックポイントを削除（最新3つのみ保持）
        checkpoint_files = sorted([f for f in os.listdir(config['checkpoint_dir']) if f.startswith('checkpoint_epoch_')])
        if len(checkpoint_files) > 3:
            for old_checkpoint in checkpoint_files[:-3]:
                os.remove(os.path.join(config['checkpoint_dir'], old_checkpoint))

    # 最終評価
    print("\n最終評価...")

    # ベストモデルをロード
    checkpoint = torch.load(f"{config['model_dir']}/best_model_wmt14.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # テストセットでBLEU評価
    print("グリーディサーチ:")
    bleu_greedy = calculate_bleu(model, test_loader, tgt_vocab, device, beam_size=1)
    print(f"  BLEUスコア: {bleu_greedy:.2f}")

    print("ビームサーチ (beam_size=3):")
    bleu_beam = calculate_bleu(model, test_loader, tgt_vocab, device, beam_size=3)
    print(f"  BLEUスコア: {bleu_beam:.2f}")

    # 翻訳例を表示
    print("\n翻訳例:")
    model.eval()

    # テストセットから5つサンプル
    sample_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn
    )

    for i, batch in enumerate(sample_loader):
        if i >= 5:
            break

        src = batch['src'].to(device)
        src_text = batch['src_texts'][0]
        tgt_text = batch['tgt_texts'][0]

        # 翻訳
        with torch.no_grad():
            translation = model.translate(
                src,
                max_len=50,
                sos_idx=tgt_vocab.sos_idx,
                eos_idx=tgt_vocab.eos_idx,
                beam_size=3
            )

        # デコード
        pred_indices = translation[0].cpu().tolist()
        pred_tokens = tgt_vocab.decode(pred_indices)
        pred_text = ' '.join(pred_tokens)

        print(f"\n例 {i+1}:")
        print(f"  ソース（独語）: {src_text}")
        print(f"  予測（英語）  : {pred_text}")
        print(f"  正解（英語）  : {tgt_text}")


# ================== インタラクティブ翻訳 ==================

def interactive_translation(model_path='best_model.pt'):
    """インタラクティブな翻訳デモ"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # モデルとデータをロード
    print("モデルをロード中...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    config = checkpoint['config']

    # モデル再構築
    model = TranslationTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_encoder_layers=config['n_encoder_layers'],
        n_decoder_layers=config['n_decoder_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        pad_idx=src_vocab.pad_idx
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # トークナイザー
    try:
        nlp_de = spacy.load('de_core_news_sm')
        spacy_de = lambda text: [token.text for token in nlp_de.tokenizer(text)]
    except:
        print("ドイツ語トークナイザーが利用できません。簡易トークナイザーを使用します。")
        spacy_de = lambda x: x.lower().split()

    print("\n独→英翻訳システム（1層Transformer）")
    print("ドイツ語を入力してください（'quit'で終了）：")

    while True:
        german_text = input("\n独語> ").strip()

        if german_text.lower() == 'quit':
            break

        if not german_text:
            continue

        # 前処理
        src_indices = [src_vocab.sos_idx] + src_vocab.encode(german_text, spacy_de) + [src_vocab.eos_idx]
        src_tensor = torch.tensor([src_indices]).to(device)

        # 翻訳
        with torch.no_grad():
            # グリーディ
            translation_greedy = model.translate(
                src_tensor,
                max_len=50,
                sos_idx=tgt_vocab.sos_idx,
                eos_idx=tgt_vocab.eos_idx,
                beam_size=1
            )

            # ビームサーチ
            translation_beam = model.translate(
                src_tensor,
                max_len=50,
                sos_idx=tgt_vocab.sos_idx,
                eos_idx=tgt_vocab.eos_idx,
                beam_size=3
            )

        # デコード
        pred_greedy = ' '.join(tgt_vocab.decode(translation_greedy[0].cpu().tolist()))
        pred_beam = ' '.join(tgt_vocab.decode(translation_beam[0].cpu().tolist()))

        print(f"英語（グリーディ）: {pred_greedy}")
        print(f"英語（ビーム検索）: {pred_beam}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # インタラクティブモード
        interactive_translation()
    else:
        # 訓練モード
        main()
