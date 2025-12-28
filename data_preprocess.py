"""
Data Preprocessing for Chinese-English Neural Machine Translation
- Tokenization: Jieba (Chinese) + NLTK (English)
- Vocabulary construction with min_freq filtering
- Pretrained word embeddings support
- PyTorch Dataset and DataLoader
"""

import os
import re
import json
import pickle
import unicodedata
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Tokenization tools
import jieba
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize


# ==================== Constants ====================
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

MAX_SEQ_LEN = 128
MIN_FREQ = 2


# ==================== Data Cleaning ====================
class DataCleaner:
    """Clean and normalize text data."""

    @staticmethod
    def clean_chinese(text: str) -> str:
        """Clean Chinese text."""
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cc')
        # Normalize unicode (full-width to half-width for numbers/letters)
        text = unicodedata.normalize('NFKC', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def clean_english(text: str) -> str:
        """Clean English text."""
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cc')
        # Normalize unicode
        text = unicodedata.normalize('NFKC', text)
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def is_valid_pair(zh_text: str, en_text: str) -> bool:
        """Check if a sentence pair is valid."""
        if not zh_text or not en_text:
            return False
        if len(zh_text) < 2 or len(en_text) < 2:
            return False
        return True


# ==================== Tokenizers ====================
class ChineseTokenizer:
    """Chinese tokenizer using Jieba."""

    def __init__(self):
        jieba.initialize()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text."""
        text = DataCleaner.clean_chinese(text)
        tokens = list(jieba.cut(text))
        tokens = [t.strip() for t in tokens if t.strip()]
        return tokens


class EnglishTokenizer:
    """English tokenizer using NLTK."""

    def tokenize(self, text: str) -> List[str]:
        """Tokenize English text."""
        text = DataCleaner.clean_english(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t.strip()]
        return tokens


# ==================== Vocabulary ====================
class Vocabulary:
    """Vocabulary class for mapping tokens to indices."""

    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self._built = False

        # Initialize with special tokens
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def add_sentence(self, tokens: List[str]):
        """Add tokens from a sentence to frequency counter."""
        self.word_freq.update(tokens)

    def build(self, max_size: Optional[int] = None):
        """Build vocabulary from collected word frequencies."""
        filtered_words = [
            word for word, freq in self.word_freq.most_common()
            if freq >= self.min_freq
        ]

        if max_size is not None:
            filtered_words = filtered_words[:max_size - len(SPECIAL_TOKENS)]

        for word in filtered_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

        self._built = True
        print(f"Vocabulary built: {len(self.word2idx)} words "
              f"(filtered from {len(self.word_freq)} unique words)")

    def __len__(self) -> int:
        return len(self.word2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.word2idx.get(t, UNK_IDX) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2word.get(i, UNK_TOKEN) for i in indices]

    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_freq': self.word_freq,
                'min_freq': self.min_freq
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Vocabulary':
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls(min_freq=data['min_freq'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = data['idx2word']
        vocab.word_freq = data['word_freq']
        vocab._built = True
        return vocab



# ==================== Pretrained Embeddings ====================
def load_pretrained_embeddings(
    vocab: Vocabulary,
    embedding_path: str,
    embedding_dim: int = 300,
    binary: bool = False
) -> torch.Tensor:
    """
    Load pretrained word embeddings.

    Supports formats:
    - GloVe format: "word dim1 dim2 ... dimN"
    - Word2Vec text format: same as GloVe
    - Word2Vec binary format: set binary=True

    Args:
        vocab: Vocabulary object
        embedding_path: Path to embedding file
        embedding_dim: Dimension of embeddings
        binary: Whether the file is in binary format (for word2vec)

    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    vocab_size = len(vocab)
    # Initialize with random embeddings (Xavier uniform)
    embedding_matrix = np.random.uniform(
        -np.sqrt(3.0 / embedding_dim),
        np.sqrt(3.0 / embedding_dim),
        (vocab_size, embedding_dim)
    )

    # Set special tokens to zeros
    embedding_matrix[PAD_IDX] = np.zeros(embedding_dim)

    found_count = 0

    if binary:
        # Load word2vec binary format
        try:
            from gensim.models import KeyedVectors
            word_vectors = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
            for word, idx in vocab.word2idx.items():
                if word in word_vectors:
                    embedding_matrix[idx] = word_vectors[word]
                    found_count += 1
        except ImportError:
            print("Warning: gensim not installed, cannot load binary word2vec format")
    else:
        # Load text format (GloVe/word2vec text)
        with open(embedding_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.rstrip().split(' ')
                if len(parts) <= embedding_dim:
                    continue  # Skip header or malformed lines
                word = parts[0]
                if word in vocab.word2idx:
                    try:
                        vector = np.array([float(x) for x in parts[1:embedding_dim+1]])
                        embedding_matrix[vocab.word2idx[word]] = vector
                        found_count += 1
                    except ValueError:
                        continue

    print(f"Loaded {found_count}/{vocab_size} pretrained embeddings "
          f"({100*found_count/vocab_size:.1f}%)")

    return torch.FloatTensor(embedding_matrix)


# ==================== Translation Dataset ====================
class TranslationDataset(Dataset):
    """PyTorch Dataset for translation task."""

    def __init__(
        self,
        data_path: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        src_tokenizer: ChineseTokenizer,
        tgt_tokenizer: EnglishTokenizer,
        max_len: int = MAX_SEQ_LEN,
        src_lang: str = 'zh',
        tgt_lang: str = 'en'
    ):
        """
        Args:
            data_path: Path to JSONL file
            src_vocab: Source language vocabulary
            tgt_vocab: Target language vocabulary
            src_tokenizer: Source language tokenizer
            tgt_tokenizer: Target language tokenizer
            max_len: Maximum sequence length
            src_lang: Source language key in JSON ('zh' or 'en')
            tgt_lang: Target language key in JSON ('zh' or 'en')
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.data = []
        self._load_data(data_path)

    def _load_data(self, data_path: str):
        """Load and preprocess data from JSONL file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                src_text = item.get(self.src_lang, '')
                tgt_text = item.get(self.tgt_lang, '')

                if not DataCleaner.is_valid_pair(src_text, tgt_text):
                    continue

                # Tokenize
                src_tokens = self.src_tokenizer.tokenize(src_text)
                tgt_tokens = self.tgt_tokenizer.tokenize(tgt_text)

                # Truncate if too long (leave room for SOS/EOS)
                src_tokens = src_tokens[:self.max_len - 2]
                tgt_tokens = tgt_tokens[:self.max_len - 2]

                # Encode
                src_ids = self.src_vocab.encode(src_tokens)
                tgt_ids = self.tgt_vocab.encode(tgt_tokens)

                # Add SOS and EOS to target
                tgt_ids = [SOS_IDX] + tgt_ids + [EOS_IDX]

                self.data.append({
                    'src_ids': src_ids,
                    'tgt_ids': tgt_ids,
                    'src_text': src_text,
                    'tgt_text': tgt_text
                })

        print(f"Loaded {len(self.data)} sentence pairs from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return {
            'src': torch.LongTensor(item['src_ids']),
            'tgt': torch.LongTensor(item['tgt_ids']),
            'src_len': len(item['src_ids']),
            'tgt_len': len(item['tgt_ids'])
        }



# ==================== Collate Function ====================
def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences to the same length within a batch.

    Args:
        batch: List of dictionaries from TranslationDataset

    Returns:
        Dictionary with padded tensors and lengths
    """
    # Sort by source length (descending) for pack_padded_sequence
    batch = sorted(batch, key=lambda x: x['src_len'], reverse=True)

    src_seqs = [item['src'] for item in batch]
    tgt_seqs = [item['tgt'] for item in batch]
    src_lens = torch.LongTensor([item['src_len'] for item in batch])
    tgt_lens = torch.LongTensor([item['tgt_len'] for item in batch])

    # Pad sequences
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=PAD_IDX)

    return {
        'src': src_padded,       # (batch_size, max_src_len)
        'tgt': tgt_padded,       # (batch_size, max_tgt_len)
        'src_lens': src_lens,    # (batch_size,)
        'tgt_lens': tgt_lens     # (batch_size,)
    }


# ==================== Build Vocabulary from Data ====================
def build_vocab_from_data(
    data_path: str,
    min_freq: int = MIN_FREQ,
    max_size: Optional[int] = None
) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build source and target vocabularies from training data.

    Args:
        data_path: Path to training JSONL file
        min_freq: Minimum word frequency
        max_size: Maximum vocabulary size (None for no limit)

    Returns:
        Tuple of (src_vocab, tgt_vocab)
    """
    src_tokenizer = ChineseTokenizer()
    tgt_tokenizer = EnglishTokenizer()

    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)

    print(f"Building vocabulary from {data_path}...")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            zh_text = item.get('zh', '')
            en_text = item.get('en', '')

            if not DataCleaner.is_valid_pair(zh_text, en_text):
                continue

            # Tokenize and add to vocabulary
            src_tokens = src_tokenizer.tokenize(zh_text)
            tgt_tokens = tgt_tokenizer.tokenize(en_text)

            src_vocab.add_sentence(src_tokens)
            tgt_vocab.add_sentence(tgt_tokens)

    # Build vocabularies
    print("Building Chinese vocabulary...")
    src_vocab.build(max_size=max_size)
    print("Building English vocabulary...")
    tgt_vocab.build(max_size=max_size)

    return src_vocab, tgt_vocab


# ==================== Get DataLoader ====================
def get_dataloader(
    data_path: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    max_len: int = MAX_SEQ_LEN
) -> DataLoader:
    """
    Create a DataLoader for translation data.

    Args:
        data_path: Path to JSONL file
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        max_len: Maximum sequence length

    Returns:
        PyTorch DataLoader
    """
    src_tokenizer = ChineseTokenizer()
    tgt_tokenizer = EnglishTokenizer()

    dataset = TranslationDataset(
        data_path=data_path,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        max_len=max_len
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return dataloader


# ==================== Main Entry Point ====================
def prepare_data(
    data_dir: str,
    train_file: str = 'train_10k.jsonl',
    valid_file: str = 'valid.jsonl',
    test_file: str = 'test.jsonl',
    batch_size: int = 32,
    min_freq: int = MIN_FREQ,
    max_len: int = MAX_SEQ_LEN,
    vocab_max_size: Optional[int] = None,
    save_vocab: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Prepare all data loaders and vocabularies.

    Args:
        data_dir: Directory containing data files
        train_file: Training data filename
        valid_file: Validation data filename
        test_file: Test data filename
        batch_size: Batch size
        min_freq: Minimum word frequency for vocabulary
        max_len: Maximum sequence length
        vocab_max_size: Maximum vocabulary size
        save_vocab: Whether to save vocabularies

    Returns:
        Tuple of (train_loader, valid_loader, test_loader, src_vocab, tgt_vocab)
    """
    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)
    test_path = os.path.join(data_dir, test_file)

    # Build vocabulary from training data
    src_vocab, tgt_vocab = build_vocab_from_data(
        train_path, min_freq=min_freq, max_size=vocab_max_size
    )

    # Save vocabularies
    if save_vocab:
        src_vocab.save(os.path.join(data_dir, 'src_vocab.pkl'))
        tgt_vocab.save(os.path.join(data_dir, 'tgt_vocab.pkl'))
        print(f"Vocabularies saved to {data_dir}")

    # Create data loaders
    train_loader = get_dataloader(
        train_path, src_vocab, tgt_vocab,
        batch_size=batch_size, shuffle=True, max_len=max_len
    )
    valid_loader = get_dataloader(
        valid_path, src_vocab, tgt_vocab,
        batch_size=batch_size, shuffle=False, max_len=max_len
    )
    test_loader = get_dataloader(
        test_path, src_vocab, tgt_vocab,
        batch_size=batch_size, shuffle=False, max_len=max_len
    )

    return train_loader, valid_loader, test_loader, src_vocab, tgt_vocab


# ==================== Test Script ====================
if __name__ == '__main__':
    # Test the preprocessing pipeline
    DATA_DIR = 'AP0004_Midterm&Final_translation_dataset_zh_en'

    print("=" * 60)
    print("Testing Data Preprocessing Pipeline")
    print("=" * 60)

    # Prepare data
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = prepare_data(
        data_dir=DATA_DIR,
        train_file='train_10k.jsonl',
        batch_size=4,
        min_freq=MIN_FREQ,
        max_len=MAX_SEQ_LEN
    )

    print(f"\nSource (Chinese) vocabulary size: {len(src_vocab)}")
    print(f"Target (English) vocabulary size: {len(tgt_vocab)}")
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    print("\n" + "=" * 60)
    print("Sample batch:")
    print("=" * 60)
    batch = next(iter(train_loader))
    print(f"Source shape: {batch['src'].shape}")
    print(f"Target shape: {batch['tgt'].shape}")
    print(f"Source lengths: {batch['src_lens'][:5]}")
    print(f"Target lengths: {batch['tgt_lens'][:5]}")

    # Decode first sample
    src_tokens = src_vocab.decode(batch['src'][0].tolist())
    tgt_tokens = tgt_vocab.decode(batch['tgt'][0].tolist())
    print(f"\nFirst source (decoded): {' '.join(src_tokens[:20])}...")
    print(f"First target (decoded): {' '.join(tgt_tokens[:20])}...")
