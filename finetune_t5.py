"""
Fine-tuning T5 model for Chinese-to-English translation.

This script fine-tunes a pretrained T5 model (e.g., google/mt5-small)
for neural machine translation.
"""

import os
import json
import argparse
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from transformers import (
        T5ForConditionalGeneration,
        T5Tokenizer,
        MT5ForConditionalGeneration,
        MT5Tokenizer,
        AdamW,
        get_linear_schedule_with_warmup
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not installed. Run: pip install transformers")


class TranslationDataset(Dataset):
    """Dataset for translation task."""

    def __init__(self, data_path: str, tokenizer, max_len: int = 128, prefix: str = "translate Chinese to English: "):
        self.data = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prefix = prefix

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    'src': item['zh'],
                    'tgt': item['en']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        item = self.data[idx]

        # Encode source with prefix
        src_text = self.prefix + item['src']
        src_encoding = self.tokenizer(
            src_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Encode target
        tgt_encoding = self.tokenizer(
            item['tgt'],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Replace padding token id with -100 for loss computation
        labels = tgt_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': src_encoding['input_ids'].squeeze(),
            'attention_mask': src_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


def translate(model, tokenizer, text: str, device, max_len: int = 128,
              prefix: str = "translate Chinese to English: ") -> str:
    """Translate a single sentence."""
    model.eval()

    input_text = prefix + text
    input_ids = tokenizer(input_text, return_tensors='pt', max_length=max_len, truncation=True)
    input_ids = input_ids['input_ids'].to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_len,
            num_beams=4,
            early_stopping=True
        )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation


def main():
    parser = argparse.ArgumentParser(description='Fine-tune T5 for translation')
    parser.add_argument('--model_name', type=str, default='google/mt5-small',
                        help='Pretrained model name (e.g., google/mt5-small, google/mt5-base)')
    parser.add_argument('--data_dir', type=str,
                        default='AP0004_Midterm&Final_translation_dataset_zh_en',
                        help='Data directory')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-5,
                        help='Learning rate')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='checkpoints/t5_nmt',
                        help='Directory to save model')
    parser.add_argument('--warmup_steps', type=int, default=500,
                        help='Number of warmup steps')

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Error: transformers library required. Run: pip install transformers sentencepiece")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load tokenizer and model
    print(f"\nLoading model: {args.model_name}")
    if 'mt5' in args.model_name:
        tokenizer = MT5Tokenizer.from_pretrained(args.model_name)
        model = MT5ForConditionalGeneration.from_pretrained(args.model_name)
    else:
        tokenizer = T5Tokenizer.from_pretrained(args.model_name)
        model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    print("\nLoading data...")
    train_dataset = TranslationDataset(
        os.path.join(args.data_dir, 'train_10k.jsonl'),
        tokenizer, args.max_len
    )
    valid_dataset = TranslationDataset(
        os.path.join(args.data_dir, 'valid.jsonl'),
        tokenizer, args.max_len
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training...")
    best_valid_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        valid_loss = evaluate(model, valid_loader, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            # Save best model
            model.save_pretrained(args.save_dir)
            tokenizer.save_pretrained(args.save_dir)
            print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f} | * BEST *')
        else:
            print(f'Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {valid_loss:.4f}')

    print(f"\nTraining complete! Best validation loss: {best_valid_loss:.4f}")
    print(f"Model saved to: {args.save_dir}")

    # Test translation
    print("\n" + "="*50)
    print("Sample Translations:")
    print("="*50)

    test_sentences = [
        "这是一个测试句子。",
        "今天天气很好。",
        "机器学习是人工智能的一个分支。"
    ]

    for sent in test_sentences:
        translation = translate(model, tokenizer, sent, device)
        print(f"Source: {sent}")
        print(f"Translation: {translation}\n")


if __name__ == '__main__':
    main()

