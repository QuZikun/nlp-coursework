"""
Training script for RNN-based NMT.
Supports Teacher Forcing and Free Running training strategies.
"""

import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config, get_config
from data_preprocess import (
    prepare_data, load_pretrained_embeddings,
    Vocabulary, PAD_IDX, EOS_IDX
)
from models.rnn_nmt import build_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_epoch(model, dataloader, optimizer, criterion, clip,
                teacher_forcing_ratio, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0

    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        src_lens = batch['src_lens']

        optimizer.zero_grad()

        # Forward pass
        outputs, _ = model(src, src_lens, tgt, teacher_forcing_ratio)

        # Calculate loss (ignore padding and first token)
        # outputs: (batch, tgt_len, vocab_size)
        # tgt: (batch, tgt_len)
        output_dim = outputs.shape[-1]
        outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
        tgt = tgt[:, 1:].contiguous().view(-1)

        loss = criterion(outputs, tgt)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            src_lens = batch['src_lens']

            # Forward pass (no teacher forcing during evaluation)
            outputs, _ = model(src, src_lens, tgt, teacher_forcing_ratio=0.0)

            output_dim = outputs.shape[-1]
            outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(config: Config):
    """Main training function."""
    print("=" * 60)
    print(f"Training: {config.exp_name}")
    print("=" * 60)

    # Set seed
    set_seed(config.train.seed)

    # Device
    device = torch.device(config.train.device)
    print(f"Using device: {device}")

    # Prepare data
    print("\nLoading data...")
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = prepare_data(
        data_dir=config.data.data_dir,
        train_file=config.data.train_file,
        valid_file=config.data.valid_file,
        test_file=config.data.test_file,
        batch_size=config.train.batch_size,
        min_freq=config.data.min_freq,
        max_len=config.data.max_seq_len,
        vocab_max_size=config.data.max_vocab_size,
        save_vocab=True
    )

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")

    # Load pretrained embeddings if specified
    src_embeddings = None
    tgt_embeddings = None

    if config.data.src_embedding_path:
        print(f"\nLoading source embeddings from {config.data.src_embedding_path}")
        src_embeddings = load_pretrained_embeddings(
            src_vocab, config.data.src_embedding_path, config.model.embed_dim
        )

    if config.data.tgt_embedding_path:
        print(f"Loading target embeddings from {config.data.tgt_embedding_path}")
        tgt_embeddings = load_pretrained_embeddings(
            tgt_vocab, config.data.tgt_embedding_path, config.model.embed_dim
        )

    # Build model
    print("\nBuilding model...")
    model = build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        rnn_type=config.model.rnn_type,
        attention_type=config.model.attention_type,
        device=device,
        src_embeddings=src_embeddings,
        tgt_embeddings=tgt_embeddings
    )

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"RNN type: {config.model.rnn_type}")
    print(f"Attention type: {config.model.attention_type}")

    # Loss function (ignore padding)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay
    )

    # Scheduler
    scheduler = None
    if config.train.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            factor=config.train.scheduler_factor,
            patience=config.train.scheduler_patience
        )

    # Training loop
    best_valid_loss = float('inf')
    patience_counter = 0
    teacher_forcing_ratio = config.train.teacher_forcing_ratio

    print(f"\nStarting training...")
    print(f"Teacher forcing ratio: {teacher_forcing_ratio}")
    print(f"Epochs: {config.train.epochs}")

    for epoch in range(config.train.epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            config.train.clip_grad, teacher_forcing_ratio, device
        )

        # Evaluate
        valid_loss = evaluate(model, valid_loader, criterion, device)

        # Update scheduler
        if scheduler:
            scheduler.step(valid_loss)

        # Decay teacher forcing ratio
        if config.train.teacher_forcing_decay > 0:
            teacher_forcing_ratio = max(
                0.0, teacher_forcing_ratio - config.train.teacher_forcing_decay
            )

        end_time = time.time()
        epoch_mins = int((end_time - start_time) // 60)
        epoch_secs = int((end_time - start_time) % 60)

        # Check for improvement
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'config': config
            }, os.path.join(config.train.save_dir, 'best_model.pt'))

            print(f'Epoch {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | '
                  f'Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | '
                  f'TF: {teacher_forcing_ratio:.2f} | * BEST *')
        else:
            patience_counter += 1
            print(f'Epoch {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | '
                  f'Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | '
                  f'TF: {teacher_forcing_ratio:.2f}')

        # Early stopping
        if patience_counter >= config.train.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete! Best validation loss: {best_valid_loss:.3f}")
    print(f"Model saved to: {config.train.save_dir}")

    return model, src_vocab, tgt_vocab


def main():
    parser = argparse.ArgumentParser(description='Train RNN NMT model')
    parser.add_argument('--experiment', type=str, default='default',
                        choices=['default', 'attention_dot', 'attention_general',
                                'attention_concat', 'teacher_forcing', 'free_running',
                                'lstm', 'full_data'],
                        help='Experiment configuration')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    parser.add_argument('--tf_ratio', type=float, default=None,
                        help='Override teacher forcing ratio')

    args = parser.parse_args()

    # Get config
    config = get_config(args.experiment)

    # Override with command line arguments
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr
    if args.tf_ratio is not None:
        config.train.teacher_forcing_ratio = args.tf_ratio

    # Train
    train(config)


if __name__ == '__main__':
    main()

