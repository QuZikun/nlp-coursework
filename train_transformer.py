"""
Training script for Transformer-based NMT.
Supports different position encodings and normalization methods.
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

from config import TransformerConfig, get_transformer_config
from data_preprocess import prepare_data, Vocabulary, PAD_IDX
from models.transformer_nmt import build_transformer_model


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


class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization."""

    def __init__(self, vocab_size: int, padding_idx: int = 0, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch * seq_len, vocab_size)
            target: (batch * seq_len,)
        """
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))  # Exclude padding and target
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = (target == self.padding_idx)
            true_dist[mask] = 0

        loss = (-true_dist * pred).sum(dim=-1)
        loss = loss.masked_fill(mask, 0).sum() / (~mask).sum()
        return loss


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0

    for batch in dataloader:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)

        optimizer.zero_grad()

        # Forward pass (teacher forcing is implicit in Transformer)
        outputs, _ = model(src, tgt[:, :-1])

        # Calculate loss
        output_dim = outputs.shape[-1]
        outputs = outputs.contiguous().view(-1, output_dim)
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

            outputs, _ = model(src, tgt[:, :-1])

            output_dim = outputs.shape[-1]
            outputs = outputs.contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)

            loss = criterion(outputs, tgt)
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def train(config: TransformerConfig):
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

    # Build model
    print("\nBuilding model...")
    model = build_transformer_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        dropout=config.model.dropout,
        max_len=config.data.max_seq_len,
        pos_encoding=config.model.pos_encoding,
        norm_type=config.model.norm_type,
        device=device
    )

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"d_model: {config.model.d_model}, n_heads: {config.model.n_heads}")
    print(f"n_layers: {config.model.n_layers}, d_ff: {config.model.d_ff}")
    print(f"Position encoding: {config.model.pos_encoding}")
    print(f"Normalization: {config.model.norm_type}")

    # Loss function with label smoothing
    criterion = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=0.1)

    # Optimizer (Adam with beta2=0.98 as in original paper)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.train.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
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

    print(f"\nStarting training...")
    print(f"Learning rate: {config.train.learning_rate}")
    print(f"Epochs: {config.train.epochs}")

    for epoch in range(config.train.epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            config.train.clip_grad, device
        )

        # Evaluate
        valid_loss = evaluate(model, valid_loader, criterion, device)

        # Update scheduler
        if scheduler:
            scheduler.step(valid_loss)

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
                  f'Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f} | * BEST *')
        else:
            patience_counter += 1
            print(f'Epoch {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s | '
                  f'Train Loss: {train_loss:.3f} | Val Loss: {valid_loss:.3f}')

        # Early stopping
        if patience_counter >= config.train.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print(f"\nTraining complete! Best validation loss: {best_valid_loss:.3f}")
    print(f"Model saved to: {config.train.save_dir}")

    return model, src_vocab, tgt_vocab


def main():
    parser = argparse.ArgumentParser(description='Train Transformer NMT model')
    parser.add_argument('--experiment', type=str, default='default',
                        choices=['default', 'pos_absolute', 'pos_learned', 'pos_relative',
                                'norm_layernorm', 'norm_rmsnorm',
                                'small', 'base', 'large',
                                'lr_1e-3', 'lr_5e-4', 'lr_1e-4',
                                'batch_32', 'batch_128'],
                        help='Experiment configuration')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')

    args = parser.parse_args()

    # Get config
    config = get_transformer_config(args.experiment)

    # Override with command line arguments
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.lr:
        config.train.learning_rate = args.lr

    # Train
    train(config)


if __name__ == '__main__':
    main()

