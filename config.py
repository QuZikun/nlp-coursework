"""
Configuration for RNN-based NMT training.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str = 'AP0004_Midterm&Final_translation_dataset_zh_en'
    train_file: str = 'train_10k.jsonl'  # Use 'train_100k.jsonl' for full training
    valid_file: str = 'valid.jsonl'
    test_file: str = 'test.jsonl'
    
    # Vocabulary
    min_freq: int = 2
    max_vocab_size: Optional[int] = None  # None for no limit
    
    # Sequence length
    max_seq_len: int = 128
    
    # Pretrained embeddings (optional)
    src_embedding_path: Optional[str] = None  # Path to Chinese embeddings
    tgt_embedding_path: Optional[str] = None  # Path to English embeddings (e.g., GloVe)


@dataclass
class RNNModelConfig:
    """RNN Model configuration."""
    # Architecture
    embed_dim: int = 256
    hidden_dim: int = 512
    n_layers: int = 2
    dropout: float = 0.3

    # RNN type: 'GRU' or 'LSTM'
    rnn_type: Literal['GRU', 'LSTM'] = 'LSTM'

    # Attention type: 'dot', 'general', 'concat'
    attention_type: Literal['dot', 'general', 'concat'] = 'general'


@dataclass
class TransformerModelConfig:
    """Transformer Model configuration."""
    # Architecture
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    # Position encoding: 'absolute', 'learned', 'relative'
    pos_encoding: Literal['absolute', 'learned', 'relative'] = 'absolute'

    # Normalization: 'layernorm', 'rmsnorm'
    norm_type: Literal['layernorm', 'rmsnorm'] = 'layernorm'


# Alias for backward compatibility
ModelConfig = RNNModelConfig


@dataclass
class TrainConfig:
    """Training configuration."""
    # Training
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    clip_grad: float = 1.0
    
    # Teacher forcing
    teacher_forcing_ratio: float = 0.5  # 0.0 = free running, 1.0 = always teacher forcing
    teacher_forcing_decay: float = 0.0  # Decay rate per epoch
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Checkpointing
    save_dir: str = 'checkpoints'
    save_best_only: bool = True
    
    # Logging
    log_interval: int = 100  # Log every N batches
    
    # Device
    device: str = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    
    # Random seed
    seed: int = 42


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    # Decoding
    decode_method: Literal['greedy', 'beam'] = 'greedy'
    beam_size: int = 5
    max_decode_len: int = 128
    
    # BLEU
    bleu_max_n: int = 4


@dataclass
class Config:
    """Complete configuration for RNN models."""
    data: DataConfig = field(default_factory=DataConfig)
    model: RNNModelConfig = field(default_factory=RNNModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment name
    exp_name: str = 'rnn_nmt'

    def __post_init__(self):
        # Create save directory
        self.train.save_dir = os.path.join(self.train.save_dir, self.exp_name)
        os.makedirs(self.train.save_dir, exist_ok=True)


@dataclass
class TransformerConfig:
    """Complete configuration for Transformer models."""
    data: DataConfig = field(default_factory=DataConfig)
    model: TransformerModelConfig = field(default_factory=TransformerModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment name
    exp_name: str = 'transformer_nmt'

    def __post_init__(self):
        # Create save directory
        self.train.save_dir = os.path.join(self.train.save_dir, self.exp_name)
        os.makedirs(self.train.save_dir, exist_ok=True)
        # Transformer usually uses smaller learning rate with warmup
        if self.train.learning_rate == 0.001:
            self.train.learning_rate = 0.0001


# Predefined configurations for experiments
def get_config(experiment: str = 'default') -> Config:
    """Get configuration for different experiments."""
    
    if experiment == 'default':
        return Config()
    
    elif experiment == 'attention_dot':
        config = Config(exp_name='rnn_attention_dot')
        config.model.attention_type = 'dot'
        return config
    
    elif experiment == 'attention_general':
        config = Config(exp_name='rnn_attention_general')
        config.model.attention_type = 'general'
        return config
    
    elif experiment == 'attention_concat':
        config = Config(exp_name='rnn_attention_concat')
        config.model.attention_type = 'concat'
        return config
    
    elif experiment == 'teacher_forcing':
        config = Config(exp_name='rnn_teacher_forcing')
        config.train.teacher_forcing_ratio = 1.0
        return config
    
    elif experiment == 'free_running':
        config = Config(exp_name='rnn_free_running')
        config.train.teacher_forcing_ratio = 0.0
        return config
    
    elif experiment == 'lstm':
        config = Config(exp_name='rnn_lstm')
        config.model.rnn_type = 'LSTM'
        return config
    
    elif experiment == 'full_data':
        config = Config(exp_name='rnn_full_data')
        config.data.train_file = 'train_100k.jsonl'
        config.train.epochs = 30
        return config

    else:
        raise ValueError(f"Unknown experiment: {experiment}")


def get_transformer_config(experiment: str = 'default') -> TransformerConfig:
    """Get Transformer configuration for different experiments."""

    if experiment == 'default':
        return TransformerConfig()

    # Position encoding experiments
    elif experiment == 'pos_absolute':
        config = TransformerConfig(exp_name='transformer_pos_absolute')
        config.model.pos_encoding = 'absolute'
        return config

    elif experiment == 'pos_learned':
        config = TransformerConfig(exp_name='transformer_pos_learned')
        config.model.pos_encoding = 'learned'
        return config

    elif experiment == 'pos_relative':
        config = TransformerConfig(exp_name='transformer_pos_relative')
        config.model.pos_encoding = 'relative'
        return config

    # Normalization experiments
    elif experiment == 'norm_layernorm':
        config = TransformerConfig(exp_name='transformer_layernorm')
        config.model.norm_type = 'layernorm'
        return config

    elif experiment == 'norm_rmsnorm':
        config = TransformerConfig(exp_name='transformer_rmsnorm')
        config.model.norm_type = 'rmsnorm'
        return config

    # Model scale experiments
    elif experiment == 'small':
        config = TransformerConfig(exp_name='transformer_small')
        config.model.d_model = 128
        config.model.n_heads = 4
        config.model.n_layers = 2
        config.model.d_ff = 512
        return config

    elif experiment == 'base':
        config = TransformerConfig(exp_name='transformer_base')
        config.model.d_model = 256
        config.model.n_heads = 8
        config.model.n_layers = 4
        config.model.d_ff = 1024
        return config

    elif experiment == 'large':
        config = TransformerConfig(exp_name='transformer_large')
        config.model.d_model = 512
        config.model.n_heads = 8
        config.model.n_layers = 6
        config.model.d_ff = 2048
        return config

    # Hyperparameter experiments
    elif experiment == 'lr_1e-3':
        config = TransformerConfig(exp_name='transformer_lr_1e-3')
        config.train.learning_rate = 0.001
        return config

    elif experiment == 'lr_5e-4':
        config = TransformerConfig(exp_name='transformer_lr_5e-4')
        config.train.learning_rate = 0.0005
        return config

    elif experiment == 'lr_1e-4':
        config = TransformerConfig(exp_name='transformer_lr_1e-4')
        config.train.learning_rate = 0.0001
        return config

    elif experiment == 'batch_32':
        config = TransformerConfig(exp_name='transformer_batch_32')
        config.train.batch_size = 32
        return config

    elif experiment == 'batch_128':
        config = TransformerConfig(exp_name='transformer_batch_128')
        config.train.batch_size = 128
        return config

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

