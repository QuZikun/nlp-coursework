"""
Transformer-based Neural Machine Translation.

Features:
- Standard Transformer Encoder-Decoder architecture
- Position Encoding: Absolute (sinusoidal) or Relative
- Normalization: LayerNorm or RMSNorm
- Multi-head Self-Attention and Cross-Attention
"""

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import special token indices from data_preprocess to ensure consistency
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocess import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX


# ==================== Normalization Layers ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def get_norm_layer(norm_type: str, dim: int) -> nn.Module:
    """Get normalization layer by type."""
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim)
    elif norm_type == 'rmsnorm':
        return RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")


# ==================== Position Encodings ====================

class AbsolutePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (from 'Attention is All You Need')."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


def get_position_encoding(pos_type: str, d_model: int, max_len: int = 5000,
                          dropout: float = 0.1) -> nn.Module:
    """Get position encoding by type."""
    if pos_type == 'absolute' or pos_type == 'sinusoidal':
        return AbsolutePositionalEncoding(d_model, max_len, dropout)
    elif pos_type == 'learned':
        return LearnedPositionalEncoding(d_model, max_len, dropout)
    elif pos_type == 'relative':
        # Relative position encoding is handled in attention
        return nn.Dropout(dropout)
    else:
        raise ValueError(f"Unknown position encoding type: {pos_type}")


# ==================== Multi-Head Attention ====================

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional relative position encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 use_relative_pos: bool = False, max_relative_pos: int = 32):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_relative_pos = use_relative_pos
        self.max_relative_pos = max_relative_pos

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        if use_relative_pos:
            # Relative position embeddings
            vocab_size = 2 * max_relative_pos + 1
            self.relative_pos_k = nn.Embedding(vocab_size, self.d_k)
            self.relative_pos_v = nn.Embedding(vocab_size, self.d_k)

    def _get_relative_positions(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get relative position indices."""
        range_vec = torch.arange(seq_len, device=device)
        distance = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance = torch.clamp(distance, -self.max_relative_pos, self.max_relative_pos)
        return distance + self.max_relative_pos

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        """
        Args:
            query: (batch, seq_len_q, d_model)
            key: (batch, seq_len_k, d_model)
            value: (batch, seq_len_v, d_model)
            mask: (batch, 1, seq_len_q, seq_len_k) or (batch, 1, 1, seq_len_k)

        Returns:
            output: (batch, seq_len_q, d_model)
            attn_weights: (batch, n_heads, seq_len_q, seq_len_k)
        """
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_k = key.size(1)

        # Linear projections
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position bias if enabled
        if self.use_relative_pos:
            rel_pos_indices = self._get_relative_positions(seq_len_k, query.device)
            rel_k = self.relative_pos_k(rel_pos_indices)  # (seq_len_k, seq_len_k, d_k)

            # Compute relative position scores
            Q_reshaped = Q.permute(2, 0, 1, 3).reshape(seq_len_q, batch_size * self.n_heads, self.d_k)
            rel_scores = torch.bmm(Q_reshaped.transpose(0, 1), rel_k.transpose(-2, -1))
            rel_scores = rel_scores.view(batch_size, self.n_heads, seq_len_q, seq_len_k)
            scores = scores + rel_scores

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        context = torch.matmul(attn_weights, V)

        # Add relative position values if enabled
        if self.use_relative_pos:
            rel_pos_indices = self._get_relative_positions(seq_len_k, query.device)
            rel_v = self.relative_pos_v(rel_pos_indices)
            attn_flat = attn_weights.permute(2, 0, 1, 3).reshape(seq_len_q, batch_size * self.n_heads, seq_len_k)
            rel_context = torch.bmm(attn_flat, rel_v)
            rel_context = rel_context.view(seq_len_q, batch_size, self.n_heads, self.d_k)
            rel_context = rel_context.permute(1, 2, 0, 3)
            context = context + rel_context

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output, attn_weights


# ==================== Feed-Forward Network ====================

class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==================== Encoder ====================

class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_type: str = 'layernorm', use_relative_pos: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, max_len: int = 5000,
                 pos_encoding: str = 'absolute', norm_type: str = 'layernorm'):
        super().__init__()
        self.d_model = d_model
        use_relative_pos = (pos_encoding == 'relative')

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = get_position_encoding(pos_encoding, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, norm_type, use_relative_pos)
            for _ in range(n_layers)
        ])
        self.scale = math.sqrt(d_model)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(src) * self.scale
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ==================== Decoder ====================

class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 norm_type: str = 'layernorm', use_relative_pos: bool = False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout, use_relative_pos=False)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = get_norm_layer(norm_type, d_model)
        self.norm2 = get_norm_layer(norm_type, d_model)
        self.norm3 = get_norm_layer(norm_type, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> tuple:
        # Masked self-attention
        attn_out, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Cross-attention
        attn_out, cross_attn_weights = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x, self_attn_weights, cross_attn_weights


class TransformerDecoder(nn.Module):
    """Transformer Decoder."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float = 0.1, max_len: int = 5000,
                 pos_encoding: str = 'absolute', norm_type: str = 'layernorm'):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        use_relative_pos = (pos_encoding == 'relative')

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.pos_encoding = get_position_encoding(pos_encoding, d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, norm_type, use_relative_pos)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.scale = math.sqrt(d_model)

    def forward(self, tgt: torch.Tensor, enc_output: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None) -> tuple:
        x = self.embedding(tgt) * self.scale
        x = self.pos_encoding(x)

        attn_weights = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, tgt_mask, src_mask)
            attn_weights.append(cross_attn)

        output = self.fc_out(x)
        return output, attn_weights


# ==================== Full Transformer Model ====================

class Transformer(nn.Module):
    """Full Transformer for Sequence-to-Sequence tasks."""

    def __init__(self, encoder: TransformerEncoder, decoder: TransformerDecoder,
                 device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create source padding mask. (batch, 1, 1, src_len)"""
        src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def create_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        """Create target mask (padding + causal). (batch, 1, tgt_len, tgt_len)"""
        tgt_len = tgt.size(1)
        # Padding mask
        tgt_pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
        # Causal mask (lower triangular)
        tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=self.device)).bool()
        tgt_causal_mask = tgt_causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)
        # Combine masks
        tgt_mask = tgt_pad_mask & tgt_causal_mask
        return tgt_mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> tuple:
        """
        Forward pass.

        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)

        Returns:
            output: (batch, tgt_len, vocab_size)
            attn_weights: list of attention weights
        """
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)

        enc_output = self.encoder(src, src_mask)
        output, attn_weights = self.decoder(tgt, enc_output, tgt_mask, src_mask)

        return output, attn_weights

    def translate(self, src: torch.Tensor, max_len: int = 128,
                  method: str = 'greedy', beam_size: int = 5) -> tuple:
        """Translate source sequence."""
        self.eval()
        src_mask = self.create_src_mask(src)
        enc_output = self.encoder(src, src_mask)

        if method == 'greedy':
            return self._greedy_decode(enc_output, src_mask, max_len)
        else:
            return self._beam_search(enc_output, src_mask, max_len, beam_size)

    def _greedy_decode(self, enc_output: torch.Tensor, src_mask: torch.Tensor,
                       max_len: int) -> tuple:
        """Greedy decoding."""
        batch_size = enc_output.size(0)

        # Start with SOS token
        decoder_input = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for _ in range(max_len - 1):
                tgt_mask = self.create_tgt_mask(decoder_input)
                output, _ = self.decoder(decoder_input, enc_output, tgt_mask, src_mask)

                # Get last token prediction
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                decoder_input = torch.cat([decoder_input, next_token], dim=1)

                # Stop if all sequences have EOS
                if (next_token == EOS_IDX).all():
                    break

        return decoder_input[0].tolist()[1:], None  # Exclude SOS

    def _beam_search(self, enc_output: torch.Tensor, src_mask: torch.Tensor,
                     max_len: int, beam_size: int) -> tuple:
        """Beam search decoding."""
        # Expand for beam search
        enc_output = enc_output.repeat(beam_size, 1, 1)
        src_mask = src_mask.repeat(beam_size, 1, 1, 1)

        # Initialize beams
        beams = [(0.0, [SOS_IDX])]
        completed = []

        for _ in range(max_len):
            all_candidates = []

            for score, seq in beams:
                if seq[-1] == EOS_IDX:
                    completed.append((score, seq))
                    continue

                decoder_input = torch.LongTensor([seq]).to(self.device)
                tgt_mask = self.create_tgt_mask(decoder_input)
                output, _ = self.decoder(decoder_input, enc_output[:1], tgt_mask, src_mask[:1])

                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                top_probs, top_ids = log_probs.topk(beam_size)

                for prob, idx in zip(top_probs[0], top_ids[0]):
                    new_score = score + prob.item()
                    new_seq = seq + [idx.item()]
                    all_candidates.append((new_score, new_seq))

            if not all_candidates:
                break

            # Select top beams
            beams = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]

            if len(completed) >= beam_size:
                break

        completed.extend(beams)
        best = max(completed, key=lambda x: x[0] / len(x[1]))
        return best[1][1:], None


def build_transformer_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 1024,
    dropout: float = 0.1,
    max_len: int = 5000,
    pos_encoding: str = 'absolute',
    norm_type: str = 'layernorm',
    device: torch.device = None
) -> Transformer:
    """Build a Transformer model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = TransformerEncoder(
        vocab_size=src_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pos_encoding=pos_encoding,
        norm_type=norm_type
    )

    decoder = TransformerDecoder(
        vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_len=max_len,
        pos_encoding=pos_encoding,
        norm_type=norm_type
    )

    model = Transformer(encoder, decoder, device).to(device)
    return model


