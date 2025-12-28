"""
Transformer模型
支持: 多种位置编码(正弦/相对/可学习) + 多种归一化(LayerNorm/RMSNorm) + Greedy/Beam Search
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ==================== 位置编码 ====================

class SinusoidalPositionalEncoding(nn.Module):
    """绝对正弦位置编码 (Vaswani et al., 2017)"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """可学习位置编码"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


class RelativePositionalEncoding(nn.Module):
    """相对位置编码 (简化版，用于消融实验)"""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        # 相对位置范围: [-max_len+1, max_len-1]
        self.rel_pos_emb = nn.Embedding(2 * max_len - 1, d_model)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 简化实现：仍然加绝对位置，但使用可学习的相对位置感知嵌入
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        # 偏移到正数范围
        positions = positions + self.max_len - 1
        x = x + self.rel_pos_emb(positions).unsqueeze(0)
        return self.dropout(x)


def get_positional_encoding(pe_type: str, d_model: int, max_len: int = 512, dropout: float = 0.1):
    """获取位置编码"""
    if pe_type == "sinusoidal":
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    elif pe_type == "learned":
        return LearnedPositionalEncoding(d_model, max_len, dropout)
    elif pe_type == "relative":
        return RelativePositionalEncoding(d_model, max_len, dropout)
    else:
        raise ValueError(f"Unknown positional encoding type: {pe_type}")


# ==================== 归一化层 ====================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (更高效的归一化)"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


# ==================== 自定义Transformer层(支持RMSNorm) ====================

class CustomTransformerEncoderLayer(nn.Module):
    """自定义Transformer编码器层，支持不同归一化方法"""
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 2048,
                 dropout: float = 0.1, norm_type: str = "layernorm"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_key_padding_mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout(attn_out))
        # Feed-forward with residual
        src = self.norm2(src + self.ff(src))
        return src


class CustomTransformerDecoderLayer(nn.Module):
    """自定义Transformer解码器层，支持不同归一化方法"""
    def __init__(self, d_model: int, nhead: int, dim_ff: int = 2048,
                 dropout: float = 0.1, norm_type: str = "layernorm"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

        if norm_type == "rmsnorm":
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))
        # Cross-attention
        attn_out, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = self.norm2(tgt + self.dropout(attn_out))
        # Feed-forward
        tgt = self.norm3(tgt + self.ff(tgt))
        return tgt


# ==================== Transformer模型 ====================

class TransformerMT(nn.Module):
    """Transformer机器翻译模型 - 支持多种位置编码和归一化方法"""
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 nhead: int = 8, num_enc_layers: int = 6, num_dec_layers: int = 6,
                 dim_ff: int = 2048, dropout: float = 0.1, max_len: int = 512,
                 pad_idx: int = 0, pos_encoding_type: str = "sinusoidal",
                 norm_type: str = "layernorm"):
        super().__init__()

        self.d_model = d_model
        self.pad_idx = pad_idx
        self.tgt_vocab_size = tgt_vocab_size

        # 嵌入层
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)

        # 位置编码 (支持多种类型)
        self.pos_enc = get_positional_encoding(pos_encoding_type, d_model, max_len, dropout)

        # 自定义Transformer层 (支持RMSNorm)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, nhead, dim_ff, dropout, norm_type)
            for _ in range(num_enc_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            CustomTransformerDecoderLayer(d_model, nhead, dim_ff, dropout, norm_type)
            for _ in range(num_dec_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_src_mask(self, src: torch.Tensor) -> torch.Tensor:
        return (src == self.pad_idx)

    def make_tgt_mask(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt_len = tgt.size(1)
        causal = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt.device)
        return causal

    def encode(self, src: torch.Tensor):
        src_mask = self.make_src_mask(src)
        x = self.pos_enc(self.src_emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_mask)
        return x, src_mask

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        tgt_mask = self.make_tgt_mask(tgt)
        tgt_pad_mask = (tgt == self.pad_idx)
        x = self.pos_enc(self.tgt_emb(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder_layers:
            x = layer(x, enc_out, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask,
                     memory_key_padding_mask=src_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        enc_out, src_mask = self.encode(src)
        dec_out = self.decode(tgt, enc_out, src_mask)
        return self.fc_out(dec_out)

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int, max_len: int = 100):
        """贪心解码"""
        self.eval()
        enc_out, src_mask = self.encode(src)
        batch_size = src.size(0)
        ys = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            dec_out = self.decode(ys, enc_out, src_mask)
            logits = self.fc_out(dec_out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
        return ys

    @torch.no_grad()
    def beam_decode(self, src: torch.Tensor, sos_idx: int, eos_idx: int,
                    max_len: int = 100, beam_size: int = 5):
        """Beam Search解码"""
        self.eval()
        enc_out, src_mask = self.encode(src)
        device = src.device

        # 初始化beam
        beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]
        done = []

        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == eos_idx:
                    done.append((seq, score))
                    continue

                dec_out = self.decode(seq, enc_out, src_mask)
                logits = self.fc_out(dec_out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)
                topv, topi = log_probs.topk(beam_size)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, topi[:, i:i+1]], dim=1)
                    new_score = score + topv[0, i].item()
                    candidates.append((new_seq, new_score))

            beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            if not beams:
                break

        done.extend(beams)
        if done:
            best = max(done, key=lambda x: x[1] / x[0].size(1))
            return best[0]
        return torch.tensor([[sos_idx]], device=device)


class LabelSmoothingLoss(nn.Module):
    """标签平滑损失"""
    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.contiguous().view(-1, self.vocab_size)
        target = target.contiguous().view(-1)

        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0

        mask = (target != self.pad_idx)
        true_dist = true_dist * mask.unsqueeze(1)

        return F.kl_div(F.log_softmax(pred, dim=-1), true_dist, reduction='sum') / mask.sum()


# ==================== 模型构建函数 ====================

def build_transformer_model(src_vocab_size: int, tgt_vocab_size: int,
                            d_model: int = 512, nhead: int = 8,
                            num_enc_layers: int = 6, num_dec_layers: int = 6,
                            dim_ff: int = 2048, dropout: float = 0.1,
                            max_len: int = 512, pad_idx: int = 0,
                            pos_encoding_type: str = "sinusoidal",
                            norm_type: str = "layernorm",
                            device: str = "cuda") -> TransformerMT:
    """构建Transformer模型"""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = TransformerMT(
        src_vocab_size, tgt_vocab_size, d_model, nhead,
        num_enc_layers, num_dec_layers, dim_ff, dropout, max_len,
        pad_idx, pos_encoding_type, norm_type
    )
    return model.to(device)

