"""
RNN-based Neural Machine Translation Model
- Encoder: 2-layer unidirectional GRU/LSTM
- Decoder: 2-layer unidirectional GRU/LSTM with Attention
- Attention: dot-product, multiplicative (Luong), additive (Bahdanau)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, Literal

import sys
sys.path.append('..')
from data_preprocess import PAD_IDX, SOS_IDX, EOS_IDX


class Encoder(nn.Module):
    """Encoder: 2-layer unidirectional RNN."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: Literal['GRU', 'LSTM'] = 'GRU',
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=PAD_IDX
            )
            embed_dim = pretrained_embeddings.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.embed_dim = embed_dim
        RNN = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = RNN(
            embed_dim, hidden_dim, n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True, bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor):
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, src_lens.cpu(), batch_first=True, enforce_sorted=True)
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return outputs, hidden


class Attention(nn.Module):
    """Attention: dot, general (multiplicative), concat (additive)."""

    def __init__(self, hidden_dim: int, attention_type: Literal['dot', 'general', 'concat'] = 'general'):
        super().__init__()
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim

        if attention_type == 'general':
            self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attention_type == 'concat':
            self.W = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        batch_size, src_len, _ = encoder_outputs.size()

        if self.attention_type == 'dot':
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'general':
            transformed = self.W(encoder_outputs)
            scores = torch.bmm(transformed, decoder_hidden.unsqueeze(2)).squeeze(2)
        elif self.attention_type == 'concat':
            decoder_expanded = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
            concat = torch.cat([decoder_expanded, encoder_outputs], dim=2)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    """Decoder with attention mechanism."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.3,
        rnn_type: Literal['GRU', 'LSTM'] = 'GRU',
        attention_type: Literal['dot', 'general', 'concat'] = 'general',
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type

        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=False, padding_idx=PAD_IDX
            )
            embed_dim = pretrained_embeddings.size(1)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.embed_dim = embed_dim
        self.attention = Attention(hidden_dim, attention_type)

        RNN = nn.GRU if rnn_type == 'GRU' else nn.LSTM
        self.rnn = RNN(embed_dim + hidden_dim, hidden_dim, n_layers,
                       dropout=dropout if n_layers > 1 else 0, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden, encoder_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        """
        Forward pass for one decoding step.

        Args:
            input_token: (batch_size,) - current input token
            hidden: Decoder hidden state
            encoder_outputs: (batch_size, src_len, hidden_dim)
            mask: (batch_size, src_len) - padding mask

        Returns:
            output: (batch_size, vocab_size)
            hidden: Updated hidden state
            attn_weights: (batch_size, src_len)
        """
        # embedded: (batch_size, 1, embed_dim)
        embedded = self.dropout(self.embedding(input_token)).unsqueeze(1)

        # Get attention hidden state
        if self.rnn_type == 'LSTM':
            attn_hidden = hidden[0][-1]  # (batch_size, hidden_dim)
        else:
            attn_hidden = hidden[-1]  # (batch_size, hidden_dim)

        # Compute attention
        context, attn_weights = self.attention(attn_hidden, encoder_outputs, mask)

        # Concatenate embedded and context for RNN input
        # rnn_input: (batch_size, 1, embed_dim + hidden_dim)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)

        # RNN forward
        rnn_output, hidden = self.rnn(rnn_input, hidden)

        # Prepare output
        rnn_output = rnn_output.squeeze(1)  # (batch_size, hidden_dim)
        embedded = embedded.squeeze(1)  # (batch_size, embed_dim)

        # Concatenate for final prediction
        output = self.fc_out(torch.cat([rnn_output, context, embedded], dim=1))

        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining Encoder and Decoder.
    Supports Teacher Forcing and Free Running training strategies.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def create_mask(self, src: torch.Tensor) -> torch.Tensor:
        """Create padding mask for source sequence."""
        return src == PAD_IDX

    def forward(self, src: torch.Tensor, src_lens: torch.Tensor,
                tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        Forward pass with optional teacher forcing.

        Args:
            src: Source sequences (batch_size, src_len)
            src_lens: Source lengths (batch_size,)
            tgt: Target sequences (batch_size, tgt_len)
            teacher_forcing_ratio: Probability of using teacher forcing (0.0 = free running)

        Returns:
            outputs: Decoder outputs (batch_size, tgt_len, vocab_size)
            attentions: Attention weights (batch_size, tgt_len, src_len)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.vocab_size

        # Store outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attentions = torch.zeros(batch_size, tgt_len, src.size(1)).to(self.device)

        # Encode
        encoder_outputs, hidden = self.encoder(src, src_lens)

        # Create source mask
        mask = self.create_mask(src)

        # First decoder input is SOS token
        input_token = tgt[:, 0]  # (batch_size,)

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, mask
            )

            outputs[:, t, :] = output
            attentions[:, t, :attn_weights.size(1)] = attn_weights

            # Teacher forcing decision
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            if use_teacher_forcing:
                input_token = tgt[:, t]
            else:
                input_token = output.argmax(dim=1)

        return outputs, attentions

    def translate(self, src: torch.Tensor, src_lens: torch.Tensor,
                  max_len: int = 128, method: str = 'greedy', beam_size: int = 5):
        """
        Translate source sequence.

        Args:
            src: Source sequence (1, src_len) - single sentence
            src_lens: Source length (1,)
            max_len: Maximum output length
            method: 'greedy' or 'beam'
            beam_size: Beam size for beam search

        Returns:
            output_tokens: List of token ids
            attention_weights: Attention weights
        """
        self.eval()
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_lens)
            mask = self.create_mask(src)

            if method == 'greedy':
                return self._greedy_decode(encoder_outputs, hidden, mask, max_len)
            else:
                return self._beam_search(encoder_outputs, hidden, mask, max_len, beam_size)

    def _greedy_decode(self, encoder_outputs, hidden, mask, max_len):
        """Greedy decoding."""
        batch_size = encoder_outputs.size(0)
        input_token = torch.LongTensor([SOS_IDX] * batch_size).to(self.device)

        output_tokens = []
        attention_weights = []

        for _ in range(max_len):
            output, hidden, attn_weights = self.decoder(
                input_token, hidden, encoder_outputs, mask
            )

            pred_token = output.argmax(dim=1)
            output_tokens.append(pred_token.item())
            attention_weights.append(attn_weights.cpu().numpy())

            if pred_token.item() == EOS_IDX:
                break

            input_token = pred_token

        return output_tokens, attention_weights

    def _beam_search(self, encoder_outputs, hidden, mask, max_len, beam_size):
        """Beam search decoding."""
        batch_size = encoder_outputs.size(0)
        assert batch_size == 1, "Beam search only supports batch_size=1"

        # Initialize beams: (score, tokens, hidden)
        if self.encoder.rnn_type == 'LSTM':
            init_hidden = (hidden[0], hidden[1])
        else:
            init_hidden = hidden

        beams = [(0.0, [SOS_IDX], init_hidden)]
        completed = []

        for _ in range(max_len):
            new_beams = []

            for score, tokens, h in beams:
                if tokens[-1] == EOS_IDX:
                    completed.append((score, tokens, None))
                    continue

                input_token = torch.LongTensor([tokens[-1]]).to(self.device)
                output, new_h, _ = self.decoder(input_token, h, encoder_outputs, mask)

                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_ids = log_probs.topk(beam_size)

                for i in range(beam_size):
                    new_score = score + topk_probs[0, i].item()
                    new_tokens = tokens + [topk_ids[0, i].item()]
                    new_beams.append((new_score, new_tokens, new_h))

            # Keep top beam_size beams
            new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_size]
            beams = new_beams

            if len(completed) >= beam_size:
                break

        # Add remaining beams to completed
        completed.extend(beams)

        # Return best sequence (excluding SOS)
        best = max(completed, key=lambda x: x[0] / len(x[1]))  # Normalize by length
        return best[1][1:], None  # Exclude SOS token


def build_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    n_layers: int = 2,
    dropout: float = 0.3,
    rnn_type: str = 'GRU',
    attention_type: str = 'general',
    device: torch.device = None,
    src_embeddings: Optional[torch.Tensor] = None,
    tgt_embeddings: Optional[torch.Tensor] = None
) -> Seq2Seq:
    """Build a Seq2Seq model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(
        vocab_size=src_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        rnn_type=rnn_type,
        pretrained_embeddings=src_embeddings
    )

    decoder = Decoder(
        vocab_size=tgt_vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        rnn_type=rnn_type,
        attention_type=attention_type,
        pretrained_embeddings=tgt_embeddings
    )

    model = Seq2Seq(encoder, decoder, device).to(device)
    return model

