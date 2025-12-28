"""
Evaluation script for Transformer-based NMT.
"""

import os
import json
import argparse
import torch
from typing import List, Tuple

from config import TransformerConfig
from data_preprocess import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
from models.transformer_nmt import build_transformer_model
from evaluate import compute_bleu


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load vocabularies
    src_vocab = Vocabulary.load(os.path.join(config.data.data_dir, 'src_vocab.pkl'))
    tgt_vocab = Vocabulary.load(os.path.join(config.data.data_dir, 'tgt_vocab.pkl'))

    # Build model
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

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, src_vocab, tgt_vocab, config





def translate(model, src_text: str, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
              device: torch.device, max_len: int = 128,
              decode_method: str = 'greedy', beam_size: int = 5) -> str:
    """Translate a single sentence using model's built-in translate method."""
    import jieba

    # Tokenize source
    tokens = list(jieba.cut(src_text))
    src_indices = [src_vocab.word2idx.get(t, UNK_IDX) for t in tokens]
    src_indices = [SOS_IDX] + src_indices + [EOS_IDX]
    src = torch.tensor([src_indices]).to(device)

    # Use model's translate method
    output_indices, _ = model.translate(src, max_len, decode_method, beam_size)

    # Handle tensor output
    if isinstance(output_indices, torch.Tensor):
        output_indices = output_indices.squeeze().tolist()

    # Convert to words
    output_words = []
    for idx in output_indices:
        if idx == EOS_IDX:
            break
        word = tgt_vocab.idx2word.get(idx, '<unk>')
        if word not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
            output_words.append(word)

    return ' '.join(output_words)


def evaluate_on_test(model, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                     test_file: str, device: torch.device,
                     max_len: int = 128, decode_method: str = 'greedy',
                     beam_size: int = 5) -> Tuple[float, List[Tuple[str, str, str]]]:
    """Evaluate model on test set."""
    import jieba

    hypotheses = []
    references = []
    examples = []

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            src_text = item['zh']
            ref_text = item['en']

            # Translate
            hyp_text = translate(model, src_text, src_vocab, tgt_vocab, device,
                                max_len, decode_method, beam_size)

            hypotheses.append(hyp_text.split())
            references.append(ref_text.split())  # Single reference per example
            examples.append((src_text, ref_text, hyp_text))

    # Calculate BLEU
    bleu_score, _ = compute_bleu(references, hypotheses)

    return bleu_score, examples


def main():
    parser = argparse.ArgumentParser(description='Evaluate Transformer NMT model')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/transformer_nmt/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--decode', type=str, default='greedy', choices=['greedy', 'beam'],
                        help='Decoding method')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for beam search')
    parser.add_argument('--num_examples', type=int, default=10,
                        help='Number of examples to show')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, src_vocab, tgt_vocab, config = load_model(args.checkpoint, device)
    print(f"Model loaded successfully")
    print(f"Position encoding: {config.model.pos_encoding}")
    print(f"Normalization: {config.model.norm_type}")

    # Evaluate
    test_file = os.path.join(config.data.data_dir, config.data.test_file)
    print(f"\nEvaluating on {test_file}...")
    print(f"Decoding method: {args.decode}")

    bleu_score, examples = evaluate_on_test(
        model, src_vocab, tgt_vocab, test_file, device,
        config.data.max_seq_len, args.decode, args.beam_size
    )

    print(f"\nBLEU Score: {bleu_score:.4f}")

    # Show examples
    print("\n" + "="*60)
    print("Sample Translations:")
    print("="*60)

    for i, (src, ref, hyp) in enumerate(examples[:args.num_examples]):
        print(f"\n[{i+1}]")
        print(f"Source: {src}")
        print(f"Reference: {ref}")
        print(f"Hypothesis: {hyp}")


if __name__ == '__main__':
    main()

