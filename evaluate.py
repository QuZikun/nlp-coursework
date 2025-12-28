"""
Evaluation script for RNN-based NMT.
- Greedy and Beam Search decoding
- BLEU score calculation
"""

import os
import json
import argparse
from typing import List, Tuple

import torch
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from data_preprocess import (
    Vocabulary, ChineseTokenizer, EnglishTokenizer,
    SOS_IDX, EOS_IDX
)
from models.rnn_nmt import build_model, Seq2Seq


# ==================== BLEU Score ====================
def compute_bleu(references: List[List[str]], hypotheses: List[List[str]],
                 max_n: int = 4) -> Tuple[float, List[float]]:
    """
    Compute BLEU score using NLTK with smoothing.

    Args:
        references: List of reference token lists
        hypotheses: List of hypothesis token lists
        max_n: Maximum n-gram order

    Returns:
        bleu_score: BLEU score (0-1)
        precisions: Precision for each n-gram order (1-gram to 4-gram) (0-1)
    """
    # NLTK expects references as list of lists (multiple references per hypothesis)
    # We have single reference per hypothesis, so wrap each in a list
    references_wrapped = [[ref] for ref in references]

    # Use smoothing method 4 (recommended for corpus-level BLEU)
    # This handles cases where higher-order n-grams have zero matches
    smoothing = SmoothingFunction()

    # Compute corpus-level BLEU with smoothing
    bleu_score = corpus_bleu(
        references_wrapped,
        hypotheses,
        smoothing_function=smoothing.method4
    )

    # Compute individual n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        weights = [1.0/n if i < n else 0.0 for i in range(max_n)]
        prec = corpus_bleu(
            references_wrapped,
            hypotheses,
            weights=weights,
            smoothing_function=smoothing.method4
        )
        precisions.append(prec)

    return bleu_score, precisions


# ==================== Translation ====================
def translate_sentence(
    model: Seq2Seq,
    src_text: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    src_tokenizer: ChineseTokenizer,
    device: torch.device,
    method: str = 'greedy',
    beam_size: int = 5,
    max_len: int = 128
) -> Tuple[str, List[str]]:
    """
    Translate a single sentence.

    Args:
        model: Trained Seq2Seq model
        src_text: Source text (Chinese)
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        src_tokenizer: Source tokenizer
        device: Device to use
        method: 'greedy' or 'beam'
        beam_size: Beam size for beam search
        max_len: Maximum output length

    Returns:
        translation: Translated text
        tokens: List of output tokens
    """
    model.eval()

    # Tokenize and encode source
    src_tokens = src_tokenizer.tokenize(src_text)
    src_ids = src_vocab.encode(src_tokens)

    # Convert to tensor
    src_tensor = torch.LongTensor([src_ids]).to(device)
    src_lens = torch.LongTensor([len(src_ids)])

    # Translate
    output_ids, _ = model.translate(
        src_tensor, src_lens, max_len=max_len, method=method, beam_size=beam_size
    )

    # Remove EOS token if present
    if EOS_IDX in output_ids:
        output_ids = output_ids[:output_ids.index(EOS_IDX)]

    # Decode
    output_tokens = tgt_vocab.decode(output_ids)

    # Remove special tokens and UNK
    output_tokens = [t for t in output_tokens if t not in ['<PAD>', '<SOS>', '<EOS>']]

    translation = ' '.join(output_tokens)
    return translation, output_tokens


def evaluate_model(
    model: Seq2Seq,
    test_data_path: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    method: str = 'greedy',
    beam_size: int = 5,
    max_len: int = 128,
    max_samples: int = None
) -> dict:
    """Evaluate model on test set."""
    src_tokenizer = ChineseTokenizer()
    tgt_tokenizer = EnglishTokenizer()

    references = []
    hypotheses = []
    results = []

    # Load test data
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    if max_samples:
        test_data = test_data[:max_samples]

    print(f"Evaluating on {len(test_data)} samples...")
    print(f"Decoding method: {method}" + (f" (beam_size={beam_size})" if method == 'beam' else ""))

    for item in tqdm(test_data, desc="Translating"):
        src_text = item['zh']
        ref_text = item['en']

        # Get reference tokens
        ref_tokens = tgt_tokenizer.tokenize(ref_text)

        # Translate
        translation, hyp_tokens = translate_sentence(
            model, src_text, src_vocab, tgt_vocab, src_tokenizer,
            device, method=method, beam_size=beam_size, max_len=max_len
        )

        references.append(ref_tokens)
        hypotheses.append(hyp_tokens)

        results.append({
            'source': src_text,
            'reference': ref_text,
            'hypothesis': translation
        })

    # Compute BLEU
    bleu_score, precisions = compute_bleu(references, hypotheses)

    print(f"\n{'='*50}")
    print(f"BLEU Score: {bleu_score:.4f}")
    print(f"Precision 1-gram: {precisions[0]:.4f}")
    print(f"Precision 2-gram: {precisions[1]:.4f}")
    print(f"Precision 3-gram: {precisions[2]:.4f}")
    print(f"Precision 4-gram: {precisions[3]:.4f}")
    print(f"{'='*50}")

    return {
        'bleu': bleu_score,
        'precisions': precisions,
        'results': results
    }


def load_model(checkpoint_path: str, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Load vocabularies
    vocab_dir = config.data.data_dir
    src_vocab = Vocabulary.load(os.path.join(vocab_dir, 'src_vocab.pkl'))
    tgt_vocab = Vocabulary.load(os.path.join(vocab_dir, 'tgt_vocab.pkl'))

    # Build model
    model = build_model(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        n_layers=config.model.n_layers,
        dropout=config.model.dropout,
        rnn_type=config.model.rnn_type,
        attention_type=config.model.attention_type,
        device=device
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, src_vocab, tgt_vocab, config


def main():
    parser = argparse.ArgumentParser(description='Evaluate RNN NMT model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--method', type=str, default='greedy',
                        choices=['greedy', 'beam'],
                        help='Decoding method')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='Beam size for beam search')
    parser.add_argument('--max_len', type=int, default=128,
                        help='Maximum output length')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, src_vocab, tgt_vocab, config = load_model(args.checkpoint, device)

    # Test data path
    test_path = os.path.join(config.data.data_dir, config.data.test_file)

    # Evaluate
    results = evaluate_model(
        model, test_path, src_vocab, tgt_vocab, device,
        method=args.method, beam_size=args.beam_size,
        max_len=args.max_len, max_samples=args.max_samples
    )

    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.output}")

    # Print some examples
    print("\n" + "="*50)
    print("Sample Translations:")
    print("="*50)
    for i, r in enumerate(results['results'][:5]):
        print(f"\n[{i+1}] Source: {r['source'][:80]}...")
        print(f"    Reference: {r['reference'][:80]}...")
        print(f"    Hypothesis: {r['hypothesis'][:80]}...")


if __name__ == '__main__':
    main()

