"""
Generate report for Attention Mechanism experiments.
"""

import os
import json
from datetime import datetime


def load_results():
    """Load attention experiment results."""
    results = {}
    
    experiments = {
        'rnn_attention_dot': 'Dot-product Attention',
        'rnn_attention_general': 'General (Multiplicative) Attention',
        'rnn_attention_concat': 'Concat (Additive) Attention',
    }
    
    for exp_key, exp_name in experiments.items():
        results[exp_key] = {
            'name': exp_name,
            'greedy': None,
            'beam': None
        }
        
        # Load greedy results
        greedy_file = f'results/{exp_key}_greedy.json'
        if os.path.exists(greedy_file):
            with open(greedy_file, 'r') as f:
                data = json.load(f)
                results[exp_key]['greedy'] = {
                    'bleu': data['bleu'],
                    'precisions': data['precisions']
                }
        
        # Load beam results
        beam_file = f'results/{exp_key}_beam.json'
        if os.path.exists(beam_file):
            with open(beam_file, 'r') as f:
                data = json.load(f)
                results[exp_key]['beam'] = {
                    'bleu': data['bleu'],
                    'precisions': data['precisions']
                }
    
    return results


def generate_report(results, output_file='results/ATTENTION_REPORT.md'):
    """Generate markdown report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Attention Mechanism Comparison Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Model:** 2-layer unidirectional LSTM\n\n")
        f.write("---\n\n")
        
        # Summary Table
        f.write("## Summary\n\n")
        f.write("| Attention Type | Greedy BLEU | Beam BLEU | Improvement | Best |\n")
        f.write("|----------------|-------------|-----------|-------------|------|\n")
        
        best_bleu = 0
        best_attention = None
        
        for exp_key in ['rnn_attention_dot', 'rnn_attention_general', 'rnn_attention_concat']:
            if exp_key in results and results[exp_key]['greedy'] and results[exp_key]['beam']:
                greedy_bleu = results[exp_key]['greedy']['bleu']
                beam_bleu = results[exp_key]['beam']['bleu']
                improvement = beam_bleu - greedy_bleu
                
                is_best = ""
                if beam_bleu > best_bleu:
                    best_bleu = beam_bleu
                    best_attention = exp_key
                    is_best = "⭐"
                
                f.write(f"| {results[exp_key]['name']} | {greedy_bleu:.4f} | {beam_bleu:.4f} | "
                       f"+{improvement:.4f} | {is_best} |\n")

        f.write("\n")

        # Conclusion
        f.write("## Conclusion\n\n")
        if best_attention:
            f.write(f"**Best performing attention mechanism:** {results[best_attention]['name']}\n\n")
            f.write(f"**Best BLEU score:** {best_bleu:.4f}\n\n")
        
        # Detailed Results
        f.write("## Detailed Results\n\n")
        
        for exp_key in ['rnn_attention_dot', 'rnn_attention_general', 'rnn_attention_concat']:
            if exp_key in results:
                f.write(f"### {results[exp_key]['name']}\n\n")
                
                if results[exp_key]['greedy']:
                    prec = results[exp_key]['greedy']['precisions']
                    f.write(f"**Greedy Search:**\n")
                    f.write(f"- BLEU: {results[exp_key]['greedy']['bleu']:.4f}\n")
                    f.write(f"- 1-gram: {prec[0]:.4f}\n")
                    f.write(f"- 2-gram: {prec[1]:.4f}\n")
                    f.write(f"- 3-gram: {prec[2]:.4f}\n")
                    f.write(f"- 4-gram: {prec[3]:.4f}\n\n")

                if results[exp_key]['beam']:
                    prec = results[exp_key]['beam']['precisions']
                    f.write(f"**Beam Search (size=5):**\n")
                    f.write(f"- BLEU: {results[exp_key]['beam']['bleu']:.4f}\n")
                    f.write(f"- 1-gram: {prec[0]:.4f}\n")
                    f.write(f"- 2-gram: {prec[1]:.4f}\n")
                    f.write(f"- 3-gram: {prec[2]:.4f}\n")
                    f.write(f"- 4-gram: {prec[3]:.4f}\n\n")
    
    print(f"✓ Report generated: {output_file}")


def main():
    print("="*60)
    print("Generating Attention Mechanism Report")
    print("="*60)
    
    results = load_results()
    generate_report(results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

