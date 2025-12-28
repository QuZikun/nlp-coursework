"""
Generate comprehensive comparison report for RNN experiments.
"""

import os
import json
from datetime import datetime
from pathlib import Path


def load_results(results_dir='results'):
    """Load all evaluation results."""
    results = {}
    
    experiments = {
        'rnn_attention_dot': 'Dot-product Attention',
        'rnn_attention_general': 'General Attention',
        'rnn_attention_concat': 'Concat Attention',
        'rnn_teacher_forcing': 'Teacher Forcing (TF=1.0)',
        'rnn_free_running': 'Free Running (TF=0.0)',
        'rnn_nmt': 'Scheduled Sampling (TF=0.5)',
        'rnn_lstm': 'LSTM',
    }
    
    for exp_key, exp_name in experiments.items():
        results[exp_key] = {
            'name': exp_name,
            'greedy': None,
            'beam': None
        }
        
        # Load greedy results
        greedy_file = os.path.join(results_dir, f'{exp_key}_greedy.json')
        if os.path.exists(greedy_file):
            with open(greedy_file, 'r') as f:
                data = json.load(f)
                results[exp_key]['greedy'] = {
                    'bleu': data['bleu'],
                    'precisions': data['precisions']
                }
        
        # Load beam results
        beam_file = os.path.join(results_dir, f'{exp_key}_beam.json')
        if os.path.exists(beam_file):
            with open(beam_file, 'r') as f:
                data = json.load(f)
                results[exp_key]['beam'] = {
                    'bleu': data['bleu'],
                    'precisions': data['precisions']
                }
    
    return results


def generate_markdown_report(results, output_file='results/RNN_EXPERIMENT_REPORT.md'):
    """Generate markdown report."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RNN-based Neural Machine Translation - Experiment Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents the results of comprehensive experiments on RNN-based Neural Machine Translation ")
        f.write("for Chinese-English translation, comparing:\n\n")
        f.write("1. **Attention Mechanisms**: Dot-product, General (Multiplicative), Concat (Additive)\n")
        f.write("2. **Training Policies**: Teacher Forcing, Free Running, Scheduled Sampling\n")
        f.write("3. **Decoding Strategies**: Greedy Search, Beam Search\n")
        f.write("4. **RNN Types**: GRU vs LSTM\n\n")
        f.write("---\n\n")
        
        # 1. Attention Mechanism Comparison
        f.write("## 1. Attention Mechanism Comparison\n\n")
        f.write("Comparing different alignment functions in the attention mechanism.\n\n")
        f.write("| Attention Type | Greedy BLEU | Beam BLEU | Improvement | Best |\n")
        f.write("|----------------|-------------|-----------|-------------|------|\n")
        
        attention_exps = ['rnn_attention_dot', 'rnn_attention_general', 'rnn_attention_concat']
        best_bleu = 0
        best_attention = None
        
        for exp in attention_exps:
            if exp in results and results[exp]['greedy'] and results[exp]['beam']:
                greedy_bleu = results[exp]['greedy']['bleu']
                beam_bleu = results[exp]['beam']['bleu']
                improvement = beam_bleu - greedy_bleu
                
                is_best = ""
                if beam_bleu > best_bleu:
                    best_bleu = beam_bleu
                    best_attention = exp
                    is_best = "⭐"
                
                f.write(f"| {results[exp]['name']} | {greedy_bleu:.2f} | {beam_bleu:.2f} | "
                       f"+{improvement:.2f} | {is_best} |\n")
        
        f.write("\n**Conclusion:** ")
        if best_attention:
            f.write(f"{results[best_attention]['name']} achieves the best performance.\n\n")
        else:
            f.write("Results pending.\n\n")
        
        # 2. Training Policy Comparison
        f.write("## 2. Training Policy Comparison\n\n")
        f.write("Comparing Teacher Forcing vs Free Running vs Scheduled Sampling.\n\n")
        f.write("| Training Policy | TF Ratio | Greedy BLEU | Beam BLEU | Improvement | Best |\n")
        f.write("|-----------------|----------|-------------|-----------|-------------|------|\n")
        
        training_exps = [
            ('rnn_teacher_forcing', '1.0'),
            ('rnn_free_running', '0.0'),
            ('rnn_nmt', '0.5')
        ]
        best_bleu = 0
        best_policy = None
        
        for exp, tf_ratio in training_exps:
            if exp in results and results[exp]['greedy'] and results[exp]['beam']:
                greedy_bleu = results[exp]['greedy']['bleu']
                beam_bleu = results[exp]['beam']['bleu']
                improvement = beam_bleu - greedy_bleu
                
                is_best = ""
                if beam_bleu > best_bleu:
                    best_bleu = beam_bleu
                    best_policy = exp
                    is_best = "⭐"
                
                f.write(f"| {results[exp]['name']} | {tf_ratio} | {greedy_bleu:.2f} | {beam_bleu:.2f} | "
                       f"+{improvement:.2f} | {is_best} |\n")
        
        f.write("\n**Conclusion:** ")
        if best_policy:
            f.write(f"{results[best_policy]['name']} achieves the best performance.\n\n")
        else:
            f.write("Results pending.\n\n")
        
        # 3. Decoding Strategy Comparison
        f.write("## 3. Decoding Strategy Comparison\n\n")
        f.write("Comparing Greedy Search vs Beam Search (beam_size=5) across all models.\n\n")
        f.write("| Model | Greedy BLEU | Beam BLEU | Improvement | Ratio |\n")
        f.write("|-------|-------------|-----------|-------------|-------|\n")
        
        for exp_key in results:
            if results[exp_key]['greedy'] and results[exp_key]['beam']:
                greedy_bleu = results[exp_key]['greedy']['bleu']
                beam_bleu = results[exp_key]['beam']['bleu']
                improvement = beam_bleu - greedy_bleu
                ratio = beam_bleu / greedy_bleu if greedy_bleu > 0 else 0
                
                f.write(f"| {results[exp_key]['name']} | {greedy_bleu:.2f} | {beam_bleu:.2f} | "
                       f"+{improvement:.2f} | {ratio:.3f}x |\n")
        
        f.write("\n")
        
        # 4. RNN Type Comparison (GRU vs LSTM)
        f.write("## 4. RNN Type Comparison\n\n")
        if 'rnn_nmt' in results and 'rnn_lstm' in results:
            if results['rnn_nmt']['beam'] and results['rnn_lstm']['beam']:
                gru_bleu = results['rnn_nmt']['beam']['bleu']
                lstm_bleu = results['rnn_lstm']['beam']['bleu']
                
                f.write("| RNN Type | Beam BLEU | Winner |\n")
                f.write("|----------|-----------|--------|\n")
                f.write(f"| GRU | {gru_bleu:.2f} | {'⭐' if gru_bleu > lstm_bleu else ''} |\n")
                f.write(f"| LSTM | {lstm_bleu:.2f} | {'⭐' if lstm_bleu > gru_bleu else ''} |\n\n")
        
        # Detailed N-gram Precisions
        f.write("## 5. Detailed N-gram Precisions\n\n")
        
        for exp_key in results:
            if results[exp_key]['greedy'] or results[exp_key]['beam']:
                f.write(f"### {results[exp_key]['name']}\n\n")
                
                if results[exp_key]['greedy']:
                    prec = results[exp_key]['greedy']['precisions']
                    f.write(f"**Greedy Search:**\n")
                    f.write(f"- BLEU: {results[exp_key]['greedy']['bleu']:.2f}\n")
                    f.write(f"- 1-gram: {prec[0]*100:.2f}%\n")
                    f.write(f"- 2-gram: {prec[1]*100:.2f}%\n")
                    f.write(f"- 3-gram: {prec[2]*100:.2f}%\n")
                    f.write(f"- 4-gram: {prec[3]*100:.2f}%\n\n")
                
                if results[exp_key]['beam']:
                    prec = results[exp_key]['beam']['precisions']
                    f.write(f"**Beam Search:**\n")
                    f.write(f"- BLEU: {results[exp_key]['beam']['bleu']:.2f}\n")
                    f.write(f"- 1-gram: {prec[0]*100:.2f}%\n")
                    f.write(f"- 2-gram: {prec[1]*100:.2f}%\n")
                    f.write(f"- 3-gram: {prec[2]*100:.2f}%\n")
                    f.write(f"- 4-gram: {prec[3]*100:.2f}%\n\n")
    
    print(f"✓ Report generated: {output_file}")


def main():
    print("="*60)
    print("Generating RNN Experiment Report")
    print("="*60)
    
    # Load results
    results = load_results()
    
    # Generate report
    generate_markdown_report(results)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

