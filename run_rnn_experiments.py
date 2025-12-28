"""
Run RNN experiments for Training Policy and Decoding Policy comparison.

This script:
1. Trains models with different teacher forcing ratios
2. Evaluates each model with both greedy and beam search
3. Generates comparison report
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command and log the output."""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"Command: {cmd}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"Completed in {elapsed/60:.2f} minutes")
    print(f"{'='*80}\n")
    
    return result.returncode == 0


def train_models(experiments, epochs=15, batch_size=64):
    """Train RNN models with different configurations."""
    results = {}
    
    for exp_name, exp_config in experiments.items():
        print(f"\n{'#'*80}")
        print(f"# Training: {exp_name}")
        print(f"# Config: {exp_config}")
        print(f"{'#'*80}\n")
        
        cmd = f"cd final && python train_rnn.py --experiment {exp_config} --epochs {epochs} --batch_size {batch_size}"
        success = run_command(cmd, f"Training {exp_name}")
        
        results[exp_name] = {
            'config': exp_config,
            'success': success,
            'checkpoint': f"checkpoints/rnn_{exp_config}/best_model.pt"
        }
    
    return results


def evaluate_models(train_results, decoding_methods):
    """Evaluate trained models with different decoding strategies."""
    eval_results = {}
    
    for exp_name, train_info in train_results.items():
        if not train_info['success']:
            print(f"Skipping {exp_name} - training failed")
            continue
        
        checkpoint = train_info['checkpoint']
        if not os.path.exists(checkpoint):
            print(f"Checkpoint not found: {checkpoint}")
            continue
        
        eval_results[exp_name] = {}
        
        for method in decoding_methods:
            print(f"\n{'#'*80}")
            print(f"# Evaluating: {exp_name} with {method} decoding")
            print(f"{'#'*80}\n")
            
            output_file = f"results/{exp_name}_{method}.json"
            os.makedirs("results", exist_ok=True)
            
            if method == 'greedy':
                cmd = f"cd final && python evaluate.py --checkpoint {checkpoint} --method greedy --output ../{output_file}"
            else:  # beam search
                cmd = f"cd final && python evaluate.py --checkpoint {checkpoint} --method beam --beam_size 5 --output ../{output_file}"
            
            success = run_command(cmd, f"Evaluating {exp_name} with {method}")
            
            # Load results
            if success and os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    result_data = json.load(f)
                    eval_results[exp_name][method] = {
                        'bleu': result_data['bleu'],
                        'precisions': result_data['precisions']
                    }
            else:
                eval_results[exp_name][method] = None
    
    return eval_results


def generate_report(eval_results, output_file='results/rnn_comparison_report.md'):
    """Generate markdown report comparing all experiments."""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# RNN-based NMT Experiment Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Training Policy Comparison
        f.write("## 1. Training Policy Comparison\n\n")
        f.write("Comparing Teacher Forcing vs Free Running vs Scheduled Sampling\n\n")
        f.write("| Training Policy | TF Ratio | Greedy BLEU | Beam BLEU | Improvement |\n")
        f.write("|----------------|----------|-------------|-----------|-------------|\n")
        
        for exp_name in ['teacher_forcing', 'free_running', 'default']:
            if exp_name in eval_results:
                greedy_bleu = eval_results[exp_name].get('greedy', {}).get('bleu', 0)
                beam_bleu = eval_results[exp_name].get('beam', {}).get('bleu', 0)
                improvement = beam_bleu - greedy_bleu
                
                tf_ratio = {'teacher_forcing': '1.0', 'free_running': '0.0', 'default': '0.5'}[exp_name]
                policy_name = {'teacher_forcing': 'Teacher Forcing', 'free_running': 'Free Running', 'default': 'Scheduled Sampling'}[exp_name]
                
                f.write(f"| {policy_name} | {tf_ratio} | {greedy_bleu:.2f} | {beam_bleu:.2f} | +{improvement:.2f} |\n")
        
        # Decoding Policy Comparison
        f.write("\n## 2. Decoding Policy Comparison\n\n")
        f.write("Comparing Greedy Search vs Beam Search (beam_size=5)\n\n")
        f.write("| Model | Greedy BLEU | Beam BLEU | Improvement | Beam/Greedy Ratio |\n")
        f.write("|-------|-------------|-----------|-------------|-------------------|\n")
        
        for exp_name, results in eval_results.items():
            if 'greedy' in results and 'beam' in results:
                greedy_bleu = results['greedy'].get('bleu', 0)
                beam_bleu = results['beam'].get('bleu', 0)
                improvement = beam_bleu - greedy_bleu
                ratio = beam_bleu / greedy_bleu if greedy_bleu > 0 else 0
                
                f.write(f"| {exp_name} | {greedy_bleu:.2f} | {beam_bleu:.2f} | +{improvement:.2f} | {ratio:.3f}x |\n")
        
        # Detailed Results
        f.write("\n## 3. Detailed N-gram Precisions\n\n")
        for exp_name, results in eval_results.items():
            f.write(f"\n### {exp_name}\n\n")
            
            for method in ['greedy', 'beam']:
                if method in results and results[method]:
                    f.write(f"\n**{method.capitalize()} Search:**\n")
                    f.write(f"- BLEU: {results[method]['bleu']:.2f}\n")
                    prec = results[method]['precisions']
                    f.write(f"- 1-gram: {prec[0]*100:.2f}%\n")
                    f.write(f"- 2-gram: {prec[1]*100:.2f}%\n")
                    f.write(f"- 3-gram: {prec[2]*100:.2f}%\n")
                    f.write(f"- 4-gram: {prec[3]*100:.2f}%\n")
    
    print(f"\nReport generated: {output_file}")


def main():
    """Main experiment pipeline."""
    
    # Define experiments
    experiments = {
        'teacher_forcing': 'teacher_forcing',  # TF ratio = 1.0
        'free_running': 'free_running',        # TF ratio = 0.0
        'scheduled_sampling': 'default',       # TF ratio = 0.5
    }
    
    decoding_methods = ['greedy', 'beam']
    
    print("="*80)
    print("RNN-based NMT Experiments")
    print("="*80)
    print("\nExperiments to run:")
    print("1. Training Policy: Teacher Forcing, Free Running, Scheduled Sampling")
    print("2. Decoding Policy: Greedy Search, Beam Search")
    print("\n" + "="*80 + "\n")
    
    # Step 1: Train models
    print("\n" + "#"*80)
    print("# STEP 1: Training Models")
    print("#"*80 + "\n")
    train_results = train_models(experiments, epochs=15, batch_size=64)
    
    # Step 2: Evaluate models
    print("\n" + "#"*80)
    print("# STEP 2: Evaluating Models")
    print("#"*80 + "\n")
    eval_results = evaluate_models(train_results, decoding_methods)
    
    # Step 3: Generate report
    print("\n" + "#"*80)
    print("# STEP 3: Generating Report")
    print("#"*80 + "\n")
    
    # Save evaluation results
    os.makedirs("results", exist_ok=True)
    with open("results/rnn_eval_results.json", 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    generate_report(eval_results)
    
    print("\n" + "="*80)
    print("All experiments completed!")
    print("="*80)


if __name__ == '__main__':
    main()

