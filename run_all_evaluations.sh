#!/bin/bash

# RNN-based NMT Evaluation Script
# Evaluates all trained models with both greedy and beam search

set -e

echo "=========================================="
echo "RNN-based NMT Evaluation Pipeline"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

mkdir -p results

# List of experiments to evaluate
EXPERIMENTS=(
    "rnn_attention_dot"
    "rnn_attention_general"
    "rnn_attention_concat"
    "rnn_teacher_forcing"
    "rnn_free_running"
    "rnn_lstm"
    "rnn_nmt"
)

EXPERIMENT_NAMES=(
    "Dot-product Attention"
    "General Attention"
    "Concat Attention"
    "Teacher Forcing"
    "Free Running"
    "LSTM"
    "Default (GRU + Scheduled)"
)

# Evaluate each model with both greedy and beam search
for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NAME="${EXPERIMENT_NAMES[$i]}"
    CHECKPOINT="checkpoints/${EXP}/best_model.pt"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $NAME"
    echo "=========================================="
    
    # Greedy decoding
    echo ""
    echo "  [Greedy Search]"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --method greedy \
        --max_samples 1000 \
        --output "results/${EXP}_greedy.json" \
        2>&1 | tee "logs/eval_${EXP}_greedy.log"
    
    # Beam search
    echo ""
    echo "  [Beam Search (size=5)]"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --method beam \
        --beam_size 5 \
        --max_samples 1000 \
        --output "results/${EXP}_beam.json" \
        2>&1 | tee "logs/eval_${EXP}_beam.log"
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results saved in: results/"
ls -lh results/*.json
echo ""
echo "Next step: python generate_report.py"
echo "=========================================="

