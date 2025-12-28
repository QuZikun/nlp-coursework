#!/bin/bash

# Evaluate Attention Mechanism Experiments
# Both Greedy and Beam Search

set -e

echo "=========================================="
echo "Attention Mechanism Evaluations"
echo "=========================================="
echo "Start time: $(date)"
echo "=========================================="

mkdir -p results

EXPERIMENTS=(
    "rnn_attention_dot"
    "rnn_attention_general"
    "rnn_attention_concat"
)

NAMES=(
    "Dot-product Attention"
    "General Attention"
    "Concat Attention"
)

for i in "${!EXPERIMENTS[@]}"; do
    EXP="${EXPERIMENTS[$i]}"
    NAME="${NAMES[$i]}"
    CHECKPOINT="checkpoints/${EXP}/best_model.pt"
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "Warning: Checkpoint not found: $CHECKPOINT"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Evaluating: $NAME"
    echo "=========================================="
    
    # Greedy Search
    echo ""
    echo "  [Greedy Search]"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --method greedy \
        --output "results/${EXP}_greedy.json" \
        2>&1 | tee "logs/eval_${EXP}_greedy.log"
    
    # Beam Search
    echo ""
    echo "  [Beam Search (size=5)]"
    python evaluate.py \
        --checkpoint "$CHECKPOINT" \
        --method beam \
        --beam_size 5 \
        --output "results/${EXP}_beam.json" \
        2>&1 | tee "logs/eval_${EXP}_beam.log"
done

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Results:"
ls -lh results/rnn_attention_*.json
echo ""
echo "Next: python generate_attention_report.py"
echo "=========================================="

