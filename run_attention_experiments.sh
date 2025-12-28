#!/bin/bash

# Attention Mechanism Comparison Experiments
# All models use LSTM (2-layer unidirectional)

set -e

echo "=========================================="
echo "Attention Mechanism Experiments"
echo "=========================================="
echo "GPU: CUDA:2"
echo "RNN Type: LSTM"
echo "Start time: $(date)"
echo "=========================================="

mkdir -p checkpoints logs results

EPOCHS=20
BATCH_SIZE=64
LR=0.001

# Experiment 1: Dot-product Attention
echo ""
echo "[1/3] Training with Dot-product Attention..."
python train_rnn.py \
    --experiment attention_dot \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_dot.log

# Experiment 2: General (Multiplicative) Attention
echo ""
echo "[2/3] Training with General Attention..."
python train_rnn.py \
    --experiment attention_general \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_general.log

# Experiment 3: Concat (Additive) Attention
echo ""
echo "[3/3] Training with Concat Attention..."
python train_rnn.py \
    --experiment attention_concat \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_concat.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Trained models:"
ls -lh checkpoints/rnn_attention_*/best_model.pt
echo ""
echo "Next: Run evaluations with run_attention_evaluations.sh"
echo "=========================================="

