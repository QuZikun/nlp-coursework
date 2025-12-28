#!/bin/bash

# RNN-based NMT Training Script
# This script trains all required RNN experiments:
# 1. Different attention mechanisms (dot, general, concat)
# 2. Different training policies (teacher forcing, free running, scheduled sampling)
# 3. Different RNN types (GRU, LSTM)

set -e  # Exit on error

echo "=========================================="
echo "RNN-based NMT Training Pipeline"
echo "=========================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"
echo "=========================================="

# Create directories
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# Training parameters
EPOCHS=20
BATCH_SIZE=64
LR=0.001

# ==========================================
# Part 1: Attention Mechanism Comparison
# ==========================================
echo ""
echo "=========================================="
echo "Part 1: Training with Different Attention Mechanisms"
echo "=========================================="

# 1.1 Dot-product Attention
echo ""
echo "[1/7] Training with Dot-product Attention..."
python train_rnn.py \
    --experiment attention_dot \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_dot.log

# 1.2 General (Multiplicative) Attention
echo ""
echo "[2/7] Training with General Attention..."
python train_rnn.py \
    --experiment attention_general \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_general.log

# 1.3 Concat (Additive) Attention
echo ""
echo "[3/7] Training with Concat Attention..."
python train_rnn.py \
    --experiment attention_concat \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_attention_concat.log

# ==========================================
# Part 2: Training Policy Comparison
# ==========================================
echo ""
echo "=========================================="
echo "Part 2: Training with Different Training Policies"
echo "=========================================="

# 2.1 Teacher Forcing (ratio = 1.0)
echo ""
echo "[4/7] Training with Teacher Forcing (TF=1.0)..."
python train_rnn.py \
    --experiment teacher_forcing \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_teacher_forcing.log

# 2.2 Free Running (ratio = 0.0)
echo ""
echo "[5/7] Training with Free Running (TF=0.0)..."
python train_rnn.py \
    --experiment free_running \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_free_running.log

# ==========================================
# Part 3: RNN Type Comparison
# ==========================================
echo ""
echo "=========================================="
echo "Part 3: Training with Different RNN Types"
echo "=========================================="

# 3.1 LSTM
echo ""
echo "[6/7] Training with LSTM..."
python train_rnn.py \
    --experiment lstm \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_lstm.log

# 3.2 Default (GRU with scheduled sampling)
echo ""
echo "[7/7] Training Default Model (GRU, TF=0.5)..."
python train_rnn.py \
    --experiment default \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    2>&1 | tee logs/train_default.log

echo ""
echo "=========================================="
echo "Training Complete!"
echo "End time: $(date)"
echo "=========================================="
echo ""
echo "Trained models:"
ls -lh checkpoints/
echo ""
echo "Next steps:"
echo "1. Run evaluation: bash run_all_evaluations.sh"
echo "2. Check logs in: logs/"
echo "=========================================="

