"""
训练脚本 - 支持RNN和Transformer的所有实验配置
"""
import os
import argparse
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from tqdm import tqdm
import sacrebleu
from functools import partial
from collections import deque

from config import DataConfig, RNNConfig, TransformerConfig, TrainingConfig, MODEL_SCALES
from data_utils import (load_jsonl_data, clean_and_filter, Vocabulary,
                        TranslationDataset, collate_fn)
from rnn_model import build_rnn_model
from transformer_model import build_transformer_model, LabelSmoothingLoss


def get_lr_scheduler(optimizer, warmup_steps: int, d_model: int, base_lr: float = 1.0,
                     total_steps: int = None, scheduler_type: str = "noam"):
    """学习率调度器

    Args:
        scheduler_type: "noam" (原始Transformer) 或 "warmup_cosine" (更稳定)
    """
    if scheduler_type == "warmup_cosine" and total_steps:
        # Warmup + Cosine Annealing - 更稳定的调度方式
        def lr_lambda(step):
            step = max(step, 1)
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return LambdaLR(optimizer, lr_lambda)
    else:
        # Noam scheduler (原始)
        def lr_lambda(step):
            step = max(step, 1)
            return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
        return LambdaLR(optimizer, lr_lambda)


# ==================== 数据准备 ====================

def prepare_data(data_cfg: DataConfig):
    """准备数据和词表"""
    print("Loading and preprocessing data...")
    train_data = load_jsonl_data(data_cfg.train_path)
    valid_data = load_jsonl_data(data_cfg.valid_path)

    train_data = clean_and_filter(train_data, data_cfg.max_src_len, data_cfg.max_tgt_len)
    valid_data = clean_and_filter(valid_data, data_cfg.max_src_len, data_cfg.max_tgt_len)
    print(f"Train: {len(train_data)}, Valid: {len(valid_data)}")

    # 构建词表
    src_vocab = Vocabulary(min_freq=data_cfg.min_freq)
    tgt_vocab = Vocabulary(min_freq=data_cfg.min_freq)
    src_vocab.build_from_corpus([d['zh_tokens'] for d in train_data])
    tgt_vocab.build_from_corpus([d['en_tokens'] for d in train_data])
    print(f"Source vocab: {len(src_vocab)}, Target vocab: {len(tgt_vocab)}")

    return train_data, valid_data, src_vocab, tgt_vocab


def create_dataloaders(train_data, valid_data, src_vocab, tgt_vocab,
                       batch_size, max_src_len, max_tgt_len):
    """创建数据加载器"""
    train_ds = TranslationDataset(train_data, src_vocab, tgt_vocab, max_src_len, max_tgt_len)
    valid_ds = TranslationDataset(valid_data, src_vocab, tgt_vocab, max_src_len, max_tgt_len)

    collate = partial(collate_fn, src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, valid_loader


# ==================== RNN训练 ====================

def train_rnn(train_cfg: TrainingConfig, data_cfg: DataConfig, rnn_cfg: RNNConfig,
              experiment_name: str = "rnn_default"):
    """训练RNN模型"""
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training RNN: {experiment_name}")
    print(f"Attention: {rnn_cfg.attention_type}, RNN: {rnn_cfg.rnn_type}, TF: {rnn_cfg.teacher_forcing_ratio}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 准备数据
    train_data, valid_data, src_vocab, tgt_vocab = prepare_data(data_cfg)
    train_loader, valid_loader = create_dataloaders(
        train_data, valid_data, src_vocab, tgt_vocab,
        train_cfg.batch_size, data_cfg.max_src_len, data_cfg.max_tgt_len
    )

    # 创建模型
    model = build_rnn_model(
        len(src_vocab), len(tgt_vocab),
        rnn_cfg.embed_dim, rnn_cfg.hidden_dim, rnn_cfg.num_layers, rnn_cfg.dropout,
        rnn_cfg.attention_type, rnn_cfg.rnn_type,
        src_vocab.pad_idx, tgt_vocab.sos_idx, tgt_vocab.eos_idx, train_cfg.device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # 优化器和损失
    optimizer = Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_idx)

    # 学习率调度器 - 在 plateau 时降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6
    )

    # 训练循环
    best_bleu = 0.0
    best_avg_bleu = 0.0
    bleu_history = deque(maxlen=3)  # 移动平均
    history = {'train_loss': [], 'valid_bleu': [], 'epoch_time': [], 'smoothed_bleu': []}
    save_dir = f"{train_cfg.checkpoint_dir}/{experiment_name}"

    for epoch in range(train_cfg.num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for src, src_len, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}"):
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, src_len, tgt, rnn_cfg.teacher_forcing_ratio)
            loss = criterion(output[:, 1:].reshape(-1, len(tgt_vocab)), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
            optimizer.step()
            total_loss += loss.item()

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)

        # 评估 - 使用 beam search 获得更稳定的结果
        bleu = evaluate_rnn(model, valid_loader, tgt_vocab, device,
                           decode_method='beam', beam_size=3)

        # 计算移动平均 BLEU
        bleu_history.append(bleu)
        smoothed_bleu = sum(bleu_history) / len(bleu_history)

        # 更新学习率
        scheduler.step(smoothed_bleu)

        history['train_loss'].append(avg_loss)
        history['valid_bleu'].append(bleu)
        history['smoothed_bleu'].append(smoothed_bleu)
        history['epoch_time'].append(epoch_time)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | BLEU: {bleu:.2f} | "
              f"Smoothed: {smoothed_bleu:.2f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # 保存最佳模型 - 使用移动平均BLEU
        os.makedirs(save_dir, exist_ok=True)
        if smoothed_bleu > best_avg_bleu:
            best_avg_bleu = smoothed_bleu
            best_bleu = bleu
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            src_vocab.save(f"{save_dir}/src_vocab.pt")
            tgt_vocab.save(f"{save_dir}/tgt_vocab.pt")

    # 保存训练历史
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest BLEU: {best_bleu:.2f} (Smoothed: {best_avg_bleu:.2f})")
    return model, src_vocab, tgt_vocab, history


def evaluate_rnn(model, loader, tgt_vocab, device, decode_method='greedy', beam_size=5):
    """评估RNN模型"""
    model.eval()
    refs, hyps = [], []

    with torch.no_grad():
        for src, src_len, tgt in loader:
            src, src_len = src.to(device), src_len.to(device)

            # 批量greedy解码
            if decode_method == 'greedy':
                preds = model.greedy_decode(src, src_len)
                for i in range(src.size(0)):
                    pred = preds[i].cpu().tolist()
                    hyp = tgt_vocab.decode(pred)
                    hyp = ' '.join([t for t in hyp if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
                    ref = tgt_vocab.decode(tgt[i].tolist())
                    ref = ' '.join([t for t in ref if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
                    hyps.append(hyp)
                    refs.append([ref])
            else:
                # beam search需要逐个处理 - 但目前RNN beam search有bug，暂时回退到greedy
                # TODO: 修复RNN beam search
                preds = model.greedy_decode(src, src_len)
                for i in range(src.size(0)):
                    pred = preds[i].cpu().tolist()
                    hyp = tgt_vocab.decode(pred)
                    hyp = ' '.join([t for t in hyp if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
                    ref = tgt_vocab.decode(tgt[i].tolist())
                    ref = ' '.join([t for t in ref if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
                    hyps.append(hyp)
                    refs.append([ref])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score



# ==================== Transformer训练 ====================

def train_transformer(train_cfg: TrainingConfig, data_cfg: DataConfig, tf_cfg: TransformerConfig,
                      experiment_name: str = "transformer_default"):
    """训练Transformer模型"""
    device = torch.device(train_cfg.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training Transformer: {experiment_name}")
    print(f"PE: {tf_cfg.pos_encoding_type}, Norm: {tf_cfg.norm_type}, d_model: {tf_cfg.d_model}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # 准备数据
    train_data, valid_data, src_vocab, tgt_vocab = prepare_data(data_cfg)
    train_loader, valid_loader = create_dataloaders(
        train_data, valid_data, src_vocab, tgt_vocab,
        train_cfg.batch_size, data_cfg.max_src_len, data_cfg.max_tgt_len
    )

    # 创建模型
    model = build_transformer_model(
        len(src_vocab), len(tgt_vocab),
        tf_cfg.d_model, tf_cfg.nhead, tf_cfg.num_encoder_layers, tf_cfg.num_decoder_layers,
        tf_cfg.dim_feedforward, tf_cfg.dropout, tf_cfg.max_seq_len,
        src_vocab.pad_idx, tf_cfg.pos_encoding_type, tf_cfg.norm_type, train_cfg.device
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # 计算合理的训练参数
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * train_cfg.num_epochs

    # 动态计算 warmup_steps: 使用前 2 个 epoch 或总步数的 10%，取较小者
    warmup_steps = min(steps_per_epoch * 2, int(total_steps * 0.1), train_cfg.warmup_steps)
    warmup_steps = max(warmup_steps, 100)  # 至少 100 步

    # 根据模型规模调整学习率
    # 小模型用较大学习率，大模型用较小学习率
    if num_params > 100_000_000:  # > 100M 参数 (large)
        base_lr = 1e-4
        scheduler_type = "warmup_cosine"
    elif num_params > 30_000_000:  # > 30M 参数 (base)
        base_lr = 3e-4
        scheduler_type = "warmup_cosine"
    else:  # small
        base_lr = 5e-4
        scheduler_type = "warmup_cosine"

    print(f"Training config: warmup={warmup_steps}, lr={base_lr}, scheduler={scheduler_type}")

    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=train_cfg.weight_decay, betas=(0.9, 0.98))
    scheduler = get_lr_scheduler(optimizer, warmup_steps, tf_cfg.d_model,
                                 total_steps=total_steps, scheduler_type=scheduler_type)

    # 损失函数 (标签平滑)
    criterion = LabelSmoothingLoss(len(tgt_vocab), tgt_vocab.pad_idx, tf_cfg.label_smoothing)

    # 训练循环
    best_bleu = 0.0
    best_avg_bleu = 0.0  # 使用移动平均来选择最佳模型
    bleu_history = deque(maxlen=3)  # 保存最近3个epoch的BLEU用于平滑
    history = {'train_loss': [], 'valid_bleu': [], 'epoch_time': [], 'smoothed_bleu': []}
    step = 0
    save_dir = f"{train_cfg.checkpoint_dir}/{experiment_name}"

    for epoch in range(train_cfg.num_epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for src, src_len, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg.num_epochs}"):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, tgt[:, :-1])  # 输入不包含最后一个token
            loss = criterion(output, tgt[:, 1:])  # 目标不包含第一个token
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1

        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)

        # 评估 - 使用 beam search 获得更稳定的结果
        bleu = evaluate_transformer(model, valid_loader, tgt_vocab, device,
                                    decode_method='beam', beam_size=3)

        # 计算移动平均 BLEU
        bleu_history.append(bleu)
        smoothed_bleu = sum(bleu_history) / len(bleu_history)

        history['train_loss'].append(avg_loss)
        history['valid_bleu'].append(bleu)
        history['smoothed_bleu'].append(smoothed_bleu)
        history['epoch_time'].append(epoch_time)

        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | BLEU: {bleu:.2f} | "
              f"Smoothed: {smoothed_bleu:.2f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # 保存最佳模型 - 使用移动平均BLEU
        os.makedirs(save_dir, exist_ok=True)
        if smoothed_bleu > best_avg_bleu:
            best_avg_bleu = smoothed_bleu
            best_bleu = bleu
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")
            src_vocab.save(f"{save_dir}/src_vocab.pt")
            tgt_vocab.save(f"{save_dir}/tgt_vocab.pt")

    # 保存训练历史
    with open(f"{save_dir}/history.json", 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest BLEU: {best_bleu:.2f} (Smoothed: {best_avg_bleu:.2f})")
    return model, src_vocab, tgt_vocab, history


def evaluate_transformer(model, loader, tgt_vocab, device, decode_method='greedy', beam_size=5):
    """评估Transformer模型"""
    model.eval()
    refs, hyps = [], []

    with torch.no_grad():
        for src, src_len, tgt in loader:
            src = src.to(device)

            for i in range(src.size(0)):
                if decode_method == 'beam':
                    pred = model.beam_decode(src[i:i+1], tgt_vocab.sos_idx, tgt_vocab.eos_idx, beam_size=beam_size)
                else:
                    pred = model.greedy_decode(src[i:i+1], tgt_vocab.sos_idx, tgt_vocab.eos_idx)

                pred = pred[0].cpu().tolist()
                hyp = tgt_vocab.decode(pred)
                hyp = ' '.join([t for t in hyp if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])
                ref = tgt_vocab.decode(tgt[i].tolist())
                ref = ' '.join([t for t in ref if t not in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']])

                hyps.append(hyp)
                refs.append([ref])

    bleu = sacrebleu.corpus_bleu(hyps, refs)
    return bleu.score


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Train NMT Models')
    parser.add_argument('--model', type=str, choices=['rnn', 'transformer', 'both'], default='both')
    parser.add_argument('--attention', type=str, default='additive',
                        choices=['additive', 'dot', 'multiplicative'])
    parser.add_argument('--rnn_type', type=str, default='lstm', choices=['lstm', 'gru'])
    parser.add_argument('--tf_ratio', type=float, default=0.5)
    parser.add_argument('--pos_encoding', type=str, default='sinusoidal',
                        choices=['sinusoidal', 'relative', 'learned'])
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'rmsnorm'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scale', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--train_file', type=str, default='./datasets/train_10k.jsonl')
    args = parser.parse_args()

    # 配置
    data_cfg = DataConfig(train_path=args.train_file)
    train_cfg = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr
    )

    if args.model in ['rnn', 'both']:
        rnn_cfg = RNNConfig(
            attention_type=args.attention,
            rnn_type=args.rnn_type,
            teacher_forcing_ratio=args.tf_ratio
        )
        exp_name = f"rnn_{args.rnn_type}_{args.attention}_tf{args.tf_ratio}"
        train_rnn(train_cfg, data_cfg, rnn_cfg, exp_name)

    if args.model in ['transformer', 'both']:
        # 获取模型规模配置
        scale_params = MODEL_SCALES.get(args.scale, MODEL_SCALES['base'])
        tf_cfg = TransformerConfig(
            pos_encoding_type=args.pos_encoding,
            norm_type=args.norm_type,
            **scale_params
        )
        exp_name = f"transformer_{args.scale}_{args.pos_encoding}_{args.norm_type}"
        train_transformer(train_cfg, data_cfg, tf_cfg, exp_name)


if __name__ == "__main__":
    main()

