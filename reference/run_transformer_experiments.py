"""
Transformer实验运行脚本 - 只运行Transformer部分的实验
"""
import os
import json
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from config import DataConfig, TransformerConfig, TrainingConfig, MODEL_SCALES
from train import train_transformer
from transformer_model import build_transformer_model

# 创建结果目录
os.makedirs('./results', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(name, train_fn, *args, **kwargs):
    """运行单个实验并记录时间"""
    print(f"\n{'='*60}")
    print(f"实验: {name}")
    print(f"{'='*60}")
    start = time.time()
    result = train_fn(*args, **kwargs)
    elapsed = time.time() - start
    print(f"实验 {name} 完成，耗时: {elapsed/60:.1f} 分钟")
    return result, elapsed

def main():
    print("="*80)
    print("中英机器翻译 - Transformer实验")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    # 配置
    data_cfg = DataConfig(train_path="./datasets/train_10k.jsonl")
    results = {}
    
    # ==================== Transformer 实验 ====================
    print("\n" + "#"*80)
    print("# Transformer 实验")
    print("#"*80)
    
    # 实验1: 位置编码消融
    print("\n>>> 实验1: 位置编码消融")
    results['pos_encoding'] = {}
    for pe in ['sinusoidal', 'learned', 'relative']:
        train_cfg = TrainingConfig(num_epochs=15, batch_size=64, warmup_steps=1000)
        tf_cfg = TransformerConfig(pos_encoding_type=pe, **MODEL_SCALES['small'])
        (model, _, _, history), _ = run_experiment(
            f"Transformer-PE-{pe}", train_transformer, train_cfg, data_cfg, tf_cfg, f"tf_pe_{pe}"
        )
        results['pos_encoding'][pe] = {
            'best_bleu': max(history['valid_bleu']),
            'total_time': sum(history['epoch_time']),
            'params': count_parameters(model),
            'history': history
        }

    # 实验2: 归一化方式
    print("\n>>> 实验2: 归一化方式对比")
    results['normalization'] = {}
    for norm in ['layernorm', 'rmsnorm']:
        train_cfg = TrainingConfig(num_epochs=15, batch_size=64, warmup_steps=1000)
        tf_cfg = TransformerConfig(norm_type=norm, **MODEL_SCALES['small'])
        (model, _, _, history), _ = run_experiment(
            f"Transformer-{norm}", train_transformer, train_cfg, data_cfg, tf_cfg, f"tf_norm_{norm}"
        )
        results['normalization'][norm] = {
            'best_bleu': max(history['valid_bleu']),
            'total_time': sum(history['epoch_time']),
            'params': count_parameters(model),
            'history': history
        }

    # 实验3: 模型规模
    print("\n>>> 实验3: 模型规模对比")
    results['model_scale'] = {}
    # 根据模型规模调整 batch size 以避免 OOM
    scale_batch_sizes = {'small': 64, 'base': 32, 'large': 16}
    for scale in ['small', 'base', 'large']:
        batch_size = scale_batch_sizes[scale]
        train_cfg = TrainingConfig(num_epochs=15, batch_size=batch_size, warmup_steps=1000)
        tf_cfg = TransformerConfig(**MODEL_SCALES[scale])
        (model, _, _, history), _ = run_experiment(
            f"Transformer-{scale}", train_transformer, train_cfg, data_cfg, tf_cfg, f"tf_scale_{scale}"
        )
        results['model_scale'][scale] = {
            'best_bleu': max(history['valid_bleu']),
            'total_time': sum(history['epoch_time']),
            'params': count_parameters(model),
            'history': history
        }
    
    # 保存结果
    print("\n保存Transformer实验结果...")
    with open('./results/transformer_results.json', 'w') as f:
        # 转换history中的numpy数组为列表
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        json.dump(convert(results), f, indent=2)
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return results

if __name__ == "__main__":
    results = main()

