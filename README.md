# Chinese-English Neural Machine Translation

## 📋 项目概述

本项目实现了基于RNN和Transformer的中英文神经机器翻译系统，并对比了不同模型架构和训练策略的性能。

**目标：**
- 实现RNN-based NMT（使用LSTM，支持多种注意力机制）
- 实现Transformer-based NMT
- 对比不同注意力机制、训练策略和解码策略的效果

---

## 📁 项目结构

```
final/
├── README.md                           # 本文档
├── config.py                           # 配置文件（模型、训练、数据参数）
├── data_preprocess.py                  # 数据预处理（分词、词汇表、DataLoader）
│
├── models/                             # 模型定义
│   ├── __init__.py
│   ├── rnn_nmt.py                     # RNN模型（Encoder-Decoder + Attention）
│   └── transformer_nmt.py             # Transformer模型
│
├── train_rnn.py                       # RNN训练脚本
├── train_transformer.py               # Transformer训练脚本
├── evaluate.py                        # RNN评估脚本
├── evaluate_transformer.py            # Transformer评估脚本
│
├── run_attention_experiments.sh       # 运行注意力机制对比实验
├── run_attention_evaluations.sh       # 评估注意力机制实验
├── generate_attention_report.py       # 生成注意力机制对比报告
│
├── run_all_rnn_experiments.sh         # 运行所有RNN实验
├── run_all_evaluations.sh             # 评估所有RNN实验
├── generate_report.py                 # 生成完整实验报告
│
├── AP0004_Midterm&Final_translation_dataset_zh_en/  # 数据集
│   ├── train_10k.jsonl                # 训练集（10k样本）
│   ├── train_100k.jsonl               # 训练集（100k样本）
│   ├── valid.jsonl                    # 验证集
│   ├── test.jsonl                     # 测试集
│   ├── src_vocab.pkl                  # 源语言词汇表
│   └── tgt_vocab.pkl                  # 目标语言词汇表
│
├── checkpoints/                       # 模型检查点
│   ├── rnn_attention_dot/
│   ├── rnn_attention_general/
│   ├── rnn_attention_concat/
│   ├── rnn_teacher_forcing/
│   ├── rnn_free_running/
│   ├── rnn_nmt/
│   └── transformer_nmt/
│
├── logs/                              # 训练日志
└── results/                           # 评估结果
```

---

## 🔧 环境配置

### 依赖项
```bash
torch>=2.0.0
jieba                  # 中文分词
nltk                   # 英文分词和BLEU评估
tqdm                   # 进度条
numpy
```

### GPU设置
本项目使用 **CUDA:2**，如需修改请编辑 `config.py` 第98行：
```python
device: str = 'cuda:2'  # 改为你的GPU编号
```

---

## 📚 核心文件说明

### 1. 配置文件

#### `config.py`
定义所有实验的配置参数：
- **DataConfig**: 数据路径、词汇表大小、序列长度
- **RNNModelConfig**: RNN模型参数（embed_dim, hidden_dim, attention_type等）
- **TransformerModelConfig**: Transformer模型参数（d_model, n_heads, n_layers等）
- **TrainConfig**: 训练参数（batch_size, learning_rate, epochs等）
- **EvalConfig**: 评估参数（decode_method, beam_size等）

**预定义实验配置：**
- `get_config('attention_dot')`: Dot-product注意力
- `get_config('attention_general')`: General注意力
- `get_config('attention_concat')`: Concat注意力
- `get_config('teacher_forcing')`: Teacher Forcing (TF=1.0)
- `get_config('free_running')`: Free Running (TF=0.0)
- `get_config('default')`: 默认配置（LSTM + General Attention + TF=0.5）

### 2. 数据处理

#### `data_preprocess.py`
- **ChineseTokenizer**: 使用Jieba进行中文分词
- **EnglishTokenizer**: 使用NLTK进行英文分词
- **Vocabulary**: 词汇表类（word2idx, idx2word）
- **TranslationDataset**: PyTorch Dataset
- **prepare_data()**: 一键准备训练/验证/测试数据

### 3. 模型定义

#### `models/rnn_nmt.py`
RNN-based Seq2Seq模型：
- **Encoder**: 2层单向LSTM
- **Decoder**: 2层单向LSTM + Attention
- **Attention**: 支持3种注意力机制
  - `dot`: 点积注意力
  - `general`: 通用注意力（Luong）
  - `concat`: 拼接注意力（Bahdanau）
- **Seq2Seq**: 完整的序列到序列模型
  - 支持Teacher Forcing和Free Running
  - 支持Greedy和Beam Search解码

#### `models/transformer_nmt.py`
Transformer模型：
- **MultiHeadAttention**: 多头注意力
- **PositionwiseFeedForward**: 前馈网络
- **PositionalEncoding**: 位置编码
- **TransformerEncoder/Decoder**: 编码器和解码器
- **Transformer**: 完整模型

---

## 🚀 使用方法

### RNN实验

#### 1. 注意力机制对比实验

**训练3个模型（Dot, General, Concat）：**
```bash
cd /workspace/users/zikun/course_project/final
CUDA_VISIBLE_DEVICES=2 bash run_attention_experiments.sh
```

**评估：**
```bash
CUDA_VISIBLE_DEVICES=2 bash run_attention_evaluations.sh
```

#### 4. 运行所有RNN实验

```bash
# 训练所有RNN变体（注意力机制 + 训练策略）
CUDA_VISIBLE_DEVICES=2 bash run_all_rnn_experiments.sh

# 评估所有模型
CUDA_VISIBLE_DEVICES=2 bash run_all_evaluations.sh

# 生成完整报告
python generate_report.py
# 输出: results/RNN_EXPERIMENT_REPORT.md
```

---

### Transformer实验

#### 1. 训练Transformer模型

```bash
# 默认配置
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment default --epochs 20 --batch_size 64

# 不同位置编码
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment pos_absolute --epochs 20
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment pos_learned --epochs 20

# 不同模型大小
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment small --epochs 20
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment base --epochs 20
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment large --epochs 20
```

#### 2. 评估Transformer模型

```bash
# Greedy Search
CUDA_VISIBLE_DEVICES=2 python evaluate_transformer.py \
    --checkpoint checkpoints/transformer_nmt/best_model.pt \
    --method greedy \
    --output results/transformer_greedy.json

# Beam Search
CUDA_VISIBLE_DEVICES=2 python evaluate_transformer.py \
    --checkpoint checkpoints/transformer_nmt/best_model.pt \
    --method beam \
    --beam_size 5 \
    --output results/transformer_beam.json
```

---

## 📊 实验设置

### RNN实验

#### 实验1: 注意力机制对比
- **Dot-product Attention**: 简单点积
- **General Attention**: 带权重矩阵（Luong）
- **Concat Attention**: 拼接+前馈网络（Bahdanau）

#### 实验2: 训练策略对比
- **Teacher Forcing (TF=1.0)**: 总是使用真实标签
- **Free Running (TF=0.0)**: 总是使用模型预测
- **Scheduled Sampling (TF=0.5)**: 50%概率使用真实标签

#### 实验3: 解码策略对比
- **Greedy Search**: 每步选择概率最高的词
- **Beam Search**: 保留top-k候选序列（k=5）

### 模型参数

#### RNN模型
```python
embed_dim = 256          # 词嵌入维度
hidden_dim = 512         # 隐藏层维度
n_layers = 2             # 层数（编码器和解码器各2层）
dropout = 0.3            # Dropout率
rnn_type = 'LSTM'        # RNN类型
```

#### Transformer模型
```python
d_model = 256            # 模型维度
n_heads = 8              # 注意力头数
n_layers = 4             # 编码器和解码器层数
d_ff = 1024              # 前馈网络维度
dropout = 0.1            # Dropout率
```

### 训练参数
```python
batch_size = 64          # 批次大小
epochs = 20              # 训练轮数
learning_rate = 0.001    # 学习率（RNN）
learning_rate = 0.0001   # 学习率（Transformer）
optimizer = Adam         # 优化器
clip_grad = 1.0          # 梯度裁剪
early_stopping = 5       # 早停patience
```

---

## 📈 评估指标

### BLEU Score
- **范围**: 0-1（越高越好）
- **计算方法**: NLTK corpus_bleu with smoothing method 4
- **N-gram**: 1-gram, 2-gram, 3-gram, 4-gram precisions

### 输出示例
```
==================================================
BLEU Score: 0.0046
Precision 1-gram: 0.1215
Precision 2-gram: 0.0357
Precision 3-gram: 0.0096
Precision 4-gram: 0.0046
==================================================
```

---

## 📝 输出文件

### 训练输出
- **Checkpoints**: `checkpoints/{experiment_name}/best_model.pt`
  - 包含：模型权重、优化器状态、配置、训练/验证损失
- **日志**: `logs/train_{experiment_name}.log`
  - 每个epoch的训练和验证损失

### 评估输出
- **结果JSON**: `results/{experiment_name}_{method}.json`
  ```json
  {
    "bleu": 0.0046,
    "precisions": [0.1215, 0.0357, 0.0096, 0.0046],
    "results": [
      {
        "source": "中文句子",
        "reference": "参考翻译",
        "hypothesis": "模型翻译"
      },
      ...
    ]
  }
  ```

- **对比报告**: `results/ATTENTION_REPORT.md` 或 `results/RNN_EXPERIMENT_REPORT.md`
  - Markdown格式的实验对比表格
  - 最佳模型标注
  - 详细的N-gram precisions

---

## 🔍 监控训练

### 查看训练日志
```bash
# 实时查看
tail -f logs/train_attention_dot.log

# 查看最后50行
tail -50 logs/train_attention_dot.log
```

### 检查GPU使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 查看GPU 2
nvidia-smi --id=2
```

### 查看已训练模型
```bash
ls -lh checkpoints/*/best_model.pt
```

---

## ⚙️ 自定义参数

### 修改训练参数
```bash
python train_rnn.py \
    --experiment attention_dot \
    --epochs 30 \              # 自定义epoch数
    --batch_size 32 \          # 自定义batch size
    --lr 0.0005 \              # 自定义学习率
    --tf_ratio 0.7             # 自定义teacher forcing ratio
```

### 修改评估参数
```bash
python evaluate.py \
    --checkpoint checkpoints/rnn_attention_dot/best_model.pt \
    --method beam \
    --beam_size 10 \           # 自定义beam size
    --max_len 150 \            # 自定义最大输出长度
    --max_samples 500          # 只评估前500个样本
```

---

## 🐛 常见问题

### 1. CUDA Out of Memory
**解决方案**: 减小batch size
```bash
python train_rnn.py --experiment default --batch_size 32
```

### 2. 训练速度慢
**检查**:
- GPU是否被正确使用：`nvidia-smi`
- 是否使用了正确的GPU：检查`config.py`中的device设置

### 3. BLEU分数为0或很低
**可能原因**:
- 模型训练不充分（增加epochs）
- 数据量太小（使用train_100k.jsonl）
- 学习率不合适（调整lr）

### 4. 词汇表未找到
**解决方案**: 首次训练会自动生成词汇表，确保：
```bash
ls AP0004_Midterm&Final_translation_dataset_zh_en/*.pkl
# 应该看到 src_vocab.pkl 和 tgt_vocab.pkl
```

---

## 📊 实验流程建议

### 快速验证（1小时）
```bash
# 1. 训练一个模型（30分钟）
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment attention_dot --epochs 20

# 2. 评估（5分钟）
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --checkpoint checkpoints/rnn_attention_dot/best_model.pt \
    --method greedy

# 3. 查看结果
cat results/rnn_attention_dot_greedy.json | grep bleu
```

### 完整RNN实验（4-5小时）
```bash
# 1. 训练所有RNN模型（3-4小时）
CUDA_VISIBLE_DEVICES=2 bash run_all_rnn_experiments.sh

# 2. 评估所有模型（30-45分钟）
CUDA_VISIBLE_DEVICES=2 bash run_all_evaluations.sh

# 3. 生成报告
python generate_report.py

# 4. 查看报告
cat results/RNN_EXPERIMENT_REPORT.md
```

### RNN vs Transformer对比（8-10小时）
```bash
# 1. 完成所有RNN实验（4-5小时）
CUDA_VISIBLE_DEVICES=2 bash run_all_rnn_experiments.sh
CUDA_VISIBLE_DEVICES=2 bash run_all_evaluations.sh

# 2. 训练Transformer（2-3小时）
CUDA_VISIBLE_DEVICES=2 python train_transformer.py --experiment default --epochs 20

# 3. 评估Transformer（15分钟）
CUDA_VISIBLE_DEVICES=2 python evaluate_transformer.py \
    --checkpoint checkpoints/transformer_nmt/best_model.pt \
    --method greedy
CUDA_VISIBLE_DEVICES=2 python evaluate_transformer.py \
    --checkpoint checkpoints/transformer_nmt/best_model.pt \
    --method beam

# 4. 对比分析
# 手动对比RNN和Transformer的BLEU分数
```

---

## 📖 参考文献

1. **Attention Mechanism**:
   - Bahdanau et al. (2015) - Neural Machine Translation by Jointly Learning to Align and Translate
   - Luong et al. (2015) - Effective Approaches to Attention-based Neural Machine Translation

2. **Transformer**:
   - Vaswani et al. (2017) - Attention Is All You Need

3. **Training Strategies**:
   - Bengio et al. (2015) - Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

4. **BLEU Score**:
   - Papineni et al. (2002) - BLEU: a Method for Automatic Evaluation of Machine Translation
   - Chen and Cherry (2014) - A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU

---

## 👥 作者

课程项目 - Chinese-English Neural Machine Translation

---

## 📄 许可

本项目仅用于学术研究和课程作业。

**生成报告：**
```bash
python generate_attention_report.py
# 输出: results/ATTENTION_REPORT.md
```

#### 2. 单独训练某个RNN模型

```bash
# Dot-product Attention
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment attention_dot --epochs 20 --batch_size 64

# General Attention
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment attention_general --epochs 20 --batch_size 64

# Concat Attention
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment attention_concat --epochs 20 --batch_size 64

# Teacher Forcing
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment teacher_forcing --epochs 20 --batch_size 64

# Free Running
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment free_running --epochs 20 --batch_size 64

# 默认配置（LSTM + General + TF=0.5）
CUDA_VISIBLE_DEVICES=2 python train_rnn.py --experiment default --epochs 20 --batch_size 64
```

#### 3. 评估RNN模型

```bash
# Greedy Search
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --checkpoint checkpoints/rnn_attention_dot/best_model.pt \
    --method greedy \
    --output results/rnn_attention_dot_greedy.json

# Beam Search
CUDA_VISIBLE_DEVICES=2 python evaluate.py \
    --checkpoint checkpoints/rnn_attention_dot/best_model.pt \
    --method beam \
    --beam_size 5 \
    --output results/rnn_attention_dot_beam.json
```


