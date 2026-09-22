# AdaptiveNN 雷达信号识别训练指南

## 概述

本指南说明如何使用预训练权重在雷达CWD图像数据集上训练AdaptiveNN模型。

## 前置条件

### 1. 环境准备
确保已激活conda环境：
```bash
conda activate adaptivenn
```

### 2. 数据集结构
确保数据集位于正确位置：
```
d:/project/测试coatnet (2)/
├── CWDtrain/          # 训练数据
│   ├── BPSK/
│   ├── Costas/
│   ├── CP/
│   ├── Frank/
│   ├── FSK4_Baker5/
│   ├── FSK4_LFM/
│   ├── LFM/
│   ├── NLFM/
│   ├── P1/
│   ├── P2/
│   ├── P3/
│   ├── P4/
│   ├── T1/
│   ├── T2/
│   ├── T3/
│   └── T4/
└── CWDtest/           # 测试数据
    └── (相同的16个类别)
```

### 3. 预训练权重
将预训练权重文件 `AdaptiveNN_ckpt.pth` 放在以下位置：
```
测试coatnet/测试coatnet/AdaptiveNN-main/AdaptiveNN-main/AdaptiveNN_ckpt.pth
```

## 训练配置

### 当前配置（已优化用于RTX 2080Ti）

| 参数 | 值 | 说明 |
|------|-----|------|
| 类别数 | 16 | 雷达信号类型 |
| Batch Size | 128 | 针对11GB显存优化 |
| Epochs | 50 | 训练轮数 |
| Learning Rate | 5e-4 | 降低以防止NaN |
| Gradient Clipping | 0.5 | 梯度裁剪，提高稳定性 |
| 输入尺寸 | 224×224 | 标准输入 |
| Glance输入尺寸 | 112×112 | 快速浏览网络 |
| Focus Patch尺寸 | 112×112 | 聚焦区域 |
| Seq Length | 4 | 序列长度 |
| Gamma | 0.5 | RL折扣因子 |
| PPO Lambda | 0.84 | PPO优势估计 |
| PPO更新步数 | 5 | 策略更新次数 |
| 数据加载线程 | 8 | 并行加载 |

## 开始训练

### 方法1：使用批处理脚本（推荐）

直接双击运行：
```
测试coatnet/测试coatnet/AdaptiveNN-main/AdaptiveNN-main/train_radar.bat
```

### 方法2：命令行运行

在AdaptiveNN目录下执行：
```bash
cd 测试coatnet/测试coatnet/AdaptiveNN-main/AdaptiveNN-main
call conda activate adaptivenn
train_radar.bat
```

## 训练过程

### 1. 预训练权重加载
脚本会自动加载 `AdaptiveNN_ckpt.pth` 中的预训练权重：
- 加载除分类头外的所有权重
- 分类头会被重新初始化以适应16类任务
- 如果权重形状不匹配，会自动跳过该层

### 2. 训练监控
训练过程中会显示：
- 每个epoch的训练损失
- 验证集准确率（整体准确率和Glance网络准确率）
- 最佳模型会自动保存

### 3. 输出文件
训练结果保存在 `./output_radar/` 目录：
```
output_radar/
├── checkpoint-best.pth      # 最佳模型
├── checkpoint-XX.pth         # 定期保存的检查点
└── log.txt                   # 训练日志
```

## 模型架构说明

AdaptiveNN采用Glance-Focus双网络架构：

### Glance Network（快速浏览网络）
- 输入：112×112低分辨率图像
- 作用：快速获取全局信息
- 输出：初步分类结果

### Focus Network（聚焦网络）
- 输入：224×224高分辨率图像的局部区域
- 作用：关注重要区域进行精细分类
- 由策略网络（PPO）决定关注位置

### Policy Network（策略网络）
- 使用强化学习（PPO算法）
- 学习选择最有信息量的图像区域
- Actor-Critic架构

## 训练技巧

### 1. 如果遇到显存不足
减小batch size：
```bash
--batch_size 64  # 或更小
```

### 2. 如果训练不稳定（出现NaN）
- 降低学习率：`--lr 1e-4`
- 增强梯度裁剪：`--clip_grad 0.3`
- 检查数据是否有异常值

### 3. 如果想要更快收敛
- 增加学习率：`--lr 1e-3`（谨慎使用）
- 增加warmup epochs：`--warmup_epochs 10`

### 4. 调整RL参数
- `--gamma`：折扣因子，控制长期奖励权重
- `--ppo_lam`：GAE lambda，控制优势估计
- `--ppo_update_steps`：每次更新策略的步数

## 评估模型

训练完成后，最佳模型会自动保存为 `checkpoint-best.pth`。

查看训练日志：
```bash
type output_radar\log.txt
```

## 常见问题

### Q1: 如何确认预训练权重已加载？
A: 查看训练开始时的输出，应该看到：
```
Load ckpt from AdaptiveNN_ckpt.pth
Load state_dict by model_key = model
```

### Q2: 训练多久能看到效果？
A: 通常在5-10个epoch后就能看到明显的准确率提升。

### Q3: 如何从中断的训练继续？
A: 脚本支持自动恢复，会从 `output_radar/` 中的最新检查点继续。

### Q4: 如何调整训练轮数？
A: 修改 `train_radar.bat` 中的 `--epochs 50` 参数。

## 性能预期

基于预训练权重，在雷达CWD数据集上：
- 预期准确率：>90%（取决于数据质量和SNR）
- 训练时间：约2-3小时（50 epochs，RTX 2080Ti）
- Glance网络准确率：通常比整体准确率低5-10%

## 模型评估

训练完成后，使用评估脚本生成详细的SNR分析结果。

### 评估方法

#### 方法1：使用批处理脚本（推荐）
直接双击运行：
```
测试coatnet/测试coatnet/AdaptiveNN-main/AdaptiveNN-main/evaluate_radar.bat
```

#### 方法2：命令行运行
```bash
cd 测试coatnet/测试coatnet/AdaptiveNN-main/AdaptiveNN-main
call conda activate adaptivenn
python evaluate_adaptivenn_snr.py --checkpoint ./output_radar/checkpoint-best.pth
```

### 评估输出

评估脚本会生成以下文件：

#### 1. adaptivenn_snr_results.csv
包含每个SNR级别的识别率数据：
```csv
Model Information
Model,AdaptiveNN
Parameters,XX.XXM
FLOPs,XX.XXG
Overall Accuracy,XX.XX%

SNR(dB),Accuracy(%)
-10,XX.XX
-8,XX.XX
...
10,XX.XX
```

#### 2. adaptivenn_snr_curve.png
SNR-准确率曲线图，直观显示模型在不同信噪比下的性能。

#### 3. adaptivenn_stats.csv
详细的模型统计信息：
- 模型名称
- 整体准确率
- 总参数量
- 可训练参数量
- FLOPs（浮点运算次数）
- 检查点路径

### 评估参数说明

可以通过命令行参数自定义评估：

```bash
python evaluate_adaptivenn_snr.py \
    --eval_data_path "测试数据路径" \
    --checkpoint "检查点路径" \
    --batch_size 64 \
    --output_prefix "输出文件前缀"
```

主要参数：
- `--eval_data_path`: 测试数据集路径（默认：../../CWDtest）
- `--checkpoint`: 模型检查点路径（默认：./output_radar/checkpoint-best.pth）
- `--batch_size`: 评估批次大小（默认：64）
- `--output_prefix`: 输出文件前缀（默认：adaptivenn）

### 结果解读

#### 整体准确率
所有SNR级别的平均识别准确率，反映模型的总体性能。

#### SNR曲线
- **横轴**：信噪比（dB），从-10dB到10dB
- **纵轴**：识别准确率（%）
- **趋势**：通常SNR越高，准确率越高
- **关键点**：
  - 低SNR（-10dB到-4dB）：噪声较大，识别难度高
  - 中SNR（-2dB到2dB）：性能快速提升
  - 高SNR（4dB到10dB）：接近最佳性能

#### 参数量和FLOPs
- **参数量**：模型的总参数数量，影响模型大小和内存占用
- **FLOPs**：浮点运算次数，反映计算复杂度
- AdaptiveNN通过动态选择关注区域，可以在保持高准确率的同时降低计算量

## 与其他模型比较

### 性能对比

将AdaptiveNN的结果与其他基线模型（ResNet, ViT）比较：

| 模型 | 参数量 | FLOPs | 整体准确率 | 低SNR性能 | 高SNR性能 |
|------|--------|-------|-----------|----------|----------|
| ResNet-50 | ~25M | ~4G | - | - | - |
| ViT-Base | ~86M | ~17G | - | - | - |
| AdaptiveNN | ~XX M | ~XX G | - | - | - |

### 优势分析

AdaptiveNN的主要优势：
1. **自适应关注**：通过强化学习动态选择重要区域
2. **计算效率**：Glance-Focus架构减少不必要的计算
3. **鲁棒性**：在低SNR条件下表现更稳定

## 下一步

训练和评估完成后，可以：
1. ✅ 分析不同SNR级别的性能（已通过评估脚本完成）
2. 可视化策略网络的关注区域
3. 与其他基线模型（ResNet, ViT）比较性能
4. 针对特定SNR范围进行微调
5. 分析混淆矩阵，了解哪些信号类型容易混淆

## 技术支持

如遇问题，请检查：
1. 数据集路径是否正确
2. 预训练权重文件是否存在
3. conda环境是否正确激活
4. GPU驱动和CUDA是否正常
5. 检查点文件是否存在（评估时）
