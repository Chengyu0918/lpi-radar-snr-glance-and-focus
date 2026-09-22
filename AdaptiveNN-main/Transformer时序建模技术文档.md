# Transformer时序建模 + 视角轨迹约束 技术文档

## 1. 算法概述

本版本在可微分AdaptiveNN的基础上进行了重大改进，引入了**Transformer时序建模**和**轨迹多样性约束**两个核心创新点。

### 1.1 核心改进

| 特性 | 原可微分版本 | Transformer时序版本 |
|------|-------------|-------------------|
| 历史建模 | 无显式历史建模 | Transformer编码token序列 |
| 决策依据 | 仅当前特征 | 完整历史上下文 |
| 位置约束 | 无 | L_div轨迹多样性约束 |
| Token表示 | 无 | 局部特征+位置编码 |

### 1.2 算法流程

```
输入图像 x
    ↓
Glance Net → F_global → Pool → s_0 (初始token)
    ↓
初始化序列 S = [s_0]
    ↓
循环 t = 1 to T:
    │
    ├─→ S → Transformer编码器 → H_t
    │
    ├─→ h_t = H_t[-1] (取最后token输出)
    │
    ├─→ h_t → Policy MLP → Z_t (注意力logits)
    │
    ├─→ Z_t → Softmax → A_t (注意力分布)
    │
    ├─→ A_t → Soft-argmax → (x_t, y_t) (位置坐标)
    │
    ├─→ (x_t, y_t) → STN → patch_t
    │
    ├─→ patch_t → Focus Net → F_local_t
    │
    ├─→ F_local_t + (x_t, y_t) → Token Generator → s_t
    │
    └─→ S ← S ∪ {s_t}
    ↓
最终分类: S → Transformer → Pool → Classifier → 类别
    ↓
轨迹约束: {(x_t, y_t)} → L_div
    ↓
总损失: L_total = L_cls + λ_div × L_div
```

## 2. 核心组件详解

### 2.1 Transformer序列编码器 (SequenceTransformer)

```python
class SequenceTransformer(nn.Module):
    """
    用于处理token序列的Transformer编码器
    
    结构:
    - 位置编码: 正弦编码 + 可学习编码
    - 多层Transformer编码器
    - 最终LayerNorm
    """
```

**关键特性:**
- **混合位置编码**: 结合固定正弦编码和可学习编码，既保持位置信息的稳定性，又允许模型学习任务特定的位置表示
- **Pre-LN结构**: 使用Pre-LayerNorm提高训练稳定性
- **DropPath正则化**: 防止过拟合

**参数配置:**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| num_layers | 3 | Transformer层数 |
| nhead | 8 | 注意力头数 |
| dim_feedforward | 1536 | FFN隐藏维度 |
| dropout | 0.1 | Dropout率 |
| drop_path | 0.1 | DropPath率 |

### 2.2 策略头 (PolicyHead)

```python
class PolicyHead(nn.Module):
    """
    从Transformer输出生成位置注意力分布
    
    流程:
    h_t → MLP → logits → Softmax → prob → Soft-argmax → (x, y)
    """
```

**Soft-argmax机制:**
```python
# 计算期望坐标
x_exp = Σ(prob * grid_x_coords)
y_exp = Σ(prob * grid_y_coords)
```

这种方式保证了:
1. **完全可微分**: 梯度可以从位置坐标反向传播到策略网络
2. **平滑输出**: 避免离散采样带来的高方差
3. **概率解释**: 注意力分布可以可视化和分析

### 2.3 Token生成器 (TokenGenerator)

```python
class TokenGenerator(nn.Module):
    """
    从局部特征和位置生成新token
    
    token = Fusion(Feature_proj(local_feat), Pos_encoder(position))
    """
```

**设计理念:**
- 局部特征提供"看到了什么"的信息
- 位置编码提供"在哪里看"的信息
- 融合层学习如何组合这两种信息

### 2.4 轨迹多样性正则化器 (DiversityRegularizer)

```python
class DiversityRegularizer(nn.Module):
    """
    计算轨迹多样性损失
    
    L_div = (1/T(T-1)) × Σ_{t≠t'} exp(-||p_t - p_t'||² / σ²)
    """
```

**数学原理:**
- 使用RBF核度量位置相似度
- 位置越近，相似度越高，损失越大
- 鼓励模型选择分散的观察位置

**σ参数的影响:**
| σ值 | 效果 |
|-----|------|
| 小 (0.1) | 只惩罚非常接近的位置 |
| 中 (0.3) | 平衡的多样性约束 |
| 大 (0.5) | 强烈鼓励位置分散 |

## 3. 损失函数

### 3.1 总损失组成

```
L_total = L_cls + λ_div × L_div + λ_entropy × L_entropy
```

### 3.2 分类损失 (L_cls)

```python
L_cls = L_reg_focus    # Focus网络正则化
      + L_glance       # Glance分类
      + L_focus        # 各step分类
      + L_final        # 最终分类
      + α × L_KD       # 知识蒸馏
```

### 3.3 轨迹多样性损失 (L_div)

```python
L_div = (1/T(T-1)) × Σ_{t≠t'} exp(-||p_t - p_t'||² / σ²)
```

**梯度分析:**
```
∂L_div/∂p_t = (2/σ²T(T-1)) × Σ_{t'≠t} (p_t - p_t') × exp(-||p_t - p_t'||² / σ²)
```

这个梯度会推动每个位置远离其他位置。

### 3.4 注意力熵正则化 (可选)

```python
L_entropy = -Σ_t H(A_t) = -Σ_t Σ_i A_t[i] × log(A_t[i])
```

鼓励注意力分布更加分散，增加探索性。

## 4. 训练配置

### 4.1 推荐超参数

```bash
# 基础训练参数
--batch_size 32
--epochs 100
--lr 5e-4
--min_lr 1e-6
--warmup_epochs 5
--weight_decay 0.05

# 模型结构参数
--seq_l 4                          # Focus步数
--glance_input_size 112            # Glance输入尺寸
--focus_patch_size 96              # Focus patch尺寸
--transformer_num_layers 3         # Transformer层数
--transformer_nhead 8              # 注意力头数

# 轨迹约束参数
--diversity_weight 0.1             # λ_div
--diversity_sigma 0.3              # σ

# 损失权重
--loss_reg_focus_net_weight 0.5
--kd_alpha 0.5
--kd_temp 4.0
```

### 4.2 参数调优建议

**diversity_weight (λ_div):**
- 太小 (< 0.05): 位置可能聚集
- 推荐 (0.1-0.2): 平衡分类和多样性
- 太大 (> 0.5): 可能影响分类性能

**diversity_sigma (σ):**
- 根据图像内容调整
- 对于细粒度任务，使用较小的σ
- 对于全局任务，使用较大的σ

**transformer_num_layers:**
- 2-3层通常足够
- 更多层可能导致过拟合

## 5. 与原版本的对比

### 5.1 架构对比

```
原可微分版本:
┌─────────────────────────────────────────┐
│  特征 → 策略网络 → 位置 → Focus → 分类  │
│         (无历史)                        │
└─────────────────────────────────────────┘

Transformer时序版本:
┌─────────────────────────────────────────────────────┐
│  Token序列 → Transformer → 策略头 → 位置 → Focus   │
│      ↑                                      │      │
│      └──────── Token生成器 ←────────────────┘      │
│                                                     │
│  + L_div轨迹约束                                    │
└─────────────────────────────────────────────────────┘
```

### 5.2 优势分析

| 方面 | 原版本 | Transformer版本 |
|------|--------|-----------------|
| 历史感知 | ❌ | ✅ 完整历史上下文 |
| 位置多样性 | ❌ 可能聚集 | ✅ L_div约束 |
| 长程依赖 | ❌ | ✅ 自注意力机制 |
| 可解释性 | 一般 | ✅ 注意力可视化 |
| 计算开销 | 较低 | 略高 |

## 6. 使用指南

### 6.1 训练

```bash
# Windows
train_radar_transformer_seq.bat

# Linux/Mac
python main_transformer_seq.py \
    --data_path ./CWDtrain \
    --eval_data_path ./CWDtest \
    --output_dir ./output_transformer_seq \
    --diversity_weight 0.1 \
    --diversity_sigma 0.3
```

### 6.2 评估

```bash
python main_transformer_seq.py \
    --eval True \
    --resume ./output_transformer_seq/checkpoint-best.pth
```

### 6.3 可视化

训练过程中会自动生成注意力可视化图像（每10个epoch），保存在输出目录中。

## 7. 实现细节

### 7.1 位置编码设计

```python
# 混合位置编码
pos_enc = sinusoidal_pe + learnable_pe

# 正弦编码提供稳定的位置信息
pe[:, 0::2] = sin(position * div_term)
pe[:, 1::2] = cos(position * div_term)

# 可学习编码允许任务适应
learnable_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
```

### 7.2 Token序列管理

```python
# 初始化
token_sequence = [s_0]  # 全局特征token

# 每步添加新token
for t in range(T):
    # ... 生成新token s_t
    token_sequence.append(s_t)

# 最终序列长度: T+1
```

### 7.3 梯度流

```
L_total
   │
   ├─→ L_cls ─→ 分类器 ─→ Transformer ─→ Token序列
   │                                         │
   │                                         ↓
   │                              Token生成器 ← Focus Net ← STN
   │                                                         │
   │                                                         ↓
   └─→ L_div ─────────────────────────────────────→ 位置坐标
                                                         │
                                                         ↓
                                              Soft-argmax ← 策略头
                                                              │
                                                              ↓
                                                        Transformer
```

## 8. 常见问题

### Q1: 位置仍然聚集怎么办？
A: 增加`diversity_weight`或减小`diversity_sigma`。

### Q2: 训练不稳定？
A: 
- 减小学习率
- 增加warmup epochs
- 使用梯度裁剪 (`--clip_grad 1.0`)

### Q3: 显存不足？
A: 
- 减小batch_size
- 减少transformer_num_layers
- 减少seq_l

### Q4: 如何选择seq_l？
A: 
- 简单任务: 2-3步
- 复杂任务: 4-6步
- 更多步数不一定更好，需要实验验证

## 9. 文件结构

```
AdaptiveNN-main/
├── models/
│   └── dynamic_deitS_transformer_seq.py  # 模型定义
├── engine_transformer_seq.py              # 训练引擎
├── main_transformer_seq.py                # 主训练脚本
├── train_radar_transformer_seq.bat        # Windows训练脚本
└── Transformer时序建模技术文档.md          # 本文档
```

## 10. 参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021
3. AdaptiveNN原论文
