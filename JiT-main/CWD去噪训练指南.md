# JiT-CWD 去噪模型训练指南

## 概述

本项目将 JiT (Just image Transformer) 模型改造为 **条件去噪模型**，用于对含噪 CWD (Choi-Williams Distribution) 时频图进行去噪。

### 核心修改

1. **输入接口修改**：`forward(x, t, y)` → `forward(x, t, cond)`
   - `x`: 当前扩散状态图 z_t
   - `t`: 时间步
   - `cond`: 条件图（含噪 CWD 图）

2. **通道数修改**：从 RGB 3通道改为单通道灰度图

3. **条件机制**：删除类别条件，改为图像条件（token concat 方案）

4. **模型规模**：针对单卡 4080 优化，提供 Tiny/Small/Base 三种规模

---

## 文件结构

```
JiT-main/
├── model_jit_cwd.py      # 修改后的模型架构
├── denoiser_cwd.py       # 去噪器封装类
├── dataset_cwd.py        # CWD数据集加载器
├── train_cwd.py          # 训练脚本
├── inference_cwd.py      # 推理脚本
├── run_train_cwd.bat     # Windows训练启动脚本
├── run_inference_cwd.bat # Windows推理启动脚本
└── CWD去噪训练指南.md    # 本文档
```

---

## 环境配置

### 1. 创建 conda 环境

```bash
conda env create -f environment.yaml
conda activate jit
```

### 2. 安装依赖（如果 environment.yaml 不可用）

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pillow tensorboard tqdm einops
```

---

## 数据准备

### 方案1：自监督训练（推荐，无需干净图像）

只需要含噪 CWD 图像，模型通过添加额外噪声进行自监督学习。

```
CWDtrain/
├── BPSK/
│   ├── BPSK_-10dB_1.png
│   ├── BPSK_-10dB_2.png
│   └── ...
├── Costas/
├── LFM/
└── ...（其他类别）
```

### 方案2：配对训练（推荐，使用高信噪比图像作为干净参考）

利用数据集中不同信噪比的图像进行配对训练：
- **含噪图像**：-20dB 到 0dB（低信噪比）
- **干净图像**：10dB（高信噪比，作为去噪目标）

#### 数据集结构

```
CWDtrain_noisy/          # 含噪图像 (SNR: -20dB 到 0dB)
├── BPSK/
│   ├── BPSK_-20dB_1.png
│   ├── BPSK_-18dB_1.png
│   ├── ...
│   └── BPSK_0dB_800.png
├── Costas/
├── LFM/
└── ...（共16个类别）

CWDtrain_clean/          # 干净图像 (实际是10dB图像，与含噪图同名配对)
├── BPSK/
│   ├── BPSK_-20dB_1.png  # 实际内容是 BPSK_10dB_1.png
│   ├── BPSK_-18dB_1.png  # 实际内容是 BPSK_10dB_1.png
│   └── ...
└── ...
```

#### 准备配对数据集

运行以下脚本自动创建配对数据集：

```bash
# 在项目根目录运行
python run_prepare_paired_dataset_v2.py
```

脚本会：
1. 从 `CWDtrain/` 读取原始数据
2. 将 -20dB 到 0dB 的图像复制到 `CWDtrain_noisy/`
3. 将对应的 10dB 图像复制到 `CWDtrain_clean/`（文件名与含噪图相同）
4. 总共创建约 140,800 个配对（16类 × 800样本 × 11个含噪SNR级别）

#### 配对原理

相同编号的图像代表同一信号在不同信噪比下的版本：
- `BPSK_-10dB_1.png` 和 `BPSK_10dB_1.png` 是同一信号
- 训练时，模型学习从低SNR图像恢复到高SNR图像

---

## 训练

### 方式1：配对训练（推荐）

使用配对数据集进行训练，双击运行 `run_train_paired.bat`，或：

```bash
cd 测试coatnet/测试coatnet/JiT-main

python train_cwd.py \
    --model JiT-CWD-Tiny \
    --data_path ../../../CWDtrain_noisy \
    --clean_path ../../../CWDtrain_clean \
    --train_mode paired \
    --output_dir ./output_cwd_paired \
    --img_size 224 \
    --batch_size 8 \
    --epochs 100
```

### 方式2：自监督训练

只使用含噪图像，双击运行 `run_train_cwd.bat`，或：

```bash
cd 测试coatnet/测试coatnet/JiT-main

python train_cwd.py \
    --model JiT-CWD-Tiny \
    --data_path ../../CWDtrain \
    --train_mode self_supervised \
    --added_noise_level 0.1 \
    --output_dir ./output_cwd \
    --img_size 224 \
    --batch_size 8 \
    --epochs 100
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | JiT-CWD-Tiny | 模型大小：Tiny(6层)/Small(8层)/Base(12层) |
| `--img_size` | 224 | 图像尺寸，CWD图为224x224 |
| `--batch_size` | 8 | 批次大小，4080建议8-16 |
| `--epochs` | 100 | 训练轮数 |
| `--lr` | 1e-4 | 学习率 |
| `--train_mode` | self_supervised | 训练模式：self_supervised/paired |
| `--added_noise_level` | 0.1 | 自监督模式下添加的噪声级别 |
| `--sampling_method` | heun | ODE求解器：euler/heun |
| `--num_sampling_steps` | 50 | 采样步数 |

### 模型规模对比

| 模型 | 层数 | 隐藏维度 | 参数量 | 显存占用(224x224) |
|------|------|----------|--------|-------------------|
| JiT-CWD-Tiny | 6 | 384 | ~5M | ~4GB |
| JiT-CWD-Small | 8 | 512 | ~15M | ~6GB |
| JiT-CWD-Base | 12 | 768 | ~50M | ~12GB |

### 显存优化建议

如果显存不足，可以尝试：

1. **减小图像尺寸**：`--img_size 64`
2. **减小批次大小**：`--batch_size 4`
3. **使用更小的模型**：`--model JiT-CWD-Tiny`
4. **启用混合精度**：`--use_amp`

---

## 推理

### 快速开始

训练完成后，双击运行 `run_inference_cwd.bat`，或：

```bash
python inference_cwd.py \
    --checkpoint ./output_cwd/checkpoint-final.pth \
    --input ../../CWDtest \
    --output ./denoised_output \
    --model JiT-CWD-Tiny \
    --img_size 224
```

### 单张图像去噪

```bash
python inference_cwd.py \
    --checkpoint ./output_cwd/checkpoint-final.pth \
    --input path/to/noisy_image.png \
    --output path/to/denoised_image.png
```

---

## 训练监控

使用 TensorBoard 查看训练过程：

```bash
tensorboard --logdir ./output_cwd
```

然后在浏览器打开 http://localhost:6006

---

## 技术细节

### 扩散过程

采用 Flow Matching 框架：
- 前向过程：`z_t = t * x + (1 - t) * ε`
- 速度场：`v = x - ε`
- 损失函数：`L = MSE(v_pred, v)`

### 条件注入

使用 Token Concat 方案：
1. 对 z_t 和 cond 分别做 patch embedding
2. 在 token 维度拼接：`[cond_tokens, x_tokens]`
3. Transformer 处理拼接后的序列
4. 只取后 N 个 token 输出

### ODE 采样

支持两种求解器：
- **Euler**：一阶精度，速度快
- **Heun**：二阶精度，质量更好

---

## 常见问题

### Q1: 显存不足 (CUDA out of memory)

A: 尝试以下方法：
```bash
python train_cwd.py --img_size 64 --batch_size 4 --model JiT-CWD-Tiny
```

### Q2: 训练损失不下降

A: 可能原因：
1. 学习率过大/过小，尝试调整 `--lr`
2. 噪声级别不合适，调整 `--added_noise_level`
3. 数据量太少，增加训练数据

### Q3: 去噪效果不好

A: 尝试：
1. 增加训练轮数
2. 增加采样步数 `--num_sampling_steps 100`
3. 使用更大的模型

### Q4: 如何恢复训练

A: 使用 `--resume` 参数：
```bash
python train_cwd.py --resume ./output_cwd
```

---

## 代码修改说明

### 1. model_jit_cwd.py

主要修改：
- 删除 `LabelEmbedder` 类
- 添加 `cond_embedder` 用于条件图嵌入
- 修改 `forward` 接口支持图像条件
- 修改 RoPE 支持 2N 长度序列

### 2. denoiser_cwd.py

主要修改：
- `forward(clean_cwd, noisy_cwd)` 训练接口
- `denoise(noisy_cwd)` 推理接口
- 删除 CFG 相关代码（不需要类别引导）

### 3. dataset_cwd.py

新增：
- `CWDPairedDataset`: 配对数据集
- `CWDSelfSupervisedDataset`: 自监督数据集
- `CWDInferenceDataset`: 推理数据集

---

## 模型评估

训练完成后，可以使用评估脚本对模型性能进行全面评估。

### 评估指标

评估脚本计算以下指标：
- **PSNR (峰值信噪比)**：衡量去噪后图像与干净参考图像的相似度，单位为dB，越高越好
- **SSIM (结构相似性)**：衡量图像结构的保持程度，范围0-1，越高越好
- **MSE (均方误差)**：基础误差指标，越低越好

### 运行评估

双击运行 `run_evaluate_cwd.bat`，或手动执行：

```bash
python evaluate_cwd.py \
    --checkpoint ./output_cwd_paired/checkpoint-last.pth \
    --noisy_dir ../../../CWDtrain_noisy \
    --clean_dir ../../../CWDtrain_clean \
    --output ./evaluation_results \
    --model JiT-CWD-Tiny \
    --num_samples 50 \
    --num_vis_samples 5
```

### 评估参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | 必需 | 模型checkpoint路径 |
| `--noisy_dir` | 必需 | 含噪图像目录 |
| `--clean_dir` | 必需 | 干净参考图像目录 |
| `--output` | ./evaluation_results | 评估结果输出目录 |
| `--num_samples` | 100 | 每个类别评估的样本数，-1表示全部 |
| `--num_vis_samples` | 5 | 每个类别可视化的样本数 |
| `--num_sampling_steps` | 50 | 去噪采样步数 |

### 评估输出

评估完成后，在输出目录中会生成：

1. **evaluation_summary.json**：评估结果摘要
   - 总体平均指标
   - 按类别统计
   - 按SNR级别统计

2. **evaluation_details.csv**：每个样本的详细评估结果

3. **visualizations/**：可视化图像
   - `{类别}_comparison.png`：每个类别的去噪对比图
   - `snr_vs_psnr.png`：SNR vs PSNR曲线
   - `psnr_improvement.png`：PSNR改进柱状图

### 评估结果解读

```
                     Overall Results                      
------------------------------------------------------------
Total samples evaluated: 800

Metric               Noisy           Denoised        Improvement    
------------------------------------------------------------
PSNR (dB)            15.23           22.45           +7.22
SSIM                 0.4521          0.7834          +0.3313
```

- **PSNR改进 > 3dB**：表示去噪效果显著
- **SSIM > 0.7**：表示结构保持良好
- **按SNR分析**：低SNR（如-20dB）通常改进更大

---

## 参考资料

- [JiT 原论文](https://arxiv.org/abs/2511.13720)
- [Flow Matching](https://arxiv.org/abs/2210.02747)
- [DiT](https://github.com/facebookresearch/DiT)

---

## 带类别条件的去噪模型（推荐）

### 问题背景

原始无条件模型在去噪时可能出现**类别混淆**问题：
- BPSK信号去噪后可能看起来像P1或其他信号
- 原因：模型学习了16种信号的混合分布，去噪时会生成任意类型的特征

### 解决方案

添加**类别条件**，使用 **Classifier-Free Guidance (CFG)** 技术：
- 训练时：10%概率丢弃类别标签（用于无条件训练）
- 推理时：同时计算有条件和无条件预测，通过CFG引导生成

### 文件说明

```
JiT-main/
├── model_jit_cwd_class.py      # 带类别条件的模型架构
├── denoiser_cwd_class.py       # 带类别条件的去噪器
├── train_cwd_class.py          # 带类别条件的训练脚本
├── inference_cwd_class.py      # 带类别条件的推理脚本
├── run_train_cwd_class.bat     # 训练批处理
└── run_inference_cwd_class.bat # 推理批处理
```

### 训练带类别条件的模型

双击运行 `run_train_cwd_class.bat`，或：

```bash
python train_cwd_class.py \
    --data_path ../../../CWDtrain_noisy \
    --clean_path ../../../CWDtrain_clean \
    --output_dir ./output_cwd_class \
    --model JiT-CWD-Class-Tiny \
    --img_size 224 \
    --batch_size 8 \
    --epochs 100 \
    --class_dropout_prob 0.1
```

### 推理（带类别引导）

双击运行 `run_inference_cwd_class.bat`，或：

```bash
python inference_cwd_class.py \
    --checkpoint ./output_cwd_class/best_model.pth \
    --input ../../../CWDtrain_noisy \
    --output ../../../CWDtrain_denoised_class \
    --cfg_scale 2.0
```

### CFG强度参数

`--cfg_scale` 控制类别引导的强度：
- `1.0`：无引导（等同于无条件模型）
- `2.0`：推荐值，平衡去噪质量和类别保持
- `3.0+`：更强的类别约束，但可能过度平滑

### 类别自动推断

推理时，脚本会自动从文件路径或文件名推断类别：
- 从路径中的文件夹名（如 `BPSK/`）
- 从文件名前缀（如 `BPSK_-10dB_1.png`）

也可以手动指定类别：
```bash
python inference_cwd_class.py \
    --checkpoint ./output_cwd_class/best_model.pth \
    --input path/to/image.png \
    --output path/to/output.png \
    --class_label BPSK
```

### 支持的16个类别

| 索引 | 类别名 | 索引 | 类别名 |
|------|--------|------|--------|
| 0 | BPSK | 8 | P1 |
| 1 | Costas | 9 | P2 |
| 2 | CP | 10 | P3 |
| 3 | Frank | 11 | P4 |
| 4 | FSK4_Baker5 | 12 | T1 |
| 5 | FSK4_LFM | 13 | T2 |
| 6 | LFM | 14 | T3 |
| 7 | NLFM | 15 | T4 |
