"""
AdaptiveNN系列模型 - 综合训练与评估脚本
包含：AdaptiveNN (PPO)、Differentiable、Transformer时序版本
SNR范围：-20dB 到 0dB，步长2dB
生成与VIT/ResNet相同格式的图表
"""
import os
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import csv
import argparse
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 添加models路径
sys.path.insert(0, './models')

from timm.models import create_model
import models.dynamic_deitS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别名称（16个信号类型）
CLASS_NAMES = ['BPSK', 'CP', 'Costas', 'FSK4_Baker5', 'FSK4_LFM', 'Frank',
               'LFM', 'NLFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']

# SNR级别：-20dB 到 0dB，步长2dB
SNR_LEVELS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0]

# 数据路径（使用绝对路径）
TRAIN_DATA_PATH = 'd:/project/测试coatnet (2)/CWDtrain'
TEST_DATA_PATH = 'd:/project/测试coatnet (2)/CWDtest'


class RadarDataset(Dataset):
    """雷达信号数据集"""
    def __init__(self, root_dir, transform=None, snr_filter=None):
        self.root_dir = root_dir
        self.transform = transform
        self.snr_filter = snr_filter
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
        
        self._load_samples()
    
    def _load_samples(self):
        """加载样本"""
        for class_name in CLASS_NAMES:
            class_path = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_path):
                if not filename.endswith('.png'):
                    continue
                
                # 提取SNR信息
                match = re.search(r'_(-?\d+)dB_', filename)
                if match:
                    snr = int(match.group(1))
                    
                    # 如果指定了SNR过滤，只加载特定SNR的数据
                    if self.snr_filter is not None and snr not in self.snr_filter:
                        continue
                    
                    filepath = os.path.join(class_path, filename)
                    self.samples.append((filepath, class_idx, snr))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label, snr = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, snr


def get_transforms(is_train=True):
    """获取数据转换"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def create_adaptivenn_model(args, model_type='ppo'):
    """创建AdaptiveNN模型"""
    print(f"创建AdaptiveNN模型 (类型: {model_type})...")
    
    model = create_model(
        args.model,
        seq_len=args.seq_l,
        feature_in_chans=args.feature_in_chans,
        policy_net_hidden_chans=args.policy_net_hidden_chans,
        policy_net_kernel_size=args.policy_net_kernel_size,
        recover_n=args.recover_n,
        remaining_blocks=args.remaining_blocks,
        glance_input_size=args.glance_input_size,
        glance_net_depth=args.glance_net_depth,
        glance_net_mlp_ratio=args.glance_net_mlp_ratio,
        glance_net_drop_path=args.glance_net_drop_path,
        focus_patch_size=args.focus_patch_size,
        focus_net_reg_size=args.focus_net_reg_size,
        focus_net_depth=args.focus_net_depth,
        focus_net_mlp_ratio=args.focus_net_mlp_ratio,
        focus_net_drop_path=args.focus_net_drop_path,
        multi_cls_drop_path=args.multi_cls_drop_path,
        pretrained=False,
        num_classes=args.nb_classes,
    )
    
    return model


def load_checkpoint(model, checkpoint_path):
    """加载检查点"""
    print(f"加载检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠ 检查点不存在: {checkpoint_path}")
        return model, False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 尝试不同的键名
    state_dict = None
    for key in ['model', 'model_ema', 'state_dict']:
        if key in checkpoint:
            state_dict = checkpoint[key]
            print(f"✓ 使用键名 '{key}' 加载模型")
            break
    
    if state_dict is None:
        state_dict = checkpoint
        print("✓ 直接使用checkpoint作为state_dict")
    
    # 处理DDP模型的键名
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("✓ 模型权重加载成功")
        return model, True
    except Exception as e:
        print(f"⚠ 加载失败: {e}")
        return model, False


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # AdaptiveNN返回字典格式的输出
            outputs = model(images)
            
            # 处理不同的输出格式
            if isinstance(outputs, dict):
                if 'x_focus' in outputs and len(outputs['x_focus']) > 0:
                    logits = outputs['x_focus'][-1]
                elif 'x_glance' in outputs and len(outputs['x_glance']) > 0:
                    logits = outputs['x_glance'][0]
                else:
                    raise ValueError("输出字典中没有找到有效的预测结果")
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                logits = outputs
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def evaluate_by_snr(model, test_data_path, device, batch_size=64):
    """按SNR评估模型（总体准确率）"""
    snr_accuracies = {}
    transform = get_transforms(is_train=False)
    
    for snr in SNR_LEVELS:
        temp_dataset = RadarDataset(test_data_path, transform=transform, snr_filter=[snr])
        
        if len(temp_dataset) == 0:
            continue
        
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        
        accuracy = evaluate_model(model, temp_loader, device)
        snr_accuracies[snr] = accuracy
        print(f"SNR {snr:3d}dB: 准确率 {accuracy:.2f}%")
    
    return snr_accuracies


def evaluate_by_snr_and_class(model, test_data_path, device, batch_size=64):
    """按SNR和类别详细评估模型"""
    transform = get_transforms(is_train=False)
    snr_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for snr in SNR_LEVELS:
        temp_dataset = RadarDataset(test_data_path, transform=transform, snr_filter=[snr])
        
        if len(temp_dataset) == 0:
            continue
        
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=0, pin_memory=True)
        
        model.eval()
        with torch.no_grad():
            for images, labels, snrs in temp_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                
                if isinstance(outputs, dict):
                    if 'x_focus' in outputs and len(outputs['x_focus']) > 0:
                        logits = outputs['x_focus'][-1]
                    elif 'x_glance' in outputs and len(outputs['x_glance']) > 0:
                        logits = outputs['x_glance'][0]
                    else:
                        raise ValueError("输出字典中没有找到有效的预测结果")
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[-1]
                else:
                    logits = outputs
                
                _, predicted = logits.max(1)
                
                for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    class_name = CLASS_NAMES[label]
                    snr_class_stats[snr][class_name]['total'] += 1
                    if pred == label:
                        snr_class_stats[snr][class_name]['correct'] += 1
    
    return snr_class_stats


def get_predictions_and_labels(model, test_data_path, device, batch_size=64, snr_filter=None):
    """获取模型预测结果和真实标签"""
    transform = get_transforms(is_train=False)
    test_dataset = RadarDataset(test_data_path, transform=transform, snr_filter=snr_filter)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            
            if isinstance(outputs, dict):
                if 'x_focus' in outputs and len(outputs['x_focus']) > 0:
                    logits = outputs['x_focus'][-1]
                elif 'x_glance' in outputs and len(outputs['x_glance']) > 0:
                    logits = outputs['x_glance'][0]
                else:
                    logits = outputs
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                logits = outputs
            
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(model, test_data_path, device, model_name, save_path, batch_size=64):
    """绘制混淆矩阵"""
    print(f"生成{model_name}混淆矩阵...")
    
    preds, labels = get_predictions_and_labels(model, test_data_path, device, batch_size)
    
    cm = confusion_matrix(labels, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar_kws={'label': '准确率 (%)'})
    plt.xlabel('预测类别', fontsize=12)
    plt.ylabel('真实类别', fontsize=12)
    plt.title(f'{model_name} 混淆矩阵 (SNR: -20dB ~ 0dB)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"混淆矩阵已保存到: {save_path}")
    plt.close()
    
    return cm, cm_normalized


def plot_confusion_matrix_by_snr(model, test_data_path, device, model_name, save_dir, batch_size=64):
    """按不同SNR绘制混淆矩阵"""
    print(f"生成{model_name}各SNR下的混淆矩阵...")
    
    # 选择几个代表性的SNR级别（-20dB到0dB范围内）
    selected_snrs = [-20, -14, -8, 0]
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()
    
    for idx, snr in enumerate(selected_snrs):
        preds, labels = get_predictions_and_labels(model, test_data_path, device, batch_size, snr_filter=[snr])
        
        if len(preds) == 0:
            continue
        
        cm = confusion_matrix(labels, preds, labels=range(len(CLASS_NAMES)))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10) * 100
        
        sns.heatmap(cm_normalized, annot=True, fmt='.0f', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                    ax=axes[idx], cbar_kws={'label': '%'}, annot_kws={'size': 8})
        axes[idx].set_xlabel('预测类别', fontsize=10)
        axes[idx].set_ylabel('真实类别', fontsize=10)
        axes[idx].set_title(f'SNR = {snr}dB', fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)
    
    plt.suptitle(f'{model_name} 不同SNR下的混淆矩阵', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name.lower().replace(" ", "_").replace("-", "_")}_confusion_matrix_by_snr.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"各SNR混淆矩阵已保存到: {save_path}")
    plt.close()


def save_snr_class_results(snr_class_stats, model_name, save_path):
    """保存SNR-类别准确率结果到CSV"""
    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        header = ['SNR(dB)'] + CLASS_NAMES + ['Average']
        writer.writerow(header)
        
        for snr in SNR_LEVELS:
            if snr not in snr_class_stats:
                continue
            
            row = [snr]
            accs = []
            for class_name in CLASS_NAMES:
                if class_name in snr_class_stats[snr]:
                    stats = snr_class_stats[snr][class_name]
                    if stats['total'] > 0:
                        acc = 100.0 * stats['correct'] / stats['total']
                        row.append(f'{acc:.2f}')
                        accs.append(acc)
                    else:
                        row.append('N/A')
                else:
                    row.append('N/A')
            
            if accs:
                row.append(f'{np.mean(accs):.2f}')
            else:
                row.append('N/A')
            
            writer.writerow(row)
    
    print(f"{model_name} SNR-类别准确率已保存到: {save_path}")


def plot_class_snr_curves(snr_class_stats, model_name, save_path):
    """绘制各类别的SNR-准确率曲线（16个信号分别的结果图）"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    axes = axes.flatten()
    
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    
    for idx, class_name in enumerate(CLASS_NAMES):
        ax = axes[idx]
        snrs = []
        accs = []
        
        for snr in SNR_LEVELS:
            if snr in snr_class_stats and class_name in snr_class_stats[snr]:
                stats = snr_class_stats[snr][class_name]
                if stats['total'] > 0:
                    snrs.append(snr)
                    accs.append(100.0 * stats['correct'] / stats['total'])
        
        if snrs:
            ax.plot(snrs, accs, 'o-', linewidth=2, markersize=6, color=colors[idx])
            ax.fill_between(snrs, accs, alpha=0.2, color=colors[idx])
        
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_ylabel('准确率 (%)', fontsize=10)
        ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        ax.set_xticks(SNR_LEVELS[::2])
        
        if accs:
            avg_acc = np.mean(accs)
            ax.axhline(y=avg_acc, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.text(max(snrs), avg_acc + 2, f'Avg: {avg_acc:.1f}%', fontsize=8, color='red')
    
    plt.suptitle(f'{model_name} 各类别信号在不同SNR下的识别准确率', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{model_name} 各类别曲线图已保存到: {save_path}")
    plt.close()


def plot_all_classes_in_one(snr_class_stats, model_name, save_path):
    """在一张图中绘制所有16个类别的SNR-准确率曲线"""
    plt.figure(figsize=(14, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', 'h', 'D', 'd', '*', 'X', 'P', 'H', '8', '+']
    
    for idx, class_name in enumerate(CLASS_NAMES):
        snrs = []
        accs = []
        
        for snr in SNR_LEVELS:
            if snr in snr_class_stats and class_name in snr_class_stats[snr]:
                stats = snr_class_stats[snr][class_name]
                if stats['total'] > 0:
                    snrs.append(snr)
                    accs.append(100.0 * stats['correct'] / stats['total'])
        
        if snrs:
            plt.plot(snrs, accs, marker=markers[idx], linewidth=1.5, markersize=5,
                    color=colors[idx], label=class_name, alpha=0.8)
    
    plt.xlabel('信噪比 (dB)', fontsize=14)
    plt.ylabel('识别准确率 (%)', fontsize=14)
    plt.title(f'{model_name} 16类信号在不同SNR下的识别准确率', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, ncol=1)
    plt.xticks(SNR_LEVELS)
    plt.ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{model_name} 所有类别曲线图已保存到: {save_path}")
    plt.close()


def plot_heatmap(snr_class_stats, model_name, save_path):
    """绘制SNR-类别准确率热力图"""
    data = np.zeros((len(SNR_LEVELS), len(CLASS_NAMES)))
    
    for i, snr in enumerate(SNR_LEVELS):
        for j, class_name in enumerate(CLASS_NAMES):
            if snr in snr_class_stats and class_name in snr_class_stats[snr]:
                stats = snr_class_stats[snr][class_name]
                if stats['total'] > 0:
                    data[i, j] = 100.0 * stats['correct'] / stats['total']
    
    plt.figure(figsize=(16, 10))
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn',
                xticklabels=CLASS_NAMES, yticklabels=[f'{snr}dB' for snr in SNR_LEVELS],
                cbar_kws={'label': '准确率 (%)'}, vmin=0, vmax=100)
    plt.xlabel('信号类别', fontsize=12)
    plt.ylabel('信噪比', fontsize=12)
    plt.title(f'{model_name} SNR-类别准确率热力图', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"{model_name} 热力图已保存到: {save_path}")
    plt.close()


def plot_comparison_curves(all_results, save_dir):
    """绘制所有模型的对比曲线"""
    # 1. 总体对比曲线
    plt.figure(figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'v', 'D']
    
    for idx, (model_name, snr_results) in enumerate(all_results.items()):
        snrs = sorted(snr_results.keys())
        accs = [snr_results[snr] for snr in snrs]
        
        plt.plot(snrs, accs, marker=markers[idx % len(markers)], linewidth=2, markersize=8,
                label=model_name, color=colors[idx % len(colors)])
    
    plt.xlabel('信噪比 (dB)', fontsize=14)
    plt.ylabel('识别准确率 (%)', fontsize=14)
    plt.title('AdaptiveNN系列模型 不同SNR下的识别准确率对比', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(SNR_LEVELS)
    plt.ylim([0, 105])
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'adaptivenn_comparison_overall.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"总体对比曲线已保存到: {save_path}")
    plt.close()


def get_default_args():
    """获取默认参数"""
    class Args:
        # 模型参数
        model = 'dynamic_deitS'
        nb_classes = 16
        input_size = 224
        
        # AdaptiveNN特定参数
        feature_in_chans = 384
        policy_net_hidden_chans = 128
        policy_net_kernel_size = 3
        recover_n = 5
        remaining_blocks = 4
        
        glance_input_size = 112
        glance_net_depth = 12
        glance_net_mlp_ratio = 4
        glance_net_drop_path = 0.1
        
        focus_patch_size = 112
        focus_net_reg_size = 160
        focus_net_depth = 12
        focus_net_mlp_ratio = 4
        focus_net_drop_path = 0.1
        
        multi_cls_drop_path = 0.1
        seq_l = 4
        
        # 数据参数
        batch_size = 64
    
    return Args()


def evaluate_single_model(model_name, checkpoint_path, args, save_dir):
    """评估单个模型"""
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"{'='*60}")
    
    # 创建模型
    model = create_adaptivenn_model(args)
    
    # 加载检查点
    model, success = load_checkpoint(model, checkpoint_path)
    
    if not success:
        print(f"⚠ 跳过模型 {model_name}（检查点加载失败）")
        return None, None
    
    model = model.to(device)
    model.eval()
    
    # 按SNR评估
    print(f"\n按SNR评估{model_name}...")
    snr_results = evaluate_by_snr(model, TEST_DATA_PATH, device, args.batch_size)
    
    # 按SNR和类别详细评估
    print(f"\n详细评估{model_name}...")
    snr_class_results = evaluate_by_snr_and_class(model, TEST_DATA_PATH, device, args.batch_size)
    
    # 生成文件名前缀
    prefix = model_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
    
    # 保存结果
    save_snr_class_results(snr_class_results, model_name, os.path.join(save_dir, f'{prefix}_snr_class_accuracy.csv'))
    
    # 绘制图表
    plot_class_snr_curves(snr_class_results, model_name, os.path.join(save_dir, f'{prefix}_class_snr_curves.png'))
    plot_all_classes_in_one(snr_class_results, model_name, os.path.join(save_dir, f'{prefix}_all_classes_curve.png'))
    plot_heatmap(snr_class_results, model_name, os.path.join(save_dir, f'{prefix}_snr_class_heatmap.png'))
    plot_confusion_matrix(model, TEST_DATA_PATH, device, model_name, os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), args.batch_size)
    plot_confusion_matrix_by_snr(model, TEST_DATA_PATH, device, model_name, save_dir, args.batch_size)
    
    return snr_results, snr_class_results


def main():
    print("=" * 60)
    print("AdaptiveNN系列模型 - 综合评估")
    print("SNR范围: -20dB ~ 0dB")
    print("=" * 60)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 检查数据路径
    if not os.path.exists(TEST_DATA_PATH):
        print(f"\n✗ 错误：测试数据路径不存在: {TEST_DATA_PATH}")
        return
    
    # 创建输出目录
    save_dir = './output_evaluation_extended'
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取默认参数
    args = get_default_args()
    
    # 定义要评估的模型和检查点
    models_to_evaluate = {
        'AdaptiveNN (PPO)': './output_radar_fresh/checkpoint-best.pth',
        'AdaptiveNN (Differentiable)': './output_radar_differentiable/checkpoint-best.pth',
        'AdaptiveNN (Transformer)': './transformer_seq_version/output_radar_transformer_seq/checkpoint-best.pth',
    }
    
    # 存储所有结果
    all_snr_results = {}
    all_snr_class_results = {}
    
    # 评估每个模型
    for model_name, checkpoint_path in models_to_evaluate.items():
        snr_results, snr_class_results = evaluate_single_model(model_name, checkpoint_path, args, save_dir)
        
        if snr_results is not None:
            all_snr_results[model_name] = snr_results
            all_snr_class_results[model_name] = snr_class_results
    
    # 绘制对比曲线
    if len(all_snr_results) > 0:
        print("\n生成对比曲线图...")
        plot_comparison_curves(all_snr_results, save_dir)
        
        # 保存总体对比数据
        with open(os.path.join(save_dir, 'comparison_results.csv'), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['SNR(dB)'] + list(all_snr_results.keys())
            writer.writerow(header)
            
            for snr in SNR_LEVELS:
                row = [snr]
                for model_name in all_snr_results.keys():
                    if snr in all_snr_results[model_name]:
                        row.append(f'{all_snr_results[model_name][snr]:.2f}')
                    else:
                        row.append('N/A')
                writer.writerow(row)
        
        print(f"对比结果已保存到: {os.path.join(save_dir, 'comparison_results.csv')}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"\n评估的模型数量: {len(all_snr_results)}")
    print(f"SNR范围: {SNR_LEVELS[0]}dB ~ {SNR_LEVELS[-1]}dB")
    print(f"\n所有结果保存在: {save_dir}")
    print("\n生成的文件:")
    for model_name in all_snr_results.keys():
        prefix = model_name.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '')
        print(f"\n  {model_name}:")
        print(f"    - {prefix}_class_snr_curves.png (16类信号分别结果)")
        print(f"    - {prefix}_all_classes_curve.png (所有类别在一张图)")
        print(f"    - {prefix}_snr_class_heatmap.png (热力图)")
        print(f"    - {prefix}_confusion_matrix.png (混淆矩阵)")
        print(f"    - {prefix}_confusion_matrix_by_snr.png (各SNR混淆矩阵)")
        print(f"    - {prefix}_snr_class_accuracy.csv (准确率数据)")
    
    if len(all_snr_results) > 1:
        print(f"\n  对比图:")
        print(f"    - adaptivenn_comparison_overall.png")
        print(f"    - comparison_results.csv")


if __name__ == '__main__':
    main()
