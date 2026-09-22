"""
AdaptiveNN模型 - 按SNR评估脚本
生成不同SNR下的识别率曲线和详细统计
"""
import os
import re
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import csv
import argparse

# 添加models路径
sys.path.insert(0, './models')

from timm.models import create_model
import models.dynamic_deitS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别名称
CLASS_NAMES = ['BPSK', 'CP', 'Costas', 'FSK4_Baker5', 'FSK4_LFM', 'Frank',
               'LFM', 'NLFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']

# SNR级别
SNR_LEVELS = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]


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


def get_transforms():
    """获取数据转换（评估用）"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def create_adaptivenn_model(args):
    """创建AdaptiveNN模型"""
    print("创建AdaptiveNN模型...")
    
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
    print(f"\n加载检查点: {checkpoint_path}")
    
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
    
    model.load_state_dict(new_state_dict, strict=True)
    print("✓ 模型权重加载成功")
    
    return model


def count_parameters(model):
    """计算模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def format_number(num):
    """格式化数字（转换为M或G）"""
    if num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def calculate_flops(model, input_size=(1, 3, 224, 224)):
    """计算模型的FLOPs"""
    try:
        from thop import profile, clever_format
        model.eval()
        input_tensor = torch.randn(input_size).to(next(model.parameters()).device)
        
        # AdaptiveNN需要特殊处理，因为它有多个输入
        # 这里简化处理，只计算主要的前向传播
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)
        flops_str, params_str = clever_format([flops, params], "%.3f")
        return flops, params, flops_str, params_str
    except Exception as e:
        print(f"⚠ FLOPs计算失败: {e}")
        print("  使用参数量估算...")
        total_params, _ = count_parameters(model)
        params_str = format_number(total_params)
        return 0, total_params, "N/A", params_str


def evaluate_model(model, test_loader, device):
    """评估模型（整体准确率）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='评估中'):
            images, labels = images.to(device), labels.to(device)
            
            # AdaptiveNN返回字典格式的输出
            outputs = model(images)
            
            # 处理不同的输出格式
            if isinstance(outputs, dict):
                # 优先使用最后的focus输出，如果没有则使用glance输出
                if 'x_focus' in outputs and len(outputs['x_focus']) > 0:
                    logits = outputs['x_focus'][-1]  # 取最后一个focus输出
                elif 'x_glance' in outputs and len(outputs['x_glance']) > 0:
                    logits = outputs['x_glance'][0]  # 使用glance输出
                else:
                    raise ValueError("输出字典中没有找到有效的预测结果")
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[-1]  # 取最后的融合输出
            else:
                logits = outputs  # 直接使用tensor输出
            
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total


def evaluate_by_snr(model, test_data_path, device, batch_size=64):
    """按SNR评估模型（总体准确率）"""
    snr_accuracies = {}
    
    print("\n" + "=" * 60)
    print("按SNR评估AdaptiveNN模型")
    print("=" * 60)
    
    transform = get_transforms()
    
    for snr in SNR_LEVELS:
        # 创建临时数据集
        temp_dataset = RadarDataset(test_data_path,
                                    transform=transform,
                                    snr_filter=[snr])
        
        if len(temp_dataset) == 0:
            print(f"SNR {snr:3d}dB: 无数据")
            continue
        
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
        
        accuracy = evaluate_model(model, temp_loader, device)
        snr_accuracies[snr] = accuracy
        print(f"SNR {snr:3d}dB: 准确率 {accuracy:.2f}% (样本数: {len(temp_dataset)})")
    
    print("=" * 60)
    return snr_accuracies


def evaluate_by_snr_and_class(model, test_data_path, device, batch_size=64):
    """按SNR和类别详细评估模型"""
    from collections import defaultdict
    
    print("\n" + "=" * 80)
    print("按SNR和类别详细评估AdaptiveNN模型")
    print("=" * 80)
    
    transform = get_transforms()
    snr_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    for snr in SNR_LEVELS:
        # 创建临时数据集
        temp_dataset = RadarDataset(test_data_path,
                                    transform=transform,
                                    snr_filter=[snr])
        
        if len(temp_dataset) == 0:
            continue
        
        temp_loader = DataLoader(temp_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
        
        model.eval()
        with torch.no_grad():
            for images, labels, snrs in temp_loader:
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
                
                # 统计每个样本
                for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    class_name = CLASS_NAMES[label]
                    snr_class_stats[snr][class_name]['total'] += 1
                    if pred == label:
                        snr_class_stats[snr][class_name]['correct'] += 1
    
    # 打印详细结果
    for snr in SNR_LEVELS:
        if snr not in snr_class_stats:
            continue
        
        print(f"\n{snr}dB 准确率详情:")
        print(f"{'类别':<15} {'准确率':<10} {'正确数/总数'}")
        print("-" * 45)
        
        class_accs = []
        for class_name in CLASS_NAMES:
            if class_name in snr_class_stats[snr]:
                stats = snr_class_stats[snr][class_name]
                if stats['total'] > 0:
                    acc = 100.0 * stats['correct'] / stats['total']
                    class_accs.append(acc)
                    print(f"{class_name:<15} {acc:>6.2f}%    {stats['correct']:>4}/{stats['total']:<4}")
        
        # 该SNR下的平均准确率
        if class_accs:
            snr_avg = np.mean(class_accs)
            print("-" * 45)
            print(f"{'平均':<15} {snr_avg:>6.2f}%")
    
    print("=" * 80)
    return snr_class_stats


def save_results_to_csv(snr_results, model_info, save_path='adaptivenn_snr_results.csv'):
    """保存SNR结果到CSV"""
    with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 写入模型信息
        writer.writerow(['Model Information'])
        writer.writerow(['Model', model_info['model_name']])
        writer.writerow(['Parameters', model_info['params_str']])
        writer.writerow(['FLOPs', model_info['flops_str']])
        writer.writerow(['Overall Accuracy', f"{model_info['overall_acc']:.2f}%"])
        writer.writerow([])
        
        # 写入SNR结果
        writer.writerow(['SNR(dB)', 'Accuracy(%)'])
        for snr in sorted(snr_results.keys()):
            writer.writerow([snr, f'{snr_results[snr]:.2f}'])
    
    print(f"\n✓ SNR结果已保存到: {save_path}")


def plot_snr_curve(snr_results, model_info, save_path='adaptivenn_snr_curve.png'):
    """绘制SNR-准确率曲线"""
    plt.figure(figsize=(12, 8))
    
    snrs = sorted(snr_results.keys())
    accs = [snr_results[snr] for snr in snrs]
    
    plt.plot(snrs, accs, 'o-', linewidth=2, markersize=10,
             label=f"AdaptiveNN ({model_info['params_str']} params)", color='red')
    
    plt.xlabel('信噪比 (dB)', fontsize=14)
    plt.ylabel('识别准确率 (%)', fontsize=14)
    plt.title('AdaptiveNN: 不同信噪比下的雷达信号识别准确率', fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xticks(snrs)
    plt.ylim([0, 105])
    
    # 添加数值标注
    for snr, acc in zip(snrs, accs):
        plt.text(snr, acc + 2, f'{acc:.1f}%', ha='center',
                fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 曲线图已保存到: {save_path}")
    plt.close()


def get_args_parser():
    """获取命令行参数"""
    parser = argparse.ArgumentParser('AdaptiveNN SNR Evaluation', add_help=False)
    
    # 数据集参数
    parser.add_argument('--eval_data_path', default='../../CWDtest', type=str,
                        help='测试数据集路径')
    parser.add_argument('--checkpoint', default='./output_radar_fresh/checkpoint-best.pth', type=str,
                        help='模型检查点路径')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='批次大小')
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--model', default='dynamic_deitS', type=str)
    parser.add_argument('--nb_classes', default=16, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    
    # AdaptiveNN特定参数
    parser.add_argument('--feature_in_chans', default=384, type=int)
    parser.add_argument('--policy_net_hidden_chans', default=128, type=int)
    parser.add_argument('--policy_net_kernel_size', default=3, type=int)
    parser.add_argument('--recover_n', default=5, type=int)
    parser.add_argument('--remaining_blocks', default=4, type=int)
    
    parser.add_argument('--glance_input_size', default=112, type=int)
    parser.add_argument('--glance_net_depth', default=12, type=int)
    parser.add_argument('--glance_net_mlp_ratio', default=4, type=int)
    parser.add_argument('--glance_net_drop_path', default=0.1, type=float)
    
    parser.add_argument('--focus_patch_size', default=112, type=int)
    parser.add_argument('--focus_net_reg_size', default=160, type=int)
    parser.add_argument('--focus_net_depth', default=12, type=int)
    parser.add_argument('--focus_net_mlp_ratio', default=4, type=int)
    parser.add_argument('--focus_net_drop_path', default=0.1, type=float)
    
    parser.add_argument('--multi_cls_drop_path', default=0.1, type=float)
    parser.add_argument('--seq_l', default=4, type=int)
    
    # 输出参数
    parser.add_argument('--output_prefix', default='adaptivenn', type=str,
                        help='输出文件前缀')
    
    return parser


def main(args):
    print("=" * 60)
    print("AdaptiveNN模型 - SNR评估")
    print("=" * 60)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    print(f"测试数据路径: {args.eval_data_path}")
    print(f"检查点路径: {args.checkpoint}")
    print(f"Batch Size: {args.batch_size}")
    
    # 检查路径
    if not os.path.exists(args.eval_data_path):
        print(f"\n✗ 错误：测试数据路径不存在: {args.eval_data_path}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"\n✗ 错误：检查点文件不存在: {args.checkpoint}")
        return
    
    # 创建模型
    print("\n创建AdaptiveNN模型...")
    model = create_adaptivenn_model(args)
    
    # 加载检查点
    model = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()
    
    # 计算参数量
    print("\n计算模型复杂度...")
    total_params, trainable_params = count_parameters(model)
    flops, params, flops_str, params_str = calculate_flops(model)
    
    print(f"✓ 总参数量: {format_number(total_params)} ({total_params:,})")
    print(f"✓ 可训练参数: {format_number(trainable_params)} ({trainable_params:,})")
    if flops_str != "N/A":
        print(f"✓ FLOPs: {flops_str}")
    
    # 整体评估
    print("\n整体评估...")
    test_dataset = RadarDataset(args.eval_data_path, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    
    overall_acc = evaluate_model(model, test_loader, device)
    print(f"✓ 整体准确率: {overall_acc:.2f}%")
    
    # 按SNR评估（总体）
    snr_results = evaluate_by_snr(model, args.eval_data_path, device, args.batch_size)
    
    # 按SNR和类别详细评估
    snr_class_results = evaluate_by_snr_and_class(model, args.eval_data_path, device, args.batch_size)
    
    # 准备模型信息
    model_info = {
        'model_name': 'AdaptiveNN',
        'params_str': params_str,
        'flops_str': flops_str,
        'overall_acc': overall_acc,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    # 保存结果
    csv_path = f'{args.output_prefix}_snr_results.csv'
    curve_path = f'{args.output_prefix}_snr_curve.png'
    
    save_results_to_csv(snr_results, model_info, csv_path)
    plot_snr_curve(snr_results, model_info, curve_path)
    
    # 保存详细统计
    stats_path = f'{args.output_prefix}_stats.csv'
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Model', 'AdaptiveNN'])
        writer.writerow(['Overall Accuracy (%)', f'{overall_acc:.2f}'])
        writer.writerow(['Total Parameters', total_params])
        writer.writerow(['Trainable Parameters', trainable_params])
        writer.writerow(['Parameters (formatted)', params_str])
        writer.writerow(['FLOPs', flops_str])
        writer.writerow(['Checkpoint', args.checkpoint])
    
    print(f"✓ 详细统计已保存到: {stats_path}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("评估完成！")
    print("=" * 60)
    print(f"模型: AdaptiveNN")
    print(f"整体准确率: {overall_acc:.2f}%")
    print(f"参数量: {params_str}")
    print(f"FLOPs: {flops_str}")
    print("\n保存的文件:")
    print(f"  - {csv_path} (SNR识别率数据)")
    print(f"  - {curve_path} (SNR曲线图)")
    print(f"  - {stats_path} (详细统计)")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AdaptiveNN SNR Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
