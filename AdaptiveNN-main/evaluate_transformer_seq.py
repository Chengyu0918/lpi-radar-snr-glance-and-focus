"""
Transformer时序建模版本的评估脚本
支持按SNR分类评估和注意力可视化
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import matplotlib.pyplot as plt

# 导入模型
from models.dynamic_deitS_transformer_seq import dynamic_deitS_transformer_seq


def get_args_parser():
    parser = argparse.ArgumentParser('Transformer-Seq AdaptiveNN Evaluation', add_help=False)
    
    # 数据参数
    parser.add_argument('--data_path', default='../../CWDtest', type=str,
                        help='dataset path for evaluation')
    parser.add_argument('--checkpoint', default='./output_transformer_seq/checkpoint-best.pth', type=str,
                        help='path to checkpoint')
    parser.add_argument('--output_dir', default='./eval_results_transformer_seq', type=str,
                        help='output directory for results')
    
    # 模型参数
    parser.add_argument('--nb_classes', default=16, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--seq_l', default=4, type=int)
    parser.add_argument('--feature_in_chans', default=384, type=int)
    parser.add_argument('--recover_n', default=3, type=int)
    parser.add_argument('--remaining_blocks', default=2, type=int)
    parser.add_argument('--glance_input_size', default=112, type=int)
    parser.add_argument('--glance_net_depth', default=6, type=int)
    parser.add_argument('--glance_net_mlp_ratio', default=4.0, type=float)
    parser.add_argument('--glance_net_drop_path', default=0.1, type=float)
    parser.add_argument('--focus_patch_size', default=96, type=int)
    parser.add_argument('--focus_net_reg_size', default=96, type=int)
    parser.add_argument('--focus_net_depth', default=6, type=int)
    parser.add_argument('--focus_net_mlp_ratio', default=4.0, type=float)
    parser.add_argument('--focus_net_drop_path', default=0.1, type=float)
    parser.add_argument('--multi_cls_drop_path', default=0.1, type=float)
    parser.add_argument('--policy_net_hidden_chans', default=256, type=int)
    parser.add_argument('--policy_net_kernel_size', default=7, type=int)
    parser.add_argument('--transformer_num_layers', default=3, type=int)
    parser.add_argument('--transformer_nhead', default=8, type=int)
    parser.add_argument('--transformer_dim_feedforward', default=1536, type=int)
    parser.add_argument('--transformer_dropout', default=0.1, type=float)
    parser.add_argument('--transformer_drop_path', default=0.1, type=float)
    parser.add_argument('--diversity_sigma', default=0.3, type=float)
    
    # 评估参数
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--visualize', action='store_true', help='Generate attention visualizations')
    parser.add_argument('--num_visualize', default=16, type=int, help='Number of samples to visualize')
    
    return parser


def parse_snr_from_filename(filename):
    """从文件名解析SNR值"""
    import re
    match = re.search(r'_(-?\d+)dB_', filename)
    if match:
        return int(match.group(1))
    return None


def load_dataset(data_path, input_size):
    """加载数据集"""
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    samples = []
    class_names = sorted(os.listdir(data_path))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_path, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_name)
                snr = parse_snr_from_filename(img_name)
                samples.append({
                    'path': img_path,
                    'class': class_name,
                    'class_idx': class_to_idx[class_name],
                    'snr': snr,
                    'transform': transform
                })
    
    return samples, class_names, class_to_idx


def create_model(args):
    """创建模型"""
    model = dynamic_deitS_transformer_seq(
        seq_len=args.seq_l,
        feature_in_chans=args.feature_in_chans,
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
        policy_net_hidden_chans=args.policy_net_hidden_chans,
        policy_net_kernel_size=args.policy_net_kernel_size,
        transformer_num_layers=args.transformer_num_layers,
        transformer_nhead=args.transformer_nhead,
        transformer_dim_feedforward=args.transformer_dim_feedforward,
        transformer_dropout=args.transformer_dropout,
        transformer_drop_path=args.transformer_drop_path,
        diversity_sigma=args.diversity_sigma,
        num_classes=args.nb_classes,
        drop_path_rate=0.1,
    )
    return model


def load_checkpoint(model, checkpoint_path, device):
    """加载检查点"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # 处理DDP模型的state_dict
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model


@torch.no_grad()
def evaluate_model(model, samples, device, args):
    """评估模型"""
    model.eval()
    
    # 按SNR和类别统计
    results_by_snr = defaultdict(lambda: {'correct': 0, 'total': 0})
    results_by_class = defaultdict(lambda: {'correct': 0, 'total': 0})
    results_by_snr_class = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    # 位置统计
    all_positions = [[] for _ in range(args.seq_l)]
    
    # 逐样本评估
    for i, sample in enumerate(samples):
        # 加载图像
        img = Image.open(sample['path']).convert('RGB')
        img_tensor = sample['transform'](img).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(img_tensor, seq_l=args.seq_l)
        
        # 获取预测
        if outputs['x_final'] is not None:
            logits = outputs['x_final']
        else:
            logits = outputs['x_focus'][-1]
        
        pred = logits.argmax(dim=1).item()
        target = sample['class_idx']
        correct = int(pred == target)
        
        # 更新统计
        snr = sample['snr']
        class_name = sample['class']
        
        if snr is not None:
            results_by_snr[snr]['correct'] += correct
            results_by_snr[snr]['total'] += 1
            results_by_snr_class[snr][class_name]['correct'] += correct
            results_by_snr_class[snr][class_name]['total'] += 1
        
        results_by_class[class_name]['correct'] += correct
        results_by_class[class_name]['total'] += 1
        
        # 收集位置信息
        for t, pos in enumerate(outputs['actions']):
            all_positions[t].append(pos.cpu().numpy())
        
        if (i + 1) % 100 == 0:
            print(f"Evaluated {i + 1}/{len(samples)} samples")
    
    return results_by_snr, results_by_class, results_by_snr_class, all_positions


def visualize_samples(model, samples, device, args, output_dir, num_samples=16):
    """可视化样本的注意力分布"""
    model.eval()
    
    # 随机选择样本
    indices = np.random.choice(len(samples), min(num_samples, len(samples)), replace=False)
    
    fig, axes = plt.subplots(len(indices), args.seq_l + 2, figsize=(4*(args.seq_l+2), 4*len(indices)))
    
    for row, idx in enumerate(indices):
        sample = samples[idx]
        
        # 加载原图
        img_pil = Image.open(sample['path']).convert('RGB')
        img_tensor = sample['transform'](img_pil).unsqueeze(0).to(device)
        
        # 前向传播
        with torch.cuda.amp.autocast():
            outputs = model(img_tensor, seq_l=args.seq_l)
        
        # 显示原图
        img_np = np.array(img_pil.resize((args.input_size, args.input_size)))
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title(f'{sample["class"]}\nSNR: {sample["snr"]}dB')
        axes[row, 0].axis('off')
        
        # 显示每个step的注意力
        for t in range(args.seq_l):
            attn = outputs['attention_probs'][t][0].cpu().numpy()
            H = W = int(np.sqrt(len(attn)))
            attn = attn.reshape(H, W)
            
            # 上采样注意力图
            attn_upsampled = np.array(Image.fromarray(attn).resize((args.input_size, args.input_size), Image.BILINEAR))
            
            axes[row, t+1].imshow(img_np)
            axes[row, t+1].imshow(attn_upsampled, alpha=0.6, cmap='hot')
            
            # 标记位置
            pos = outputs['actions'][t][0].cpu().numpy()
            y_pos = pos[0] * args.input_size
            x_pos = pos[1] * args.input_size
            axes[row, t+1].scatter([x_pos], [y_pos], c='cyan', s=150, marker='x', linewidths=3)
            
            axes[row, t+1].set_title(f'Step {t+1}')
            axes[row, t+1].axis('off')
        
        # 显示所有位置轨迹
        axes[row, -1].imshow(img_np)
        colors = plt.cm.rainbow(np.linspace(0, 1, args.seq_l))
        for t in range(args.seq_l):
            pos = outputs['actions'][t][0].cpu().numpy()
            y_pos = pos[0] * args.input_size
            x_pos = pos[1] * args.input_size
            axes[row, -1].scatter([x_pos], [y_pos], c=[colors[t]], s=150, marker='o', label=f'Step {t+1}')
            if t > 0:
                prev_pos = outputs['actions'][t-1][0].cpu().numpy()
                prev_y = prev_pos[0] * args.input_size
                prev_x = prev_pos[1] * args.input_size
                axes[row, -1].plot([prev_x, x_pos], [prev_y, y_pos], c=colors[t], linewidth=2)
        
        axes[row, -1].set_title('Trajectory')
        axes[row, -1].axis('off')
        if row == 0:
            axes[row, -1].legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'attention_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def plot_results(results_by_snr, results_by_class, all_positions, output_dir, args):
    """绘制结果图表"""
    
    # 1. SNR vs Accuracy曲线
    if results_by_snr:
        snrs = sorted(results_by_snr.keys())
        accs = [results_by_snr[snr]['correct'] / results_by_snr[snr]['total'] * 100 
                for snr in snrs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, accs, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.title('Accuracy vs SNR', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'snr_accuracy_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. 类别准确率柱状图
    classes = sorted(results_by_class.keys())
    class_accs = [results_by_class[c]['correct'] / results_by_class[c]['total'] * 100 
                  for c in classes]
    
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(classes)), class_accs, color='steelblue')
    plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Accuracy by Class', fontsize=14)
    plt.ylim(0, 105)
    
    # 添加数值标签
    for bar, acc in zip(bars, class_accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 位置分布图
    fig, axes = plt.subplots(1, args.seq_l, figsize=(4*args.seq_l, 4))
    
    for t in range(args.seq_l):
        positions = np.concatenate(all_positions[t], axis=0)
        
        axes[t].hexbin(positions[:, 1], positions[:, 0], gridsize=20, cmap='YlOrRd')
        axes[t].set_xlim(0, 1)
        axes[t].set_ylim(0, 1)
        axes[t].set_xlabel('X')
        axes[t].set_ylabel('Y')
        axes[t].set_title(f'Step {t+1} Position Distribution')
        axes[t].set_aspect('equal')
        axes[t].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 位置多样性分析
    print("\n=== Position Diversity Analysis ===")
    for t in range(args.seq_l):
        positions = np.concatenate(all_positions[t], axis=0)
        mean_pos = positions.mean(axis=0)
        std_pos = positions.std(axis=0)
        print(f"Step {t+1}: Mean=({mean_pos[0]:.4f}, {mean_pos[1]:.4f}), Std=({std_pos[0]:.4f}, {std_pos[1]:.4f})")


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据集
    print(f"Loading dataset from {args.data_path}")
    samples, class_names, class_to_idx = load_dataset(args.data_path, args.input_size)
    print(f"Loaded {len(samples)} samples, {len(class_names)} classes")
    
    # 创建模型
    print("Creating model...")
    model = create_model(args)
    model = model.to(device)
    
    # 加载检查点
    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_checkpoint(model, args.checkpoint, device)
    
    # 评估
    print("Evaluating...")
    results_by_snr, results_by_class, results_by_snr_class, all_positions = evaluate_model(
        model, samples, device, args
    )
    
    # 计算总体准确率
    total_correct = sum(r['correct'] for r in results_by_class.values())
    total_samples = sum(r['total'] for r in results_by_class.values())
    overall_acc = total_correct / total_samples * 100
    
    print(f"\n=== Overall Results ===")
    print(f"Total Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    
    # 按SNR打印结果
    if results_by_snr:
        print(f"\n=== Results by SNR ===")
        for snr in sorted(results_by_snr.keys()):
            r = results_by_snr[snr]
            acc = r['correct'] / r['total'] * 100
            print(f"SNR {snr:3d}dB: {acc:6.2f}% ({r['correct']:4d}/{r['total']:4d})")
    
    # 按类别打印结果
    print(f"\n=== Results by Class ===")
    for class_name in sorted(results_by_class.keys()):
        r = results_by_class[class_name]
        acc = r['correct'] / r['total'] * 100
        print(f"{class_name:15s}: {acc:6.2f}% ({r['correct']:4d}/{r['total']:4d})")
    
    # 绘制结果图表
    print("\nGenerating plots...")
    plot_results(results_by_snr, results_by_class, all_positions, args.output_dir, args)
    
    # 可视化
    if args.visualize:
        print("\nGenerating attention visualizations...")
        visualize_samples(model, samples, device, args, args.output_dir, args.num_visualize)
    
    # 保存结果到文件
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"Overall Accuracy: {overall_acc:.2f}%\n\n")
        
        if results_by_snr:
            f.write("Results by SNR:\n")
            for snr in sorted(results_by_snr.keys()):
                r = results_by_snr[snr]
                acc = r['correct'] / r['total'] * 100
                f.write(f"  SNR {snr:3d}dB: {acc:6.2f}%\n")
            f.write("\n")
        
        f.write("Results by Class:\n")
        for class_name in sorted(results_by_class.keys()):
            r = results_by_class[class_name]
            acc = r['correct'] / r['total'] * 100
            f.write(f"  {class_name:15s}: {acc:6.2f}%\n")
    
    print(f"\nResults saved to {results_file}")
    print("Evaluation completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Transformer-Seq AdaptiveNN Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
