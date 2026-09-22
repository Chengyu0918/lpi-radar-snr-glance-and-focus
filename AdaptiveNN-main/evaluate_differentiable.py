"""
可微分AdaptiveNN评估脚本 - 雷达信号分类
支持按SNR级别评估准确率
"""
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, './models')

from timm.models import create_model
from dataset_utils import build_dataset
import utils
import models.dynamic_deitS_differentiable


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
    parser = argparse.ArgumentParser('Differentiable AdaptiveNN evaluation script', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='dynamic_deitS_diff', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--checkpoint', default='./output_radar_differentiable/checkpoint-best.pth', type=str,
                        help='Path to checkpoint')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='', type=str, help='dataset path')
    parser.add_argument('--eval_data_path', default=None, type=str)
    parser.add_argument('--nb_classes', default=16, type=int)
    parser.add_argument('--data_set', default='image_folder', type=str)
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    
    # AdaptiveNN parameters
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
    
    # Evaluation parameters
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    
    # SNR evaluation
    parser.add_argument('--evaluate_by_snr', type=str2bool, default=True,
                        help='Evaluate accuracy for each SNR level')
    
    # Transform parameters (required by dataset_utils)
    parser.add_argument('--color_jitter', type=float, default=0.4)
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1')
    parser.add_argument('--smoothing', type=float, default=0.1)
    parser.add_argument('--train_interpolation', type=str, default='bicubic')
    parser.add_argument('--reprob', type=float, default=0.25)
    parser.add_argument('--remode', type=str, default='pixel')
    parser.add_argument('--recount', type=int, default=1)
    parser.add_argument('--crop_pct', type=float, default=None)
    
    return parser


@torch.no_grad()
def evaluate_model(model, data_loader, device, args):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_outputs = []
    
    print("开始评估...")
    for batch_idx, (images, targets) in enumerate(data_loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # 前向传播
        with torch.cuda.amp.autocast():
            expected_outputs = model(images, seq_l=args.seq_l)
            output = expected_outputs['x_focus'][-1]  # 使用最后一个focus step的输出
        
        # 收集预测结果
        _, preds = output.max(1)
        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())
        all_outputs.append(output.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            print(f"已处理 {batch_idx + 1}/{len(data_loader)} batches")
    
    # 合并所有结果
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)
    
    # 计算总体准确率
    correct = (all_preds == all_targets).sum().item()
    total = len(all_targets)
    accuracy = 100.0 * correct / total
    
    return accuracy, all_preds, all_targets, all_outputs


def evaluate_by_snr(data_path, all_preds, all_targets):
    """按SNR级别评估准确率"""
    import os
    from collections import defaultdict
    
    # SNR级别列表
    snr_levels = ['-10dB', '-8dB', '-6dB', '-4dB', '-2dB', '0dB', 
                  '2dB', '4dB', '6dB', '8dB', '10dB']
    
    # 获取所有类别
    classes = sorted([d for d in os.listdir(data_path) 
                     if os.path.isdir(os.path.join(data_path, d))])
    
    print(f"\n找到 {len(classes)} 个类别: {classes}")
    
    # 按SNR统计（总体）
    snr_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    # 按SNR和类别统计（详细）
    snr_class_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    # 遍历数据集，根据文件名提取SNR信息
    dataset_path = Path(data_path)
    sample_idx = 0
    
    for class_idx, class_name in enumerate(classes):
        class_path = dataset_path / class_name
        if not class_path.exists():
            continue
            
        # 获取该类别的所有图像文件
        image_files = sorted(list(class_path.glob('*.png')))
        
        for img_file in image_files:
            if sample_idx >= len(all_preds):
                break
                
            # 从文件名提取SNR信息
            # 文件名格式: BPSK_-10dB_1.png
            filename = img_file.stem
            snr = None
            for snr_level in snr_levels:
                if snr_level in filename:
                    snr = snr_level
                    break
            
            if snr is None:
                sample_idx += 1
                continue
            
            # 统计该SNR的准确率
            pred = all_preds[sample_idx].item()
            target = all_targets[sample_idx].item()
            
            # 总体统计
            snr_stats[snr]['total'] += 1
            if pred == target:
                snr_stats[snr]['correct'] += 1
            
            # 按类别统计
            snr_class_stats[snr][class_name]['total'] += 1
            if pred == target:
                snr_class_stats[snr][class_name]['correct'] += 1
            
            sample_idx += 1
    
    return snr_stats, snr_class_stats, classes


def main(args):
    print(args)
    
    device = torch.device(args.device)
    
    # 固定随机种子
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    
    # 构建数据集
    print(f"\n加载数据集: {args.eval_data_path}")
    dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        shuffle=False  # 重要：不打乱顺序，以便按SNR统计
    )
    
    # 创建模型
    print(f"\n创建模型: {args.model}")
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
    
    # 加载checkpoint
    print(f"\n加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'module' in checkpoint:
        state_dict = checkpoint['module']
    else:
        state_dict = checkpoint
    
    # 加载权重
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"加载权重: {msg}")
    
    model.to(device)
    model.eval()
    
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f'\n模型参数量: {n_parameters:,}')
    
    # 评估模型
    print(f"\n{'='*60}")
    print("开始评估可微分AdaptiveNN模型")
    print(f"{'='*60}")
    
    accuracy, all_preds, all_targets, all_outputs = evaluate_model(
        model, data_loader_val, device, args
    )
    
    print(f"\n{'='*60}")
    print(f"总体准确率: {accuracy:.2f}%")
    print(f"{'='*60}")
    
    # 按SNR评估
    if args.evaluate_by_snr and args.eval_data_path:
        print(f"\n{'='*60}")
        print("按SNR级别评估准确率")
        print(f"{'='*60}")
        
        snr_stats, snr_class_stats, classes = evaluate_by_snr(
            args.eval_data_path, all_preds, all_targets
        )
        
        # 打印总体SNR准确率
        print(f"\n{'SNR级别':<10} {'准确率':<10} {'正确数/总数'}")
        print("-" * 40)
        
        snr_levels = ['-10dB', '-8dB', '-6dB', '-4dB', '-2dB', '0dB', 
                      '2dB', '4dB', '6dB', '8dB', '10dB']
        
        for snr in snr_levels:
            if snr in snr_stats:
                stats = snr_stats[snr]
                acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
                print(f"{snr:<10} {acc:>6.2f}%    {stats['correct']:>4}/{stats['total']:<4}")
        
        # 计算平均准确率
        avg_acc = np.mean([100.0 * stats['correct'] / stats['total'] 
                          for stats in snr_stats.values() if stats['total'] > 0])
        print("-" * 40)
        print(f"{'平均':<10} {avg_acc:>6.2f}%")
        
        # 打印详细的SNR×类别准确率矩阵
        print(f"\n{'='*80}")
        print("每个SNR下各类别的详细准确率")
        print(f"{'='*80}")
        
        for snr in snr_levels:
            if snr not in snr_class_stats:
                continue
                
            print(f"\n{snr} 准确率详情:")
            print(f"{'类别':<15} {'准确率':<10} {'正确数/总数'}")
            print("-" * 45)
            
            class_accs = []
            for class_name in classes:
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
    
    # 按类别统计
    print(f"\n{'='*60}")
    print("按类别统计准确率")
    print(f"{'='*60}")
    
    num_classes = args.nb_classes
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for pred, target in zip(all_preds, all_targets):
        class_total[target] += 1
        if pred == target:
            class_correct[target] += 1
    
    # 获取类别名称
    if args.eval_data_path:
        import os
        classes = sorted([d for d in os.listdir(args.eval_data_path) 
                         if os.path.isdir(os.path.join(args.eval_data_path, d))])
    else:
        classes = [f"Class_{i}" for i in range(num_classes)]
    
    print(f"\n{'类别':<15} {'准确率':<10} {'正确数/总数'}")
    print("-" * 45)
    
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_name = classes[i] if i < len(classes) else f"Class_{i}"
            print(f"{class_name:<15} {acc:>6.2f}%    {int(class_correct[i]):>4}/{int(class_total[i]):<4}")
    
    print("\n评估完成！")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Differentiable AdaptiveNN evaluation script', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
