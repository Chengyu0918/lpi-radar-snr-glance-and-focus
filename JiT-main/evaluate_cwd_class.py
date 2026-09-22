"""
带类别条件的CWD去噪模型评估脚本
用于评估训练好的JiT-CWD-Class去噪模型的性能

评估指标：
- PSNR (峰值信噪比)
- SSIM (结构相似性)
- MSE (均方误差)

使用方法：
python evaluate_cwd_class.py --checkpoint ./output_cwd_class/checkpoint-last.pth --noisy_dir ../../../CWDtrain_noisy --clean_dir ../../../CWDtrain_clean --output ./evaluation_results_class
"""
import argparse
import os
import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 尝试导入skimage，如果没有则使用自定义实现
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: skimage not found, using custom PSNR/SSIM implementation")

from denoiser_cwd_class import CWDDenoiserWithClass


# 类别名称列表
CLASS_NAMES = [
    'BPSK', 'Costas', 'CP', 'Frank', 
    'FSK4_Baker5', 'FSK4_LFM', 'LFM', 'NLFM',
    'P1', 'P2', 'P3', 'P4', 
    'T1', 'T2', 'T3', 'T4'
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def calculate_psnr(img1, img2, data_range=255):
    """计算PSNR"""
    if HAS_SKIMAGE:
        return psnr(img1, img2, data_range=data_range)
    else:
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(data_range / np.sqrt(mse))


def calculate_ssim(img1, img2, data_range=255):
    """计算SSIM"""
    if HAS_SKIMAGE:
        return ssim(img1, img2, data_range=data_range)
    else:
        # 简化版SSIM实现
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2
        
        img1 = img1.astype(float)
        img2 = img2.astype(float)
        
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1_sq = np.var(img1)
        sigma2_sq = np.var(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        ssim_val = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_val


def calculate_mse(img1, img2):
    """计算MSE"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-CWD-Class Model Evaluation', add_help=False)
    
    # 模型参数
    parser.add_argument('--model', default='JiT-CWD-Class-Base', type=str,
                        choices=['JiT-CWD-Class-Tiny', 'JiT-CWD-Class-Small', 'JiT-CWD-Class-Base'])
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--class_dropout_prob', default=0.1, type=float)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    
    # 扩散参数
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.05, type=float)
    
    # 采样参数
    parser.add_argument('--sampling_method', default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--cfg_scale', default=2.0, type=float,
                        help='Classifier-free guidance scale')
    
    # 输入输出
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='模型checkpoint路径')
    parser.add_argument('--noisy_dir', required=True, type=str,
                        help='含噪图像目录')
    parser.add_argument('--clean_dir', required=True, type=str,
                        help='干净图像目录（用于计算指标）')
    parser.add_argument('--output', default='./evaluation_results_class', type=str,
                        help='评估结果输出目录')
    
    # 评估参数
    parser.add_argument('--num_samples', default=50, type=int,
                        help='每个类别评估的样本数量，-1表示全部')
    parser.add_argument('--save_images', action='store_true', default=True,
                        help='是否保存对比图像')
    parser.add_argument('--num_vis_samples', default=3, type=int,
                        help='每个类别可视化的样本数量')
    
    # 其他
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    
    return parser


def load_model(args):
    """加载模型"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = CWDDenoiserWithClass(args)
    model.to(device)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 尝试不同的键名加载模型权重
    if 'model' in checkpoint:
        # 完整的训练检查点
        state_dict = checkpoint['model']
        # 提取model.前缀的权重
        model_state = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                model_state[k[6:]] = v  # 去掉'model.'前缀
            else:
                model_state[k] = v
        model.model.load_state_dict(model_state, strict=False)
        print(f"Loaded model weights from checkpoint['model']")
    elif 'model_state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from checkpoint['model_state_dict']")
    elif 'state_dict' in checkpoint:
        model.model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded model weights from checkpoint['state_dict']")
    else:
        # 直接尝试加载
        model.model.load_state_dict(checkpoint)
        print(f"Loaded model weights directly")
    
    # 尝试加载EMA权重（如果有）
    if 'model_ema1' in checkpoint:
        try:
            ema_state = checkpoint['model_ema1']
            model_state = {}
            for k, v in ema_state.items():
                if k.startswith('model.'):
                    model_state[k[6:]] = v
                else:
                    model_state[k] = v
            model.model.load_state_dict(model_state, strict=False)
            print("Loaded EMA parameters (model_ema1)")
        except Exception as e:
            print(f"Warning: Could not load EMA parameters: {e}")
    
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    
    return model, device


def get_snr_from_filename(filename):
    """从文件名提取SNR值"""
    # 文件名格式: BPSK_-10dB_1.png
    parts = filename.split('_')
    for part in parts:
        if 'dB' in part:
            snr_str = part.replace('dB', '')
            try:
                return int(snr_str)
            except ValueError:
                pass
    return None


def get_class_from_path(path):
    """从路径获取类别名称"""
    # 路径格式: .../BPSK/BPSK_-10dB_1.png
    parts = Path(path).parts
    for part in parts:
        if part in CLASS_TO_IDX:
            return part
    # 从文件名提取
    filename = Path(path).stem
    for class_name in CLASS_NAMES:
        if filename.startswith(class_name):
            return class_name
    return None


def denoise_image(model, noisy_img_path, class_label, device, img_size=224, cfg_scale=2.0):
    """对单张图像进行去噪"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img = Image.open(noisy_img_path).convert('L')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 类别标签
    class_idx = CLASS_TO_IDX.get(class_label, 0)
    class_tensor = torch.tensor([class_idx], device=device)
    
    with torch.no_grad():
        denoised = model.denoise(img_tensor, class_tensor, cfg_scale=cfg_scale)
    
    # 反归一化
    denoised_np = (denoised[0, 0].cpu().numpy() + 1) / 2 * 255
    denoised_np = np.clip(denoised_np, 0, 255).astype(np.uint8)
    
    return denoised_np


def load_image(img_path, img_size=224):
    """加载并预处理图像"""
    img = Image.open(img_path).convert('L')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img)


def evaluate_model(args):
    """主评估函数"""
    print('=' * 60)
    print('JiT-CWD-Class Model Evaluation')
    print('=' * 60)
    print(f'CFG Scale: {args.cfg_scale}')
    print(f'Sampling Steps: {args.num_sampling_steps}')
    print('=' * 60)
    
    # 加载模型
    model, device = load_model(args)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 收集所有图像对
    noisy_dir = Path(args.noisy_dir)
    clean_dir = Path(args.clean_dir)
    
    # 按类别和SNR组织结果
    results_by_class = defaultdict(list)
    results_by_snr = defaultdict(list)
    all_results = []
    
    # 获取所有类别
    categories = [d.name for d in noisy_dir.iterdir() if d.is_dir()]
    print(f"\nFound {len(categories)} categories: {categories}")
    
    # 可视化样本收集
    vis_samples = defaultdict(list)
    
    for category in tqdm(categories, desc="Processing categories"):
        noisy_cat_dir = noisy_dir / category
        clean_cat_dir = clean_dir / category
        
        if not clean_cat_dir.exists():
            print(f"Warning: Clean directory not found for {category}")
            continue
        
        # 获取该类别的所有图像
        noisy_images = list(noisy_cat_dir.glob('*.png'))
        
        # 限制样本数量
        if args.num_samples > 0:
            noisy_images = noisy_images[:args.num_samples]
        
        for noisy_path in tqdm(noisy_images, desc=f"  {category}", leave=False):
            clean_path = clean_cat_dir / noisy_path.name
            
            if not clean_path.exists():
                continue
            
            # 获取SNR
            snr = get_snr_from_filename(noisy_path.name)
            
            # 加载图像
            noisy_img = load_image(noisy_path, args.img_size)
            clean_img = load_image(clean_path, args.img_size)
            
            # 去噪（使用类别条件）
            denoised_img = denoise_image(model, noisy_path, category, device, 
                                         args.img_size, args.cfg_scale)
            
            # 计算指标
            # 去噪后 vs 干净图像
            psnr_denoised = calculate_psnr(denoised_img, clean_img)
            ssim_denoised = calculate_ssim(denoised_img, clean_img)
            mse_denoised = calculate_mse(denoised_img, clean_img)
            
            # 含噪 vs 干净图像（基线）
            psnr_noisy = calculate_psnr(noisy_img, clean_img)
            ssim_noisy = calculate_ssim(noisy_img, clean_img)
            mse_noisy = calculate_mse(noisy_img, clean_img)
            
            result = {
                'category': category,
                'filename': noisy_path.name,
                'snr': snr,
                'psnr_denoised': psnr_denoised,
                'ssim_denoised': ssim_denoised,
                'mse_denoised': mse_denoised,
                'psnr_noisy': psnr_noisy,
                'ssim_noisy': ssim_noisy,
                'mse_noisy': mse_noisy,
                'psnr_improvement': psnr_denoised - psnr_noisy,
                'ssim_improvement': ssim_denoised - ssim_noisy,
            }
            
            all_results.append(result)
            results_by_class[category].append(result)
            if snr is not None:
                results_by_snr[snr].append(result)
            
            # 收集可视化样本
            if len(vis_samples[category]) < args.num_vis_samples:
                vis_samples[category].append({
                    'noisy': noisy_img,
                    'clean': clean_img,
                    'denoised': denoised_img,
                    'filename': noisy_path.name,
                    'psnr_improvement': psnr_denoised - psnr_noisy,
                    'ssim_denoised': ssim_denoised,
                })
    
    # 计算统计结果
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    # 总体结果
    if all_results:
        avg_psnr_denoised = np.mean([r['psnr_denoised'] for r in all_results])
        avg_ssim_denoised = np.mean([r['ssim_denoised'] for r in all_results])
        avg_psnr_noisy = np.mean([r['psnr_noisy'] for r in all_results])
        avg_ssim_noisy = np.mean([r['ssim_noisy'] for r in all_results])
        avg_psnr_improvement = np.mean([r['psnr_improvement'] for r in all_results])
        avg_ssim_improvement = np.mean([r['ssim_improvement'] for r in all_results])
        
        print(f"\n{'Overall Results':^60}")
        print("-" * 60)
        print(f"Total samples evaluated: {len(all_results)}")
        print(f"CFG Scale: {args.cfg_scale}")
        print(f"\n{'Metric':<20} {'Noisy':<15} {'Denoised':<15} {'Improvement':<15}")
        print("-" * 60)
        print(f"{'PSNR (dB)':<20} {avg_psnr_noisy:<15.2f} {avg_psnr_denoised:<15.2f} {avg_psnr_improvement:+.2f}")
        print(f"{'SSIM':<20} {avg_ssim_noisy:<15.4f} {avg_ssim_denoised:<15.4f} {avg_ssim_improvement:+.4f}")
    
    # 按类别结果
    print(f"\n{'Results by Category':^60}")
    print("-" * 60)
    print(f"{'Category':<15} {'PSNR(N)':<10} {'PSNR(D)':<10} {'ΔPSNR':<10} {'SSIM(D)':<10}")
    print("-" * 60)
    
    category_stats = {}
    for category in sorted(results_by_class.keys()):
        results = results_by_class[category]
        avg_psnr_n = np.mean([r['psnr_noisy'] for r in results])
        avg_psnr_d = np.mean([r['psnr_denoised'] for r in results])
        avg_ssim_d = np.mean([r['ssim_denoised'] for r in results])
        delta_psnr = avg_psnr_d - avg_psnr_n
        
        print(f"{category:<15} {avg_psnr_n:<10.2f} {avg_psnr_d:<10.2f} {delta_psnr:+10.2f} {avg_ssim_d:<10.4f}")
        
        category_stats[category] = {
            'psnr_noisy': avg_psnr_n,
            'psnr_denoised': avg_psnr_d,
            'ssim_denoised': avg_ssim_d,
            'psnr_improvement': delta_psnr,
            'num_samples': len(results)
        }
    
    # 按SNR结果
    snr_stats = {}
    if results_by_snr:
        print(f"\n{'Results by SNR Level':^60}")
        print("-" * 60)
        print(f"{'SNR (dB)':<10} {'Samples':<10} {'PSNR(N)':<10} {'PSNR(D)':<10} {'ΔPSNR':<10} {'SSIM(D)':<10}")
        print("-" * 60)
        
        for snr in sorted(results_by_snr.keys()):
            results = results_by_snr[snr]
            avg_psnr_n = np.mean([r['psnr_noisy'] for r in results])
            avg_psnr_d = np.mean([r['psnr_denoised'] for r in results])
            avg_ssim_d = np.mean([r['ssim_denoised'] for r in results])
            delta_psnr = avg_psnr_d - avg_psnr_n
            
            print(f"{snr:<10} {len(results):<10} {avg_psnr_n:<10.2f} {avg_psnr_d:<10.2f} {delta_psnr:+10.2f} {avg_ssim_d:<10.4f}")
            
            snr_stats[snr] = {
                'psnr_noisy': avg_psnr_n,
                'psnr_denoised': avg_psnr_d,
                'ssim_denoised': avg_ssim_d,
                'psnr_improvement': delta_psnr,
                'num_samples': len(results)
            }
    
    # 保存结果
    summary = {
        'config': {
            'model': args.model,
            'cfg_scale': args.cfg_scale,
            'num_sampling_steps': args.num_sampling_steps,
            'checkpoint': args.checkpoint,
        },
        'overall': {
            'total_samples': len(all_results),
            'avg_psnr_noisy': float(avg_psnr_noisy) if all_results else 0,
            'avg_psnr_denoised': float(avg_psnr_denoised) if all_results else 0,
            'avg_ssim_noisy': float(avg_ssim_noisy) if all_results else 0,
            'avg_ssim_denoised': float(avg_ssim_denoised) if all_results else 0,
            'avg_psnr_improvement': float(avg_psnr_improvement) if all_results else 0,
            'avg_ssim_improvement': float(avg_ssim_improvement) if all_results else 0,
        },
        'by_category': category_stats,
        'by_snr': snr_stats,
    }
    
    # 保存JSON结果
    with open(os.path.join(args.output, 'evaluation_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存详细结果CSV
    import csv
    with open(os.path.join(args.output, 'evaluation_details.csv'), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys() if all_results else [])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nResults saved to {args.output}")
    
    # 生成可视化
    if args.save_images and vis_samples:
        print("\nGenerating visualization images...")
        generate_visualizations(vis_samples, args.output, snr_stats, category_stats)
    
    return summary


def generate_visualizations(vis_samples, output_dir, snr_stats=None, category_stats=None):
    """生成可视化图像"""
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # 为每个类别生成对比图
    for category, samples in vis_samples.items():
        if not samples:
            continue
        
        n_samples = len(samples)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Denoising Results - {category}', fontsize=14, fontweight='bold')
        
        for i, sample in enumerate(samples):
            # 含噪图像
            axes[i, 0].imshow(sample['noisy'], cmap='gray')
            axes[i, 0].set_title('Noisy Input')
            axes[i, 0].axis('off')
            
            # 去噪结果
            axes[i, 1].imshow(sample['denoised'], cmap='gray')
            axes[i, 1].set_title(f'Denoised (ΔPSNR: {sample["psnr_improvement"]:+.2f}dB, SSIM: {sample["ssim_denoised"]:.3f})')
            axes[i, 1].axis('off')
            
            # 干净参考
            axes[i, 2].imshow(sample['clean'], cmap='gray')
            axes[i, 2].set_title('Clean Reference')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f'{category}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 生成SNR vs PSNR曲线图
    if snr_stats:
        snrs = sorted(snr_stats.keys())
        psnr_noisy = [snr_stats[s]['psnr_noisy'] for s in snrs]
        psnr_denoised = [snr_stats[s]['psnr_denoised'] for s in snrs]
        
        plt.figure(figsize=(10, 6))
        plt.plot(snrs, psnr_noisy, 'b-o', label='Noisy', linewidth=2, markersize=8)
        plt.plot(snrs, psnr_denoised, 'r-s', label='Denoised', linewidth=2, markersize=8)
        plt.xlabel('Input SNR (dB)', fontsize=12)
        plt.ylabel('PSNR (dB)', fontsize=12)
        plt.title('Denoising Performance vs Input SNR (Class-Conditional)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'snr_vs_psnr.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # PSNR改进曲线
        psnr_improvement = [snr_stats[s]['psnr_improvement'] for s in snrs]
        
        plt.figure(figsize=(10, 6))
        colors = ['green' if x > 0 else 'red' for x in psnr_improvement]
        plt.bar(snrs, psnr_improvement, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel('Input SNR (dB)', fontsize=12)
        plt.ylabel('PSNR Improvement (dB)', fontsize=12)
        plt.title('PSNR Improvement by Input SNR (Class-Conditional)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'psnr_improvement.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    # 生成类别性能对比图
    if category_stats:
        categories = sorted(category_stats.keys())
        psnr_improvements = [category_stats[c]['psnr_improvement'] for c in categories]
        ssim_values = [category_stats[c]['ssim_denoised'] for c in categories]
        
        # PSNR改进柱状图
        plt.figure(figsize=(14, 6))
        colors = ['green' if x > 0 else 'red' for x in psnr_improvements]
        bars = plt.bar(range(len(categories)), psnr_improvements, color=colors, alpha=0.7, edgecolor='black')
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.xlabel('Signal Category', fontsize=12)
        plt.ylabel('PSNR Improvement (dB)', fontsize=12)
        plt.title('PSNR Improvement by Signal Category', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'category_psnr_improvement.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # SSIM柱状图
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(categories)), ssim_values, color='steelblue', alpha=0.7, edgecolor='black')
        plt.xticks(range(len(categories)), categories, rotation=45, ha='right')
        plt.xlabel('Signal Category', fontsize=12)
        plt.ylabel('SSIM', fontsize=12)
        plt.title('SSIM by Signal Category (After Denoising)', fontsize=14, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'category_ssim.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to {vis_dir}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    evaluate_model(args)
