"""
带类别条件的CWD去噪模型训练脚本

使用方法：
python train_cwd_class.py --data_path ../../../CWDtrain_noisy --clean_path ../../../CWDtrain_clean --output_dir ./output_cwd_class
"""
import argparse
import os
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from denoiser_cwd_class import CWDDenoiserWithClass


class CWDPairedDatasetWithClass(Dataset):
    """带类别标签的配对CWD数据集"""
    def __init__(self, noisy_dir, clean_dir, img_size=224, class_names=None):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.img_size = img_size
        
        # 类别名称
        if class_names is None:
            self.class_names = [
                'BPSK', 'Costas', 'CP', 'Frank', 
                'FSK4_Baker5', 'FSK4_LFM', 'LFM', 'NLFM',
                'P1', 'P2', 'P3', 'P4', 
                'T1', 'T2', 'T3', 'T4'
            ]
        else:
            self.class_names = class_names
        
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 收集所有图像对
        self.pairs = []
        for class_name in self.class_names:
            noisy_class_dir = self.noisy_dir / class_name
            clean_class_dir = self.clean_dir / class_name
            
            if not noisy_class_dir.exists() or not clean_class_dir.exists():
                continue
            
            class_idx = self.class_to_idx[class_name]
            
            for noisy_path in noisy_class_dir.glob('*.png'):
                clean_path = clean_class_dir / noisy_path.name
                if clean_path.exists():
                    self.pairs.append((noisy_path, clean_path, class_idx))
        
        print(f"Found {len(self.pairs)} image pairs across {len(self.class_names)} classes")
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy_path, clean_path, class_idx = self.pairs[idx]
        
        noisy_img = Image.open(noisy_path).convert('L')
        clean_img = Image.open(clean_path).convert('L')
        
        noisy_tensor = self.transform(noisy_img)
        clean_tensor = self.transform(clean_img)
        
        return noisy_tensor, clean_tensor, class_idx


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-CWD Class-Conditional Training', add_help=False)
    
    # 模型参数
    parser.add_argument('--model', default='JiT-CWD-Class-Tiny', type=str,
                        choices=['JiT-CWD-Class-Tiny', 'JiT-CWD-Class-Small', 'JiT-CWD-Class-Base'])
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--class_dropout_prob', default=0.1, type=float,
                        help='类别标签dropout概率，用于classifier-free guidance')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    
    # 扩散参数
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.05, type=float)
    
    # 采样参数
    parser.add_argument('--sampling_method', default='heun', type=str)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    
    # 训练参数
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    
    # EMA参数
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    
    # 数据参数
    parser.add_argument('--data_path', required=True, type=str, help='含噪图像目录')
    parser.add_argument('--clean_path', required=True, type=str, help='干净图像目录')
    parser.add_argument('--output_dir', default='./output_cwd_class', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # 其他
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--save_every', default=10, type=int)
    parser.add_argument('--sample_every', default=10, type=int)
    
    return parser


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """更新EMA模型"""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


def save_samples(model, dataloader, device, output_dir, epoch, num_samples=4):
    """保存去噪样本"""
    model.eval()
    sample_dir = os.path.join(output_dir, f'samples_epoch{epoch}')
    os.makedirs(sample_dir, exist_ok=True)
    
    # 获取一个batch
    noisy, clean, class_labels = next(iter(dataloader))
    noisy = noisy[:num_samples].to(device)
    clean = clean[:num_samples].to(device)
    class_labels = class_labels[:num_samples].to(device)
    
    # 去噪
    denoised = model.denoise(noisy, class_labels, cfg_scale=2.0)
    
    # 保存图像
    for i in range(num_samples):
        # 反归一化
        noisy_np = (noisy[i, 0].cpu().numpy() + 1) / 2 * 255
        clean_np = (clean[i, 0].cpu().numpy() + 1) / 2 * 255
        denoised_np = (denoised[i, 0].cpu().numpy() + 1) / 2 * 255
        
        noisy_np = np.clip(noisy_np, 0, 255).astype(np.uint8)
        clean_np = np.clip(clean_np, 0, 255).astype(np.uint8)
        denoised_np = np.clip(denoised_np, 0, 255).astype(np.uint8)
        
        # 拼接保存
        concat = np.concatenate([noisy_np, denoised_np, clean_np], axis=1)
        Image.fromarray(concat).save(os.path.join(sample_dir, f'sample_{i}_class{class_labels[i].item()}.png'))
    
    model.train()


def main(args):
    print('=' * 50)
    print('JiT-CWD Class-Conditional Training')
    print('=' * 50)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建数据集
    dataset = CWDPairedDatasetWithClass(
        noisy_dir=args.data_path,
        clean_dir=args.clean_path,
        img_size=args.img_size
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型
    model = CWDDenoiserWithClass(args)
    model.to(device)
    
    # 创建EMA模型
    ema_model1 = copy.deepcopy(model)
    ema_model2 = copy.deepcopy(model)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # TensorBoard
    writer = SummaryWriter(args.output_dir)
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint_path = os.path.join(args.resume, 'checkpoint-last.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            ema_model1.load_state_dict(checkpoint['model_ema1'])
            ema_model2.load_state_dict(checkpoint['model_ema2'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resumed from epoch {start_epoch}')
    
    # 训练循环
    print(f'\nStarting training for {args.epochs} epochs...')
    global_step = start_epoch * len(dataloader)
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')
        for noisy, clean, class_labels in pbar:
            noisy = noisy.to(device)
            clean = clean.to(device)
            class_labels = class_labels.to(device)
            
            # 前向传播
            loss = model(clean, noisy, class_labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # 更新EMA
            update_ema(ema_model1, model, args.ema_decay1)
            update_ema(ema_model2, model, args.ema_decay2)
            
            epoch_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            writer.add_scalar('train/loss', loss.item(), global_step)
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch}: avg_loss = {avg_loss:.4f}')
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch)
        
        # 保存样本
        if (epoch + 1) % args.sample_every == 0:
            save_samples(ema_model1, dataloader, device, args.output_dir, epoch)
        
        # 保存检查点
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'model_ema1': ema_model1.state_dict(),
                'model_ema2': ema_model2.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': args,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'checkpoint-last.pth'))
            print(f'Saved checkpoint at epoch {epoch}')
    
    writer.close()
    print('Training completed!')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
