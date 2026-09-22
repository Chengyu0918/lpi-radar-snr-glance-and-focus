"""
CWD去噪训练脚本
针对单卡RTX 4080优化

使用方法：
python train_cwd.py --data_path ../../CWDtrain --img_size 224 --batch_size 8 --epochs 100
"""
import argparse
import datetime
import os
import time
import copy
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from denoiser_cwd import CWDDenoiser
from dataset_cwd import get_cwd_dataloader, get_inference_dataloader
from PIL import Image


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-CWD Denoising Training', add_help=False)

    # 模型架构
    parser.add_argument('--model', default='JiT-CWD-Tiny', type=str,
                        choices=['JiT-CWD-Tiny', 'JiT-CWD-Small', 'JiT-CWD-Base'],
                        help='模型大小，Tiny适合4080调试')
    parser.add_argument('--img_size', default=224, type=int, 
                        help='图像尺寸，CWD图为224x224')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='输入通道数，CWD图为单通道')
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)

    # 训练参数
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=8, type=int,
                        help='批次大小，4080建议8-16')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    
    # EMA参数
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    
    # 扩散参数
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.05, type=float)

    # 采样参数
    parser.add_argument('--sampling_method', default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--num_sampling_steps', default=50, type=int)

    # 数据参数
    parser.add_argument('--data_path', default='../../CWDtrain', type=str,
                        help='训练数据路径')
    parser.add_argument('--clean_path', default=None, type=str,
                        help='干净图像路径（可选，用于配对训练）')
    parser.add_argument('--train_mode', default='self_supervised', type=str,
                        choices=['paired', 'self_supervised'],
                        help='训练模式')
    parser.add_argument('--added_noise_level', default=0.1, type=float,
                        help='自监督模式下添加的噪声级别')
    parser.add_argument('--num_workers', default=4, type=int)

    # 保存和日志
    parser.add_argument('--output_dir', default='./output_cwd',
                        help='输出目录')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='保存频率（epochs）')
    parser.add_argument('--log_freq', default=50, type=int,
                        help='日志频率（iterations）')
    parser.add_argument('--eval_freq', type=int, default=20,
                        help='评估频率（epochs）')

    # 其他
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--resume', default='', type=str,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--use_amp', action='store_true',
                        help='使用混合精度训练')

    return parser


def adjust_learning_rate(optimizer, epoch, args):
    """学习率调度：warmup + cosine decay"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.lr * 0.5 * (1.0 + np.cos(np.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def save_checkpoint(model, optimizer, epoch, args, filename='checkpoint.pth'):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': args,
    }
    
    # 保存EMA参数
    if model.ema_params1 is not None:
        ema_state_dict1 = {}
        ema_state_dict2 = {}
        for i, (name, _) in enumerate(model.named_parameters()):
            ema_state_dict1[name] = model.ema_params1[i]
            ema_state_dict2[name] = model.ema_params2[i]
        checkpoint['model_ema1'] = ema_state_dict1
        checkpoint['model_ema2'] = ema_state_dict2
    
    save_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(model, optimizer, args):
    """加载checkpoint"""
    checkpoint_path = os.path.join(args.resume, 'checkpoint-last.pth')
    if not os.path.exists(checkpoint_path):
        checkpoint_path = args.resume
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # 加载EMA参数
        if 'model_ema1' in checkpoint:
            ema_state_dict1 = checkpoint['model_ema1']
            ema_state_dict2 = checkpoint['model_ema2']
            model.ema_params1 = [ema_state_dict1[name].cuda() for name, _ in model.named_parameters()]
            model.ema_params2 = [ema_state_dict2[name].cuda() for name, _ in model.named_parameters()]
        
        print(f"Resumed from epoch {start_epoch}")
        return start_epoch
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return 0


def train_one_epoch(model, dataloader, optimizer, device, epoch, log_writer, args):
    """训练一个epoch"""
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    
    for batch_idx, (clean_cwd, noisy_cwd) in enumerate(dataloader):
        clean_cwd = clean_cwd.to(device)
        noisy_cwd = noisy_cwd.to(device)
        
        optimizer.zero_grad()
        
        if args.use_amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(clean_cwd, noisy_cwd)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(clean_cwd, noisy_cwd)
            loss.backward()
            optimizer.step()
        
        # 更新EMA
        model.update_ema()
        
        total_loss += loss.item()
        num_batches += 1
        
        # 日志
        if batch_idx % args.log_freq == 0:
            print(f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.6f}")
            
            if log_writer is not None:
                global_step = epoch * len(dataloader) + batch_idx
                log_writer.add_scalar('train/loss', loss.item(), global_step)
    
    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def evaluate(model, dataloader, device, epoch, output_dir, num_samples=8):
    """评估：生成去噪样本"""
    model.eval()
    
    # 获取一个batch
    for noisy_cwd, _ in dataloader:
        noisy_cwd = noisy_cwd[:num_samples].to(device)
        break
    
    # 切换到EMA参数
    original_state = copy.deepcopy(model.state_dict())
    if model.ema_params1 is not None:
        ema_state_dict = copy.deepcopy(model.state_dict())
        for i, (name, _) in enumerate(model.named_parameters()):
            ema_state_dict[name] = model.ema_params1[i]
        model.load_state_dict(ema_state_dict)
    
    # 去噪
    denoised = model.denoise(noisy_cwd)
    
    # 恢复原始参数
    model.load_state_dict(original_state)
    
    # 保存结果
    save_dir = os.path.join(output_dir, f'samples_epoch{epoch}')
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_samples, noisy_cwd.size(0))):
        # 反归一化
        noisy_img = (noisy_cwd[i, 0].cpu().numpy() + 1) / 2 * 255
        denoised_img = (denoised[i, 0].cpu().numpy() + 1) / 2 * 255
        
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
        
        # 保存
        Image.fromarray(noisy_img).save(os.path.join(save_dir, f'{i}_noisy.png'))
        Image.fromarray(denoised_img).save(os.path.join(save_dir, f'{i}_denoised.png'))
    
    print(f"Saved {num_samples} samples to {save_dir}")


def main(args):
    print('=' * 50)
    print('JiT-CWD Denoising Training')
    print('=' * 50)
    print(f"Arguments:\n{args}")
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # TensorBoard
    log_writer = SummaryWriter(log_dir=args.output_dir)
    
    # 创建数据加载器
    print(f"\nLoading data from {args.data_path}")
    train_loader = get_cwd_dataloader(
        data_dir=args.data_path,
        clean_dir=args.clean_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=True,
        mode=args.train_mode,
        noise_level=args.added_noise_level
    )
    
    # 创建推理用数据加载器（用于评估）
    eval_loader = get_inference_dataloader(
        data_dir=args.data_path,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )
    
    # 创建模型
    print(f"\nCreating model: {args.model}")
    model = CWDDenoiser(args)
    model.to(device)
    
    # 打印模型信息
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_params / 1e6:.2f}M")
    
    # 初始化EMA参数
    model.ema_params1 = copy.deepcopy(list(model.parameters()))
    model.ema_params2 = copy.deepcopy(list(model.parameters()))
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args)
    
    # 训练循环
    print(f"\nStarting training for {args.epochs} epochs")
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        # 调整学习率
        lr = adjust_learning_rate(optimizer, epoch, args)
        print(f"\nEpoch {epoch}, LR: {lr:.6f}")
        
        # 训练
        avg_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, log_writer, args
        )
        print(f"Epoch {epoch} average loss: {avg_loss:.6f}")
        
        # 保存checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
            save_checkpoint(model, optimizer, epoch, args, 'checkpoint-last.pth')
        
        # 评估
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            evaluate(model, eval_loader, device, epoch, args.output_dir)
        
        # 记录到TensorBoard
        log_writer.add_scalar('train/epoch_loss', avg_loss, epoch)
        log_writer.add_scalar('train/lr', lr, epoch)
        log_writer.flush()
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    
    # 保存最终模型
    save_checkpoint(model, optimizer, args.epochs - 1, args, 'checkpoint-final.pth')
    
    log_writer.close()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
