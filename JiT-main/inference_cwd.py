"""
CWD去噪推理脚本
用于对含噪CWD图像进行去噪

使用方法：
python inference_cwd.py --checkpoint ./output_cwd/checkpoint-final.pth --input ../../CWDtest --output ./denoised_output
"""
import argparse
import os
import copy
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from denoiser_cwd import CWDDenoiser
from dataset_cwd import get_inference_dataloader


def get_args_parser():
    parser = argparse.ArgumentParser('JiT-CWD Denoising Inference', add_help=False)
    
    # 模型参数（需要与训练时一致）
    parser.add_argument('--model', default='JiT-CWD-Tiny', type=str,
                        choices=['JiT-CWD-Tiny', 'JiT-CWD-Small', 'JiT-CWD-Base'])
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--attn_dropout', type=float, default=0.0)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    
    # 扩散参数
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=1.0, type=float)
    parser.add_argument('--t_eps', default=0.05, type=float)
    
    # EMA参数
    parser.add_argument('--ema_decay1', type=float, default=0.9999)
    parser.add_argument('--ema_decay2', type=float, default=0.9996)
    
    # 采样参数
    parser.add_argument('--sampling_method', default='heun', type=str,
                        choices=['euler', 'heun'])
    parser.add_argument('--num_sampling_steps', default=50, type=int,
                        help='采样步数，越多质量越好但越慢')
    
    # 输入输出
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='模型checkpoint路径')
    parser.add_argument('--input', required=True, type=str,
                        help='输入图像目录')
    parser.add_argument('--output', default='./denoised_output', type=str,
                        help='输出目录')
    
    # 其他
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='使用EMA参数进行推理')
    
    return parser


def load_model(args):
    """加载模型和checkpoint"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model = CWDDenoiser(args)
    model.to(device)
    
    # 加载checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    
    # 加载EMA参数
    if args.use_ema and 'model_ema1' in checkpoint:
        ema_state_dict = checkpoint['model_ema1']
        current_state = model.state_dict()
        for name in ema_state_dict:
            if name in current_state:
                current_state[name] = ema_state_dict[name]
        model.load_state_dict(current_state)
        print("Loaded EMA parameters")
    
    model.eval()
    print(f"Loaded model from {args.checkpoint}")
    
    return model, device


def denoise_images(model, dataloader, device, output_dir):
    """对所有图像进行去噪"""
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for noisy_batch, names in tqdm(dataloader, desc="Denoising"):
            noisy_batch = noisy_batch.to(device)
            
            # 去噪
            denoised_batch = model.denoise(noisy_batch)
            
            # 保存结果
            for i, name in enumerate(names):
                # 反归一化
                denoised_img = (denoised_batch[i, 0].cpu().numpy() + 1) / 2 * 255
                denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
                
                # 创建子目录
                save_path = os.path.join(output_dir, name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # 保存
                Image.fromarray(denoised_img).save(save_path)
    
    print(f"Saved denoised images to {output_dir}")


def denoise_single_image(model, image_path, output_path, device, img_size=224):
    """对单张图像进行去噪"""
    import torchvision.transforms as transforms
    
    # 加载图像
    img = Image.open(image_path).convert('L')
    original_size = img.size
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 去噪
    with torch.no_grad():
        denoised = model.denoise(img_tensor)
    
    # 后处理
    denoised_img = (denoised[0, 0].cpu().numpy() + 1) / 2 * 255
    denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
    
    # 恢复原始尺寸
    denoised_pil = Image.fromarray(denoised_img)
    denoised_pil = denoised_pil.resize(original_size, Image.BILINEAR)
    
    # 保存
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    denoised_pil.save(output_path)
    
    print(f"Saved denoised image to {output_path}")


def main(args):
    print('=' * 50)
    print('JiT-CWD Denoising Inference')
    print('=' * 50)
    
    # 加载模型
    model, device = load_model(args)
    
    # 检查输入是文件还是目录
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 单张图像
        output_path = args.output if args.output.endswith('.png') else os.path.join(args.output, input_path.name)
        denoise_single_image(model, str(input_path), output_path, device, args.img_size)
    else:
        # 目录
        dataloader = get_inference_dataloader(
            data_dir=args.input,
            batch_size=args.batch_size,
            img_size=args.img_size,
            num_workers=args.num_workers
        )
        denoise_images(model, dataloader, device, args.output)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
