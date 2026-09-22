"""
带类别条件的CWD去噪推理脚本
使用Classifier-Free Guidance进行条件去噪
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from denoiser_cwd_class import CWDDenoiserWithClass

# 类别映射
CLASS_NAMES = ['BPSK', 'Costas', 'CP', 'Frank', 'FSK4_Baker5', 'FSK4_LFM', 
               'LFM', 'NLFM', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_image(image_path, image_size=224):
    """加载并预处理图像为灰度图"""
    image = Image.open(image_path).convert('L')  # 转为灰度
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    
    # 转为tensor并归一化到[-1, 1]
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = image_np * 2 - 1  # [0,1] -> [-1,1]
    
    tensor = torch.from_numpy(image_np).unsqueeze(0)  # [1, H, W]
    return tensor


def save_image(tensor, save_path):
    """保存张量为图像"""
    # 反归一化 [-1,1] -> [0,1]
    tensor = (tensor + 1) / 2
    tensor = tensor.clamp(0, 1)
    
    # 转换为numpy
    if tensor.dim() == 4:
        tensor = tensor[0]  # 移除batch维度
    if tensor.shape[0] == 1:
        tensor = tensor[0]  # 移除channel维度
    
    image_np = tensor.cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = Image.fromarray(image_np, mode='L')
    image.save(save_path)


def get_class_from_path(image_path):
    """从文件路径推断类别"""
    path_parts = image_path.replace('\\', '/').split('/')
    for part in path_parts:
        if part in CLASS_TO_IDX:
            return CLASS_TO_IDX[part]
    
    # 从文件名推断
    filename = os.path.basename(image_path)
    for class_name in CLASS_NAMES:
        if filename.startswith(class_name):
            return CLASS_TO_IDX[class_name]
    
    return None


def denoise_single_image(denoiser, image_path, output_path, class_label=None, 
                         cfg_scale=2.0, image_size=224, device='cuda'):
    """对单张图像进行去噪"""
    # 加载图像
    noisy_image = load_image(image_path, image_size).unsqueeze(0).to(device)
    
    # 确定类别
    if class_label is None:
        class_label = get_class_from_path(image_path)
        if class_label is None:
            print(f"警告: 无法从路径推断类别，使用默认类别0 (BPSK)")
            class_label = 0
    
    class_labels = torch.tensor([class_label], device=device, dtype=torch.long)
    
    # 去噪
    with torch.no_grad():
        denoised = denoiser.denoise(noisy_image, class_labels, cfg_scale=cfg_scale)
    
    # 保存结果
    save_image(denoised, output_path)
    
    return CLASS_NAMES[class_label]


def denoise_directory(denoiser, input_dir, output_dir, cfg_scale=2.0, 
                      image_size=224, device='cuda'):
    """对目录中的所有图像进行去噪"""
    processed = 0
    # 遍历所有子目录和图像
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                input_path = os.path.join(root, file)
                
                # 构建输出路径，保持目录结构
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # 去噪
                class_name = denoise_single_image(
                    denoiser, input_path, output_path,
                    cfg_scale=cfg_scale, image_size=image_size, device=device
                )
                processed += 1
                if processed % 10 == 0:
                    print(f"已处理 {processed} 张图像...")
                    
    return processed


def main():
    parser = argparse.ArgumentParser(description='带类别条件的CWD去噪推理')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--input', type=str, required=True,
                        help='输入图像或目录路径')
    parser.add_argument('--output', type=str, required=True,
                        help='输出图像或目录路径')
    parser.add_argument('--class_label', type=str, default=None,
                        help='指定类别名称（如BPSK, LFM等），不指定则自动推断')
    parser.add_argument('--cfg_scale', type=float, default=2.0,
                        help='Classifier-Free Guidance强度 (默认: 2.0)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='图像尺寸 (默认: 224)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    denoiser = CWDDenoiserWithClass.from_checkpoint(args.checkpoint, device=args.device)
    print("模型加载完成")
    
    # 解析类别标签
    class_label = None
    if args.class_label is not None:
        if args.class_label in CLASS_TO_IDX:
            class_label = CLASS_TO_IDX[args.class_label]
        else:
            print(f"警告: 未知类别 '{args.class_label}'，将自动推断")
    
    # 判断输入是文件还是目录
    if os.path.isfile(args.input):
        # 单张图像
        class_name = denoise_single_image(
            denoiser, args.input, args.output,
            class_label=class_label,
            cfg_scale=args.cfg_scale,
            image_size=args.image_size,
            device=args.device
        )
        print(f"去噪完成: {args.output} (类别: {class_name})")
    else:
        # 目录
        total = denoise_directory(
            denoiser, args.input, args.output,
            cfg_scale=args.cfg_scale,
            image_size=args.image_size,
            device=args.device
        )
        print(f"批量去噪完成，共处理 {total} 张图像，结果保存在: {args.output}")


if __name__ == '__main__':
    main()
