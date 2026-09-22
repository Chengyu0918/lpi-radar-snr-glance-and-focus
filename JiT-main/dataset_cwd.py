"""
CWD数据集加载器
用于加载含噪CWD图像对（noisy, clean）进行去噪训练

注意：您需要准备配对的数据集：
- noisy_dir: 含噪CWD图像目录
- clean_dir: 对应的干净CWD图像目录（可选，如果没有则使用自监督方式）
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path


class CWDPairedDataset(Dataset):
    """
    配对CWD数据集：同时加载含噪图和干净图
    
    目录结构应为：
    noisy_dir/
        class1/
            img1.png
            img2.png
        class2/
            ...
    clean_dir/
        class1/
            img1.png  # 与noisy对应
            img2.png
        class2/
            ...
    """
    def __init__(self, noisy_dir, clean_dir, img_size=256, transform=None):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir)
        self.img_size = img_size
        
        # 收集所有图像路径
        self.image_pairs = []
        
        # 遍历noisy目录
        for class_dir in sorted(self.noisy_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                clean_class_dir = self.clean_dir / class_name
                
                if clean_class_dir.exists():
                    for img_file in sorted(class_dir.glob('*.png')):
                        clean_img = clean_class_dir / img_file.name
                        if clean_img.exists():
                            self.image_pairs.append((str(img_file), str(clean_img)))
        
        print(f"Found {len(self.image_pairs)} image pairs")
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        noisy_path, clean_path = self.image_pairs[idx]
        
        # 加载图像
        noisy_img = Image.open(noisy_path).convert('L')  # 转为灰度图
        clean_img = Image.open(clean_path).convert('L')
        
        # 应用变换
        noisy_tensor = self.transform(noisy_img)
        clean_tensor = self.transform(clean_img)
        
        return clean_tensor, noisy_tensor


class CWDSelfSupervisedDataset(Dataset):
    """
    自监督CWD数据集：只有含噪图，通过添加额外噪声进行训练
    
    训练策略：
    - 将含噪图作为"干净"目标
    - 添加额外噪声作为输入条件
    - 模型学习去除额外添加的噪声
    
    目录结构：
    data_dir/
        class1/
            img1.png
            img2.png
        class2/
            ...
    """
    def __init__(self, data_dir, img_size=256, noise_level=0.1, transform=None):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.noise_level = noise_level
        
        # 收集所有图像路径
        self.image_paths = []
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                for img_file in sorted(class_dir.glob('*.png')):
                    self.image_paths.append(str(img_file))
        
        print(f"Found {len(self.image_paths)} images for self-supervised training")
        
        # 默认变换
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # 加载图像
        img = Image.open(img_path).convert('L')  # 转为灰度图
        
        # 应用变换
        img_tensor = self.transform(img)  # [1, H, W], 范围[0, 1]
        
        # 归一化到[-1, 1]
        clean_tensor = img_tensor * 2.0 - 1.0
        
        # 添加额外噪声作为"含噪"输入
        noise = torch.randn_like(clean_tensor) * self.noise_level
        noisy_tensor = clean_tensor + noise
        noisy_tensor = torch.clamp(noisy_tensor, -1.0, 1.0)
        
        return clean_tensor, noisy_tensor


class CWDInferenceDataset(Dataset):
    """
    推理用数据集：只加载含噪图像
    """
    def __init__(self, data_dir, img_size=256):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        
        # 收集所有图像路径
        self.image_paths = []
        self.image_names = []
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in sorted(class_dir.glob('*.png')):
                    self.image_paths.append(str(img_file))
                    self.image_names.append(f"{class_name}/{img_file.name}")
        
        print(f"Found {len(self.image_paths)} images for inference")
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = self.image_names[idx]
        
        img = Image.open(img_path).convert('L')
        img_tensor = self.transform(img)
        
        return img_tensor, img_name


def get_cwd_dataloader(
    data_dir,
    clean_dir=None,
    batch_size=16,
    img_size=256,
    num_workers=4,
    shuffle=True,
    mode='self_supervised',
    noise_level=0.1
):
    """
    获取CWD数据加载器
    
    Args:
        data_dir: 数据目录（含噪图像）
        clean_dir: 干净图像目录（可选，用于配对训练）
        batch_size: 批次大小
        img_size: 图像尺寸
        num_workers: 数据加载线程数
        shuffle: 是否打乱
        mode: 'paired' 或 'self_supervised'
        noise_level: 自监督模式下添加的噪声级别
    
    Returns:
        DataLoader
    """
    if mode == 'paired' and clean_dir is not None:
        dataset = CWDPairedDataset(data_dir, clean_dir, img_size)
    elif mode == 'self_supervised':
        dataset = CWDSelfSupervisedDataset(data_dir, img_size, noise_level)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_inference_dataloader(data_dir, batch_size=16, img_size=256, num_workers=4):
    """获取推理用数据加载器"""
    dataset = CWDInferenceDataset(data_dir, img_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataloader
