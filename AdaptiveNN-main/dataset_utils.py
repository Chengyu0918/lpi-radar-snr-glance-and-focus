import os
import re
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

# SNR级别配置：-20dB 到 0dB，步长2dB
DEFAULT_SNR_LEVELS = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0]


class SNRFilteredImageFolder(Dataset):
    """支持SNR过滤的图像文件夹数据集"""
    
    def __init__(self, root, transform=None, snr_filter=None):
        """
        Args:
            root: 数据集根目录
            transform: 图像变换
            snr_filter: SNR过滤列表，如 [-20, -18, ..., 0]，None表示不过滤
        """
        self.root = root
        self.transform = transform
        self.snr_filter = snr_filter
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        self._load_samples()
    
    def _load_samples(self):
        """加载样本，支持SNR过滤"""
        # 获取所有类别目录
        class_dirs = sorted([d for d in os.listdir(self.root) 
                           if os.path.isdir(os.path.join(self.root, d))])
        
        self.classes = class_dirs
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_dirs)}
        
        total_loaded = 0
        total_filtered = 0
        
        for class_name in class_dirs:
            class_path = os.path.join(self.root, class_name)
            class_idx = self.class_to_idx[class_name]
            
            for filename in os.listdir(class_path):
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    continue
                
                # 提取SNR信息
                match = re.search(r'_(-?\d+)dB_', filename)
                if match:
                    snr = int(match.group(1))
                    
                    # 如果指定了SNR过滤，只加载特定SNR的数据
                    if self.snr_filter is not None and snr not in self.snr_filter:
                        total_filtered += 1
                        continue
                
                filepath = os.path.join(class_path, filename)
                self.samples.append((filepath, class_idx))
                total_loaded += 1
        
        if self.snr_filter is not None:
            print(f"  SNR过滤: 加载 {total_loaded} 个样本, 过滤掉 {total_filtered} 个样本")
            print(f"  SNR范围: {min(self.snr_filter)}dB ~ {max(self.snr_filter)}dB")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        image = Image.open(filepath).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        
        # 使用SNR过滤的数据集
        # 默认使用 -20dB 到 0dB 的数据
        snr_filter = getattr(args, 'snr_filter', DEFAULT_SNR_LEVELS)
        
        print(f"使用SNR过滤数据集: {root}")
        dataset = SNRFilteredImageFolder(root, transform=transform, snr_filter=snr_filter)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
