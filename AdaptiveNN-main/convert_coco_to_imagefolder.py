"""
将COCO数据集转换为ImageFolder格式
用于AdaptiveNN分类任务
"""

import os
import json
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

def convert_coco_to_imagefolder(
    coco_dir="coco_data",
    output_dir="coco_imagefolder",
    split="val2017",
    max_images_per_class=100,
    min_images_per_class=50
):
    """
    将COCO数据集转换为ImageFolder格式
    
    参数:
        coco_dir: COCO数据集目录
        output_dir: 输出目录
        split: 数据集分割 (val2017 或 train2017)
        max_images_per_class: 每个类别最多保留的图像数
        min_images_per_class: 每个类别最少需要的图像数
    """
    
    print("=" * 60)
    print("COCO to ImageFolder 转换工具")
    print("=" * 60)
    print()
    
    # 路径设置
    images_dir = os.path.join(coco_dir, split)
    annotations_file = os.path.join(coco_dir, "annotations", f"instances_{split}.json")
    
    # 检查文件是否存在
    if not os.path.exists(images_dir):
        print(f"✗ 错误: 找不到图像目录 {images_dir}")
        print("请先运行 download_coco.py 下载数据")
        return
    
    if not os.path.exists(annotations_file):
        print(f"✗ 错误: 找不到标注文件 {annotations_file}")
        print("请先运行 download_coco.py 下载数据")
        return
    
    print(f"✓ 图像目录: {images_dir}")
    print(f"✓ 标注文件: {annotations_file}")
    print()
    
    # 加载COCO标注
    print("加载COCO标注...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # 创建类别映射
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print(f"✓ 找到 {len(categories)} 个类别")
    
    # 统计每个图像的类别
    image_to_categories = defaultdict(set)
    for ann in coco_data['annotations']:
        image_to_categories[ann['image_id']].add(ann['category_id'])
    
    # 为每个类别收集图像
    category_to_images = defaultdict(list)
    for img in coco_data['images']:
        img_id = img['id']
        if img_id in image_to_categories:
            # 只使用主要类别（第一个类别）
            main_category = list(image_to_categories[img_id])[0]
            category_to_images[main_category].append(img['file_name'])
    
    # 过滤类别（保留图像数量足够的类别）
    valid_categories = {
        cat_id: cat_name 
        for cat_id, cat_name in categories.items()
        if len(category_to_images[cat_id]) >= min_images_per_class
    }
    
    print(f"✓ 过滤后保留 {len(valid_categories)} 个类别（每类至少{min_images_per_class}张图像）")
    print()
    
    # 创建输出目录
    if os.path.exists(output_dir):
        response = input(f"输出目录 {output_dir} 已存在，是否删除并重新创建? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(output_dir)
        else:
            print("取消转换")
            return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制图像到对应类别文件夹
    print("开始转换...")
    total_images = 0
    
    for cat_id, cat_name in tqdm(valid_categories.items(), desc="处理类别"):
        # 创建类别文件夹（使用安全的文件名）
        safe_cat_name = cat_name.replace(' ', '_').replace('/', '_')
        cat_dir = os.path.join(output_dir, safe_cat_name)
        os.makedirs(cat_dir, exist_ok=True)
        
        # 复制图像（限制数量）
        images = category_to_images[cat_id][:max_images_per_class]
        
        for img_name in images:
            src = os.path.join(images_dir, img_name)
            dst = os.path.join(cat_dir, img_name)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                total_images += 1
    
    print()
    print("=" * 60)
    print("转换完成！")
    print("=" * 60)
    print(f"\n统计信息:")
    print(f"- 类别数: {len(valid_categories)}")
    print(f"- 总图像数: {total_images}")
    print(f"- 平均每类: {total_images // len(valid_categories)} 张")
    print(f"\n输出目录: {os.path.abspath(output_dir)}")
    print(f"\n目录结构:")
    print(f"{output_dir}/")
    for cat_name in list(valid_categories.values())[:5]:
        safe_name = cat_name.replace(' ', '_').replace('/', '_')
        print(f"├── {safe_name}/")
    print("└── ...")
    print("\n下一步:")
    print(f"运行训练: python main_ppo.py --data_path {output_dir} --nb_classes {len(valid_categories)}")
    print("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将COCO转换为ImageFolder格式')
    parser.add_argument('--coco_dir', default='coco_data', help='COCO数据集目录')
    parser.add_argument('--output_dir', default='coco_imagefolder', help='输出目录')
    parser.add_argument('--split', default='val2017', choices=['val2017', 'train2017'], 
                        help='数据集分割')
    parser.add_argument('--max_per_class', type=int, default=100, 
                        help='每个类别最多保留的图像数')
    parser.add_argument('--min_per_class', type=int, default=50,
                        help='每个类别最少需要的图像数')
    
    args = parser.parse_args()
    
    convert_coco_to_imagefolder(
        coco_dir=args.coco_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_images_per_class=args.max_per_class,
        min_images_per_class=args.min_per_class
    )

if __name__ == "__main__":
    main()
