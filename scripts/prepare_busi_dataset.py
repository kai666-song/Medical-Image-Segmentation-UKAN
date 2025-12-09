"""
BUSI 数据集清洗脚本
功能：
1. 将原图和 mask 分离到 images/ 和 masks/ 文件夹
2. 合并多个 mask 文件（多病灶情况）为单一 mask
3. 统一命名格式：{category}_{index:04d}.png
4. 可选：排除 normal 类别（无病灶）

输出结构：
datasets/BUSI_processed/
├── images/
│   ├── benign_0001.png
│   ├── malignant_0001.png
│   └── ...
└── masks/
    ├── benign_0001.png
    ├── malignant_0001.png
    └── ...
"""

import os
import re
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

# 配置
SRC_DIR = Path("./datasets/BUSI")
DST_DIR = Path("./datasets/BUSI_processed")
CATEGORIES = ["benign", "malignant"]  # 排除 normal（无病灶）
INCLUDE_NORMAL = False  # 是否包含 normal 类别


def parse_filename(filename: str):
    """
    解析文件名，提取类别、编号和 mask 信息
    例如：
    - "benign (1).png" -> ("benign", 1, False, None)
    - "benign (1)_mask.png" -> ("benign", 1, True, 0)
    - "benign (1)_mask_1.png" -> ("benign", 1, True, 1)
    """
    # 匹配原图: category (number).png
    img_pattern = r"^(benign|malignant|normal) \((\d+)\)\.png$"
    # 匹配 mask: category (number)_mask.png 或 category (number)_mask_N.png
    mask_pattern = r"^(benign|malignant|normal) \((\d+)\)_mask(?:_(\d+))?\.png$"
    
    img_match = re.match(img_pattern, filename)
    if img_match:
        category, idx = img_match.groups()
        return category, int(idx), False, None
    
    mask_match = re.match(mask_pattern, filename)
    if mask_match:
        category, idx, mask_idx = mask_match.groups()
        mask_idx = int(mask_idx) if mask_idx else 0
        return category, int(idx), True, mask_idx
    
    return None, None, None, None


def merge_masks(mask_paths: list) -> np.ndarray:
    """
    合并多个 mask 文件（OR 操作）
    """
    merged = None
    for path in mask_paths:
        mask = np.array(Image.open(path).convert("L"))
        if merged is None:
            merged = mask
        else:
            merged = np.maximum(merged, mask)
    return merged


def process_dataset():
    """
    处理数据集
    """
    # 创建输出目录
    images_dir = DST_DIR / "images"
    masks_dir = DST_DIR / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    categories = CATEGORIES + (["normal"] if INCLUDE_NORMAL else [])
    
    stats = {
        "total_images": 0,
        "multi_mask_cases": 0,
        "by_category": defaultdict(int)
    }
    
    for category in categories:
        src_category_dir = SRC_DIR / category
        if not src_category_dir.exists():
            print(f"[WARN] 目录不存在: {src_category_dir}")
            continue
        
        # 收集所有文件
        files = list(src_category_dir.glob("*.png"))
        
        # 按编号分组
        grouped = defaultdict(lambda: {"image": None, "masks": []})
        
        for f in files:
            cat, idx, is_mask, mask_idx = parse_filename(f.name)
            if cat is None:
                print(f"[WARN] 无法解析文件名: {f.name}")
                continue
            
            if is_mask:
                grouped[idx]["masks"].append((mask_idx, f))
            else:
                grouped[idx]["image"] = f
        
        # 处理每个样本
        for idx in sorted(grouped.keys()):
            data = grouped[idx]
            
            if data["image"] is None:
                print(f"[WARN] {category} ({idx}) 缺少原图")
                continue
            
            if not data["masks"] and category != "normal":
                print(f"[WARN] {category} ({idx}) 缺少 mask")
                continue
            
            # 生成新文件名
            new_name = f"{category}_{idx:04d}.png"
            
            # 复制原图
            shutil.copy2(data["image"], images_dir / new_name)
            
            # 处理 mask
            if data["masks"]:
                # 按 mask_idx 排序
                data["masks"].sort(key=lambda x: x[0])
                mask_paths = [p for _, p in data["masks"]]
                
                if len(mask_paths) > 1:
                    # 多个 mask，需要合并
                    stats["multi_mask_cases"] += 1
                    merged = merge_masks(mask_paths)
                    Image.fromarray(merged).save(masks_dir / new_name)
                else:
                    # 单个 mask，直接复制
                    shutil.copy2(mask_paths[0], masks_dir / new_name)
            elif category == "normal":
                # normal 类别创建全黑 mask
                img = Image.open(data["image"])
                black_mask = Image.new("L", img.size, 0)
                black_mask.save(masks_dir / new_name)
            
            stats["total_images"] += 1
            stats["by_category"][category] += 1
    
    return stats


def generate_split_file(train_ratio=0.7, val_ratio=0.15):
    """
    生成训练/验证/测试集划分文件
    """
    images_dir = DST_DIR / "images"
    all_images = sorted([f.stem for f in images_dir.glob("*.png")])
    
    np.random.seed(42)
    np.random.shuffle(all_images)
    
    n = len(all_images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_set = all_images[:n_train]
    val_set = all_images[n_train:n_train + n_val]
    test_set = all_images[n_train + n_val:]
    
    # 保存划分文件
    for name, data in [("train", train_set), ("val", val_set), ("test", test_set)]:
        with open(DST_DIR / f"{name}.txt", "w") as f:
            f.write("\n".join(data))
    
    return len(train_set), len(val_set), len(test_set)


if __name__ == "__main__":
    print("=" * 50)
    print("BUSI 数据集清洗脚本")
    print("=" * 50)
    
    print(f"\n源目录: {SRC_DIR}")
    print(f"目标目录: {DST_DIR}")
    print(f"包含类别: {CATEGORIES + (['normal'] if INCLUDE_NORMAL else [])}")
    
    print("\n[1/2] 处理数据集...")
    stats = process_dataset()
    
    print(f"\n处理完成:")
    print(f"  - 总图像数: {stats['total_images']}")
    print(f"  - 多 mask 合并: {stats['multi_mask_cases']} 例")
    print(f"  - 各类别统计:")
    for cat, count in stats["by_category"].items():
        print(f"      {cat}: {count}")
    
    print("\n[2/2] 生成数据集划分...")
    n_train, n_val, n_test = generate_split_file()
    print(f"  - 训练集: {n_train}")
    print(f"  - 验证集: {n_val}")
    print(f"  - 测试集: {n_test}")
    
    print("\n" + "=" * 50)
    print("完成！输出目录结构:")
    print(f"  {DST_DIR}/")
    print(f"  ├── images/")
    print(f"  ├── masks/")
    print(f"  ├── train.txt")
    print(f"  ├── val.txt")
    print(f"  └── test.txt")
    print("=" * 50)
