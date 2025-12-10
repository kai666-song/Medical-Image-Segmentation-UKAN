"""
U-KAN 预测脚本 - 生成分割结果可视化
"""
import argparse
import os
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from albumentations import Resize
from albumentations.core.composition import Compose
from albumentations.augmentations import transforms

import archs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='实验名称')
    parser.add_argument('--arch', default='UKAN', help='模型架构: UKAN 或 UKAN_CBAM')
    parser.add_argument('--model_path', default=None, help='模型路径，默认使用 outputs/{name}/model.pth')
    parser.add_argument('--data_dir', default='../datasets', help='数据集目录')
    parser.add_argument('--dataset', default='BUSI_processed', help='数据集名称')
    parser.add_argument('--output_dir', default='../outputs', help='输出目录')
    parser.add_argument('--input_size', default=256, type=int, help='输入尺寸')
    parser.add_argument('--num_samples', default=20, type=int, help='可视化样本数量，-1表示全部')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 路径设置
    exp_dir = f'{args.output_dir}/{args.name}'
    model_path = args.model_path or f'{exp_dir}/model.pth'
    result_dir = f'{exp_dir}/predictions'
    os.makedirs(result_dir, exist_ok=True)
    
    # 自动检测模型架构
    arch = args.arch
    if 'cbam' in args.name.lower() and arch == 'UKAN':
        arch = 'UKAN_CBAM'
    
    # 加载模型 (使用训练时的 embed_dims=[128, 160, 256])
    print(f'Loading model from {model_path}')
    print(f'Architecture: {arch}')
    model_class = getattr(archs, arch)
    model = model_class(num_classes=1, input_channels=3, deep_supervision=False, 
                        embed_dims=[128, 160, 256])
    model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=True))
    model = model.cuda()
    model.eval()
    
    # 数据预处理
    transform = Compose([
        Resize(args.input_size, args.input_size),
        transforms.Normalize(),
    ])
    
    # 获取图像列表
    img_dir = os.path.join(args.data_dir, args.dataset, 'images')
    mask_dir = os.path.join(args.data_dir, args.dataset, 'masks', '0')
    img_paths = sorted(glob(os.path.join(img_dir, '*.png')))
    
    if args.num_samples > 0:
        img_paths = img_paths[:args.num_samples]
    
    print(f'Processing {len(img_paths)} images...')
    
    with torch.no_grad():
        for img_path in tqdm(img_paths):
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # 读取原图和 mask
            img_orig = cv2.imread(img_path)
            mask_path = os.path.join(mask_dir, f'{img_id}.png')
            mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if os.path.exists(mask_path) else None
            
            # 预处理
            augmented = transform(image=img_orig)
            img = augmented['image']
            img = img.astype('float32') / 255
            img = img.transpose(2, 0, 1)
            img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()
            
            # 预测
            output = model(img_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_binary = (pred > 0.5).astype(np.uint8) * 255
            
            # 调整尺寸回原图大小
            h, w = img_orig.shape[:2]
            pred_resized = cv2.resize(pred_binary, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 创建可视化图像
            # 原图 | GT Mask | 预测 Mask | 叠加效果
            img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            
            # GT mask 可视化
            if mask_gt is not None:
                mask_gt_vis = cv2.cvtColor(mask_gt, cv2.COLOR_GRAY2RGB)
            else:
                mask_gt_vis = np.zeros_like(img_rgb)
            
            # 预测 mask 可视化
            pred_vis = cv2.cvtColor(pred_resized, cv2.COLOR_GRAY2RGB)
            
            # 叠加效果：红色为预测，绿色为GT
            overlay = img_orig.copy()
            overlay[pred_resized > 127] = [0, 0, 255]  # 红色预测区域
            if mask_gt is not None:
                overlay[mask_gt > 127] = [0, 255, 0]  # 绿色GT区域
                # 重叠区域为黄色
                overlap = (pred_resized > 127) & (mask_gt > 127)
                overlay[overlap] = [0, 255, 255]
            blended = cv2.addWeighted(img_orig, 0.6, overlay, 0.4, 0)
            
            # 拼接
            row1 = np.hstack([img_orig, mask_gt_vis if mask_gt is not None else np.zeros_like(img_orig)])
            row2 = np.hstack([pred_vis, blended])
            result = np.vstack([row1, row2])
            
            # 添加标签
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(result, 'Original', (10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result, 'Ground Truth', (w + 10, 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result, 'Prediction', (10, h + 30), font, 1, (255, 255, 255), 2)
            cv2.putText(result, 'Overlay', (w + 10, h + 30), font, 1, (255, 255, 255), 2)
            
            # 保存
            cv2.imwrite(os.path.join(result_dir, f'{img_id}_result.png'), result)
            
            # 单独保存预测 mask
            cv2.imwrite(os.path.join(result_dir, f'{img_id}_pred.png'), pred_resized)
    
    print(f'\nResults saved to {result_dir}')
    print(f'  - {len(img_paths)} visualization images (*_result.png)')
    print(f'  - {len(img_paths)} prediction masks (*_pred.png)')


if __name__ == '__main__':
    main()
