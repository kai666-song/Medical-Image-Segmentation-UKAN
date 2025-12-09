# U-KAN 乳腺超声图像分割复现报告

## 1. 项目概述

本项目基于 U-KAN (U-Net + Kolmogorov-Arnold Networks) 架构，在 BUSI (Breast Ultrasound Images) 数据集上完成乳腺超声图像分割任务的复现。

### 1.1 技术栈

| 组件 | 版本/配置 |
|------|----------|
| Python | 3.9 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| GPU | NVIDIA RTX 4060 (8GB) |
| 操作系统 | Windows |

### 1.2 项目结构

```
U-KAN/
├── datasets/
│   ├── BUSI/                    # 原始数据集
│   │   ├── benign/
│   │   ├── malignant/
│   │   └── normal/
│   └── BUSI_processed/          # 处理后的数据集
│       ├── images/
│       ├── masks/0/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt
├── Seg_UKAN/                    # 分割模型代码
│   ├── train.py                 # 训练脚本（已添加断点续训）
│   ├── predict.py               # 预测可视化脚本
│   ├── archs.py                 # 模型架构
│   └── ...
├── outputs/
│   └── busi_ukan/               # 训练输出
│       ├── model.pth            # 最佳模型
│       ├── checkpoint.pth       # 断点续训检查点
│       ├── log.csv              # 训练日志
│       └── predictions/         # 可视化结果
└── scripts/
    ├── prepare_busi_dataset.py  # 数据清洗脚本
    └── setup_env.bat            # 环境配置脚本
```

## 2. 数据准备

### 2.1 原始数据问题

BUSI 数据集原始结构存在以下问题：
- 原图和 Mask 混在同一文件夹
- 文件命名不规范：`benign (1).png`, `benign (1)_mask.png`
- 部分图像存在多个 Mask：`_mask.png`, `_mask_1.png`, `_mask_2.png`

### 2.2 数据清洗方案

编写 `scripts/prepare_busi_dataset.py` 实现：

1. **分离图像和掩码**：将原图放入 `images/`，掩码放入 `masks/0/`
2. **多掩码合并**：使用 OR 操作合并多个病灶掩码
3. **统一命名**：`{category}_{index:04d}.png`
4. **数据集划分**：70% 训练 / 15% 验证 / 15% 测试

### 2.3 数据统计

| 类别 | 数量 |
|------|------|
| Benign (良性) | 437 |
| Malignant (恶性) | 210 |
| **总计** | **647** |
| 多掩码合并 | 17 例 |

划分结果：
- 训练集：452 张
- 验证集：97 张
- 测试集：98 张

## 3. 模型训练

### 3.1 训练配置

```yaml
arch: UKAN
embed_dims: [128, 160, 256]
input_size: 256x256
batch_size: 4
epochs: 200
optimizer: Adam
learning_rate: 1e-4
kan_learning_rate: 1e-2
scheduler: CosineAnnealingLR
loss: BCEDiceLoss
```

### 3.2 训练命令

```powershell
cd Seg_UKAN
python train.py \
    --arch UKAN \
    --dataset BUSI_processed \
    --data_dir ../datasets \
    --input_w 256 --input_h 256 \
    --batch_size 4 \
    --epochs 200 \
    --lr 1e-4 \
    --name busi_ukan \
    --output_dir ../outputs \
    --num_workers 4
```

### 3.3 断点续训

训练过程中如果中断，可使用以下命令继续：

```powershell
python train.py \
    --arch UKAN \
    --dataset BUSI_processed \
    --data_dir ../datasets \
    --input_w 256 --input_h 256 \
    --batch_size 4 \
    --epochs 200 \
    --lr 1e-4 \
    --name busi_ukan \
    --output_dir ../outputs \
    --num_workers 4 \
    --resume ../outputs/busi_ukan/checkpoint.pth
```

### 3.4 训练结果

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| IoU | 0.849 | 0.677 |
| Dice | - | 0.795 |
| Loss | 0.118 | 0.299 |

最佳验证结果（Epoch 141）：
- 最佳验证 Dice: 0.7997
- 最佳验证 IoU: 0.6835

训练时间：约 200 epochs × 2 min/epoch ≈ 6-7 小时

### 3.5 训练曲线

#### Loss 曲线
![Loss曲线](loss_curve.png)

训练过程中，Loss 从初始的 1.07 逐渐下降到 0.12，验证 Loss 从 1.04 下降到 0.30，模型收敛良好。

#### IoU 曲线
![IoU曲线](iou_curve.png)

训练 IoU 从 0.28 提升到 0.85，验证 IoU 从 0.29 提升到 0.68，表明模型分割能力持续提升。

#### Dice 曲线
![Dice曲线](dice_curve.png)

验证 Dice 从 0.44 提升到 0.79，在 Epoch 141 达到最佳值 0.7997。

## 4. 模型预测与可视化

### 4.1 预测命令

```powershell
cd Seg_UKAN
# 预测 20 张样本
python predict.py --name busi_ukan --num_samples 20

# 预测全部图像
python predict.py --name busi_ukan --num_samples -1
```

### 4.2 输出说明

预测结果保存在 `outputs/busi_ukan/predictions/`：
- `*_result.png`：四宫格对比图（原图、GT、预测、叠加）
- `*_pred.png`：预测的分割掩码

叠加图颜色说明：
- 🟢 绿色：Ground Truth 区域
- 🔴 红色：预测区域
- 🟡 黄色：重叠区域（正确预测）

## 5. 代码修改记录

### 5.1 archs.py
- 添加 `__all__ = ['UKAN']` 导出声明

### 5.2 train.py
- 添加 `--resume` 参数支持断点续训
- 每个 epoch 保存 `checkpoint.pth`
- 移除训练中不必要的 `indicators()` 调用（避免内存溢出）
- 添加自定义数据集的 mask 扩展名支持

### 5.3 新增文件
- `scripts/prepare_busi_dataset.py`：数据清洗脚本
- `Seg_UKAN/predict.py`：预测可视化脚本

## 6. 环境配置

### 6.1 创建 Conda 环境

```powershell
conda create -n medicalimage python=3.9
conda activate medicalimage
```

### 6.2 安装依赖

```powershell
# PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 其他依赖
pip install albumentations==1.3.1
pip install tqdm tensorboardX pandas scikit-image scipy opencv-python timm addict yapf
```

## 7. 总结

本项目成功复现了 U-KAN 在 BUSI 乳腺超声数据集上的分割任务：

✅ 数据清洗与预处理  
✅ 模型训练（200 epochs）  
✅ 断点续训功能  
✅ 预测可视化  
✅ 验证 Dice 达到 0.795

### 后续改进方向

1. 数据增强：添加更多增强策略提升泛化能力
2. 超参数调优：调整学习率、embed_dims 等参数
3. 模型集成：结合多个模型提升性能
4. 后处理：添加 CRF 等后处理方法优化边界
