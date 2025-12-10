"""
生成 U-KAN vs U-KAN+CBAM 对比图表
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取训练日志
df_ukan = pd.read_csv('outputs/busi_ukan/log.csv')
df_cbam = pd.read_csv('outputs/busi_ukan_cbam/log.csv')

# 1. Loss对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_ukan['epoch'], df_ukan['val_loss'], label='U-KAN', color='#2196F3', linewidth=1.5)
ax.plot(df_cbam['epoch'], df_cbam['val_loss'], label='U-KAN+CBAM', color='#FF5722', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Loss')
ax.set_title('Validation Loss Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/comparison_loss.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. IoU对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_ukan['epoch'], df_ukan['val_iou'], label='U-KAN', color='#2196F3', linewidth=1.5)
ax.plot(df_cbam['epoch'], df_cbam['val_iou'], label='U-KAN+CBAM', color='#FF5722', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation IoU')
ax.set_title('Validation IoU Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/comparison_iou.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Dice对比
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_ukan['epoch'], df_ukan['val_dice'], label='U-KAN', color='#2196F3', linewidth=1.5)
ax.plot(df_cbam['epoch'], df_cbam['val_dice'], label='U-KAN+CBAM', color='#FF5722', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation Dice')
ax.set_title('Validation Dice Comparison')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/comparison_dice.png', dpi=150, bbox_inches='tight')
plt.close()

print("对比图表已保存:")
print("  - docs/comparison_loss.png")
print("  - docs/comparison_iou.png")
print("  - docs/comparison_dice.png")

# 打印对比结果
print("\n" + "=" * 60)
print("U-KAN vs U-KAN+CBAM 对比结果")
print("=" * 60)
print(f"{'指标':<20} {'U-KAN':<15} {'U-KAN+CBAM':<15} {'提升':<10}")
print("-" * 60)
best_dice_ukan = df_ukan['val_dice'].max()
best_dice_cbam = df_cbam['val_dice'].max()
best_iou_ukan = df_ukan['val_iou'].max()
best_iou_cbam = df_cbam['val_iou'].max()
print(f"{'最佳 Val Dice':<20} {best_dice_ukan:<15.4f} {best_dice_cbam:<15.4f} {'+' + f'{(best_dice_cbam-best_dice_ukan)*100:.2f}%':<10}")
print(f"{'最佳 Val IoU':<20} {best_iou_ukan:<15.4f} {best_iou_cbam:<15.4f} {'+' + f'{(best_iou_cbam-best_iou_ukan)*100:.2f}%':<10}")
print("=" * 60)
