"""
生成训练曲线图 - 分开保存
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取训练日志
df = pd.read_csv('outputs/busi_ukan/log.csv')

# 1. Loss曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['loss'], label='Train Loss', color='#2196F3', linewidth=1.5)
ax.plot(df['epoch'], df['val_loss'], label='Val Loss', color='#FF5722', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 2. IoU曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['iou'], label='Train IoU', color='#2196F3', linewidth=1.5)
ax.plot(df['epoch'], df['val_iou'], label='Val IoU', color='#FF5722', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('IoU')
ax.set_title('Training & Validation IoU')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/iou_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 3. Dice曲线
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['epoch'], df['val_dice'], label='Val Dice', color='#4CAF50', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Dice Score')
ax.set_title('Validation Dice Score')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('docs/dice_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# 打印最佳结果
best_val_dice_idx = df['val_dice'].idxmax()
best_val_iou_idx = df['val_iou'].idxmax()

print("=" * 50)
print("训练结果统计")
print("=" * 50)
print(f"最佳验证 Dice: {df.loc[best_val_dice_idx, 'val_dice']:.4f} (Epoch {best_val_dice_idx})")
print(f"最佳验证 IoU:  {df.loc[best_val_iou_idx, 'val_iou']:.4f} (Epoch {best_val_iou_idx})")
print(f"最终训练 Loss: {df.iloc[-1]['loss']:.4f}")
print(f"最终验证 Loss: {df.iloc[-1]['val_loss']:.4f}")
print(f"最终训练 IoU:  {df.iloc[-1]['iou']:.4f}")
print(f"最终验证 IoU:  {df.iloc[-1]['val_iou']:.4f}")
print(f"最终验证 Dice: {df.iloc[-1]['val_dice']:.4f}")
print("=" * 50)
print("图表已保存:")
print("  - docs/loss_curve.png")
print("  - docs/iou_curve.png")
print("  - docs/dice_curve.png")
