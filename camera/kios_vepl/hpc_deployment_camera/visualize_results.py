"""
Generate visualizations from training results for thesis
"""
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate training visualizations')
parser.add_argument('--output_dir', type=str, default='outputs',
                    help='Output directory containing training results')
args = parser.parse_args()

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Load training history
history_file = f'{args.output_dir}/training_history.json'
with open(history_file, 'r') as f:
    history = json.load(f)

train_loss = history['train_loss']
val_iou = history['val_iou']
val_dice = history['val_dice']
val_acc = history['val_acc']

# Generate epochs list from data length
epochs = list(range(1, len(train_loss) + 1))

# Create output directory for visualizations
viz_dir = Path(f'{args.output_dir}/visualizations')
viz_dir.mkdir(exist_ok=True, parents=True)

# 1. Training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss over Epochs', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(viz_dir / 'training_loss.png', bbox_inches='tight')
plt.close()

# 2. IoU and Dice Score Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, val_iou, 'g-', linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('IoU Score', fontsize=12)
ax1.set_title('Validation IoU over Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(val_iou) * 1.1])

ax2.plot(epochs, val_dice, 'r-', linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Dice Score', fontsize=12)
ax2.set_title('Validation Dice Score over Epochs', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, max(val_dice) * 1.1])

plt.tight_layout()
plt.savefig(viz_dir / 'iou_dice_curves.png', bbox_inches='tight')
plt.close()

# 3. Combined Metrics Plot
fig, ax1 = plt.subplots(figsize=(12, 6))

color1 = 'tab:blue'
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', color=color1, fontsize=12)
ax1.plot(epochs, train_loss, color=color1, linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color2 = 'tab:green'
ax2.set_ylabel('IoU / Dice Score', color=color2, fontsize=12)
ax2.plot(epochs, val_iou, color='green', linewidth=2, marker='o', markersize=4, label='Val IoU')
ax2.plot(epochs, val_dice, color='orange', linewidth=2, marker='s', markersize=4, label='Val Dice')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim([0, 1])

# Add legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=11)

plt.title('Training Progress: Loss and Metrics', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(viz_dir / 'combined_metrics.png', bbox_inches='tight')
plt.close()

# 4. Accuracy Curve
plt.figure(figsize=(10, 6))
plt.plot(epochs, [acc * 100 for acc in val_acc], 'purple', linewidth=2, marker='d', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(viz_dir / 'accuracy_curve.png', bbox_inches='tight')
plt.close()

# 5. Summary Statistics Table
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('tight')
ax.axis('off')

# Load summary
summary_file = f'{args.output_dir}/training_summary.json'
with open(summary_file, 'r') as f:
    summary = json.load(f)

table_data = [
    ['Metric', 'Best Value', 'Final Value'],
    ['IoU Score', f"{max(val_iou):.4f}", f"{val_iou[-1]:.4f}"],
    ['Dice Score', f"{max(val_dice):.4f}", f"{val_dice[-1]:.4f}"],
    ['Accuracy', f"{max(val_acc)*100:.2f}%", f"{val_acc[-1]*100:.2f}%"],
    ['Training Loss', f"{min(train_loss):.4f}", f"{train_loss[-1]:.4f}"],
    ['Test IoU', '-', f"{summary['test_iou']:.4f}"],
    ['Test Dice', '-', f"{summary['test_dice']:.4f}"],
    ['Test Accuracy', '-', f"{summary['test_acc']:.2f}%"],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.3, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)

# Style header row
for i in range(3):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#E7E6E6')

plt.title('Training Summary Statistics', fontsize=14, fontweight='bold', pad=20)
plt.savefig(viz_dir / 'summary_table.png', bbox_inches='tight')
plt.close()

# 6. Improvement Rate Analysis
improvements = np.diff(val_iou)
plt.figure(figsize=(10, 6))
plt.bar(epochs[1:], improvements, color=['green' if x > 0 else 'red' for x in improvements])
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('IoU Improvement', fontsize=12)
plt.title('Per-Epoch IoU Improvement', fontsize=14, fontweight='bold')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(viz_dir / 'improvement_rate.png', bbox_inches='tight')
plt.close()

print(f"✅ All visualizations saved to {viz_dir}/")
print("\nGenerated files:")
print("  1. training_loss.png - Training loss curve")
print("  2. iou_dice_curves.png - IoU and Dice metrics")
print("  3. combined_metrics.png - Combined loss and metrics")
print("  4. accuracy_curve.png - Validation accuracy")
print("  5. summary_table.png - Summary statistics table")
print("  6. improvement_rate.png - Per-epoch improvement analysis")
