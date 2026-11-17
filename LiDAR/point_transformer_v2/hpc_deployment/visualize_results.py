"""
Visualize Training Results
===========================
Creates plots from PyTorch Lightning training logs.

Usage:
    python visualize_results.py --output_dir outputs_lidar
"""

import os
import argparse
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# ==============================================================================
# Parse Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description='Visualize training results')
parser.add_argument('--output_dir', type=str, required=True,
                    help='Output directory containing logs/')
args = parser.parse_args()

output_dir = Path(args.output_dir)
log_dir = output_dir / 'logs' / 'point_transformer'
viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*80)
print("VISUALIZING TRAINING RESULTS")
print("="*80)
print(f"Output directory: {args.output_dir}")
print(f"Log directory: {log_dir}")
print(f"Visualization directory: {viz_dir}")
print("="*80 + "\n")

# ==============================================================================
# Find TensorBoard Event Files
# ==============================================================================
print("Searching for TensorBoard event files...")

event_files = list(log_dir.rglob('events.out.tfevents.*'))
if not event_files:
    print(f"ERROR: No TensorBoard event files found in {log_dir}")
    print("Make sure training has completed and logs were saved.")
    exit(1)

# Use the most recent event file
event_file = sorted(event_files, key=lambda x: x.stat().st_mtime)[-1]
print(f"Found event file: {event_file.name}\n")

# ==============================================================================
# Extract Metrics from TensorBoard
# ==============================================================================
print("Extracting metrics from TensorBoard logs...")

ea = event_accumulator.EventAccumulator(str(event_file.parent))
ea.Reload()

# Get available scalar tags
scalar_tags = ea.Tags()['scalars']
print(f"Available metrics: {scalar_tags}\n")

# Extract metrics
metrics = {}
for tag in scalar_tags:
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    # Convert steps to epochs for validation metrics (logged once per epoch)
    # Validation metrics are logged at end of epoch, so step number ≈ epoch number
    if tag.startswith('val_') or tag.endswith('_epoch'):
        # For validation: step number IS the epoch number (0-indexed)
        epochs = [s for s in steps]
    else:
        # For training steps: divide by steps_per_epoch
        # Estimate steps_per_epoch from first validation step
        if 'val_loss' in scalar_tags:
            val_events = ea.Scalars('val_loss')
            if len(val_events) > 1:
                steps_per_epoch = val_events[1].step - val_events[0].step
            else:
                steps_per_epoch = val_events[0].step if val_events else 1
        else:
            steps_per_epoch = max(steps) / len(steps) if steps else 1
        epochs = [s / steps_per_epoch for s in steps]

    metrics[tag] = {'steps': steps, 'epochs': epochs, 'values': values}

# ==============================================================================
# Plot Training Curves
# ==============================================================================
print("Creating training curve visualizations...")

# 1. Loss curves
if 'train_loss_epoch' in metrics and 'val_loss' in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))

    if metrics['train_loss_epoch']['values']:
        ax.plot(metrics['train_loss_epoch']['epochs'],
               metrics['train_loss_epoch']['values'],
               label='Training Loss', linewidth=2)

    if metrics['val_loss']['values']:
        ax.plot(metrics['val_loss']['epochs'],
               metrics['val_loss']['values'],
               label='Validation Loss', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'loss_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: loss_curves.png")

# 2. Accuracy curves
if 'train_acc' in metrics and 'val_acc' in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))

    if metrics['train_acc']['values']:
        ax.plot(metrics['train_acc']['epochs'],
               metrics['train_acc']['values'],
               label='Training Accuracy', linewidth=2)

    if metrics['val_acc']['values']:
        ax.plot(metrics['val_acc']['epochs'],
               metrics['val_acc']['values'],
               label='Validation Accuracy', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: accuracy_curves.png")

# 3. mIoU curve (most important metric)
if 'val_miou' in metrics:
    fig, ax = plt.subplots(figsize=(10, 6))

    if metrics['val_miou']['values']:
        ax.plot(metrics['val_miou']['epochs'],
               metrics['val_miou']['values'],
               label='Validation mIoU', linewidth=2, color='green')

        # Mark best mIoU
        best_idx = max(range(len(metrics['val_miou']['values'])),
                      key=lambda i: metrics['val_miou']['values'][i])
        best_epoch = metrics['val_miou']['epochs'][best_idx]
        best_value = metrics['val_miou']['values'][best_idx]

        ax.scatter([best_epoch], [best_value], color='red', s=100, zorder=5,
                  label=f'Best mIoU: {best_value:.4f}')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Mean IoU', fontsize=12)
    ax.set_title('Validation Mean IoU', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(viz_dir / 'miou_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: miou_curve.png")

# 4. Combined plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Point Transformer V2 Training Summary', fontsize=16, fontweight='bold')

# Loss
if 'train_loss_epoch' in metrics and 'val_loss' in metrics:
    ax = axes[0, 0]
    if metrics['train_loss_epoch']['values']:
        ax.plot(metrics['train_loss_epoch']['epochs'],
               metrics['train_loss_epoch']['values'],
               label='Train', linewidth=2)
    if metrics['val_loss']['values']:
        ax.plot(metrics['val_loss']['epochs'],
               metrics['val_loss']['values'],
               label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Accuracy
if 'train_acc' in metrics and 'val_acc' in metrics:
    ax = axes[0, 1]
    if metrics['train_acc']['values']:
        ax.plot(metrics['train_acc']['epochs'],
               metrics['train_acc']['values'],
               label='Train', linewidth=2)
    if metrics['val_acc']['values']:
        ax.plot(metrics['val_acc']['epochs'],
               metrics['val_acc']['values'],
               label='Val', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

# mIoU
if 'val_miou' in metrics:
    ax = axes[1, 0]
    if metrics['val_miou']['values']:
        ax.plot(metrics['val_miou']['epochs'],
               metrics['val_miou']['values'],
               linewidth=2, color='green')
        best_idx = max(range(len(metrics['val_miou']['values'])),
                      key=lambda i: metrics['val_miou']['values'][i])
        best_epoch = metrics['val_miou']['epochs'][best_idx]
        best_value = metrics['val_miou']['values'][best_idx]
        ax.scatter([best_epoch], [best_value], color='red', s=100, zorder=5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean IoU')
    ax.set_title(f'Validation mIoU (Best: {best_value:.4f})')
    ax.grid(True, alpha=0.3)

# Learning Rate
if 'lr-AdamW' in metrics:
    ax = axes[1, 1]
    if metrics['lr-AdamW']['values']:
        ax.plot(metrics['lr-AdamW']['epochs'],
               metrics['lr-AdamW']['values'],
               linewidth=2, color='purple')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(viz_dir / 'training_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print("  Saved: training_summary.png")

# ==============================================================================
# Save Metrics Summary
# ==============================================================================
print("\nSaving metrics summary...")

summary = {
    'best_val_miou': float(best_value) if 'val_miou' in metrics else None,
    'best_epoch': int(best_epoch) if 'val_miou' in metrics else None,
    'final_train_loss': float(metrics['train_loss_epoch']['values'][-1]) if 'train_loss_epoch' in metrics else None,
    'final_val_loss': float(metrics['val_loss']['values'][-1]) if 'val_loss' in metrics else None,
    'final_train_acc': float(metrics['train_acc']['values'][-1]) if 'train_acc' in metrics else None,
    'final_val_acc': float(metrics['val_acc']['values'][-1]) if 'val_acc' in metrics else None,
}

# Load training summary if exists
training_summary_path = output_dir / 'training_summary.json'
if training_summary_path.exists():
    with open(training_summary_path, 'r') as f:
        training_summary = json.load(f)
    training_summary['metrics'] = summary
else:
    training_summary = {'metrics': summary}

with open(training_summary_path, 'w') as f:
    json.dump(training_summary, f, indent=2)

print(f"  Saved: training_summary.json")

print(f"\n{'='*80}")
print("VISUALIZATION COMPLETE!")
print(f"{'='*80}")
print(f"\nAll visualizations saved to: {viz_dir}/")
print(f"  - loss_curves.png")
print(f"  - accuracy_curves.png")
print(f"  - miou_curve.png")
print(f"  - training_summary.png")
print(f"\nBest Validation mIoU: {best_value:.4f} (Epoch {int(best_epoch)})")
