"""
Assemble and Visualize Full Point Cloud Predictions
====================================================
Reconstruct full point cloud from test chunks and visualize predictions.

Usage:
    python assemble_predictions_hpc.py \
        --checkpoint_path outputs_lidar/checkpoints/best_model.ckpt \
        --data_dir ../data/processed_data_optimized \
        --output_dir outputs_lidar/full_reconstructions \
        --num_points 8192
"""

import os
import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# ==============================================================================
# Parse Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description='Assemble full point cloud predictions')
parser.add_argument('--checkpoint_path', type=str, required=True,
                    help='Path to model checkpoint')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to processed data directory')
parser.add_argument('--output_dir', type=str, default='./full_reconstructions',
                    help='Output directory for visualizations')
parser.add_argument('--num_points', type=int, default=8192,
                    help='Number of points per sample (must match training)')
parser.add_argument('--max_chunks', type=int, default=None,
                    help='Maximum chunks to process (None for all)')
parser.add_argument('--downsample_factor', type=int, default=1,
                    help='Downsample factor for visualization (1=no downsample)')
args = parser.parse_args()

# Create output directory
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Class information
CLASS_NAMES = ['High Vegetation', 'Power Lines', 'Pylons', 'Others']
CLASS_COLORS = ['darkgreen', 'orange', 'blue', 'gray']

print("\n" + "="*80)
print("FULL POINT CLOUD ASSEMBLY AND VISUALIZATION")
print("="*80)
print(f"Checkpoint: {args.checkpoint_path}")
print(f"Data directory: {args.data_dir}")
print(f"Output directory: {args.output_dir}")
print("="*80 + "\n")

# ==============================================================================
# Model Architecture (same as training)
# ==============================================================================
def knn(x, k):
    """Find k nearest neighbors"""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


class TransformerBlock(nn.Module):
    """Point Transformer Block"""
    def __init__(self, d_model, num_heads=8, k=16):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.q = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, features, positions, k=None):
        if k is None:
            k = self.k

        B, N, C = features.shape
        idx = knn(positions.transpose(1, 2), k)

        idx_base = torch.arange(0, B, device=features.device).view(-1, 1, 1) * N
        idx_expanded = (idx + idx_base).view(B * N * k)

        features_flat = features.view(B * N, C)
        neighbor_features = features_flat[idx_expanded].view(B, N, k, C)

        positions_flat = positions.view(B * N, 3)
        neighbor_positions = positions_flat[idx_expanded].view(B, N, k, 3)

        relative_pos = positions.unsqueeze(2) - neighbor_positions
        pos_encoding = self.pos_mlp(relative_pos)

        q = self.q(features).unsqueeze(2)
        k_feat = self.k_linear(neighbor_features) + pos_encoding
        v = self.v(neighbor_features) + pos_encoding

        attn = torch.matmul(q, k_feat.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)

        features = self.norm1(features + out)
        out = self.ffn(features)
        features = self.norm2(features + out)
        return features


class PointTransformerV2(nn.Module):
    """Point Transformer V2 for semantic segmentation"""
    def __init__(self, num_classes=4, d_model=256, num_layers=4, k=16, dropout=0.3):
        super().__init__()
        self.num_classes = num_classes
        self.input_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, k=k) for _ in range(num_layers)
        ])
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, points):
        features = self.input_embed(points)
        for block in self.blocks:
            features = block(features, points)
        logits = self.seg_head(features)
        return logits


# ==============================================================================
# Load Model
# ==============================================================================
print(f"Loading model from {args.checkpoint_path}...")
checkpoint = torch.load(args.checkpoint_path, map_location=device)

# Extract hyperparameters
hparams = checkpoint['hyper_parameters']
print(f"Model hyperparameters:")
print(f"  num_classes: {hparams['num_classes']}")
print(f"  d_model: {hparams['d_model']}")
print(f"  num_layers: {hparams['num_layers']}")
print(f"  k: {hparams.get('k', 16)}")

# Create model
model = PointTransformerV2(
    num_classes=hparams['num_classes'],
    d_model=hparams['d_model'],
    num_layers=hparams['num_layers'],
    k=hparams.get('k', 16),
    dropout=0.3
)

# Load state dict
state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
              if k.startswith('model.')}
model.load_state_dict(state_dict)

model = model.to(device)
model.eval()

print(f"Model loaded successfully!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ==============================================================================
# Load and Predict on Test Chunks
# ==============================================================================
test_dir = Path(args.data_dir) / 'test'
chunk_files = sorted(list(test_dir.glob('*.pt')))

if args.max_chunks is not None:
    chunk_files = chunk_files[:args.max_chunks]

print(f"Found {len(chunk_files)} test chunks\n")

# Initialize lists
all_points = []
all_labels = []
all_predictions = []

print("Processing chunks...")
with torch.no_grad():
    for chunk_file in tqdm(chunk_files, desc="Predicting"):
        # Load chunk
        chunk_data = torch.load(chunk_file, weights_only=False)
        points = chunk_data['input_pcd']
        labels = chunk_data['semantic_labels']

        # Ensure labels is 1D
        labels = labels.reshape(-1)

        # Convert to 0-indexed
        labels = labels - 1

        # Get metadata
        metadata = chunk_data.get('metadata', {})
        centroid = metadata.get('centroid', None)

        # Denormalize points
        if centroid is not None:
            if isinstance(centroid, list):
                centroid = np.array(centroid)
            elif isinstance(centroid, torch.Tensor):
                centroid = centroid.numpy()
            points_denorm = points.numpy() + centroid
        else:
            points_denorm = points.numpy()

        # Predict
        n_points = points.shape[0]

        if n_points > args.num_points:
            # Sample points
            indices = np.random.choice(n_points, args.num_points, replace=False)
            points_pred = points[indices].unsqueeze(0).to(device)
            logits = model(points_pred)
            pred = logits.argmax(dim=-1).cpu().numpy()[0]

            # Create full prediction array
            full_pred = np.zeros(n_points, dtype=np.int64)
            full_pred[indices] = pred
            # For unsampled points, use most common class
            unsampled_mask = np.ones(n_points, dtype=bool)
            unsampled_mask[indices] = False
            if unsampled_mask.sum() > 0:
                most_common_class = np.bincount(pred).argmax()
                full_pred[unsampled_mask] = most_common_class
            pred = full_pred

        elif n_points < args.num_points:
            # Pad points
            padded_points = torch.zeros(args.num_points, 3)
            padded_points[:n_points] = points

            points_pred = padded_points.unsqueeze(0).to(device)
            logits = model(points_pred)
            pred = logits.argmax(dim=-1).cpu().numpy()[0][:n_points]

        else:
            points_pred = points.unsqueeze(0).to(device)
            logits = model(points_pred)
            pred = logits.argmax(dim=-1).cpu().numpy()[0]

        # Downsample if needed
        if args.downsample_factor > 1:
            indices = np.arange(0, n_points, args.downsample_factor)
            points_denorm = points_denorm[indices]
            labels = labels[indices]
            pred = pred[indices]

        # Append
        all_points.append(points_denorm)
        all_labels.append(labels.numpy())
        all_predictions.append(pred)

# Concatenate all chunks
all_points = np.vstack(all_points)
all_labels = np.concatenate(all_labels)
all_predictions = np.concatenate(all_predictions)

print(f"\nReconstructed point cloud:")
print(f"  Total points: {len(all_points):,}")
print(f"  X range: [{all_points[:, 0].min():.2f}, {all_points[:, 0].max():.2f}]")
print(f"  Y range: [{all_points[:, 1].min():.2f}, {all_points[:, 1].max():.2f}]")
print(f"  Z range: [{all_points[:, 2].min():.2f}, {all_points[:, 2].max():.2f}]\n")

# ==============================================================================
# Compute Metrics
# ==============================================================================
print("="*80)
print("FULL POINT CLOUD METRICS")
print("="*80)

metrics = {}
for cls in range(4):
    pred_mask = all_predictions == cls
    true_mask = all_labels == cls

    tp = np.sum(pred_mask & true_mask)
    fp = np.sum(pred_mask & ~true_mask)
    fn = np.sum(~pred_mask & true_mask)

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics[CLASS_NAMES[cls]] = {
        'IoU': iou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'Count': np.sum(true_mask)
    }

    print(f"\n{CLASS_NAMES[cls]}:")
    print(f"  IoU:       {iou:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Points:    {np.sum(true_mask):,} ({100*np.sum(true_mask)/len(all_labels):.2f}%)")

mean_iou = np.mean([m['IoU'] for m in metrics.values()])
print(f"\n{'='*80}")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Overall Accuracy: {(all_predictions == all_labels).mean():.4f}")
print(f"{'='*80}\n")

# ==============================================================================
# Visualize - Top View
# ==============================================================================
print("Creating visualizations...")

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle('Full Point Cloud - Top View (X-Y)', fontsize=16, fontweight='bold')

# Ground Truth
ax1 = axes[0]
ax1.set_title('Ground Truth', fontsize=14)
for cls in range(4):
    mask = all_labels == cls
    if mask.sum() > 0:
        ax1.scatter(all_points[mask, 0], all_points[mask, 1],
                   c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.legend(loc='upper right', markerscale=10)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# Prediction
ax2 = axes[1]
ax2.set_title('Model Prediction', fontsize=14)
for cls in range(4):
    mask = all_predictions == cls
    if mask.sum() > 0:
        ax2.scatter(all_points[mask, 0], all_points[mask, 1],
                   c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.legend(loc='upper right', markerscale=10)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{args.output_dir}/full_pointcloud_topview.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: full_pointcloud_topview.png")

# ==============================================================================
# Visualize - 3D View
# ==============================================================================
fig = plt.figure(figsize=(20, 10))
fig.suptitle('Full Point Cloud - 3D View', fontsize=16, fontweight='bold')

# Ground Truth
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Ground Truth', fontsize=14)
for cls in range(4):
    mask = all_labels == cls
    if mask.sum() > 0:
        # Sample for visualization
        if mask.sum() > 50000:
            indices = np.random.choice(np.where(mask)[0], 50000, replace=False)
            ax1.scatter(all_points[indices, 0], all_points[indices, 1], all_points[indices, 2],
                       c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
        else:
            ax1.scatter(all_points[mask, 0], all_points[mask, 1], all_points[mask, 2],
                       c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend(loc='upper right', markerscale=10)

# Prediction
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Model Prediction', fontsize=14)
for cls in range(4):
    mask = all_predictions == cls
    if mask.sum() > 0:
        # Sample for visualization
        if mask.sum() > 50000:
            indices = np.random.choice(np.where(mask)[0], 50000, replace=False)
            ax2.scatter(all_points[indices, 0], all_points[indices, 1], all_points[indices, 2],
                       c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
        else:
            ax2.scatter(all_points[mask, 0], all_points[mask, 1], all_points[mask, 2],
                       c=CLASS_COLORS[cls], label=CLASS_NAMES[cls], s=1, alpha=0.6)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend(loc='upper right', markerscale=10)

plt.tight_layout()
plt.savefig(f'{args.output_dir}/full_pointcloud_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: full_pointcloud_3d.png")

# ==============================================================================
# Visualize - Power Infrastructure Focus
# ==============================================================================
fig = plt.figure(figsize=(20, 10))
fig.suptitle('Power Infrastructure Focus', fontsize=16, fontweight='bold')

# Ground Truth
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_title('Ground Truth', fontsize=14)

powerline_mask_gt = all_labels == 1
pylon_mask_gt = all_labels == 2

if powerline_mask_gt.sum() > 0:
    ax1.scatter(all_points[powerline_mask_gt, 0], all_points[powerline_mask_gt, 1],
               all_points[powerline_mask_gt, 2], c='orange', label='Power Lines', s=5, alpha=0.8)
if pylon_mask_gt.sum() > 0:
    ax1.scatter(all_points[pylon_mask_gt, 0], all_points[pylon_mask_gt, 1],
               all_points[pylon_mask_gt, 2], c='blue', label='Pylons', s=5, alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend(loc='upper right')

# Prediction
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_title('Model Prediction', fontsize=14)

powerline_mask_pred = all_predictions == 1
pylon_mask_pred = all_predictions == 2

if powerline_mask_pred.sum() > 0:
    ax2.scatter(all_points[powerline_mask_pred, 0], all_points[powerline_mask_pred, 1],
               all_points[powerline_mask_pred, 2], c='orange', label='Power Lines', s=5, alpha=0.8)
if pylon_mask_pred.sum() > 0:
    ax2.scatter(all_points[pylon_mask_pred, 0], all_points[pylon_mask_pred, 1],
               all_points[pylon_mask_pred, 2], c='blue', label='Pylons', s=5, alpha=0.8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f'{args.output_dir}/powerlines_focus.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: powerlines_focus.png")

# ==============================================================================
# Save Results
# ==============================================================================
# Save point cloud data
output_data = {
    'points': all_points,
    'labels': all_labels,
    'predictions': all_predictions,
    'class_names': CLASS_NAMES,
    'metrics': metrics
}

np.save(f'{args.output_dir}/full_pointcloud.npy', output_data)
print(f"Saved: full_pointcloud.npy")

# Save metrics as text
with open(f'{args.output_dir}/full_metrics.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("FULL POINT CLOUD RECONSTRUCTION METRICS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Total points: {len(all_points):,}\n")
    f.write(f"Number of chunks: {len(chunk_files)}\n")
    f.write(f"Checkpoint: {args.checkpoint_path}\n\n")

    for class_name, class_metrics in metrics.items():
        f.write(f"{class_name}:\n")
        f.write(f"  IoU:       {class_metrics['IoU']:.4f}\n")
        f.write(f"  Precision: {class_metrics['Precision']:.4f}\n")
        f.write(f"  Recall:    {class_metrics['Recall']:.4f}\n")
        f.write(f"  F1-Score:  {class_metrics['F1']:.4f}\n")
        f.write(f"  Points:    {class_metrics['Count']:,}\n\n")

    f.write(f"Mean IoU: {mean_iou:.4f}\n")
    f.write(f"Overall Accuracy: {(all_predictions == all_labels).mean():.4f}\n")

print(f"Saved: full_metrics.txt")

print(f"\n{'='*80}")
print("RECONSTRUCTION COMPLETE!")
print(f"{'='*80}")
print(f"All outputs saved to: {args.output_dir}/")
print(f"  - full_pointcloud_topview.png")
print(f"  - full_pointcloud_3d.png")
print(f"  - powerlines_focus.png")
print(f"  - full_pointcloud.npy")
print(f"  - full_metrics.txt")
print(f"\nMean IoU: {mean_iou:.4f}")
print(f"Overall Accuracy: {(all_predictions == all_labels).mean():.4f}")
