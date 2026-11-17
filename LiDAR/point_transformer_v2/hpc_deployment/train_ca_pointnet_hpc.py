"""
CA-PointNet++ Training for HPC
==============================
Train CA-PointNet++ model on LiDAR point cloud data for semantic segmentation.
Based on "Semantic Segmentation of Transmission Corridor 3D Point Clouds
Based on CA-PointNet++" paper.

Classes:
  - 0: Tower (Pylons)
  - 1: Transmission Line (Power Lines)
  - 2: Ground Wire
  - 3: Ground (includes vegetation)

Architecture:
  - PointNet++ backbone with Single-Scale Grouping (SSG)
  - Coordinate Attention (CA) modules in Set Abstraction layers
  - 4 encoder layers + 4 decoder layers
  - Input: 4096 points, 6 channels (XYZ + RGB)

Usage:
    python train_ca_pointnet_hpc.py \
        --data_dir ../data/processed_data_optimized \
        --output_dir ./outputs_ca_pointnet \
        --batch_size 32 \
        --epochs 100
"""

import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
from tqdm import tqdm

# ==============================================================================
# Parse Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description='Train CA-PointNet++ on LiDAR data')
parser.add_argument('--data_dir', type=str, default='../data/processed_data_optimized',
                    help='Path to processed data directory')
parser.add_argument('--output_dir', type=str, default='./outputs_ca_pointnet',
                    help='Output directory for checkpoints and logs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size (32 as per paper)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Maximum number of training epochs (100 as per paper)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate (0.001 as per paper)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Momentum for optimizer (0.9 as per paper)')
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points per sample (4096 as per paper)')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of dataloader workers')
parser.add_argument('--early_stop_patience', type=int, default=15,
                    help='Early stopping patience (epochs)')
parser.add_argument('--gradient_clip', type=float, default=1.0,
                    help='Gradient clipping value')
parser.add_argument('--max_train_samples', type=int, default=None,
                    help='Maximum training samples (None for all)')
args = parser.parse_args()

# Create output directories
Path(args.output_dir).mkdir(parents=True, exist_ok=True)
checkpoint_dir = Path(args.output_dir) / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True)
log_dir = Path(args.output_dir) / 'logs'
log_dir.mkdir(exist_ok=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()  # Clear any cached memory
    print("CUDA optimizations enabled (cudnn.benchmark=True)")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Record start time
start_time = time.time()
start_datetime = datetime.now()

print("\n" + "="*80)
print("CA-POINTNET++ TRAINING")
print("="*80)
print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Data directory: {args.data_dir}")
print(f"Output directory: {args.output_dir}")
print(f"Batch size: {args.batch_size}")
print(f"Max epochs: {args.epochs}")
print(f"Points per sample: {args.num_points}")
print(f"Input channels: 6 (XYZ + RGB)")
print("="*80 + "\n")

# ==============================================================================
# Dataset Classes
# ==============================================================================
class PowerLineDatasetRGB(Dataset):
    """Dataset for power line point cloud segmentation with RGB"""

    def __init__(self, data_dir, split='train', num_points=4096, use_fps=True, max_samples=None):
        self.data_dir = Path(data_dir) / split
        self.num_points = num_points
        self.use_fps = use_fps
        self.split = split
        self.max_samples = max_samples

        # Get all sample files
        self.samples = sorted(list(self.data_dir.glob('*.pt')))

        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]

        if not self.samples:
            raise ValueError(f"No samples found in {self.data_dir}")

        print(f"Loaded {len(self.samples)} samples from {split} split")

        # Load metadata
        metadata_path = Path(data_dir) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                print(f"  Metadata: {self.metadata['n_classes']} classes")
        else:
            self.metadata = {'n_classes': 4}

    def farthest_point_sampling(self, points, num_samples):
        """Farthest Point Sampling (FPS)"""
        N, D = points.shape
        if N <= num_samples:
            return np.arange(N)

        centroids = np.zeros(num_samples, dtype=np.int32)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)

        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)

        return centroids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = torch.load(self.samples[idx], weights_only=False)
        points = sample['input_pcd']  # [N, 3] - XYZ coordinates
        labels = sample['semantic_labels'].reshape(-1)  # [N]

        # Convert to 0-indexed
        labels = labels - 1

        # Generate synthetic RGB based on labels (since we don't have real RGB)
        # This is a placeholder - ideally you'd have real RGB from your data
        rgb = self.generate_synthetic_rgb(labels, points.shape[0])

        # Combine XYZ + RGB -> [N, 6]
        points_rgb = torch.cat([points, rgb], dim=1)

        n_points = points_rgb.shape[0]

        if n_points > self.num_points:
            if self.use_fps:
                fps_idx = self.farthest_point_sampling(points_rgb[:, :3].numpy(), self.num_points)
                points_rgb = points_rgb[fps_idx]
                labels = labels[fps_idx]
            else:
                choice = np.random.choice(n_points, self.num_points, replace=False)
                points_rgb = points_rgb[choice]
                labels = labels[choice]
        elif n_points < self.num_points:
            choice = np.random.choice(n_points, self.num_points - n_points, replace=True)
            extra_points = points_rgb[choice]
            extra_labels = labels[choice]
            points_rgb = torch.cat([points_rgb, extra_points], dim=0)
            labels = torch.cat([labels, extra_labels], dim=0)

        return {'points': points_rgb, 'labels': labels, 'idx': idx}

    def generate_synthetic_rgb(self, labels, num_points):
        """Generate synthetic RGB colors based on labels"""
        rgb = torch.zeros(num_points, 3)

        # Tower/Pylon: Gray
        rgb[labels == 0] = torch.tensor([0.5, 0.5, 0.5])
        # Transmission Line: Yellow
        rgb[labels == 1] = torch.tensor([1.0, 1.0, 0.0])
        # Ground Wire: Orange
        rgb[labels == 2] = torch.tensor([1.0, 0.5, 0.0])
        # Ground/Vegetation: Green
        rgb[labels == 3] = torch.tensor([0.2, 0.8, 0.2])

        # Add some noise for variation
        rgb += torch.randn_like(rgb) * 0.1
        rgb = torch.clamp(rgb, 0, 1)

        return rgb


class PowerLineDataModule(L.LightningDataModule):
    """Lightning DataModule for power line segmentation"""

    def __init__(self, data_dir, batch_size=32, num_points=4096, num_workers=8,
                 use_fps=True, max_samples=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_workers = num_workers
        self.use_fps = use_fps
        self.max_samples = max_samples

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PowerLineDatasetRGB(
                self.data_dir, split='train', num_points=self.num_points,
                use_fps=self.use_fps, max_samples=self.max_samples
            )
            self.val_dataset = PowerLineDatasetRGB(
                self.data_dir, split='test', num_points=self.num_points,
                use_fps=False, max_samples=None
            )

        if stage == 'test' or stage is None:
            self.test_dataset = PowerLineDatasetRGB(
                self.data_dir, split='test', num_points=self.num_points,
                use_fps=False, max_samples=None
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )


# ==============================================================================
# Model Architecture - CA-PointNet++
# ==============================================================================
def square_distance(src, dst):
    """Calculate squared Euclidean distance between points"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """Farthest Point Sampling"""
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Query ball point grouping"""
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def index_points(points, idx):
    """Index points"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


class CoordinateAttention(nn.Module):
    """Coordinate Attention Module"""
    def __init__(self, in_channels, reduction=8):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [B, C, H, W] where H=npoint, W=nsample
        identity = x

        n, c, h, w = x.size()

        # Pool along two directions
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1] -> [B, C, 1, W]

        # Concatenate and transform
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1] or [B, C, 1, H+W]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split back
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Generate attention weights
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        # Apply attention
        out = identity * a_h * a_w

        return out


class PointNetSetAbstractionCA(nn.Module):
    """Set Abstraction layer with Coordinate Attention"""
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super(PointNetSetAbstractionCA, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        # Insert CA module before final max pooling
        self.ca = CoordinateAttention(last_channel)
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: [B, N, 3] - point coordinates
            points: [B, N, C] - point features (or None)
        Return:
            new_xyz: [B, npoint, 3] - sampled points
            new_points: [B, mlp[-1], npoint] - new point features
        """
        xyz = xyz.permute(0, 2, 1)  # [B, 3, N]
        if points is not None:
            points = points.permute(0, 2, 1)  # [B, C, N]

        if self.group_all:
            new_xyz = torch.zeros(xyz.shape[0], 1, 3).to(xyz.device)
            grouped_xyz = xyz.permute(0, 2, 1).unsqueeze(1)
            if points is not None:
                new_points = torch.cat([grouped_xyz, points.permute(0, 2, 1).unsqueeze(1)], dim=-1)
            else:
                new_points = grouped_xyz
        else:
            fps_idx = farthest_point_sample(xyz.permute(0, 2, 1), self.npoint)
            new_xyz = index_points(xyz.permute(0, 2, 1), fps_idx)  # [B, npoint, 3]
            idx = query_ball_point(self.radius, self.nsample, xyz.permute(0, 2, 1), new_xyz)
            grouped_xyz = index_points(xyz.permute(0, 2, 1), idx)  # [B, npoint, nsample, 3]
            grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

            if points is not None:
                grouped_points = index_points(points.permute(0, 2, 1), idx)
                new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
            else:
                new_points = grouped_xyz_norm

        # [B, npoint, nsample, C] -> [B, C, npoint, nsample]
        new_points = new_points.permute(0, 3, 1, 2)

        # MLP
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        # Apply Coordinate Attention
        new_points = self.ca(new_points)

        # Max pooling
        new_points = torch.max(new_points, 3)[0]  # [B, mlp[-1], npoint]

        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    """Feature Propagation layer"""
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel

        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: [B, N, 3] - input points position data
            xyz2: [B, S, 3] - sampled input points position data
            points1: [B, C1, N] - input points data
            points2: [B, C2, S] - sampled points data
        Return:
            new_points: [B, mlp[-1], N] - upsampled points data
        """
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, 1, N)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2.permute(0, 2, 1), idx) * weight.view(B, N, 3, 1), dim=2)
            interpolated_points = interpolated_points.permute(0, 2, 1)

        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points

        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))

        return new_points


class CAPointNetPlusPlus(nn.Module):
    """CA-PointNet++ for Semantic Segmentation"""
    def __init__(self, num_classes=4, in_channels=6):
        super(CAPointNetPlusPlus, self).__init__()
        self.num_classes = num_classes

        # Encoder - 4 Set Abstraction layers with CA
        # SA1: 4096 -> 1024 points
        self.sa1 = PointNetSetAbstractionCA(npoint=1024, radius=0.1, nsample=32,
                                            in_channel=in_channels, mlp=[64, 64, 128])
        # SA2: 1024 -> 256 points
        self.sa2 = PointNetSetAbstractionCA(npoint=256, radius=0.2, nsample=32,
                                            in_channel=128 + 3, mlp=[128, 128, 256])
        # SA3: 256 -> 64 points
        self.sa3 = PointNetSetAbstractionCA(npoint=64, radius=0.4, nsample=32,
                                            in_channel=256 + 3, mlp=[256, 256, 512])
        # SA4: 64 -> 16 points
        self.sa4 = PointNetSetAbstractionCA(npoint=16, radius=0.8, nsample=32,
                                            in_channel=512 + 3, mlp=[512, 512, 1024])

        # Decoder - 4 Feature Propagation layers
        self.fp4 = PointNetFeaturePropagation(in_channel=1024 + 512, mlp=[512, 512])
        self.fp3 = PointNetFeaturePropagation(in_channel=512 + 256, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=256 + 128, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=128 + in_channels, mlp=[128, 128, 128])

        # Classifier
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        """
        Input:
            xyz: [B, N, 6] - point cloud with XYZ + RGB
        Output:
            logits: [B, N, num_classes] - per-point class logits
        """
        B, N, C = xyz.shape

        # Split coordinates and features
        l0_points = xyz  # [B, N, 6]
        l0_xyz = xyz[:, :, :3]  # [B, N, 3]

        # Encoder
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points.permute(0, 2, 1))
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points.permute(0, 2, 1))
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points.permute(0, 2, 1))

        # Decoder
        l3_points_up = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points_up = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points_up)
        l1_points_up = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_up)
        l0_points_up = self.fp1(l0_xyz, l1_xyz, l0_points.permute(0, 2, 1), l1_points_up)

        # Classifier
        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points_up))))
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # [B, N, num_classes]

        return x


class CAPointNetLightning(L.LightningModule):
    """Lightning module for CA-PointNet++ training"""
    def __init__(self, num_classes=4, lr=1e-3, momentum=0.9,
                 class_weights=None, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = CAPointNetPlusPlus(num_classes=num_classes, in_channels=6)

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            class_weights_tensor = torch.where(
                class_weights_tensor < 0.01,
                torch.ones_like(class_weights_tensor),
                class_weights_tensor
            )
            self.class_weights = class_weights_tensor
        else:
            self.class_weights = torch.ones(num_classes)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=-1)

    def forward(self, points):
        return self.model(points)

    def calculate_iou(self, pred, target, num_classes):
        ious = []
        pred = pred.view(-1)
        target = target.view(-1)
        for cls in range(num_classes):
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds & target_inds).sum().float()
            union = (pred_inds | target_inds).sum().float()
            ious.append((intersection / union).item() if union > 0 else float('nan'))
        return ious

    def training_step(self, batch, batch_idx):
        points, labels = batch['points'], batch['labels']
        logits = self(points)
        loss = self.criterion(logits.reshape(-1, self.hparams.num_classes), labels.reshape(-1))
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        points, labels = batch['points'], batch['labels']
        logits = self(points)
        loss = self.criterion(logits.reshape(-1, self.hparams.num_classes), labels.reshape(-1))
        pred = logits.argmax(dim=-1)
        acc = (pred == labels).float().mean()
        ious = self.calculate_iou(pred, labels, self.hparams.num_classes)
        mean_iou = np.nanmean([iou for iou in ious if not np.isnan(iou)])

        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_miou', mean_iou, prog_bar=True, on_step=False, on_epoch=True)

        return {'val_loss': loss, 'val_miou': mean_iou}

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.momentum, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.7
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }


# ==============================================================================
# Main Training
# ==============================================================================
if __name__ == '__main__':
    # Class weights (same as Point Transformer)
    # Tower, Transmission Line, Ground Wire, Ground
    class_weights = [50.0, 10.0, 50.0, 1.0]
    print(f"\nUsing class weights: {class_weights}")
    print("  [Tower, Transmission Line, Ground Wire, Ground]")
    print("  Addressing severe class imbalance\n")

    # Data module
    print("Setting up data...")
    dm = PowerLineDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_points=args.num_points,
        num_workers=args.num_workers,
        use_fps=True,
        max_samples=args.max_train_samples
    )

    # Model
    print("Initializing CA-PointNet++ model...")
    model = CAPointNetLightning(
        num_classes=4,
        lr=args.lr,
        momentum=args.momentum,
        class_weights=class_weights,
        max_epochs=args.epochs
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='best_model',
        monitor='val_miou',
        mode='max',
        save_top_k=1,
        save_last=False,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_miou',
        patience=args.early_stop_patience,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    progress_bar = TQDMProgressBar(refresh_rate=1)

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name='ca_pointnet'
    )

    # Trainer
    print("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='32-true',
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=2  # Gradient accumulation to simulate larger batch
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")

    trainer.fit(model, dm)

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

    # Test
    print("Testing best model...")
    test_results = trainer.test(model, dm, ckpt_path='best')

    # Extract training history
    print("\nExtracting training history...")
    training_history = {
        'train_loss_epoch': [],
        'train_acc_epoch': [],
        'val_loss': [],
        'val_acc': [],
        'val_miou': [],
        'epoch': []
    }

    # Extract from TensorBoard logs
    import glob
    tensorboard_files = glob.glob(str(log_dir / 'ca_pointnet' / 'version_*' / 'events.out.tfevents.*'))

    if tensorboard_files:
        from tensorboard.backend.event_processing import event_accumulator

        try:
            ea = event_accumulator.EventAccumulator(str(Path(tensorboard_files[0]).parent))
            ea.Reload()

            for tag in ['train_loss_epoch', 'train_acc_epoch', 'val_loss', 'val_acc', 'val_miou']:
                if tag in ea.Tags()['scalars']:
                    events = ea.Scalars(tag)
                    training_history[tag] = [{'step': e.step, 'value': e.value} for e in events]

            print(f"Extracted {len(training_history['val_miou'])} epochs of training history")
        except Exception as e:
            print(f"Warning: Could not extract TensorBoard history: {e}")

    # Save training history
    history_path = Path(args.output_dir) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Calculate and log training duration
    end_time = time.time()
    end_datetime = datetime.now()
    total_duration = end_time - start_time
    hours = int(total_duration // 3600)
    minutes = int((total_duration % 3600) // 60)
    seconds = int(total_duration % 60)

    # Save training summary
    best_model_path = checkpoint_dir / 'best_model.ckpt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)

        summary = {
            'model': 'CA-PointNet++',
            'paper': 'Semantic Segmentation of Transmission Corridor 3D Point Clouds Based on CA-PointNet++',
            'best_model_path': str(best_model_path),
            'best_val_miou': float(checkpoint_callback.best_model_score),
            'test_results': test_results[0] if test_results else {},
            'total_epochs_trained': trainer.current_epoch + 1,
            'training_duration': {
                'total_seconds': total_duration,
                'formatted': f"{hours}h {minutes}m {seconds}s"
            },
            'start_time': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'hyperparameters': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'momentum': args.momentum,
                'num_points': args.num_points,
                'input_channels': 6,
                'input_format': 'XYZ + RGB'
            },
            'architecture': {
                'encoder_layers': 4,
                'decoder_layers': 4,
                'sa1': {'npoint': 1024, 'radius': 0.1, 'nsample': 32, 'out_channel': 64},
                'sa2': {'npoint': 256, 'radius': 0.2, 'nsample': 32, 'out_channel': 128},
                'sa3': {'npoint': 64, 'radius': 0.4, 'nsample': 32, 'out_channel': 256},
                'sa4': {'npoint': 16, 'radius': 0.8, 'nsample': 32, 'out_channel': 512}
            },
            'class_weights': class_weights,
            'classes': ['Tower', 'Transmission Line', 'Ground Wire', 'Ground']
        }

        summary_path = Path(args.output_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Model: CA-PointNet++")
        print(f"Start Time: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Training Duration: {hours}h {minutes}m {seconds}s ({total_duration:.2f} seconds)")
        print(f"Best validation mIoU: {summary['best_val_miou']:.4f}")
        print(f"Total epochs trained: {summary['total_epochs_trained']}")
        print(f"\nTraining summary saved to {summary_path}")
        print(f"Best model saved to {best_model_path}")
        print(f"\nAll results saved to {args.output_dir}/")
        print("="*80)
