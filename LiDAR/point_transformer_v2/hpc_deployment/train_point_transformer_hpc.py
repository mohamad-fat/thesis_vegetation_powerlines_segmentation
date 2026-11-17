"""
Point Transformer V2 Training for HPC
=====================================
Train Point Transformer V2 model on LiDAR point cloud data for semantic segmentation.

Classes:
  - 0: High Vegetation
  - 1: Power Lines
  - 2: Pylons
  - 3: Others

Usage:
    python train_point_transformer_hpc.py \
        --data_dir ../data/processed_data_optimized \
        --output_dir ./outputs_lidar \
        --batch_size 16 \
        --epochs 50 \
        --num_points 8192 \
        --d_model 256 \
        --num_layers 4
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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

import numpy as np
from tqdm import tqdm

# ==============================================================================
# Parse Arguments
# ==============================================================================
parser = argparse.ArgumentParser(description='Train Point Transformer V2 on LiDAR data')
parser.add_argument('--data_dir', type=str, default='../data/processed_data_optimized',
                    help='Path to processed data directory')
parser.add_argument('--output_dir', type=str, default='./outputs_lidar',
                    help='Output directory for checkpoints and logs')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size (16 for V100 GPU)')
parser.add_argument('--epochs', type=int, default=50,
                    help='Maximum number of training epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay for optimizer')
parser.add_argument('--num_points', type=int, default=8192,
                    help='Number of points per sample (8192 for GPU)')
parser.add_argument('--d_model', type=int, default=256,
                    help='Model dimension')
parser.add_argument('--num_layers', type=int, default=4,
                    help='Number of transformer layers')
parser.add_argument('--k_neighbors', type=int, default=16,
                    help='Number of nearest neighbors for attention')
parser.add_argument('--dropout', type=float, default=0.3,
                    help='Dropout rate')
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
    # Enable CUDA optimizations for V100
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels for best performance
    torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
    print("CUDA optimizations enabled (cudnn.benchmark=True)")

print("\n" + "="*80)
print("POINT TRANSFORMER V2 TRAINING")
print("="*80)
print(f"Data directory: {args.data_dir}")
print(f"Output directory: {args.output_dir}")
print(f"Batch size: {args.batch_size}")
print(f"Max epochs: {args.epochs}")
print(f"Points per sample: {args.num_points}")
print(f"Model: {args.num_layers} layers, {args.d_model} dimensions")
print("="*80 + "\n")

# ==============================================================================
# Dataset Classes
# ==============================================================================
class PowerLineDataset(Dataset):
    """Dataset for power line point cloud segmentation"""

    def __init__(self, data_dir, split='train', num_points=8192, use_fps=True, max_samples=None):
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
        points = sample['input_pcd']
        labels = sample['semantic_labels']

        # Ensure labels is 1D
        labels = labels.reshape(-1)

        # Convert to 0-indexed
        labels = labels - 1

        n_points = points.shape[0]

        if n_points > self.num_points:
            if self.use_fps:
                fps_idx = self.farthest_point_sampling(points.numpy(), self.num_points)
                points = points[fps_idx]
                labels = labels[fps_idx]
            else:
                choice = np.random.choice(n_points, self.num_points, replace=False)
                points = points[choice]
                labels = labels[choice]
        elif n_points < self.num_points:
            choice = np.random.choice(n_points, self.num_points - n_points, replace=True)
            extra_points = points[choice]
            extra_labels = labels[choice]
            points = torch.cat([points, extra_points], dim=0)
            labels = torch.cat([labels, extra_labels], dim=0)

        return {'points': points, 'labels': labels, 'idx': idx}


class PowerLineDataModule(L.LightningDataModule):
    """Lightning DataModule for power line segmentation"""

    def __init__(self, data_dir, batch_size=16, num_points=8192, num_workers=8,
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
            self.train_dataset = PowerLineDataset(
                self.data_dir, split='train', num_points=self.num_points,
                use_fps=self.use_fps, max_samples=self.max_samples
            )
            self.val_dataset = PowerLineDataset(
                self.data_dir, split='test', num_points=self.num_points,
                use_fps=False, max_samples=None
            )

        if stage == 'test' or stage is None:
            self.test_dataset = PowerLineDataset(
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
# Model Architecture
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

        # KNN
        idx = knn(positions.transpose(1, 2), k)

        # Gather neighbor features
        idx_base = torch.arange(0, B, device=features.device).view(-1, 1, 1) * N
        idx_expanded = (idx + idx_base).view(B * N * k)

        features_flat = features.view(B * N, C)
        neighbor_features = features_flat[idx_expanded].view(B, N, k, C)

        positions_flat = positions.view(B * N, 3)
        neighbor_positions = positions_flat[idx_expanded].view(B, N, k, 3)

        # Relative positions
        relative_pos = positions.unsqueeze(2) - neighbor_positions
        pos_encoding = self.pos_mlp(relative_pos)

        # Attention
        q = self.q(features).unsqueeze(2)
        k_feat = self.k_linear(neighbor_features) + pos_encoding
        v = self.v(neighbor_features) + pos_encoding

        attn = torch.matmul(q, k_feat.transpose(-2, -1)) / (C ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).squeeze(2)

        # Residual connections
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


class PointTransformerLightning(L.LightningModule):
    """Lightning module for Point Transformer training"""
    def __init__(self, num_classes=4, d_model=256, num_layers=4, k=16,
                 lr=1e-3, weight_decay=1e-4, class_weights=None, max_epochs=50, dropout=0.3):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointTransformerV2(
            num_classes=num_classes, d_model=d_model, num_layers=num_layers,
            k=k, dropout=dropout
        )

        if class_weights is not None:
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            # Replace near-zero weights with 1.0
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
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }


# ==============================================================================
# Calculate Class Weights
# ==============================================================================
def calculate_class_weights(data_dir, num_samples=100):
    """Calculate class weights from training data"""
    print("\nCalculating class weights from training data...")
    dataset = PowerLineDataset(data_dir, split='train', use_fps=False)

    class_counts = torch.zeros(4)
    sample_size = min(len(dataset), num_samples)
    print(f"Sampling {sample_size} chunks to estimate class distribution...")

    for i in tqdm(range(sample_size), desc="Sampling"):
        sample = dataset[i]
        labels = sample['labels']
        for cls in range(4):
            class_counts[cls] += (labels == cls).sum()

    total = class_counts.sum()
    class_weights = total / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 4

    class_names = ['High Vegetation', 'Power Lines', 'Pylons', 'Others']
    print("\nClass distribution:")
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, class_weights)):
        percentage = 100 * count / total
        print(f"  {name}: {int(count):,} points ({percentage:.2f}%) -> weight: {weight:.3f}")

    return class_weights.tolist()


# ==============================================================================
# Main Training
# ==============================================================================
if __name__ == '__main__':
    # Class weights to address severe imbalance
    # High Vegetation: 60% → weight 1.0
    # Power Lines: 0.2% → weight 10.0 (50x rarer, 10x weight)
    # Pylons: 0.01% → weight 50.0 (6000x rarer, 50x weight)
    # Others: 40% → weight 1.5
    class_weights = [1.0, 10.0, 50.0, 1.5]
    print(f"\nUsing manual class weights: {class_weights}")
    print("  [High Vegetation, Power Lines, Pylons, Others]")
    print("  Addressing severe class imbalance (Pylons: 0.01%, Power Lines: 0.2%)\n")

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
    print("Initializing model...")
    model = PointTransformerLightning(
        num_classes=4,
        d_model=args.d_model,
        num_layers=args.num_layers,
        k=args.k_neighbors,
        lr=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights,
        max_epochs=args.epochs,
        dropout=args.dropout
    )
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    # Callbacks - SAVE ONLY BEST MODEL
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='best_model',
        monitor='val_miou',
        mode='max',
        save_top_k=1,  # ONLY save best model
        save_last=False,  # Don't save last checkpoint
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_miou',
        patience=args.early_stop_patience,
        mode='max',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Compact progress bar for cleaner output
    progress_bar = TQDMProgressBar(refresh_rate=1)

    logger = TensorBoardLogger(
        save_dir=str(log_dir),
        name='point_transformer'
    )

    # Trainer
    print("Setting up trainer...")
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator='gpu',
        devices=1,
        precision='32-true',  # Use full precision for stability (mixed precision causes NaN)
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor, progress_bar],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=args.gradient_clip
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

    # Extract training history from logged metrics
    print("\nExtracting training history...")
    metrics = trainer.callback_metrics

    # Get all logged metrics from trainer
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
    tensorboard_files = glob.glob(str(log_dir / 'point_transformer' / 'version_*' / 'events.out.tfevents.*'))

    if tensorboard_files:
        from tensorboard.backend.event_processing import event_accumulator

        try:
            ea = event_accumulator.EventAccumulator(str(Path(tensorboard_files[0]).parent))
            ea.Reload()

            # Extract scalars
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

    # Save training summary
    best_model_path = checkpoint_dir / 'best_model.ckpt'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path)

        summary = {
            'best_model_path': str(best_model_path),
            'best_val_miou': float(checkpoint_callback.best_model_score),
            'test_results': test_results[0] if test_results else {},
            'total_epochs_trained': trainer.current_epoch + 1,
            'hyperparameters': {
                'batch_size': args.batch_size,
                'epochs': args.epochs,
                'lr': args.lr,
                'num_points': args.num_points,
                'd_model': args.d_model,
                'num_layers': args.num_layers,
                'k_neighbors': args.k_neighbors,
                'dropout': args.dropout
            },
            'class_weights': class_weights
        }

        summary_path = Path(args.output_dir) / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nTraining summary saved to {summary_path}")
        print(f"Best model saved to {best_model_path}")
        print(f"Best validation mIoU: {summary['best_val_miou']:.4f}")
        print(f"Total epochs trained: {summary['total_epochs_trained']}")
        print(f"\nAll results saved to {args.output_dir}/")
