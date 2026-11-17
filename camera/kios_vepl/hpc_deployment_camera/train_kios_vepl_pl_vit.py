"""
KIOS VEPL Power Line Detection Training - Enhanced Transformer-Based
=====================================================================

This script trains a binary power line segmentation model using Mix Transformer (MiT) architecture
with advanced training techniques optimized for challenging thin-structure detection.

Model: U-Net + Mix Transformer (SegFormer backbone: mit_b0-b5)
Loss: Combined Focal Tversky + Weighted BCE (handles extreme class imbalance)
Optimizer: AdamW with cosine annealing and warmup
Regularization: Gradient clipping, EMA, enhanced augmentations

Features:
- Focal Tversky + Weighted BCE loss (optimized for thin power lines)
- Tversky α=0.3, β=0.7 (emphasizes recall - catching all power lines)
- Cosine annealing scheduler with warmup (better convergence)
- Exponential Moving Average (EMA) for stable predictions
- Gradient clipping for training stability
- Enhanced augmentations (elastic transforms, cutout, noise)
- Comprehensive visualizations (training curves, predictions)
- Early stopping with reduced patience
- CUDA-optimized with AMP support

Usage:
    python train_kios_vepl_pl_vit.py --batch_size 32 --epochs 100 --lr 0.0001 --encoder mit_b2
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json
import copy

import cv2
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for HPC


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='KIOS VEPL Power Line Detection Training - Vision Transformer')

    # Data paths
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs_pl_vit',
                        help='Path to output directory for models and logs')

    # Model configuration
    parser.add_argument('--architecture', type=str, default='Unet',
                        choices=['Unet', 'DeepLabV3', 'FPN', 'UnetPlusPlus'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='mit_b2',
                        choices=['mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5'],
                        help='Mix Transformer encoder (SegFormer backbone)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size (256, 384, 512, or 640 for MiT)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (smaller for ViT due to memory)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (higher for better convergence)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--min_lr', type=float, default=1e-7,
                        help='Minimum learning rate')

    # Data split
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='Training data ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation data ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Training options
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (reduced for faster stopping)')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.999,
                        help='EMA decay rate')

    # Loss function parameters (Tversky for power lines)
    parser.add_argument('--tversky_alpha', type=float, default=0.3,
                        help='Tversky alpha (FP weight) - lower values reduce false positives')
    parser.add_argument('--tversky_beta', type=float, default=0.7,
                        help='Tversky beta (FN weight) - higher values reduce false negatives')
    parser.add_argument('--tversky_weight', type=float, default=0.7,
                        help='Tversky loss weight in combined loss')
    parser.add_argument('--bce_weight', type=float, default=0.3,
                        help='BCE loss weight in combined loss')
    parser.add_argument('--bce_pos_weight', type=float, default=10.0,
                        help='BCE positive class weight for class imbalance')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision (AMP)')

    # Checkpointing
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
class TverskyLoss(nn.Module):
    """
    Tversky Loss with focal component for extreme class imbalance.

    Tversky index is a generalization of Dice coefficient that allows
    weighting of false positives and false negatives differently.

    Args:
        alpha: Weight for false positives (lower = penalize FP less)
        beta: Weight for false negatives (higher = penalize FN more)
        smooth: Smoothing factor to avoid division by zero
    """

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()

        # Tversky index
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Focal component to focus on hard examples
        focal_weight = torch.pow(1 - tversky_index, 2)

        return focal_weight * (1 - tversky_index)


class CombinedLoss(nn.Module):
    """
    Combined Focal Tversky + Weighted BCE loss for power line detection.

    Tversky loss handles extreme class imbalance and emphasizes recall,
    while BCE provides pixel-wise supervision.
    """

    def __init__(self, tversky_alpha=0.3, tversky_beta=0.7,
                 tversky_weight=0.7, bce_weight=0.3, pos_weight=10.0):
        super(CombinedLoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, pred, target):
        tversky = self.tversky_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.tversky_weight * tversky + self.bce_weight * bce


# =============================================================================
# EXPONENTIAL MOVING AVERAGE
# =============================================================================
class ModelEMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.model = model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()

        for param in self.shadow.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        """Update EMA parameters."""
        for ema_param, model_param in zip(self.shadow.parameters(), model.parameters()):
            ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Apply EMA weights to model."""
        self.backup = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.shadow.state_dict())

    def restore(self):
        """Restore original weights."""
        self.model.load_state_dict(self.backup)
        del self.backup


# =============================================================================
# COSINE ANNEALING WITH WARMUP
# =============================================================================
class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations.
        eta_min: Minimum learning rate.
        warmup_steps: Number of warmup steps.
        last_epoch: The index of last epoch.
    """

    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.T_max - self.warmup_steps)
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(output_dir):
    """Setup logging to both file and console."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f'training_{datetime.now():%Y%m%d_%H%M%S}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


# =============================================================================
# DATASET CLASS
# =============================================================================
class KIOSPowerLineDataset(Dataset):
    """Dataset class for KIOS VEPL power line segmentation."""

    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load RGB image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load binary mask (already preprocessed as grayscale for KIOS VEPL)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.float32)
        binary_mask = np.expand_dims(binary_mask, axis=-1)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
            binary_mask = binary_mask.permute(2, 0, 1)

        return image, binary_mask


# =============================================================================
# DATA AUGMENTATION
# =============================================================================
def get_train_transform(image_size=512):
    """Enhanced data augmentation for power line detection with Mix Transformer."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.15,
            rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5
        ),

        # Elastic deformations
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=1.0),
        ], p=0.3),

        # Spectral augmentations - optimized for power line detection
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.4),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=15, p=0.4),

        # Additional augmentations for robustness
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),

        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.3
        ),

        # ImageNet normalization (required for pretrained MiT/SegFormer)
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


def get_val_transform(image_size=512):
    """Validation transform without augmentation."""
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ])


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(loader, model, optimizer, loss_fn, device, scaler=None, use_amp=False, ema=None, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping and EMA."""
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    loop = tqdm(loader, desc="Training", leave=False)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device, non_blocking=True)
        targets = targets.float().to(device, non_blocking=True)

        if data.size(0) < 2:
            continue

        # Forward pass
        if use_amp:
            with torch.autocast(device_type="cuda", enabled=True):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        total_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Update EMA
        if ema is not None:
            ema.update(model)

        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })

    return total_loss / num_batches


def evaluate(loader, model, device):
    """Evaluate model on validation/test set."""
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if x.size(0) < 1:
                continue

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

            tp, fp, fn, tn = smp.metrics.get_stats(preds.long(), y.long(), mode='binary')
            iou_score += smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            dice_score += smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")
            num_batches += 1

    iou = iou_score / num_batches
    dice = dice_score / num_batches
    acc = num_correct / num_pixels * 100

    model.train()
    return iou.item(), dice.item(), acc.item()


def save_checkpoint(state, filepath):
    """Save checkpoint to file."""
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def plot_training_history(history, output_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # IoU curve
    axes[0, 1].plot(history['val_iou'], label='Validation IoU', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('IoU', fontsize=12)
    axes[0, 1].set_title('Validation IoU', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dice curve
    axes[1, 0].plot(history['val_dice'], label='Validation Dice', color='blue', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Dice Score', fontsize=12)
    axes[1, 0].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate curve
    axes[1, 1].plot(history['learning_rates'], label='Learning Rate', color='red', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved: {save_path}")


def visualize_predictions(model, loader, device, output_dir, num_samples=8):
    """Visualize model predictions on sample images."""
    model.eval()

    images_to_visualize = []
    masks_to_visualize = []
    preds_to_visualize = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)

            # Get predictions
            pred = torch.sigmoid(model(x))
            pred_binary = (pred > 0.5).float()

            # Move to CPU
            x = x.cpu()
            y = y.cpu()
            pred_binary = pred_binary.cpu()

            images_to_visualize.append(x)
            masks_to_visualize.append(y)
            preds_to_visualize.append(pred_binary)

            if len(images_to_visualize) * x.size(0) >= num_samples:
                break

    # Concatenate batches
    images = torch.cat(images_to_visualize, dim=0)[:num_samples]
    masks = torch.cat(masks_to_visualize, dim=0)[:num_samples]
    preds = torch.cat(preds_to_visualize, dim=0)[:num_samples]

    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)

    # Create visualization
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # Original image
        img = images[i].permute(1, 2, 0).numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image', fontsize=10)
        axes[i, 0].axis('off')

        # Ground truth mask
        mask = masks[i, 0].numpy()
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth', fontsize=10)
        axes[i, 1].axis('off')

        # Prediction
        pred = preds[i, 0].numpy()
        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 2].axis('off')

        # Overlay (green=correct, red=false positive, blue=false negative)
        overlay = img.copy()

        # True positives - green
        tp_mask = (mask == 1) & (pred == 1)
        overlay[tp_mask] = overlay[tp_mask] * 0.5 + np.array([0, 1, 0]) * 0.5

        # False positives - red
        fp_mask = (mask == 0) & (pred == 1)
        overlay[fp_mask] = overlay[fp_mask] * 0.5 + np.array([1, 0, 0]) * 0.5

        # False negatives - blue
        fn_mask = (mask == 1) & (pred == 0)
        overlay[fn_mask] = overlay[fn_mask] * 0.5 + np.array([0, 0, 1]) * 0.5

        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay (G=TP, R=FP, B=FN)', fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'predictions_visualization.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    logging.info(f"Predictions visualization saved: {save_path}")

    model.train()


def plot_metrics_comparison(history, output_dir):
    """Plot comprehensive metrics comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Combined loss and IoU
    ax1 = axes[0]
    ax1_twin = ax1.twinx()

    line1 = ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    line2 = ax1_twin.plot(epochs, history['val_iou'], 'g-', label='Val IoU', linewidth=2)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax1_twin.set_ylabel('IoU', color='g', fontsize=12)
    ax1.set_title('Loss vs IoU', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='g')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # IoU and Dice comparison
    axes[1].plot(epochs, history['val_iou'], label='IoU', linewidth=2, color='green')
    axes[1].plot(epochs, history['val_dice'], label='Dice', linewidth=2, color='blue')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('IoU vs Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Accuracy
    axes[2].plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2, color='purple')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Accuracy (%)', fontsize=12)
    axes[2].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Metrics comparison saved: {save_path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def main():
    args = parse_args()

    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("="*70)
    logging.info("KIOS VEPL Power Line Detection - Mix Transformer (SegFormer)")
    logging.info("="*70)
    logging.info(f"Log file: {log_file}")

    # Log all arguments
    logging.info("\nConfiguration:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # Set random seed
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    logging.info(f"\nDevice: {device}")
    if device.type == 'cuda':
        logging.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    use_amp = args.amp and device.type == 'cuda'
    logging.info(f"AMP enabled: {use_amp}")

    # Load data
    logging.info("\n" + "="*70)
    logging.info("Loading data...")
    logging.info("="*70)

    img_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'pl_masks')  # Power line masks

    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.error(f"Data directories not found: {img_dir} or {mask_dir}")
        sys.exit(1)

    all_image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    all_mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    logging.info(f"Total samples: {len(all_image_paths)}")

    if len(all_image_paths) != len(all_mask_paths):
        logging.error(f"Mismatch: {len(all_image_paths)} images vs {len(all_mask_paths)} masks")
        sys.exit(1)

    # Data split
    test_size = 1.0 - args.train_split - args.val_split

    train_val_imgs, test_imgs, train_val_msks, test_msks = train_test_split(
        all_image_paths, all_mask_paths,
        test_size=test_size,
        random_state=args.random_seed
    )

    val_ratio = args.val_split / (args.train_split + args.val_split)
    train_imgs, val_imgs, train_msks, val_msks = train_test_split(
        train_val_imgs, train_val_msks,
        test_size=val_ratio,
        random_state=args.random_seed
    )

    logging.info(f"Training samples: {len(train_imgs)} ({len(train_imgs)/len(all_image_paths)*100:.1f}%)")
    logging.info(f"Validation samples: {len(val_imgs)} ({len(val_imgs)/len(all_image_paths)*100:.1f}%)")
    logging.info(f"Test samples: {len(test_imgs)} ({len(test_imgs)/len(all_image_paths)*100:.1f}%)")

    # Create data loaders
    logging.info(f"\nCreating data loaders (batch_size={args.batch_size}, num_workers={args.num_workers})...")

    train_loader = DataLoader(
        KIOSPowerLineDataset(train_imgs, train_msks, transform=get_train_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        KIOSPowerLineDataset(val_imgs, val_msks, transform=get_val_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    test_loader = DataLoader(
        KIOSPowerLineDataset(test_imgs, test_msks, transform=get_val_transform(args.image_size)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    logging.info(f"Training batches: {len(train_loader)}")
    logging.info(f"Validation batches: {len(val_loader)}")
    logging.info(f"Test batches: {len(test_loader)}")

    # Initialize model
    logging.info("\n" + "="*70)
    logging.info("Initializing Mix Transformer model...")
    logging.info("="*70)

    logging.info(f"Architecture: {args.architecture} + {args.encoder}")
    logging.info(f"Image size: {args.image_size}x{args.image_size}")
    logging.info(f"Encoder: Mix Transformer (SegFormer)")

    try:
        if args.architecture == 'Unet':
            model = smp.Unet(
                encoder_name=args.encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        elif args.architecture == 'DeepLabV3':
            model = smp.DeepLabV3(
                encoder_name=args.encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        elif args.architecture == 'FPN':
            model = smp.FPN(
                encoder_name=args.encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
        elif args.architecture == 'UnetPlusPlus':
            model = smp.UnetPlusPlus(
                encoder_name=args.encoder,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1,
            )
    except Exception as e:
        logging.error(f"Failed to initialize model: {e}")
        logging.error(f"Available encoders: mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5")
        sys.exit(1)

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function - Combined Focal Tversky + Weighted BCE for power lines
    loss_fn = CombinedLoss(
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_weight=args.tversky_weight,
        bce_weight=args.bce_weight,
        pos_weight=args.bce_pos_weight
    )
    logging.info(f"\nLoss: Combined ({args.tversky_weight*100:.0f}% Focal Tversky + {args.bce_weight*100:.0f}% Weighted BCE)")
    logging.info(f"Tversky α={args.tversky_alpha}, β={args.tversky_beta} (emphasizes recall)")
    logging.info(f"BCE positive weight: {args.bce_pos_weight}x (handles class imbalance)")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logging.info(f"\nOptimizer: AdamW")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")

    # Learning rate scheduler - Cosine Annealing with Warmup
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
        warmup_steps=args.warmup_epochs
    )
    logging.info(f"LR Scheduler: CosineAnnealingWarmup (warmup={args.warmup_epochs}, T_max={args.epochs}, min_lr={args.min_lr})")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Exponential Moving Average
    ema = ModelEMA(model, decay=args.ema_decay) if args.use_ema else None
    if ema:
        logging.info(f"EMA: Enabled (decay={args.ema_decay})")

    # Gradient clipping
    logging.info(f"Gradient Clipping: Max norm = {args.max_grad_norm}")

    # Resume from checkpoint
    start_epoch = 0
    best_iou = -1

    if args.resume:
        if os.path.exists(args.resume):
            logging.info(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_iou = checkpoint.get('best_iou', -1)
            logging.info(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")
        else:
            logging.warning(f"Checkpoint not found: {args.resume}")

    # Training loop
    logging.info("\n" + "="*70)
    logging.info("Starting training...")
    logging.info("="*70)

    training_history = {
        'train_loss': [],
        'val_iou': [],
        'val_dice': [],
        'val_acc': [],
        'learning_rates': []
    }

    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e}")

        # Training with EMA and gradient clipping
        train_loss = train_epoch(
            train_loader, model, optimizer, loss_fn, device,
            scaler, use_amp, ema, args.max_grad_norm
        )

        # Validation
        val_iou, val_dice, val_acc = evaluate(val_loader, model, device)

        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.2f}%")

        # Update scheduler (cosine annealing updates every epoch)
        scheduler.step()

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_iou'].append(val_iou)
        training_history['val_dice'].append(val_dice)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)

        # Check for LR changes
        new_lr = optimizer.param_groups[0]['lr']
        if abs(new_lr - current_lr) > 1e-10:
            logging.info(f"Learning rate changed: {current_lr:.2e} -> {new_lr:.2e}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'val_dice': val_dice,
                'val_acc': val_acc,
                'args': vars(args)
            }

            # Save EMA if enabled
            if ema is not None:
                checkpoint['ema_state_dict'] = ema.shadow.state_dict()

            best_model_path = os.path.join(args.output_dir, 'best_model_pl_vit.pth')
            save_checkpoint(checkpoint, best_model_path)
            logging.info(f"NEW BEST MODEL (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epoch(s) (best: {best_iou:.4f})")

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logging.info(f"\nEarly stopping triggered (patience: {args.early_stop_patience})")
            break

        logging.info("-" * 70)

    # Generate training visualizations
    logging.info("\n" + "="*70)
    logging.info("Generating training visualizations...")
    logging.info("="*70)

    plot_training_history(training_history, args.output_dir)
    plot_metrics_comparison(training_history, args.output_dir)

    # Final testing
    logging.info("\n" + "="*70)
    logging.info("Final testing on unseen data...")
    logging.info("="*70)

    # Load best model
    best_model_path = os.path.join(args.output_dir, 'best_model_pl_vit.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load EMA if available
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.shadow.load_state_dict(checkpoint['ema_state_dict'])
            logging.info(f"Loaded best model with EMA from epoch {checkpoint['epoch']+1}")
        else:
            logging.info(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    else:
        logging.warning("Best model not found, using current model state")

    # Evaluate with EMA if available
    if ema is not None:
        ema.apply_shadow()
        test_iou_ema, test_dice_ema, test_acc_ema = evaluate(test_loader, model, device)
        ema.restore()

        logging.info(f"\nFINAL RESULTS (EMA):")
        logging.info(f"Test IoU: {test_iou_ema:.4f}")
        logging.info(f"Test Dice: {test_dice_ema:.4f}")
        logging.info(f"Test Accuracy: {test_acc_ema:.2f}%")

        test_iou, test_dice, test_acc = test_iou_ema, test_dice_ema, test_acc_ema
    else:
        test_iou, test_dice, test_acc = evaluate(test_loader, model, device)

        logging.info(f"\nFINAL RESULTS:")
        logging.info(f"Test IoU: {test_iou:.4f}")
        logging.info(f"Test Dice: {test_dice:.4f}")
        logging.info(f"Test Accuracy: {test_acc:.2f}%")

    logging.info(f"Best Val IoU: {best_iou:.4f}")

    # Visualize predictions on validation set
    logging.info("\nGenerating prediction visualizations...")
    if ema is not None:
        ema.apply_shadow()
    visualize_predictions(model, val_loader, device, args.output_dir, num_samples=8)
    if ema is not None:
        ema.restore()

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history_pl_vit.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logging.info(f"\nTraining history saved: {history_path}")

    # Save final summary
    summary = {
        'config': vars(args),
        'best_val_iou': best_iou,
        'test_iou': test_iou,
        'test_dice': test_dice,
        'test_acc': test_acc,
        'total_epochs': len(training_history['train_loss']),
        'final_lr': training_history['learning_rates'][-1] if training_history['learning_rates'] else args.lr
    }

    summary_path = os.path.join(args.output_dir, 'training_summary_pl_vit.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Training summary saved: {summary_path}")

    logging.info("\n" + "="*70)
    logging.info("Training completed successfully!")
    logging.info("="*70)


if __name__ == '__main__':
    main()
