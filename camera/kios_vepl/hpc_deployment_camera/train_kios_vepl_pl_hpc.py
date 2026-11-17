"""
KIOS VEPL Power Line Detection Training - HPC Optimized
========================================================

This script trains a binary power line segmentation model optimized for HPC/CUDA environments.

Model: DeepLabV3 + ResNet34
Loss: Combined Focal Tversky + Weighted BCE
Device: CUDA-optimized with AMP support

Usage:
    python train_kios_vepl_hpc.py --batch_size 32 --epochs 20 --lr 0.0001
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import json

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


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='KIOS VEPL Power Line Detection Training')

    # Data paths
    parser.add_argument('--data_dir', type=str, default='./processed_data',
                        help='Path to processed data directory')
    parser.add_argument('--output_dir', type=str, default='./output_pl',
                        help='Path to output directory for models and logs')

    # Model configuration
    parser.add_argument('--architecture', type=str, default='DeepLabV3',
                        choices=['DeepLabV3', 'Unet', 'FPN', 'UnetPlusPlus'],
                        help='Model architecture')
    parser.add_argument('--encoder', type=str, default='resnet34',
                        help='Encoder backbone')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32 for HPC)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00001,
                        help='Weight decay')

    # Data split
    parser.add_argument('--train_split', type=float, default=0.6,
                        help='Training data ratio')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation data ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Training options
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--early_stop_patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='Learning rate scheduler patience')

    # Loss function parameters
    parser.add_argument('--tversky_alpha', type=float, default=0.3,
                        help='Tversky alpha (FP weight)')
    parser.add_argument('--tversky_beta', type=float, default=0.7,
                        help='Tversky beta (FN weight)')
    parser.add_argument('--bce_pos_weight', type=float, default=15.0,
                        help='BCE positive class weight')

    # Device configuration
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='Use automatic mixed precision (AMP)')

    # Checkpointing
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    return parser.parse_args()


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
# LOSS FUNCTIONS
# =============================================================================
class TverskyLoss(nn.Module):
    """Tversky Loss with focal component for extreme class imbalance."""

    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        denominator = TP + self.alpha*FP + self.beta*FN + self.smooth
        Tversky = (TP + self.smooth) / denominator
        focal_weight = torch.pow(1 - Tversky, 2)

        return focal_weight * (1 - Tversky)


class CombinedLoss(nn.Module):
    """Combined Focal Tversky + Weighted BCE loss."""

    def __init__(self, tversky_alpha, tversky_beta, tversky_weight=0.7,
                 bce_weight=0.3, pos_weight=10.0):
        super(CombinedLoss, self).__init__()
        self.tversky_weight = tversky_weight
        self.bce_weight = bce_weight
        self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, inputs, targets):
        tversky = self.tversky_loss(inputs, targets)
        bce = self.bce_loss(inputs, targets)
        return self.tversky_weight * tversky + self.bce_weight * bce


# =============================================================================
# DATASET CLASS
# =============================================================================
class KIOSPowerLineDataset(Dataset):
    """Dataset class for KIOS power line segmentation."""

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

        # Load binary mask
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
def get_train_transform():
    """Aggressive augmentation for small dataset."""
    return A.Compose([
        A.Resize(height=256, width=256),

        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

        # Spectral augmentations
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_val_transform():
    """Validation transform without augmentation."""
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(loader, model, optimizer, loss_fn, device, scaler=None, use_amp=False):
    """Train for one epoch."""
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
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

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
    valid_batches = 0

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
            batch_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
            batch_dice = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

            if not torch.isnan(batch_iou):
                iou_score += batch_iou
                valid_batches += 1
            if not torch.isnan(batch_dice):
                dice_score += batch_dice

    if valid_batches > 0:
        iou = iou_score / valid_batches
    else:
        iou = torch.tensor(0.0)
        logging.warning("All IoU calculations returned NaN")

    dice = dice_score / len(loader)
    acc = num_correct / num_pixels * 100

    model.train()
    return iou.item(), dice.item(), acc.item()


def save_checkpoint(state, filepath):
    """Save checkpoint to file."""
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved: {filepath}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def main():
    args = parse_args()

    # Setup logging
    log_file = setup_logging(args.output_dir)
    logging.info("="*70)
    logging.info("KIOS VEPL Power Line Detection Training - HPC Mode")
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
        KIOSPowerLineDataset(train_imgs, train_msks, transform=get_train_transform()),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    val_loader = DataLoader(
        KIOSPowerLineDataset(val_imgs, val_msks, transform=get_val_transform()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )

    test_loader = DataLoader(
        KIOSPowerLineDataset(test_imgs, test_msks, transform=get_val_transform()),
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
    logging.info("Initializing model...")
    logging.info("="*70)

    logging.info(f"Architecture: {args.architecture} + {args.encoder}")

    if args.architecture == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    elif args.architecture == 'Unet':
        model = smp.Unet(
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

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    loss_fn = CombinedLoss(
        tversky_alpha=args.tversky_alpha,
        tversky_beta=args.tversky_beta,
        tversky_weight=0.7,
        bce_weight=0.3,
        pos_weight=args.bce_pos_weight
    )
    logging.info(f"\nLoss: Combined (70% Focal Tversky + 30% Weighted BCE)")
    logging.info(f"Tversky α={args.tversky_alpha}, β={args.tversky_beta}")
    logging.info(f"BCE positive weight: {args.bce_pos_weight}x")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    logging.info(f"\nOptimizer: AdamW")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")

    # Learning rate scheduler
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )
    logging.info(f"LR Scheduler: ReduceLROnPlateau (patience={args.lr_patience})")

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

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

        # Training
        train_loss = train_epoch(train_loader, model, optimizer, loss_fn, device, scaler, use_amp)

        # Validation
        val_iou, val_dice, val_acc = evaluate(val_loader, model, device)

        logging.info(f"Train Loss: {train_loss:.4f}")
        logging.info(f"Val IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.2f}%")

        # Update scheduler
        scheduler.step(val_iou)

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_iou'].append(val_iou)
        training_history['val_dice'].append(val_dice)
        training_history['val_acc'].append(val_acc)
        training_history['learning_rates'].append(current_lr)

        # Check for LR reduction
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < current_lr:
            logging.info(f"Learning rate reduced: {current_lr:.2e} -> {new_lr:.2e}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'val_dice': val_dice,
                'val_acc': val_acc,
                'args': vars(args)
            }

            best_model_path = os.path.join(args.output_dir, 'best_model.pth')
            save_checkpoint(checkpoint, best_model_path)
            logging.info(f"✅ NEW BEST MODEL (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            logging.info(f"No improvement for {patience_counter} epoch(s) (best: {best_iou:.4f})")

        # Periodic checkpoint (disabled - only saving best model)
        # if (epoch + 1) % args.save_freq == 0:
        #     checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'best_iou': best_iou,
        #         'args': vars(args)
        #     }
        #     save_checkpoint(checkpoint, checkpoint_path)

        # Early stopping
        if patience_counter >= args.early_stop_patience:
            logging.info(f"\n🛑 Early stopping triggered (patience: {args.early_stop_patience})")
            break

        logging.info("-" * 70)

    # Final testing
    logging.info("\n" + "="*70)
    logging.info("Final testing on unseen data...")
    logging.info("="*70)

    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_iou, test_dice, test_acc = evaluate(test_loader, model, device)

    logging.info(f"\nFINAL RESULTS:")
    logging.info(f"Test IoU: {test_iou:.4f}")
    logging.info(f"Test Dice: {test_dice:.4f}")
    logging.info(f"Test Accuracy: {test_acc:.2f}%")
    logging.info(f"Best Val IoU: {best_iou:.4f}")

    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
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
        'final_lr': training_history['learning_rates'][-1]
    }

    summary_path = os.path.join(args.output_dir, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Training summary saved: {summary_path}")

    logging.info("\n" + "="*70)
    logging.info("✅ Training completed successfully!")
    logging.info("="*70)


if __name__ == '__main__':
    main()
