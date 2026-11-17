"""
Re-evaluate Power Line Base Model (ResNet34)
"""

import os
import torch
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Dataset class
class KIOSPowerLineDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        binary_mask = (mask > 0).astype(np.float32)
        binary_mask = np.expand_dims(binary_mask, axis=-1)

        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']
            binary_mask = binary_mask.permute(2, 0, 1)

        return image, binary_mask


def get_val_transform():
    """Validation transform - same as base model (256x256)."""
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def evaluate(loader, model, device, threshold=0.5):
    """Evaluate model on validation/test set."""
    model.eval()
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_score = 0
    valid_batches = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if x.size(0) < 1:
                continue

            preds = torch.sigmoid(model(x))
            preds_binary = (preds > threshold).float()

            num_correct += (preds_binary == y).sum()
            num_pixels += torch.numel(preds_binary)

            tp, fp, fn, tn = smp.metrics.get_stats(preds_binary.long(), y.long(), mode='binary')
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

    dice = dice_score / len(loader)
    acc = num_correct / num_pixels * 100

    return iou.item(), dice.item(), acc.item()


def main():
    print("="*70)
    print("Re-evaluating Power Line Base Model (ResNet34 + DeepLabV3)")
    print("="*70)

    # Configuration - matches base model training
    data_dir = '../data/processed_data'
    checkpoint_path = '../output_pl/best_model.pth'
    batch_size = 8
    random_seed = 42

    # Device (with MPS support for Mac)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}\n")

    # Load data
    img_dir = os.path.join(data_dir, 'images')
    mask_dir = os.path.join(data_dir, 'pl_masks')

    all_image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    all_mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    print(f"Total samples: {len(all_image_paths)}")

    # Data split (same as training)
    train_split = 0.6
    val_split = 0.2
    test_size = 1.0 - train_split - val_split

    train_val_imgs, test_imgs, train_val_msks, test_msks = train_test_split(
        all_image_paths, all_mask_paths,
        test_size=test_size,
        random_state=random_seed
    )

    val_ratio = val_split / (train_split + val_split)
    _, val_imgs, _, val_msks = train_test_split(
        train_val_imgs, train_val_msks,
        test_size=val_ratio,
        random_state=random_seed
    )

    print(f"Validation samples: {len(val_imgs)}")
    print(f"Test samples: {len(test_imgs)}\n")

    # Create data loaders
    val_loader = DataLoader(
        KIOSPowerLineDataset(val_imgs, val_msks, transform=get_val_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    test_loader = DataLoader(
        KIOSPowerLineDataset(test_imgs, test_msks, transform=get_val_transform()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # Load model - DeepLabV3 + ResNet34 (same as base model)
    print("Loading model...")
    model = smp.DeepLabV3(
        encoder_name='resnet34',
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    print(f"Architecture: DeepLabV3 + ResNet34")
    print(f"Image size: 256x256\n")

    # Evaluate with different thresholds
    print("="*70)
    print("Evaluating with different thresholds:")
    print("="*70)

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        print(f"\nThreshold: {threshold}")
        print("-"*40)

        # Validation
        val_iou, val_dice, val_acc = evaluate(val_loader, model, device, threshold)
        print(f"Val  IoU: {val_iou:.4f} | Dice: {val_dice:.4f} | Acc: {val_acc:.2f}%")

        # Test
        test_iou, test_dice, test_acc = evaluate(test_loader, model, device, threshold)
        print(f"Test IoU: {test_iou:.4f} | Dice: {test_dice:.4f} | Acc: {test_acc:.2f}%")

    print("\n" + "="*70)
    print("Re-evaluation complete!")
    print("="*70)


if __name__ == '__main__':
    main()
