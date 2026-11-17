"""
Visualize sample predictions for thesis
Shows input image, ground truth mask, and predicted mask side-by-side
"""
import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize model predictions')
parser.add_argument('--model_path', type=str, default='outputs/best_model.pth',
                    help='Path to model checkpoint')
parser.add_argument('--data_dir', type=str, default='processed_data',
                    help='Path to processed data directory')
parser.add_argument('--output_dir', type=str, default='outputs/visualizations/predictions',
                    help='Output directory for visualizations')
parser.add_argument('--architecture', type=str, default='DeepLabV3',
                    choices=['DeepLabV3', 'Unet', 'FPN', 'UnetPlusPlus'],
                    help='Model architecture')
parser.add_argument('--encoder', type=str, default='resnet34',
                    help='Encoder backbone')
parser.add_argument('--num_samples', type=int, default=6,
                    help='Number of samples to visualize')
args = parser.parse_args()

# Configuration
MODEL_PATH = args.model_path
DATA_DIR = args.data_dir
OUTPUT_DIR = args.output_dir
NUM_SAMPLES = args.num_samples

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load model
print(f"Loading best model ({args.architecture} + {args.encoder})...")
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Initialize model based on architecture
if args.architecture == 'DeepLabV3':
    model = smp.DeepLabV3(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
elif args.architecture == 'Unet':
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
elif args.architecture == 'FPN':
    model = smp.FPN(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
elif args.architecture == 'UnetPlusPlus':
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print("Model loaded successfully")

# Transform for inference
transform = A.Compose([
    A.Resize(512, 512),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# Get all images and apply same split as training
from sklearn.model_selection import train_test_split

# Determine mask directory based on model type
# Vegetation models use vgg16 encoder, power line models use resnet/efficientnet
if args.encoder == 'vgg16' or 'veg' in args.model_path.lower():
    mask_subdir = 'veg_masks'
else:
    mask_subdir = 'pl_masks'

all_images = sorted(Path(f'{DATA_DIR}/images').glob('*.png'))
all_masks = sorted(Path(f'{DATA_DIR}/{mask_subdir}').glob('*.png'))
print(f"Using mask directory: {mask_subdir}")

# Apply same split as training (train=60%, val=20%, test=20%, seed=42)
train_split = 0.6
val_split = 0.2
random_seed = 42
test_size = 1.0 - train_split - val_split

# Split: train+val vs test
train_val_imgs, test_imgs, train_val_msks, test_msks = train_test_split(
    all_images, all_masks,
    test_size=test_size,
    random_state=random_seed
)

print(f"Total images: {len(all_images)}")
print(f"Test set size: {len(test_imgs)}")

# Select random samples from TEST SET ONLY
np.random.seed(42)
indices = np.random.choice(len(test_imgs), min(NUM_SAMPLES, len(test_imgs)), replace=False)

print(f"\nGenerating predictions for {len(indices)} samples...")

# Process each sample
for idx, i in enumerate(indices):
    # Load image and mask
    img_path = test_imgs[i]
    msk_path = test_msks[i]

    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)

    # Store original for visualization
    orig_img = image.copy()
    orig_mask = mask.copy()

    # Apply transform
    transformed = transform(image=image, mask=mask)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = (pred > 0.5).float()

    # Convert to numpy
    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_prob = pred.squeeze().cpu().numpy()

    # Resize predictions back to original size
    h, w = orig_mask.shape
    pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_prob_resized = cv2.resize(pred_prob, (w, h), interpolation=cv2.INTER_LINEAR)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Image, Ground Truth, Prediction
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('Input Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(orig_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(pred_mask_resized, cmap='gray')
    axes[0, 2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')

    # Row 2: Overlay comparisons
    # Ground truth overlay
    overlay_gt = orig_img.copy()
    overlay_gt[orig_mask > 0] = [0, 255, 0]  # Green for GT
    axes[1, 0].imshow(cv2.addWeighted(orig_img, 0.6, overlay_gt, 0.4, 0))
    axes[1, 0].set_title('Ground Truth Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Prediction overlay
    overlay_pred = orig_img.copy()
    overlay_pred[pred_mask_resized > 0.5] = [255, 0, 0]  # Red for prediction
    axes[1, 1].imshow(cv2.addWeighted(orig_img, 0.6, overlay_pred, 0.4, 0))
    axes[1, 1].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')

    # Probability heatmap
    im = axes[1, 2].imshow(pred_prob_resized, cmap='jet', vmin=0, vmax=1)
    axes[1, 2].set_title('Prediction Probability', fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Calculate metrics for this sample
    intersection = np.logical_and(orig_mask > 0, pred_mask_resized > 0.5).sum()
    union = np.logical_or(orig_mask > 0, pred_mask_resized > 0.5).sum()
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (orig_mask.sum() + pred_mask_resized.sum() + 1e-8)

    plt.suptitle(f'Sample {idx+1} - IoU: {iou:.3f}, Dice: {dice:.3f}',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()

    # Save
    output_path = f'{OUTPUT_DIR}/prediction_sample_{idx+1}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path} (IoU: {iou:.3f}, Dice: {dice:.3f})")

# Create comparison grid
print("\nCreating comparison grid...")
fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, NUM_SAMPLES * 3))

if NUM_SAMPLES == 1:
    axes = axes.reshape(1, -1)

for idx, i in enumerate(indices):
    # Load image and mask
    image = cv2.imread(str(test_imgs[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(test_msks[i]), cv2.IMREAD_GRAYSCALE)

    # Transform and predict
    transformed = transform(image=image, mask=mask)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = (pred > 0.5).float().squeeze().cpu().numpy()

    # Resize
    h, w = mask.shape
    pred_mask_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Plot
    axes[idx, 0].imshow(image)
    axes[idx, 0].axis('off')
    if idx == 0:
        axes[idx, 0].set_title('Input', fontsize=11, fontweight='bold')

    axes[idx, 1].imshow(mask, cmap='gray')
    axes[idx, 1].axis('off')
    if idx == 0:
        axes[idx, 1].set_title('Ground Truth', fontsize=11, fontweight='bold')

    axes[idx, 2].imshow(pred_mask_resized, cmap='gray')
    axes[idx, 2].axis('off')
    if idx == 0:
        axes[idx, 2].set_title('Prediction', fontsize=11, fontweight='bold')

task_name = "Vegetation" if "veg" in args.model_path.lower() or args.encoder == "vgg16" else "Power Line"
plt.suptitle(f'{task_name} Segmentation Results', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison_grid.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✅ All prediction visualizations saved to {OUTPUT_DIR}/")
print("\nGenerated files:")
print(f"  - {NUM_SAMPLES} individual prediction samples")
print("  - 1 comparison grid")
