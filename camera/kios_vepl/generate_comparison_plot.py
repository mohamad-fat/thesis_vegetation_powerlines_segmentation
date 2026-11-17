"""
Generate comparison plot for power line detection models.

This script creates a visualization comparing VEPL-Net and Vision Transformer
predictions on test set samples.

Layout: 3 rows (3 samples) × 3 columns (Ground Truth, VEPL-Net, ViT)

Usage:
    python generate_comparison_plot.py tile_000051 tile_000123 tile_000456

Or edit the SAMPLE_NAMES list in the script directly.
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths
DATA_DIR = './data/processed_data'
VEPL_NET_MODEL = './output_pl/best_model.pth'
VIT_MODEL = './outputs_pl_vit/best_model_pl_vit.pth'
OUTPUT_PATH = './power_line_comparison_plot.png'

# Specify sample names to visualize (without .png extension)
# You can modify this list or pass them as command line arguments
SAMPLE_NAMES = ['tile_001120', 'tile_001737', 'tile_000429']  # Change these to your desired samples

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_test_indices():
    """Get test set indices using the same split strategy as training."""
    img_dir = os.path.join(DATA_DIR, 'images')
    mask_dir = os.path.join(DATA_DIR, 'pl_masks')

    all_image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
    all_mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png')])

    # Same split as training: 60% train + 20% val = 80% train_val, 20% test
    # First split: separate out the test set (last 20%)
    train_val_imgs, test_imgs, train_val_msks, test_msks = train_test_split(
        all_image_paths, all_mask_paths,
        test_size=0.2,  # 20% for test (last 20%)
        random_state=RANDOM_SEED
    )

    # The second split in training scripts separates train_val into train and val
    # but we only need the test set here
    return test_imgs, test_msks


def get_transform_vepl():
    """Transform for VEPL-Net model (256x256)."""
    return A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def get_transform_vit():
    """Transform for ViT model (512x512)."""
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def load_vepl_net_model():
    """Load VEPL-Net model (DeepLabV3 + ResNet34)."""
    print("Loading VEPL-Net model...")
    model = smp.DeepLabV3(
        encoder_name='resnet34',
        encoder_weights=None,  # Load from checkpoint
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(VEPL_NET_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"VEPL-Net loaded from epoch {checkpoint['epoch']+1}")
    print(f"Best Val IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")

    return model


def load_vit_model():
    """Load Vision Transformer model (U-Net + MiT-B2)."""
    print("\nLoading Vision Transformer model...")
    model = smp.Unet(
        encoder_name='mit_b2',
        encoder_weights=None,  # Load from checkpoint
        in_channels=3,
        classes=1,
    )

    checkpoint = torch.load(VIT_MODEL, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"ViT loaded from epoch {checkpoint['epoch']+1}")
    print(f"Best Val IoU: {checkpoint.get('best_iou', 'N/A'):.4f}")

    return model


def predict(model, image, transform, threshold=0.5):
    """Generate prediction from a model."""
    # Apply transform
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
        pred_binary = (pred > threshold).float()

    # Convert to numpy
    pred_np = pred_binary.squeeze().cpu().numpy()

    return pred_np


def create_overlay(image, mask, prediction, show_ground_truth_only=False):
    """
    Create visualization overlay.
    - Green: True Positives (correct detections)
    - Red: False Negatives (missed detections)
    - Blue: False Positives (incorrect detections)

    If show_ground_truth_only=True, only shows the ground truth without color coding.
    """
    # Ensure image is RGB and in [0, 1] range
    if image.max() > 1.0:
        image = image / 255.0

    overlay = image.copy()

    # Resize prediction to match original mask size
    if prediction.shape != mask.shape:
        prediction = cv2.resize(prediction, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    if show_ground_truth_only:
        # For ground truth, just show the mask in green without any comparison
        gt_mask = mask == 1
        overlay[gt_mask] = overlay[gt_mask] * 0.5 + np.array([0, 1, 0]) * 0.5
    else:
        # True Positives - Green
        tp_mask = (mask == 1) & (prediction == 1)
        overlay[tp_mask] = overlay[tp_mask] * 0.5 + np.array([0, 1, 0]) * 0.5

        # False Negatives - Red (missed power lines)
        fn_mask = (mask == 1) & (prediction == 0)
        overlay[fn_mask] = overlay[fn_mask] * 0.5 + np.array([1, 0, 0]) * 0.5

        # False Positives - Blue (incorrect detections)
        fp_mask = (mask == 0) & (prediction == 1)
        overlay[fp_mask] = overlay[fp_mask] * 0.5 + np.array([0, 0, 1]) * 0.5

    return overlay


def main():
    print("="*70)
    print("Power Line Detection Comparison Plot Generator")
    print("="*70)

    # Check for command line arguments
    sample_names = SAMPLE_NAMES
    if len(sys.argv) > 1:
        sample_names = sys.argv[1:]
        print(f"\nUsing samples from command line: {sample_names}")
    else:
        print(f"\nUsing samples from script: {sample_names}")

    # Validate we have at least 3 samples
    if len(sample_names) < 3:
        print("Error: Please specify at least 3 sample names")
        print("Usage: python generate_comparison_plot.py tile_000051 tile_000123 tile_000456")
        sys.exit(1)

    # We'll only use the first 3 samples for the 3-row layout
    sample_names = sample_names[:3]

    # Get test set
    test_imgs, test_msks = get_test_indices()
    print(f"\nTotal test samples: {len(test_imgs)}")

    # Save all test images info to a data folder
    test_data_dir = './test_set_data'
    os.makedirs(test_data_dir, exist_ok=True)

    print(f"\nSaving test set information to: {test_data_dir}")

    # Save list of test images
    with open(os.path.join(test_data_dir, 'test_images_list.txt'), 'w') as f:
        for i, (img_path, mask_path) in enumerate(zip(test_imgs, test_msks)):
            f.write(f"{i+1},{os.path.basename(img_path)},{os.path.basename(mask_path)}\n")

    print(f"  - Saved test_images_list.txt ({len(test_imgs)} images)")

    # Copy test images and masks to the data folder
    import shutil
    test_imgs_out = os.path.join(test_data_dir, 'images')
    test_masks_out = os.path.join(test_data_dir, 'masks')
    os.makedirs(test_imgs_out, exist_ok=True)
    os.makedirs(test_masks_out, exist_ok=True)

    print(f"  - Copying {len(test_imgs)} test images and masks...")
    for img_path, mask_path in zip(test_imgs, test_msks):
        shutil.copy2(img_path, test_imgs_out)
        shutil.copy2(mask_path, test_masks_out)

    print(f"  - Test images saved to: {test_imgs_out}")
    print(f"  - Test masks saved to: {test_masks_out}")

    # Find the specified samples in the test set
    print(f"\nSearching for specified samples in test set...")
    sample_indices = []
    sample_paths = []

    for sample_name in sample_names:
        found = False
        for idx, (img_path, mask_path) in enumerate(zip(test_imgs, test_msks)):
            img_basename = os.path.basename(img_path)
            # Match with or without .png extension
            if sample_name in img_basename or f"{sample_name}.png" == img_basename:
                sample_indices.append(idx)
                sample_paths.append((img_path, mask_path))
                print(f"  ✓ Found '{sample_name}': {img_basename}")
                found = True
                break

        if not found:
            print(f"  ✗ ERROR: Sample '{sample_name}' not found in test set!")
            print(f"\nAvailable test samples:")
            for i, img_path in enumerate(test_imgs[:10]):
                print(f"    {i+1}. {os.path.basename(img_path)}")
            print(f"    ... and {len(test_imgs)-10} more")
            print(f"\nCheck test_set_data/test_images_list.txt for full list")
            sys.exit(1)

    print(f"\nUsing test indices: {sample_indices}")

    # Load models
    vepl_net_model = load_vepl_net_model()
    vit_model = load_vit_model()

    # Get transforms
    transform_vepl = get_transform_vepl()
    transform_vit = get_transform_vit()

    # Create figure (3 rows, 3 columns)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    print("\n" + "="*70)
    print("Generating predictions...")
    print("="*70)

    for row_idx, (sample_idx, (img_path, mask_path)) in enumerate(zip(sample_indices, sample_paths)):
        print(f"\nProcessing sample {row_idx+1}/3: {os.path.basename(img_path)}")
        print(f"  Test index: {sample_idx+1}")
        print(f"  Image: {os.path.basename(img_path)}")
        print(f"  Mask:  {os.path.basename(mask_path)}")

        # Load image and mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask > 0).astype(np.float32)

        # Generate predictions
        print("  Generating VEPL-Net prediction...")
        vepl_pred = predict(vepl_net_model, image, transform_vepl, threshold=0.5)

        print("  Generating ViT prediction...")
        vit_pred = predict(vit_model, image, transform_vit, threshold=0.7)  # Optimized threshold

        # Create overlays
        print("  Creating overlays...")
        vepl_overlay = create_overlay(image.copy(), mask_binary, vepl_pred)
        vit_overlay = create_overlay(image.copy(), mask_binary, vit_pred)

        # Plot Ground Truth (just the original image)
        axes[row_idx, 0].imshow(image)
        axes[row_idx, 0].set_title('Ground Truth', fontsize=14, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Plot VEPL-Net
        axes[row_idx, 1].imshow(vepl_overlay)
        axes[row_idx, 1].set_title('VEPL-Net\n(DeepLabV3 + ResNet34)', fontsize=14, fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Plot ViT
        axes[row_idx, 2].imshow(vit_overlay)
        axes[row_idx, 2].set_title('Vision Transformer\n(U-Net + MiT-B2)', fontsize=14, fontweight='bold')
        axes[row_idx, 2].axis('off')

        # Add row label with sample number
        row_label = f"Sample {row_idx + 1}"
        # Adjust vertical position for 3 rows: start at 0.83, space by 0.33
        y_position = 0.83 - row_idx * 0.33
        fig.text(0.02, y_position, row_label,
                fontsize=12, fontweight='bold', rotation=90,
                va='center', ha='center')

    # Add title
    fig.suptitle('Power Line Segmentation: VEPL-Net and ViT',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.5, label='True Positives'),
        Patch(facecolor='red', alpha=0.5, label='False Negatives'),
        Patch(facecolor='blue', alpha=0.5, label='False Positives')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
              fontsize=12, frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.5, -0.01))

    # Adjust layout
    plt.tight_layout(rect=[0.03, 0.02, 1, 0.99])

    # Save figure
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print("\n" + "="*70)
    print(f"Plot saved to: {OUTPUT_PATH}")
    print("="*70)

    # Also save a high-resolution version
    high_res_path = OUTPUT_PATH.replace('.png', '_high_res.png')
    plt.savefig(high_res_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"High-resolution version saved to: {high_res_path}")

    plt.close()

    print("\n" + "="*70)
    print("Caption for thesis:")
    print("="*70)
    print("""
Sample power line detection predictions comparing VEPL-Net baseline
(DeepLabV3 + ResNet34) with Vision Transformer (U-Net + MiT-B2). Green overlay
indicates true positives, red indicates false negatives, and blue indicates
false positives.
    """)

    print("\nDone!")


if __name__ == '__main__':
    main()
