"""
Model Comparison Visualization Script
======================================
Compare predictions from 4 different vegetation segmentation models:
1. Multispectral VEPL-Net (U-Net + VGG16, 12 channels)
2. VGG from Scratch (U-Net + VGG16 scratch, 9 channels)
3. Multispectral ViT (U-Net + MiT-B2, 9 channels)
4. RGB-only VEPL-Net (U-Net + VGG16, 3 channels)

Usage:
    python compare_models_visualization.py --data_dir ./data/ml_ready --sample_indices 0 1
"""

import argparse
import json
from pathlib import Path

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser(description='Compare Model Predictions')
    parser.add_argument('--data_dir', type=str, default='data/ml_ready',
                        help='Path to ml_ready dataset directory')
    parser.add_argument('--output_dir', type=str, default='model_comparison',
                        help='Directory to save visualization')
    parser.add_argument('--sample_indices', type=int, nargs='+', default=[0, 1, 2],
                        help='Indices of test samples to visualize')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    return parser.parse_args()


def load_data_for_inference(tile_name, data_dir, use_multispectral=True, use_indices=True, use_all_indices=True):
    """
    Load data for a single tile with specific channel configuration.

    Args:
        use_multispectral: Include 4 spectral bands
        use_indices: Whether to include any indices at all
        use_all_indices: If True, use all 5 indices; if False, use only NDVI and GNDVI
    """
    data_dir = Path(data_dir)

    # Load RGB image
    rgb_path = data_dir / 'rgb_tiles' / f"{tile_name}.png"
    rgb_image = cv2.imread(str(rgb_path))
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

    # Load VEGETATION mask
    mask_path = data_dir / 'mask_tiles' / f"{tile_name}_vegetation.npy"
    mask = np.load(mask_path).astype(np.float32)

    # Initialize combined image with RGB
    combined_image = rgb_image.copy()

    # Load and concatenate multispectral bands
    if use_multispectral:
        bands = ['green', 'red', 'rededge', 'nir']
        band_data = []
        for band in bands:
            band_path = data_dir / 'band_tiles' / f"{tile_name}_{band}.npy"
            band_array = np.load(band_path)

            # Handle NaN and inf values
            band_array = np.nan_to_num(band_array, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize band to 0-255 range
            band_min = band_array.min()
            band_max = band_array.max()

            if band_max - band_min > 1e-8:
                band_normalized = ((band_array - band_min) / (band_max - band_min) * 255).astype(np.uint8)
            else:
                band_normalized = np.full_like(band_array, 128, dtype=np.uint8)

            band_data.append(band_normalized)

        bands_stacked = np.stack(band_data, axis=-1)
        combined_image = np.concatenate([combined_image, bands_stacked], axis=-1)

    # Load and concatenate vegetation indices (only if use_indices is True)
    if use_indices:
        if use_all_indices:
            indices = ['ndvi', 'gndvi', 'ndre', 'lci', 'osavi']
        else:
            indices = ['ndvi', 'gndvi']

        index_data = []
        for index in indices:
            index_path = data_dir / 'index_tiles' / f"{tile_name}_{index}.npy"
            index_array = np.load(index_path)

            # Handle NaN and inf values
            index_array = np.nan_to_num(index_array, nan=0.0, posinf=1.0, neginf=-1.0)
            index_array = np.clip(index_array, -1, 1)

            # Normalize to 0-255
            index_normalized = ((index_array + 1) / 2 * 255).astype(np.uint8)
            index_data.append(index_normalized)

        if index_data:
            indices_stacked = np.stack(index_data, axis=-1)
            combined_image = np.concatenate([combined_image, indices_stacked], axis=-1)

    return combined_image, mask, rgb_image


def preprocess_image(image, input_channels):
    """Preprocess image for model input"""
    transform = A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(mean=[0.485]*input_channels, std=[0.229]*input_channels, max_pixel_value=255.0),
        ToTensorV2(),
    ])

    augmented = transform(image=image)
    return augmented['image']


def load_model(model_path, architecture, encoder, input_channels, device):
    """Load a trained model from checkpoint"""
    print(f"  Loading model from {model_path}")
    print(f"    Architecture: {architecture}, Encoder: {encoder}, Channels: {input_channels}")

    # Create model
    if architecture == 'Unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=input_channels,
            classes=1,
            activation=None
        )
    elif architecture == 'DeepLabV3':
        model = smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=input_channels,
            classes=1,
            activation=None
        )

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"    ✓ Loaded (Val IoU: {checkpoint.get('val_iou', 0):.4f})")

    return model


def calculate_iou(pred, target):
    """Calculate IoU score"""
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def run_inference(model, image_tensor, device):
    """Run inference and return binary prediction"""
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        output = model(image_tensor)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
        pred_binary = (pred > 0.5).astype(np.float32)

    return pred_binary


def create_error_overlay(prediction, ground_truth):
    """
    Create a color overlay showing:
    - True Positives (TP) in green
    - False Positives (FP) in blue
    - False Negatives (FN) in red
    - True Negatives (TN) in black

    Returns RGB image
    """
    h, w = prediction.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert to binary
    pred_binary = (prediction > 0.5).astype(bool)
    gt_binary = (ground_truth > 0.5).astype(bool)

    # True Positives - Green (correctly predicted vegetation)
    tp_mask = pred_binary & gt_binary
    overlay[tp_mask] = [0, 255, 0]  # Green

    # False Positives - Blue (incorrectly predicted as vegetation)
    fp_mask = pred_binary & ~gt_binary
    overlay[fp_mask] = [0, 0, 255]  # Blue

    # False Negatives - Red (missed vegetation)
    fn_mask = ~pred_binary & gt_binary
    overlay[fn_mask] = [255, 0, 0]  # Red

    # True Negatives - Black (correctly predicted non-vegetation)
    # Already black by default

    return overlay


def visualize_comparison(samples_data, output_path=None):
    """
    Visualize model comparison with color-coded error overlays.

    samples_data: List of dicts with keys:
        - tile_name
        - rgb
        - ground_truth
        - predictions: dict of {model_name: (prediction, iou)}
    """
    num_samples = len(samples_data)
    num_cols = 6  # RGB, GT, Pred1, Pred2, Pred3, Pred4

    # Larger figure size for bigger plots
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(36, 6*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)

    model_names = [
        'Multispectral\nVEPL-Net',
        'VGG from\nScratch',
        'Multispectral\nViT',
        'RGB-only\nVEPL-Net'
    ]

    for row_idx, sample_data in enumerate(samples_data):
        # Column 0: RGB
        axes[row_idx, 0].imshow(sample_data['rgb'])
        if row_idx == 0:
            axes[row_idx, 0].set_title('RGB Image', fontsize=36, weight='bold')
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_ylabel(f'Sample {row_idx+1}', fontsize=40, rotation=0,
                                      labelpad=50, va='center', weight='bold')

        # Column 1: Ground Truth
        axes[row_idx, 1].imshow(sample_data['ground_truth'], cmap='gray', vmin=0, vmax=1)
        if row_idx == 0:
            axes[row_idx, 1].set_title('Ground Truth', fontsize=36, weight='bold')
        axes[row_idx, 1].axis('off')

        # Columns 2-5: Model Predictions with color-coded error overlays
        for col_idx, (model_name, model_key) in enumerate(zip(model_names, sample_data['predictions'].keys())):
            pred, iou = sample_data['predictions'][model_key]

            # Create error overlay
            error_overlay = create_error_overlay(pred, sample_data['ground_truth'])
            axes[row_idx, col_idx+2].imshow(error_overlay)

            if row_idx == 0:
                title = f'{model_name}'
                axes[row_idx, col_idx+2].set_title(title, fontsize=36, weight='bold')

            # Add IoU text on the image
            axes[row_idx, col_idx+2].text(0.5, 0.95, f'IoU: {iou:.3f}',
                                          transform=axes[row_idx, col_idx+2].transAxes,
                                          fontsize=28, weight='bold',
                                          ha='center', va='top',
                                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            axes[row_idx, col_idx+2].axis('off')

    # Add legend for the color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive (Correct Veg)'),
        Patch(facecolor='blue', label='False Positive (Incorrect Veg)'),
        Patch(facecolor='red', label='False Negative (Missed Veg)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               fontsize=28, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle('Vegetation Segmentation Model Comparison', fontsize=42, y=0.99, weight='bold')

    # Reduce spacing between columns and rows for tighter layout
    plt.subplots_adjust(hspace=0.05, wspace=0.02)
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])

    # if output_path:
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #     print(f"\n✓ Visualization saved to: {output_path} (300 DPI)")
    if output_path:
        # Use .pdf for the best quality in LaTeX
        # If you must use .png, dpi=600 is better for printing
        plt.savefig(
            output_path, 
            dpi=600,                # High resolution for raster formats
            bbox_inches='tight',    # Removes unnecessary white margins
            pad_inches=0.1,         # Small padding around the plot
            transparent=False,      # Better for white paper theses
            facecolor='w'           # Ensures white background
    )
    print(f"\n✓ Visualization saved to: {output_path}")

    plt.show()


def main():
    args = parse_args()

    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print("="*80)
    print("MODEL COMPARISON VISUALIZATION")
    print("="*80)
    print(f"Device: {device}\n")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and get test split
    metadata_path = data_dir / 'metadata' / 'tile_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Perform same split as training
    tiles_with_pl = [tile['tile_name'] for tile in metadata['tiles'] if tile.get('has_power_line', False)]
    tiles_without_pl = [tile['tile_name'] for tile in metadata['tiles'] if not tile.get('has_power_line', False)]

    pl_train_val, pl_test = train_test_split(tiles_with_pl, test_size=0.2, random_state=42)
    no_pl_train_val, no_pl_test = train_test_split(tiles_without_pl, test_size=0.2, random_state=42)

    test_tiles = pl_test + no_pl_test

    print(f"Total test tiles: {len(test_tiles)}")

    # Get samples to visualize
    selected_tiles = [test_tiles[idx] for idx in args.sample_indices]

    print(f"\nSelected samples:")
    for i, idx in enumerate(args.sample_indices):
        print(f"  Sample {i+1} (index {idx}): {selected_tiles[i]}")

    # Define model configurations
    models_config = {
        'multispectral_vepl': {
            'name': 'Multispectral VEPL-Net',
            'path': 'outputs_veg/best_model.pth',
            'architecture': 'Unet',
            'encoder': 'vgg16',
            'input_channels': 12,
            'use_multispectral': True,
            'use_indices': True,
            'use_all_indices': True
        },
        'vgg_scratch': {
            'name': 'VGG from Scratch',
            'path': 'outputs_multispectral_veg_vgg_scratch/best_model.pth',
            'architecture': 'Unet',
            'encoder': 'vgg16',
            'input_channels': 9,
            'use_multispectral': True,
            'use_indices': True,
            'use_all_indices': False
        },
        'multispectral_vit': {
            'name': 'Multispectral ViT',
            'path': 'outputs_multispectral_veg_vit_simple/best_model.pth',
            'architecture': 'Unet',
            'encoder': 'mit_b2',
            'input_channels': 9,
            'use_multispectral': True,
            'use_indices': True,
            'use_all_indices': False
        },
        'rgb_only': {
            'name': 'RGB-only VEPL-Net',
            'path': 'outputs_rgb_veg/best_model.pth',
            'architecture': 'Unet',
            'encoder': 'vgg16',
            'input_channels': 3,
            'use_multispectral': False,
            'use_indices': False,
            'use_all_indices': False
        }
    }

    # Load all models
    print("\nLoading models...")
    print("-"*80)

    loaded_models = {}
    for model_key, config in models_config.items():
        model_path = Path(config['path'])
        if not model_path.exists():
            print(f"  ⚠ Model not found: {model_path}")
            continue

        loaded_models[model_key] = {
            'model': load_model(model_path, config['architecture'], config['encoder'],
                               config['input_channels'], device),
            'config': config
        }

    print(f"\n✓ Loaded {len(loaded_models)}/{len(models_config)} models")

    # Process each sample
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)

    samples_data = []

    for tile_name in selected_tiles:
        print(f"\nProcessing: {tile_name}")

        sample_predictions = {}

        for model_key, model_data in loaded_models.items():
            config = model_data['config']
            model = model_data['model']

            # Load data with appropriate configuration
            combined_image, mask, rgb_image = load_data_for_inference(
                tile_name, data_dir,
                use_multispectral=config['use_multispectral'],
                use_indices=config['use_indices'],
                use_all_indices=config['use_all_indices']
            )

            # Preprocess
            image_tensor = preprocess_image(combined_image, config['input_channels'])

            # Run inference
            prediction = run_inference(model, image_tensor, device)

            # Calculate IoU
            iou = calculate_iou(prediction, mask)

            print(f"  {config['name']}: IoU = {iou:.4f}")

            sample_predictions[model_key] = (prediction, iou)

        samples_data.append({
            'tile_name': tile_name,
            'rgb': rgb_image,
            'ground_truth': mask,
            'predictions': sample_predictions
        })

    # Create visualization
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)

    viz_output_path = output_dir / 'model_comparison.png'
    visualize_comparison(samples_data, output_path=viz_output_path)

    # Calculate average IoU per model
    print("\n" + "="*80)
    print("AVERAGE IoU ACROSS SAMPLES")
    print("="*80)

    for model_key, model_data in loaded_models.items():
        config = model_data['config']
        ious = [sample['predictions'][model_key][1] for sample in samples_data]
        avg_iou = np.mean(ious)
        print(f"{config['name']}: {avg_iou:.4f}")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
