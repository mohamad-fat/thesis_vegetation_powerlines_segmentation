import os
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import re

def group_multispectral_images(data_dir):
    """
    Group multispectral images by capture timestamp.
    Returns a dictionary where keys are timestamps and values are dicts of band paths.
    """
    files = os.listdir(data_dir)

    # Pattern to extract timestamp and band type
    pattern = r'DJI_(\d+)_(\d+)_MS_([A-Z]+)\.TIF'
    visual_pattern = r'DJI_(\d+)_(\d+)_([DF])\.JPG'

    captures = {}

    for file in files:
        # Match multispectral bands
        match = re.match(pattern, file)
        if match:
            timestamp = match.group(1)
            sequence = match.group(2)
            band = match.group(3)

            key = f"{timestamp}_{sequence}"
            if key not in captures:
                captures[key] = {
                    'timestamp': timestamp,
                    'sequence': sequence,
                    'bands': {},
                    'visual': {}
                }

            captures[key]['bands'][band] = os.path.join(data_dir, file)

        # Match visual RGB images
        visual_match = re.match(visual_pattern, file)
        if visual_match:
            timestamp = visual_match.group(1)
            sequence = visual_match.group(2)
            camera_type = visual_match.group(3)

            key = f"{timestamp}_{sequence}"
            if key not in captures:
                captures[key] = {
                    'timestamp': timestamp,
                    'sequence': sequence,
                    'bands': {},
                    'visual': {}
                }

            captures[key]['visual'][camera_type] = os.path.join(data_dir, file)

    return captures

def load_multispectral_stack(capture_info, bands=['R', 'G', 'NIR', 'RE']):
    """
    Load and stack multispectral bands into a single numpy array.
    Returns: (height, width, num_bands) array
    """
    band_arrays = []

    for band in bands:
        if band in capture_info['bands']:
            img = cv2.imread(capture_info['bands'][band], cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Warning: Could not load {band} band")
                continue
            band_arrays.append(img)

    if not band_arrays:
        return None

    # Stack along channel dimension
    stacked = np.stack(band_arrays, axis=-1)
    return stacked

def create_false_color_composite(capture_info, composite_type='CIR'):
    """
    Create false color composite for visualization.

    composite_type options:
    - 'CIR': Color Infrared (NIR-R-G) - vegetation appears red
    - 'NDVI': Normalized Difference Vegetation Index visualization
    - 'RGB': Natural color (R-G-B) if available
    """

    if composite_type == 'CIR':
        # Color Infrared: NIR as Red, Red as Green, Green as Blue
        nir = cv2.imread(capture_info['bands']['NIR'], cv2.IMREAD_UNCHANGED)
        r = cv2.imread(capture_info['bands']['R'], cv2.IMREAD_UNCHANGED)
        g = cv2.imread(capture_info['bands']['G'], cv2.IMREAD_UNCHANGED)

        composite = np.stack([nir, r, g], axis=-1)

    elif composite_type == 'NDVI':
        # Calculate NDVI and colorize
        nir = cv2.imread(capture_info['bands']['NIR'], cv2.IMREAD_UNCHANGED).astype(float)
        r = cv2.imread(capture_info['bands']['R'], cv2.IMREAD_UNCHANGED).astype(float)

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi = (nir - r) / (nir + r + 1e-8)

        # Normalize to 0-255
        ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)

        # Apply colormap (green for high vegetation)
        composite = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
        composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    elif composite_type == 'RGB':
        # Natural RGB if visual image exists
        if 'D' in capture_info['visual']:
            composite = cv2.imread(capture_info['visual']['D'])
            composite = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
        else:
            # Create from multispectral RGB
            r = cv2.imread(capture_info['bands']['R'], cv2.IMREAD_UNCHANGED)
            g = cv2.imread(capture_info['bands']['G'], cv2.IMREAD_UNCHANGED)
            # Note: No Blue band in multispectral, use Red Edge as substitute
            re = cv2.imread(capture_info['bands']['RE'], cv2.IMREAD_UNCHANGED)
            composite = np.stack([r, g, re], axis=-1)

    # Normalize to 8-bit if needed
    if composite.dtype != np.uint8:
        composite = cv2.normalize(composite, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return composite

def visualize_capture(capture_info, save_path=None):
    """
    Visualize all bands and composites for a single capture.
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Position {capture_info['sequence']} - Timestamp: {capture_info['timestamp']}", fontsize=16, fontweight='bold')

    # Individual bands - TOP ROW
    band_names = ['G', 'R', 'RE', 'NIR']
    band_labels = {
        'G': 'Green (520-590nm)',
        'R': 'Red (640-680nm)',
        'RE': 'Red Edge (730-740nm)',
        'NIR': 'Near Infrared (820-890nm)'
    }

    for idx, band in enumerate(band_names):
        if band in capture_info['bands']:
            img = cv2.imread(capture_info['bands'][band], cv2.IMREAD_UNCHANGED)
            axes[0, idx].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[0, idx].set_title(band_labels[band], fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')

            # Add image dimensions
            axes[0, idx].text(0.02, 0.98, f'{img.shape[1]}x{img.shape[0]}',
                            transform=axes[0, idx].transAxes,
                            fontsize=8, color='yellow',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

    # Composites - BOTTOM ROW
    composites = [
        ('RGB', 'Natural RGB'),
        ('CIR', 'Color Infrared (NIR-R-G)\nVegetation = Red'),
        ('NDVI', 'NDVI - Vegetation Index\nGreen = Healthy Vegetation'),
    ]

    for idx, (comp_type, title) in enumerate(composites):
        try:
            composite = create_false_color_composite(capture_info, comp_type)
            axes[1, idx].imshow(composite)
            axes[1, idx].set_title(title, fontsize=11, fontweight='bold')
            axes[1, idx].axis('off')
        except Exception as e:
            axes[1, idx].text(0.5, 0.5, f'Error: {str(e)}',
                            ha='center', va='center', transform=axes[1, idx].transAxes, fontsize=8)
            axes[1, idx].axis('off')

    # Visual image if available
    if 'D' in capture_info['visual']:
        visual = cv2.imread(capture_info['visual']['D'])
        visual = cv2.cvtColor(visual, cv2.COLOR_BGR2RGB)
        axes[1, 3].imshow(visual)
        axes[1, 3].set_title('Visual RGB Camera', fontsize=11, fontweight='bold')
        axes[1, 3].axis('off')
    else:
        axes[1, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.close(fig)

def visualize_all_positions(captures, max_positions=5, save_path=None):
    """
    Visualize multiple positions side by side to compare.
    """
    sorted_keys = sorted(captures.keys())[:max_positions]

    fig, axes = plt.subplots(max_positions, 5, figsize=(20, 4*max_positions))
    if max_positions == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Multispectral Band Comparison Across Positions', fontsize=16, fontweight='bold')

    for row_idx, key in enumerate(sorted_keys):
        capture = captures[key]

        # Column headers on first row
        if row_idx == 0:
            col_titles = ['Green', 'Red', 'Red Edge', 'NIR', 'Color Infrared']
            for col_idx, title in enumerate(col_titles):
                axes[row_idx, col_idx].set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Row label
        axes[row_idx, 0].text(-0.3, 0.5, f"Position\n{capture['sequence']}",
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=11, fontweight='bold',
                             verticalalignment='center', rotation=0)

        # Show individual bands
        band_names = ['G', 'R', 'RE', 'NIR']
        for col_idx, band in enumerate(band_names):
            if band in capture['bands']:
                img = cv2.imread(capture['bands'][band], cv2.IMREAD_UNCHANGED)
                axes[row_idx, col_idx].imshow(img, cmap='gray', vmin=0, vmax=255)
            axes[row_idx, col_idx].axis('off')

        # Show CIR composite
        try:
            cir = create_false_color_composite(capture, 'CIR')
            axes[row_idx, 4].imshow(cir)
        except:
            pass
        axes[row_idx, 4].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison visualization to {save_path}")

    plt.close(fig)

def save_multispectral_stack(capture_info, output_path, bands=['R', 'G', 'NIR', 'RE']):
    """
    Save stacked multispectral bands as a multi-channel TIFF.
    This format can be used for labeling in tools like QGIS or CVAT.
    """
    stacked = load_multispectral_stack(capture_info, bands)

    if stacked is None:
        print("Error: Could not create stack")
        return False

    # Save as multi-channel TIFF
    # PIL can handle multi-channel TIFF
    from PIL import Image

    # Convert to list of 2D arrays (one per band)
    band_list = [stacked[:, :, i] for i in range(stacked.shape[2])]

    # Save as multi-page TIFF or multi-channel
    img = Image.fromarray(stacked[:, :, 0])  # Start with first band

    # For multi-channel, we need to use a different approach
    # Save as NPY for now (easier for Python processing)
    np.save(output_path.replace('.tif', '.npy'), stacked)
    print(f"Saved multispectral stack to {output_path.replace('.tif', '.npy')}")

    # Also save a CIR composite for visual reference
    cir = create_false_color_composite(capture_info, 'CIR')
    cv2.imwrite(output_path.replace('.tif', '_CIR.png'), cv2.cvtColor(cir, cv2.COLOR_RGB2BGR))
    print(f"Saved CIR composite to {output_path.replace('.tif', '_CIR.png')}")

    return True

def main():
    # Set your data directory
    data_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/mavic_multispectral/DJI_202510021041_017_Geri-MS"

    # Group images by capture
    print("Grouping multispectral images...")
    captures = group_multispectral_images(data_dir)

    print(f"Found {len(captures)} positions in the flight")
    print("\nPosition summary:")
    for key, info in sorted(captures.items())[:5]:  # Show first 5
        bands = ', '.join(info['bands'].keys())
        visual = ', '.join(info['visual'].keys())
        print(f"  Position {info['sequence']}: Bands=[{bands}] Visual=[{visual}]")

    if not captures:
        print("No captures found!")
        return

    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS:")
    print("="*60)

    # Create output directory
    output_dir = os.path.join(data_dir, "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    # Option 1: Visualize single position with all bands
    print("\n1. Creating detailed view of first position (all 4 bands + composites)")
    first_key = sorted(captures.keys())[0]
    first_viz_path = os.path.join(output_dir, f"position_{captures[first_key]['sequence']}_detail.png")
    print(f"   Position: {captures[first_key]['sequence']}")
    visualize_capture(captures[first_key], save_path=first_viz_path)

    # Option 2: Compare multiple positions
    print("\n2. Creating comparison view across first 5 positions")
    comparison_path = os.path.join(output_dir, "positions_comparison.png")
    visualize_all_positions(captures, max_positions=5, save_path=comparison_path)

    # Save stacked versions
    print("\n3. Saving stacked multispectral data for first 3 positions...")
    stack_dir = os.path.join(data_dir, "stacked")
    os.makedirs(stack_dir, exist_ok=True)

    for key in sorted(captures.keys())[:3]:  # Save first 3
        output_path = os.path.join(stack_dir, f"{key}_stack.tif")
        save_multispectral_stack(captures[key], output_path)

    print(f"\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    print(f"\nVisualizations saved to: {output_dir}")
    print(f"Stacked data saved to: {stack_dir}")
    print(f"\nGenerated files:")
    print(f"  - {first_viz_path}")
    print(f"  - {comparison_path}")
    print(f"  - Check {stack_dir} for .npy and _CIR.png files")

if __name__ == "__main__":
    main()
