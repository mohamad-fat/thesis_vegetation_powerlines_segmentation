"""
Analyze Class Density in KIOS VEPL Dataset
===========================================
Calculate the density of vegetation and power line classes in the processed dataset.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

def analyze_class_density(data_dir):
    """
    Analyze class density for vegetation and power lines.

    Args:
        data_dir: Path to processed_data directory containing veg_masks and pl_masks
    """
    data_dir = Path(data_dir)
    veg_masks_dir = data_dir / 'veg_masks'
    pl_masks_dir = data_dir / 'pl_masks'

    # Get all mask files
    veg_mask_files = sorted(list(veg_masks_dir.glob('*.png')))
    pl_mask_files = sorted(list(pl_masks_dir.glob('*.png')))

    print(f"Found {len(veg_mask_files)} vegetation masks")
    print(f"Found {len(pl_mask_files)} power line masks")

    # Initialize counters
    total_pixels = 0
    veg_pixels = 0
    pl_pixels = 0
    background_pixels = 0
    overlap_pixels = 0  # Pixels that are both vegetation and power line

    # Per-image statistics
    per_image_stats = []

    # Process all images
    print("\nAnalyzing masks...")
    for veg_file, pl_file in tqdm(zip(veg_mask_files, pl_mask_files), total=len(veg_mask_files)):
        # Load masks
        veg_mask = np.array(Image.open(veg_file))
        pl_mask = np.array(Image.open(pl_file))

        # Ensure binary masks (0 or 1)
        veg_mask = (veg_mask > 0).astype(np.uint8)
        pl_mask = (pl_mask > 0).astype(np.uint8)

        # Calculate pixels for this image
        img_total = veg_mask.size
        img_veg = veg_mask.sum()
        img_pl = pl_mask.sum()
        img_overlap = (veg_mask & pl_mask).sum()
        img_background = img_total - img_veg - img_pl + img_overlap

        # Update global counters
        total_pixels += img_total
        veg_pixels += img_veg
        pl_pixels += img_pl
        overlap_pixels += img_overlap
        background_pixels += img_background

        # Store per-image stats
        per_image_stats.append({
            'filename': veg_file.name,
            'total_pixels': int(img_total),
            'veg_pixels': int(img_veg),
            'pl_pixels': int(img_pl),
            'overlap_pixels': int(img_overlap),
            'background_pixels': int(img_background),
            'veg_density': float(img_veg / img_total * 100),
            'pl_density': float(img_pl / img_total * 100),
            'overlap_density': float(img_overlap / img_total * 100),
            'background_density': float(img_background / img_total * 100)
        })

    # Calculate overall statistics
    veg_density = veg_pixels / total_pixels * 100
    pl_density = pl_pixels / total_pixels * 100
    overlap_density = overlap_pixels / total_pixels * 100
    background_density = background_pixels / total_pixels * 100

    # Print results
    print("\n" + "="*80)
    print("CLASS DENSITY ANALYSIS")
    print("="*80)
    print(f"\nTotal images analyzed: {len(veg_mask_files)}")
    print(f"Total pixels: {total_pixels:,}")

    print("\n" + "-"*80)
    print("OVERALL CLASS DISTRIBUTION")
    print("-"*80)
    print(f"Vegetation:   {veg_pixels:>15,} pixels ({veg_density:>6.2f}%)")
    print(f"Power Lines:  {pl_pixels:>15,} pixels ({pl_density:>6.2f}%)")
    print(f"Overlap:      {overlap_pixels:>15,} pixels ({overlap_density:>6.2f}%)")
    print(f"Background:   {background_pixels:>15,} pixels ({background_density:>6.2f}%)")

    # Calculate per-image statistics
    veg_densities = [s['veg_density'] for s in per_image_stats]
    pl_densities = [s['pl_density'] for s in per_image_stats]

    print("\n" + "-"*80)
    print("PER-IMAGE STATISTICS")
    print("-"*80)
    print(f"Vegetation density:")
    print(f"  Mean:   {np.mean(veg_densities):>6.2f}%")
    print(f"  Std:    {np.std(veg_densities):>6.2f}%")
    print(f"  Min:    {np.min(veg_densities):>6.2f}%")
    print(f"  Max:    {np.max(veg_densities):>6.2f}%")
    print(f"  Median: {np.median(veg_densities):>6.2f}%")

    print(f"\nPower line density:")
    print(f"  Mean:   {np.mean(pl_densities):>6.2f}%")
    print(f"  Std:    {np.std(pl_densities):>6.2f}%")
    print(f"  Min:    {np.min(pl_densities):>6.2f}%")
    print(f"  Max:    {np.max(pl_densities):>6.2f}%")
    print(f"  Median: {np.median(pl_densities):>6.2f}%")

    # Find images with highest/lowest densities
    print("\n" + "-"*80)
    print("EXTREME CASES")
    print("-"*80)

    # Highest vegetation
    max_veg_idx = np.argmax(veg_densities)
    print(f"Highest vegetation: {per_image_stats[max_veg_idx]['filename']} ({veg_densities[max_veg_idx]:.2f}%)")

    # Lowest vegetation
    min_veg_idx = np.argmin(veg_densities)
    print(f"Lowest vegetation:  {per_image_stats[min_veg_idx]['filename']} ({veg_densities[min_veg_idx]:.2f}%)")

    # Highest power lines
    max_pl_idx = np.argmax(pl_densities)
    print(f"Highest power lines: {per_image_stats[max_pl_idx]['filename']} ({pl_densities[max_pl_idx]:.2f}%)")

    # Lowest power lines
    min_pl_idx = np.argmin(pl_densities)
    print(f"Lowest power lines:  {per_image_stats[min_pl_idx]['filename']} ({pl_densities[min_pl_idx]:.2f}%)")

    # Save results to JSON
    results = {
        'summary': {
            'total_images': len(veg_mask_files),
            'total_pixels': int(total_pixels),
            'vegetation_pixels': int(veg_pixels),
            'powerline_pixels': int(pl_pixels),
            'overlap_pixels': int(overlap_pixels),
            'background_pixels': int(background_pixels),
            'vegetation_density': float(veg_density),
            'powerline_density': float(pl_density),
            'overlap_density': float(overlap_density),
            'background_density': float(background_density)
        },
        'per_image_statistics': {
            'vegetation': {
                'mean': float(np.mean(veg_densities)),
                'std': float(np.std(veg_densities)),
                'min': float(np.min(veg_densities)),
                'max': float(np.max(veg_densities)),
                'median': float(np.median(veg_densities))
            },
            'powerline': {
                'mean': float(np.mean(pl_densities)),
                'std': float(np.std(pl_densities)),
                'min': float(np.min(pl_densities)),
                'max': float(np.max(pl_densities)),
                'median': float(np.median(pl_densities))
            }
        },
        'per_image_details': per_image_stats
    }

    output_file = data_dir / 'class_density_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")

    return results


if __name__ == '__main__':
    data_dir = '/Users/user/Desktop/university/thesis/veg_enc_code/camera/kios_vepl/data/processed_data'
    results = analyze_class_density(data_dir)
