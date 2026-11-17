"""
Convert PyTorch .pt test files to LAS format
============================================

Converts test chunks from .pt format to .las point cloud files.

Usage:
    # Convert to individual .las files (one per chunk)
    python convert_pt_to_las.py --mode individual

    # Merge all chunks into a single .las file
    python convert_pt_to_las.py --mode merged

    # Both individual and merged
    python convert_pt_to_las.py --mode both
"""

import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import laspy

# Class mapping (1-indexed in .pt files -> 0-indexed to match predictions)
CLASS_MAPPING = {
    1: 0,   # High Vegetation -> class 0
    2: 1,   # Power Lines -> class 1
    3: 2,   # Pylons -> class 2
    4: 3    # Others -> class 3
}

CLASS_NAMES = {
    1: "High Vegetation",
    2: "Power Lines",
    3: "Pylons",
    4: "Others"
}

def convert_chunk_to_las(pt_file, output_path, add_centroid=False):
    """
    Convert a single .pt file to .las format

    Args:
        pt_file: Path to .pt file
        output_path: Path to save .las file
        add_centroid: If True, add centroid offset back to coordinates
    """
    # Load the .pt file
    sample = torch.load(pt_file, weights_only=False)

    # Extract data
    points = sample['input_pcd'].numpy()  # Nx3
    labels = sample['semantic_labels'].numpy().reshape(-1)  # N

    # Add centroid offset if requested (to restore original coordinates)
    if add_centroid and 'centroid' in sample:
        centroid = sample['centroid']
        points = points + centroid

    # Create LAS file (version 1.2, point format 1)
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = np.min(points, axis=0)
    header.scales = np.array([0.01, 0.01, 0.01])  # 1cm precision

    las = laspy.LasData(header)

    # Set coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Set classification (map from our labels to LAS standard codes)
    classification = np.array([CLASS_MAPPING.get(int(label), 0) for label in labels], dtype=np.uint8)
    las.classification = classification

    # Set intensity (optional - use label value for visualization)
    las.intensity = labels.astype(np.uint16) * 10000

    # Write to file
    las.write(output_path)

    return len(points)


def merge_all_chunks(pt_files, output_path, add_centroid=False):
    """
    Merge all .pt chunks into a single .las file

    Args:
        pt_files: List of paths to .pt files
        output_path: Path to save merged .las file
        add_centroid: If True, add centroid offset back to coordinates
    """
    print("Loading all chunks...")
    all_points = []
    all_labels = []

    for pt_file in tqdm(pt_files, desc="Loading chunks"):
        sample = torch.load(pt_file, weights_only=False)
        points = sample['input_pcd'].numpy()
        labels = sample['semantic_labels'].numpy().reshape(-1)

        # Add centroid offset if requested
        if add_centroid and 'centroid' in sample:
            centroid = sample['centroid']
            points = points + centroid

        all_points.append(points)
        all_labels.append(labels)

    # Concatenate all data
    print("Merging point clouds...")
    all_points = np.vstack(all_points)
    all_labels = np.concatenate(all_labels)

    print(f"Total points: {len(all_points):,}")

    # Create LAS file
    print("Creating LAS file...")
    header = laspy.LasHeader(point_format=1, version="1.2")
    header.offsets = np.min(all_points, axis=0)
    header.scales = np.array([0.01, 0.01, 0.01])  # 1cm precision

    las = laspy.LasData(header)

    # Set coordinates
    las.x = all_points[:, 0]
    las.y = all_points[:, 1]
    las.z = all_points[:, 2]

    # Set classification
    classification = np.array([CLASS_MAPPING.get(int(label), 0) for label in all_labels], dtype=np.uint8)
    las.classification = classification

    # Set intensity
    las.intensity = all_labels.astype(np.uint16) * 10000

    # Write to file
    print(f"Writing to {output_path}...")
    las.write(output_path)

    return len(all_points)


def main():
    parser = argparse.ArgumentParser(description='Convert .pt test files to .las format')
    parser.add_argument('--data_dir', type=str,
                        default='/Users/user/Desktop/university/thesis/veg_enc_code/LiDAR/data/processed_data_optimized',
                        help='Path to processed data directory')
    parser.add_argument('--mode', type=str, choices=['individual', 'merged', 'both'], default='both',
                        help='Conversion mode: individual files, merged file, or both')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: data_dir/test_las)')
    parser.add_argument('--add_centroid', action='store_true',
                        help='Add centroid offset to restore original coordinates')
    args = parser.parse_args()

    # Setup paths
    data_dir = Path(args.data_dir)
    test_dir = data_dir / 'test'

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / 'test_las'

    output_dir.mkdir(exist_ok=True)

    # Get all .pt files
    pt_files = sorted(list(test_dir.glob('*.pt')))
    print(f"\nFound {len(pt_files)} test files")

    if len(pt_files) == 0:
        print("No .pt files found!")
        return

    print("\n" + "="*80)
    print("CONVERTING PT FILES TO LAS FORMAT")
    print("="*80)
    print(f"Input directory: {test_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Add centroid offset: {args.add_centroid}")
    print(f"Number of files: {len(pt_files)}")
    print("\nClass Mapping (to match predictions file):")
    for orig_label, (class_name, output_class) in enumerate(zip(CLASS_NAMES.values(), CLASS_MAPPING.values()), 1):
        print(f"  {orig_label} ({class_name}) -> Class {output_class}")
    print("="*80 + "\n")

    # Convert individual files
    if args.mode in ['individual', 'both']:
        print("Converting individual chunks to .las files...")
        individual_dir = output_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)

        total_points = 0
        for pt_file in tqdm(pt_files, desc="Converting"):
            output_file = individual_dir / f"{pt_file.stem}.las"
            n_points = convert_chunk_to_las(pt_file, output_file, args.add_centroid)
            total_points += n_points

        print(f"\n✓ Converted {len(pt_files)} files")
        print(f"  Total points: {total_points:,}")
        print(f"  Saved to: {individual_dir}/")

    # Merge all files
    if args.mode in ['merged', 'both']:
        print("\nMerging all chunks into single .las file...")
        merged_file = output_dir / 'test_set_merged.las'
        n_points = merge_all_chunks(pt_files, merged_file, args.add_centroid)

        print(f"\n✓ Created merged file")
        print(f"  Total points: {n_points:,}")
        print(f"  Saved to: {merged_file}")

    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}/")
    print("\nYou can now open the .las files in CloudCompare, QGIS, or other point cloud viewers.")
    print("\nClassification Legend (matches predictions file):")
    print("  Class 0: High Vegetation")
    print("  Class 1: Power Lines")
    print("  Class 2: Pylons")
    print("  Class 3: Others")
    print("\nThese classes match the predictions file for easy visual comparison!")


if __name__ == '__main__':
    main()
