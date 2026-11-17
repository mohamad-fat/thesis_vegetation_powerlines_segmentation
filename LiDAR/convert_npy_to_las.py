#!/usr/bin/env python3
"""
Convert NPY Point Cloud to LAS Format
======================================
Converts a numpy array containing point cloud data to LAS format.

Usage:
    python convert_npy_to_las.py --input full_pointcloud.npy --output full_pointcloud.las
"""

import argparse
import numpy as np
import laspy
from pathlib import Path

def convert_npy_to_las(input_path, output_path, classification_field='predictions'):
    """
    Convert NPY point cloud to LAS format.

    Args:
        input_path: Path to input .npy file
        output_path: Path to output .las file
        classification_field: Which field to use for classification
                             ('predictions', 'labels', or None)
    """
    print("\n" + "="*80)
    print("CONVERTING NPY TO LAS")
    print("="*80)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print("="*80 + "\n")

    # Load NPY file
    print("Loading NPY file...")
    data = np.load(input_path, allow_pickle=True)

    # Check if it's a dictionary or array
    if isinstance(data, np.ndarray) and data.dtype == object:
        # It's a dictionary saved as object array
        data = data.item()
        print(f"Data type: Dictionary with keys: {data.keys()}")

        # Extract points
        if 'points' in data:
            points = data['points']
        elif 'xyz' in data:
            points = data['xyz']
        else:
            raise ValueError(f"No 'points' or 'xyz' field found. Available: {data.keys()}")

        # Extract classification if available
        if classification_field and classification_field in data:
            classification = data[classification_field]
            print(f"Using '{classification_field}' for classification")
        elif 'labels' in data:
            classification = data['labels']
            print("Using 'labels' for classification")
        elif 'predictions' in data:
            classification = data['predictions']
            print("Using 'predictions' for classification")
        else:
            classification = None
            print("No classification data found")

    elif isinstance(data, np.ndarray) and data.ndim == 2:
        # It's a simple array (N, 3) or (N, 4)
        print(f"Data type: Array with shape {data.shape}")

        if data.shape[1] >= 3:
            points = data[:, :3]
            if data.shape[1] >= 4:
                classification = data[:, 3].astype(np.uint8)
                print("Using 4th column for classification")
            else:
                classification = None
        else:
            raise ValueError(f"Expected at least 3 columns, got {data.shape[1]}")
    else:
        raise ValueError(f"Unexpected data format: {type(data)}, shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")

    # Print statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Total points: {len(points):,}")
    print(f"  X range: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y range: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")
    print(f"  Z range: [{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

    if classification is not None:
        unique_classes, counts = np.unique(classification, return_counts=True)
        print(f"\nClassification distribution:")
        class_names = {0: 'High Vegetation', 1: 'Power Lines', 2: 'Pylons', 3: 'Others'}
        for cls, count in zip(unique_classes, counts):
            name = class_names.get(int(cls), f'Class {cls}')
            percentage = 100 * count / len(classification)
            print(f"  {name}: {count:,} ({percentage:.2f}%)")

    # Create LAS file
    print("\nCreating LAS file...")

    # Create header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.offsets = [points[:, 0].min(), points[:, 1].min(), points[:, 2].min()]
    header.scales = [0.001, 0.001, 0.001]  # 1mm precision

    # Create LAS data
    las = laspy.LasData(header)

    # Add point coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Add classification if available
    if classification is not None:
        las.classification = classification.astype(np.uint8)

    # Add default values for other fields
    las.intensity = np.zeros(len(points), dtype=np.uint16)
    las.return_number = np.ones(len(points), dtype=np.uint8)
    las.number_of_returns = np.ones(len(points), dtype=np.uint8)

    # Save LAS file
    print(f"Writing LAS file to {output_path}...")
    las.write(output_path)

    # Verify output
    file_size_mb = Path(output_path).stat().st_size / (1024**2)
    print(f"\n{'='*80}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*80}")
    print(f"Output file: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"Total points: {len(points):,}")
    print(f"\nYou can now open this file in CloudCompare, QGIS, or any LAS viewer!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert NPY point cloud to LAS format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic conversion
    python convert_npy_to_las.py --input full_pointcloud.npy --output full_pointcloud.las

    # Use ground truth labels instead of predictions
    python convert_npy_to_las.py --input full_pointcloud.npy --output full_pointcloud.las --classification labels

    # No classification (just XYZ)
    python convert_npy_to_las.py --input full_pointcloud.npy --output full_pointcloud.las --classification none
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input NPY file path')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Output LAS file path')
    parser.add_argument('--classification', '-c', type=str, default='predictions',
                       choices=['predictions', 'labels', 'none'],
                       help='Which field to use for point classification (default: predictions)')

    args = parser.parse_args()

    # Convert paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Check input exists
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        exit(1)

    # Determine classification field
    if args.classification == 'none':
        classification_field = None
    else:
        classification_field = args.classification

    # Convert
    try:
        convert_npy_to_las(input_path, output_path, classification_field)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
