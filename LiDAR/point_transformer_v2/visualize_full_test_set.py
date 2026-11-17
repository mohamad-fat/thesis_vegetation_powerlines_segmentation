"""
Visualize the ENTIRE test set as one combined 3D point cloud
Interactive viewer for the full dataset
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from tqdm import tqdm
import argparse

# Enable interactive mode
plt.ion()

# Class information
CLASS_NAMES = ['High Vegetation', 'Power Lines', 'Pylons', 'Others']
CLASS_COLORS = ['darkgreen', 'orange', 'blue', 'gray']


def load_and_combine_test_set(data_dir, max_samples=None, subsample_rate=1.0):
    """
    Load all test samples and combine into one point cloud

    Args:
        data_dir: Path to processed data
        max_samples: Maximum number of samples to load (None = all)
        subsample_rate: Fraction of points to keep from each sample (1.0 = all)
    """
    test_dir = Path(data_dir) / 'test'
    test_samples = sorted(list(test_dir.glob('*.pt')))

    if max_samples:
        test_samples = test_samples[:max_samples]

    print(f"Loading {len(test_samples)} test samples...")
    print(f"Subsample rate: {subsample_rate}")

    all_points = []
    all_labels = []
    all_centroids = []

    for sample_path in tqdm(test_samples, desc="Loading samples"):
        sample = torch.load(sample_path, weights_only=False)
        points = sample['input_pcd'].numpy()
        labels = (sample['semantic_labels'].squeeze() - 1).numpy()  # 0-indexed
        centroid = sample.get('centroid', np.zeros(3))

        # Subsample if needed
        if subsample_rate < 1.0:
            n_keep = int(len(points) * subsample_rate)
            indices = np.random.choice(len(points), n_keep, replace=False)
            points = points[indices]
            labels = labels[indices]

        # Denormalize to original coordinates
        points_denormalized = points + centroid

        all_points.append(points_denormalized)
        all_labels.append(labels)
        all_centroids.append(centroid)

    # Combine all samples
    combined_points = np.vstack(all_points)
    combined_labels = np.concatenate(all_labels)

    print(f"\nCombined point cloud:")
    print(f"  Total points: {len(combined_points):,}")
    print(f"  Memory size: ~{combined_points.nbytes / 1024**2:.1f} MB")

    print(f"\n  Class distribution:")
    for i, name in enumerate(CLASS_NAMES):
        count = (combined_labels == i).sum()
        pct = 100 * count / len(combined_labels)
        print(f"    {name}: {count:,} ({pct:.2f}%)")

    print(f"\n  Spatial extent:")
    print(f"    X: [{combined_points[:, 0].min():.2f}, {combined_points[:, 0].max():.2f}]")
    print(f"    Y: [{combined_points[:, 1].min():.2f}, {combined_points[:, 1].max():.2f}]")
    print(f"    Z: [{combined_points[:, 2].min():.2f}, {combined_points[:, 2].max():.2f}]")

    return combined_points, combined_labels


def visualize_full_test_set(points, labels, downsample_for_display=None):
    """
    Create interactive 3D visualization of the full test set

    Args:
        points: Combined point cloud (N, 3)
        labels: Combined labels (N,)
        downsample_for_display: Max points to display (for performance)
    """

    # Downsample for display if needed
    if downsample_for_display and len(points) > downsample_for_display:
        print(f"\nDownsampling to {downsample_for_display:,} points for display...")
        indices = np.random.choice(len(points), downsample_for_display, replace=False)
        points_display = points[indices]
        labels_display = labels[indices]
    else:
        points_display = points
        labels_display = labels

    print(f"\nDisplaying {len(points_display):,} points")

    # Create figure with single view
    fig = plt.figure(figsize=(14, 10))
    fig.canvas.manager.set_window_title('Full Test Set - 3D Interactive View')
    fig.suptitle(f'Complete Test Set - {len(points):,} total points', fontsize=16, fontweight='bold')

    # Single view: Full point cloud
    print("\nRendering Complete Test Set View...")
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Complete Test Set - All Classes', fontsize=12, fontweight='bold')

    for cls in range(4):
        mask = labels_display == cls
        if mask.sum() > 0:
            size = 10 if cls in [1, 2] else 1
            ax.scatter(points_display[mask, 0], points_display[mask, 1], points_display[mask, 2],
                       c=CLASS_COLORS[cls], label=f"{CLASS_NAMES[cls]} ({(labels == cls).sum():,})",
                       s=size, alpha=0.6)

    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)

    plt.tight_layout()

    print("\n" + "="*80)
    print("INTERACTIVE CONTROLS:")
    print("  - Mouse: Click and drag to rotate")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Close window to exit")
    print("="*80)

    plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description='Visualize Full Test Set')
    parser.add_argument('--data-dir', type=str, default='../data/processed_data_optimized',
                        help='Path to processed data directory')
    parser.add_argument('--max-samples', type=int, default=1363,
                        help='Maximum number of test samples to load (default: 1363 = 10% of test set)')
    parser.add_argument('--subsample', type=float, default=1.0,
                        help='Fraction of points to keep from each sample (0.0-1.0)')
    parser.add_argument('--display-limit', type=int, default=500000,
                        help='Max points to display (for performance)')

    args = parser.parse_args()

    print("="*80)
    print("FULL TEST SET 3D VIEWER - 10% DATASET")
    print("="*80)
    print()

    # Load and combine test set
    combined_points, combined_labels = load_and_combine_test_set(
        args.data_dir,
        max_samples=args.max_samples,
        subsample_rate=args.subsample
    )

    # Visualize
    visualize_full_test_set(
        combined_points,
        combined_labels,
        downsample_for_display=args.display_limit
    )

    print("\n" + "="*80)
    print("VIEWER CLOSED")
    print("="*80)


if __name__ == "__main__":
    main()
