"""
Interactive 3D Visualization of Test Set
Uses matplotlib interactive window for rotating/zooming point clouds
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import json
import argparse

# Enable interactive mode
plt.ion()

# Configuration
class Config:
    def __init__(self):
        self.data_dir = '../data/processed_data_optimized'
        self.output_dir = 'test_visualizations'
        self.filter_by_class = None  # 'powerlines', 'pylons', 'both', or None
        self.sample_index = None  # Specific index or None for random

# Class information
CLASS_NAMES = ['High Vegetation', 'Power Lines', 'Pylons', 'Others']
CLASS_COLORS = ['darkgreen', 'orange', 'blue', 'gray']


def load_test_samples(data_dir):
    """Load all test sample paths"""
    test_dir = Path(data_dir) / 'test'
    test_samples = sorted(list(test_dir.glob('*.pt')))

    print(f"Found {len(test_samples)} test samples")

    # Load metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
            print(f"\nMetadata:")
            print(f"  Classes: {metadata['n_classes']}")
            print(f"  Class names: {metadata['class_names']}")
            print(f"  Test samples: {metadata['n_test']}")

    return test_samples


def analyze_sample(sample_path):
    """Analyze a single sample"""
    sample = torch.load(sample_path, weights_only=False)
    points = sample['input_pcd'].numpy()
    labels = (sample['semantic_labels'].squeeze() - 1).numpy()

    # Calculate statistics
    stats = {
        'total_points': len(points),
        'has_powerlines': (labels == 1).sum() > 10,
        'has_pylons': (labels == 2).sum() > 5,
        'powerline_count': (labels == 1).sum(),
        'pylon_count': (labels == 2).sum(),
        'class_counts': {}
    }

    for i in range(4):
        stats['class_counts'][CLASS_NAMES[i]] = (labels == i).sum()

    return stats, points, labels


def find_samples_with_infrastructure(test_samples, filter_type='both', max_to_scan=200):
    """Find samples with power lines and/or pylons"""
    print(f"\nScanning for samples with {filter_type}...")

    matching_samples = []

    for idx in range(min(max_to_scan, len(test_samples))):
        stats, _, _ = analyze_sample(test_samples[idx])

        if filter_type == 'powerlines' and stats['has_powerlines']:
            matching_samples.append((idx, stats['powerline_count']))
        elif filter_type == 'pylons' and stats['has_pylons']:
            matching_samples.append((idx, stats['pylon_count']))
        elif filter_type == 'both' and stats['has_powerlines'] and stats['has_pylons']:
            matching_samples.append((idx, stats['powerline_count'] + stats['pylon_count']))

    # Sort by count
    matching_samples.sort(key=lambda x: x[1], reverse=True)

    print(f"Found {len(matching_samples)} matching samples")
    if matching_samples:
        print(f"Top 5 samples:")
        for i, (idx, count) in enumerate(matching_samples[:5], 1):
            print(f"  {i}. Index {idx}: {count} points")

    return [idx for idx, _ in matching_samples]


def visualize_sample_interactive(points, labels, sample_idx, stats):
    """Create interactive 3D visualization"""

    # Create figure with 3 subplots
    fig = plt.figure(figsize=(20, 6))
    fig.canvas.manager.set_window_title(f'Test Sample {sample_idx} - Interactive View')
    fig.suptitle(f'Test Sample {sample_idx}', fontsize=16, fontweight='bold')

    # Full view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title('Full Point Cloud (Rotate/Zoom with mouse)', fontsize=12, fontweight='bold')
    for cls in range(4):
        mask = labels == cls
        if mask.sum() > 0:
            size = 20 if cls in [1, 2] else 1
            ax1.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                       c=CLASS_COLORS[cls], label=f"{CLASS_NAMES[cls]} ({mask.sum()})",
                       s=size, alpha=0.7)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Power Lines Focus
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title('Power Lines Focus', fontsize=12, fontweight='bold', color='orange')
    powerline_mask = labels == 1
    if powerline_mask.sum() > 0:
        # Power lines
        ax2.scatter(points[powerline_mask, 0], points[powerline_mask, 1],
                   points[powerline_mask, 2], c='orange', s=25, alpha=0.9, label='Power Lines')
        # Context (vegetation)
        veg_mask = labels == 0
        if veg_mask.sum() > 0:
            sample_size = min(3000, veg_mask.sum())
            sample_veg = np.random.choice(np.where(veg_mask)[0], sample_size, replace=False)
            ax2.scatter(points[sample_veg, 0], points[sample_veg, 1],
                       points[sample_veg, 2], c='lightgreen', s=1, alpha=0.15, label='Vegetation (context)')
        ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 0.5, 'No power lines\nin this sample',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Pylons Focus
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title('Pylons Focus', fontsize=12, fontweight='bold', color='blue')
    pylon_mask = labels == 2
    if pylon_mask.sum() > 0:
        # Pylons
        ax3.scatter(points[pylon_mask, 0], points[pylon_mask, 1],
                   points[pylon_mask, 2], c='blue', s=40, alpha=0.9, marker='^', label='Pylons')
        # Context
        context_mask = (labels == 0) | (labels == 1)
        if context_mask.sum() > 0:
            sample_size = min(2000, context_mask.sum())
            sample_ctx = np.random.choice(np.where(context_mask)[0], sample_size, replace=False)
            ctx_colors = ['lightgreen' if labels[i] == 0 else 'lightsalmon' for i in sample_ctx]
            ax3.scatter(points[sample_ctx, 0], points[sample_ctx, 1],
                       points[sample_ctx, 2], c=ctx_colors, s=1, alpha=0.15, label='Context')
        ax3.legend(loc='upper right', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 0.5, 'No pylons\nin this sample',
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Add statistics text
    stats_text = f"Total Points: {stats['total_points']:,}\n"
    stats_text += "\n".join([f"{name}: {count:,}" for name, count in stats['class_counts'].items()])
    fig.text(0.02, 0.02, stats_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show(block=False)

    return fig


def interactive_viewer(config):
    """Main interactive viewer"""
    print("="*80)
    print("INTERACTIVE 3D TEST SET VIEWER")
    print("="*80)
    print("\nControls:")
    print("  - Mouse: Click and drag to rotate")
    print("  - Mouse wheel: Zoom in/out")
    print("  - Close window to view next sample")
    print("  - Ctrl+C to exit\n")
    print("="*80)

    # Load test samples
    test_samples = load_test_samples(config.data_dir)

    if not test_samples:
        print("No test samples found!")
        return

    # Determine which samples to show
    if config.filter_by_class:
        sample_indices = find_samples_with_infrastructure(
            test_samples,
            filter_type=config.filter_by_class
        )
        if not sample_indices:
            print(f"No samples found with {config.filter_by_class}")
            return
    elif config.sample_index is not None:
        sample_indices = [config.sample_index]
    else:
        # Show first 10 samples
        sample_indices = list(range(min(10, len(test_samples))))
        print(f"\nShowing first {len(sample_indices)} samples")

    # View each sample
    for i, idx in enumerate(sample_indices):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(sample_indices)} - Index: {idx}")
        print(f"File: {test_samples[idx].name}")
        print(f"{'='*80}")

        # Load and analyze sample
        stats, points, labels = analyze_sample(test_samples[idx])

        # Print statistics
        print(f"\nStatistics:")
        print(f"  Total points: {stats['total_points']:,}")
        for name, count in stats['class_counts'].items():
            pct = 100 * count / stats['total_points']
            print(f"  {name}: {count:,} ({pct:.2f}%)")

        # Create interactive visualization
        fig = visualize_sample_interactive(points, labels, idx, stats)

        # Wait for user to close window
        try:
            print("\nViewing... Close window to continue to next sample (or Ctrl+C to exit)")
            plt.show(block=True)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            plt.close('all')
            break

        plt.close(fig)

    print("\n" + "="*80)
    print("VIEWER CLOSED")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Interactive 3D Test Set Viewer')
    parser.add_argument('--data-dir', type=str, default='../data/processed_data_optimized',
                        help='Path to processed data directory')
    parser.add_argument('--index', type=int, default=None,
                        help='Specific sample index to view')
    parser.add_argument('--filter', type=str, choices=['powerlines', 'pylons', 'both'],
                        default=None, help='Filter samples by class')

    args = parser.parse_args()

    # Create config
    config = Config()
    config.data_dir = args.data_dir
    config.sample_index = args.index
    config.filter_by_class = args.filter

    # Run viewer
    interactive_viewer(config)


if __name__ == "__main__":
    main()
