"""
Step 1: Data Preprocessing - Convert .las files to PyTorch format (Optimized Version)
"""
import numpy as np
import torch
import laspy
from pathlib import Path
from tqdm import tqdm
import json

class LASDataProcessor:
    """Process .las files and convert to a format suitable for the model"""

    def __init__(self, data_dir, output_dir, filter_others=True, downsample_ratio=1.0):
        """
        Args:
            data_dir: Input directory with .las files
            output_dir: Output directory for processed files
            filter_others: If True, remove points labeled as 'Others' (class 4) to reduce file size
            downsample_ratio: Keep only this fraction of points (e.g., 0.5 = keep 50% of points)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.filter_others = filter_others
        self.downsample_ratio = downsample_ratio

        print(f"Configuration:")
        print(f"  Filter 'Others' class: {filter_others}")
        print(f"  Downsample ratio: {downsample_ratio}")

    def read_las_file(self, las_path):
        """Read a .las file and extract points and labels"""
        print(f"Reading {las_path}...")
        las = laspy.read(las_path)

        # Extract XYZ coordinates
        points = np.vstack((las.x, las.y, las.z)).T

        # Extract labels (classification field in LAS)
        if hasattr(las, 'classification'):
            labels = las.classification
        elif hasattr(las, 'label'):
            labels = las.label
        else:
            raise ValueError("No classification/label field found in LAS file")

        # Convert labels to numpy array if needed
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        print(f"  Total points: {len(points):,}")
        print(f"  Original label distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    Class {label}: {count:,} ({100*count/len(labels):.2f}%)")

        return points.astype(np.float32), labels.astype(np.int32)

    def normalize_points(self, points):
        """Normalize point cloud to zero mean"""
        centroid = np.mean(points, axis=0)
        points = points - centroid
        return points, centroid

    def map_labels(self, labels):
        """
        Map your dataset labels to the desired classes.

        Original Dataset Classes:
         1: Unclassified, 2: Ground, 3: Low Vegetation, 4: Medium Vegetation,
         5: High Vegetation, 6: Building, 7: Low Point (low noise),
         8: Model Key/Reserved, 9: Water, 10: Power Lines, 11: Pylons.

        New Target Classes:
         1: High Vegetation
         2: Power Lines
         3: Pylons
         4: Others
        """
        # Default all points to the 'Others' class (4)
        mapped_labels = np.full(labels.shape, 4, dtype=np.int32)

        # Map High Vegetation (original class 5) to new class 1
        mapped_labels[labels == 5] = 1

        # Map Power Lines (original class 10) to new class 2
        mapped_labels[labels == 10] = 2

        # Map Pylons (original class 11) to new class 3
        mapped_labels[labels == 11] = 3

        return mapped_labels

    def filter_and_downsample(self, points, labels):
        """Filter out 'Others' class and/or downsample points"""
        mask = np.ones(len(points), dtype=bool)

        # Filter out 'Others' class (4) if enabled
        if self.filter_others:
            mask = labels != 4
            print(f"  Filtering 'Others' class: {np.sum(~mask):,} points removed")

        points = points[mask]
        labels = labels[mask]

        # Downsample if ratio < 1.0
        if self.downsample_ratio < 1.0 and len(points) > 0:
            n_keep = int(len(points) * self.downsample_ratio)
            indices = np.random.choice(len(points), n_keep, replace=False)
            points = points[indices]
            labels = labels[indices]
            print(f"  Downsampling to {self.downsample_ratio*100}%: kept {len(points):,} points")

        return points, labels

    def split_into_chunks(self, points, labels, chunk_size=4096, spatial_grid=True):
        """
        Split large point cloud into manageable chunks.

        Args:
            points: Nx3 array of point coordinates
            labels: N array of labels
            chunk_size: Target points per chunk
            spatial_grid: If True, use spatial grid slicing (like image tiles).
                         If False, use random shuffling (old method).
        """
        n_points = len(points)
        chunks = []

        if n_points == 0:
            return []

        if n_points <= chunk_size:
            return [(points, labels)]

        if spatial_grid:
            # Grid-based slicing (like 2D image tiling)
            # Find spatial bounds in X-Y plane
            x_min, y_min = points[:, 0].min(), points[:, 1].min()
            x_max, y_max = points[:, 0].max(), points[:, 1].max()

            # Calculate approximate grid size
            # Estimate how many points per unit area
            area = (x_max - x_min) * (y_max - y_min)
            if area == 0:
                # Fallback to random if all points are at same X-Y location
                spatial_grid = False
            else:
                points_per_area = n_points / area
                tile_area = chunk_size / points_per_area
                tile_size = np.sqrt(tile_area)

                # Calculate number of tiles in each dimension
                n_tiles_x = max(1, int(np.ceil((x_max - x_min) / tile_size)))
                n_tiles_y = max(1, int(np.ceil((y_max - y_min) / tile_size)))

                print(f"  Creating {n_tiles_x}x{n_tiles_y} spatial grid (tile size: {tile_size:.2f}m)")

                # Create grid tiles
                x_edges = np.linspace(x_min, x_max, n_tiles_x + 1)
                y_edges = np.linspace(y_min, y_max, n_tiles_y + 1)

                for i in range(n_tiles_x):
                    for j in range(n_tiles_y):
                        # Find points in this grid cell
                        mask = (
                            (points[:, 0] >= x_edges[i]) &
                            (points[:, 0] < x_edges[i+1]) &
                            (points[:, 1] >= y_edges[j]) &
                            (points[:, 1] < y_edges[j+1])
                        )

                        # Handle edge case for last tile (include boundary)
                        if i == n_tiles_x - 1:
                            mask |= (points[:, 0] == x_edges[i+1]) & (points[:, 1] >= y_edges[j]) & (points[:, 1] <= y_edges[j+1])
                        if j == n_tiles_y - 1:
                            mask |= (points[:, 1] == y_edges[j+1]) & (points[:, 0] >= x_edges[i]) & (points[:, 0] <= x_edges[i+1])

                        if mask.sum() > 0:
                            tile_points = points[mask]
                            tile_labels = labels[mask]

                            # If tile is too large, split it further randomly
                            if len(tile_points) > chunk_size * 1.5:
                                n_sub_chunks = int(np.ceil(len(tile_points) / chunk_size))
                                shuffle_idx = np.random.permutation(len(tile_points))
                                for k in range(n_sub_chunks):
                                    start_idx = k * chunk_size
                                    end_idx = min((k + 1) * chunk_size, len(tile_points))
                                    sub_idx = shuffle_idx[start_idx:end_idx]
                                    chunks.append((tile_points[sub_idx], tile_labels[sub_idx]))
                            else:
                                chunks.append((tile_points, tile_labels))

        if not spatial_grid:
            # Fallback: Random shuffling (old method)
            print("  Using random shuffling for chunking")
            shuffle_idx = np.random.permutation(n_points)
            points = points[shuffle_idx]
            labels = labels[shuffle_idx]

            n_chunks = int(np.ceil(n_points / chunk_size))
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n_points)
                chunk_points = points[start_idx:end_idx]
                chunk_labels = labels[start_idx:end_idx]
                chunks.append((chunk_points, chunk_labels))

        return chunks

    def process_all_files(self, train_split=0.8, max_files=None):
        """Process all .las files in the directory"""
        # Filter only files starting with 'segmented_croped_chunk_'
        all_las_files = sorted(self.data_dir.glob('*.las'))
        las_files = [f for f in all_las_files if f.name.startswith('segmented_croped_chunk_')]

        if not las_files:
            print(f"No files starting with 'segmented_croped_chunk_' found in {self.data_dir}")
            print(f"Available .las files: {[f.name for f in all_las_files[:5]]}")
            raise ValueError(f"No .las files starting with 'segmented_croped_chunk_' found in {self.data_dir}")

        if max_files is not None:
            las_files = las_files[:max_files]

        print(f"Found {len(las_files)} .las files starting with 'segmented_croped_chunk_'")

        all_samples = []
        total_stats = {
            'total_points_original': 0,
            'total_points_after_filtering': 0,
            'total_chunks': 0,
            'class_distribution': {1: 0, 2: 0, 3: 0, 4: 0}
        }

        for las_file in tqdm(las_files, desc="Processing LAS files"):
            # Read data
            points, labels = self.read_las_file(las_file)
            total_stats['total_points_original'] += len(points)

            # Normalize
            points, centroid = self.normalize_points(points)

            # Map labels to the new classes
            labels = self.map_labels(labels)

            # Filter and downsample
            points, labels = self.filter_and_downsample(points, labels)
            total_stats['total_points_after_filtering'] += len(points)

            # Update class distribution
            unique, counts = np.unique(labels, return_counts=True)
            for label, count in zip(unique, counts):
                total_stats['class_distribution'][int(label)] += int(count)

            print(f"  After filtering - remaining points: {len(points):,}")
            print(f"  Class distribution after mapping:")
            for label, count in zip(unique, counts):
                class_names = {1: 'High Vegetation', 2: 'Power Lines', 3: 'Pylons', 4: 'Others'}
                print(f"    {class_names.get(label, f'Class {label}')}: {count:,} ({100*count/len(labels):.2f}%)")

            # Split into chunks
            chunks = self.split_into_chunks(points, labels)
            total_stats['total_chunks'] += len(chunks)
            print(f"  Created {len(chunks)} chunks")

            # Save each chunk
            for idx, (chunk_points, chunk_labels) in enumerate(chunks):
                sample_dict = {
                    'input_pcd': torch.from_numpy(chunk_points).float(),
                    'semantic_labels': torch.from_numpy(chunk_labels).long().unsqueeze(1),
                    'source_file': las_file.name,
                    'chunk_id': idx,
                    'centroid': centroid
                }
                all_samples.append(sample_dict)

        # Print summary statistics
        print(f"\nDataset Statistics:")
        print(f"  Original total points: {total_stats['total_points_original']:,}")
        print(f"  After filtering: {total_stats['total_points_after_filtering']:,}")
        print(f"  Reduction: {100*(1-total_stats['total_points_after_filtering']/total_stats['total_points_original']):.1f}%")
        print(f"  Total chunks: {total_stats['total_chunks']}")
        print(f"  Final class distribution:")
        class_names = {1: 'High Vegetation', 2: 'Power Lines', 3: 'Pylons', 4: 'Others'}
        for cls_id, count in sorted(total_stats['class_distribution'].items()):
            if count > 0:
                pct = 100 * count / total_stats['total_points_after_filtering']
                print(f"    {class_names[cls_id]}: {count:,} ({pct:.2f}%)")

        # Split into train/test
        n_train = int(len(all_samples) * train_split)
        train_samples = all_samples[:n_train]
        test_samples = all_samples[n_train:]

        # Save
        train_dir = self.output_dir / 'train'
        test_dir = self.output_dir / 'test'
        train_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)

        print(f"\nSaving {len(train_samples)} training samples...")
        for i, sample in enumerate(tqdm(train_samples, desc="Saving train")):
            torch.save(sample, train_dir / f'sample_{i:06d}.pt')

        print(f"Saving {len(test_samples)} test samples...")
        for i, sample in enumerate(tqdm(test_samples, desc="Saving test")):
            torch.save(sample, test_dir / f'sample_{i:06d}.pt')

        # Save metadata with the new class information
        metadata = {
            'n_train': len(train_samples),
            'n_test': len(test_samples),
            'n_classes': 4,
            'class_names': ['High Vegetation', 'Power Lines', 'Pylons', 'Others'],
            'filter_others': self.filter_others,
            'downsample_ratio': self.downsample_ratio,
            'statistics': total_stats
        }

        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\nProcessing complete!")
        print(f"Train samples: {len(train_samples)}")
        print(f"Test samples: {len(test_samples)}")
        print(f"Output directory: {self.output_dir}")

        return metadata

class InverseLASProcessor:
    """Inverse processor to reconstruct original point cloud data from processed PyTorch files"""

    def __init__(self, processed_dir):
        """
        Args:
            processed_dir: Directory containing processed .pt files
        """
        self.processed_dir = Path(processed_dir)

        # Load metadata
        metadata_path = self.processed_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata from {metadata_path}")
        else:
            self.metadata = None
            print("Warning: No metadata.json found")

    def inverse_map_labels(self, mapped_labels):
        """
        Inverse map labels from new classes back to original classes.

        New Classes -> Original Classes:
         1: High Vegetation -> 5
         2: Power Lines -> 10
         3: Pylons -> 11
         4: Others -> 0 (using 0 as placeholder for filtered 'Others')
        """
        original_labels = np.zeros(mapped_labels.shape, dtype=np.int32)

        # Map back to original classes
        original_labels[mapped_labels == 1] = 5   # High Vegetation
        original_labels[mapped_labels == 2] = 10  # Power Lines
        original_labels[mapped_labels == 3] = 11  # Pylons
        original_labels[mapped_labels == 4] = 0   # Others (placeholder)

        return original_labels

    def denormalize_points(self, normalized_points, centroid):
        """Denormalize points by adding back the centroid"""
        return normalized_points + centroid

    def load_and_reconstruct_chunk(self, pt_file_path):
        """
        Load a processed .pt file and reconstruct the original data.

        Returns:
            dict with keys: 'points', 'labels', 'source_file', 'chunk_id', 'centroid'
        """
        sample = torch.load(pt_file_path)

        # Extract data
        normalized_points = sample['input_pcd'].numpy()
        mapped_labels = sample['semantic_labels'].squeeze().numpy()
        centroid = sample['centroid']
        source_file = sample['source_file']
        chunk_id = sample['chunk_id']

        # Inverse transform
        original_points = self.denormalize_points(normalized_points, centroid)
        original_labels = self.inverse_map_labels(mapped_labels)

        return {
            'points': original_points,
            'labels': original_labels,
            'source_file': source_file,
            'chunk_id': chunk_id,
            'centroid': centroid,
            'n_points': len(original_points)
        }

    def reconstruct_all_chunks_from_file(self, source_filename, split='train', percentage=100.0):
        """
        Reconstruct all chunks from a specific source .las file.

        Args:
            source_filename: Name of the original .las file (e.g., 'file.las')
            split: 'train' or 'test'
            percentage: Percentage of chunks to reconstruct (e.g., 10.0 for first 10%)

        Returns:
            Combined point cloud and labels from all chunks
        """
        split_dir = self.processed_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")

        # Find all chunks from this source file
        all_pt_files = sorted(split_dir.glob('*.pt'))
        matching_chunks = []

        print(f"Searching for chunks from {source_filename}...")
        for pt_file in all_pt_files:
            sample = torch.load(pt_file)
            if sample['source_file'] == source_filename:
                matching_chunks.append((sample['chunk_id'], pt_file))

        matching_chunks.sort(key=lambda x: x[0])  # Sort by chunk_id
        print(f"Found {len(matching_chunks)} chunks from {source_filename}")

        if not matching_chunks:
            return None

        # Limit to specified percentage
        if percentage < 100.0:
            n_chunks_to_use = max(1, int(len(matching_chunks) * percentage / 100.0))
            matching_chunks = matching_chunks[:n_chunks_to_use]
            print(f"Using first {percentage}% of chunks: {len(matching_chunks)} chunks")

        # Reconstruct all chunks
        all_points = []
        all_labels = []

        for chunk_id, pt_file in matching_chunks:
            reconstructed = self.load_and_reconstruct_chunk(pt_file)
            all_points.append(reconstructed['points'])
            all_labels.append(reconstructed['labels'])
            print(f"  Chunk {chunk_id}: {reconstructed['n_points']} points")

        # Combine all chunks
        combined_points = np.vstack(all_points)
        combined_labels = np.hstack(all_labels)

        return {
            'points': combined_points,
            'labels': combined_labels,
            'source_file': source_filename,
            'n_chunks': len(matching_chunks),
            'total_points': len(combined_points)
        }

    def compare_with_original(self, reconstructed_data, original_las_path):
        """
        Compare reconstructed data with original .las file to verify processing.

        Args:
            reconstructed_data: Output from reconstruct_all_chunks_from_file()
            original_las_path: Path to original .las file

        Returns:
            Comparison statistics
        """
        print(f"\nComparing with original file: {original_las_path}")

        # Load original
        las = laspy.read(original_las_path)
        original_points = np.vstack((las.x, las.y, las.z)).T

        if hasattr(las, 'classification'):
            original_labels = np.array(las.classification)
        elif hasattr(las, 'label'):
            original_labels = np.array(las.label)
        else:
            raise ValueError("No classification/label field found")

        # Get reconstructed data
        recon_points = reconstructed_data['points']
        recon_labels = reconstructed_data['labels']

        # Statistics
        stats = {
            'original_points': len(original_points),
            'reconstructed_points': len(recon_points),
            'point_ratio': len(recon_points) / len(original_points),
            'original_label_distribution': {},
            'reconstructed_label_distribution': {}
        }

        # Original label distribution
        unique_orig, counts_orig = np.unique(original_labels, return_counts=True)
        for label, count in zip(unique_orig, counts_orig):
            stats['original_label_distribution'][int(label)] = int(count)

        # Reconstructed label distribution
        unique_recon, counts_recon = np.unique(recon_labels, return_counts=True)
        for label, count in zip(unique_recon, counts_recon):
            stats['reconstructed_label_distribution'][int(label)] = int(count)

        # Print comparison
        print(f"\nPoint Count Comparison:")
        print(f"  Original: {stats['original_points']:,}")
        print(f"  Reconstructed: {stats['reconstructed_points']:,}")
        print(f"  Ratio: {stats['point_ratio']:.2%}")

        print(f"\nOriginal Label Distribution:")
        for label, count in sorted(stats['original_label_distribution'].items()):
            pct = 100 * count / stats['original_points']
            print(f"  Class {label}: {count:,} ({pct:.2f}%)")

        print(f"\nReconstructed Label Distribution:")
        class_names = {5: 'High Vegetation', 10: 'Power Lines', 11: 'Pylons', 0: 'Others/Filtered'}
        for label, count in sorted(stats['reconstructed_label_distribution'].items()):
            pct = 100 * count / stats['reconstructed_points']
            name = class_names.get(label, f'Class {label}')
            print(f"  {name} ({label}): {count:,} ({pct:.2f}%)")

        # Check if key classes are preserved
        print(f"\nClass Preservation Check:")
        key_classes = [5, 10, 11]  # High Veg, Power Lines, Pylons
        for cls in key_classes:
            orig_count = stats['original_label_distribution'].get(cls, 0)
            recon_count = stats['reconstructed_label_distribution'].get(cls, 0)
            if orig_count > 0:
                preservation_ratio = recon_count / orig_count
                class_name = class_names.get(cls, f'Class {cls}')
                print(f"  {class_name}: {preservation_ratio:.2%} preserved")

        return stats

    def visualize_sample_chunk(self, pt_file_path):
        """
        Visualize a single chunk's reconstruction for inspection.

        Returns:
            Reconstructed data for visualization
        """
        reconstructed = self.load_and_reconstruct_chunk(pt_file_path)

        print(f"\nChunk Information:")
        print(f"  Source file: {reconstructed['source_file']}")
        print(f"  Chunk ID: {reconstructed['chunk_id']}")
        print(f"  Number of points: {reconstructed['n_points']}")
        print(f"  Centroid: {reconstructed['centroid']}")
        print(f"  Point cloud bounds:")
        print(f"    X: [{reconstructed['points'][:, 0].min():.2f}, {reconstructed['points'][:, 0].max():.2f}]")
        print(f"    Y: [{reconstructed['points'][:, 1].min():.2f}, {reconstructed['points'][:, 1].max():.2f}]")
        print(f"    Z: [{reconstructed['points'][:, 2].min():.2f}, {reconstructed['points'][:, 2].max():.2f}]")

        # Label distribution
        unique, counts = np.unique(reconstructed['labels'], return_counts=True)
        print(f"  Label distribution:")
        class_names = {5: 'High Vegetation', 10: 'Power Lines', 11: 'Pylons', 0: 'Others'}
        for label, count in zip(unique, counts):
            pct = 100 * count / reconstructed['n_points']
            name = class_names.get(label, f'Class {label}')
            print(f"    {name}: {count} ({pct:.2f}%)")

        return reconstructed


if __name__ == "__main__":
    # Process your data with optimizations
    # Option 1: Filter out 'Others' class to focus on relevant classes only
    processor = LASDataProcessor(
        data_dir='../data/behms_corridor',
        output_dir='../data/processed_data_optimized',
        filter_others=False,  # Remove 'Others' class to reduce size
        downsample_ratio=1.0  # Keep all points (or set to 0.5 to keep 50%)
    )

    # Option 2: Keep 'Others' but downsample heavily
    # processor = LASDataProcessor(
    #     data_dir='../data/behms_corridor',
    #     output_dir='../data/processed_data_optimized',
    #     filter_others=False,
    #     downsample_ratio=0.3  # Keep only 30% of points
    # )

    metadata = processor.process_all_files(train_split=0.8, max_files=None)  # None = process all files
    print("\nMetadata:", json.dumps(metadata, indent=2))

    # Example usage of inverse processor
    print("\n" + "="*80)
    print("INVERSE PROCESSING - VERIFICATION")
    print("="*80)

    # Uncomment to test inverse processing
    # inverse_processor = InverseLASProcessor(processed_dir='../data/processed_data_optimized')
    #
    # # Example 1: Visualize a single chunk
    # sample_file = Path('../data/processed_data_optimized/train/sample_000000.pt')
    # if sample_file.exists():
    #     reconstructed_chunk = inverse_processor.visualize_sample_chunk(sample_file)
    #
    # # Example 2: Reconstruct all chunks from a specific source file
    # source_filename = 'your_file.las'  # Replace with actual filename
    # reconstructed_full = inverse_processor.reconstruct_all_chunks_from_file(
    #     source_filename=source_filename,
    #     split='train'
    # )
    #
    # # Example 3: Compare with original
    # if reconstructed_full:
    #     original_las_path = Path('../data/behms_corridor') / source_filename
    #     if original_las_path.exists():
    #         comparison_stats = inverse_processor.compare_with_original(
    #             reconstructed_full,
    #             original_las_path
    #         )
