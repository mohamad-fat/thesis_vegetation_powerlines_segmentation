"""
Step 2: PyTorch Dataset and Lightning DataModule
"""
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from pathlib import Path
import json
import numpy as np

class PowerLineDataset(Dataset):
    """Dataset for power line point cloud segmentation"""
    
    def __init__(self, data_dir, split='train', num_points=4096, use_fps=True):
        """
        Args:
            data_dir: Directory containing processed .pt files
            split: 'train' or 'test'
            num_points: Number of points per sample (will use FPS if more)
            use_fps: Use Farthest Point Sampling for downsampling
        """
        self.data_dir = Path(data_dir) / split
        self.num_points = num_points
        self.use_fps = use_fps
        self.split = split
        
        # Get all sample files
        self.samples = sorted(list(self.data_dir.glob('*.pt')))
        
        if not self.samples:
            raise ValueError(f"No samples found in {self.data_dir}")
        
        print(f"Loaded {len(self.samples)} samples from {split} split")
        
        # Load metadata
        metadata_path = Path(data_dir) / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'n_classes': 6}
    
    def farthest_point_sampling(self, points, num_samples):
        """
        Farthest Point Sampling (FPS) - maintains geometry
        From the paper: Used to address class imbalance
        """
        N, D = points.shape
        centroids = np.zeros(num_samples, dtype=np.int32)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N)
        
        for i in range(num_samples):
            centroids[i] = farthest
            centroid = points[farthest, :]
            dist = np.sum((points - centroid) ** 2, axis=1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        
        return centroids
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Load sample
        sample = torch.load(self.samples[idx])
        
        points = sample['input_pcd']  # (N, 3)
        labels = sample['semantic_labels'].squeeze()  # (N,)
        
        # Resample to fixed number of points
        n_points = points.shape[0]
        
        if n_points > self.num_points:
            if self.use_fps:
                # Use FPS as in the paper
                fps_idx = self.farthest_point_sampling(points.numpy(), self.num_points)
                points = points[fps_idx]
                labels = labels[fps_idx]
            else:
                # Random sampling
                choice = np.random.choice(n_points, self.num_points, replace=False)
                points = points[choice]
                labels = labels[choice]
        elif n_points < self.num_points:
            # Pad with random sampling
            choice = np.random.choice(n_points, self.num_points - n_points, replace=True)
            extra_points = points[choice]
            extra_labels = labels[choice]
            points = torch.cat([points, extra_points], dim=0)
            labels = torch.cat([labels, extra_labels], dim=0)
        
        return {
            'points': points,  # (num_points, 3)
            'labels': labels,  # (num_points,)
            'idx': idx
        }


class PowerLineDataModule(L.LightningDataModule):
    """Lightning DataModule for power line segmentation"""
    
    def __init__(
        self,
        data_dir='processed_data',
        batch_size=8,
        num_points=4096,
        num_workers=4,
        use_fps=True
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_points = num_points
        self.num_workers = num_workers
        self.use_fps = use_fps
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PowerLineDataset(
                self.data_dir,
                split='train',
                num_points=self.num_points,
                use_fps=self.use_fps
            )
            
        if stage == 'test' or stage is None:
            self.test_dataset = PowerLineDataset(
                self.data_dir,
                split='test',
                num_points=self.num_points,
                use_fps=False  # No FPS for testing
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return self.test_dataloader()


# Test the dataset
if __name__ == "__main__":
    # Create datamodule
    dm = PowerLineDataModule(
        data_dir='processed_data',
        batch_size=4,
        num_points=4096
    )
    
    dm.setup('fit')
    
    # Test train loader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    
    print("Batch keys:", batch.keys())
    print("Points shape:", batch['points'].shape)
    print("Labels shape:", batch['labels'].shape)
    print("Unique labels:", torch.unique(batch['labels']))