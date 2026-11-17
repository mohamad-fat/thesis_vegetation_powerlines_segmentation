"""
Step 4: Complete Training Script
Run this after preprocessing your .las files
"""
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our modules (make sure they're in the same directory or installed)
from powerline_dataset import PowerLineDataModule
from point_transformer_model import PointTransformerLightning


def calculate_class_weights(data_dir='processed_data'):
    """
    Calculate class weights based on dataset statistics
    Paper uses weighted cross entropy for class imbalance
    """
    from torch.utils.data import DataLoader
    from powerline_dataset import PowerLineDataset
    
    print("Calculating class weights from training data...")
    dataset = PowerLineDataset(data_dir, split='train', use_fps=False)
    
    # Count class occurrences
    class_counts = torch.zeros(6)
    
    for i in range(min(len(dataset), 100)):  # Sample first 100
        sample = dataset[i]
        labels = sample['labels']
        for cls in range(6):
            class_counts[cls] += (labels == cls).sum()
    
    # Calculate weights (inverse frequency)
    class_weights = class_counts.sum() / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * 6  # Normalize
    
    print("Class counts:", class_counts.numpy())
    print("Class weights:", class_weights.numpy())
    
    return class_weights.tolist()


def train_model(
    data_dir='processed_data',
    batch_size=8,
    num_points=4096,
    max_epochs=200,
    lr=1e-3,
    d_model=256,
    num_layers=4,
    gpus=1,
    calculate_weights=True,
    checkpoint_dir='checkpoints'
):
    """
    Main training function
    
    Args:
        data_dir: Directory with processed data
        batch_size: Batch size for training
        num_points: Number of points per sample
        max_epochs: Maximum training epochs
        lr: Learning rate
        d_model: Model dimension
        num_layers: Number of transformer layers
        gpus: Number of GPUs to use (0 for CPU)
        calculate_weights: Whether to calculate class weights
        checkpoint_dir: Directory to save checkpoints
    """
    
    # Set up paths
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Calculate class weights if needed
    if calculate_weights:
        class_weights = calculate_class_weights(data_dir)
    else:
        # Default weights favoring underrepresented classes
        class_weights = [1.0, 1.0, 1.0, 2.0, 5.0, 5.0]
    
    print(f"\nUsing class weights: {class_weights}")
    
    # Data module
    print("\nSetting up data...")
    dm = PowerLineDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_points=num_points,
        num_workers=4,
        use_fps=True  # FPS as in paper
    )
    
    # Model
    print("Initializing model...")
    model = PointTransformerLightning(
        num_classes=6,
        d_model=d_model,
        num_layers=num_layers,
        lr=lr,
        weight_decay=1e-4,
        class_weights=class_weights,
        max_epochs=max_epochs
    )
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='pointtransformer-{epoch:02d}-{val_miou:.3f}',
        monitor='val_miou',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_miou',
        patience=20,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Logger
    logger = TensorBoardLogger('lightning_logs', name='point_transformer')
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        precision='16-mixed' if gpus > 0 else '32',  # Mixed precision for speed
        gradient_clip_val=1.0,
    )
    
    # Train
    print("\nStarting training...")
    print(f"Batch size: {batch_size}")
    print(f"Number of points: {num_points}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Model dimension: {d_model}")
    print(f"Number of layers: {num_layers}\n")
    
    trainer.fit(model, dm)
    
    # Test
    print("\nTesting best model...")
    trainer.test(model, dm, ckpt_path='best')
    
    print(f"\nTraining complete! Best model saved to {checkpoint_dir}")
    print(f"TensorBoard logs saved to lightning_logs/")
    print(f"To view logs, run: tensorboard --logdir=lightning_logs")
    
    return model, trainer


def visualize_predictions(
    model_path,
    data_dir='processed_data',
    num_samples=5,
    output_dir='predictions'
):
    """
    Visualize model predictions
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = PointTransformerLightning.load_from_checkpoint(model_path)
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test data
    from powerline_dataset import PowerLineDataset
    dataset = PowerLineDataset(data_dir, split='test', use_fps=False)
    
    class_names = ['Noise', 'Ground', 'Low Veg', 'Med Veg', 'Tower', 'Power Line']
    colors = ['gray', 'brown', 'lightgreen', 'darkgreen', 'blue', 'orange']
    
    # Predict on random samples
    for i in range(min(num_samples, len(dataset))):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        
        points = sample['points'].unsqueeze(0).to(model.device)
        labels = sample['labels'].numpy()
        
        # Predict
        with torch.no_grad():
            logits = model(points)
            pred = logits.argmax(dim=-1).cpu().numpy()[0]
        
        # Visualize
        fig = plt.figure(figsize=(15, 5))
        
        # Ground truth
        ax1 = fig.add_subplot(121, projection='3d')
        points_np = sample['points'].numpy()
        for cls in range(6):
            mask = labels == cls
            if mask.sum() > 0:
                ax1.scatter(
                    points_np[mask, 0],
                    points_np[mask, 1],
                    points_np[mask, 2],
                    c=colors[cls],
                    label=class_names[cls],
                    s=1
                )
        ax1.set_title('Ground Truth')
        ax1.legend()
        
        # Prediction
        ax2 = fig.add_subplot(122, projection='3d')
        for cls in range(6):
            mask = pred == cls
            if mask.sum() > 0:
                ax2.scatter(
                    points_np[mask, 0],
                    points_np[mask, 1],
                    points_np[mask, 2],
                    c=colors[cls],
                    label=class_names[cls],
                    s=1
                )
        ax2.set_title('Prediction')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/sample_{i:03d}.png', dpi=150)
        plt.close()
        
        print(f"Saved prediction visualization {i+1}/{num_samples}")
    
    print(f"\nVisualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Configuration
    CONFIG = {
        'data_dir': 'processed_data',
        'batch_size': 8,
        'num_points': 4096,
        'max_epochs': 200,
        'lr': 1e-3,
        'd_model': 256,
        'num_layers': 4,
        'gpus': 1,  # Set to 0 for CPU training
        'calculate_weights': True,
        'checkpoint_dir': 'checkpoints'
    }
    
    print("=" * 60)
    print("POWER LINE POINT CLOUD SEGMENTATION TRAINING")
    print("Model: Point Transformer V2 (Best from TS40K paper)")
    print("=" * 60)
    
    # Train
    model, trainer = train_model(**CONFIG)
    
    # Visualize predictions
    best_model_path = Path(CONFIG['checkpoint_dir']) / 'last.ckpt'
    if best_model_path.exists():
        print("\nGenerating prediction visualizations...")
        visualize_predictions(
            model_path=str(best_model_path),
            data_dir=CONFIG['data_dir'],
            num_samples=5
        )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)