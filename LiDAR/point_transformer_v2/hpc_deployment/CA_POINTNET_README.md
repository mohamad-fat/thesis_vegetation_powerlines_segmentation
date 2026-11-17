# CA-PointNet++ Implementation

## Overview

This implementation is based on the paper **"Semantic Segmentation of Transmission Corridor 3D Point Clouds Based on CA-PointNet++"**. The model introduces **Coordinate Attention (CA)** modules into the standard PointNet++ architecture to enhance feature extraction for semantic segmentation of power transmission infrastructure.

## Architecture

### Key Components

1. **Coordinate Attention (CA) Module**
   - Captures long-range dependencies by encoding channels along spatial dimensions
   - Inserted into each Set Abstraction layer before max pooling
   - Operations:
     - Global average pooling along two directions (height and width)
     - Concatenation and 1x1 convolution transformation
     - Hard Swish activation
     - Split and generate attention weights via Sigmoid
     - Reweight original features

2. **Encoder (4 Set Abstraction Layers with CA)**

   | Layer | Input Points | Output Points | Radius | Neighbors | Output Channels |
   |-------|--------------|---------------|--------|-----------|-----------------|
   | SA1   | 4096         | 1024          | 0.1    | 32        | 64              |
   | SA2   | 1024         | 256           | 0.2    | 32        | 128             |
   | SA3   | 256          | 64            | 0.4    | 32        | 256             |
   | SA4   | 64           | 16            | 0.8    | 32        | 512             |

3. **Decoder (4 Feature Propagation Layers)**
   - FP4: Upsamples 16 → 64 points
   - FP3: Upsamples 64 → 256 points
   - FP2: Upsamples 256 → 1024 points
   - FP1: Upsamples 1024 → 4096 points
   - Uses Inverse Distance Weighting (IDW) and skip connections

4. **Classifier**
   - Two Conv1D layers with BatchNorm and Dropout
   - Final output: 4 classes per point

## Training Configuration (As Per Paper)

### Model Input
- **Number of Points**: 4096
- **Input Channels**: 6 (XYZ coordinates + RGB colors)

### Hyperparameters
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Max Epochs**: 100
- **LR Scheduler**: StepLR (step_size=20, gamma=0.7)
- **Gradient Clipping**: 1.0

### Classes (4 Total)
1. **Tower** (Pylons)
2. **Transmission Line** (Power Lines)
3. **Ground Wire**
4. **Ground** (includes vegetation)

## Class Weights

To address severe class imbalance:
```python
class_weights = [50.0, 10.0, 50.0, 1.0]
# [Tower, Transmission Line, Ground Wire, Ground]
```

- **Tower/Pylon**: 50.0 (extremely rare: ~0.01%)
- **Transmission Line**: 10.0 (very rare: ~0.2%)
- **Ground Wire**: 50.0 (extremely rare)
- **Ground**: 1.0 (abundant: ~40%)

## Usage

### Local Training
```bash
python train_ca_pointnet_hpc.py \
    --data_dir ../data/processed_data_optimized \
    --output_dir ./outputs_ca_pointnet \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --momentum 0.9 \
    --num_points 4096
```

### HPC Training (SLURM)
```bash
# Submit job to SLURM
sbatch submit_ca_pointnet_job.sh

# Check job status
squeue -u $USER

# Monitor output
tail -f logs/ca_pointnet_<job_id>.out

# Check for errors
tail -f logs/ca_pointnet_<job_id>.err
```

## Files

- `train_ca_pointnet_hpc.py` - Main training script
- `submit_ca_pointnet_job.sh` - SLURM batch submission script
- `CA_POINTNET_README.md` - This file

## Output Structure

```
outputs_ca_pointnet/
├── checkpoints/
│   └── best_model.ckpt          # Best model checkpoint (highest val_miou)
├── logs/
│   └── ca_pointnet/
│       └── version_X/
│           └── events.out.tfevents.*  # TensorBoard logs
├── training_history.json         # Per-epoch metrics
└── training_summary.json         # Final summary with architecture details
```

## Key Differences from Point Transformer V2

| Feature | CA-PointNet++ | Point Transformer V2 |
|---------|---------------|---------------------|
| **Architecture** | PointNet++ + CA modules | Transformer blocks |
| **Attention Mechanism** | Coordinate Attention | Self-attention |
| **Input Channels** | 6 (XYZ + RGB) | 3 (XYZ only) |
| **Sampling Strategy** | Farthest Point Sampling (FPS) | FPS + KNN |
| **Grouping** | Ball query | KNN |
| **Paper Reference** | Transmission corridor segmentation | General point cloud tasks |

## RGB Handling

**Note**: Since the current dataset does not contain real RGB values, the implementation generates **synthetic RGB colors** based on semantic labels for training:

- **Tower/Pylon**: Gray (0.5, 0.5, 0.5)
- **Transmission Line**: Yellow (1.0, 1.0, 0.0)
- **Ground Wire**: Orange (1.0, 0.5, 0.0)
- **Ground/Vegetation**: Green (0.2, 0.8, 0.2)

Small Gaussian noise is added for variation. If real RGB data becomes available, modify the `generate_synthetic_rgb()` method in `PowerLineDatasetRGB` class.

## Expected Performance

Based on the paper's results on transmission corridor data:
- **Overall Accuracy**: ~95%+
- **Mean IoU**: Target > 0.70
- **Per-class IoU**:
  - Ground: High (due to abundance)
  - Tower: Moderate (improved with high weight)
  - Transmission Line: Moderate (improved with CA module)
  - Ground Wire: Challenging (very sparse)

## Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs_ca_pointnet/logs
```

Access at: http://localhost:6006

### Metrics Logged
- `train_loss` - Training cross-entropy loss
- `train_acc` - Training per-point accuracy
- `val_loss` - Validation loss
- `val_acc` - Validation accuracy
- `val_miou` - Validation mean IoU (primary metric)

## Citation

If you use this implementation, please cite the original paper:

```
@article{ca_pointnet_2023,
  title={Semantic Segmentation of Transmission Corridor 3D Point Clouds Based on CA-PointNet++},
  author={[Authors]},
  journal={[Journal]},
  year={2023}
}
```

## Troubleshooting

### Out of Memory (OOM) Errors
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--num_points` (try 2048)

### Poor Performance on Rare Classes
- Increase class weights for rare classes
- Ensure sufficient training samples with rare classes
- Consider data augmentation (rotation, scaling, jittering)

### Training Too Slow
- Increase `--num_workers` (up to CPU cores available)
- Enable `cudnn.benchmark` (already enabled by default)
- Reduce `--num_points` if accuracy permits

### Model Not Converging
- Check learning rate (try 0.0001 if 0.001 is too high)
- Verify class weights are appropriate
- Check for NaN/Inf in loss (enable gradient clipping)
- Ensure data preprocessing is correct

## Hardware Requirements

### Minimum
- GPU: NVIDIA V100 (16GB VRAM) or equivalent
- RAM: 32GB
- CPUs: 8 cores

### Recommended
- GPU: NVIDIA A100 (40GB VRAM)
- RAM: 64GB
- CPUs: 16 cores

## Notes

1. **Coordinate Attention** provides better spatial feature encoding than standard PointNet++
2. **RGB features** enhance geometric features, especially for distinguishing similar structures
3. **Class weights** are critical for handling extreme imbalance in transmission corridor data
4. **Early stopping** prevents overfitting (patience=15 epochs)
5. **Gradient clipping** ensures training stability

## Future Improvements

- [ ] Add real RGB data from multispectral sensors
- [ ] Implement multi-scale testing
- [ ] Add test-time augmentation (TTA)
- [ ] Experiment with focal loss for imbalanced classes
- [ ] Add confusion matrix visualization
- [ ] Implement sliding window inference for large point clouds
