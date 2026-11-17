# Mix Transformer (SegFormer) Training for KIOS VEPL Vegetation Segmentation

## Overview

This document describes the Mix Transformer (MiT) based training setup for vegetation segmentation, which replaces the VGG16-based U-Net model with SegFormer's transformer architecture specifically designed for semantic segmentation.

## Key Differences from VGG16 Version

### Model Architecture
- **Encoder**: Mix Transformer (MiT) from SegFormer instead of VGG16
- **Default**: `mit_b2` (MiT-B2 - balanced performance and efficiency)
- **Alternatives**:
  - `mit_b0` (smallest, fastest, ~4M params)
  - `mit_b1` (small, ~14M params)
  - `mit_b2` (medium, ~25M params) ⭐ **Recommended**
  - `mit_b3` (large, ~45M params)
  - `mit_b4` (very large, ~62M params)
  - `mit_b5` (largest, highest quality, ~82M params)

### Training Configuration
- **Image Size**: 512x512 (default for MiT, can use 256, 384, or 640)
- **Batch Size**: 32 (reduced from 64 due to higher memory requirements)
- **Memory**: 64GB RAM (increased from 32GB)
- **Time Limit**: 48 hours (increased for transformer training)
- **Weight Decay**: 0.01 (L2 regularization for better generalization)
- **Learning Rate**: 0.00001 (same as VGG16)
- **Encoder**: mit_b2 by default (good balance of speed and accuracy)

### Data Augmentation
- **Normalization**: ImageNet standard (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- **Additional Augmentations**:
  - Vertical flips (p=0.3)
  - Random 90-degree rotations (p=0.3)
  - Hue-Saturation-Value adjustments

### Dependencies
- **No new requirements**: MiT encoders are built into `segmentation-models-pytorch`
- All dependencies are the same as VGG16 version

## Files Created

1. **train_kios_vepl_veg_VT.py**: Main training script with Mix Transformer (SegFormer)
2. **submit_veg_vit_job.sh**: SBATCH job submission script for HPC
3. **VIT_TRAINING_README.md**: This documentation file

## How to Use

### 1. Submit Training Job

From the HPC deployment directory:

```bash
sbatch submit_veg_vit_job.sh
```

### 2. Monitor Training

Check job status:
```bash
squeue -u $USER
```

View output logs:
```bash
tail -f kios_veg_vit_<JOB_ID>.out
```

View error logs:
```bash
tail -f kios_veg_vit_<JOB_ID>.err
```

### 3. Results Location

All outputs will be saved to: `./outputs_veg_vit/`

- **Best Model**: `outputs_veg_vit/best_model_vit.pth`
- **Training History**: `outputs_veg_vit/training_history_vit.json`
- **Training Summary**: `outputs_veg_vit/training_summary_vit.json`
- **Training Logs**: `outputs_veg_vit/training_*.log`

## Customizing Training Parameters

Edit the `submit_veg_vit_job.sh` file to modify:

### Change Model Size
```bash
# For smaller/faster model:
--encoder mit_b0 \
--image_size 384 \
--batch_size 64

# For balanced model (default - recommended):
--encoder mit_b2 \
--image_size 512 \
--batch_size 32

# For larger/better model:
--encoder mit_b5 \
--image_size 512 \
--batch_size 16
```

### Adjust Training Duration
```bash
--epochs 100 \          # Number of epochs
--early_stop_patience 20  # Stop if no improvement for N epochs
```

### Modify Learning Rate Schedule
```bash
--lr 0.00001 \          # Initial learning rate
--lr_patience 10 \      # Reduce LR after N epochs without improvement
--lr_factor 0.5         # Multiply LR by this factor when reducing
```

### Change Batch Size (if memory issues)
```bash
--batch_size 16         # Reduce if OOM errors occur
```

## Expected Performance

### Training Time
- **Per Epoch**: ~10-15 minutes (depends on dataset size)
- **Total Training**: 10-20 hours (with early stopping)

### Memory Usage
- **GPU Memory**: ~10-14 GB (for batch size 32)
- **System RAM**: ~20-30 GB

### Performance Metrics
Mix Transformers (SegFormer) typically achieve:
- **Higher IoU**: +3-7% improvement over VGG16
- **Better Generalization**: Improved performance on test set
- **More Stable Training**: Smoother convergence
- **Better Multi-scale Features**: Hierarchical transformer design

## Troubleshooting

### Out of Memory (OOM) Errors
1. Reduce batch size: `--batch_size 16` or `--batch_size 8`
2. Use smaller image size: `--image_size 384` or `--image_size 256`
3. Use smaller model: `--encoder mit_b0` or `--encoder mit_b1`

### Slow Training
1. Increase number of workers: `--num_workers 16`
2. Use smaller image size: `--image_size 384`
3. Reduce model size: `--encoder mit_b0` or `--encoder mit_b1`

### Poor Performance
1. Increase weight decay: `--weight_decay 0.05`
2. Adjust learning rate: `--lr 0.00005` (higher) or `--lr 0.000005` (lower)
3. Increase augmentation probability
4. Train longer: `--epochs 150`

### Model Loading Issues
If you encounter encoder-related errors, ensure:
```bash
# Check segmentation-models-pytorch version
pip show segmentation-models-pytorch

# Should be version 0.3.3 or higher
# Reinstall if needed:
pip install --user --upgrade segmentation-models-pytorch
```

## Comparing with VGG16 Results

To compare performance:

1. **VGG16 results**: `outputs_veg/training_summary.json`
2. **ViT results**: `outputs_veg_vit/training_summary_vit.json`

Key metrics to compare:
- `test_iou`: Intersection over Union on test set
- `test_dice`: Dice coefficient on test set
- `test_acc`: Pixel-wise accuracy on test set

## Architecture Details

### Mix Transformer (MiT) Encoder
The Mix Transformer is a hierarchical transformer specifically designed for dense prediction tasks like segmentation.

**Key Features:**
- **Hierarchical Structure**: Produces multi-scale features (unlike standard ViT)
- **Efficient Self-Attention**: Reduces computational complexity
- **Overlapped Patch Merging**: Better local continuity than standard patching

**Model Variants:**
- **mit_b0**: ~4M params, 4 stages, smallest and fastest
- **mit_b1**: ~14M params, 4 stages, good for quick experiments
- **mit_b2**: ~25M params, 4 stages, balanced (recommended)
- **mit_b3**: ~45M params, 4 stages, high performance
- **mit_b4**: ~62M params, 4 stages, very high performance
- **mit_b5**: ~82M params, 4 stages, best quality

### Decoder
- U-Net style decoder with skip connections
- Progressive upsampling from transformer features
- Final output: Single-channel binary mask

## References

- SegFormer: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (Xie et al., 2021)
- Paper: https://arxiv.org/abs/2105.15203
- segmentation-models-pytorch: https://github.com/qubvel/segmentation_models.pytorch
- Mix Transformer (MiT) is the encoder backbone from SegFormer, optimized for dense prediction

## Contact

For questions or issues, contact: mfatfa01@ucy.ac.cy
