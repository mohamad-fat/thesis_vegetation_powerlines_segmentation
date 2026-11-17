# Power Line Detection - Enhanced ViT Training Script

## Overview

Enhanced transformer-based training script (`train_kios_vepl_pl_vit.py`) for power line detection using Mix Transformer (MiT) encoder with advanced training techniques.

## Key Features

### Optimized for Power Lines
- **Tversky Loss** (α=0.3, β=0.7) - Emphasizes recall to catch all power lines
- **Focal component** - Focuses on hard-to-detect thin structures  
- **Weighted BCE** (pos_weight=10.0) - Handles extreme class imbalance
- **Higher resolution** (512x512) - Better detection of thin power lines

### Advanced Training Techniques
- Cosine annealing LR scheduler with warmup
- Exponential Moving Average (EMA)
- Gradient clipping (stability)
- Enhanced augmentations
- Comprehensive visualizations

### Improvements Over Base Model

| Aspect | Base (ResNet34) | Enhanced (ViT) |
|--------|-----------------|----------------|
| Encoder | ResNet34 | Mix Transformer (mit_b2) |
| Image Size | 256x256 | 512x512 |
| LR Schedule | ReduceLROnPlateau | Cosine Annealing + Warmup |
| Initial LR | 1e-4 | 1e-4 (with warmup) |
| Regularization | Weight decay | + EMA + Grad clipping |
| Augmentations | Standard | Enhanced (elastic, cutout) |
| Visualizations | None | 3 comprehensive plots |

## Quick Start

### Basic Usage

```bash
cd camera/kios_vepl/hpc_deployment_camera
python train_kios_vepl_pl_vit.py
```

Default configuration:
- Encoder: mit_b2 (27M parameters)
- Image size: 512x512
- Batch size: 32
- Epochs: 100 (with early stopping at 10)
- Learning rate: 1e-4 → 1e-7 (cosine)
- Tversky α=0.3, β=0.7
- BCE pos_weight=10.0

### Custom Configuration

```bash
# Larger model for better performance
python train_kios_vepl_pl_vit.py --encoder mit_b3

# Smaller GPU memory
python train_kios_vepl_pl_vit.py --batch_size 16 --image_size 384

# Adjust class imbalance handling
python train_kios_vepl_pl_vit.py \
    --tversky_alpha 0.2 \
    --tversky_beta 0.8 \
    --bce_pos_weight 15.0

# More aggressive recall (fewer missed power lines)
python train_kios_vepl_pl_vit.py --tversky_beta 0.8

# More precision (fewer false positives)
python train_kios_vepl_pl_vit.py --tversky_alpha 0.5 --tversky_beta 0.5
```

## Understanding Tversky Parameters

### Alpha (FP weight) - Controls Precision
- **Lower (0.1-0.3)**: Reduces false positives (fewer wrong detections)
- **Higher (0.6-0.9)**: Allows more false positives
- **Default: 0.3** - Good balance for power lines

### Beta (FN weight) - Controls Recall  
- **Lower (0.1-0.3)**: Allows missing power lines
- **Higher (0.6-0.9)**: Strongly penalizes missing power lines
- **Default: 0.7** - Emphasizes catching all power lines

### When to Adjust

**Too many false positives** (detecting non-power-lines):
```bash
python train_kios_vepl_pl_vit.py --tversky_alpha 0.2 --tversky_beta 0.8
```

**Missing power lines** (false negatives):
```bash
python train_kios_vepl_pl_vit.py --tversky_alpha 0.4 --tversky_beta 0.6
```

**Extreme class imbalance** (very few power line pixels):
```bash
python train_kios_vepl_pl_vit.py --bce_pos_weight 20.0
```

## Output Files

Generated in `outputs_pl_vit/`:

### Models
- `best_model_pl_vit.pth` - Best model (highest validation IoU)

### Visualizations
- `training_curves.png` - Loss, IoU, Dice, LR schedule
- `metrics_comparison.png` - Multi-metric analysis
- `predictions_visualization.png` - Sample predictions with overlays
  - Green: Correctly detected power lines (TP)
  - Red: False detections (FP)
  - Blue: Missed power lines (FN)

### Data
- `training_history_pl_vit.json` - Metrics per epoch
- `training_summary_pl_vit.json` - Final results
- `training_YYYYMMDD_HHMMSS.log` - Detailed log

## Expected Performance

Based on previous base model results (IoU ~50%), we expect:

- **Validation IoU**: 55-60% (improved by higher resolution + ViT)
- **Test IoU**: 55-60% (reduced overfitting with EMA)
- **Recall**: High (β=0.7 emphasizes catching all power lines)
- **Precision**: Moderate (controlled by α=0.3)

## Mix Transformer Encoders

| Encoder | Parameters | Speed | Memory | Recommended Use |
|---------|-----------|-------|--------|-----------------|
| mit_b0 | 3.7M | Fastest | Low | Quick experiments |
| mit_b1 | 14M | Fast | Medium | Balanced |
| **mit_b2** | 27M | Medium | Medium | **Recommended** |
| mit_b3 | 47M | Slower | High | Best performance |
| mit_b4 | 62M | Slow | High | Research |
| mit_b5 | 82M | Slowest | Very High | Maximum quality |

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train_kios_vepl_pl_vit.py --batch_size 16

# Or use smaller model
python train_kios_vepl_pl_vit.py --encoder mit_b1 --batch_size 16

# Or reduce image size
python train_kios_vepl_pl_vit.py --image_size 384
```

### IoU Not Improving
```bash
# Try higher learning rate with more warmup
python train_kios_vepl_pl_vit.py --lr 0.0002 --warmup_epochs 10

# Or adjust Tversky weights
python train_kios_vepl_pl_vit.py --tversky_beta 0.8
```

### Too Many False Positives
```bash
# Reduce alpha (penalize FP more)
python train_kios_vepl_pl_vit.py --tversky_alpha 0.2

# Or increase BCE weight
python train_kios_vepl_pl_vit.py --tversky_weight 0.6 --bce_weight 0.4
```

### Missing Power Lines
```bash
# Increase beta (penalize FN more)
python train_kios_vepl_pl_vit.py --tversky_beta 0.8

# Or increase pos_weight
python train_kios_vepl_pl_vit.py --bce_pos_weight 15.0
```

## Monitoring Training

### Good Signs
- Learning rate smoothly decreasing (1e-4 → 1e-7)
- Training loss decreasing steadily
- Validation IoU increasing
- "NEW BEST MODEL" messages

### Warning Signs
- Loss increasing or oscillating
- IoU plateauing early (before epoch 20)
- Many epochs without improvement

### Visualization Analysis

**training_curves.png**:
- LR should follow smooth cosine curve (log scale)
- Loss should decrease monotonically
- IoU should plateau (not decrease)

**predictions_visualization.png**:
- Mostly green (correct detections)
- Some red acceptable (false positives)
- Minimal blue desired (missed power lines)

## Comparison with Vegetation Model

The power line model differs from vegetation model:

1. **Loss Function**: Tversky (not Dice) - better for thin structures
2. **Class Balance**: More extreme (fewer power line pixels)
3. **α/β weights**: Emphasizes recall (catching all power lines)
4. **pos_weight**: Higher (10.0 vs 1.0) for class imbalance

## Next Steps

1. Run training with defaults
2. Check visualizations in `outputs_pl_vit/`
3. Analyze prediction errors (red/blue regions)
4. Adjust Tversky α/β if needed
5. Experiment with different encoders (mit_b3 for better performance)

## References

- See `IMPROVEMENTS_SUMMARY.md` for vegetation model (similar architecture)
- Tversky Loss: Balances precision/recall for imbalanced data
- Mix Transformer: SegFormer backbone for vision tasks
