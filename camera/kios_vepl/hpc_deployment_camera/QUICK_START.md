# Quick Start Guide - Enhanced Training Script

## What Changed?

The training script has been significantly improved to address overfitting and convergence issues identified in previous training runs.

### Key Improvements at a Glance

| Aspect | Before | After |
|--------|--------|-------|
| Test IoU | 70.6% | Expected: >75% |
| Val-Test Gap | 9.0% | Expected: <5% |
| Learning Rate | Fixed at 1e-5 | Dynamic: 1e-4 → 1e-7 |
| Loss Function | BCE only | BCE + Dice |
| Augmentations | Basic (5 types) | Enhanced (10+ types) |
| Regularization | Weight decay | + EMA + Grad clipping |
| Visualizations | None | 3 comprehensive plots |

## Running the Enhanced Script

### Default Settings (Recommended)

```bash
cd camera/kios_vepl/hpc_deployment_camera
python train_kios_vepl_veg_VT.py
```

This uses the optimized defaults:
- Learning rate: 1e-4 (10x higher)
- Batch size: 32
- Epochs: 100 (with early stopping at 10 patience)
- Combined loss: BCE + Dice
- EMA: Enabled
- Enhanced augmentations: Enabled

### Custom Configuration

```bash
# Smaller GPU memory
python train_kios_vepl_veg_VT.py --batch_size 16

# Different encoder
python train_kios_vepl_veg_VT.py --encoder mit_b3

# Faster experimentation
python train_kios_vepl_veg_VT.py --epochs 50 --early_stop_patience 5

# Custom learning rate
python train_kios_vepl_veg_VT.py --lr 0.0005 --warmup_epochs 10
```

## Output Files

The script generates these files in `outputs_veg_vit/`:

### Model Checkpoints
- `best_model_vit.pth` - Best model based on validation IoU


### Visualizations (NEW!)
- `training_curves.png` - Loss, IoU, Dice, LR over time
- `metrics_comparison.png` - Multi-metric analysis
- `predictions_visualization.png` - Sample predictions with error overlay

### Training Data
- `training_history_vit.json` - All metrics per epoch
- `training_summary_vit.json` - Final results and config
- `training_YYYYMMDD_HHMMSS.log` - Detailed training log

## Understanding the Visualizations

### 1. Training Curves (`training_curves.png`)

Four subplots showing:
- **Top-left**: Training loss (should decrease smoothly)
- **Top-right**: Validation IoU (should increase then plateau)
- **Bottom-left**: Validation Dice score (similar to IoU)
- **Bottom-right**: Learning rate schedule (log scale, should show cosine decay)

**What to look for:**
- Smooth, decreasing loss curve
- IoU/Dice plateauing (not decreasing)
- LR following cosine pattern from 1e-4 to 1e-7

### 2. Metrics Comparison (`metrics_comparison.png`)

Three analysis plots:
- **Left**: Loss vs IoU (inversely correlated)
- **Middle**: IoU vs Dice comparison (should track together)
- **Right**: Validation accuracy over time

**What to look for:**
- IoU increasing as loss decreases
- IoU and Dice curves parallel
- Accuracy >97% at convergence

### 3. Predictions Visualization (`predictions_visualization.png`)

8 sample predictions with:
- **Column 1**: Original RGB image
- **Column 2**: Ground truth vegetation mask
- **Column 3**: Model prediction
- **Column 4**: Error overlay:
  - **Green**: True Positives (correct vegetation)
  - **Red**: False Positives (predicted vegetation incorrectly)
  - **Blue**: False Negatives (missed vegetation)

**What to look for:**
- Mostly green (correct predictions)
- Minimal red/blue (few errors)
- Good boundary delineation

## Expected Training Behavior

### Phase 1: Warmup (Epochs 1-5)
- Learning rate increases from ~0 to 1e-4
- Loss decreases rapidly
- IoU increases from ~0.15 to ~0.4-0.5

### Phase 2: Main Training (Epochs 6-40)
- Learning rate follows cosine decay
- Loss continues decreasing steadily
- IoU increases to >0.75
- Best model typically found here

### Phase 3: Fine-tuning (Epochs 41-60)
- Learning rate approaches minimum
- Small improvements in IoU
- May trigger early stopping

### Phase 4: Convergence (Epochs 60+)
- Minimal changes
- Early stopping likely triggered
- Training ends

## Monitoring Progress

During training, watch the console for:

```
Epoch 25/100 | LR: 5.23e-05
Train Loss: 0.1234
Val IoU: 0.7856 | Dice: 0.8734 | Acc: 97.84%
NEW BEST MODEL (IoU: 0.7856)
```

**Good signs:**
- LR decreasing each epoch
- Loss decreasing
- IoU increasing
- "NEW BEST MODEL" messages

**Warning signs:**
- Loss increasing
- IoU decreasing
- No improvement for many epochs

## Comparing with Previous Results

### Previous Training (100 epochs)
- Best Val IoU: **79.6%**
- Test IoU: **70.6%**
- Gap: **9.0%** (overfitting!)
- LR: Constant at 1e-5
- Training time: Full 100 epochs

### Expected with Improvements
- Best Val IoU: **>80%**
- Test IoU: **>75%**
- Gap: **<5%** (less overfitting)
- LR: Dynamic 1e-4 → 1e-7
- Training time: 40-60 epochs (early stopping)

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python train_kios_vepl_veg_VT.py --batch_size 16

# Or reduce image size
python train_kios_vepl_veg_VT.py --image_size 384
```

### Training Too Slow
```bash
# Reduce workers or image size
python train_kios_vepl_veg_VT.py --num_workers 4 --image_size 384
```

### Validation IoU Not Improving
```bash
# Try higher learning rate or more warmup
python train_kios_vepl_veg_VT.py --lr 0.0002 --warmup_epochs 10
```

### Still Overfitting
```bash
# Increase regularization
python train_kios_vepl_veg_VT.py --weight_decay 0.02
```

## Next Steps

1. **Run training** with default settings
2. **Check visualizations** in `outputs_veg_vit/`
3. **Compare results** with previous training
4. **Analyze predictions** for error patterns
5. **Iterate** if needed with custom hyperparameters

## Questions?

See `IMPROVEMENTS_SUMMARY.md` for detailed technical explanations of all changes.
