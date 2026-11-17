# HPC Job Submission Guide

## Available Training Scripts

### 1. Vegetation Detection
- **Base Model**: `submit_veg_job.sh` - ResNet34 encoder
- **Enhanced ViT**: `submit_veg_vit_job.sh` - Mix Transformer encoder

### 2. Power Line Detection  
- **Base Model**: `submit_pl_job.sh` - ResNet34 encoder
- **Enhanced ViT**: `submit_pl_vit_job.sh` - Mix Transformer encoder ✨ NEW

## Quick Start

### Submit Power Line ViT Job

```bash
# Navigate to deployment directory
cd camera/kios_vepl/hpc_deployment_camera

# Submit job
sbatch submit_pl_vit_job.sh
```

### Monitor Job

```bash
# Check job status
squeue -u $USER

# View output (while running)
tail -f kios_pl_vit_<JOB_ID>.out

# View errors (if any)
tail -f kios_pl_vit_<JOB_ID>.err
```

### Check Results

```bash
# After completion
ls -lh outputs_pl_vit/

# View summary
cat outputs_pl_vit/training_summary_pl_vit.json

# Download visualizations
scp user@hpc:path/to/outputs_pl_vit/*.png .
```

## Configuration Comparison

| Feature | Base (ResNet34) | Enhanced (ViT) |
|---------|----------------|----------------|
| **Script** | `submit_pl_job.sh` | `submit_pl_vit_job.sh` |
| **Encoder** | ResNet34 | mit_b2 (27M params) |
| **Image Size** | 256×256 | 512×512 |
| **Batch Size** | 64 | 32 |
| **LR Schedule** | ReduceLROnPlateau | Cosine + Warmup |
| **Initial LR** | 1e-4 | 1e-4 |
| **Early Stop** | 15 epochs | 10 epochs |
| **Regularization** | Weight decay | + EMA + Grad clip |
| **Augmentations** | Standard | Enhanced |
| **Visualizations** | None | 3 plots |
| **Runtime** | ~4-6 hours | ~8-12 hours |

## Resource Allocation

All scripts request:
- **GPU**: 1× V100 (16GB)
- **CPUs**: 8 cores
- **Memory**: 64GB RAM
- **Time**: 48 hours (max)
- **Partition**: GPU

## Customizing Job Parameters

### Modify SLURM Settings

Edit the `#SBATCH` directives in the script:

```bash
# For longer training
#SBATCH --time=72:00:00

# For multiple GPUs (if available)
#SBATCH --gres=gpu:v100:2

# For more memory
#SBATCH --mem=128G
```

### Modify Training Parameters

Edit the `python train_kios_vepl_pl_vit.py` command:

```bash
# Smaller batch (if OOM)
--batch_size 16

# Different encoder
--encoder mit_b3

# Adjust Tversky for more recall
--tversky_beta 0.8

# Higher BCE weight for class imbalance
--bce_pos_weight 15.0

# More warmup epochs
--warmup_epochs 10
```

## Common Issues

### Out of Memory (OOM)

**Symptoms**: Job fails with CUDA OOM error

**Solutions**:
```bash
# Option 1: Reduce batch size
--batch_size 16

# Option 2: Reduce image size
--image_size 384

# Option 3: Use smaller encoder
--encoder mit_b1
```

### Job Timeout

**Symptoms**: Job killed after 48 hours

**Solutions**:
```bash
# Increase time limit
#SBATCH --time=72:00:00

# Or reduce epochs
--epochs 50

# Or increase early stopping
--early_stop_patience 15
```

### Data Not Found

**Symptoms**: "ERROR: Processed data not found!"

**Solutions**:
```bash
# Verify data exists
ls processed_data/images/
ls processed_data/pl_masks/

# Check counts match
ls processed_data/images/*.png | wc -l
ls processed_data/pl_masks/*.png | wc -l
```

### Dependencies Fail

**Symptoms**: ImportError or module not found

**Solutions**:
```bash
# Install manually before job
pip install --user segmentation-models-pytorch==0.3.3
pip install --user albumentations matplotlib

# Or add to job script
```

## Output Files

### Standard Output
- `kios_pl_vit_<JOB_ID>.out` - Training logs, metrics
- `kios_pl_vit_<JOB_ID>.err` - Error messages

### Training Outputs (in `outputs_pl_vit/`)

**Models**:
- `best_model_pl_vit.pth` - Best model checkpoint

**Visualizations**:
- `training_curves.png` - Loss, IoU, Dice, LR
- `metrics_comparison.png` - Multi-metric plots  
- `predictions_visualization.png` - Sample predictions

**Data**:
- `training_history_pl_vit.json` - All epochs
- `training_summary_pl_vit.json` - Final results
- `training_*.log` - Detailed log

## Job Management Commands

```bash
# Submit job
sbatch submit_pl_vit_job.sh

# Check queue
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# Check past jobs
sacct -u $USER

# Job details
scontrol show job <JOB_ID>

# Node info
sinfo -p GPU
```

## Performance Expectations

### Power Line Detection (Base)
- Val IoU: ~48%
- Test IoU: ~52%
- Training time: 4-6 hours
- Epochs: Usually 60-80

### Power Line Detection (ViT - Expected)
- Val IoU: ~55-60%
- Test IoU: ~55-60%
- Training time: 8-12 hours  
- Epochs: Usually 40-60 (earlier convergence)

## Tips for Success

1. **Test Locally First** (if possible)
   - Verify data loading works
   - Test with 1-2 epochs
   - Check memory usage

2. **Monitor Early**
   - Watch first few epochs
   - Verify metrics are reasonable
   - Cancel if issues detected

3. **Save Intermediate Results**
   - Model saves best automatically
   - Visualizations generated at end
   - History saved per epoch

4. **Adjust Hyperparameters**
   - Start with defaults
   - Tune based on results
   - Document changes

5. **Resource Optimization**
   - Use appropriate batch size
   - Don't request excessive time
   - Release resources when done

## Next Steps After Training

1. **Check Results**
   ```bash
   cat outputs_pl_vit/training_summary_pl_vit.json
   ```

2. **Download Visualizations**
   ```bash
   scp user@hpc:outputs_pl_vit/*.png ./local_dir/
   ```

3. **Analyze Predictions**
   - Look at `predictions_visualization.png`
   - Check for systematic errors
   - Adjust Tversky α/β if needed

4. **Compare with Base**
   ```bash
   # Compare IoU scores
   grep "test_iou" output_pl/training_summary.json
   grep "test_iou" outputs_pl_vit/training_summary_pl_vit.json
   ```

5. **Iterate if Needed**
   - Adjust hyperparameters
   - Try different encoder
   - Modify augmentations

## Support

For issues:
1. Check error logs: `kios_pl_vit_<JOB_ID>.err`
2. Review training log: `outputs_pl_vit/training_*.log`
3. Verify data integrity
4. Check resource usage with `sacct`

For documentation:
- See `PL_VIT_README.md` for detailed usage
- See `train_kios_vepl_pl_vit.py` for code
- See previous job outputs for examples
