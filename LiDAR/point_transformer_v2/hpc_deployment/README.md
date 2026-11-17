# Point Transformer V2 HPC Deployment

Complete HPC deployment for training Point Transformer V2 on LiDAR point cloud data for semantic segmentation of power line infrastructure.

## 🎯 Overview

This deployment trains a deep learning model to segment 4 classes in LiDAR point clouds:
- **Class 0**: High Vegetation
- **Class 1**: Power Lines
- **Class 2**: Pylons
- **Class 3**: Others

### Model Architecture
- **Point Transformer V2** - Transformer-based architecture for 3D point clouds
- **GPU-Optimized**: Configured for V100 GPU (batch_size=16, num_points=8192)
- **Lightweight**: ~1.5M parameters (d_model=256, num_layers=4)

---

## 📁 Files

```
hpc_deployment/
├── train_point_transformer_hpc.py     # Main training script
├── assemble_predictions_hpc.py        # Full test set assembly & visualization
├── visualize_results.py               # Training metrics visualization
├── submit_lidar_job.sh                # SLURM job submission script
├── prepare_for_hpc.sh                 # Packaging script
└── README.md                          # This file
```

---

## 🚀 Quick Start

### 1. Prepare Deployment Package

```bash
# On local machine
cd /path/to/LiDAR/point_transformer_v2/hpc_deployment
bash prepare_for_hpc.sh
```

### 2. Transfer to HPC

```bash
rsync -avz --progress lidar_hpc_deployment.tar.gz mfatfa01@turing1.ucy.ac.cy:~/
```

### 3. Setup on HPC

```bash
# SSH to HPC
ssh mfatfa01@turing1.ucy.ac.cy

# Extract files
cd ~
tar -xzf lidar_hpc_deployment.tar.gz
mkdir -p hpc_deployment
mv train_point_transformer_hpc.py hpc_deployment/
mv assemble_predictions_hpc.py hpc_deployment/
mv visualize_results.py hpc_deployment/
mv submit_lidar_job.sh hpc_deployment/
mv README.md hpc_deployment/
cd hpc_deployment
```

### 4. Verify Data

```bash
# Check training data
ls -lh ../data/processed_data_optimized/train/ | head

# Check test data
ls -lh ../data/processed_data_optimized/test/ | head

# Count chunks
echo "Train chunks: $(ls ../data/processed_data_optimized/train/*.pt | wc -l)"
echo "Test chunks: $(ls ../data/processed_data_optimized/test/*.pt | wc -l)"
```

### 5. Submit Training Job

```bash
sbatch submit_lidar_job.sh
```

### 6. Monitor Training

```bash
# Check job status
squeue -u $USER

# Watch output in real-time
tail -f lidar_train_<job_id>.out

# Check for errors
tail -f lidar_train_<job_id>.err
```

---

## 📊 Training Configuration

### Default Parameters (GPU-Optimized)

```python
Batch Size:       16        # Optimized for V100 GPU
Epochs:           50        # With early stopping (patience=15)
Learning Rate:    0.001     # With cosine annealing
Points/Sample:    8192      # Increased from 4096 for better coverage
Model Dimension:  256       # d_model
Transformer Layers: 4       # Increased from 3
K-Nearest:        16        # For local attention
Dropout:          0.3       # Regularization
Gradient Clip:    1.0       # Stability
```

### Hardware Specifications

```bash
Partition:     GPU
GPU:           1x NVIDIA V100 (16GB)
CPUs:          8 cores
RAM:           64GB
Time Limit:    48 hours
```

---

## 📈 Output Structure

After training completes, you'll have:

```
outputs_lidar/
├── checkpoints/
│   └── best_model.ckpt              # Best model (highest val_miou)
├── logs/
│   └── point_transformer/
│       └── version_0/               # TensorBoard logs
├── visualizations/
│   ├── loss_curves.png              # Training/validation loss
│   ├── accuracy_curves.png          # Training/validation accuracy
│   ├── miou_curve.png               # Validation mIoU over time
│   └── training_summary.png         # Combined view
├── full_reconstructions/
│   ├── full_pointcloud_topview.png  # Top-down view
│   ├── full_pointcloud_3d.png       # 3D visualization
│   ├── powerlines_focus.png         # Power infrastructure focus
│   ├── full_pointcloud.npy          # Complete point cloud data
│   └── full_metrics.txt             # Full reconstruction metrics
└── training_summary.json            # Training metadata
```

---

## 🔍 Key Metrics

### Primary Metric: **Mean IoU (mIoU)**
- Intersection over Union averaged across all classes
- Best model is saved based on highest validation mIoU

### Per-Class Metrics:
- **IoU**: Intersection over Union
- **Precision**: Correct predictions / All predictions
- **Recall**: Correct predictions / All ground truth
- **F1-Score**: Harmonic mean of precision and recall

---

## 💡 Customizing Training

### Modify Hyperparameters

Edit `submit_lidar_job.sh` and change the training command:

```bash
python train_point_transformer_hpc.py \
    --data_dir ../data/processed_data_optimized \
    --output_dir ./outputs_lidar \
    --batch_size 32 \              # Increase if you have more GPU memory
    --epochs 100 \                 # Train longer
    --lr 0.0005 \                  # Adjust learning rate
    --num_points 16384 \           # More points per sample
    --d_model 512 \                # Larger model
    --num_layers 6 \               # Deeper model
    --early_stop_patience 20       # More patience
```

### Available Arguments

```
--data_dir              Path to processed data (default: ../data/processed_data_optimized)
--output_dir            Output directory (default: ./outputs_lidar)
--batch_size            Batch size (default: 16)
--epochs                Max epochs (default: 50)
--lr                    Learning rate (default: 0.001)
--weight_decay          L2 regularization (default: 0.0001)
--num_points            Points per sample (default: 8192)
--d_model               Model dimension (default: 256)
--num_layers            Transformer layers (default: 4)
--k_neighbors           KNN neighbors (default: 16)
--dropout               Dropout rate (default: 0.3)
--num_workers           DataLoader workers (default: 8)
--early_stop_patience   Early stop patience (default: 15)
--gradient_clip         Gradient clipping (default: 1.0)
--max_train_samples     Limit training samples (default: None)
```

---

## 🐛 Troubleshooting

### Job Won't Start

```bash
# Check job queue
squeue -u $USER

# Check job details
scontrol show job <job_id>

# Check partition availability
sinfo -p GPU
```

### Out of Memory Error

If you see CUDA out of memory errors:

1. **Reduce batch size**:
   ```bash
   --batch_size 8
   ```

2. **Reduce points per sample**:
   ```bash
   --num_points 4096
   ```

3. **Reduce model size**:
   ```bash
   --d_model 128 --num_layers 3
   ```

### Data Not Found

```bash
# Check data directory structure
ls -R ../data/processed_data_optimized/

# Should have:
# ../data/processed_data_optimized/train/*.pt
# ../data/processed_data_optimized/test/*.pt
# ../data/processed_data_optimized/metadata.json
```

### Dependencies Not Installing

If pip install fails, try:

```bash
# Load Python module first
module use /share/apps/eb/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Then install manually
pip install --user lightning==2.5.0
pip install --user tensorboard matplotlib scikit-learn tqdm
```

---

## 📥 Downloading Results

### Download Entire Output Directory

```bash
# On local machine
rsync -avz --progress mfatfa01@turing1.ucy.ac.cy:hpc_deployment/outputs_lidar/ ./results_lidar/
```

### Download Only Visualizations

```bash
rsync -avz --progress mfatfa01@turing1.ucy.ac.cy:hpc_deployment/outputs_lidar/visualizations/ ./visualizations/
rsync -avz --progress mfatfa01@turing1.ucy.ac.cy:hpc_deployment/outputs_lidar/full_reconstructions/ ./reconstructions/
```

### Download Only Best Model

```bash
rsync -avz --progress mfatfa01@turing1.ucy.ac.cy:hpc_deployment/outputs_lidar/checkpoints/best_model.ckpt ./
```

---

## 🔬 Using the Trained Model

### Load Model for Inference

```python
import torch
from train_point_transformer_hpc import PointTransformerV2

# Load checkpoint
checkpoint = torch.load('best_model.ckpt')
hparams = checkpoint['hyper_parameters']

# Create model
model = PointTransformerV2(
    num_classes=hparams['num_classes'],
    d_model=hparams['d_model'],
    num_layers=hparams['num_layers'],
    k=hparams.get('k', 16)
)

# Load weights
state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()
              if k.startswith('model.')}
model.load_state_dict(state_dict)
model.eval()

# Predict
with torch.no_grad():
    logits = model(points)  # points: [B, N, 3]
    predictions = logits.argmax(dim=-1)  # [B, N]
```

### Run Full Test Set Assembly

```bash
python assemble_predictions_hpc.py \
    --checkpoint_path outputs_lidar/checkpoints/best_model.ckpt \
    --data_dir ../data/processed_data_optimized \
    --output_dir outputs_lidar/full_reconstructions \
    --num_points 8192
```

---

## 📊 Expected Performance

Based on similar Point Transformer architectures:

| Metric | Expected Range | Notes |
|--------|---------------|-------|
| Mean IoU | 0.40 - 0.60 | Depends on data quality |
| High Vegetation IoU | 0.60 - 0.75 | Most common class |
| Power Lines IoU | 0.30 - 0.55 | Challenging (thin structures) |
| Pylons IoU | 0.00 - 0.30 | Very rare class |
| Others IoU | 0.40 - 0.60 | Varied structures |
| Overall Accuracy | 0.70 - 0.85 | Dominated by vegetation |

---

## 🎓 Key Differences from Multispectral Deployment

| Aspect | Multispectral (2D) | LiDAR (3D) |
|--------|-------------------|------------|
| **Architecture** | U-Net + VGG16 | Point Transformer V2 |
| **Input** | 12-channel images (512×512) | Point clouds (8192 points) |
| **Framework** | PyTorch | PyTorch Lightning |
| **Metrics** | IoU, Dice, Pixel Accuracy | IoU, F1, Point Accuracy |
| **Pretrained** | ImageNet (RGB only) | None (custom architecture) |
| **Batch Size** | 32 | 16 (more memory intensive) |
| **Training Time** | ~2-4 hours | ~6-12 hours |

---

## 📝 Notes

### Class Imbalance
- High Vegetation dominates (~60% of points)
- Pylons are very rare (~0.01% of points)
- Class weights are automatically calculated during training

### Model Checkpointing
- **ONLY best model is saved** (based on val_miou)
- No intermediate checkpoints to save disk space
- Best model overwrites previous best

### Visualization
- Full point cloud assembly can take 10-30 minutes
- Visualizations downsample large point clouds for speed
- 3D plots may look sparse due to sampling

---

## 🆘 Support

If you encounter issues:

1. Check SLURM logs: `lidar_train_<job_id>.out` and `.err`
2. Verify data availability: `ls ../data/processed_data_optimized/train/`
3. Check GPU availability: `sinfo -p GPU`
4. Review training summary: `cat outputs_lidar/training_summary.json`

---

## ✅ Completion Checklist

- [ ] Data transferred to HPC (`../data/processed_data_optimized/`)
- [ ] Deployment package created and transferred
- [ ] Files extracted to `hpc_deployment/`
- [ ] Data verified (train & test chunks exist)
- [ ] Job submitted successfully (`sbatch submit_lidar_job.sh`)
- [ ] Training completed without errors
- [ ] Best model saved (`outputs_lidar/checkpoints/best_model.ckpt`)
- [ ] Visualizations generated (`outputs_lidar/visualizations/`)
- [ ] Full reconstruction completed (`outputs_lidar/full_reconstructions/`)
- [ ] Results downloaded to local machine

---

**Good luck with your Point Transformer V2 training!** 🚀
