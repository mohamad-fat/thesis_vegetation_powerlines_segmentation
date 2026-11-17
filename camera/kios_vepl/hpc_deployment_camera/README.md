# KIOS VEPL Segmentation - HPC Deployment Guide

This folder contains everything needed to train **both Power Line Detection and Vegetation Segmentation** models on an HPC cluster with GPU support.

## 🎯 Two Training Modes

### 1. Power Line Detection
- **Model**: DeepLabV3 + ResNet34
- **Loss**: Combined Tversky + Weighted BCE
- **Script**: `train_kios_vepl_hpc.py`
- **Submission**: `submit_job.sh`
- **Output**: `outputs/`

### 2. Vegetation Segmentation
- **Model**: U-Net + VGG16
- **Loss**: BCEWithLogitsLoss
- **Script**: `train_kios_vepl_veg_hpc.py`
- **Submission**: `submit_veg_job.sh`
- **Output**: `outputs_veg/`

## 📁 Directory Structure

```
hpc_deployment/
├── README.md                        # This file
├── train_kios_vepl_hpc.py          # Power line training script
├── train_kios_vepl_veg_hpc.py      # Vegetation training script (NEW)
├── submit_job.sh                    # SLURM script for power line
├── submit_veg_job.sh                # SLURM script for vegetation (NEW)
├── visualize_results.py             # Generate training plots (shared)
├── visualize_predictions.py         # Generate prediction samples (shared)
├── requirements.txt                 # Python dependencies
├── processed_data/                  # Preprocessed dataset (TO BE COPIED)
│   ├── images/                      # Tiled images (256x256)
│   └── masks/                       # Binary masks (power line OR vegetation)
├── outputs/                         # Power line training outputs
│   ├── best_model.pth              # Best power line model
│   ├── training_history.json       # Training metrics
│   └── visualizations/             # Generated plots
├── outputs_veg/                     # Vegetation training outputs (NEW)
│   ├── best_model.pth              # Best vegetation model
│   ├── training_history.json       # Training metrics
│   └── visualizations/             # Generated plots
└── logs/                            # SLURM logs (created automatically)
```

---

## 🚀 Quick Start

### Step 1: Prepare Data Locally

**IMPORTANT:** Before transferring to HPC, you must first preprocess your data locally:

```bash
# On your local machine:
cd /Users/user/Desktop/university/thesis/veg_enc_code/camera/kios_vepl/notebooks

# Run the preprocessing notebook
jupyter notebook preprocess_kios_vepl.ipynb
# Execute all cells to generate processed_data/
```

This will create `camera/kios_vepl/processed_data/` with:
- `images/` - Tiled images (256x256 PNG files)
- `masks/` - Binary power line masks
- `statistics/` - Dataset statistics

### Step 2: Copy Processed Data

Copy the preprocessed data to the HPC deployment folder:

```bash
# On your local machine:
cd /Users/user/Desktop/university/thesis/veg_enc_code/camera/kios_vepl

# Copy processed data
cp -r processed_data/ hpc_deployment/
```

### Step 3: Transfer to HPC

Transfer the entire `hpc_deployment/` folder to your HPC cluster:

```bash
# Replace with your HPC details:
HPC_USER="your_username"
HPC_HOST="hpc.university.edu"
HPC_PATH="/home/$HPC_USER/kios_vepl"

# Transfer files (will take time depending on dataset size)
rsync -avz --progress hpc_deployment/ $HPC_USER@$HPC_HOST:$HPC_PATH/
```

**Alternative using scp:**
```bash
scp -r hpc_deployment/ $HPC_USER@$HPC_HOST:$HPC_PATH/
```

### Step 4: Setup on HPC

SSH into your HPC cluster:

```bash
ssh $HPC_USER@$HPC_HOST
cd $HPC_PATH
```

**Configure the job script** - Edit `submit_job.sh`:

```bash
nano submit_job.sh
```

**Important settings to adjust:**
1. `--partition=gpu` → Change to your HPC's GPU partition name
2. `--mail-user=your.email@university.edu` → Change to your email
3. `--gres=gpu:1` → Adjust GPU type if needed (e.g., `gpu:v100:1`, `gpu:a100:1`)
4. Module names → Check available modules with `module avail`

**Check available GPU partitions:**
```bash
sinfo -o "%20P %5a %.10l %16F"  # List partitions
module avail cuda                # Available CUDA versions
module avail python              # Available Python versions
```

### Step 5: Install Dependencies

Option A - Create virtual environment (recommended):
```bash
module load python/3.10
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option B - Use conda (if available):
```bash
module load anaconda3
conda create -n kios_vepl python=3.10
conda activate kios_vepl
pip install -r requirements.txt
```

**For PyTorch with CUDA support:**
```bash
# Check CUDA version first
module load cuda/11.8  # or your version
nvcc --version

# Install PyTorch with matching CUDA version
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Step 6: Verify Setup

Test that everything works:

```bash
# Load modules
module load cuda/11.8 python/3.10
source venv/bin/activate

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test data loading
python -c "import os; print(f'Images: {len(os.listdir(\"processed_data/images\"))}')"
```

### Step 7: Submit Job

**Choose which model to train:**

#### For Power Line Detection:
```bash
sbatch submit_job.sh
```

#### For Vegetation Segmentation:
```bash
sbatch submit_veg_job.sh
```

#### To train BOTH models (submit 2 jobs):
```bash
sbatch submit_job.sh         # Power line
sbatch submit_veg_job.sh     # Vegetation
```

**Monitor your job(s):**
```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f logs/train_<jobid>.out

# View errors
tail -f logs/train_<jobid>.err

# Cancel job if needed
scancel <jobid>
```

---

## 🔧 Training Script Options

Both training scripts support many command-line options:

### Power Line Detection - Basic Usage

```bash
python train_kios_vepl_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./outputs \
    --batch_size 32 \
    --epochs 20 \
    --architecture DeepLabV3 \
    --encoder resnet34
```

### Vegetation Segmentation - Basic Usage

```bash
python train_kios_vepl_veg_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./outputs_veg \
    --batch_size 32 \
    --epochs 20 \
    --architecture Unet \
    --encoder vgg16
```

### Key Differences Between Models

| Parameter | Power Line | Vegetation |
|-----------|-----------|------------|
| Script | `train_kios_vepl_hpc.py` | `train_kios_vepl_veg_hpc.py` |
| Architecture | DeepLabV3 | U-Net |
| Encoder | resnet34 | vgg16 |
| Loss Function | Tversky + BCE | BCEWithLogitsLoss |
| Learning Rate | 0.0001 | 0.00001 |
| Output Directory | `outputs/` | `outputs_veg/` |
| Tversky Parameters | ✅ (α=0.3, β=0.7) | ❌ (not used) |
| BCE Pos Weight | ✅ (15.0) | ❌ (not used) |

### All Available Options (Both Scripts)

```bash
# For power line:
python train_kios_vepl_hpc.py \
    # Data paths
    --data_dir ./processed_data \
    --output_dir ./outputs \

    # Model configuration
    --architecture DeepLabV3 \        # DeepLabV3, Unet, FPN, UnetPlusPlus
    --encoder resnet34 \               # resnet34, resnet50, efficientnet-b0, etc.

    # Training hyperparameters
    --batch_size 32 \                  # Larger for more GPU memory
    --epochs 20 \
    --lr 0.0001 \
    --weight_decay 0.00001 \

    # Data split
    --train_split 0.6 \
    --val_split 0.2 \
    --random_seed 42 \

    # Training options
    --num_workers 8 \                  # Parallel data loading
    --early_stop_patience 7 \
    --lr_patience 5 \

    # Loss function
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --bce_pos_weight 15.0 \

    # Device settings
    --device cuda \                    # cuda, cpu, or auto
    --amp \                            # Enable mixed precision

    # Checkpointing
    --save_freq 5 \                    # Save checkpoint every 5 epochs
    --resume ./outputs/checkpoint_epoch_10.pth  # Resume from checkpoint
```

### Resume Training

If training was interrupted:

```bash
python train_kios_vepl_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./outputs \
    --resume ./outputs/checkpoint_epoch_15.pth
```

---

## 📊 Output Files

After training completes, you'll find results in separate directories:

### Power Line Detection Outputs (`outputs/`)
- `best_model.pth` - Best power line model
- `training_*.log` - Detailed training log
- `training_history.json` - Epoch-by-epoch metrics
- `training_summary.json` - Final summary
- `visualizations/` - Training plots and predictions

### Vegetation Segmentation Outputs (`outputs_veg/`)
- `best_model.pth` - Best vegetation model
- `training_*.log` - Detailed training log
- `training_history.json` - Epoch-by-epoch metrics
- `training_summary.json` - Final summary
- `visualizations/` - Training plots and predictions

### SLURM Logs (`logs/`)
- `kios_vepl_train_<jobid>.out` - Power line standard output
- `kios_veg_train_<jobid>.out` - Vegetation standard output
- `*.err` - Error logs (if any)

### Example training_summary.json:
```json
{
  "config": {
    "batch_size": 32,
    "epochs": 20,
    "lr": 0.0001,
    ...
  },
  "best_val_iou": 0.2519,
  "test_iou": 0.2493,
  "test_dice": 0.3763,
  "test_acc": 99.12,
  "total_epochs": 18
}
```

---

## 📥 Download Results

After training, download results back to your local machine:

### Download Power Line Results:
```bash
# On your local machine:
rsync -avz --progress $HPC_USER@$HPC_HOST:$HPC_PATH/outputs/ ./hpc_results_powerline/
```

### Download Vegetation Results:
```bash
rsync -avz --progress $HPC_USER@$HPC_HOST:$HPC_PATH/outputs_veg/ ./hpc_results_vegetation/
```

### Download Both:
```bash
rsync -avz --progress $HPC_USER@$HPC_HOST:$HPC_PATH/outputs* ./hpc_results/
```

---

## 🐛 Troubleshooting

### Problem: "CUDA out of memory"

**Solution:** Reduce batch size
```bash
# In submit_job.sh, change:
BATCH_SIZE=16  # or 8 for smaller GPU
```

### Problem: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch with CUDA support
```bash
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Problem: "CUDA not available"

**Solutions:**
1. Check if CUDA module is loaded: `module list`
2. Load CUDA: `module load cuda/11.8`
3. Verify GPU allocation: `nvidia-smi`
4. Check SLURM script has `--gres=gpu:1`

### Problem: Job fails with "Out of memory (not CUDA)"

**Solution:** Increase memory in SLURM script
```bash
#SBATCH --mem=64G  # Increase from 32G
```

### Problem: "No data files found"

**Solution:** Verify data was transferred correctly
```bash
ls -la processed_data/images/*.png | wc -l
ls -la processed_data/masks/*.png | wc -l
```

### Problem: Job pending for long time

**Check queue:**
```bash
squeue -u $USER
sinfo -p gpu  # Check GPU partition status
```

**Solutions:**
1. Request fewer resources (smaller GPU, less time)
2. Try different partition
3. Check with HPC support about queue priorities

---

## 💡 Tips for Better Training

### 1. Batch Size Selection

Based on GPU memory:
- **V100 (16GB)**: batch_size=32-64
- **V100 (32GB)**: batch_size=64-128
- **A100 (40GB)**: batch_size=128-256
- **T4 (16GB)**: batch_size=16-32

Start with suggested size and increase if no OOM errors.

### 2. Number of Workers

Set based on CPU cores:
```bash
# In submit_job.sh
#SBATCH --cpus-per-task=8
# Then set:
NUM_WORKERS=8
```

### 3. Monitor GPU Usage

While job is running:
```bash
# SSH to the compute node (find node with: squeue -u $USER)
ssh compute-node-name
watch -n 1 nvidia-smi  # Update every second
```

### 4. Early Stopping

For small datasets, use patience:
- `--early_stop_patience 7` (default)
- `--lr_patience 5` (default)

This prevents overfitting and saves time.

### 5. Try Different Architectures

```bash
# U-Net (lighter, faster)
--architecture Unet --encoder resnet34

# FPN (good for power lines)
--architecture FPN --encoder resnet50

# U-Net++ (more complex)
--architecture UnetPlusPlus --encoder efficientnet-b3
```

---

## 📧 Getting Help

### HPC-Specific Issues
Contact your university's HPC support with:
- Job ID
- Error messages from logs
- SLURM script configuration

### Model/Training Issues
Check training logs in `outputs/training_*.log` for:
- Loss trends (should decrease)
- IoU trends (should increase)
- Early stopping triggers
- LR reductions

### Performance Issues
If IoU < 0.15:
1. **Label more images** (aim for 500+ tiles minimum)
2. Try different architecture (U-Net, FPN)
3. Adjust loss function weights
4. Increase training epochs
5. Check data quality in preprocessing

---

## 📚 Additional Resources

### PyTorch with CUDA
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Compatibility](https://pytorch.org/get-started/previous-versions/)

### SLURM Commands
- [SLURM Cheat Sheet](https://slurm.schedmd.com/pdfs/summary.pdf)
- Common commands:
  - `squeue` - Check job queue
  - `scancel` - Cancel job
  - `sinfo` - Cluster info
  - `sacct` - Job accounting

### Segmentation Models
- [SMP Documentation](https://smp.readthedocs.io/)
- [Available Encoders](https://smp.readthedocs.io/en/latest/encoders.html)
- [Available Architectures](https://smp.readthedocs.io/en/latest/models.html)

---

## ✅ Checklist Before Submission

- [ ] Preprocessed data locally and copied to `processed_data/`
- [ ] Transferred folder to HPC
- [ ] Updated submission scripts (`submit_job.sh` and/or `submit_veg_job.sh`) with correct partition and email
- [ ] Loaded required modules (CUDA, Python)
- [ ] Created and activated virtual environment
- [ ] Installed all dependencies from `requirements.txt`
- [ ] Verified CUDA availability with PyTorch
- [ ] Verified data files are present
- [ ] Adjusted batch size for your GPU
- [ ] Adjusted number of workers for your CPUs
- [ ] Decided which model(s) to train: Power Line, Vegetation, or Both
- [ ] Ready to submit: `sbatch submit_job.sh` and/or `sbatch submit_veg_job.sh`

---

## 📝 Quick Reference

### Power Line Detection
```bash
# Transfer data to HPC
rsync -avz hpc_deployment/ user@hpc:/path/

# Setup on HPC
ssh user@hpc
cd /path
module load cuda/11.8 python/3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Submit power line job
sbatch submit_job.sh

# Monitor
squeue -u $USER
tail -f kios_vepl_train_<jobid>.out

# Download results
rsync -avz user@hpc:/path/outputs/ ./results_powerline/
```

### Vegetation Segmentation
```bash
# Submit vegetation job (after setup above)
sbatch submit_veg_job.sh

# Monitor
tail -f kios_veg_train_<jobid>.out

# Download results
rsync -avz user@hpc:/path/outputs_veg/ ./results_vegetation/
```

### Train Both Models
```bash
# Submit both jobs simultaneously
sbatch submit_job.sh
sbatch submit_veg_job.sh

# Monitor both
squeue -u $USER

# Download all results
rsync -avz user@hpc:/path/outputs* ./results/
```

---

**Good luck with your training! 🚀**

For questions about this deployment, refer to the main training notebook documentation or contact your thesis supervisor.
