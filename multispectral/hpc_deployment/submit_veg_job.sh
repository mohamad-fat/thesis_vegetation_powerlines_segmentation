#!/bin/bash
#SBATCH --job-name=multispectral_veg_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfatfa01@ucy.ac.cy

# ==============================================================================
# Multispectral Vegetation Segmentation Training - SLURM Job Script
# ==============================================================================
# This script trains a U-Net + VGG16 model for vegetation segmentation
# using multispectral imagery (RGB + spectral bands + vegetation indices).
#
# Usage: sbatch submit_veg_job.sh
# ==============================================================================

echo "================================================================"
echo "Multispectral Vegetation Segmentation Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# Load PyTorch module (includes Python 3, CUDA, and PyTorch)
module use /share/apps/eb/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Loaded modules:"
module list
echo ""

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Check data availability
if [ ! -d "ml_ready/rgb_tiles" ] || [ ! -d "ml_ready/mask_tiles" ]; then
    echo "ERROR: Processed data not found!"
    echo "Expected: ml_ready/rgb_tiles and ml_ready/mask_tiles"
    exit 1
fi

NUM_RGB=$(ls ml_ready/rgb_tiles/*.png 2>/dev/null | wc -l)
NUM_VEG_MASKS=$(ls ml_ready/mask_tiles/*_vegetation.npy 2>/dev/null | wc -l)
echo "Found $NUM_RGB RGB tiles and $NUM_VEG_MASKS vegetation masks"
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."

# Install in stages to avoid conflicts
pip install --user torchvision==0.16.2
pip install --user pretrainedmodels efficientnet-pytorch
pip install --user segmentation-models-pytorch==0.3.3
pip install --user albumentations opencv-python-headless scikit-learn tqdm

echo "Dependencies installed"
echo ""

# Run training with optimized parameters
echo "Starting training..."
python train_multispectral_veg_hpc.py \
    --data_dir ./ml_ready \
    --output_dir ./outputs_veg \
    --batch_size 32 \
    --epochs 50 \
    --lr 0.00001 \
    --num_workers 8 \
    --architecture Unet \
    --encoder vgg16 \
    --early_stop_patience 10 \
    --lr_patience 5 \
    --amp

EXIT_STATUS=$?

echo ""
echo "================================================================"
echo "Training completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "================================================================"

# Generate visualizations if training succeeded
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "Generating training metrics visualizations..."
    python visualize_results.py --output_dir outputs_veg
    echo ""
    echo "Generating sample predictions..."
    python visualize_predictions.py \
        --model_path outputs_veg/best_model.pth \
        --data_dir ml_ready \
        --architecture Unet \
        --encoder vgg16 \
        --mask_type vegetation \
        --output_dir outputs_veg/visualizations/predictions \
        --num_samples 6
    echo ""
    echo "================================================================"
    echo "✅ All results and visualizations ready!"
    echo "Results: outputs_veg/"
    echo "Visualizations: outputs_veg/visualizations/"
    echo "Predictions: outputs_veg/visualizations/predictions/"
    echo "================================================================"
fi

exit $EXIT_STATUS
