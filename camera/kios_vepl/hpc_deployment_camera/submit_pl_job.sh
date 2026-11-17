#!/bin/bash
#SBATCH --job-name=kios_vepl_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfatfa01@ucy.ac.cy

echo "================================================================"
echo "KIOS VEPL Power Line Detection Training"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# Load modules
module use /share/apps/eb/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Loaded modules:"
module list
echo ""

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Check data availability
if [ ! -d "processed_data/images" ] || [ ! -d "processed_data/pl_masks" ]; then
    echo "ERROR: Processed data not found!"
    exit 1
fi

NUM_IMAGES=$(ls processed_data/images/*.png 2>/dev/null | wc -l)
NUM_MASKS=$(ls processed_data/pl_masks/*.png 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images and $NUM_MASKS pl_masks"
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
python train_kios_vepl_pl_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./output_pl \
    --batch_size 64 \
    --epochs 100 \
    --lr 0.0001 \
    --num_workers 8 \
    --architecture DeepLabV3 \
    --encoder resnet34 \
    --early_stop_patience 15 \
    --lr_patience 8 \
    --bce_pos_weight 10.0 \
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
    python visualize_results.py
    echo ""
    echo "Generating sample predictions..."
    python visualize_predictions.py
    echo ""
    echo "================================================================"
    echo "✅ All results and visualizations ready!"
    echo "Results: outputs/"
    echo "Visualizations: outputs/visualizations/"
    echo "Predictions: outputs/visualizations/predictions/"
    echo "================================================================"
fi

exit $EXIT_STATUS
