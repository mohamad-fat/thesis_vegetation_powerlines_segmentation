#!/bin/bash
#SBATCH --job-name=ca_pointnet_train
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
# CA-PointNet++ Training Job for HPC
# ==============================================================================
# Based on paper: "Semantic Segmentation of Transmission Corridor 3D Point
# Clouds Based on CA-PointNet++"
#
# Architecture:
#   - PointNet++ backbone with Single-Scale Grouping (SSG)
#   - Coordinate Attention (CA) modules in Set Abstraction layers
#   - 4 encoder layers + 4 decoder layers
#   - Input: 4096 points, 6 channels (XYZ + RGB)
#
# Training Config (as per paper):
#   - Batch Size: 32
#   - Optimizer: Adam (lr=0.001, momentum=0.9)
#   - Epochs: 100
#   - Classes: 4 (Tower, Transmission Line, Ground Wire, Ground)
# ==============================================================================

echo "========================================="
echo "CA-PointNet++ Training Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="
echo ""

# Load PyTorch module (includes Python 3, CUDA, and PyTorch)
module use /share/apps/eb/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Loaded modules:"
module list
echo ""

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."

# Install in stages to avoid conflicts
pip install --user numpy==1.26.4 --force-reinstall
pip install --user torchvision==0.16.2
pip install --user lightning==2.5.0
pip install --user tensorboard matplotlib
pip install --user scikit-learn tqdm

echo "Dependencies installed"
echo ""

# Check data availability
if [ ! -d "./data/processed_data_optimized/train" ]; then
    echo "ERROR: Processed data not found!"
    echo "Expected: ./data/processed_data_optimized/train"
    exit 1
fi

NUM_TRAIN=$(ls ./data/processed_data_optimized/train/*.pt 2>/dev/null | wc -l)
NUM_TEST=$(ls ./data/processed_data_optimized/test/*.pt 2>/dev/null | wc -l)
echo "Found $NUM_TRAIN training chunks and $NUM_TEST test chunks"
echo ""

# Training parameters (adjusted for V100 GPU memory constraints)
DATA_DIR="./data/processed_data_optimized"
OUTPUT_DIR="./outputs_ca_pointnet"
BATCH_SIZE=8  # Reduced from 32 (paper) for V100 GPU - CA module uses significant memory
EPOCHS=100
LR=0.001
MOMENTUM=0.9
NUM_POINTS=2048  # Reduced from 4096 for memory efficiency
NUM_WORKERS=4  # Reduced workers to avoid memory pressure
EARLY_STOP_PATIENCE=15
GRADIENT_CLIP=1.0

echo "========================================="
echo "Training Configuration"
echo "========================================="
echo "Model: CA-PointNet++"
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE (reduced from 32 for V100 memory constraints)"
echo "Max epochs: $EPOCHS (as per paper)"
echo "Learning rate: $LR (as per paper)"
echo "Momentum: $MOMENTUM (as per paper)"
echo "Points per sample: $NUM_POINTS (reduced from 4096 for memory)"
echo "Input channels: 6 (XYZ + RGB) - 2x memory vs 3 channels"
echo "Number of workers: $NUM_WORKERS (reduced for memory efficiency)"
echo "Early stopping patience: $EARLY_STOP_PATIENCE"
echo "Gradient clipping: $GRADIENT_CLIP"
echo "========================================="
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Start training
echo "Starting CA-PointNet++ training..."
echo "Start time: $(date)"
echo ""

python train_ca_pointnet_hpc.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR \
    --momentum $MOMENTUM \
    --num_points $NUM_POINTS \
    --num_workers $NUM_WORKERS \
    --early_stop_patience $EARLY_STOP_PATIENCE \
    --gradient_clip $GRADIENT_CLIP

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Training completed"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"

    # Display summary if available
    if [ -f "$OUTPUT_DIR/training_summary.json" ]; then
        echo ""
        echo "Training Summary:"
        cat "$OUTPUT_DIR/training_summary.json"
    fi
else
    echo ""
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs for details: ca_pointnet_train_${SLURM_JOB_ID}.err"
fi

echo ""
echo "========================================="
echo "Job Statistics"
echo "========================================="
sstat --format=JobID,MaxRSS,MaxVMSize,AveCPU -j $SLURM_JOB_ID
echo "========================================="

exit $TRAIN_EXIT_CODE
