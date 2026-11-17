#!/bin/bash
#SBATCH --job-name=lidar_train
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
# Point Transformer V2 LiDAR Training - SLURM Job Script
# ==============================================================================
# This script trains a Point Transformer V2 model for semantic segmentation
# of LiDAR point clouds (power lines, pylons, vegetation).
#
# Usage: sbatch submit_lidar_job.sh
# ==============================================================================

echo "================================================================"
echo "Point Transformer V2 LiDAR Training"
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
if [ ! -d "./data/processed_data_optimized/train" ]; then
    echo "ERROR: Processed data not found!"
    echo "Expected: ./data/processed_data_optimized/train"
    exit 1
fi

NUM_TRAIN=$(ls ./data/processed_data_optimized/train/*.pt 2>/dev/null | wc -l)
NUM_TEST=$(ls ./data/processed_data_optimized/test/*.pt 2>/dev/null | wc -l)
echo "Found $NUM_TRAIN training chunks and $NUM_TEST test chunks"
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."

# Install in stages to avoid conflicts
# Fix numpy version compatibility issue (numpy 2.0+ has _core module issues with PyTorch 2.1.2)
pip install --user numpy=1.26.4 --force-reinstall
pip install --user torchvision==0.16.2
pip install --user lightning==2.5.0
pip install --user tensorboard matplotlib
pip install --user scikit-learn tqdm

echo "Dependencies installed"
echo ""

# Run training with GPU-optimized parameters (memory-efficient for V100)
echo "Starting Point Transformer V2 training..."
python train_point_transformer_hpc.py \
    --data_dir ./data/processed_data_optimized \
    --output_dir ./outputs_lidar \
    --batch_size 16 \
    --epochs 5 \
    --lr 0.0001 \
    --num_points 4096 \
    --d_model 156 \
    --num_layers 3 \
    --k_neighbors 16 \
    --dropout 0.0 \
    --num_workers 8 \
    --early_stop_patience 3 \
    --gradient_clip 1.0

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
    python visualize_results.py --output_dir outputs_lidar
    echo ""
    echo "Assembling and visualizing full test set predictions..."
    python assemble_predictions_hpc.py \
        --checkpoint_path outputs_lidar/checkpoints/best_model.ckpt \
        --data_dir ./data/processed_data_optimized \
        --output_dir outputs_lidar/full_reconstructions \
        --num_points 4096
    echo ""
    echo "================================================================"
    echo "✅ All results and visualizations ready!"
    echo "Results: outputs_lidar/"
    echo "Checkpoints: outputs_lidar/checkpoints/best_model.ckpt"
    echo "Logs: outputs_lidar/logs/"
    echo "Full Reconstructions: outputs_lidar/full_reconstructions/"
    echo "================================================================"
fi

exit $EXIT_STATUS
