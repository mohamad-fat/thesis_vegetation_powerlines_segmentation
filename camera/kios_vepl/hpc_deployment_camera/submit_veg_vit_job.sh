#!/bin/bash
#SBATCH --job-name=kios_veg_vit
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

echo "================================================================"
echo "KIOS VEPL Vegetation Segmentation - Mix Transformer (SegFormer)"
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
if [ ! -d "processed_data/images" ] || [ ! -d "processed_data/veg_masks" ]; then
    echo "ERROR: Processed data not found!"
    exit 1
fi

NUM_IMAGES=$(ls processed_data/images/*.png 2>/dev/null | wc -l)
NUM_MASKS=$(ls processed_data/veg_masks/*.png 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images and $NUM_MASKS veg_masks"
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

# Run training with Mix Transformer (SegFormer)
echo "Starting Mix Transformer (SegFormer) training..."
python train_kios_vepl_veg_VT.py \
    --data_dir ./processed_data \
    --output_dir ./outputs_veg_vit \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.00001 \
    --weight_decay 0.01 \
    --num_workers 8 \
    --architecture Unet \
    --encoder mit_b2 \
    --image_size 512 \
    --early_stop_patience 20 \
    --lr_patience 10 \
    --lr_factor 0.5 \
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
    echo "================================================================"
    echo "Mix Transformer training completed successfully!"
    echo "Results: outputs_veg_vit/"
    echo "Best model: outputs_veg_vit/best_model_vit.pth"
    echo "Training history: outputs_veg_vit/training_history_vit.json"
    echo "Training summary: outputs_veg_vit/training_summary_vit.json"
    echo "Encoder used: mit_b2 (Mix Transformer / SegFormer)"
    echo "================================================================"
fi

exit $EXIT_STATUS
