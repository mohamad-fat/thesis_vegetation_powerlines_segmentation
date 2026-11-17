#!/bin/bash
#SBATCH --job-name=kios_pl_vit
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mfatfa01@ucy.ac.cy

echo "================================================================"
echo "KIOS VEPL Power Line Detection - Mix Transformer (SegFormer)"
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
    echo "Expected: processed_data/images/ and processed_data/pl_masks/"
    exit 1
fi

NUM_IMAGES=$(ls processed_data/images/*.png 2>/dev/null | wc -l)
NUM_MASKS=$(ls processed_data/pl_masks/*.png 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES images and $NUM_MASKS power line masks"

if [ $NUM_IMAGES -ne $NUM_MASKS ]; then
    echo "WARNING: Number of images and masks do not match!"
fi
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."

# Install in stages to avoid conflicts
pip install --user torchvision==0.16.2
pip install --user pretrainedmodels efficientnet-pytorch
pip install --user segmentation-models-pytorch==0.3.3
pip install --user albumentations opencv-python-headless scikit-learn tqdm matplotlib

echo "Dependencies installed"
echo ""

# Display configuration
echo "================================================================"
echo "Training Configuration:"
echo "================================================================"
echo "Model: Mix Transformer (SegFormer) - Encoder: mit_b2"
echo "Architecture: U-Net with MiT-B2 encoder"
echo "Image size: 512x512"
echo "Batch size: 16"
echo "Loss: Combined Focal Tversky (70%) + Weighted BCE (30%)"
echo "Tversky α=0.3, β=0.7 (emphasizes recall)"
echo "BCE pos_weight=10.0 (handles class imbalance)"
echo "LR Schedule: Cosine Annealing with 5-epoch warmup"
echo "Initial LR: 1e-4 → Min LR: 1e-7"
echo "Regularization: EMA (decay=0.999) + Gradient Clipping (1.0)"
echo "Early stopping: 10 epochs patience"
echo "================================================================"
echo ""

# Run training with Mix Transformer (SegFormer) for Power Lines
echo "Starting Mix Transformer (SegFormer) power line detection training..."
python train_kios_vepl_pl_vit.py \
    --data_dir ./processed_data \
    --output_dir ./outputs_pl_vit \
    --batch_size 16 \
    --epochs 100 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --warmup_epochs 5 \
    --min_lr 1e-7 \
    --num_workers 4 \
    --architecture Unet \
    --encoder mit_b2 \
    --image_size 512 \
    --early_stop_patience 10 \
    --max_grad_norm 1.0 \
    --use_ema \
    --ema_decay 0.999 \
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --tversky_weight 0.7 \
    --bce_weight 0.3 \
    --bce_pos_weight 10.0 \
    --amp

EXIT_STATUS=$?

echo ""
echo "================================================================"
echo "Training completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "================================================================"

# Display results if training succeeded
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "Mix Transformer Power Line Detection Training Completed!"
    echo "================================================================"
    echo "Output directory: outputs_pl_vit/"
    echo ""
    echo "Model Files:"
    echo "  - best_model_pl_vit.pth (best model checkpoint)"
    echo ""
    echo "Visualizations:"
    echo "  - training_curves.png (loss, IoU, Dice, LR schedule)"
    echo "  - metrics_comparison.png (multi-metric analysis)"
    echo "  - predictions_visualization.png (sample predictions with overlays)"
    echo ""
    echo "Data Files:"
    echo "  - training_history_pl_vit.json (metrics per epoch)"
    echo "  - training_summary_pl_vit.json (final results)"
    echo "  - training_*.log (detailed training log)"
    echo ""
    echo "Model Configuration:"
    echo "  - Encoder: mit_b2 (Mix Transformer / SegFormer)"
    echo "  - Loss: Focal Tversky + Weighted BCE"
    echo "  - Optimizer: AdamW with Cosine Annealing"
    echo "  - Enhancements: EMA, Gradient Clipping, Enhanced Augmentations"
    echo ""

    # Display final metrics if available
    if [ -f "outputs_pl_vit/training_summary_pl_vit.json" ]; then
        echo "Final Results:"
        python -c "
import json
try:
    with open('outputs_pl_vit/training_summary_pl_vit.json', 'r') as f:
        summary = json.load(f)
    print(f\"  Best Val IoU: {summary.get('best_val_iou', 'N/A'):.4f}\")
    print(f\"  Test IoU: {summary.get('test_iou', 'N/A'):.4f}\")
    print(f\"  Test Dice: {summary.get('test_dice', 'N/A'):.4f}\")
    print(f\"  Test Accuracy: {summary.get('test_acc', 'N/A'):.2f}%\")
    print(f\"  Total Epochs: {summary.get('total_epochs', 'N/A')}\")
except:
    pass
"
    fi

    echo "================================================================"
else
    echo ""
    echo "================================================================"
    echo "ERROR: Training failed with exit status: $EXIT_STATUS"
    echo "Check the error log: kios_pl_vit_${SLURM_JOB_ID}.err"
    echo "================================================================"
fi

exit $EXIT_STATUS
