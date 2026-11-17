#!/bin/bash
#SBATCH --job-name=ms_veg_vit_adapt
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
# Multispectral Vegetation Segmentation - ADAPTED Vision Transformer
# ==============================================================================
# This script trains U-Net + Mix Transformer adapted for 9-channel multispectral input
# Input: RGB (3ch) + Bands (4ch: Green, Red, RedEdge, NIR) + Indices (2ch: NDVI, GNDVI)
#
# Solution: Adapts pretrained Mix Transformer from 3→9 channels
# ==============================================================================

echo "================================================================"
echo "Multispectral Vegetation Segmentation - Adapted Vision Transformer"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================================================================"

# Load PyTorch module
module use /share/apps/eb/modules/all
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

echo "Loaded modules:"
module list
echo ""

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
echo ""

# Check data availability
if [ ! -d "ml_ready/rgb_tiles" ] || [ ! -d "ml_ready/band_tiles" ] || [ ! -d "ml_ready/index_tiles" ] || [ ! -d "ml_ready/mask_tiles" ]; then
    echo "ERROR: Processed data not found!"
    exit 1
fi

echo "Data Summary:"
echo "  RGB tiles: $(ls ml_ready/rgb_tiles/*.png 2>/dev/null | wc -l)"
echo "  Band tiles: $(ls ml_ready/band_tiles/*_nir.npy 2>/dev/null | wc -l)"
echo "  Index tiles: $(ls ml_ready/index_tiles/*_ndvi.npy 2>/dev/null | wc -l)"
echo "  Vegetation masks: $(ls ml_ready/mask_tiles/*_vegetation.npy 2>/dev/null | wc -l)"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install --user torchvision==0.16.2
pip install --user pretrainedmodels efficientnet-pytorch
pip install --user segmentation-models-pytorch==0.3.3
pip install --user albumentations opencv-python-headless scikit-learn tqdm matplotlib
echo "Dependencies installed"
echo ""

# Run training
echo "================================================================"
echo "Training Configuration:"
echo "================================================================"
echo "  Model: U-Net + Mix Transformer (mit_b2)"
echo "  Input Channels: 9"
echo "    - RGB: 3 channels (Red, Green, Blue)"
echo "    - Spectral Bands: 4 channels (Green, Red, RedEdge, NIR)"
echo "    - Vegetation Indices: 2 channels (NDVI, GNDVI)"
echo ""
echo "  Strategy: Adapt pretrained MiT encoder"
echo "    1. Load pretrained weights (ImageNet, 3 channels)"
echo "    2. Modify first conv layer: 3 → 9 channels"
echo "    3. Replicate RGB weights across new channels"
echo ""
echo "  Training Setup:"
echo "    - Architecture: U-Net"
echo "    - Encoder: mit_b2 (adapted)"
echo "    - Batch Size: 32"
echo "    - Learning Rate: 0.0003 (with 3-epoch warmup)"
echo "    - Optimizer: AdamW (weight_decay=0.01)"
echo "    - Loss: Combined BCE + Dice (0.3/0.7)"
echo "    - Pos Weight: 3.0 (class imbalance handling)"
echo "    - EMA: Enabled (decay=0.999)"
echo "    - Gradient Clipping: 1.0"
echo "================================================================"
echo ""

python train_multispectral_veg_vit_adapted_hpc.py \
    --data_dir ./ml_ready \
    --output_dir ./outputs_multispectral_veg_vit \
    --batch_size 32 \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 0.01 \
    --warmup_epochs 3 \
    --min_lr 1e-6 \
    --pos_weight 3.0 \
    --num_workers 8 \
    --architecture Unet \
    --encoder mit_b2 \
    --image_size 512 \
    --early_stop_patience 15 \
    --max_grad_norm 1.0 \
    --use_ema \
    --ema_decay 0.999 \
    --amp

EXIT_STATUS=$?

echo ""
echo "================================================================"
echo "Training completed with exit status: $EXIT_STATUS"
echo "End Time: $(date)"
echo "================================================================"

if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo "✅ Training completed successfully!"
    echo "================================================================"
    echo ""
    echo "📊 RESULTS SUMMARY"
    echo "================================================================"
    echo "Output directory: outputs_multispectral_veg_vit/"
    echo ""
    echo "Generated files:"
    echo "  - best_model.pth: Best model checkpoint (with EMA)"
    echo "  - training_history.json: Training metrics per epoch"
    echo "  - training_summary.json: Final results summary"
    echo "  - training_curves.png: Loss/IoU/Dice/LR visualizations"
    echo "  - training_*.log: Detailed training log"
    echo ""
    echo "================================================================"
    echo "📈 TECHNICAL DETAILS"
    echo "================================================================"
    echo ""
    echo "✅ Successfully adapted Mix Transformer for multispectral data!"
    echo ""
    echo "Adaptation Strategy:"
    echo "  1. Loaded pretrained MiT encoder (ImageNet weights)"
    echo "  2. Modified first conv layer from 3 → 9 input channels"
    echo "  3. Replicated RGB weights across 9 channels"
    echo "  4. Normalized weights to maintain activation magnitude"
    echo "  5. Added small noise for channel diversity"
    echo ""
    echo "Input Composition (9 channels):"
    echo "  RGB:     [0-2]   - Red, Green, Blue"
    echo "  Bands:   [3-6]   - Green, Red, RedEdge, NIR"
    echo "  Indices: [7-8]   - NDVI, GNDVI"
    echo ""
    echo "Expected Performance:"
    echo "  - Better than RGB-only (3ch) due to spectral information"
    echo "  - Better than CNN encoders due to transformer attention"
    echo "  - Estimated IoU: 75-85%"
    echo ""
    echo "================================================================"
    echo "📊 MODEL COMPARISON"
    echo "================================================================"
    echo ""
    echo "Compare with:"
    echo "  1. RGB + VGG16:        outputs_rgb_veg/"
    echo "  2. RGB + ViT:          outputs_rgb_veg_vit/"
    echo "  3. MS + ViT (adapted): outputs_multispectral_veg_vit/ (this)"
    echo ""
    echo "================================================================"
fi

exit $EXIT_STATUS
