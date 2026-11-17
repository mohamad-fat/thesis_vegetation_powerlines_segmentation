#!/bin/bash
# ==============================================================================
# Prepare Multispectral HPC Deployment
# ==============================================================================
# This script helps you prepare the HPC deployment folder with processed data
# and creates a compressed archive ready for transfer to HPC.
#
# Usage: ./prepare_for_hpc.sh
# ==============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "================================================================"
echo "Multispectral Segmentation HPC Deployment Preparation"
echo "================================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. Check if processed data exists
# ------------------------------------------------------------------------------
echo "Step 1: Checking for preprocessed data..."

PROCESSED_DATA_DIR="$PROJECT_DIR/data/ml_ready"

if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    echo ""
    echo "❌ ERROR: Processed data not found!"
    echo ""
    echo "Expected location: $PROCESSED_DATA_DIR"
    echo ""
    echo "You need to run the preprocessing notebook first:"
    echo "  1. cd $PROJECT_DIR/notebooks"
    echo "  2. jupyter notebook preprocess_multispectral.ipynb"
    echo "  3. Execute all cells to generate data/ml_ready/"
    echo ""
    exit 1
fi

# Check subdirectories
REQUIRED_SUBDIRS=("rgb_tiles" "band_tiles" "index_tiles" "mask_tiles" "metadata")
for subdir in "${REQUIRED_SUBDIRS[@]}"; do
    if [ ! -d "$PROCESSED_DATA_DIR/$subdir" ]; then
        echo ""
        echo "❌ ERROR: $subdir directory not found!"
        echo "   Expected: $PROCESSED_DATA_DIR/$subdir"
        echo ""
        exit 1
    fi
done

# Count files
NUM_RGB=$(ls "$PROCESSED_DATA_DIR/rgb_tiles"/*.png 2>/dev/null | wc -l)
NUM_PL_MASKS=$(ls "$PROCESSED_DATA_DIR/mask_tiles"/*_powerline.npy 2>/dev/null | wc -l)
NUM_VEG_MASKS=$(ls "$PROCESSED_DATA_DIR/mask_tiles"/*_vegetation.npy 2>/dev/null | wc -l)

echo "✅ Found processed data:"
echo "   RGB tiles: $NUM_RGB"
echo "   Power line masks: $NUM_PL_MASKS"
echo "   Vegetation masks: $NUM_VEG_MASKS"

if [ "$NUM_RGB" -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No RGB tiles found!"
    exit 1
fi

if [ "$NUM_PL_MASKS" -eq 0 ] && [ "$NUM_VEG_MASKS" -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No mask files found!"
    exit 1
fi

echo ""

# ------------------------------------------------------------------------------
# 2. Copy processed data to HPC deployment folder
# ------------------------------------------------------------------------------
echo "Step 2: Copying processed data to HPC deployment folder..."

HPC_DATA="$SCRIPT_DIR/ml_ready"

# Remove old data if exists
if [ -d "$HPC_DATA" ]; then
    echo "   Removing old processed data..."
    rm -rf "$HPC_DATA"
fi

# Copy entire ml_ready directory
echo "   Copying ml_ready directory..."
cp -r "$PROCESSED_DATA_DIR" "$HPC_DATA"

echo "✅ Data copied successfully"
echo ""

# ------------------------------------------------------------------------------
# 3. Verify HPC deployment folder structure
# ------------------------------------------------------------------------------
echo "Step 3: Verifying HPC deployment structure..."

REQUIRED_FILES=(
    "README.md"
    "train_multispectral_pl_hpc.py"
    "train_multispectral_veg_hpc.py"
    "submit_pl_job.sh"
    "submit_veg_job.sh"
    "requirements.txt"
    "ml_ready/rgb_tiles"
    "ml_ready/mask_tiles"
)

MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -e "$SCRIPT_DIR/$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo ""
    echo "❌ ERROR: Missing required files:"
    for file in "${MISSING_FILES[@]}"; do
        echo "   - $file"
    done
    echo ""
    exit 1
fi

echo "✅ All required files present"
echo ""

# ------------------------------------------------------------------------------
# 4. Calculate sizes
# ------------------------------------------------------------------------------
echo "Step 4: Calculating deployment size..."

TOTAL_SIZE=$(du -sh "$SCRIPT_DIR" | cut -f1)
DATA_SIZE=$(du -sh "$HPC_DATA" | cut -f1)

echo "   Total deployment size: $TOTAL_SIZE"
echo "   Processed data size: $DATA_SIZE"
echo ""

# ------------------------------------------------------------------------------
# 5. Create compressed archive
# ------------------------------------------------------------------------------
echo "Step 5: Creating compressed archive..."

ARCHIVE_NAME="multispectral_hpc_$(date +%Y%m%d_%H%M%S).tar.gz"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_FOLDER="$(basename "$SCRIPT_DIR")"

cd "$PARENT_DIR"

echo "   Creating: $ARCHIVE_NAME"
echo "   This may take a few minutes..."

tar -czf "$ARCHIVE_NAME" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude=".ipynb_checkpoints" \
    --exclude="*.tar.gz" \
    "$DEPLOYMENT_FOLDER"

ARCHIVE_SIZE=$(du -sh "$ARCHIVE_NAME" | cut -f1)

echo "✅ Archive created: $ARCHIVE_NAME"
echo "   Compressed size: $ARCHIVE_SIZE"
echo ""

# ------------------------------------------------------------------------------
# 6. Display summary and next steps
# ------------------------------------------------------------------------------
echo "================================================================"
echo "✅ HPC DEPLOYMENT PREPARATION COMPLETE"
echo "================================================================"
echo ""
echo "📁 Deployment folder: $SCRIPT_DIR"
echo "📦 Archive: $PARENT_DIR/$ARCHIVE_NAME"
echo ""
echo "📊 Statistics:"
echo "   - RGB tiles: $NUM_RGB"
echo "   - Power line masks: $NUM_PL_MASKS"
echo "   - Vegetation masks: $NUM_VEG_MASKS"
echo "   - Deployment size: $TOTAL_SIZE"
echo "   - Archive size: $ARCHIVE_SIZE"
echo ""
echo "================================================================"
echo "NEXT STEPS"
echo "================================================================"
echo ""
echo "Option A: Transfer the folder directly"
echo "   rsync -avz --progress $SCRIPT_DIR/ user@hpc:/path/to/destination/"
echo ""
echo "Option B: Transfer the compressed archive (faster)"
echo "   scp $PARENT_DIR/$ARCHIVE_NAME user@hpc:/path/to/destination/"
echo "   # Then on HPC:"
echo "   tar -xzf $ARCHIVE_NAME"
echo ""
echo "After transfer, on the HPC:"
echo "   1. cd /path/to/destination"
echo "   2. Edit submit_pl_job.sh and submit_veg_job.sh (partition, email, modules)"
echo "   3. Setup environment: module load python/3.10 cuda/11.8"
echo "   4. Create venv: python -m venv venv"
echo "   5. Install deps: source venv/bin/activate && pip install -r requirements.txt"
echo "   6. Submit jobs: sbatch submit_pl_job.sh && sbatch submit_veg_job.sh"
echo ""
echo "For detailed instructions, see: $SCRIPT_DIR/README.md"
echo ""
echo "================================================================"
echo "Good luck with your training! 🚀"
echo "================================================================"
