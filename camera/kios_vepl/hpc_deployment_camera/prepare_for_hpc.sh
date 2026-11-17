#!/bin/bash
# ==============================================================================
# Prepare KIOS VEPL HPC Deployment
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
echo "KIOS VEPL HPC Deployment Preparation"
echo "================================================================"
echo ""

# ------------------------------------------------------------------------------
# 1. Check if processed data exists
# ------------------------------------------------------------------------------
echo "Step 1: Checking for preprocessed data..."

PROCESSED_DATA_DIR="$PROJECT_DIR/data/processed_data"

if [ ! -d "$PROCESSED_DATA_DIR" ]; then
    echo ""
    echo "❌ ERROR: Processed data not found!"
    echo ""
    echo "Expected location: $PROCESSED_DATA_DIR"
    echo ""
    echo "You need to run the preprocessing notebook first:"
    echo "  1. cd $PROJECT_DIR/notebooks"
    echo "  2. jupyter notebook preprocess_kios_vepl.ipynb"
    echo "  3. Execute all cells to generate data/processed_data/"
    echo ""
    exit 1
fi

# Check subdirectories
if [ ! -d "$PROCESSED_DATA_DIR/images" ]; then
    echo ""
    echo "❌ ERROR: Images directory not found!"
    echo "   Expected: $PROCESSED_DATA_DIR/images"
    echo ""
    exit 1
fi

if [ ! -d "$PROCESSED_DATA_DIR/veg_masks" ] && [ ! -d "$PROCESSED_DATA_DIR/pl_masks" ]; then
    echo ""
    echo "❌ ERROR: Mask directories not found!"
    echo ""
    echo "Expected structure:"
    echo "  data/processed_data/"
    echo "  ├── images/           # Shared images"
    echo "  ├── veg_masks/        # Vegetation masks"
    echo "  └── pl_masks/         # Power line masks"
    echo ""
    echo "Please run preprocess_kios_vepl.ipynb to generate both mask types."
    echo ""
    exit 1
fi

# Count files
NUM_IMAGES=$(ls "$PROCESSED_DATA_DIR/images"/*.png 2>/dev/null | wc -l)
NUM_VEG_MASKS=0
NUM_PL_MASKS=0

if [ -d "$PROCESSED_DATA_DIR/veg_masks" ]; then
    NUM_VEG_MASKS=$(ls "$PROCESSED_DATA_DIR/veg_masks"/*.png 2>/dev/null | wc -l)
fi

if [ -d "$PROCESSED_DATA_DIR/pl_masks" ]; then
    NUM_PL_MASKS=$(ls "$PROCESSED_DATA_DIR/pl_masks"/*.png 2>/dev/null | wc -l)
fi

echo "✅ Found processed data:"
echo "   Images: $NUM_IMAGES"
echo "   Vegetation masks: $NUM_VEG_MASKS"
echo "   Power line masks: $NUM_PL_MASKS"

if [ "$NUM_IMAGES" -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No image files found!"
    exit 1
fi

if [ "$NUM_VEG_MASKS" -eq 0 ] && [ "$NUM_PL_MASKS" -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No mask files found!"
    exit 1
fi

# Verify counts match
MISMATCH=0

if [ "$NUM_VEG_MASKS" -gt 0 ] && [ "$NUM_IMAGES" -ne "$NUM_VEG_MASKS" ]; then
    echo ""
    echo "⚠️  WARNING: Images ($NUM_IMAGES) and vegetation masks ($NUM_VEG_MASKS) don't match!"
    MISMATCH=1
fi

if [ "$NUM_PL_MASKS" -gt 0 ] && [ "$NUM_IMAGES" -ne "$NUM_PL_MASKS" ]; then
    echo ""
    echo "⚠️  WARNING: Images ($NUM_IMAGES) and power line masks ($NUM_PL_MASKS) don't match!"
    MISMATCH=1
fi

if [ "$MISMATCH" -eq 1 ]; then
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""

# ------------------------------------------------------------------------------
# 2. Copy processed data to HPC deployment folder
# ------------------------------------------------------------------------------
echo "Step 2: Copying processed data to HPC deployment folder..."

HPC_PROCESSED_DATA="$SCRIPT_DIR/processed_data"

# Remove old data if exists
if [ -d "$HPC_PROCESSED_DATA" ]; then
    echo "   Removing old processed data..."
    rm -rf "$HPC_PROCESSED_DATA"
fi

# Copy data
echo "   Copying images and masks..."
mkdir -p "$HPC_PROCESSED_DATA"
cp -r "$PROCESSED_DATA_DIR/images" "$HPC_PROCESSED_DATA/"

# Copy vegetation masks if they exist
if [ -d "$PROCESSED_DATA_DIR/veg_masks" ]; then
    echo "   Copying vegetation masks..."
    cp -r "$PROCESSED_DATA_DIR/veg_masks" "$HPC_PROCESSED_DATA/"
fi

# Copy power line masks if they exist
if [ -d "$PROCESSED_DATA_DIR/pl_masks" ]; then
    echo "   Copying power line masks..."
    cp -r "$PROCESSED_DATA_DIR/pl_masks" "$HPC_PROCESSED_DATA/"
fi

# Optional: Copy statistics if available
if [ -d "$PROCESSED_DATA_DIR/statistics" ]; then
    echo "   Copying statistics..."
    cp -r "$PROCESSED_DATA_DIR/statistics" "$HPC_PROCESSED_DATA/"
fi

echo "✅ Data copied successfully"
echo ""

# ------------------------------------------------------------------------------
# 3. Verify HPC deployment folder structure
# ------------------------------------------------------------------------------
echo "Step 3: Verifying HPC deployment structure..."

REQUIRED_FILES=(
    "README.md"
    "train_kios_vepl_pl_hpc.py"
    "train_kios_vepl_veg_hpc.py"
    "submit_pl_job.sh"
    "submit_veg_job.sh"
    "visualize_results.py"
    "visualize_predictions.py"
    "requirements.txt"
    "processed_data/images"
)

# Optional files (at least one mask type should exist)
OPTIONAL_MASK_DIRS=(
    "processed_data/veg_masks"
    "processed_data/pl_masks"
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

# Check that at least one mask directory exists
MASK_COUNT=0
for mask_dir in "${OPTIONAL_MASK_DIRS[@]}"; do
    if [ -d "$SCRIPT_DIR/$mask_dir" ]; then
        MASK_COUNT=$((MASK_COUNT + 1))
    fi
done

if [ "$MASK_COUNT" -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No mask directories found!"
    echo "   Expected at least one of:"
    for mask_dir in "${OPTIONAL_MASK_DIRS[@]}"; do
        echo "   - $mask_dir"
    done
    echo ""
    exit 1
fi

echo "✅ All required files present"
echo "✅ Found $MASK_COUNT mask type(s)"
echo ""

# ------------------------------------------------------------------------------
# 4. Calculate sizes
# ------------------------------------------------------------------------------
echo "Step 4: Calculating deployment size..."

TOTAL_SIZE=$(du -sh "$SCRIPT_DIR" | cut -f1)
DATA_SIZE=$(du -sh "$HPC_PROCESSED_DATA" | cut -f1)

echo "   Total deployment size: $TOTAL_SIZE"
echo "   Processed data size: $DATA_SIZE"
echo ""

# ------------------------------------------------------------------------------
# 5. Create compressed archive (optional)
# ------------------------------------------------------------------------------
echo "Step 5: Creating compressed archive..."

ARCHIVE_NAME="kios_vepl_hpc_$(date +%Y%m%d_%H%M%S).tar.gz"
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
echo "   - Images: $NUM_IMAGES"
echo "   - Vegetation masks: $NUM_VEG_MASKS"
echo "   - Power line masks: $NUM_PL_MASKS"
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
echo "   2. Edit submit_job.sh (partition, email, modules)"
echo "   3. Setup environment: module load python/3.10 cuda/11.8"
echo "   4. Create venv: python -m venv venv"
echo "   5. Install deps: source venv/bin/activate && pip install -r requirements.txt"
echo "   6. Submit job: sbatch submit_job.sh"
echo ""
echo "For detailed instructions, see: $SCRIPT_DIR/README.md"
echo ""
echo "================================================================"
echo "Good luck with your training! 🚀"
echo "================================================================"
