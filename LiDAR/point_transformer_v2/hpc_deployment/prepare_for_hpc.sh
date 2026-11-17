#!/bin/bash

# ==============================================================================
# Prepare LiDAR Point Transformer HPC Deployment
# ==============================================================================
# This script packages all necessary files for HPC deployment
# Usage: bash prepare_for_hpc.sh
# ==============================================================================

echo "================================================================"
echo "Preparing LiDAR Point Transformer HPC Deployment"
echo "================================================================"
echo ""

# Define source and destination
SOURCE_DIR="."
ARCHIVE_NAME="lidar_hpc_deployment.tar.gz"

# Files to include
FILES=(
    "train_point_transformer_hpc.py"
    "assemble_predictions_hpc.py"
    "visualize_results.py"
    "submit_lidar_job.sh"
    "README.md"
)

# Check all files exist
echo "Checking files..."
MISSING_FILES=0
for file in "${FILES[@]}"; do
    if [ -f "$SOURCE_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (MISSING)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING_FILES file(s) missing!"
    echo "Please ensure all files are present before packaging."
    exit 1
fi

echo ""
echo "Creating archive..."

# Create tar.gz archive
tar -czf "$ARCHIVE_NAME" "${FILES[@]}"

if [ $? -eq 0 ]; then
    SIZE=$(du -h "$ARCHIVE_NAME" | cut -f1)
    echo "✓ Archive created: $ARCHIVE_NAME ($SIZE)"
    echo ""
    echo "================================================================"
    echo "HPC Deployment Package Ready!"
    echo "================================================================"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Transfer to HPC:"
    echo "   rsync -avz --progress $ARCHIVE_NAME mfatfa01@turing1.ucy.ac.cy:~/"
    echo ""
    echo "2. On HPC, extract and setup:"
    echo "   cd ~"
    echo "   tar -xzf $ARCHIVE_NAME"
    echo "   mkdir -p hpc_deployment"
    echo "   mv train_point_transformer_hpc.py hpc_deployment/"
    echo "   mv assemble_predictions_hpc.py hpc_deployment/"
    echo "   mv visualize_results.py hpc_deployment/"
    echo "   mv submit_lidar_job.sh hpc_deployment/"
    echo "   mv README.md hpc_deployment/"
    echo "   cd hpc_deployment"
    echo ""
    echo "3. Ensure data is available:"
    echo "   ls -lh ../data/processed_data_optimized/train/"
    echo "   ls -lh ../data/processed_data_optimized/test/"
    echo ""
    echo "4. Submit training job:"
    echo "   sbatch submit_lidar_job.sh"
    echo ""
    echo "5. Monitor job:"
    echo "   squeue -u \$USER"
    echo "   tail -f lidar_train_<job_id>.out"
    echo ""
    echo "================================================================"
else
    echo "ERROR: Failed to create archive!"
    exit 1
fi
