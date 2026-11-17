# Point Transformer V2 for Power Line Segmentation

A deep learning solution for semantic segmentation of LiDAR point clouds to detect power lines, pylons, and vegetation for power line encroachment analysis.

## Overview

This project uses Point Transformer V2 architecture to classify LiDAR point clouds into four categories:
- **High Vegetation** (Class 0): Trees and tall vegetation
- **Power Lines** (Class 1): Electrical transmission lines
- **Pylons** (Class 2): Power line support structures
- **Others** (Class 3): Ground, buildings, and other objects

## Quick Start

### 1. Training (Local - Apple Silicon or CPU)
```bash
# Open and run the training notebook
jupyter notebook point_transformer_training.ipynb
```

### 2. Training (Google Colab - GPU)
```bash
# Upload point_transformer_training_colab.ipynb to Google Colab
# Enable GPU: Runtime → Change runtime type → GPU
# Run all cells
```

### 3. Evaluation
```bash
# Open the evaluation notebook
jupyter notebook model_evaluation.ipynb
```

## Project Structure

```
point_transformer_v2/
├── README.md                              # This file (high-level overview)
├── DETAILED_DOCS.md                       # Detailed technical documentation
├── point_transformer_training.ipynb       # Training notebook (local)
├── point_transformer_training_colab.ipynb # Training notebook (GPU/Colab)
├── model_evaluation.ipynb                 # Model evaluation and visualization
├── visualize_test_set.ipynb              # Test set visualization
├── inference_on_las.ipynb                 # Inference on raw LAS files
├── las_processor.py                       # LiDAR data preprocessing
├── checkpoints/                           # Saved model checkpoints
└── lightning_logs/                        # TensorBoard training logs
```

## Key Features

- **Point Transformer V2 Architecture**: State-of-the-art attention mechanism for point clouds
- **GPU Acceleration**: Optimized for CUDA GPUs with mixed precision training
- **Apple Silicon Support**: Native MPS acceleration for M1/M2/M3 Macs
- **Class Balancing**: Automatic class weight calculation for imbalanced datasets
- **Spatial Grid Chunking**: Efficient processing of large point clouds
- **3D Visualization**: Interactive visualization of predictions vs ground truth

## Model Architecture

```
Input (N × 3 points)
    ↓
Input Embedding (3 → 64 → d_model)
    ↓
Transformer Blocks × L layers
    ├─ K-Nearest Neighbors
    ├─ Position Encoding
    ├─ Self-Attention
    └─ Feed-Forward Network
    ↓
Segmentation Head (d_model → 256 → 128 → 4 classes)
    ↓
Output (N × 4 logits)
```

## Training Configuration

### Local (Apple Silicon MPS)
- Batch size: 2
- Model: 128 dimensions, 2 layers
- Training time: ~1-2 min/epoch
- Memory: ~8 GB

### Google Colab (T4 GPU)
- Batch size: 16
- Model: 256 dimensions, 4 layers
- Training time: ~10-30 sec/epoch
- Memory: ~12 GB

## Results

Example performance metrics:
- **Mean IoU**: 0.42
- **Validation Accuracy**: 71%
- **Power Line Detection**: Specialized for critical infrastructure

## Data Format

Expected data structure:
```
data/processed_data_optimized/
├── train/
│   ├── chunk_0000.pt
│   ├── chunk_0001.pt
│   └── ...
├── test/
│   ├── chunk_0000.pt
│   └── ...
└── metadata.json
```

Each `.pt` file contains:
- `input_pcd`: Point cloud coordinates (N × 3)
- `semantic_labels`: Class labels (N,) with values 1-4

## Dependencies

```
torch >= 2.0
lightning >= 2.0
numpy
matplotlib
tqdm
```

## Workflow

1. **Data Preprocessing** (`las_processor.py`)
   - Load LAS files
   - Split into spatial grid chunks
   - Normalize coordinates
   - Save as PyTorch tensors

2. **Training** (`point_transformer_training.ipynb`)
   - Load preprocessed data
   - Calculate class weights
   - Train Point Transformer V2
   - Monitor with TensorBoard
   - Save best checkpoints

3. **Evaluation** (`model_evaluation.ipynb`)
   - Load trained model
   - Compute per-class metrics (IoU, F1, Precision, Recall)
   - Generate 3D visualizations
   - Create confusion matrix

4. **Inference** (`inference_on_las.ipynb`)
   - Load raw LAS files
   - Process and predict
   - Save predictions back to LAS format

## Tips

- **Start with small data**: Use `MAX_TRAIN_SAMPLES` parameter to test with subset
- **Monitor training**: Use TensorBoard to track loss and IoU metrics
- **Adjust batch size**: Reduce if you encounter out-of-memory errors
- **Use GPU**: Training on GPU is 3-5x faster than CPU/MPS

## Troubleshooting

**Out of Memory?**
- Reduce `BATCH_SIZE` in configuration
- Reduce `D_MODEL` or `NUM_LAYERS`
- Reduce `NUM_POINTS` (number of points per sample)

**Training too slow?**
- Use Google Colab with GPU
- Increase `BATCH_SIZE` if you have memory
- Set `NUM_WORKERS > 0` for parallel data loading

**Poor accuracy?**
- Train for more epochs
- Adjust class weights
- Use Farthest Point Sampling (`USE_FPS=True`)
- Increase model capacity (`D_MODEL`, `NUM_LAYERS`)

## Citation

If you use this code, please cite:
```
Point Transformer V2 for Power Line Segmentation
Vegetation Encroachment Detection System
```

## License

See project root for license information.

## Contact

For questions or issues, please refer to the project repository or contact the maintainers.
