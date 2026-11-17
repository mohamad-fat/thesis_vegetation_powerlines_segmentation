# Point Transformer V2 for LiDAR Semantic Segmentation

## Overview
The method uses **Point Transformer V2**, a deep learning architecture designed for semantic segmentation of 3D LiDAR point clouds. The task is to classify power line infrastructure into 4 classes:
- **Class 0**: High Vegetation
- **Class 1**: Power Lines
- **Class 2**: Pylons
- **Class 3**: Others (ground, low vegetation, etc.)

## Model Architecture

**Point Transformer V2** is a transformer-based architecture that processes 3D point clouds directly without voxelization:

### 1. Input Embedding Layer
Transforms raw 3D coordinates (x, y, z) through a two-layer MLP:
```
3D coordinates → 64 dimensions → 256 dimensions
```

### 2. Transformer Blocks (4 layers, 256 dimensions each)

Each transformer block consists of:

- **K-Nearest Neighbors (KNN)**: For each point, finds k=16 nearest neighbors based on spatial proximity

- **Position Encoding**: MLP network encodes relative positions between each point and its neighbors
  ```
  Relative position (3D) → 256 → 256 dimensions
  ```

- **Local Self-Attention**:
  - Query (Q), Key (K), Value (V) projections on local neighborhoods
  - Position encoding added to both K and V to incorporate geometric information
  - Scaled dot-product attention: `attention = softmax(Q·K^T / √d)`
  - Aggregates information from local neighborhood

- **Feed-Forward Network**: Two-layer MLP with 4x expansion
  ```
  256 → 1024 → 256 dimensions
  ```

- **Residual Connections & Layer Normalization** after both attention and FFN modules

### 3. Segmentation Head
Three-layer MLP with dropout for final classification:
```
256 → 256 → 128 → 4 classes
```

**Total Parameters**: ~4.2 million

## Data Preprocessing

- **Point Sampling**: 8,192 points per training sample
- **Farthest Point Sampling (FPS)**: Used during training to downsample larger point clouds while preserving geometric structure and spatial coverage
- **Random Sampling with Replacement**: Used for upsampling smaller chunks to reach 8,192 points
- **Data Split**: Train/test split with 6,203 test chunks
- **Total Test Points**: 22,490,678 points

## Training Methodology

### Framework
PyTorch Lightning for distributed training and experiment tracking

### Optimization
- **Optimizer**: AdamW
  - Learning rate: 1e-3
  - Weight decay: 1e-4
- **Scheduler**: Cosine Annealing
  - T_max: 50 epochs
  - Minimum learning rate: 1e-6
- **Batch Size**: 16 samples
- **Max Epochs**: 50
- **Gradient Clipping**: 1.0 to prevent exploding gradients
- **Precision**: Full 32-bit precision (mixed precision caused NaN issues)

### Loss Function
Cross-Entropy Loss with **weighted classes** (see Class Imbalance section below)

### Regularization
- **Dropout**: 0.3 in segmentation head
- **Early Stopping**: Patience of 15 epochs, monitoring validation mean IoU

### Hardware
NVIDIA V100 GPU with CUDA optimizations (`cudnn.benchmark=True`)

## Handling Class Imbalance

The dataset exhibits **severe class imbalance**, which is a critical challenge in power line infrastructure detection:

### Data Distribution
- **High Vegetation**: ~60% of points (dominant class)
- **Others**: ~40% of points
- **Power Lines**: ~0.2% of points (300x rarer than vegetation)
- **Pylons**: ~0.01% of points (6000x rarer than vegetation)

### Strategies Employed

#### 1. Weighted Loss Function
The primary strategy uses **class-weighted Cross-Entropy loss** with manually tuned weights that reflect the inverse of class frequency:

```python
class_weights = [1.0, 10.0, 50.0, 1.5]
# [High Vegetation, Power Lines, Pylons, Others]
```

- **High Vegetation** (weight 1.0): Baseline weight for the most common class
- **Power Lines** (weight 10.0): 10x weight for rare but critical class (~0.2% of data)
- **Pylons** (weight 50.0): 50x weight for extremely rare class (~0.01% of data)
- **Others** (weight 1.5): Slightly increased weight to balance against vegetation

This weighting scheme penalizes the model much more heavily for misclassifying rare classes (power lines and pylons), forcing it to pay attention to minority classes during training.

#### 2. Mean IoU as Primary Metric
Instead of overall accuracy (which would be dominated by vegetation), the model monitoring uses **mean Intersection over Union (mIoU)**:
- Treats all classes equally regardless of frequency
- Prevents the model from achieving high accuracy by simply predicting the majority class
- Early stopping and best model selection based on validation mIoU

#### 3. Farthest Point Sampling (FPS)
During training, FPS is used instead of random sampling:
- Maintains better spatial coverage of the point cloud
- Preserves geometric structure and local neighborhoods
- Reduces bias toward dense regions (which may not contain rare classes)

#### 4. Safety Measures in Weighting
The code includes safeguards against extreme weights:
```python
# Replace near-zero weights with 1.0 to prevent numerical instability
class_weights_tensor = torch.where(
    class_weights_tensor < 0.01,
    torch.ones_like(class_weights_tensor),
    class_weights_tensor
)
```

### Impact of Class Imbalance Handling

The weighted loss function proved **highly effective for power lines** but **insufficient for pylons**:

- **Power Lines**: Despite being only 0.2% of data, achieved excellent 88% IoU with balanced precision (95.8%) and recall (91.6%)
- **Pylons**: Extremely rare class (0.01%) still struggled with 2.49% IoU. High precision (98.25%) but very low recall (2.49%) indicates the model learned to be conservative—predictions are accurate but most pylon points are missed

The pylon results suggest that class weighting alone may be insufficient for classes with extreme rarity (< 0.01% of data), and additional techniques such as:
- Focal loss
- Data augmentation specifically for rare classes
- Two-stage detection (detect then refine)
- Synthetic minority oversampling

may be necessary for further improvement.

## Results

### Overall Performance
- **Mean IoU**: 55.10%
- **Overall Accuracy**: 79.47%
- **Test Set**: 22,490,678 points across 6,203 chunks

### Per-Class Performance

| Class | IoU | Precision | Recall | F1-Score | Points | % of Data |
|-------|-----|-----------|--------|----------|---------|-----------|
| **High Vegetation** | 70.79% | 82.39% | 83.41% | 82.90% | 13,404,112 | 59.6% |
| **Power Lines** | **88.04%** | **95.80%** | **91.58%** | **93.64%** | 40,615 | 0.18% |
| **Pylons** | 2.49% | 98.25% | 2.49% | 4.86% | 2,246 | 0.01% |
| **Others** | 59.07% | 74.94% | 73.61% | 74.27% | 9,043,705 | 40.2% |

### Key Findings

1. **Excellent Power Line Detection**: Despite representing only 0.18% of the data, power lines achieved 88% IoU with high precision (95.8%) and recall (91.6%). The class weighting strategy was highly effective for this critical infrastructure component.

2. **Pylon Detection Challenge**: Pylons (only 0.01% of data, ~2,200 points) proved extremely difficult with 2.49% IoU. High precision (98.25%) but very low recall (2.49%) indicates the model is very conservative—when it predicts pylons, it's usually correct, but it misses most pylon points. The extreme class imbalance (6000x less common than vegetation) overwhelmed even aggressive class weighting (50x weight).

3. **Strong Vegetation Classification**: High vegetation achieved solid performance (70.79% IoU) as the dominant class, showing the model learned good general geometric features.

4. **Balanced "Others" Class**: The "Others" class (ground, low vegetation, etc.) achieved reasonable performance (59.07% IoU), benefiting from moderate class weighting.

5. **Class Imbalance Impact**: The results demonstrate that class weighting is effective for moderately rare classes (power lines at 0.2%) but may be insufficient for extremely rare classes (pylons at 0.01%) without additional techniques.

## Technical Implementation

- **File**: `LiDAR/point_transformer_v2/hpc_deployment/train_point_transformer_hpc.py`
- **Results**: `LiDAR/point_transformer_v2/outputs_lidar/full_reconstructions/full_metrics.txt`
- **Best Model**: `outputs_lidar/checkpoints/best_model.ckpt`
- **Training History**: `outputs_lidar/training_history.json`
