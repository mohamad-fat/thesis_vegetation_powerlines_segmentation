# Point Transformer V2 - Detailed Technical Documentation

This document provides in-depth technical details about the Point Transformer V2 implementation for power line segmentation.

## Table of Contents

1. [Architecture Details](#architecture-details)
2. [Data Processing Pipeline](#data-processing-pipeline)
3. [Model Components](#model-components)
4. [Training Process](#training-process)
5. [Configuration Parameters](#configuration-parameters)
6. [File-by-File Documentation](#file-by-file-documentation)
7. [Algorithm Details](#algorithm-details)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Architecture Details

### Point Transformer V2 Overview

The Point Transformer V2 is a neural network architecture designed for point cloud processing. It uses self-attention mechanisms to capture both local geometric patterns and long-range dependencies in 3D point clouds.

```
Architecture Flow:
Input Points (B, N, 3)
    ↓
Input Embedding Layer
    Linear(3 → 64) → ReLU → Linear(64 → d_model)
    Output: (B, N, d_model)
    ↓
Transformer Blocks (× L layers)
    For each block:
        1. K-Nearest Neighbors (KNN)
        2. Position Encoding
        3. Multi-head Self-Attention
        4. Feed-Forward Network
        5. Layer Normalization
    Output: (B, N, d_model)
    ↓
Segmentation Head
    Linear(d_model → 256) → ReLU → Dropout(0.3)
    Linear(256 → 128) → ReLU → Dropout(0.3)
    Linear(128 → num_classes)
    Output: (B, N, 4)
```

**Dimensions:**
- `B`: Batch size
- `N`: Number of points per sample (4096)
- `d_model`: Model dimension (128 for MPS, 256 for GPU)
- `L`: Number of transformer layers (2 for MPS, 4 for GPU)
- `K`: Number of nearest neighbors (16)

### Transformer Block Details

Each Transformer Block consists of:

1. **K-Nearest Neighbors (KNN)**
   - Finds K nearest neighbors for each point
   - Uses Euclidean distance in 3D space
   - Creates local neighborhood for attention

2. **Position Encoding**
   - Computes relative positions between points and their neighbors
   - MLP: Linear(3 → d_model) → ReLU → Linear(d_model → d_model)
   - Encodes geometric relationships

3. **Self-Attention**
   - Query (Q): Linear transformation of point features
   - Key (K): Linear transformation of neighbor features + position encoding
   - Value (V): Linear transformation of neighbor features + position encoding
   - Attention weights: softmax(Q·K^T / √d_model)
   - Output: Attention-weighted sum of values

4. **Feed-Forward Network (FFN)**
   - MLP: Linear(d_model → 4×d_model) → ReLU → Linear(4×d_model → d_model)
   - Applied point-wise after attention

5. **Residual Connections & Normalization**
   - Residual connection after attention: features = LayerNorm(features + attention_output)
   - Residual connection after FFN: features = LayerNorm(features + ffn_output)

---

## Data Processing Pipeline

### 1. Raw Data Format

**Input**: LAS files (LiDAR point clouds)
- Each point has: X, Y, Z coordinates
- Each point has: semantic label (1=Vegetation, 2=PowerLines, 3=Pylons, 4=Others)

### 2. Preprocessing (`las_processor.py`)

#### Step 1: Load LAS File
```python
las = laspy.read(las_file_path)
points = np.column_stack([las.x, las.y, las.z])
labels = las.classification
```

#### Step 2: Filter by Filename
- Only processes files starting with `segmented_croped_chunk_`
- Ensures consistent data source

#### Step 3: Spatial Grid Chunking

Instead of random sampling, uses spatial grid:

```python
# Calculate grid dimensions
x_min, x_max = points[:, 0].min(), points[:, 0].max()
y_min, y_max = points[:, 1].min(), points[:, 1].max()

# Determine tile size based on point density
area = (x_max - x_min) * (y_max - y_min)
points_per_area = n_points / area
tile_area = chunk_size / points_per_area
tile_size = sqrt(tile_area)

# Create grid
n_tiles_x = ceil((x_max - x_min) / tile_size)
n_tiles_y = ceil((y_max - y_min) / tile_size)

# Assign points to grid cells
tile_x = floor((points[:, 0] - x_min) / tile_size)
tile_y = floor((points[:, 1] - y_min) / tile_size)
```

**Benefits:**
- Preserves spatial coherence
- Like image tiling for 2D images
- Better for visualization and testing

#### Step 4: Normalization

For each chunk:
```python
# Center around origin
centroid = points.mean(axis=0)
normalized_points = points - centroid

# Store centroid for inverse transformation
metadata = {
    'centroid': centroid,
    'original_bounds': [x_min, x_max, y_min, y_max, z_min, z_max]
}
```

#### Step 5: Label Mapping

Original → Processed:
- 5 (High Vegetation) → 1
- 10 (Power Lines) → 2
- 11 (Pylons) → 3
- 0 (Others) → 4

Then 1-indexed → 0-indexed (during training):
- 1 → 0 (High Vegetation)
- 2 → 1 (Power Lines)
- 3 → 2 (Pylons)
- 4 → 3 (Others)

#### Step 6: Save as PyTorch Tensor

```python
chunk_data = {
    'input_pcd': torch.tensor(normalized_points, dtype=torch.float32),
    'semantic_labels': torch.tensor(labels, dtype=torch.int64),
    'metadata': metadata
}
torch.save(chunk_data, output_path)
```

### 3. Dataset Class (`PowerLineDataset`)

#### Loading
```python
sample = torch.load(sample_path)
points = sample['input_pcd']  # (N, 3)
labels = sample['semantic_labels']  # (N,)
```

#### Point Sampling

**If N > 4096 (too many points):**

Option A: Farthest Point Sampling (FPS)
```python
# Start with random point
farthest = random_point
centroids = [farthest]

for i in range(num_samples):
    # Find distances to farthest point
    distances = compute_distances(points, farthest)

    # Update minimum distances
    min_distances = min(min_distances, distances)

    # Pick point with maximum minimum distance
    farthest = argmax(min_distances)
    centroids.append(farthest)
```

Option B: Random Sampling
```python
indices = random.choice(N, 4096, replace=False)
points = points[indices]
labels = labels[indices]
```

**If N < 4096 (too few points):**
```python
# Duplicate random points to reach 4096
num_needed = 4096 - N
indices = random.choice(N, num_needed, replace=True)
extra_points = points[indices]
extra_labels = labels[indices]
points = concatenate([points, extra_points])
labels = concatenate([labels, extra_labels])
```

---

## Model Components

### 1. K-Nearest Neighbors (KNN)

```python
def knn(x, k):
    """
    Find k nearest neighbors

    Args:
        x: (B, 3, N) - point coordinates
        k: number of neighbors

    Returns:
        idx: (B, N, k) - indices of k nearest neighbors for each point
    """
    # Compute pairwise distances
    inner = -2 * matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = sum(x**2, dim=1, keepdim=True)  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)  # (B, N, N)

    # Get top-k
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx
```

**Complexity**: O(N²) for distance computation, O(N²log(k)) for top-k selection

### 2. Neighbor Gathering

**Problem**: Need to gather features of K neighbors for each point

**Old Approach (Buggy)**:
```python
# This could fail with 0-dimensional tensors
idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, C)
neighbor_features = torch.gather(features, 1, idx_expanded)
```

**New Approach (Safe)**:
```python
# Flatten and use direct indexing
B, N, C = features.shape
idx_base = torch.arange(0, B, device=device).view(-1, 1, 1) * N
idx_expanded = (idx + idx_base).view(B * N * k)

features_flat = features.view(B * N, C)
neighbor_features = features_flat[idx_expanded].view(B, N, k, C)
```

**Why this works**:
- Converts batch indices to global indices
- Uses simple 1D indexing on flattened tensor
- Reshapes back to (B, N, k, C)
- No dimension mismatches

### 3. Position Encoding

```python
# Compute relative positions
relative_pos = positions.unsqueeze(2) - neighbor_positions  # (B, N, k, 3)

# Encode with MLP
pos_encoding = pos_mlp(relative_pos)  # (B, N, k, d_model)

# Add to key and value
k_feat = k_linear(neighbor_features) + pos_encoding
v_feat = v_linear(neighbor_features) + pos_encoding
```

**Purpose**: Encodes geometric relationships between points and neighbors

### 4. Attention Mechanism

```python
# Query: current point features
q = q_linear(features).unsqueeze(2)  # (B, N, 1, C)

# Key: neighbor features + position
k = k_linear(neighbor_features) + pos_encoding  # (B, N, k, C)

# Value: neighbor features + position
v = v_linear(neighbor_features) + pos_encoding  # (B, N, k, C)

# Compute attention weights
attn = matmul(q, k.transpose(-2, -1)) / sqrt(C)  # (B, N, 1, k)
attn = softmax(attn, dim=-1)  # (B, N, 1, k)

# Weighted sum
out = matmul(attn, v).squeeze(2)  # (B, N, C)
```

**Interpretation**:
- Attention weights determine importance of each neighbor
- High weight = neighbor is relevant for classifying this point
- Position encoding ensures geometric awareness

---

## Training Process

### 1. Initialization

```python
# Create model
model = PointTransformerLightning(
    num_classes=4,
    d_model=128,
    num_layers=2,
    k=16,
    lr=1e-3,
    class_weights=calculated_weights
)

# Create data module
dm = PowerLineDataModule(
    data_dir='path/to/data',
    batch_size=2,
    num_points=4096,
    use_fps=True
)
```

### 2. Forward Pass

```python
# Input: batch of point clouds
points = batch['points']  # (B, 4096, 3)
labels = batch['labels']  # (B, 4096)

# Forward
logits = model(points)  # (B, 4096, 4)

# Loss
loss = CrossEntropyLoss(logits.view(-1, 4), labels.view(-1))
```

### 3. Loss Calculation

**Cross-Entropy Loss with Class Weights**:
```python
# Class weights (handle imbalance)
weights = [1.0, 5.0, 10.0, 1.0]  # Emphasize power lines and pylons

# Per-point loss
loss = -sum(weight[y] * log(softmax(logits)[y]))

# Average over all points
total_loss = mean(loss)
```

**Why class weights?**
- Power lines: ~0-5% of points
- Pylons: ~0-2% of points
- Vegetation: ~60-70% of points
- Others: ~25-35% of points

Without weights, model ignores rare classes.

### 4. Backpropagation

```python
# Compute gradients
loss.backward()

# Clip gradients (prevent exploding gradients)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Update weights
optimizer.step()
optimizer.zero_grad()
```

### 5. Learning Rate Scheduling

**Cosine Annealing**:
```python
lr(epoch) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * epoch / max_epochs))
```

- Starts at `lr_max = 1e-3`
- Gradually decreases to `lr_min = 1e-6`
- Smooth decay helps final convergence

### 6. Validation

Every epoch:
```python
# Evaluate on validation set
for batch in val_loader:
    logits = model(batch['points'])
    pred = logits.argmax(dim=-1)

    # Compute IoU per class
    for cls in range(4):
        pred_mask = (pred == cls)
        true_mask = (labels == cls)
        intersection = (pred_mask & true_mask).sum()
        union = (pred_mask | true_mask).sum()
        iou[cls] = intersection / union

    mean_iou = mean(iou)
```

### 7. Checkpointing

**Save best model**:
```python
if val_miou > best_miou:
    save_checkpoint({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_miou': val_miou,
        'hyperparameters': {...}
    })
```

---

## Configuration Parameters

### Model Architecture

| Parameter | MPS/CPU | GPU | Description |
|-----------|---------|-----|-------------|
| `D_MODEL` | 128 | 256 | Feature dimension |
| `NUM_LAYERS` | 2 | 4 | Number of transformer blocks |
| `NUM_HEADS` | 8 | 8 | Attention heads (not used in current impl) |
| `K_NEIGHBORS` | 16 | 16 | Number of neighbors for attention |
| `DROPOUT` | 0.3 | 0.3 | Dropout rate in segmentation head |

**Model size calculation**:
- 2 layers, 128 dims: ~473K parameters (~1.9 MB)
- 4 layers, 256 dims: ~1.8M parameters (~7.2 MB)

### Training Hyperparameters

| Parameter | MPS/CPU | GPU | Description |
|-----------|---------|-----|-------------|
| `BATCH_SIZE` | 2 | 16 | Samples per batch |
| `MAX_EPOCHS` | 10-50 | 100 | Maximum training epochs |
| `LEARNING_RATE` | 1e-3 | 1e-3 | Initial learning rate |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 | L2 regularization |
| `GRADIENT_CLIP` | 1.0 | 1.0 | Gradient clipping threshold |

### Data Processing

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_POINTS` | 4096 | Points per sample |
| `USE_FPS` | True | Use Farthest Point Sampling |
| `NUM_WORKERS` | 0 (MPS), 4 (GPU) | Parallel data loading |

### Callbacks

| Parameter | Value | Description |
|-----------|-------|-------------|
| `EARLY_STOPPING_PATIENCE` | 20 | Epochs to wait before stopping |
| `SAVE_TOP_K` | 3 | Number of best checkpoints to keep |
| `CHECKPOINT_MONITOR` | val_miou | Metric to monitor |
| `LOG_EVERY_N_STEPS` | 10 | Logging frequency |

---

## File-by-File Documentation

### `point_transformer_training.ipynb`

**Purpose**: Train Point Transformer V2 model locally (MPS/CPU)

**Sections**:
1. **Configuration**: All tunable parameters
2. **Imports**: Required libraries
3. **Dataset Classes**: Data loading and sampling
4. **Utility Functions**: Device detection, class weight calculation
5. **Model Classes**: KNN, TransformerBlock, PointTransformerV2
6. **Class Weights**: Calculate or use provided weights
7. **Setup**: Initialize data module and model
8. **Callbacks**: Checkpointing, early stopping, learning rate monitoring
9. **Trainer**: Configure Lightning Trainer
10. **Training**: Run training loop
11. **Testing**: Evaluate best model
12. **Visualization**: Generate prediction visualizations
13. **Model Loading**: Load saved checkpoint

**Key Features**:
- All configuration at top for easy tuning
- Automatic device detection (MPS/CUDA/CPU)
- Class weight calculation for imbalanced data
- TensorBoard logging
- Checkpoint saving

### `point_transformer_training_colab.ipynb`

**Purpose**: Train on Google Colab with GPU acceleration

**Additional Features**:
- Google Drive mounting
- Automatic checkpoint backup to Drive
- GPU detection and configuration
- 16-bit mixed precision training
- Checkpoint download functionality
- Embedded TensorBoard

**Differences from local version**:
- Larger batch size (16 vs 2)
- Larger model (256 dims, 4 layers vs 128 dims, 2 layers)
- More data workers (4 vs 0)
- Mixed precision enabled
- Drive persistence

### `model_evaluation.ipynb`

**Purpose**: Evaluate trained model and generate visualizations

**Sections**:
1. **Setup**: Load dependencies
2. **Dataset**: Load test dataset
3. **Model**: Load trained checkpoint
4. **Configuration**: Set evaluation parameters
5. **Load Model**: Load from checkpoint with error handling
6. **Load Test Data**: Prepare test dataset
7. **Evaluation**: Compute metrics on full test set
8. **Display Results**: Print per-class metrics
9. **Visualize Metrics**: Bar charts of IoU, precision, recall, F1
10. **3D Visualization**: Generate 3D point cloud visualizations
11. **Generate Samples**: Visualize predictions vs ground truth
12. **Confusion Matrix**: Show classification confusion
13. **Summary Report**: Generate text report

**Metrics Computed**:
- Per-class IoU (Intersection over Union)
- Per-class Precision
- Per-class Recall
- Per-class F1-Score
- Overall Mean IoU
- Confusion Matrix

### `las_processor.py`

**Purpose**: Preprocess raw LAS files into PyTorch tensors

**Classes**:

#### `LASProcessor`
**Methods**:
- `process_all_files()`: Process all LAS files in directory
- `process_single_file()`: Process one LAS file
- `split_into_chunks()`: Split point cloud into spatial grid chunks
- `map_labels()`: Map LAS classification codes to model labels
- `save_chunk()`: Save chunk as PyTorch tensor

#### `InverseLASProcessor`
**Methods**:
- `inverse_map_labels()`: Map model labels back to LAS codes
- `denormalize_points()`: Add centroid back to points
- `load_and_reconstruct_chunk()`: Load single chunk
- `reconstruct_all_chunks_from_file()`: Reconstruct full file from chunks
- `compare_with_original()`: Compare reconstructed vs original
- `visualize_sample_chunk()`: Visualize single chunk

**Usage**:
```python
# Process LAS files
processor = LASProcessor(
    data_dir='path/to/las/files',
    output_dir='processed_data',
    chunk_size=4096,
    split_ratio=(0.8, 0.2)
)
processor.process_all_files()

# Verify processing
inverse = InverseLASProcessor(
    processed_dir='processed_data',
    original_dir='path/to/las/files'
)
inverse.compare_with_original('file.las', percentage=10.0)
```

### `visualize_test_set.ipynb`

**Purpose**: Visualize test set samples before training

**Features**:
- Load processed test data
- Display 3D point clouds
- Show class distribution
- Inspect spatial grid chunks

### `inference_on_las.ipynb`

**Purpose**: Run inference on new LAS files

**Workflow**:
1. Load trained model
2. Load raw LAS file
3. Split into chunks
4. Predict labels for each chunk
5. Combine predictions
6. Save as new LAS file with predictions

---

## Algorithm Details

### Farthest Point Sampling (FPS)

**Goal**: Sample N points from M points (N < M) while maintaining geometric coverage

**Algorithm**:
```
1. Initialize:
   - Pick random starting point p₀
   - distances = [∞, ∞, ..., ∞]  (M values)
   - sampled = [p₀]

2. For i = 1 to N-1:
   a. For each point pⱼ:
      - distances[j] = min(distances[j], dist(pⱼ, sampled[-1]))
   b. next_point = argmax(distances)
   c. sampled.append(next_point)

3. Return sampled
```

**Time Complexity**: O(N × M)

**Benefits**:
- Maintains geometric diversity
- Better than random sampling for point clouds
- Preserves shape structure

### K-Nearest Neighbors

**Goal**: Find K nearest points for each point

**Naive Algorithm**:
```
For each point p:
    distances = [dist(p, q) for all q]
    neighbors = argsort(distances)[:K]
```
**Time**: O(N² log N)

**Our Implementation**:
```
# Vectorized computation
pairwise_dist = compute_all_distances(points)  # O(N²)
neighbors = topk(pairwise_dist, K)  # O(N² log K)
```
**Time**: O(N² log K)

**Space**: O(N²) for distance matrix

### Attention Mechanism

**Standard Transformer Attention**:
```
Q = W_q × X
K = W_k × X
V = W_v × X
Attention = softmax(Q·K^T / √d) · V
```

**Point Transformer Attention**:
```
For each point i:
    neighbors = KNN(i, K)
    Q_i = W_q × features[i]

    For each neighbor j:
        position_encoding = MLP(position[i] - position[j])
        K_j = W_k × features[j] + position_encoding
        V_j = W_v × features[j] + position_encoding

    attention_weights = softmax(Q_i · K^T / √d)
    output[i] = sum(attention_weights · V)
```

**Key Difference**: Position encoding based on relative 3D coordinates

---

## Performance Optimization

### 1. Mixed Precision Training (GPU Only)

**16-bit vs 32-bit**:
- 16-bit floats: 2 bytes per number
- 32-bit floats: 4 bytes per number
- **2x memory savings**
- **1.5-2x speed improvement on modern GPUs**

**How it works**:
```python
# Forward pass in FP16
with autocast():
    logits = model(points.half())
    loss = criterion(logits, labels)

# Backward pass with loss scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Trade-offs**:
- Slightly less precise
- May need loss scaling to prevent underflow
- Not beneficial on CPU/MPS

### 2. Data Loading Optimization

**Parallel Workers**:
```python
DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # 4 parallel processes
    pin_memory=True  # Pin memory for faster GPU transfer
)
```

**Benefits**:
- Data loading happens in parallel with training
- GPU never waits for data
- **Can reduce training time by 30-50%**

**Recommendation**:
- CPU: num_workers=0
- MPS: num_workers=0 (MPS has issues with multiprocessing)
- GPU: num_workers=4-8

### 3. Gradient Accumulation

**For larger effective batch size**:
```python
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Effect**: Simulates batch_size × accumulation_steps

### 4. Checkpoint Efficiency

**Save only essential data**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'val_miou': val_miou
}
# Don't save: training data, intermediate activations, full history
```

---

## Troubleshooting Guide

### Error: `IndexError: too many indices for tensor of dimension 0`

**Cause**: Labels tensor was squeezed to 0 dimensions (scalar)

**Fix**:
```python
# Bad
labels = sample['semantic_labels'].squeeze()  # May become scalar

# Good
labels = sample['semantic_labels'].reshape(-1)  # Always 1D
```

**Location**: `PowerLineDataset.__getitem__()` in both training and evaluation notebooks

### Error: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce batch size
2. Reduce model size (d_model, num_layers)
3. Reduce num_points per sample
4. Enable gradient checkpointing
5. Use mixed precision training

**Example**:
```python
# Before (OOM on 12GB GPU)
BATCH_SIZE = 32
D_MODEL = 512
NUM_LAYERS = 6

# After (fits on 12GB GPU)
BATCH_SIZE = 16
D_MODEL = 256
NUM_LAYERS = 4
```

### Error: Missing keys in state_dict

**Cause**: Model architecture mismatch between training and evaluation

**Symptoms**:
```
Missing key(s): "blocks.0.k_linear.weight"
Unexpected key(s): "blocks.0.k.weight"
```

**Fix**: Ensure model definitions match:
- Training notebook uses `k_linear`
- Evaluation notebook also uses `k_linear`
- Delete old incompatible checkpoints

### Warning: `num_workers > 0` on MPS

**Cause**: MPS doesn't support multiprocessing well

**Fix**:
```python
# For MPS/Apple Silicon
NUM_WORKERS = 0

# For CUDA
NUM_WORKERS = 4
```

### Low Validation mIoU

**Possible causes**:

1. **Class imbalance not addressed**
   - Check class weights are calculated
   - Ensure weights aren't too small (< 0.01)

2. **Insufficient training**
   - Train for more epochs
   - Check if loss is still decreasing

3. **Model too small**
   - Increase d_model
   - Add more layers

4. **Overfitting**
   - Increase dropout
   - Add data augmentation
   - Reduce model size

5. **Data quality issues**
   - Check label accuracy
   - Visualize some samples
   - Ensure proper preprocessing

### Slow Training Speed

**Benchmarks**:
- CPU: ~5-10 min/epoch (not recommended)
- MPS (M1/M2): ~1-2 min/epoch
- T4 GPU: ~30-60 sec/epoch
- A100 GPU: ~10-20 sec/epoch

**If slower than expected**:

1. **Check device**:
   ```python
   print(torch.cuda.is_available())  # Should be True for GPU
   ```

2. **Enable optimizations**:
   ```python
   # GPU only
   USE_MIXED_PRECISION = True
   NUM_WORKERS = 4
   ```

3. **Profile bottlenecks**:
   ```python
   # Time data loading
   start = time.time()
   batch = next(iter(dataloader))
   print(f"Data loading: {time.time() - start:.2f}s")

   # Time forward pass
   start = time.time()
   output = model(batch['points'])
   print(f"Forward pass: {time.time() - start:.2f}s")
   ```

---

## Advanced Topics

### Custom Data Augmentation

```python
class AugmentedDataset(PowerLineDataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        points = sample['points']

        # Random rotation around Z-axis
        angle = np.random.uniform(0, 2*np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        points = points @ rotation

        # Random jittering
        points += np.random.normal(0, 0.01, points.shape)

        sample['points'] = points
        return sample
```

### Transfer Learning

```python
# Load pretrained model
checkpoint = torch.load('pretrained_model.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])

# Freeze early layers
for param in model.model.input_embed.parameters():
    param.requires_grad = False

for block in model.model.blocks[:2]:
    for param in block.parameters():
        param.requires_grad = False

# Fine-tune only segmentation head
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### Ensemble Predictions

```python
# Load multiple models
models = [
    PointTransformerLightning.load_from_checkpoint('model1.ckpt'),
    PointTransformerLightning.load_from_checkpoint('model2.ckpt'),
    PointTransformerLightning.load_from_checkpoint('model3.ckpt'),
]

# Average predictions
logits_ensemble = 0
for model in models:
    logits_ensemble += model(points) / len(models)

predictions = logits_ensemble.argmax(dim=-1)
```

---

## Appendix

### Mathematical Notation

- `B`: Batch size
- `N`: Number of points per sample
- `D`: Point dimension (usually 3 for XYZ)
- `C` or `d_model`: Feature dimension
- `L`: Number of layers
- `K`: Number of neighbors
- `⊙`: Element-wise multiplication
- `⊕`: Element-wise addition
- `∥·∥`: Euclidean distance

### Glossary

- **Point Cloud**: Set of 3D points representing a surface or object
- **Semantic Segmentation**: Assigning a class label to each point
- **Attention**: Mechanism to weight importance of different inputs
- **IoU (Intersection over Union)**: Metric for segmentation quality
- **FPS (Farthest Point Sampling)**: Downsampling method for point clouds
- **KNN (K-Nearest Neighbors)**: Finding K closest points
- **Transformer**: Neural network architecture using attention
- **MPS (Metal Performance Shaders)**: Apple's GPU acceleration
- **Mixed Precision**: Using 16-bit and 32-bit floats together

### References

- Point Transformer V2 paper: [arxiv.org/abs/2210.05666](https://arxiv.org/abs/2210.05666)
- PyTorch Lightning docs: [lightning.ai/docs](https://lightning.ai/docs)
- LAStools documentation: [rapidlasso.com](https://rapidlasso.com)

---

## Changelog

**v1.0** (Current)
- Initial implementation with Point Transformer V2
- Support for MPS, CUDA, and CPU
- Spatial grid chunking
- Fixed tensor dimension bugs
- Google Colab notebook
- Comprehensive evaluation metrics

**Future Improvements**:
- Multi-scale feature aggregation
- Test-time augmentation
- Real-time inference optimization
- Support for additional LiDAR features (intensity, return number)
- Interactive web-based visualization
