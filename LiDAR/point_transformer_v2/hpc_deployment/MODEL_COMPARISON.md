# Model Architecture Comparison

## Available Models

### 1. Point Transformer V2 (`train_point_transformer_hpc.py`)

**Architecture:**
- Transformer-based architecture for 3D point clouds
- Local attention mechanism with KNN
- Position encoding for spatial awareness

**Key Features:**
- Model size: ~1M parameters
- Layers: 3 transformer blocks
- Dimension: 156D
- K-neighbors: 16
- Attention: Multi-head self-attention on local neighborhoods

**Strengths:**
- ✅ Captures long-range dependencies through attention
- ✅ Position-aware feature learning
- ✅ Good for structured point clouds
- ✅ Smaller model size

**Training:**
```bash
sbatch submit_lidar_job.sh
```

**Output:** `outputs_lidar/`

---

### 2. PointNet++ with Attention (`train_pointnetpp_attention_hpc.py`)

**Architecture:**
- Hierarchical point cloud processing
- Set Abstraction (SA) with integrated attention modules
- Feature Propagation (FP) for dense prediction

**Key Features:**
- Model size: ~2.5M parameters
- Hierarchical: 4 SA layers + 4 FP layers
- Attention: Spatial attention in each SA layer
- Sampling: Farthest Point Sampling (FPS)
- Grouping: Ball query for local regions

**Structure:**
```
Input (4096 points)
  ↓
SA1 (1024 points, 32 neighbors) + Attention
  ↓
SA2 (256 points, 32 neighbors) + Attention
  ↓
SA3 (64 points, 32 neighbors) + Attention
  ↓
SA4 (16 points, 32 neighbors) + Attention
  ↓
FP4 → FP3 → FP2 → FP1
  ↓
Output (4096 point predictions)
```

**Strengths:**
- ✅ Hierarchical feature extraction
- ✅ Multi-scale spatial understanding
- ✅ Enhanced features through attention
- ✅ Proven architecture (PointNet++ backbone)
- ✅ Better for varying point densities

**Training:**
```bash
sbatch submit_pointnetpp_job.sh
```

**Output:** `outputs_pointnetpp/`

---

## Detailed Comparison

| Aspect | Point Transformer V2 | PointNet++ + Attention |
|--------|---------------------|------------------------|
| **Paradigm** | Transformer | Hierarchical CNN |
| **Parameters** | ~1M | ~2.5M |
| **Attention** | Multi-head self-attention | Spatial attention per layer |
| **Spatial Handling** | KNN + Position encoding | FPS + Ball query |
| **Receptive Field** | Local (16 neighbors) | Multi-scale (hierarchical) |
| **Memory Usage** | Lower | Higher |
| **Training Speed** | Faster | Slower |
| **Inference Speed** | Faster | Slower |
| **Best For** | Structured clouds | Varying density |

---

## Which Model to Use?

### Use Point Transformer V2 if:
- ✅ You need faster training/inference
- ✅ Point cloud is uniformly sampled
- ✅ You want a smaller model
- ✅ Memory is limited
- ✅ You need position-aware features

### Use PointNet++ with Attention if:
- ✅ You want multi-scale features
- ✅ Point density varies significantly
- ✅ You need hierarchical understanding
- ✅ You can afford more computation
- ✅ You want enhanced spatial features

---

## Training Both Models

### Sequential Training
```bash
# Train Point Transformer first
sbatch submit_lidar_job.sh

# After completion, train PointNet++
sbatch submit_pointnetpp_job.sh
```

### Check Results
```bash
# Point Transformer
cat outputs_lidar/training_summary.json

# PointNet++
cat outputs_pointnetpp/training_summary.json
```

### Compare mIoU
```bash
# Compare validation mIoU scores
echo "Point Transformer mIoU:"
jq '.best_val_miou' outputs_lidar/training_summary.json

echo "PointNet++ mIoU:"
jq '.best_val_miou' outputs_pointnetpp/training_summary.json
```

---

## Expected Performance

Based on similar architectures on point cloud segmentation:

| Model | Expected mIoU | Training Time |
|-------|---------------|---------------|
| Point Transformer V2 | 0.55 - 0.65 | ~2-3 hours |
| PointNet++ + Attention | 0.60 - 0.70 | ~3-4 hours |

*Note: PointNet++ typically performs better but takes longer*

---

## Attention Mechanisms Explained

### Point Transformer Attention
```python
# Multi-head self-attention on K-nearest neighbors
Q = Linear(features)
K = Linear(neighbor_features) + pos_encoding
V = Linear(neighbor_features) + pos_encoding
Attention = softmax(Q @ K.T / sqrt(d))
Output = Attention @ V
```

### PointNet++ Attention
```python
# Spatial attention on aggregated features
Query = Conv1D(features, channels // 8)
Key = Conv1D(features, channels // 8)
Value = Conv1D(features, channels)
Energy = Query @ Key.T
Attention = softmax(Energy)
Output = gamma * (Value @ Attention.T) + features
```

---

## For Your Thesis

### Contribution Statement
*"We integrated an attention module into PointNet++ to enhance spatial feature extraction, achieving X% improvement in mIoU compared to baseline PointNet++, and compared it with Point Transformer V2 architecture."*

### Comparison Table Template
| Model | Architecture | mIoU | High Veg IoU | Power Line IoU | Pylon IoU | Others IoU |
|-------|-------------|------|-------------|---------------|-----------|-----------|
| Point Transformer V2 | Transformer | ___ | ___ | ___ | ___ | ___ |
| PointNet++ + Attn | Hierarchical | ___ | ___ | ___ | ___ | ___ |

Fill in after training completes!

---

## Troubleshooting

### If PointNet++ OOM (Out of Memory)
```bash
# Reduce batch size
--batch_size 8  # or even 4

# Or reduce number of points
--num_points 2048  # instead of 4096
```

### If Training is Too Slow
```bash
# Reduce SA layers (modify train_pointnetpp_attention_hpc.py)
# Use fewer abstraction levels

# Or train on subset first
--max_train_samples 10000
```

---

Good luck with your model comparison! 🚀
