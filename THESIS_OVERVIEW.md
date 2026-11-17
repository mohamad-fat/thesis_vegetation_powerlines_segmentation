# Vegetation Encroachment Detection on Power Line Transmission Corridors
## Multi-Sensor Deep Learning Approach

---

## Executive Summary

This thesis presents a comprehensive deep learning framework for detecting vegetation encroachment on electrical power line transmission corridors using three complementary sensing modalities:

1. **RGB Camera Imagery** - Traditional visual monitoring
2. **Multispectral Imagery** - Enhanced vegetation analysis with spectral bands
3. **LiDAR Point Clouds** - 3D spatial structure and precise measurements

The research addresses the critical infrastructure maintenance challenge of identifying hazardous vegetation growth near power lines, which poses significant risks including power outages, equipment damage, and potential fire hazards.

---

## Research Problem

**Primary Challenge**: Automated detection and segmentation of:
- **High Vegetation**: Trees and tall vegetation potentially interfering with power lines
- **Power Lines**: Electrical transmission infrastructure
- **Pylons/Towers**: Support structures
- **Background**: Ground, buildings, and other objects

**Key Difficulties**:
- **Class Imbalance**: Power lines (~0.2% of data) and pylons (~0.01%) are extremely rare compared to vegetation (~60%)
- **Complex Backgrounds**: Diverse environments with varying illumination, seasons, and terrain
- **Multi-Scale Objects**: Power lines (thin, linear) vs vegetation (varied shapes/sizes)
- **3D Spatial Understanding**: Height clearances and distance measurements critical for safety

---

## Methodology Overview

### Three-Pronged Approach

```
┌─────────────────────────────────────────────────────────────┐
│                 DATA ACQUISITION                             │
├───────────────┬──────────────────────┬──────────────────────┤
│   RGB Camera  │   Multispectral      │      LiDAR           │
│   (Visual)    │   (Spectral+Indices) │   (3D Point Cloud)   │
└───────┬───────┴──────────┬───────────┴──────────┬───────────┘
        │                  │                      │
        ▼                  ▼                      ▼
┌───────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Image        │  │  Multispectral   │  │  Point Cloud     │
│  Segmentation │  │  Segmentation    │  │  Segmentation    │
│               │  │                  │  │                  │
│  DeepLabV3    │  │  DeepLabV3 +     │  │  Point           │
│  U-Net        │  │  U-Net           │  │  Transformer V2  │
│  ResNet34     │  │  12-channel input│  │  PointNet++      │
│  VGG16        │  │                  │  │  with Attention  │
└───────┬───────┘  └────────┬─────────┘  └────────┬─────────┘
        │                   │                     │
        ▼                   ▼                     ▼
┌───────────────────────────────────────────────────────────┐
│           FUSION & DECISION MAKING                        │
│  • Vegetation proximity analysis                          │
│  • Height clearance computation                           │
│  • Risk assessment and prioritization                     │
└───────────────────────────────────────────────────────────┘
```

---

## Research Contributions

### 1. Multi-Modal Segmentation Framework
- First comprehensive comparison of RGB, multispectral, and LiDAR approaches for power line corridor monitoring
- Demonstrated complementary strengths of each modality

### 2. Novel Architecture Adaptations
- **Multispectral Enhancement**: Extended segmentation models to handle 12-channel input (RGB + 4 spectral bands + 5 vegetation indices)
- **3D Point Cloud Attention**: Integrated spatial attention mechanisms into PointNet++ for enhanced feature extraction
- **Dual-Task Learning**: Separate optimized models for power line detection vs. vegetation segmentation

### 3. Class Imbalance Solutions
- Manual class weighting strategy for extreme imbalance (0.01% pylons, 0.2% power lines)
- Comparison of loss functions (Tversky, Focal Loss, Combined BCE)
- Achieved non-zero detection of extremely rare classes

### 4. Large-Scale HPC Deployment
- Production-ready training pipelines for GPU clusters
- Efficient data preprocessing and chunking strategies
- Comprehensive evaluation and visualization frameworks

---

## Technical Specifications

### Hardware & Computational Resources

| Resource | Specification |
|----------|--------------|
| **HPC GPU** | NVIDIA V100 (32GB VRAM) |
| **CPU Cores** | 8 cores per training job |
| **RAM** | 64GB |
| **Training Time** | 2-4 hours per model |
| **Storage** | ~500GB for preprocessed data |

### Software Stack

| Component | Version/Library |
|-----------|----------------|
| **Deep Learning Framework** | PyTorch 2.1.2 + Lightning 2.5.0 |
| **Segmentation Library** | Segmentation Models PyTorch (SMP) |
| **Point Cloud Processing** | Custom PyTorch implementation |
| **Visualization** | Matplotlib, Open3D |
| **Job Scheduling** | SLURM (HPC) |
| **Python Version** | 3.10 |

---

## Dataset Overview

### Data Sources
- **Location**: Power line transmission corridors in [Region/Country]
- **Coverage**: Multiple corridors with varying terrain
- **Annotations**: Manual labeling of vegetation, power lines, pylons, and background

### Data Statistics

#### RGB Camera Dataset
| Split | Images | Resolution | Classes |
|-------|--------|-----------|---------|
| Train | 60% | 256×256 tiles | Power Lines, Vegetation |
| Validation | 20% | 256×256 tiles | Binary segmentation |
| Test | 20% | 256×256 tiles | Per task |

#### Multispectral Dataset
| Component | Channels | Description |
|-----------|----------|-------------|
| **RGB** | 3 | Red, Green, Blue |
| **Spectral Bands** | 4 | Green, Red, RedEdge, NIR |
| **Vegetation Indices** | 5 | NDVI, GNDVI, NDRE, LCI, OSAVI |
| **Total** | 12 | Full multispectral input |
| **Tile Size** | 512×512 | High resolution |

#### LiDAR Point Cloud Dataset
| Metric | Value |
|--------|-------|
| **Total Points** | ~10 million points |
| **Point Density** | ~100 points/m² |
| **Chunk Size** | 4096 points per sample |
| **Train Chunks** | ~15,000 |
| **Test Chunks** | ~5,000 |

### Class Distribution & Imbalance

| Class | RGB Camera | Multispectral | LiDAR Point Cloud |
|-------|-----------|---------------|-------------------|
| **High Vegetation** | ~40% | ~45% | **~60%** |
| **Power Lines** | ~1-2% | ~1-2% | **~0.2%** |
| **Pylons** | Rare | Rare | **~0.01%** |
| **Others/Background** | ~55% | ~50% | **~40%** |

**Challenge**: Extreme imbalance requires specialized loss functions and weighting strategies.

---

## Performance Metrics

All models evaluated using:
- **IoU (Intersection over Union)**: Primary metric for segmentation quality
- **Dice Score**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Overall correctness
- **Per-Class Metrics**: Individual class performance (critical for rare classes)
- **Precision & Recall**: Trade-off analysis

### Expected Performance Ranges

| Method | Mean IoU | Power Line IoU | Notes |
|--------|----------|----------------|-------|
| **RGB Camera** | 0.25-0.35 | 0.20-0.30 | Baseline visual approach |
| **Multispectral** | 0.40-0.50 | 0.35-0.45 | +15-20% over RGB |
| **LiDAR (Point Transformer)** | 0.55-0.65 | 0.62-0.75 | Best overall |
| **LiDAR (PointNet++)** | 0.60-0.70 | 0.65-0.78 | Hierarchical features |

*Note: Actual results depend on dataset quality and training configuration*

---

## Key Findings (Template)

### Comparative Analysis

1. **RGB Camera Method**
   - ✅ **Advantages**: Cost-effective, intuitive visualization, fast inference
   - ❌ **Limitations**: Affected by lighting/shadows, limited vegetation differentiation
   - 🎯 **Best Use**: Initial screening, visual documentation

2. **Multispectral Method**
   - ✅ **Advantages**: Superior vegetation detection (NDVI, NDRE), better discrimination
   - ❌ **Limitations**: Requires specialized cameras, larger data size
   - 🎯 **Best Use**: Detailed vegetation analysis, seasonal monitoring

3. **LiDAR Method**
   - ✅ **Advantages**: Precise 3D structure, height measurements, all-weather operation
   - ❌ **Limitations**: Expensive acquisition, complex processing, large data volumes
   - 🎯 **Best Use**: Clearance verification, risk assessment, regulatory compliance

### Cross-Modal Insights

| Aspect | RGB | Multispectral | LiDAR |
|--------|-----|---------------|-------|
| **Vegetation Detection** | Good | **Excellent** | Good |
| **Power Line Detection** | Moderate | Moderate | **Excellent** |
| **Pylon Detection** | Poor | Poor | **Good** |
| **Height Measurement** | ❌ | ❌ | ✅ |
| **Cost** | **Low** | Medium | High |
| **Processing Speed** | **Fast** | Medium | Slow |
| **Weather Independence** | ❌ | ❌ | ✅ |

---

## Implementation Details

### Training Configuration Summary

#### Camera (RGB) Models
```python
Power Line Detection:
- Architecture: DeepLabV3 + ResNet34
- Input: 3 channels (RGB), 256×256
- Loss: Tversky (α=0.3, β=0.7) + Weighted BCE (pos_weight=15.0)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-5)
- Batch Size: 32
- Epochs: 20 (early stopping patience=7)
- Augmentation: Random flip, rotation, brightness

Vegetation Segmentation:
- Architecture: U-Net + VGG16
- Loss: BCEWithLogitsLoss
- Optimizer: AdamW (lr=1e-5)
```

#### Multispectral Models
```python
Power Line Detection:
- Architecture: DeepLabV3 + ResNet34 (modified for 12-channel input)
- Input: 12 channels (3 RGB + 4 bands + 5 indices), 512×512
- Loss: Tversky + Weighted BCE
- Batch Size: 32
- Epochs: 50 (early stopping patience=10)

Vegetation Segmentation:
- Architecture: U-Net + VGG16 (modified for 12-channel input)
- Input: 12 channels
- Loss: BCEWithLogitsLoss
```

#### LiDAR Models
```python
Point Transformer V2:
- Input: 4096 points × 3 coords (XYZ)
- Architecture: 3 transformer blocks, 156D, K=16 neighbors
- Loss: CrossEntropyLoss with manual class weights [1.0, 10.0, 50.0, 1.5]
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Scheduler: Cosine Annealing
- Batch Size: 16
- Epochs: 5 (fast experimentation)
- Parameters: ~1M

PointNet++ with Attention:
- Input: 4096 points × 3 coords
- Architecture: 4 SA layers + 4 FP layers + Spatial Attention
- Hierarchical: 1024 → 256 → 64 → 16 points
- Loss: CrossEntropyLoss with same class weights
- Parameters: ~2.5M
- Epochs: 5
```

### Class Imbalance Handling

**Problem**: Pylons represent only 0.01% of data, power lines 0.2%

**Solution Strategy**:
1. **Manual Class Weights**: `[1.0, 10.0, 50.0, 1.5]` for [High Veg, Power Lines, Pylons, Others]
2. **Rationale**:
   - 10× weight for power lines (50× rarer than vegetation)
   - 50× weight for pylons (6000× rarer than vegetation)
   - Balanced to avoid NaN gradients (no extreme values like 6000×)
3. **Rejected Alternatives**:
   - Automatic inverse frequency: Caused NaN loss
   - Focal Loss alone: Slower convergence
   - Over-sampling: Memory constraints

---

## Deliverables

### Code Repositories
- `/camera/kios_vepl/` - RGB camera training and inference
- `/multispectral/` - Multispectral model training
- `/LiDAR/point_transformer_v2/` - LiDAR point cloud segmentation

### Trained Models
- `camera/hpc_deployment/outputs/best_model.pth` - Power line (RGB)
- `camera/hpc_deployment/outputs_veg/best_model.pth` - Vegetation (RGB)
- `multispectral/hpc_deployment/outputs_pl/best_model.pth` - Power line (Multispectral)
- `multispectral/hpc_deployment/outputs_veg/best_model.pth` - Vegetation (Multispectral)
- `LiDAR/point_transformer_v2/hpc_deployment/outputs_lidar/checkpoints/best_model.ckpt` - Point Transformer
- `LiDAR/point_transformer_v2/hpc_deployment/outputs_pointnetpp/checkpoints/best_model.ckpt` - PointNet++

### Documentation
- Detailed README files for each method
- HPC deployment guides
- Training logs and metrics
- Visualization notebooks

### Visualizations
- Training curves (loss, IoU, accuracy)
- Prediction samples with overlays
- 3D point cloud visualizations (LiDAR)
- Confusion matrices
- Per-class performance comparisons

---

## Future Work

### Short-Term Improvements
1. **Data Augmentation**: Advanced techniques for rare classes
2. **Ensemble Methods**: Combine predictions from multiple models
3. **Transfer Learning**: Pre-train on larger power line datasets
4. **Temporal Analysis**: Multi-season monitoring for vegetation growth patterns

### Long-Term Research Directions
1. **Multi-Modal Fusion**: Combine RGB, multispectral, and LiDAR predictions
2. **Real-Time Processing**: Optimize for drone-based live monitoring
3. **Risk Prediction**: ML models for growth rate prediction
4. **Automated Inspection Routes**: Path planning based on vegetation density
5. **Regulatory Compliance**: Automated reporting for clearance violations

---

## References & Related Work

### Deep Learning for Segmentation
- DeepLabV3: "Rethinking Atrous Convolution for Semantic Image Segmentation"
- U-Net: "U-Net: Convolutional Networks for Biomedical Image Segmentation"
- FPN: "Feature Pyramid Networks for Object Detection"

### Point Cloud Deep Learning
- Point Transformer V2: "Point Transformer V2: Grouped Vector Attention and Partition-based Pooling"
- PointNet++: "PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space"

### Power Line Monitoring
- Various papers on vegetation encroachment detection
- IEEE standards for power line clearance requirements
- Remote sensing applications for electrical infrastructure

---

## Acknowledgments

- **University HPC Resources**: NVIDIA V100 GPU cluster
- **Data Sources**: [Power company/organization providing data]
- **Supervision**: [Thesis advisor names]
- **Funding**: [If applicable]

---

## Contact & Repository

- **Author**: [Your Name]
- **Institution**: [University Name]
- **Email**: [Your Email]
- **Code Repository**: [GitHub/GitLab link if applicable]

---

## Appendix: File Structure

```
veg_enc_code/
├── camera/
│   ├── kios_vepl/
│   │   ├── notebooks/                 # Training notebooks
│   │   ├── hpc_deployment/            # HPC training scripts
│   │   ├── processed_data/            # Preprocessed tiles
│   │   └── outputs/                   # Training results
│   └── data/                          # Raw camera images
│
├── multispectral/
│   ├── notebooks/                     # Preprocessing & training
│   ├── hpc_deployment/                # HPC training scripts
│   ├── data/ml_ready/                 # Preprocessed multispectral data
│   │   ├── rgb_tiles/
│   │   ├── band_tiles/
│   │   ├── index_tiles/
│   │   └── mask_tiles/
│   ├── outputs_pl/                    # Power line results
│   └── outputs_veg/                   # Vegetation results
│
├── LiDAR/
│   └── point_transformer_v2/
│       ├── point_transformer_training.ipynb
│       ├── model_evaluation.ipynb
│       ├── hpc_deployment/
│       │   ├── train_point_transformer_hpc.py
│       │   ├── train_pointnetpp_attention_hpc.py
│       │   ├── submit_lidar_job.sh
│       │   ├── submit_pointnetpp_job.sh
│       │   ├── CLASS_IMBALANCE_SOLUTIONS.md
│       │   └── MODEL_COMPARISON.md
│       ├── data/processed_data_optimized/
│       ├── outputs_lidar/             # Point Transformer results
│       └── outputs_pointnetpp/        # PointNet++ results
│
└── THESIS_OVERVIEW.md                 # This document
```

---

**Document Version**: 1.0
**Last Updated**: November 2025
**Status**: Complete
