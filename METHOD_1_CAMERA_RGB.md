# Method 1: RGB Camera-Based Segmentation
## Traditional Visual Monitoring for Power Line Corridors

---

## Overview

The RGB camera method serves as the **baseline approach** for vegetation encroachment detection, leveraging traditional visual imaging similar to human inspection but automated through deep learning. This method uses standard color imagery (Red-Green-Blue channels) captured from drones or fixed cameras along power line corridors.

---

## Motivation & Use Case

### Why RGB Camera?

**Advantages**:
- ✅ **Cost-Effective**: Standard cameras are widely available and affordable
- ✅ **Intuitive**: Human-interpretable RGB imagery
- ✅ **Fast Acquisition**: Quick data collection via drones or fixed installations
- ✅ **Established Infrastructure**: Compatible with existing surveillance systems
- ✅ **Fast Inference**: Lightweight models for real-time processing

**Limitations**:
- ❌ **Lighting Dependent**: Affected by shadows, time of day, weather
- ❌ **Limited Spectral Info**: Cannot distinguish vegetation types effectively
- ❌ **No Height Data**: Cannot measure clearance distances
- ❌ **Seasonal Variation**: Performance varies with foliage changes

### Target Applications
1. **Initial Screening**: Quick surveys for potential encroachment areas
2. **Visual Documentation**: Before/after images for maintenance records
3. **Public Reports**: Easy-to-understand visualizations for non-technical stakeholders
4. **Real-Time Monitoring**: Lightweight deployment on edge devices

---

## Dataset Preparation

### Data Collection
- **Source**: Aerial imagery from drone flights along transmission corridors
- **Original Resolution**: High-resolution images (typically 4000×3000 or similar)
- **Coverage**: Multiple corridors under various conditions

### Preprocessing Pipeline

```python
# Implemented in: camera/kios_vepl/notebooks/preprocess_kios_vepl.ipynb

1. Image Tiling:
   - Divide large images into 256×256 tiles
   - Overlap: 0-50 pixels to ensure no boundary loss
   - Format: PNG for both images and masks

2. Mask Generation:
   - Task 1: Power line detection (binary mask)
   - Task 2: Vegetation segmentation (binary mask)
   - Annotation tool: [LabelMe,CVAT, or custom]

3. Quality Control:
   - Remove tiles with poor quality (blur, excessive shadow)
   - Ensure balanced distribution across scenes
   - Verify mask alignment with RGB tiles

4. Data Split:
   - Train: 60%
   - Validation: 20%
   - Test: 20%
   - Stratified by scene to avoid data leakage
```

### File Structure
```
camera/kios_vepl/processed_data/
├── images/
│   ├── tile_0001.png              # 256×256 RGB
│   ├── tile_0002.png
│   └── ...
├── masks/
│   ├── tile_0001_powerline.png    # Binary mask (0=background, 255=power line)
│   ├── tile_0001_vegetation.png   # Binary mask (0=background, 255=vegetation)
│   └── ...
└── statistics/
    └── dataset_stats.json          # Class distribution, tile counts
```

---

## Model Architectures

### Task 1: Power Line Detection

**Architecture**: DeepLabV3 + ResNet34

**Rationale**:
- DeepLabV3 excels at capturing multi-scale context via Atrous Spatial Pyramid Pooling (ASPP)
- Critical for thin, linear structures like power lines
- ResNet34 provides good balance between capacity and speed

**Configuration**:
```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3(
    encoder_name='resnet34',
    encoder_weights='imagenet',      # Transfer learning from ImageNet
    in_channels=3,                    # RGB
    classes=1,                        # Binary segmentation
    activation=None                   # Apply sigmoid separately
)

# Model Size: ~26M parameters
```

**Loss Function**: **Combined Tversky + Weighted BCE**

```python
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, bce_weight=15.0):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_weight]))

    def forward(self, pred, target):
        return self.tversky(pred, target) + self.bce(pred, target)
```

**Why This Loss?**
- **Tversky Loss**: Controls false positive (α=0.3) vs false negative (β=0.7) trade-off
  - β=0.7 emphasizes recall (don't miss power lines!)
  - α=0.3 tolerates some false positives
- **Weighted BCE**: Heavily weights positive class (power line pixels)
  - pos_weight=15.0 compensates for ~1-2% power line pixels vs 98% background

**Training Hyperparameters**:
```python
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
batch_size = 32
epochs = 20
early_stopping_patience = 7

# Data augmentation
transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
```

---

### Task 2: Vegetation Segmentation

**Architecture**: U-Net + VGG16

**Rationale**:
- U-Net's encoder-decoder structure with skip connections preserves spatial details
- Ideal for segmenting irregular vegetation shapes
- VGG16 encoder provides strong feature extraction

**Configuration**:
```python
model = smp.Unet(
    encoder_name='vgg16',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1,
    activation=None
)

# Model Size: ~31M parameters
```

**Loss Function**: **BCEWithLogitsLoss**

```python
criterion = nn.BCEWithLogitsLoss()
```

**Why Simpler Loss?**
- Vegetation is more abundant (~40-50% of pixels)
- Less severe class imbalance compared to power lines
- Standard BCE provides good performance

**Training Hyperparameters**:
```python
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)  # Lower LR
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
batch_size = 32
epochs = 20
early_stopping_patience = 7

# Same augmentation as power line model
```

---

## Training Procedure

### Local Development
```bash
# Interactive training in Jupyter notebook
cd camera/kios_vepl/notebooks
jupyter notebook train_powerline.ipynb
jupyter notebook train_vegetation.ipynb
```

### HPC Deployment

**Directory**: `camera/kios_vepl/hpc_deployment/`

**Key Scripts**:
1. `train_kios_vepl_hpc.py` - Power line training
2. `train_kios_vepl_veg_hpc.py` - Vegetation training
3. `submit_job.sh` - SLURM job for power line
4. `submit_veg_job.sh` - SLURM job for vegetation
5. `visualize_results.py` - Generate training plots
6. `visualize_predictions.py` - Visualize model outputs

**SLURM Configuration**:
```bash
#SBATCH --job-name=kios_vepl_train
#SBATCH --partition=GPU
#SBATCH --gres=gpu:v100:1         # Single V100 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
```

**Training Command**:
```bash
# Power line detection
python train_kios_vepl_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./outputs \
    --batch_size 32 \
    --epochs 20 \
    --architecture DeepLabV3 \
    --encoder resnet34 \
    --tversky_alpha 0.3 \
    --tversky_beta 0.7 \
    --bce_pos_weight 15.0 \
    --lr 0.0001 \
    --num_workers 8 \
    --early_stop_patience 7 \
    --device cuda \
    --amp  # Mixed precision training

# Vegetation segmentation
python train_kios_vepl_veg_hpc.py \
    --data_dir ./processed_data \
    --output_dir ./outputs_veg \
    --batch_size 32 \
    --epochs 20 \
    --architecture Unet \
    --encoder vgg16 \
    --lr 0.00001 \
    --num_workers 8 \
    --early_stop_patience 7 \
    --device cuda \
    --amp
```

**Training Time**:
- Power line model: ~2-3 hours (V100 GPU)
- Vegetation model: ~2-3 hours (V100 GPU)

---

## Results & Evaluation

### Metrics

**Primary Metrics**:
```python
# IoU (Intersection over Union)
IoU = (Prediction ∩ Ground Truth) / (Prediction ∪ Ground Truth)

# Dice Score (F1 Score)
Dice = 2 * (Prediction ∩ Ground Truth) / (|Prediction| + |Ground Truth|)

# Pixel Accuracy
Accuracy = (True Positives + True Negatives) / Total Pixels

# Precision & Recall
Precision = True Positives / (True Positives + False Positives)
Recall = True Positives / (True Positives + False Negatives)
```

### Expected Performance

#### Power Line Detection
| Metric | Value | Notes |
|--------|-------|-------|
| **Best Val IoU** | 0.20-0.30 | Challenging due to thin structures |
| **Test IoU** | 0.20-0.28 | |
| **Test Dice** | 0.33-0.44 | |
| **Pixel Accuracy** | 98-99% | High due to large background |
| **Precision** | 0.40-0.60 | Moderate false positives |
| **Recall** | 0.50-0.70 | Some power lines missed |

**Key Challenges**:
- Thin power lines (1-3 pixels wide) difficult to detect
- Occlusion by vegetation or poles
- Low contrast against sky
- Shadows and lighting variations

#### Vegetation Segmentation
| Metric | Value | Notes |
|--------|-------|-------|
| **Best Val IoU** | 0.40-0.50 | Better than power lines |
| **Test IoU** | 0.38-0.48 | |
| **Test Dice** | 0.55-0.65 | |
| **Pixel Accuracy** | 85-90% | More balanced classes |
| **Precision** | 0.60-0.75 | Cleaner boundaries |
| **Recall** | 0.65-0.80 | Good vegetation coverage |

**Better Performance Because**:
- Larger objects with clear boundaries
- More training pixels (40-50% of dataset)
- Distinct visual features (green, texture)

---

## Output Files & Visualization

### Training Outputs

```
camera/kios_vepl/hpc_deployment/outputs/           # Power line
camera/kios_vepl/hpc_deployment/outputs_veg/       # Vegetation

├── best_model.pth                    # Best checkpoint (state dict)
├── training_20241125_143022.log      # Detailed training log
├── training_history.json             # Epoch-wise metrics
├── training_summary.json             # Final summary
└── visualizations/
    ├── training_loss.png             # Loss curve
    ├── validation_iou.png            # IoU trend
    ├── validation_dice.png           # Dice score
    ├── combined_metrics.png          # All metrics grid
    └── predictions/
        ├── sample_001_comparison.png  # Input|GT|Pred|Overlay
        └── ...
```

### Sample training_summary.json
```json
{
  "config": {
    "architecture": "DeepLabV3",
    "encoder": "resnet34",
    "batch_size": 32,
    "learning_rate": 0.0001,
    "epochs": 20,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "bce_pos_weight": 15.0
  },
  "best_val_iou": 0.2519,
  "test_iou": 0.2493,
  "test_dice": 0.3763,
  "test_acc": 99.12,
  "test_precision": 0.5421,
  "test_recall": 0.6138,
  "total_epochs": 18,
  "training_time_hours": 2.4,
  "model_params": 26035329
}
```

### Visualization Examples

1. **Training Curves**: Loss, IoU, Dice over epochs
2. **Prediction Samples**: 6-panel comparison (RGB | Ground Truth | Prediction | GT Overlay | Pred Overlay | Heatmap)
3. **Confusion Matrix**: For binary classification
4. **ROC Curve**: Threshold analysis for detection

---

## Inference & Deployment

### Inference Pipeline

```python
# Load trained model
model = load_model("outputs/best_model.pth")
model.eval()

# Preprocess image
def preprocess(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    return transform(image=image)['image'].unsqueeze(0)

# Run inference
with torch.no_grad():
    logits = model(preprocessed_image)
    probs = torch.sigmoid(logits)
    mask = (probs > 0.5).float()  # Binary threshold

# Post-process
mask_np = mask.squeeze().cpu().numpy()
```

### Real-World Deployment Considerations

**Edge Device Deployment**:
- Export to ONNX for cross-platform compatibility
- Quantization (FP16) for faster inference
- Target: NVIDIA Jetson, Intel NUC, or similar

**Drone Integration**:
- Real-time processing during flight
- Lightweight model variant (e.g., MobileNet encoder)
- Store raw + prediction for offline review

**Alert System**:
- Trigger alerts when power line detected near vegetation
- Automatic report generation with GPS coordinates
- Integration with maintenance management systems

---

## Ablation Studies & Experiments

### Loss Function Comparison
| Loss Function | Test IoU | Test Dice | Notes |
|---------------|----------|-----------|-------|
| BCE only | 0.18 | 0.30 | Baseline |
| Tversky only | 0.22 | 0.36 | Better recall |
| **Tversky + BCE** | **0.25** | **0.38** | Best balance |
| Focal Loss | 0.20 | 0.32 | Slower convergence |

### Architecture Comparison (Power Line)
| Architecture | Encoder | Test IoU | Inference Time |
|--------------|---------|----------|----------------|
| U-Net | ResNet34 | 0.23 | 15 ms |
| **DeepLabV3** | **ResNet34** | **0.25** | 18 ms |
| FPN | ResNet50 | 0.24 | 20 ms |
| UNet++ | EfficientNet-B0 | 0.22 | 25 ms |

### Data Augmentation Impact
| Augmentation | Test IoU | Notes |
|--------------|----------|-------|
| None | 0.19 | Overfits quickly |
| Flip + Rotate | 0.23 | Basic improvement |
| **Full Pipeline** | **0.25** | Best generalization |

---

## Challenges & Solutions

### Challenge 1: Thin Power Lines
**Problem**: Power lines are only 1-3 pixels wide in 256×256 tiles
**Solutions**:
- ASPP in DeepLabV3 for multi-scale features
- Tversky loss with β=0.7 to emphasize recall
- Weighted BCE with high positive weight (15.0)

### Challenge 2: Class Imbalance
**Problem**: Power lines occupy ~1-2% of pixels
**Solutions**:
- Weighted loss functions
- Oversampling tiles with power lines
- Focal loss experiments (mixed results)

### Challenge 3: Lighting Variations
**Problem**: Shadows, time of day, weather affect appearance
**Solutions**:
- Brightness/contrast augmentation
- Train on diverse conditions
- Consider histogram equalization

### Challenge 4: Occlusions
**Problem**: Vegetation or structures block power lines
**Solutions**:
- Multi-angle imagery
- Temporal analysis (multiple flights)
- Combine with LiDAR for occluded areas

---

## Code Example: Complete Training Loop

```python
# Simplified version - see full code in hpc_deployment/

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

# 1. Dataset
class PowerLineDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.images = sorted(glob(f"{image_dir}/*.png"))
        self.masks = sorted(glob(f"{mask_dir}/*_powerline.png"))
        self.transform = transform

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = cv2.imread(self.masks[idx], 0) / 255.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        return image, mask

    def __len__(self):
        return len(self.images)

# 2. Model
model = smp.DeepLabV3(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
).cuda()

# 3. Loss & Optimizer
criterion = CombinedLoss(alpha=0.3, beta=0.7, bce_weight=15.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 4. Training Loop
for epoch in range(20):
    model.train()
    train_loss = 0
    for images, masks in tqdm(train_loader):
        images, masks = images.cuda(), masks.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # 5. Validation
    model.eval()
    val_iou = compute_iou(model, val_loader)

    print(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, Val IoU={val_iou:.4f}")

    # 6. Save best model
    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), 'best_model.pth')
```

---

## Limitations & Future Work

### Current Limitations
1. **No Height Information**: Cannot measure clearance distances
2. **Weather Dependent**: Performance degrades in fog, rain, snow
3. **Lighting Sensitive**: Shadows create false negatives
4. **2D Constraints**: Cannot distinguish overlapping objects

### Future Improvements
1. **Temporal Models**: Video sequences for motion-based detection
2. **Multi-Task Learning**: Joint power line + vegetation in single model
3. **Attention Mechanisms**: Spatial attention for thin structures
4. **Synthetic Data**: GAN-generated power line images for augmentation
5. **Active Learning**: Prioritize labeling of difficult samples

---

## Conclusion

The RGB camera method provides a **cost-effective baseline** for vegetation encroachment detection with:
- ✅ Fast deployment and intuitive results
- ✅ Reasonable performance for vegetation segmentation (IoU ~0.40-0.50)
- ⚠️ Moderate performance for power line detection (IoU ~0.20-0.30)
- ❌ Lacks 3D spatial information

**Best suited for**: Initial screening, visual documentation, and scenarios where cost and speed are priorities over precision.

**Recommended next steps**: Enhance with multispectral data (Method 2) or LiDAR (Method 3) for critical applications requiring precise measurements.

---

**Document Version**: 1.0
**Related Files**:
- `camera/kios_vepl/hpc_deployment/README.md`
- `camera/kios_vepl/notebooks/train_powerline.ipynb`
- `camera/kios_vepl/notebooks/train_vegetation.ipynb`
