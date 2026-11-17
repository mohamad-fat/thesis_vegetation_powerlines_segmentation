# Training Configuration Comparison

## Original Notebook (Working)
```
D_MODEL = 156
NUM_LAYERS = 3
LEARNING_RATE = 1e-3 (0.001)
BATCH_SIZE = 2
EARLY_STOPPING_PATIENCE = 20
SAVE_TOP_K = 3
CLASS_WEIGHTS = calculated (but broken)
PRECISION = default (no mixed precision initially)
Model Parameters: ~968K
```

## HPC Script (Current - Slow & Bad Results)
```
d_model = 256           ❌ TOO LARGE (1.6x bigger)
num_layers = 4          ❌ TOO DEEP (33% more layers)  
lr = 0.0001            ❌ TOO LOW (10x smaller)
batch_size = 8          ✅ OK
early_stop_patience = 3 ❌ TOO AGGRESSIVE  
save_top_k = 1          ✅ OK
class_weights = None    ✅ OK (notebook's were broken)
precision = '32-true'   ⚠️ SLOW (no mixed precision)
Model Parameters: ~3.3M  ❌ 3.4x MORE PARAMETERS
```

## Issues Identified

### 1. Model Too Large
- Notebook: 156D × 3 layers = 968K params
- HPC: 256D × 4 layers = 3.3M params (3.4x larger!)
- **Result**: Much slower training, harder to optimize

### 2. Learning Rate Too Low
- Notebook: 0.001
- HPC: 0.0001 (10x smaller)
- **Result**: Model learns too slowly, can't converge in 10 epochs

### 3. Early Stopping Too Aggressive  
- Notebook: patience=20
- HPC: patience=3
- **Result**: Stops before model has chance to learn

### 4. No Mixed Precision
- Notebook: Uses default (allows 16-mixed on GPU)
- HPC: Forced '32-true' (we disabled it to fix NaN)
- **Result**: 2x slower training

## Recommended Fix

Match the notebook's working configuration:

```bash
--d_model 156           # Match notebook
--num_layers 3          # Match notebook
--lr 0.001              # Match notebook (10x higher!)
--batch_size 8          # Keep (better than 2)
--early_stop_patience 15  # More reasonable
--epochs 30             # Give it time to learn
# Try mixed precision again (NaN might be fixed now)
```
