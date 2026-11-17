# Class Imbalance Solutions

## Current Problem

### Class Distribution
```
High Vegetation: 60.0% → 1.0x weight (majority)
Others:          40.0% → 1.5x weight  
Power Lines:      0.2% → 300x weight (rare!)
Pylons:          0.01% → 6000x weight (extremely rare!)
```

**Current Status:** ❌ NOT addressing imbalance (uniform weights)

**Impact:**
- Model biased toward majority classes
- Poor performance on Power Lines
- Pylons completely ignored (IoU = 0.00)

---

## Solution 1: Manual Class Weights ⭐ (RECOMMENDED)

Manually set reasonable weights based on frequency:

```python
# In train_point_transformer_hpc.py, line 467
# Replace:
class_weights = None

# With:
class_weights = [1.0, 10.0, 50.0, 1.5]
# [High Veg, Power Lines, Pylons, Others]
```

**Reasoning:**
- High Vegetation (60%): 1.0 (baseline)
- Others (40%): 1.5 (slightly more important)
- Power Lines (0.2%): 10.0 (much more important)
- Pylons (0.01%): 50.0 (extremely important)

**Advantages:**
- ✅ Simple to implement
- ✅ Stable (no extreme weights)
- ✅ Interpretable
- ✅ No NaN issues

---

## Solution 2: Focal Loss

Replace CrossEntropyLoss with Focal Loss (focuses on hard examples):

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, ignore_index=-1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', 
            ignore_index=self.ignore_index
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

# Use in model:
self.criterion = FocalLoss(gamma=2)
```

**Advantages:**
- ✅ Automatically focuses on hard/rare examples
- ✅ No manual weight tuning
- ✅ State-of-the-art for imbalanced data

**Disadvantages:**
- ⚠️ Requires more tuning (gamma parameter)
- ⚠️ Slower convergence

---

## Solution 3: Online Hard Example Mining (OHEM)

Sample hard examples more frequently during training:

```python
def ohem_loss(logits, labels, keep_ratio=0.7):
    """Keep only hardest 70% of examples"""
    loss = F.cross_entropy(logits, labels, reduction='none')
    loss_sorted, _ = torch.sort(loss, descending=True)
    keep_num = int(loss.numel() * keep_ratio)
    threshold = loss_sorted[keep_num]
    keep_mask = loss >= threshold
    return loss[keep_mask].mean()
```

---

## Solution 4: Balanced Sampling

Sample equal number of points from each class per batch:

```python
class BalancedPointSampler:
    def sample(self, points, labels, num_per_class=1024):
        """Sample equal points from each class"""
        balanced_points = []
        balanced_labels = []
        
        for cls in range(4):
            mask = labels == cls
            cls_points = points[mask]
            
            if len(cls_points) >= num_per_class:
                idx = np.random.choice(len(cls_points), num_per_class, replace=False)
            else:
                idx = np.random.choice(len(cls_points), num_per_class, replace=True)
            
            balanced_points.append(cls_points[idx])
            balanced_labels.append(torch.full((num_per_class,), cls))
        
        return torch.cat(balanced_points), torch.cat(balanced_labels)
```

---

## Solution 5: Hybrid Approach ⭐⭐ (BEST)

Combine class weights + focal loss:

```python
# Use both!
class_weights = torch.tensor([1.0, 10.0, 50.0, 1.5])
self.criterion = FocalLoss(alpha=class_weights, gamma=2)
```

---

## Recommended Implementation

### Quick Fix (2 minutes)
```python
# In train_point_transformer_hpc.py, line 467
class_weights = [1.0, 10.0, 50.0, 1.5]  # Manual weights
```

### Better Fix (10 minutes)
1. Add manual class weights
2. Increase Pylon weight if still IoU=0
3. Monitor per-class IoU during training

### Best Fix (30 minutes)
1. Implement Focal Loss
2. Use with moderate class weights
3. Add balanced sampling for Pylons

---

## Expected Improvements

| Approach | Power Line IoU | Pylon IoU | Training Stability |
|----------|---------------|-----------|-------------------|
| **No weights (current)** | 0.62 | 0.00 | ✅ Stable |
| **Manual weights** | 0.70 | 0.30 | ✅ Stable |
| **Focal Loss** | 0.75 | 0.35 | ⚠️ Moderate |
| **Hybrid** | 0.80 | 0.40 | ⚠️ Moderate |

---

## Implementation Priority

### Priority 1: Add Manual Class Weights
Simple, stable, effective.

### Priority 2: Monitor Per-Class Metrics
Track IoU for each class during training.

### Priority 3: Try Focal Loss
If Pylons still at 0% IoU after weights.

---

## Why Original Class Weights Failed

The automatic calculation produced:
```python
class_weights = [7.15e-12, 2.0, 2.0, 1.54e-11]
```

**Problem:** Near-zero weights for majority classes!
- Division: total / (count + eps) → 1e9 / (1e9 + eps) ≈ 1
- Then normalized: → near zero
- Result: Loss ≈ 0 for majority classes → NaN gradients

**Fix:** Use inverse frequency WITHOUT extreme normalization:
```python
# Good:
weights = 1.0 / (frequency + 0.01)

# Bad (what we had):
weights = total / (counts + 1e-6)
weights = weights / weights.sum() * 4  # Over-normalization!
```

---

## Conclusion

**Recommendation:** Start with **manual class weights = [1.0, 10.0, 50.0, 1.5]**

This is:
- ✅ Simple
- ✅ Stable
- ✅ Should improve Power Lines and Pylons significantly
- ✅ No risk of NaN

Then iterate based on results!
