# V100 GPU Optimization Analysis

## Current Strategy ✅
```python
- Single GPU: 1x V100 (32GB)
- Mixed Precision: 16-mixed (FP16 + FP32) ✅
- Batch Size: 8
- Gradient Clipping: 1.0 ✅
- Data Workers: 8 (parallel data loading) ✅
- Pin Memory: True ✅
```

## Missing Optimizations ❌

### 1. **CUDA Optimizations**
```python
torch.backends.cudnn.benchmark = True  # Auto-tune kernels
torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops
```

### 2. **Gradient Accumulation** (Simulate Larger Batches)
```python
# Accumulate gradients over multiple steps
accumulate_grad_batches = 4  # Effective batch = 8 * 4 = 32
```

### 3. **Larger Batch Size** (With Smaller Model)
```python
# Now that model is 3.4x smaller (1M vs 3.3M params)
# We can increase batch size
batch_size = 16 or 24  # Instead of 8
```

### 4. **Channels Last Memory Format**
```python
# More efficient memory layout for CNNs/attention
model = model.to(memory_format=torch.channels_last_3d)
```

### 5. **Torch Compile** (PyTorch 2.0+)
```python
# JIT compile for faster execution
model = torch.compile(model)
```

## Recommended Additions

### Priority 1: Enable CUDA Benchmark (Free Speed)
```python
torch.backends.cudnn.benchmark = True
```

### Priority 2: Increase Batch Size (Now Possible)
```python
# With smaller model (1M params vs 3.3M)
batch_size = 16  # 2x larger
# This also improves gradient quality
```

### Priority 3: Gradient Accumulation (Optional)
```python
# If memory allows, simulate even larger batches
trainer = L.Trainer(
    accumulate_grad_batches=2,  # Effective batch = 16 * 2 = 32
    ...
)
```

## Expected Speedup

| Optimization | Expected Speedup |
|--------------|------------------|
| CUDA benchmark | 5-15% faster |
| Batch 8 → 16 | 20-30% faster |
| Grad accumulation | Better convergence |
| **Combined** | **~30-50% faster** |

## V100 Specs Check

- Memory: 32GB ✅ Plenty of headroom
- Compute: 15.7 TFLOPS (FP32), 125 TFLOPS (FP16) ✅
- Tensor Cores: ✅ Used by mixed precision
- NVLink: N/A (single GPU)

## Notebook Comparison

Notebook doesn't use:
- cudnn.benchmark
- Gradient accumulation
- Larger batch sizes (limited to 2 by MPS)

We should ADD these for V100!
