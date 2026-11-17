import torch

# Load one sample to check structure
sample = torch.load('/Users/user/Desktop/university/thesis/veg_enc_code/LiDAR/data/processed_data_optimized/test/sample_000000.pt', weights_only=False)

print("Keys in sample:")
for key in sample.keys():
    print(f"  {key}: {type(sample[key])}, shape: {sample[key].shape if hasattr(sample[key], 'shape') else 'N/A'}")

print("\nFirst few points:")
print(sample['input_pcd'][:5])

print("\nFirst few labels:")
print(sample['semantic_labels'][:5])
