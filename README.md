# Vegetation Encroachment Detection on Power Line Transmission Corridors
### A Multi-Sensor Deep Learning Approach (RGB Camera, Multispectral Imagery, and LiDAR Point Clouds)

This repository contains the official implementation and experimental pipelines for the Master's thesis **"Vegetation Encroachment Detection on Power Line Transmission Corridors"**. It investigates automated semantic segmentation of transmission line corridors across three complementary remote sensing modalities:

1. **Method 1 — RGB Camera Imagery (`camera/`)**: Visual segmentation of power lines and vegetation using CNNs (DeepLabV3, U-Net) and Vision Transformers (SegFormer / MiT).
2. **Method 2 — Multispectral Imagery (`multispectral/`)**: Enhanced 9- to 12-channel segmentation integrating RGB, narrowband spectral reflectance (Green, Red, RedEdge, NIR), and biophysical vegetation indices (NDVI, GNDVI, NDRE, LCI, OSAVI).
3. **Method 3 — 3D LiDAR Point Clouds (`LiDAR/`)**: Direct 3D semantic segmentation of transmission corridors (High Vegetation, Power Lines, Pylons, Background) using **Point Transformer V2** and **CA-PointNet++ (Coordinate Attention PointNet++)**, enabling 3D clearance and safety compliance analysis.

---

## Repository Structure

```text
thesis_vegetation_powerlines_segmentation/
├── README.md                                  # Main repository documentation
├── THESIS_OVERVIEW.md                         # High-level multi-modal synthesis & comparisons
├── METHOD_1_CAMERA_RGB.md                     # Detailed methodology for RGB Camera segmentation
├── METHOD_2_MULTISPECTRAL.md                  # Detailed methodology for Multispectral segmentation
├── METHOD_3_LIDAR_POINT_CLOUD.md              # Detailed methodology for LiDAR 3D segmentation
├── environment.yml                            # Conda environment specification
│
├── camera/                                    # Method 1: RGB Camera Segmentation
│   ├── kios_vepl/
│   │   ├── RGB_Method_Methodology_Summary.md  # RGB dataset & experimental summary
│   │   ├── data/
│   │   │   └── analyze_class_density.py       # Pixel-level class distribution analysis
│   │   ├── notebooks/
│   │   │   ├── preprocess_kios_vepl.ipynb     # Image tiling and binary mask generation
│   │   │   ├── train_kios_vepl.ipynb          # Interactive baseline training
│   │   │   ├── train_kios_vepl_pl.ipynb       # Power line segmentation notebook
│   │   │   └── train_kios_vepl_veg.ipynb      # Vegetation segmentation notebook
│   │   ├── hpc_deployment_camera/             # SLURM HPC training & evaluation suite
│   │   │   ├── train_kios_vepl_pl_hpc.py      # DeepLabV3 + ResNet34 power line training
│   │   │   ├── train_kios_vepl_pl_vit.py      # Vision Transformer (MiT) power line training
│   │   │   ├── train_kios_vepl_veg_hpc.py     # U-Net + VGG16 vegetation training
│   │   │   ├── train_kios_vepl_veg_VT.py      # Vision Transformer vegetation training
│   │   │   ├── reevaluate_pl_base.py          # Test-set evaluation for CNN baseline
│   │   │   ├── reevaluate_pl_vit.py           # Test-set evaluation for ViT model
│   │   │   ├── visualize_predictions.py       # Qualitative overlay & heatmap generation
│   │   │   ├── visualize_results.py           # Learning curves & metric plots
│   │   │   ├── prepare_for_hpc.sh             # Data packaging script for HPC transfer
│   │   │   ├── submit_pl_job.sh               # SLURM job script (Power Lines CNN)
│   │   │   ├── submit_pl_vit_job.sh           # SLURM job script (Power Lines ViT)
│   │   │   ├── submit_veg_job.sh              # SLURM job script (Vegetation CNN)
│   │   │   ├── submit_veg_vit_job.sh          # SLURM job script (Vegetation ViT)
│   │   │   └── requirements.txt               # Python dependencies for HPC jobs
│   │   ├── generate_comparison_plot.py        # Power line model comparison plots
│   │   └── generate_comparison_plot_vegetation.py
│   ├── original_vepl/
│   │   └── notebooks/                         # Orthomosaic preprocessing & Hough/CNN experiments
│   ├── data/prof_data_uav/                    # UAV line detection utilities
│   └── scripts/                               # Dataset label inspection & aerial visualization
│
├── multispectral/                             # Method 2: Multispectral & Vegetation Index Segmentation
│   ├── preprocess_multispectral.ipynb         # Radiometric calibration, band co-registration & indices
│   ├── train_multispectral.ipynb              # Multi-channel (RGB + Bands + Indices) model training
│   ├── predict_powerline_multispectral.ipynb  # Inference and spectral index overlay analysis
│   ├── compare_models_visualization.py        # Cross-architecture & channel ablation comparisons
│   ├── scripts/
│   │   ├── align_multispectral_bands.py       # Spectral band registration
│   │   ├── align_multispectral_best.py        # Enhanced ECC/feature-based band alignment
│   │   ├── align_multispectral_fixed.py       # Fixed-offset / homography alignment pipeline
│   │   ├── diagnose_alignment.py              # Band co-registration quality diagnostics
│   │   ├── organize_multispectral.py          # Flight & band file organization utility
│   │   ├── prepare_for_odm.py                 # OpenDroneMap orthomosaic preparation
│   │   └── view_aligned_multispectral.py      # Interactive aligned band viewer
│   ├── hpc_deployment/
│   │   ├── prepare_for_hpc.sh                 # Multispectral ml_ready packaging for HPC
│   │   ├── submit_veg_job.sh                  # SLURM job: U-Net + VGG16 (12-channel input)
│   │   ├── submit_rgb_veg_job.sh              # SLURM job: RGB baseline on multispectral scenes
│   │   └── submit_multispectral_veg_vit_adapted_job.sh # SLURM job: Adapted MiT-B2 ViT
│   └── test_set_samples/
│       └── test_tile_names.json               # Held-out test split tile identifiers
│
└── LiDAR/                                     # Method 3: 3D LiDAR Point Cloud Segmentation
    ├── LIDAR_METHOD_SUMMARY.md                # Summary of 3D point cloud processing & results
    ├── convert_bag_to_las.py                  # ROS bag to LAS point cloud converter
    ├── convert_npy_to_las.py                  # NumPy array to LAS converter
    ├── extract_pose.py                        # Trajectory / pose extraction from ROS bags
    ├── display_las.ipynb                      # 3D LAS visualization notebook
    ├── merge_pcd_to_las.ipynb                 # Merging PCD scans into georeferenced LAS
    ├── data/
    │   ├── check_pt_structure.py              # Tensor chunk structure verification
    │   ├── convert_pt_to_las.py               # PyTorch .pt chunk to LAS exporter
    │   └── merge_pcd_to_las.py                # Batch PCD-to-LAS merging script
    ├── kios_data/
    │   └── test_kios_data.ipynb               # KIOS LiDAR corridor inspection
    └── point_transformer_v2/
        ├── README.md                          # Point Transformer V2 & CA-PointNet++ guide
        ├── DETAILED_DOCS.md                   # Architecture & implementation reference
        ├── TRAINING_COMPARISON.md             # Point Transformer V2 vs. CA-PointNet++ comparison
        ├── GPU_OPTIMIZATION_CHECK.md          # Memory and throughput tuning notes
        ├── las_processor.py                   # Spatial chunking (50m×50m), normalization & FPS sampling
        ├── powerline_dataset.py               # PyTorch Dataset for 4096-point 3D chunks
        ├── point_transformer_model.py         # Grouped Vector Attention & Partition Pooling model
        ├── training_script.py                 # Local PyTorch Lightning training entrypoint
        ├── point_transformer_training.ipynb   # Interactive training notebook
        ├── model_evaluation.ipynb             # Per-class IoU, confusion matrix & 3D evaluation
        ├── inference_on_las.ipynb             # Direct inference on raw LAS corridor files
        ├── assemble_and_visualize_full.ipynb  # Full corridor chunk reassembly & clearance check
        ├── visualize_full_test_set.py         # Full test scene 3D rendering
        ├── visualize_test_interactive.py      # Interactive Open3D point cloud viewer
        ├── visualize_test_set.ipynb           # Test chunk visualization notebook
        └── hpc_deployment/                    # SLURM HPC training & full-corridor assembly
            ├── train_point_transformer_hpc.py # Point Transformer V2 HPC training script
            ├── train_ca_pointnet_hpc.py       # Coordinate Attention PointNet++ HPC training script
            ├── assemble_predictions_hpc.py    # Full test corridor reconstruction & metrics
            ├── visualize_results.py           # Training loss & validation mIoU plotting
            ├── prepare_for_hpc.sh             # Point cloud chunk packaging for HPC
            ├── submit_lidar_job.sh            # SLURM job script (Point Transformer V2)
            ├── submit_ca_pointnet_job.sh      # SLURM job script (CA-PointNet++)
            ├── CLASS_IMBALANCE_SOLUTIONS.md   # Analysis of severe 3D class imbalance
            ├── MODEL_COMPARISON.md            # Architectural trade-offs & benchmarks
            └── CA_POINTNET_README.md          # Coordinate Attention PointNet++ documentation
```

---

## Methodology & Technical Highlights

### 1. Multi-Modal Sensing Comparison

| Modality | Input Representation | Architectures Evaluated | Primary Strengths |
| :--- | :--- | :--- | :--- |
| **RGB Camera** | 3-channel RGB (`256×256` & `512×512` tiles) | DeepLabV3 (ResNet34), U-Net (VGG16), SegFormer (MiT-B2) | Low hardware cost, fast inference, high-resolution visual context |
| **Multispectral** | 9–12 channels: RGB (3) + Green, Red, RedEdge, NIR (4) + NDVI, GNDVI, NDRE, LCI, OSAVI (5) | Adapted DeepLabV3 & U-Net (VGG16 / MiT-B2) with first-layer weight expansion | Strong chlorophyll contrast, shadow resilience in NIR, superior vegetation vigor discrimination |
| **3D LiDAR** | `4096 × 3` normalized XYZ point clouds in `50m × 50m` spatial chunks | Point Transformer V2, Coordinate Attention PointNet++ (CA-PointNet++) | True 3D geometry, weather/illumination invariance, exact metric clearance computation |

### 2. Addressing Extreme Class Imbalance
Power line corridors exhibit severe class imbalance:
- **2D Imagery**: Power lines occupy only ~1–2% of image pixels. Addressed via a **Combined Tversky ($\alpha=0.3, \beta=0.7$) + Pos-Weighted BCE ($\text{pos\_weight}=15.0$) Loss**, prioritizing recall on thin linear conductors while stabilizing pixel-wise gradients.
- **3D Point Clouds**: High vegetation (~60%) and background (~40%) dominate over power lines (~0.2%) and pylons (~0.01%). Addressed via calibrated manual class weights `[1.0, 10.0, 50.0, 1.5]` in full-precision (`FP32`) Cross-Entropy loss with gradient clipping (`1.0`), avoiding gradient explosion while raising rare-class detection from `0.00` IoU to detectable levels.

### 3. Pretrained Weight Adaptation for Multispectral Input
To leverage ImageNet-pretrained encoders on 9- or 12-channel multispectral stacks, the first convolutional layer (`conv1`) is dynamically expanded:
- Channels `0..2` retain pretrained RGB weights.
- Spectral band and vegetation index channels (`3..C`) are initialized using scaled channel-averaged RGB filters, preserving early edge/texture extractors while learning spectral combinations.

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mohamad-fat/thesis_vegetation_powerlines_segmentation.git
cd thesis_vegetation_powerlines_segmentation
```

### 2. Create the Conda Environment
An exported Conda environment specification is provided in `environment.yml`:
```bash
conda env create -f environment.yml
conda activate veg_enc_env
```

Alternatively, install the core dependencies via `pip`:
```bash
pip install torch==2.1.2 torchvision==0.16.2 pytorch-lightning==2.5.0
pip install segmentation-models-pytorch==0.3.3 albumentations opencv-python-headless
pip install laspy open3d scikit-learn matplotlib tqdm
```

---

## Usage Guide

### Method 1: RGB Camera Segmentation (`camera/`)

1. **Preprocess Raw Aerial Imagery**:
   Run `camera/kios_vepl/notebooks/preprocess_kios_vepl.ipynb` to tile high-resolution UAV images and generate binary masks for power lines and vegetation.
2. **Train Models (Local or HPC)**:
   ```bash
   cd camera/kios_vepl/hpc_deployment_camera

   # Power Line Detection (DeepLabV3 + ResNet34)
   python train_kios_vepl_pl_hpc.py \
       --data_dir ../data/processed_data \
       --output_dir ../output_pl \
       --architecture DeepLabV3 \
       --encoder resnet34 \
       --batch_size 32 \
       --epochs 20 \
       --tversky_alpha 0.3 \
       --tversky_beta 0.7 \
       --bce_pos_weight 15.0 \
       --lr 1e-4 \
       --amp

   # Vegetation Segmentation (U-Net + VGG16)
   python train_kios_vepl_veg_hpc.py \
       --data_dir ../data/processed_data \
       --output_dir ../output_veg \
       --architecture Unet \
       --encoder vgg16 \
       --batch_size 32 \
       --epochs 20 \
       --lr 1e-5 \
       --amp
   ```
   For Vision Transformer training (`train_kios_vepl_pl_vit.py` / `train_kios_vepl_veg_VT.py`) or SLURM cluster submission, use `sbatch submit_pl_vit_job.sh` or `sbatch submit_veg_vit_job.sh`.

---

### Method 2: Multispectral Segmentation (`multispectral/`)

1. **Band Alignment & Orthomosaic Preparation**:
   Use the scripts in `multispectral/scripts/` (`align_multispectral_best.py`, `diagnose_alignment.py`, `prepare_for_odm.py`) to co-register raw spectral bands.
2. **Compute Vegetation Indices & Generate Tiles**:
   Execute `multispectral/preprocess_multispectral.ipynb` to compute `NDVI`, `GNDVI`, `NDRE`, `LCI`, and `OSAVI` and export aligned `.npy` tiles to `data/ml_ready/`.
3. **Train & Evaluate**:
   - Interactive training and ablation experiments: `multispectral/train_multispectral.ipynb`
   - Inference and spectral overlay visualization: `multispectral/predict_powerline_multispectral.ipynb`
   - HPC cluster deployment:
     ```bash
     cd multispectral/hpc_deployment
     ./prepare_for_hpc.sh
     sbatch submit_veg_job.sh
     sbatch submit_multispectral_veg_vit_adapted_job.sh
     ```

---

### Method 3: LiDAR 3D Point Cloud Segmentation (`LiDAR/`)

1. **Preprocess LAS Point Clouds into 3D Chunks**:
   ```bash
   cd LiDAR/point_transformer_v2
   python las_processor.py
   ```
   This partitions raw `.las` files into `50m × 50m` chunks, normalizes coordinates, and samples `4096` points per chunk into `.pt` tensors.
2. **Train 3D Segmentation Models**:
   ```bash
   cd LiDAR/point_transformer_v2/hpc_deployment

   # Train Point Transformer V2
   python train_point_transformer_hpc.py \
       --data_dir ../data/processed_data_optimized \
       --output_dir ../outputs_lidar \
       --batch_size 16 \
       --epochs 5 \
       --lr 1e-3 \
       --num_points 4096 \
       --d_model 156 \
       --num_layers 3 \
       --k_neighbors 16 \
       --precision 32-true

   # Train Coordinate Attention PointNet++ (CA-PointNet++)
   python train_ca_pointnet_hpc.py \
       --data_dir ../data/processed_data_optimized \
       --output_dir ../outputs_ca_pointnet \
       --batch_size 16 \
       --epochs 5 \
       --lr 1e-3 \
       --num_points 4096
   ```
3. **Full Corridor Reconstruction & Clearance Analysis**:
   ```bash
   python assemble_predictions_hpc.py \
       --model_path ../outputs_lidar/checkpoints/best_model.ckpt \
       --data_dir ../data/processed_data_optimized \
       --output_dir ../outputs_lidar/full_reconstructions
   ```

---

## Detailed Documentation

For in-depth architectural diagrams, mathematical formulations, loss functions, and ablation tables, refer to the dedicated method documents:
- [THESIS_OVERVIEW.md](THESIS_OVERVIEW.md) — Executive summary, cross-modal comparison, and system architecture
- [METHOD_1_CAMERA_RGB.md](METHOD_1_CAMERA_RGB.md) — RGB camera segmentation pipeline and loss function ablations
- [METHOD_2_MULTISPECTRAL.md](METHOD_2_MULTISPECTRAL.md) — Spectral bands, vegetation index formulations, and 12-channel encoder adaptation
- [METHOD_3_LIDAR_POINT_CLOUD.md](METHOD_3_LIDAR_POINT_CLOUD.md) — Point Transformer V2, CA-PointNet++, 3D class imbalance solutions, and clearance computation
