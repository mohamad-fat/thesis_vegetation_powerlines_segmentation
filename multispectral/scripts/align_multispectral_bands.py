import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.enums import Resampling as ResamplingEnum
from pathlib import Path
import cv2
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


def get_band_info(band_path: str) -> Dict:
    """
    Extract metadata from a single band orthophoto.

    Args:
        band_path: Path to the GeoTIFF file

    Returns:
        Dictionary containing band metadata
    """
    with rasterio.open(band_path) as src:
        return {
            'path': band_path,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'shape': src.shape,
            'resolution': src.res,
            'dtype': src.dtypes[0],
            'nodata': src.nodata
        }


def find_common_extent(band_infos: List[Dict]) -> Tuple:
    """
    Find the common (intersection) extent of all bands.

    Args:
        band_infos: List of band metadata dictionaries

    Returns:
        Tuple of (minx, miny, maxx, maxy) in common CRS
    """
    # Get all bounds
    all_bounds = [info['bounds'] for info in band_infos]

    # Find intersection (common area)
    minx = max([b.left for b in all_bounds])
    miny = max([b.bottom for b in all_bounds])
    maxx = min([b.right for b in all_bounds])
    maxy = min([b.top for b in all_bounds])

    print(f"Common extent: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")

    return (minx, miny, maxx, maxy)


def find_target_resolution(band_infos: List[Dict]) -> Tuple[float, float]:
    """
    Determine target resolution (use finest resolution among all bands).

    Args:
        band_infos: List of band metadata dictionaries

    Returns:
        Tuple of (x_res, y_res)
    """
    resolutions = [info['resolution'] for info in band_infos]

    # Use the finest (smallest) resolution
    target_x_res = min([r[0] for r in resolutions])
    target_y_res = min([r[1] for r in resolutions])

    print(f"Target resolution: {target_x_res:.6f} x {target_y_res:.6f}")

    return (target_x_res, target_y_res)


def align_band_to_reference(band_path: str, reference_info: Dict,
                            common_extent: Tuple, target_resolution: Tuple) -> np.ndarray:
    """
    Align a single band to match reference geometry.

    Args:
        band_path: Path to the band to align
        reference_info: Metadata of reference band
        common_extent: Common extent (minx, miny, maxx, maxy)
        target_resolution: Target resolution (x_res, y_res)

    Returns:
        Aligned numpy array
    """
    minx, miny, maxx, maxy = common_extent
    x_res, y_res = target_resolution

    # Calculate target dimensions
    width = int((maxx - minx) / x_res)
    height = int((maxy - miny) / abs(y_res))

    # Create target transform
    from rasterio.transform import from_bounds
    target_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    # Read and reproject the band
    with rasterio.open(band_path) as src:
        # Create destination array
        aligned = np.zeros((height, width), dtype=src.dtypes[0])

        # Reproject to match reference
        reproject(
            source=rasterio.band(src, 1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=reference_info['crs'],
            resampling=Resampling.bilinear
        )

    return aligned


def stack_and_save_aligned_bands(band_paths: Dict[str, str], output_dir: str,
                                 flight_name: str = "flight") -> str:
    """
    Align all bands and stack them into a single multi-band GeoTIFF.

    Args:
        band_paths: Dictionary mapping band names to file paths
                   e.g., {'G': 'path/to/G.tif', 'R': 'path/to/R.tif', ...}
        output_dir: Directory to save aligned outputs
        flight_name: Name prefix for output files

    Returns:
        Path to the stacked multi-band GeoTIFF
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print(f"ALIGNING MULTISPECTRAL BANDS FOR: {flight_name}")
    print("=" * 80)

    # Step 1: Get metadata for all bands
    print("\nStep 1: Reading band metadata...")
    band_infos = {}
    band_names = ['G', 'R', 'RE', 'NIR']

    for band_name in band_names:
        if band_name in band_paths:
            print(f"  - {band_name}: {band_paths[band_name]}")
            band_infos[band_name] = get_band_info(band_paths[band_name])
        else:
            print(f"  WARNING: {band_name} band not found!")

    if len(band_infos) == 0:
        raise ValueError("No valid bands found!")

    # Step 2: Check CRS consistency
    print("\nStep 2: Checking CRS consistency...")
    crs_list = [info['crs'] for info in band_infos.values()]
    if len(set(str(crs) for crs in crs_list)) > 1:
        print("  WARNING: Bands have different CRS!")
        for band_name, info in band_infos.items():
            print(f"    {band_name}: {info['crs']}")
    else:
        print(f"  All bands use CRS: {crs_list[0]}")

    # Step 3: Find common extent
    print("\nStep 3: Finding common extent...")
    common_extent = find_common_extent(list(band_infos.values()))

    # Step 4: Determine target resolution
    print("\nStep 4: Determining target resolution...")
    target_resolution = find_target_resolution(list(band_infos.values()))

    # Step 5: Use first available band as reference
    reference_band_name = band_names[0] if band_names[0] in band_infos else list(band_infos.keys())[0]
    reference_info = band_infos[reference_band_name]
    print(f"\nStep 5: Using {reference_band_name} band as reference")

    # Step 6: Align all bands
    print("\nStep 6: Aligning bands...")
    aligned_bands = {}

    for band_name in band_names:
        if band_name in band_paths:
            print(f"  - Aligning {band_name} band...")
            aligned_bands[band_name] = align_band_to_reference(
                band_paths[band_name],
                reference_info,
                common_extent,
                target_resolution
            )
            print(f"    Output shape: {aligned_bands[band_name].shape}")

    # Step 7: Stack bands into multi-band array
    print("\nStep 7: Stacking bands...")
    band_order = [name for name in band_names if name in aligned_bands]
    stacked = np.stack([aligned_bands[name] for name in band_order], axis=0)
    print(f"  Stacked shape: {stacked.shape} (bands, height, width)")
    print(f"  Band order: {band_order}")

    # Step 8: Save stacked GeoTIFF
    print("\nStep 8: Saving aligned multi-band GeoTIFF...")
    output_path = os.path.join(output_dir, f"{flight_name}_multispectral_aligned.tif")

    minx, miny, maxx, maxy = common_extent
    x_res, y_res = target_resolution
    height, width = aligned_bands[band_order[0]].shape

    from rasterio.transform import from_bounds
    target_transform = from_bounds(minx, miny, maxx, maxy, width, height)

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=len(band_order),
        dtype=stacked.dtype,
        crs=reference_info['crs'],
        transform=target_transform,
        compress='lzw'
    ) as dst:
        for idx, band_name in enumerate(band_order, start=1):
            dst.write(aligned_bands[band_name], idx)
            dst.set_band_description(idx, band_name)

    print(f"  Saved to: {output_path}")

    # Step 9: Save individual aligned bands
    print("\nStep 9: Saving individual aligned bands...")
    individual_dir = os.path.join(output_dir, "individual_bands")
    os.makedirs(individual_dir, exist_ok=True)

    for band_name in band_order:
        band_output_path = os.path.join(individual_dir, f"{flight_name}_band_{band_name}_aligned.tif")

        with rasterio.open(
            band_output_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=aligned_bands[band_name].dtype,
            crs=reference_info['crs'],
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(aligned_bands[band_name], 1)

        print(f"  - {band_name}: {band_output_path}")

    # Step 10: Calculate and save vegetation indices
    print("\nStep 10: Calculating vegetation indices...")
    indices_dir = os.path.join(output_dir, "vegetation_indices")
    os.makedirs(indices_dir, exist_ok=True)

    if 'NIR' in aligned_bands and 'R' in aligned_bands:
        # NDVI = (NIR - R) / (NIR + R)
        nir = aligned_bands['NIR'].astype(np.float32)
        red = aligned_bands['R'].astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)

        ndvi_path = os.path.join(indices_dir, f"{flight_name}_NDVI.tif")
        with rasterio.open(
            ndvi_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=reference_info['crs'],
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(ndvi, 1)

        print(f"  - NDVI: {ndvi_path}")

    if 'NIR' in aligned_bands and 'RE' in aligned_bands:
        # NDRE = (NIR - RedEdge) / (NIR + RedEdge)
        nir = aligned_bands['NIR'].astype(np.float32)
        re = aligned_bands['RE'].astype(np.float32)

        ndre = (nir - re) / (nir + re + 1e-8)
        ndre = np.clip(ndre, -1, 1)

        ndre_path = os.path.join(indices_dir, f"{flight_name}_NDRE.tif")
        with rasterio.open(
            ndre_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=reference_info['crs'],
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(ndre, 1)

        print(f"  - NDRE: {ndre_path}")

    if 'NIR' in aligned_bands and 'G' in aligned_bands:
        # GNDVI = (NIR - Green) / (NIR + Green)
        nir = aligned_bands['NIR'].astype(np.float32)
        green = aligned_bands['G'].astype(np.float32)

        gndvi = (nir - green) / (nir + green + 1e-8)
        gndvi = np.clip(gndvi, -1, 1)

        gndvi_path = os.path.join(indices_dir, f"{flight_name}_GNDVI.tif")
        with rasterio.open(
            gndvi_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=reference_info['crs'],
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(gndvi, 1)

        print(f"  - GNDVI: {gndvi_path}")

    # Step 11: Create visualization
    print("\nStep 11: Creating visualizations...")
    visualize_aligned_bands(aligned_bands, output_dir, flight_name)

    print("\n" + "=" * 80)
    print("ALIGNMENT COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - Multi-band GeoTIFF: {output_path}")
    print(f"  - Individual bands: {individual_dir}/")
    print(f"  - Vegetation indices: {indices_dir}/")
    print(f"  - Visualizations: {output_dir}/visualizations/")

    return output_path


def visualize_aligned_bands(aligned_bands: Dict[str, np.ndarray],
                            output_dir: str, flight_name: str):
    """
    Create visualization of aligned bands.

    Args:
        aligned_bands: Dictionary of aligned band arrays
        output_dir: Directory to save visualizations
        flight_name: Name prefix for output files
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    # Create figure with all bands
    n_bands = len(aligned_bands)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Aligned Multispectral Bands - {flight_name}', fontsize=16, fontweight='bold')

    band_names = ['G', 'R', 'RE', 'NIR']
    band_labels = {
        'G': 'Green (520-590nm)',
        'R': 'Red (640-680nm)',
        'RE': 'Red Edge (730-740nm)',
        'NIR': 'Near Infrared (820-890nm)'
    }

    # Plot individual bands
    for idx, band_name in enumerate(band_names):
        if band_name in aligned_bands:
            row = idx // 3
            col = idx % 3

            band = aligned_bands[band_name]

            # Normalize for display
            p2, p98 = np.percentile(band[band > 0], [2, 98])
            band_display = np.clip((band - p2) / (p98 - p2), 0, 1)

            axes[row, col].imshow(band_display, cmap='gray')
            axes[row, col].set_title(f'{band_labels[band_name]}', fontsize=12, fontweight='bold')
            axes[row, col].axis('off')

            # Add dimensions
            axes[row, col].text(0.02, 0.98, f'{band.shape[1]}x{band.shape[0]}',
                              transform=axes[row, col].transAxes,
                              fontsize=10, color='yellow',
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # Create false color composites
    if 'NIR' in aligned_bands and 'R' in aligned_bands and 'G' in aligned_bands:
        # Color Infrared (CIR)
        nir = aligned_bands['NIR']
        r = aligned_bands['R']
        g = aligned_bands['G']

        # Normalize each band
        nir_norm = cv2.normalize(nir, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        r_norm = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        g_norm = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        cir = np.stack([nir_norm, r_norm, g_norm], axis=-1)

        axes[1, 1].imshow(cir)
        axes[1, 1].set_title('Color Infrared\n(NIR-R-G)\nVegetation = Red', fontsize=11, fontweight='bold')
        axes[1, 1].axis('off')

    # NDVI visualization
    if 'NIR' in aligned_bands and 'R' in aligned_bands:
        nir = aligned_bands['NIR'].astype(float)
        r = aligned_bands['R'].astype(float)

        ndvi = (nir - r) / (nir + r + 1e-8)
        ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)

        ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
        ndvi_colored = cv2.cvtColor(ndvi_colored, cv2.COLOR_BGR2RGB)

        axes[1, 2].imshow(ndvi_colored)
        axes[1, 2].set_title('NDVI\nGreen = Healthy Vegetation', fontsize=11, fontweight='bold')
        axes[1, 2].axis('off')

    plt.tight_layout()

    viz_path = os.path.join(viz_dir, f'{flight_name}_aligned_bands.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"  Visualization saved: {viz_path}")
    plt.close()


def main():
    """
    Main function to align multispectral bands from ODM output.
    """
    # Define paths to your ODM outputs
    base_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/trial_alignment/reconstructed_02"

    # Map band names to their orthophoto paths
    band_paths = {
        'G': os.path.join(base_dir, "02_band_G/odm_orthophoto/odm_orthophoto_02_band_G.tif"),
        'R': os.path.join(base_dir, "02_band_R/odm_orthophoto/odm_orthophoto_02_R.tif"),
        'RE': os.path.join(base_dir, "02_band_RE/odm_orthophoto/odm_orthophoto_02_RE.tif"),
        'NIR': os.path.join(base_dir, "02_band_NIR/odm_orthophoto/odm_orthophoto_02_band_NIR.tif")
    }

    # Output directory
    output_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/trial_alignment/aligned_output"

    # Run alignment
    stack_and_save_aligned_bands(band_paths, output_dir, flight_name="flight_02")

    print("\n" + "=" * 80)
    print("SUCCESS!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check the aligned multi-band GeoTIFF in QGIS")
    print("2. Verify all 4 bands are properly aligned")
    print("3. Use this aligned stack for labeling (all bands stay aligned)")
    print("4. Process additional flights by updating band_paths in this script")


if __name__ == "__main__":
    main()
