import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def align_to_reference_band(band_paths: Dict[str, str], output_dir: str,
                            reference_band: str = 'G',
                            flight_name: str = "flight") -> str:
    """
    Align all bands to a reference band's exact coordinate grid.

    This method ensures perfect pixel-to-pixel alignment by:
    1. Selecting one band as the reference (usually G for Green)
    2. Using the reference band's EXACT transform and dimensions
    3. Reprojecting all other bands to match this grid precisely

    This avoids rounding errors from recalculating transforms.

    Args:
        band_paths: Dict mapping band names to file paths
        output_dir: Output directory
        reference_band: Which band to use as reference (default: 'G')
        flight_name: Flight identifier

    Returns:
        Path to aligned multi-band GeoTIFF
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTISPECTRAL BAND ALIGNMENT - REFERENCE BAND METHOD")
    print("=" * 80)
    print()

    # Validate reference band exists
    if reference_band not in band_paths:
        available = list(band_paths.keys())
        print(f"WARNING: Reference band '{reference_band}' not found!")
        print(f"Available bands: {available}")
        reference_band = available[0]
        print(f"Using '{reference_band}' as reference instead.\n")

    # Step 1: Read reference band metadata
    print(f"Step 1: Loading reference band ({reference_band})...")
    with rasterio.open(band_paths[reference_band]) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_shape = ref_src.shape
        ref_bounds = ref_src.bounds
        ref_dtype = ref_src.dtypes[0]

        print(f"  Transform: {ref_transform}")
        print(f"  CRS: {ref_crs}")
        print(f"  Dimensions: {ref_shape[1]} x {ref_shape[0]} (width x height)")
        print(f"  Resolution: {ref_src.res[0]:.6f} x {abs(ref_src.res[1]):.6f} m/px")
        print(f"  Bounds: ({ref_bounds.left:.2f}, {ref_bounds.bottom:.2f}) to ({ref_bounds.right:.2f}, {ref_bounds.top:.2f})")

    # Step 2: Read all band metadata
    print("\nStep 2: Reading all band metadata...")
    band_order = ['G', 'R', 'RE', 'NIR']
    band_infos = {}

    for band_name in band_order:
        if band_name not in band_paths:
            print(f"  WARNING: {band_name} band not found, skipping")
            continue

        with rasterio.open(band_paths[band_name]) as src:
            band_infos[band_name] = {
                'path': band_paths[band_name],
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': src.shape,
                'resolution': src.res,
                'dtype': src.dtypes[0]
            }

            # Check if band matches reference
            if band_name == reference_band:
                status = "(REFERENCE)"
            elif src.shape == ref_shape and src.transform == ref_transform:
                status = "(Already aligned)"
            else:
                status = f"({src.shape[1]}x{src.shape[0]}, needs alignment)"

            print(f"  {band_name}: {status}")

    if len(band_infos) == 0:
        raise ValueError("No bands found!")

    # Step 3: Check CRS consistency
    print("\nStep 3: Checking CRS consistency...")
    crs_set = set(str(info['crs']) for info in band_infos.values())
    if len(crs_set) > 1:
        print("  ERROR: Bands have different CRS!")
        for band_name, info in band_infos.items():
            print(f"    {band_name}: {info['crs']}")
        raise ValueError("CRS mismatch - cannot align bands with different coordinate systems")
    print(f"  All bands share CRS: {ref_crs}")

    # Step 4: Align all bands to reference grid
    print("\nStep 4: Aligning bands to reference grid...")
    height, width = ref_shape
    aligned_bands = {}

    for band_name in band_order:
        if band_name not in band_paths:
            continue

        print(f"  Processing {band_name}...")

        with rasterio.open(band_paths[band_name]) as src:
            # Check if band is already perfectly aligned
            if (src.shape == ref_shape and
                src.transform == ref_transform and
                src.crs == ref_crs):
                print(f"    Already perfectly aligned, reading directly...")
                aligned_bands[band_name] = src.read(1)
            else:
                print(f"    Reprojecting from {src.shape} to {ref_shape}...")

                # Create destination array
                aligned = np.zeros((height, width), dtype=src.dtypes[0])

                # Reproject to match reference exactly
                reproject(
                    source=rasterio.band(src, 1),
                    destination=aligned,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,  # Use reference transform EXACTLY
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata if src.nodata is not None else 0,
                    dst_nodata=0
                )

                aligned_bands[band_name] = aligned

            # Verify alignment
            non_zero = np.count_nonzero(aligned_bands[band_name])
            coverage = non_zero / aligned_bands[band_name].size * 100
            print(f"    Coverage: {coverage:.1f}% ({non_zero:,} pixels)")

    # Step 5: Stack bands
    print("\nStep 5: Stacking bands...")
    available_bands = [name for name in band_order if name in aligned_bands]
    stacked = np.stack([aligned_bands[name] for name in available_bands], axis=0)
    print(f"  Stacked shape: {stacked.shape} (bands × height × width)")
    print(f"  Band order: {available_bands}")

    # Step 6: Save multi-band GeoTIFF
    output_path = Path(output_dir) / f"{flight_name}_multispectral_aligned.tif"
    print(f"\nStep 6: Saving multi-band GeoTIFF...")
    print(f"  Output: {output_path}")

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=len(available_bands),
        dtype=aligned_bands[available_bands[0]].dtype,
        crs=ref_crs,
        transform=ref_transform,  # Use reference transform EXACTLY
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as dst:
        for idx, band_name in enumerate(available_bands, start=1):
            dst.write(aligned_bands[band_name], idx)
            dst.set_band_description(idx, band_name)

    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")

    # Step 7: Save individual aligned bands with IDENTICAL metadata
    print("\nStep 7: Saving individual aligned bands...")
    individual_dir = Path(output_dir) / "individual_bands"
    individual_dir.mkdir(exist_ok=True)

    for band_name in available_bands:
        band_path = individual_dir / f"{flight_name}_band_{band_name}_aligned.tif"

        with rasterio.open(
            band_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype=aligned_bands[band_name].dtype,
            crs=ref_crs,
            transform=ref_transform,  # CRITICAL: Same transform for all bands
            compress='lzw'
        ) as dst:
            dst.write(aligned_bands[band_name], 1)

        print(f"  {band_name}: {band_path.name}")

    # Step 8: Verify alignment in saved files
    print("\nStep 8: Verifying saved files have identical transforms...")
    all_match = True
    for band_name in available_bands:
        band_path = individual_dir / f"{flight_name}_band_{band_name}_aligned.tif"
        with rasterio.open(band_path) as src:
            if src.transform != ref_transform:
                print(f"  ERROR: {band_name} transform mismatch!")
                all_match = False
            elif src.crs != ref_crs:
                print(f"  ERROR: {band_name} CRS mismatch!")
                all_match = False
            elif src.shape != (height, width):
                print(f"  ERROR: {band_name} shape mismatch!")
                all_match = False

    if all_match:
        print(f"  All bands verified: Identical transforms, CRS, and dimensions")

    # Step 9: Calculate vegetation indices
    print("\nStep 9: Calculating vegetation indices...")
    indices_dir = Path(output_dir) / "vegetation_indices"
    indices_dir.mkdir(exist_ok=True)

    if 'NIR' in aligned_bands and 'R' in aligned_bands:
        nir = aligned_bands['NIR'].astype(np.float32)
        red = aligned_bands['R'].astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)

        ndvi_path = indices_dir / f"{flight_name}_NDVI.tif"
        with rasterio.open(
            ndvi_path, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype='float32',
            crs=ref_crs, transform=ref_transform, compress='lzw'
        ) as dst:
            dst.write(ndvi, 1)

        print(f"  NDVI: {ndvi_path.name}")

    if 'NIR' in aligned_bands and 'RE' in aligned_bands:
        nir = aligned_bands['NIR'].astype(np.float32)
        re = aligned_bands['RE'].astype(np.float32)

        ndre = (nir - re) / (nir + re + 1e-8)
        ndre = np.clip(ndre, -1, 1)

        ndre_path = indices_dir / f"{flight_name}_NDRE.tif"
        with rasterio.open(
            ndre_path, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype='float32',
            crs=ref_crs, transform=ref_transform, compress='lzw'
        ) as dst:
            dst.write(ndre, 1)

        print(f"  NDRE: {ndre_path.name}")

    if 'NIR' in aligned_bands and 'G' in aligned_bands:
        nir = aligned_bands['NIR'].astype(np.float32)
        green = aligned_bands['G'].astype(np.float32)

        gndvi = (nir - green) / (nir + green + 1e-8)
        gndvi = np.clip(gndvi, -1, 1)

        gndvi_path = indices_dir / f"{flight_name}_GNDVI.tif"
        with rasterio.open(
            gndvi_path, 'w', driver='GTiff',
            height=height, width=width, count=1, dtype='float32',
            crs=ref_crs, transform=ref_transform, compress='lzw'
        ) as dst:
            dst.write(gndvi, 1)

        print(f"  GNDVI: {gndvi_path.name}")

    print("\n" + "=" * 80)
    print("ALIGNMENT COMPLETE")
    print("=" * 80)
    print(f"\nMulti-band output: {output_path}")
    print(f"Individual bands: {individual_dir}/")
    print(f"Vegetation indices: {indices_dir}/")
    print(f"\nAll bands now share:")
    print(f"  - Identical CRS: {ref_crs}")
    print(f"  - Identical transform: {ref_transform}")
    print(f"  - Identical dimensions: {width} x {height} pixels")
    print(f"\nVerification in QGIS:")
    print(f"  1. Load all individual bands from {individual_dir}/")
    print(f"  2. They should overlay PERFECTLY - same pixel corresponds to same location")
    print(f"  3. Use 'Identify Features' tool - coordinates should match exactly")

    return str(output_path)


def verify_alignment_in_qgis(output_dir: str, flight_name: str):
    """
    Print instructions for verifying alignment in QGIS.
    """
    individual_dir = Path(output_dir) / "individual_bands"

    print("\n" + "=" * 80)
    print("VERIFICATION INSTRUCTIONS FOR QGIS")
    print("=" * 80)
    print(f"\n1. Open QGIS and add all rasters from:")
    print(f"   {individual_dir}/")
    print(f"\n2. All bands should be listed with identical:")
    print(f"   - CRS (in the Layers panel)")
    print(f"   - Extent (check Properties > Information)")
    print(f"   - Pixel size")
    print(f"\n3. Test pixel alignment:")
    print(f"   a. Activate 'Identify Features' tool")
    print(f"   b. Click the same visual feature across different bands")
    print(f"   c. The X,Y coordinates should be IDENTICAL (not just similar)")
    print(f"\n4. Visual overlay test:")
    print(f"   a. Change band rendering to 50% transparency")
    print(f"   b. Toggle bands on/off - edges should not shift")
    print(f"\nIf coordinates differ even slightly, the alignment has failed.")


def main():
    """
    Main function - align multispectral bands using reference band method.
    """
    # Your ODM output paths
    base_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/trial_alignment/reconstructed_02"

    band_paths = {
        'G': f"{base_dir}/02_band_G/odm_orthophoto/odm_orthophoto_02_band_G.tif",
        'R': f"{base_dir}/02_band_R/odm_orthophoto/odm_orthophoto_02_R.tif",
        'RE': f"{base_dir}/02_band_RE/odm_orthophoto/odm_orthophoto_02_RE.tif",
        'NIR': f"{base_dir}/02_band_NIR/odm_orthophoto/odm_orthophoto_02_band_NIR.tif"
    }

    output_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/trial_alignment/aligned_output_fixed"

    # Run alignment (using G band as reference)
    aligned_path = align_to_reference_band(
        band_paths,
        output_dir,
        reference_band='G',  # You can change to 'R', 'RE', or 'NIR' if preferred
        flight_name="flight_02"
    )

    # Print verification instructions
    verify_alignment_in_qgis(output_dir, "flight_02")

    print("\n" + "=" * 80)
    print("KEY DIFFERENCE FROM PREVIOUS SCRIPTS")
    print("=" * 80)
    print("\nOLD approach (align_multispectral_bands.py):")
    print("  - Calculated NEW transform using from_bounds()")
    print("  - This created a 'theoretical' grid that might not match any real band")
    print("  - Small rounding errors caused pixel misalignment")
    print("\nNEW approach (this script):")
    print("  - Uses one band's EXACT transform as reference")
    print("  - All other bands reproject to match this EXACT grid")
    print("  - No rounding errors, guaranteed pixel-perfect alignment")
    print("\nResult: All saved bands have IDENTICAL geospatial metadata")


if __name__ == "__main__":
    main()
