import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt


def align_with_geospatial(band_paths: Dict[str, str], output_dir: str,
                         flight_name: str = "flight") -> str:
    """
    Best method: Pure geospatial alignment using ODM's georeferencing.

    This is the RECOMMENDED method because:
    1. ODM already georeferenced all bands to the same coordinate system
    2. Each band has accurate geospatial metadata
    3. Reprojection preserves accuracy while ensuring perfect alignment
    4. Fast and reliable

    Args:
        band_paths: Dict mapping band names to file paths
        output_dir: Output directory
        flight_name: Flight identifier

    Returns:
        Path to aligned multi-band GeoTIFF
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MULTISPECTRAL BAND ALIGNMENT")
    print("Method: Geospatial Reprojection (ODM Metadata)")
    print("=" * 80)
    print()

    # Step 1: Read metadata from all bands
    print("Step 1: Reading band metadata...")
    band_infos = {}
    band_order = ['G', 'R', 'RE', 'NIR']

    for band_name in band_order:
        if band_name not in band_paths:
            print(f"  WARNING: {band_name} band not found, skipping")
            continue

        with rasterio.open(band_paths[band_name]) as src:
            band_infos[band_name] = {
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'shape': src.shape,
                'resolution': src.res,
                'dtype': src.dtypes[0],
                'nodata': src.nodata
            }
            print(f"  {band_name}: {src.shape} @ {src.res[0]:.4f}m/px")

    if len(band_infos) == 0:
        raise ValueError("No bands found!")

    # Step 2: Check georeferencing quality
    print("\nStep 2: Checking georeferencing quality...")
    crs_list = [info['crs'] for info in band_infos.values()]
    if len(set(str(crs) for crs in crs_list)) > 1:
        print("  ERROR: Bands have different CRS - ODM processing may have failed!")
        return None
    print(f"  All bands share same CRS: {crs_list[0]}")

    # Check if bounds are similar (they should overlap significantly)
    bounds_list = [info['bounds'] for info in band_infos.values()]
    for band_name, info in band_infos.items():
        b = info['bounds']
        area = (b.right - b.left) * (b.top - b.bottom)
        print(f"  {band_name} coverage: {area/1e6:.2f} km²")

    # Step 3: Find common extent (intersection)
    print("\nStep 3: Finding common extent (intersection)...")
    minx = max([b.left for b in bounds_list])
    miny = max([b.bottom for b in bounds_list])
    maxx = min([b.right for b in bounds_list])
    maxy = min([b.top for b in bounds_list])

    common_area = (maxx - minx) * (maxy - miny)
    print(f"  Common area: {common_area/1e6:.2f} km²")
    print(f"  Bounds: ({minx:.2f}, {miny:.2f}) to ({maxx:.2f}, {maxy:.2f})")

    if common_area <= 0:
        print("  ERROR: No overlap between bands!")
        return None

    # Step 4: Determine target resolution (finest available)
    print("\nStep 4: Determining target resolution...")
    resolutions = [info['resolution'] for info in band_infos.values()]
    target_x_res = min([r[0] for r in resolutions])
    target_y_res = min([abs(r[1]) for r in resolutions])
    print(f"  Target resolution: {target_x_res:.4f} x {target_y_res:.4f} m/px")

    # Calculate output dimensions
    width = int((maxx - minx) / target_x_res)
    height = int((maxy - miny) / target_y_res)
    print(f"  Output dimensions: {width} x {height} pixels")

    # Create target transform
    target_transform = from_bounds(minx, miny, maxx, maxy, width, height)
    target_crs = crs_list[0]

    # Step 5: Reproject and align all bands
    print("\nStep 5: Reprojecting bands to common grid...")
    aligned_bands = {}

    for band_name in band_order:
        if band_name not in band_paths:
            continue

        print(f"  Processing {band_name}...")

        with rasterio.open(band_paths[band_name]) as src:
            # Read source band
            source_data = src.read(1)

            # Create destination array
            aligned = np.zeros((height, width), dtype=source_data.dtype)

            # Reproject using bilinear interpolation
            reproject(
                source=rasterio.band(src, 1),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=0
            )

            aligned_bands[band_name] = aligned

            # Check alignment quality
            non_zero = np.count_nonzero(aligned)
            total = aligned.size
            coverage = non_zero / total * 100
            print(f"    Coverage: {coverage:.1f}% ({non_zero:,} / {total:,} pixels)")

    # Step 6: Stack and save
    print("\nStep 6: Stacking bands...")
    available_bands = [name for name in band_order if name in aligned_bands]
    stacked = np.stack([aligned_bands[name] for name in available_bands], axis=0)
    print(f"  Stacked shape: {stacked.shape} (bands × height × width)")
    print(f"  Band order: {available_bands}")

    # Save multi-band GeoTIFF
    output_path = Path(output_dir) / f"{flight_name}_multispectral_aligned.tif"
    print(f"\nStep 7: Saving to {output_path}...")

    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=len(available_bands),
        dtype=stacked.dtype,
        crs=target_crs,
        transform=target_transform,
        compress='lzw',
        tiled=True,
        blockxsize=256,
        blockysize=256
    ) as dst:
        for idx, band_name in enumerate(available_bands, start=1):
            dst.write(aligned_bands[band_name], idx)
            dst.set_band_description(idx, band_name)

    print(f"  File size: {output_path.stat().st_size / 1024**2:.1f} MB")

    # Save individual bands
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
            crs=target_crs,
            transform=target_transform,
            compress='lzw'
        ) as dst:
            dst.write(aligned_bands[band_name], 1)

    print(f"  Individual bands saved to: {individual_dir}")

    print("\n" + "=" * 80)
    print("ALIGNMENT COMPLETE!")
    print("=" * 80)
    print(f"\nOutput: {output_path}")
    print(f"Bands: {', '.join(available_bands)}")
    print(f"Dimensions: {width} × {height} pixels")
    print(f"Resolution: {target_x_res:.4f} m/pixel")
    print("\nAll bands are now perfectly aligned and ready for:")
    print("  1. Labeling in QGIS (all 4 bands move together)")
    print("  2. Vegetation index calculation")
    print("  3. Model training")

    return str(output_path)


def verify_alignment_quality(aligned_path: str):
    """
    Verify the quality of alignment by checking band statistics.

    Args:
        aligned_path: Path to aligned multi-band GeoTIFF
    """
    print("\n" + "=" * 80)
    print("ALIGNMENT QUALITY CHECK")
    print("=" * 80)

    with rasterio.open(aligned_path) as src:
        print(f"\nFile: {Path(aligned_path).name}")
        print(f"Bands: {src.count}")
        print(f"Dimensions: {src.width} × {src.height}")
        print(f"CRS: {src.crs}")
        print(f"Resolution: {src.res}")
        print()

        for i in range(1, src.count + 1):
            band_name = src.descriptions[i-1] if src.descriptions[i-1] else f"Band {i}"
            band = src.read(i)

            non_zero = np.count_nonzero(band)
            coverage = non_zero / band.size * 100

            valid_data = band[band > 0]
            if len(valid_data) > 0:
                print(f"{band_name}:")
                print(f"  Coverage: {coverage:.1f}%")
                print(f"  Range: {valid_data.min()} - {valid_data.max()}")
                print(f"  Mean: {valid_data.mean():.1f}")
                print(f"  Std: {valid_data.std():.1f}")
            else:
                print(f"{band_name}: NO DATA!")
            print()

    print("Quality Assessment:")
    print("  ✓ If coverage > 40%: Good alignment")
    print("  ✓ If all bands have similar coverage: Excellent alignment")
    print("  ✗ If coverage < 30%: Check ODM outputs")


def main():
    """
    Main function - align multispectral bands from ODM.
    """
    # Your ODM output paths
    base_dir = "camera/data/trial_alignment/reconstructed_02"

    band_paths = {
        'G': f"{base_dir}/02_band_G/odm_orthophoto/odm_orthophoto_02_band_G.tif",
        'R': f"{base_dir}/02_band_R/odm_orthophoto/odm_orthophoto_02_R.tif",
        'RE': f"{base_dir}/02_band_RE/odm_orthophoto/odm_orthophoto_02_RE.tif",
        'NIR': f"{base_dir}/02_band_NIR/odm_orthophoto/odm_orthophoto_02_band_NIR.tif"
    }

    output_dir = "camera/data/trial_alignment/aligned_output_best"

    # Run alignment
    aligned_path = align_with_geospatial(band_paths, output_dir, flight_name="flight_02")

    if aligned_path:
        # Verify quality
        verify_alignment_quality(aligned_path)

        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("\n1. Open in QGIS to visually verify:")
        print(f"   {aligned_path}")
        print("\n2. Check that all 4 bands are perfectly overlaid")
        print("\n3. Use this file for labeling - when you draw polygons,")
        print("   all 4 bands will be labeled simultaneously!")
        print("\n4. For other flights, update band_paths in this script")


if __name__ == "__main__":
    main()
