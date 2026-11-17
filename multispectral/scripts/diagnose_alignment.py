import rasterio
from pathlib import Path
import numpy as np


def diagnose_alignment(aligned_dir: str):
    """
    Diagnose alignment issues by comparing transforms of aligned bands.

    This will show you exactly where the misalignment is coming from.

    Args:
        aligned_dir: Directory containing aligned individual bands
    """
    print("=" * 80)
    print("ALIGNMENT DIAGNOSTIC TOOL")
    print("=" * 80)
    print()

    individual_dir = Path(aligned_dir) / "individual_bands"

    if not individual_dir.exists():
        print(f"ERROR: Directory not found: {individual_dir}")
        print(f"\nTrying parent directory: {aligned_dir}")
        individual_dir = Path(aligned_dir)

    # Find all .tif files
    tif_files = sorted(individual_dir.glob("*_aligned.tif"))

    if len(tif_files) == 0:
        print(f"ERROR: No aligned GeoTIFF files found in {individual_dir}")
        return

    print(f"Found {len(tif_files)} aligned band files:")
    for f in tif_files:
        print(f"  - {f.name}")
    print()

    # Read metadata from all files
    print("=" * 80)
    print("READING GEOSPATIAL METADATA")
    print("=" * 80)
    print()

    band_metadata = {}

    for tif_file in tif_files:
        band_name = tif_file.stem.split('_')[-2]  # Extract band name (G, R, RE, NIR)

        with rasterio.open(tif_file) as src:
            band_metadata[band_name] = {
                'file': tif_file.name,
                'crs': src.crs,
                'transform': src.transform,
                'shape': src.shape,
                'bounds': src.bounds,
                'resolution': src.res
            }

            print(f"{band_name} band ({tif_file.name}):")
            print(f"  CRS: {src.crs}")
            print(f"  Dimensions: {src.width} x {src.height}")
            print(f"  Transform:")
            print(f"    {src.transform}")
            print(f"  Bounds:")
            print(f"    Left:   {src.bounds.left:.10f}")
            print(f"    Bottom: {src.bounds.bottom:.10f}")
            print(f"    Right:  {src.bounds.right:.10f}")
            print(f"    Top:    {src.bounds.top:.10f}")
            print(f"  Resolution: {src.res[0]:.10f} x {abs(src.res[1]):.10f} m/px")
            print()

    # Check for alignment issues
    print("=" * 80)
    print("ALIGNMENT VERIFICATION")
    print("=" * 80)
    print()

    if len(band_metadata) < 2:
        print("ERROR: Need at least 2 bands to check alignment")
        return

    # Use first band as reference
    ref_band = list(band_metadata.keys())[0]
    ref_meta = band_metadata[ref_band]

    print(f"Using {ref_band} as reference band")
    print()

    all_aligned = True
    issues = []

    for band_name, meta in band_metadata.items():
        if band_name == ref_band:
            continue

        print(f"Comparing {band_name} to {ref_band}:")

        # Check CRS
        if str(meta['crs']) != str(ref_meta['crs']):
            print(f"  CRS MISMATCH")
            print(f"    {ref_band}: {ref_meta['crs']}")
            print(f"    {band_name}: {meta['crs']}")
            all_aligned = False
            issues.append(f"{band_name}: CRS mismatch")
        else:
            print(f"  CRS: OK (both use {meta['crs']})")

        # Check dimensions
        if meta['shape'] != ref_meta['shape']:
            print(f"  DIMENSION MISMATCH")
            print(f"    {ref_band}: {ref_meta['shape']}")
            print(f"    {band_name}: {meta['shape']}")
            all_aligned = False
            issues.append(f"{band_name}: Dimension mismatch")
        else:
            print(f"  Dimensions: OK (both {meta['shape'][1]}x{meta['shape'][0]})")

        # Check transform (CRITICAL for pixel alignment)
        if meta['transform'] != ref_meta['transform']:
            print(f"  TRANSFORM MISMATCH - THIS CAUSES MISALIGNMENT IN QGIS")

            # Show differences
            ref_t = ref_meta['transform']
            cur_t = meta['transform']

            print(f"    {ref_band} transform: {ref_t}")
            print(f"    {band_name} transform: {cur_t}")

            # Calculate differences
            diff_a = abs(cur_t.a - ref_t.a)
            diff_b = abs(cur_t.b - ref_t.b)
            diff_c = abs(cur_t.c - ref_t.c)
            diff_d = abs(cur_t.d - ref_t.d)
            diff_e = abs(cur_t.e - ref_t.e)
            diff_f = abs(cur_t.f - ref_t.f)

            print(f"    Differences:")
            print(f"      Pixel width (a):  {diff_a:.15f}")
            print(f"      Rotation (b):     {diff_b:.15f}")
            print(f"      Top-left X (c):   {diff_c:.15f} meters")
            print(f"      Rotation (d):     {diff_d:.15f}")
            print(f"      Pixel height (e): {diff_e:.15f}")
            print(f"      Top-left Y (f):   {diff_f:.15f} meters")

            # Estimate pixel shift
            if diff_a > 0:
                pixel_shift_x = diff_c / ref_t.a
                print(f"      Estimated X shift: {pixel_shift_x:.3f} pixels")

            if diff_e > 0:
                pixel_shift_y = diff_f / abs(ref_t.e)
                print(f"      Estimated Y shift: {pixel_shift_y:.3f} pixels")

            all_aligned = False
            issues.append(f"{band_name}: Transform mismatch (pixel shift)")
        else:
            print(f"  Transform: OK (identical)")

        # Check bounds
        bounds_match = (
            abs(meta['bounds'].left - ref_meta['bounds'].left) < 1e-9 and
            abs(meta['bounds'].bottom - ref_meta['bounds'].bottom) < 1e-9 and
            abs(meta['bounds'].right - ref_meta['bounds'].right) < 1e-9 and
            abs(meta['bounds'].top - ref_meta['bounds'].top) < 1e-9
        )

        if not bounds_match:
            print(f"  BOUNDS MISMATCH")
            diff_left = abs(meta['bounds'].left - ref_meta['bounds'].left)
            diff_bottom = abs(meta['bounds'].bottom - ref_meta['bounds'].bottom)
            diff_right = abs(meta['bounds'].right - ref_meta['bounds'].right)
            diff_top = abs(meta['bounds'].top - ref_meta['bounds'].top)
            print(f"    Left diff:   {diff_left:.10f} m")
            print(f"    Bottom diff: {diff_bottom:.10f} m")
            print(f"    Right diff:  {diff_right:.10f} m")
            print(f"    Top diff:    {diff_top:.10f} m")
        else:
            print(f"  Bounds: OK (identical)")

        print()

    # Final verdict
    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print()

    if all_aligned:
        print("RESULT: All bands are PERFECTLY ALIGNED")
        print()
        print("All bands have:")
        print("  - Identical CRS")
        print("  - Identical dimensions")
        print("  - Identical transforms (same pixel grid)")
        print("  - Identical bounds")
        print()
        print("These bands should overlay perfectly in QGIS with no pixel shift.")
    else:
        print("RESULT: ALIGNMENT ISSUES DETECTED")
        print()
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print()
        print("PROBLEM: Your bands have different transforms or dimensions.")
        print("This means pixels at the same row/column do NOT correspond to")
        print("the same geographic location.")
        print()
        print("SOLUTION: Use the fixed alignment script:")
        print("  python camera/align_multispectral_fixed.py")
        print()
        print("The fixed script uses a REFERENCE BAND approach:")
        print("  1. Pick one band as reference (e.g., Green)")
        print("  2. Use its EXACT transform for output")
        print("  3. Reproject all other bands to match this grid")
        print("  4. All outputs get IDENTICAL geospatial metadata")
        print()
        print("This guarantees pixel-perfect alignment in QGIS.")

    print()
    print("=" * 80)


def main():
    """Main function."""
    import sys

    if len(sys.argv) > 1:
        aligned_dir = sys.argv[1]
    else:
        # Default path
        aligned_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/trial_alignment/aligned_output"

    print(f"Diagnosing alignment in: {aligned_dir}\n")
    diagnose_alignment(aligned_dir)


if __name__ == "__main__":
    main()
