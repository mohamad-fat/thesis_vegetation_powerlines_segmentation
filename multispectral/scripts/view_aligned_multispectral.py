import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cv2
from pathlib import Path


def normalize_band(band, percentile_clip=True, lower=2, upper=98):
    """
    Normalize band data for visualization with proper contrast stretching.

    Args:
        band: numpy array of band data
        percentile_clip: If True, clip to percentiles for better contrast
        lower: Lower percentile for clipping
        upper: Upper percentile for clipping

    Returns:
        Normalized band (0-255 uint8)
    """
    # Remove zero values (no data)
    valid_data = band[band > 0]

    if len(valid_data) == 0:
        return np.zeros_like(band, dtype=np.uint8)

    if percentile_clip:
        # Use percentile clipping for better contrast
        vmin, vmax = np.percentile(valid_data, [lower, upper])
    else:
        vmin, vmax = valid_data.min(), valid_data.max()

    # Normalize to 0-255
    normalized = np.clip((band - vmin) / (vmax - vmin + 1e-8) * 255, 0, 255).astype(np.uint8)

    return normalized


def create_rgb_composite(r_band, g_band, b_band):
    """
    Create RGB composite with proper normalization.

    Args:
        r_band, g_band, b_band: Raw band data

    Returns:
        RGB image (height, width, 3) uint8
    """
    r_norm = normalize_band(r_band)
    g_norm = normalize_band(g_band)
    b_norm = normalize_band(b_band)

    rgb = np.stack([r_norm, g_norm, b_norm], axis=-1)
    return rgb


def create_false_color_cir(nir_band, r_band, g_band):
    """
    Create Color Infrared (CIR) false color composite.
    NIR -> Red, Red -> Green, Green -> Blue
    Vegetation appears red/pink.

    Args:
        nir_band, r_band, g_band: Raw band data

    Returns:
        CIR image (height, width, 3) uint8
    """
    nir_norm = normalize_band(nir_band)
    r_norm = normalize_band(r_band)
    g_norm = normalize_band(g_band)

    cir = np.stack([nir_norm, r_norm, g_norm], axis=-1)
    return cir


def visualize_multispectral(tif_path, output_path=None, show_plot=False):
    """
    Visualize aligned multispectral bands with proper contrast.

    Args:
        tif_path: Path to multi-band GeoTIFF
        output_path: Optional path to save visualization
        show_plot: If True, display plot interactively
    """
    with rasterio.open(tif_path) as src:
        print(f"Reading: {tif_path}")
        print(f"Bands: {src.count}, Shape: {src.shape}")
        print(f"CRS: {src.crs}")
        print()

        # Read all bands
        g_band = src.read(1)  # Green
        r_band = src.read(2)  # Red
        re_band = src.read(3)  # Red Edge
        nir_band = src.read(4)  # NIR

        # Print statistics
        for i, (band, name) in enumerate([(g_band, 'Green'), (r_band, 'Red'),
                                           (re_band, 'RedEdge'), (nir_band, 'NIR')], 1):
            valid = band[band > 0]
            if len(valid) > 0:
                print(f"{name} - Min: {valid.min()}, Max: {valid.max()}, Mean: {valid.mean():.1f}")
        print()

    # Create visualizations
    fig = plt.figure(figsize=(20, 12))

    # Individual bands (grayscale with proper contrast)
    band_data = [
        (g_band, 'Green (520-590nm)', 'Greens'),
        (r_band, 'Red (640-680nm)', 'Reds'),
        (re_band, 'Red Edge (730-740nm)', 'Oranges'),
        (nir_band, 'Near Infrared (820-890nm)', 'Purples')
    ]

    for idx, (band, title, cmap) in enumerate(band_data, 1):
        ax = plt.subplot(3, 4, idx)

        # Normalize for display
        band_norm = normalize_band(band)

        ax.imshow(band_norm, cmap=cmap, vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add statistics
        valid = band[band > 0]
        if len(valid) > 0:
            ax.text(0.02, 0.98, f'Range: {valid.min()}-{valid.max()}\nMean: {valid.mean():.0f}',
                   transform=ax.transAxes, fontsize=8, color='white',
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    # False color composites
    # 1. Color Infrared (CIR) - Vegetation appears red
    ax5 = plt.subplot(3, 4, 5)
    cir = create_false_color_cir(nir_band, r_band, g_band)
    ax5.imshow(cir)
    ax5.set_title('Color Infrared (NIR-R-G)\nVegetation = Red/Pink', fontsize=11, fontweight='bold')
    ax5.axis('off')

    # 2. Natural RGB (using R, G, and RE as blue substitute)
    ax6 = plt.subplot(3, 4, 6)
    rgb = create_rgb_composite(r_band, g_band, re_band)
    ax6.imshow(rgb)
    ax6.set_title('Pseudo RGB (R-G-RE)', fontsize=11, fontweight='bold')
    ax6.axis('off')

    # 3. Agriculture composite (RE-NIR-G)
    ax7 = plt.subplot(3, 4, 7)
    ag_composite = create_rgb_composite(re_band, nir_band, g_band)
    ax7.imshow(ag_composite)
    ax7.set_title('Agriculture (RE-NIR-G)', fontsize=11, fontweight='bold')
    ax7.axis('off')

    # 4. Urban composite (NIR-RE-R)
    ax8 = plt.subplot(3, 4, 8)
    urban = create_rgb_composite(nir_band, re_band, r_band)
    ax8.imshow(urban)
    ax8.set_title('Urban (NIR-RE-R)', fontsize=11, fontweight='bold')
    ax8.axis('off')

    # Vegetation indices
    # NDVI
    ax9 = plt.subplot(3, 4, 9)
    nir_f = nir_band.astype(float)
    r_f = r_band.astype(float)

    # Only calculate where both bands have data
    mask = (nir_band > 0) & (r_band > 0)
    ndvi = np.zeros_like(nir_f)
    ndvi[mask] = (nir_f[mask] - r_f[mask]) / (nir_f[mask] + r_f[mask] + 1e-8)
    ndvi = np.clip(ndvi, -1, 1)

    ndvi_colored = ax9.imshow(ndvi, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax9.set_title('NDVI\n(NIR - R) / (NIR + R)', fontsize=11, fontweight='bold')
    ax9.axis('off')
    plt.colorbar(ndvi_colored, ax=ax9, fraction=0.046)

    # NDRE (better for dense vegetation)
    ax10 = plt.subplot(3, 4, 10)
    re_f = re_band.astype(float)

    mask = (nir_band > 0) & (re_band > 0)
    ndre = np.zeros_like(nir_f)
    ndre[mask] = (nir_f[mask] - re_f[mask]) / (nir_f[mask] + re_f[mask] + 1e-8)
    ndre = np.clip(ndre, -1, 1)

    ndre_colored = ax10.imshow(ndre, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax10.set_title('NDRE (Red Edge)\n(NIR - RE) / (NIR + RE)', fontsize=11, fontweight='bold')
    ax10.axis('off')
    plt.colorbar(ndre_colored, ax=ax10, fraction=0.046)

    # GNDVI
    ax11 = plt.subplot(3, 4, 11)
    g_f = g_band.astype(float)

    mask = (nir_band > 0) & (g_band > 0)
    gndvi = np.zeros_like(nir_f)
    gndvi[mask] = (nir_f[mask] - g_f[mask]) / (nir_f[mask] + g_f[mask] + 1e-8)
    gndvi = np.clip(gndvi, -1, 1)

    gndvi_colored = ax11.imshow(gndvi, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax11.set_title('GNDVI (Green)\n(NIR - G) / (NIR + G)', fontsize=11, fontweight='bold')
    ax11.axis('off')
    plt.colorbar(gndvi_colored, ax=ax11, fraction=0.046)

    # Enhanced Vegetation Index (EVI)
    ax12 = plt.subplot(3, 4, 12)

    mask = (nir_band > 0) & (r_band > 0) & (g_band > 0)
    evi = np.zeros_like(nir_f)

    # EVI = 2.5 * (NIR - R) / (NIR + 6*R - 7.5*B + 1)
    # Using G as B substitute
    evi[mask] = 2.5 * (nir_f[mask] - r_f[mask]) / (nir_f[mask] + 6*r_f[mask] - 7.5*g_f[mask] + 1)
    evi = np.clip(evi, -1, 1)

    evi_colored = ax12.imshow(evi, cmap='RdYlGn', vmin=-0.5, vmax=1.0)
    ax12.set_title('EVI (Enhanced)\n2.5*(NIR-R)/(NIR+6R-7.5G+1)', fontsize=11, fontweight='bold')
    ax12.axis('off')
    plt.colorbar(evi_colored, ax=ax12, fraction=0.046)

    plt.suptitle(f'Aligned Multispectral Bands - {Path(tif_path).stem}',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def compare_original_vs_aligned(original_path, aligned_path, band_name, output_path=None):
    """
    Compare original ODM output vs aligned output for a single band.

    Args:
        original_path: Path to original ODM orthophoto
        aligned_path: Path to aligned band
        band_name: Name of the band (e.g., 'Green', 'NIR')
        output_path: Optional path to save comparison
    """
    with rasterio.open(original_path) as src_orig:
        orig_band = src_orig.read(1)

    with rasterio.open(aligned_path) as src_align:
        aligned_band = src_align.read(1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Original
    orig_norm = normalize_band(orig_band)
    axes[0].imshow(orig_norm, cmap='gray')
    axes[0].set_title(f'Original ODM Output - {band_name}\nShape: {orig_band.shape}',
                      fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Aligned
    aligned_norm = normalize_band(aligned_band)
    axes[1].imshow(aligned_norm, cmap='gray')
    axes[1].set_title(f'Aligned Output - {band_name}\nShape: {aligned_band.shape}',
                      fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to: {output_path}")

    plt.close()


def main():
    """
    Main function to visualize aligned multispectral data.
    """
    # Path to aligned multispectral stack
    aligned_path = "camera/data/trial_alignment/aligned_output/flight_02_multispectral_aligned.tif"

    # Output directory
    output_dir = "camera/data/trial_alignment/aligned_output/visualizations"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VISUALIZING ALIGNED MULTISPECTRAL BANDS")
    print("=" * 80)
    print()

    # Create comprehensive visualization
    output_viz = f"{output_dir}/flight_02_detailed_visualization.png"
    visualize_multispectral(aligned_path, output_path=output_viz, show_plot=False)

    print()
    print("=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nCheck the output at: {output_viz}")
    print("\nThis visualization shows:")
    print("  - Individual bands with proper contrast stretching")
    print("  - False color composites (CIR, RGB, Agriculture, Urban)")
    print("  - Vegetation indices (NDVI, NDRE, GNDVI, EVI)")
    print()
    print("If the images still look noisy:")
    print("  1. This might be normal for multispectral drone data")
    print("  2. Check the original ODM outputs for comparison")
    print("  3. Noise can be reduced during preprocessing/training")


if __name__ == "__main__":
    main()
