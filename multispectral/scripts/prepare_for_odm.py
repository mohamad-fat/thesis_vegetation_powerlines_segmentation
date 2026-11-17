import os
import shutil
import re
from pathlib import Path

def organize_bands_for_odm(source_dir, output_base_dir):
    """
    Organize multispectral images by band for OpenDroneMap processing.

    OpenDroneMap needs each band processed separately, then combined.
    This script creates separate directories for each band.

    Args:
        source_dir: Directory containing all multispectral images
        output_base_dir: Base directory where band-specific folders will be created
    """

    # Create output directory structure
    output_base = Path(output_base_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Define bands to process
    bands = ['G', 'R', 'NIR', 'RE']
    band_dirs = {}

    # Create separate directory for each band
    for band in bands:
        band_dir = output_base / f"band_{band}"
        band_dir.mkdir(exist_ok=True)
        band_dirs[band] = band_dir
        print(f"Created directory: {band_dir}")

    # Also create directory for RGB visual images
    rgb_dir = output_base / "RGB_visual"
    rgb_dir.mkdir(exist_ok=True)
    print(f"Created directory: {rgb_dir}")

    # Pattern to match multispectral files
    ms_pattern = r'DJI_(\d+)_(\d+)_MS_([A-Z]+)\.TIF'
    rgb_pattern = r'DJI_(\d+)_(\d+)_D\.JPG'

    # Get all files in source directory
    source_path = Path(source_dir)
    files = list(source_path.glob("*"))

    ms_counts = {band: 0 for band in bands}
    rgb_count = 0

    print(f"\nProcessing files from: {source_dir}")
    print("="*60)

    for file_path in files:
        filename = file_path.name

        # Match multispectral bands
        ms_match = re.match(ms_pattern, filename)
        if ms_match:
            timestamp = ms_match.group(1)
            sequence = ms_match.group(2)
            band = ms_match.group(3)

            if band in bands:
                dest_path = band_dirs[band] / filename
                shutil.copy2(file_path, dest_path)
                ms_counts[band] += 1

                if ms_counts[band] == 1:
                    print(f"Copying {band} band images...")

        # Match RGB visual images
        rgb_match = re.match(rgb_pattern, filename)
        if rgb_match:
            dest_path = rgb_dir / filename
            shutil.copy2(file_path, dest_path)
            rgb_count += 1

            if rgb_count == 1:
                print(f"Copying RGB visual images...")

    # Print summary
    print("\n" + "="*60)
    print("ORGANIZATION COMPLETE")
    print("="*60)
    print(f"\nImages organized by band:")
    for band in bands:
        print(f"  Band {band:3s}: {ms_counts[band]:3d} images -> {band_dirs[band]}")
    print(f"  RGB     : {rgb_count:3d} images -> {rgb_dir}")

    print(f"\nTotal output directory: {output_base}")

    return band_dirs, rgb_dir, ms_counts, rgb_count


def create_odm_commands(output_base_dir, project_name="multispectral_project"):
    """
    Generate OpenDroneMap commands for processing each band.

    Returns a script with commands to run ODM on each band separately.
    """
    output_base = Path(output_base_dir)

    # Create a shell script with ODM commands
    script_path = output_base / "run_odm_processing.sh"

    bands = ['G', 'R', 'NIR', 'RE']

    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# OpenDroneMap processing script for multispectral bands\n")
        f.write("# Generated automatically\n\n")

        f.write("# Set base directories\n")
        f.write(f'BASE_DIR="{output_base_dir}"\n')
        f.write(f'PROJECT_NAME="{project_name}"\n\n')

        f.write("# Check if docker is available\n")
        f.write('if ! command -v docker &> /dev/null; then\n')
        f.write('    echo "Error: Docker is not installed or not in PATH"\n')
        f.write('    echo "Please install Docker first: https://docs.docker.com/get-docker/"\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')

        f.write("echo 'Starting OpenDroneMap processing for multispectral bands'\n")
        f.write("echo '============================================================'\n\n")

        # Process each band
        for idx, band in enumerate(bands, 1):
            f.write(f"# Process Band {band}\n")
            f.write(f'echo ""\n')
            f.write(f'echo "Processing Band {band} ({idx}/4)..."\n')
            f.write(f'echo "------------------------------------------------------------"\n')

            f.write(f'docker run -ti --rm \\\n')
            f.write(f'  -v "$BASE_DIR/band_{band}":/datasets/code/images \\\n')
            f.write(f'  -v "$BASE_DIR/odm_output/band_{band}":/datasets/code/odm_output \\\n')
            f.write(f'  opendronemap/odm \\\n')
            f.write(f'  --project-path /datasets \\\n')
            f.write(f'  --radiometric-calibration camera \\\n')
            f.write(f'  --max-concurrency 4 \\\n')
            f.write(f'  --dsm \\\n')
            f.write(f'  --dtm \\\n')
            f.write(f'  --orthophoto-resolution 5.0\n\n')

        # Process RGB
        f.write(f"# Process RGB Visual\n")
        f.write(f'echo ""\n')
        f.write(f'echo "Processing RGB Visual (5/5)..."\n')
        f.write(f'echo "------------------------------------------------------------"\n')
        f.write(f'docker run -ti --rm \\\n')
        f.write(f'  -v "$BASE_DIR/RGB_visual":/datasets/code/images \\\n')
        f.write(f'  -v "$BASE_DIR/odm_output/RGB":/datasets/code/odm_output \\\n')
        f.write(f'  opendronemap/odm \\\n')
        f.write(f'  --project-path /datasets \\\n')
        f.write(f'  --max-concurrency 4 \\\n')
        f.write(f'  --dsm \\\n')
        f.write(f'  --dtm \\\n')
        f.write(f'  --orthophoto-resolution 5.0\n\n')

        f.write('echo ""\n')
        f.write('echo "============================================================"\n')
        f.write('echo "All processing complete!"\n')
        f.write('echo "Orthomosaics saved to: $BASE_DIR/odm_output/"\n')
        f.write('echo ""\n')
        f.write('echo "Output files for each band:"\n')

        for band in bands:
            f.write(f'echo "  Band {band}: $BASE_DIR/odm_output/band_{band}/code/odm_orthophoto/odm_orthophoto.tif"\n')
        f.write(f'echo "  RGB:    $BASE_DIR/odm_output/RGB/code/odm_orthophoto/odm_orthophoto.tif"\n')

    # Make script executable
    os.chmod(script_path, 0o755)

    print(f"\nCreated ODM processing script: {script_path}")
    print(f"\nTo run OpenDroneMap processing:")
    print(f"  1. Install Docker: https://docs.docker.com/get-docker/")
    print(f"  2. Pull ODM image: docker pull opendronemap/odm")
    print(f"  3. Run script: bash {script_path}")

    return script_path


def create_readme(output_base_dir):
    """
    Create a README with instructions for ODM processing.
    """
    readme_path = Path(output_base_dir) / "README_ODM_PROCESSING.md"

    with open(readme_path, 'w') as f:
        f.write("# OpenDroneMap Multispectral Processing\n\n")

        f.write("## Directory Structure\n\n")
        f.write("```\n")
        f.write("odm_ready/\n")
        f.write("├── band_G/           # Green band images\n")
        f.write("├── band_R/           # Red band images\n")
        f.write("├── band_NIR/         # Near Infrared band images\n")
        f.write("├── band_RE/          # Red Edge band images\n")
        f.write("├── RGB_visual/       # RGB camera images\n")
        f.write("├── run_odm_processing.sh  # Processing script\n")
        f.write("└── README_ODM_PROCESSING.md  # This file\n")
        f.write("```\n\n")

        f.write("## Prerequisites\n\n")
        f.write("1. Install Docker: https://docs.docker.com/get-docker/\n")
        f.write("2. Pull OpenDroneMap image:\n")
        f.write("   ```bash\n")
        f.write("   docker pull opendronemap/odm\n")
        f.write("   ```\n\n")

        f.write("## Processing Steps\n\n")
        f.write("### Option 1: Automatic Processing (Recommended)\n\n")
        f.write("Run the provided script to process all bands:\n\n")
        f.write("```bash\n")
        f.write("bash run_odm_processing.sh\n")
        f.write("```\n\n")

        f.write("This will process all 4 multispectral bands + RGB sequentially.\n\n")

        f.write("### Option 2: Manual Processing\n\n")
        f.write("Process individual bands manually:\n\n")
        f.write("```bash\n")
        f.write("# Process Green band\n")
        f.write("docker run -ti --rm \\\n")
        f.write("  -v $(pwd)/band_G:/datasets/code/images \\\n")
        f.write("  -v $(pwd)/odm_output/band_G:/datasets/code/odm_output \\\n")
        f.write("  opendronemap/odm \\\n")
        f.write("  --project-path /datasets \\\n")
        f.write("  --radiometric-calibration camera \\\n")
        f.write("  --max-concurrency 4\n")
        f.write("```\n\n")

        f.write("Repeat for other bands (R, NIR, RE).\n\n")

        f.write("## Output\n\n")
        f.write("After processing, orthomosaics will be saved to:\n\n")
        f.write("```\n")
        f.write("odm_output/\n")
        f.write("├── band_G/code/odm_orthophoto/odm_orthophoto.tif\n")
        f.write("├── band_R/code/odm_orthophoto/odm_orthophoto.tif\n")
        f.write("├── band_NIR/code/odm_orthophoto/odm_orthophoto.tif\n")
        f.write("├── band_RE/code/odm_orthophoto/odm_orthophoto.tif\n")
        f.write("└── RGB/code/odm_orthophoto/odm_orthophoto.tif\n")
        f.write("```\n\n")

        f.write("## Next Steps\n\n")
        f.write("1. Stack the orthomosaics into a single multi-band image\n")
        f.write("2. Use the stacked image for vegetation classification\n")
        f.write("3. Apply your trained model across the entire mosaic\n\n")

        f.write("## Processing Parameters\n\n")
        f.write("- `--radiometric-calibration camera`: Uses camera calibration data\n")
        f.write("- `--orthophoto-resolution 5.0`: Output resolution in cm/pixel\n")
        f.write("- `--dsm`: Generate Digital Surface Model\n")
        f.write("- `--dtm`: Generate Digital Terrain Model\n")
        f.write("- `--max-concurrency 4`: Number of parallel processes\n\n")

        f.write("## Troubleshooting\n\n")
        f.write("### Memory Issues\n")
        f.write("Add `--split 2` to split dataset into smaller chunks\n\n")
        f.write("### Resolution Issues\n")
        f.write("Adjust `--orthophoto-resolution` (lower number = higher resolution)\n\n")
        f.write("### More Options\n")
        f.write("See: https://docs.opendronemap.org/arguments/\n")

    print(f"Created README: {readme_path}")
    return readme_path


def main():
    # Configuration
    source_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/mavic_multispectral/DJI_202510021041_017_Geri-MS"
    output_base_dir = "/Users/user/Desktop/university/thesis/veg_enc_code/camera/data/odm_ready"
    project_name = "geri_multispectral"

    print("="*60)
    print("PREPARING MULTISPECTRAL DATA FOR OPENDRONEMAP")
    print("="*60)
    print(f"\nSource: {source_dir}")
    print(f"Output: {output_base_dir}\n")

    # Organize bands
    band_dirs, rgb_dir, ms_counts, rgb_count = organize_bands_for_odm(
        source_dir,
        output_base_dir
    )

    # Create processing scripts and documentation
    script_path = create_odm_commands(output_base_dir, project_name)
    readme_path = create_readme(output_base_dir)

    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Install Docker if not already installed")
    print(f"  2. Pull ODM image: docker pull opendronemap/odm")
    print(f"  3. Run processing: bash {script_path}")
    print(f"  4. Read instructions: {readme_path}")
    print(f"\nProcessing will create orthomosaics in:")
    print(f"  {output_base_dir}/odm_output/")


if __name__ == "__main__":
    main()
