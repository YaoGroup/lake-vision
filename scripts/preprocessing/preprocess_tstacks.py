#!/usr/bin/env python
"""
Preprocess raw imagery timestacks into combined datasets (imagery + water_area).

Usage:
    python preprocess_lakes.py \
        --tstack_dir /path/to/tstacks \
        --area_file /path/to/all_lakes_2019.nc \
        --output_dir /path/to/processed \
        --max_lakes 50
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lakevision.data.preprocessing import load_area_sequences, combine_lake_data


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw timestacks into combined datasets."
    )
    parser.add_argument(
        "--tstack_dir",
        type=str,
        required=True,
        help="Directory containing raw tstack_*.nc files",
    )
    parser.add_argument(
        "--area_file",
        type=str,
        required=True,
        help="Path to area sequences file (e.g., all_lakes_2019.nc)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed .nc files",
    )
    parser.add_argument(
        "--max_lakes",
        type=int,
        default=None,
        help="Maximum number of lakes to process (default: all)",
    )
    parser.add_argument(
        "--mask_band",
        type=str,
        default="mask",
        help="Name of mask band in tstacks (default: 'mask')",
    )
    parser.add_argument(
        "--no_spectral",
        action="store_true",
        help="Exclude NIR and SWIR bands (RGB + mask only)",
    )
    parser.add_argument(
        "--spectral_bands",
        type=str,
        nargs="+",
        default=["nir", "swir16", "swir22"],
        help="Spectral bands to include (default: nir swir16 swir22)",
    )

    args = parser.parse_args()

    tstack_dir = Path(args.tstack_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all tstack files
    tstack_files = sorted(tstack_dir.glob("tstack_*.nc"))
    
    if args.max_lakes is not None:
        tstack_files = tstack_files[:args.max_lakes]

    print(f"Found {len(tstack_files)} tstack files to process")
    print(f"Output directory: {output_dir}")
    print()

    # Load area sequences once
    print("Loading area sequences...")
    area_ds = load_area_sequences(args.area_file)
    print(f"Loaded area data with {len(area_ds['ids'])} lakes")
    print()

    # Process each lake
    successful = 0
    failed = 0

    for i, tstack_path in enumerate(tstack_files):
        # Extract lake_id from filename (e.g., tstack_CW2019_1579.nc -> CW2019_1579)
        lake_id = tstack_path.stem.replace("tstack_", "")
        output_path = output_dir / f"{lake_id}.nc"

        print(f"[{i+1}/{len(tstack_files)}] Processing {lake_id}...")

        try:
            combine_lake_data(
                imagery_path=str(tstack_path),
                area_ds=area_ds,
                lake_id=lake_id,
                output_path=str(output_path),
                mask_band_name=args.mask_band,
                fill_nans=True,
                include_spectral_bands=not args.no_spectral,
                spectral_bands=args.spectral_bands,
            )
            successful += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print()
    print(f"Done! Processed {successful} lakes, {failed} failed")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    main()
