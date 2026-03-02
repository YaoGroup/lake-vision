#!/usr/bin/env python
"""
Preprocess raw imagery timestacks into combined datasets (imagery + water_area),
optionally appending labels and class probabilities as attributes.

Usage:
    python preprocess_lakes.py \
        --tstack_dir /path/to/tstacks \
        --area_file /path/to/all_lakes_2019.nc \
        --output_dir /path/to/processed \
        --labels_file /path/to/labels.csv \
        --max_lakes 50
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import xarray as xr

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lakevision.data.preprocessing import load_area_sequences, combine_lake_data


def load_label_lookup(labels_path, label_columns):
    """Load CSV and build a lookup dict keyed by new_id."""
    labels = pd.read_csv(labels_path)
    print(f"Loaded {len(labels)} rows from {labels_path}")

    # Validate that expected columns exist
    missing = [c for c in ["new_id"] + label_columns if c not in labels.columns]
    if missing:
        raise ValueError(f"Missing columns in labels CSV: {missing}\n"
                         f"Available columns: {list(labels.columns)}")

    lookup = labels.set_index("new_id")[label_columns].to_dict(orient="index")
    print(f"Built label lookup for {len(lookup)} lakes")
    return lookup


def append_labels_to_nc(nc_path, label_row):
    """Open an existing .nc file and append label attributes."""
    ds = xr.open_dataset(nc_path)
    for col, val in label_row.items():
        # Convert numpy types to native Python for clean NetCDF storage
        if hasattr(val, "item"):
            val = val.item()
        ds.attrs[col] = val
    ds.to_netcdf(nc_path, mode="w")
    ds.close()


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
        "--labels_file",
        type=str,
        default=None,
        help="Path to labels CSV with columns: new_id, label_rines, rinesID, p_nd, p_ed, p_ld, p_rd",
    )
    parser.add_argument(
        "--label_columns",
        type=str,
        nargs="+",
        default=["label_rines", "rinesID", "p_nd", "p_ed", "p_ld", "p_rd"],
        help="Columns from labels CSV to append as attributes",
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
    parser.add_argument(
        "--labels_only",
        action="store_true",
        help="Skip preprocessing; only append labels to existing .nc files in output_dir",
    )

    args = parser.parse_args()

    tstack_dir = Path(args.tstack_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label lookup if provided
    label_lookup = None
    if args.labels_file:
        label_lookup = load_label_lookup(args.labels_file, args.label_columns)
        print()

    # --- Labels-only mode: just append to existing files ---
    if args.labels_only:
        if label_lookup is None:
            print("ERROR: --labels_only requires --labels_file")
            sys.exit(1)

        nc_files = sorted(output_dir.glob("*.nc"))
        print(f"Labels-only mode: found {len(nc_files)} .nc files in {output_dir}\n")

        matched, skipped = 0, 0
        for nc_path in nc_files:
            lake_id = nc_path.stem  # e.g., CW2019_1524
            if lake_id in label_lookup:
                append_labels_to_nc(str(nc_path), label_lookup[lake_id])
                matched += 1
            else:
                print(f"  SKIP {nc_path.name}: '{lake_id}' not found in CSV")
                skipped += 1

        print(f"\nDone! {matched} labeled, {skipped} skipped")
        return

    # --- Full preprocessing mode ---
    tstack_files = sorted(tstack_dir.glob("tstack_*.nc"))

    if args.max_lakes is not None:
        tstack_files = tstack_files[:args.max_lakes]

    total_files = len(tstack_files)
    print(f"Found {total_files} tstack files to process")
    print(f"Output directory: {output_dir}")
    if label_lookup:
        print(f"Labels: will append {len(args.label_columns)} attributes per lake")
    print()

    # Load area sequences once
    print("Loading area sequences...")
    area_ds = load_area_sequences(args.area_file)
    print(f"Loaded area data with {len(area_ds['ids'])} lakes")
    print()

    # Process each lake
    successful = 0
    failed = 0
    label_matched = 0
    label_skipped = 0
    start_time = time.time()
    last_checkpoint = start_time

    print(f"Starting processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    sys.stdout.flush()

    for i, tstack_path in enumerate(tstack_files):
        # Extract lake_id from filename (e.g., tstack_CW2019_1579.nc -> CW2019_1579)
        lake_id = tstack_path.stem.replace("tstack_", "")
        output_path = output_dir / f"{lake_id}.nc"

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

            # Append labels if available
            if label_lookup and lake_id in label_lookup:
                append_labels_to_nc(str(output_path), label_lookup[lake_id])
                label_matched += 1
            elif label_lookup:
                label_skipped += 1

        except Exception as e:
            print(f"  [{i+1}/{total_files}] ERROR processing {lake_id}: {e}")
            failed += 1
            sys.stdout.flush()
            continue

        # Print progress every 50 files or every 5 minutes
        current_time = time.time()
        if (i + 1) % 50 == 0 or (current_time - last_checkpoint) > 300:
            elapsed = current_time - start_time
            rate = (i + 1) / elapsed  # files per second
            remaining = (total_files - i - 1) / rate if rate > 0 else 0

            elapsed_str = str(timedelta(seconds=int(elapsed)))
            remaining_str = str(timedelta(seconds=int(remaining)))

            print(f"  [{i+1}/{total_files}] {lake_id} | "
                  f"Elapsed: {elapsed_str} | "
                  f"ETA: {remaining_str} | "
                  f"Rate: {rate:.2f} files/sec | "
                  f"Time: {datetime.now().strftime('%H:%M:%S')}")
            sys.stdout.flush()
            last_checkpoint = current_time

    # Final summary
    end_time = time.time()
    total_elapsed = end_time - start_time
    total_elapsed_str = str(timedelta(seconds=int(total_elapsed)))

    print("=" * 60)
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {total_elapsed_str}")
    print(f"Processed: {successful} successful, {failed} failed")
    if label_lookup:
        print(f"Labels: {label_matched} matched, {label_skipped} not found in CSV")
    print(f"Average rate: {successful / total_elapsed:.2f} files/sec")
    print(f"Output saved to {output_dir}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()