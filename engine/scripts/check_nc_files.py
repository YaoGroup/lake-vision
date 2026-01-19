#!/usr/bin/env python3
"""
Quick script to check NC files for corruption and cloudy_seq variables.

Usage:
    python check_nc_files.py /path/to/nc/dir [--sample N]
"""
import argparse
from pathlib import Path


def check_files(nc_dir: str, sample: int = None):
    """Check NC files for readability and variables."""
    import xarray as xr

    nc_dir = Path(nc_dir)
    files = sorted(nc_dir.glob("*.nc"))

    if sample and sample < len(files):
        # Sample evenly across the file list
        step = len(files) // sample
        files = files[::step][:sample]

    print(f"Checking {len(files)} NC files in {nc_dir}")
    print("=" * 70)

    stats = {
        "total": len(files),
        "readable": 0,
        "corrupt": 0,
        "has_cloudy_rgb": 0,
        "has_cloudy_rgbn": 0,
        "has_cloudy_bns16": 0,
    }
    corrupt_files = []

    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f)
            vars = list(ds.data_vars)
            ds.close()

            stats["readable"] += 1
            if "cloudy_seq_rgb" in vars:
                stats["has_cloudy_rgb"] += 1
            if "cloudy_seq_rgbn" in vars:
                stats["has_cloudy_rgbn"] += 1
            if "cloudy_seq_bns16" in vars:
                stats["has_cloudy_bns16"] += 1

            # Progress indicator
            if (i + 1) % 50 == 0:
                print(f"  Checked {i + 1}/{len(files)} files...")

        except Exception as e:
            stats["corrupt"] += 1
            corrupt_files.append((f.name, str(e)[:50]))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:        {stats['total']}")
    print(f"Readable:           {stats['readable']}")
    print(f"Corrupt/unreadable: {stats['corrupt']}")
    print()
    print(f"Has cloudy_seq_rgb:   {stats['has_cloudy_rgb']}")
    print(f"Has cloudy_seq_rgbn:  {stats['has_cloudy_rgbn']}")
    print(f"Has cloudy_seq_bns16: {stats['has_cloudy_bns16']}")

    if corrupt_files:
        print("\n" + "=" * 70)
        print(f"CORRUPT FILES ({len(corrupt_files)}):")
        print("=" * 70)
        for name, err in corrupt_files:
            print(f"  {name}: {err}")

    return stats, corrupt_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check NC files for corruption")
    parser.add_argument("nc_dir", help="Directory containing NC files")
    parser.add_argument("--sample", "-n", type=int, default=None,
                        help="Only check N files (sampled evenly)")
    args = parser.parse_args()

    check_files(args.nc_dir, args.sample)
