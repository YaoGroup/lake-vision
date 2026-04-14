"""
Batch-synthesize ESSD composite NetCDFs for a region × year.

Phase 1 of the preprocessing pipeline: compose
  sat-tile-stack raw .nc + Dunmire water_area + Dunmire static polygon + GUI labels
into one per-lake composite NetCDF.

Phase 2 (the cloudy-tile useful-flag inference pass) is a separate SLURM step;
see run_synthesize_region.sh.

Usage
-----
    python engine/preprocessing/synthesize_region.py \
        --region CW \
        --year 2018 \
        --raw_dir /oak/.../stacks/CW_2018 \
        --dunmire_area /oak/.../data/dunmire/all_lakes_2018.nc \
        --dunmire_polygons /oak/.../repos/sat-tile-stack/labeling/dunmire/labels_2018_volumes.geojson \
        --labels_csv /oak/.../data/essd_labels/labels_CW_2018.csv \
        --output_dir /oak/.../sherlock_lakevision/composites/CW_2018 \
        --workers 8

The ``--labels_csv`` flag is optional; if omitted, composites are still built
but lack the per-lake label attributes (useful for lakes labeled later).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import pandas as pd
import xarray as xr

# Add repo root to path so `from lakevision...` works when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.data.synthesis import LakeDatasetSynthesizer


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

def _validate_composite(path: Path) -> bool:
    """Quick check: does an existing composite look sane?"""
    try:
        ds = xr.open_dataset(path)
        ok = (
            'imagery' in ds
            and 'water_area' in ds
            and ds.sizes.get('time', 0) == 153
            and ds.sizes.get('channel', 0) == 7
            and ds.attrs.get('Conventions') == 'CF-1.8'
        )
        ds.close()
        return ok
    except Exception:
        return False


def _synthesize_one(task) -> dict:
    """Worker: synthesize one lake. Returns a status dict."""
    (lake_id, year, raw_nc, dunmire_area_path, dunmire_polygons_path,
     labels_csv_path, output_dir) = task

    outfile = Path(output_dir) / f"{lake_id}.nc"

    # Skip if already built and valid
    if outfile.exists():
        if _validate_composite(outfile):
            return {'status': 'skip', 'id': lake_id}
        else:
            outfile.unlink()

    try:
        # Lazily open deps in the worker (can't pickle xarray Datasets well)
        import geopandas as gpd
        area_ds = xr.open_dataset(dunmire_area_path)
        polygons_gdf = gpd.read_file(dunmire_polygons_path)

        label_row = None
        if labels_csv_path is not None:
            labels_df = pd.read_csv(labels_csv_path)
            # Label CSVs use 'lake_id' as the id column
            if lake_id in labels_df['lake_id'].values:
                label_row = labels_df[labels_df['lake_id'] == lake_id].iloc[0]

        synth = LakeDatasetSynthesizer(
            raw_nc=raw_nc,
            dunmire_area_ds=area_ds,
            dunmire_polygons_gdf=polygons_gdf,
            lake_id=lake_id,
            year=year,
            label_row=label_row,
        )
        out_path = synth.synthesize(output_dir)
        area_ds.close()
        return {'status': 'ok', 'id': lake_id, 'path': str(out_path)}
    except Exception as e:
        return {
            'status': 'error',
            'id': lake_id,
            'error': f'{type(e).__name__}: {e}',
            'traceback': traceback.format_exc(),
        }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Synthesize ESSD composite NetCDFs.")
    p.add_argument('--region', required=True, help='IMBIE region code (e.g. CW)')
    p.add_argument('--year', type=int, required=True, choices=[2018, 2019])
    p.add_argument('--raw_dir', required=True,
                   help='Directory of raw sat-tile-stack .nc files')
    p.add_argument('--dunmire_area', required=True,
                   help='Path to Dunmire all_lakes_{YEAR}.nc')
    p.add_argument('--dunmire_polygons', required=True,
                   help='Path to Dunmire labels_{YEAR}_volumes.geojson')
    p.add_argument('--labels_csv', default=None,
                   help='Optional: GUI labels CSV to append per-lake label attrs')
    p.add_argument('--output_dir', required=True,
                   help='Output directory for composite .nc files')
    p.add_argument('--workers', type=int, default=1,
                   help='Number of parallel workers (default: 1)')
    p.add_argument('--start', type=int, default=0,
                   help='Starting lake index (for subsetting)')
    p.add_argument('--count', type=int, default=None,
                   help='Number of lakes to process (None = all)')
    return p.parse_args()


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover lake IDs from raw .nc filenames
    raw_files = sorted(raw_dir.glob('*.nc'))
    if not raw_files:
        print(f"ERROR: no .nc files found in {raw_dir}")
        sys.exit(1)
    lake_ids = [fp.stem for fp in raw_files]

    # Subset
    end = args.start + args.count if args.count else len(lake_ids)
    end = min(end, len(lake_ids))
    subset_ids = lake_ids[args.start:end]
    subset_files = raw_files[args.start:end]

    print(f"{'='*60}")
    print(f"ESSD Composite Synthesis: {args.region} {args.year}")
    print(f"{'='*60}")
    print(f"Raw dir:          {raw_dir} ({len(lake_ids)} .nc files)")
    print(f"Building:         {len(subset_ids)} composites (idx {args.start}..{end-1})")
    print(f"Dunmire area:     {args.dunmire_area}")
    print(f"Dunmire polygons: {args.dunmire_polygons}")
    print(f"Labels CSV:       {args.labels_csv or '(none)'}")
    print(f"Output dir:       {output_dir}")
    print(f"Workers:          {args.workers}")
    print(f"{'='*60}\n", flush=True)

    # Build task list
    tasks = [
        (lid, args.year, fp, args.dunmire_area, args.dunmire_polygons,
         args.labels_csv, str(output_dir))
        for lid, fp in zip(subset_ids, subset_files)
    ]

    start_time = time.time()
    n_ok = n_skip = n_error = 0
    error_details = []

    if args.workers == 1:
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] {task[0]}", flush=True)
            result = _synthesize_one(task)
            status = result['status']
            if status == 'ok':
                n_ok += 1
            elif status == 'skip':
                n_skip += 1
            else:
                n_error += 1
                error_details.append(result)
                print(f"  ERROR: {result['error']}", flush=True)
    else:
        print(f"Starting {args.workers} parallel workers...\n", flush=True)
        with Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_synthesize_one, tasks):
                status = result['status']
                if status == 'ok':
                    n_ok += 1
                elif status == 'skip':
                    n_skip += 1
                else:
                    n_error += 1
                    error_details.append(result)
                done = n_ok + n_skip + n_error
                if done % 25 == 0 or done == len(tasks):
                    elapsed = time.time() - start_time
                    print(f"  [{done}/{len(tasks)}] {n_ok} ok, "
                          f"{n_skip} skip, {n_error} err ({elapsed:.0f}s)",
                          flush=True)

    total = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Done in {total:.1f}s ({total/60:.1f} min)")
    print(f"  Success: {n_ok}")
    print(f"  Skipped: {n_skip}")
    print(f"  Errors:  {n_error}")
    if error_details:
        print(f"\n--- First 3 error details ---")
        for e in error_details[:3]:
            print(f"  {e['id']}: {e['error']}")
    print(f"{'='*60}", flush=True)

    sys.exit(0 if n_error == 0 else 1)


if __name__ == '__main__':
    main()
