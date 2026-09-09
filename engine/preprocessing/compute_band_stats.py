#!/usr/bin/env python3
"""Per-band mean and std over the TRAINING split, for --band_stats / --fill mean.

Streams the deposit files listed in an ids JSON, accumulates sum and sum of
squares in float64 over finite pixels only (NaN = unobserved, never counted),
and writes {"red": {"mean": m, "std": s}, ...} in the format
lakevision.data.datasets.load_band_stats expects.

Sub-sampling: --stride 4 keeps every 4th pixel in y and x, --tstride 3 every
3rd day. Per file that is still ~2.7 M samples per band; the estimate is
statistically indistinguishable from the full pass and ~50x faster. Never
run this over val or test ids.

Usage (repo root):
    python engine/preprocessing/compute_band_stats.py \
        --nc_dir /path/to/stacks_v2/CW_2019 \
        --ids_file splits/essd_CW_crossyear/train_ids.json \
        --out band_stats_crossyear_train.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lakevision.data.datasets import LakeDataset  # noqa: E402  (BAND_TO_CHANNEL)


def band_names(nc):
    def _decode(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)
    if 'band_name' in nc.variables:
        raw = [''.join(_decode(c) for c in np.atleast_1d(row)).strip('\x00').strip()
               for row in nc.variables['band_name'][:]]
    else:
        raw = [_decode(b) for b in nc.variables['band'][:]]
    return [LakeDataset.BAND_TO_CHANNEL.get(b, b) for b in raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_dir", required=True, help="directory of {lake_id}.nc deposit files")
    ap.add_argument("--ids_file", required=True, help="JSON list of lake ids (the TRAIN split)")
    ap.add_argument("--out", required=True, help="band_stats.json to write")
    ap.add_argument("--stride", type=int, default=4, help="pixel stride in y and x")
    ap.add_argument("--tstride", type=int, default=3, help="stride along time")
    args = ap.parse_args()

    ids = json.load(open(args.ids_file))
    nc_dir = Path(args.nc_dir)
    paths = [nc_dir / f"{lid}.nc" for lid in ids]
    missing = [p for p in paths if not p.exists()]
    if missing:
        sys.exit(f"ERROR: {len(missing)} of {len(paths)} files missing, e.g. {missing[0]}")

    acc = {}   # channel -> [n, sum, sumsq]
    t0 = time.time()
    for i, p in enumerate(paths):
        with netCDF4.Dataset(str(p)) as nc:
            nc.set_auto_mask(False)
            names = band_names(nc)
            var = nc.variables['reflectance'] if 'reflectance' in nc.variables else nc.variables['imagery']
            arr = np.asarray(var[::args.tstride, :, ::args.stride, ::args.stride], dtype=np.float64)
        for b, name in enumerate(names):
            if name == 'mask':
                continue
            x = arr[:, b]
            fin = np.isfinite(x)
            s = acc.setdefault(name, [0, 0.0, 0.0])
            s[0] += int(fin.sum()); s[1] += float(x[fin].sum()); s[2] += float((x[fin] ** 2).sum())
        if (i + 1) % 50 == 0 or i + 1 == len(paths):
            el = time.time() - t0
            print(f"  {i + 1}/{len(paths)} files, {el:.0f}s elapsed, ~{el / (i + 1) * (len(paths) - i - 1):.0f}s left")

    stats = {}
    for name, (n, s, ss) in acc.items():
        mean = s / n
        var = max(ss / n - mean ** 2, 0.0)
        stats[name] = {"mean": mean, "std": float(np.sqrt(var)), "n_pixels": n}
    stats["_provenance"] = {"ids_file": str(args.ids_file), "n_files": len(paths),
                            "stride": args.stride, "tstride": args.tstride}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(stats, open(args.out, "w"), indent=1)
    for name in [k for k in stats if not k.startswith("_")]:
        print(f"{name:8s} mean {stats[name]['mean']:9.1f}  std {stats[name]['std']:9.1f}  n {stats[name]['n_pixels']:,}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
