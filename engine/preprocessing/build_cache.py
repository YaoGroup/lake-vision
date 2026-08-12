#!/usr/bin/env python3
"""
Build a blosc2 training cache from the ESSD v2 stacks.

Why this exists
---------------
Training reads were ~99% of wall-clock. Per sample the old path cost 2.74 s, of
which 1.94 s (71%) was zlib decompression of float32 NetCDF at 330 MB/s. This
cache changes three things, each measured:

  float32 -> uint16 (raw DN)          2.0x fewer bytes, bit-exact lossless
  mask stored once, not broadcast     removes a 153x-redundant field
  zlib -> blosc2 + bitshuffle         2.7x smaller, 4.8x faster to decode
                                      (median over 16 lakes; 1585 MB/s 1-thread)

Layout (one file per channel, so an experiment stages only what it uses):

    cache/B04/<lake_id>.b2nd            [T, 512, 512] uint16, 65535 = no data
    cache/B03/...  B02/  B08/  B11/  B12/
    cache/lake_boundary/<lake_id>.b2nd  [512, 512]    uint8   (static, ONCE)
    cache/water_mask_ndwi/<lake_id>.b2nd[T, 512, 512] uint8   (255 = fill)
    cache/scalars/<lake_id>.npz         p_water, eo_cloud_cover, pct_nans,
                                        boa_add_offset, time

Reflectance is stored as round(DN * 2) -- see DN_SCALE. Conversion to surface
reflectance is (stored/2 + boa_add_offset) / 10000 and is deferred to the
GPU -- that is what keeps the dataloader queue at uint16 and lets bs=64 fit.
boa_add_offset can be negative (ESA baseline >= 04.00 uses -1000), which is why
it must NOT be applied before the uint16 cast.

Usage
-----
    python build_cache.py \
        --stacks_root /oak/.../sherlock_sattilestack/stacks_v2 \
        --out_root    $L_SCRATCH/cache \
        --years CW_2018 CW_2019 \
        --limit 50                      # prototype; omit for the full build
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import blosc2
except ImportError:
    sys.exit("blosc2 not installed. On Sherlock: pip install --user blosc2")

import netCDF4

# Canonical band names in the v2 stacks. NOTE the axis order is red-first
# (B04, B03, B02, ...), NOT blue-first -- always index via band_name, never by
# position, or red and blue silently swap.
ALL_BANDS = ["B04", "B03", "B02", "B08", "B11", "B12"]

NODATA_U16 = 65535   # reflectance sentinel
# Reflectance is stored as round(DN * DN_SCALE). The v2 stacks average two
# observations when both fall in one cadence bin, so ~0.6% of RGB values (and
# ~40% of B11/B12, which are 20 m bands resampled to 10 m) carry a fractional
# part of exactly 0.5 -- never any other fraction. Scaling by 2 makes every
# value exactly integral, so the uint16 cast is genuinely lossless rather than
# quietly rounding. Max observed DN is ~15k, so 2x leaves >2x headroom in uint16.
DN_SCALE = 2
SCALARS = ["p_water", "eo_cloud_cover", "pct_nans", "boa_add_offset"]

CPARAMS = dict(
    codec=blosc2.Codec.LZ4,
    filters=[blosc2.Filter.BITSHUFFLE],
    clevel=5,
)


def _cparams():
    return blosc2.CParams(**CPARAMS)


def _decode_band_names(nc):
    """Read band_name as a list of strings like ['B04', 'B03', ...].

    `band_name` is a NetCDF char array [band, string3]. netCDF4 hands it back as
    a 2-D array of single characters (dtype U1/S1), so calling .tobytes() on a
    row yields UTF-32-padded garbage ('B\\x00\\x00\\x000\\x00...'). Join the
    characters per row instead, and strip NUL padding.
    """
    var = nc.variables["band_name"]
    raw = var[:]
    if raw.ndim == 2:
        names = ["".join(str(c) for c in row) for row in raw]
    else:
        names = [str(v) for v in raw]
    return [n.replace("\x00", "").strip() for n in names]


def build_one(nc_path, out_root, bands, masks, overwrite=False):
    """Convert a single v2 stack to cache files. Returns dict of bytes written."""
    lake_id = nc_path.stem
    written = {}

    with netCDF4.Dataset(str(nc_path)) as nc:
        nc.set_auto_mask(False)

        raw_names = _decode_band_names(nc)

        refl = nc.variables["reflectance"]
        T = refl.shape[0]

        for band in bands:
            if band not in raw_names:
                raise ValueError(f"{lake_id}: band {band} not in {raw_names}")
            dst = out_root / band / f"{lake_id}.b2nd"
            if dst.exists() and not overwrite:
                written[band] = dst.stat().st_size
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)

            bi = raw_names.index(band)
            # Hyperslab on the chunked channel axis: only this band decompresses.
            a = np.asarray(refl[:, bi, :, :], dtype=np.float32)
            finite = np.isfinite(a)
            if a[finite].size and a[finite].min() < 0:
                raise ValueError(
                    f"{lake_id}/{band}: negative raw DN ({a[finite].min()}); "
                    f"uint16 cast would wrap. Investigate before caching."
                )
            scaled = np.rint(a * DN_SCALE)
            hi = scaled[finite].max() if finite.any() else 0
            if hi >= NODATA_U16:
                raise ValueError(
                    f"{lake_id}/{band}: DN*{DN_SCALE} reaches {hi:.0f}, which "
                    f"collides with the {NODATA_U16} nodata sentinel.")
            u = np.where(finite, scaled, NODATA_U16).astype(np.uint16)
            del scaled
            del a, finite

            blosc2.asarray(u, urlpath=str(dst), mode="w",
                           chunks=(min(51, T), 512, 512), cparams=_cparams())
            written[band] = dst.stat().st_size
            del u

        for m in masks:
            if m not in nc.variables:
                print(f"  WARN {lake_id}: no variable {m}, skipping")
                continue
            dst = out_root / m / f"{lake_id}.b2nd"
            if dst.exists() and not overwrite:
                written[m] = dst.stat().st_size
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            v = np.asarray(nc.variables[m][:], dtype=np.uint8)
            chunks = (min(51, T), 512, 512) if v.ndim == 3 else None
            blosc2.asarray(v, urlpath=str(dst), mode="w",
                           chunks=chunks, cparams=_cparams())
            written[m] = dst.stat().st_size
            del v

        sc = {}
        for s in SCALARS:
            if s in nc.variables:
                sc[s] = np.asarray(nc.variables[s][:], dtype=np.float32)
        sc["time"] = np.asarray(nc.variables["time"][:], dtype=np.int64)
        if "drainage_label" in nc.variables:
            sc["drainage_label"] = np.asarray(nc.variables["drainage_label"][:])

        dst = out_root / "scalars" / f"{lake_id}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **sc)
        written["scalars"] = dst.stat().st_size

    return lake_id, written


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stacks_root", required=True,
                   help="Directory containing CW_2018/ and CW_2019/")
    p.add_argument("--out_root", required=True, help="Cache destination")
    p.add_argument("--years", nargs="+", default=["CW_2018", "CW_2019"])
    p.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"],
                   choices=ALL_BANDS,
                   help="Reflectance bands to cache (default RGB)")
    p.add_argument("--masks", nargs="+",
                   default=["lake_boundary", "water_mask_ndwi"],
                   help="Mask variables to cache. cloud_mask deliberately "
                        "excluded: NDWI is a normalized difference and so is "
                        "robust to cloud brightness (measured "
                        "P(water|cloudy)/P(water|clear) = 0.16x).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cache only the first N lakes per year (prototype).")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    stacks_root = Path(args.stacks_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = []
    for y in args.years:
        yr = sorted((stacks_root / y).glob("*.nc"))
        if not yr:
            sys.exit(f"No .nc files under {stacks_root / y}")
        files += yr[: args.limit] if args.limit else yr

    print(f"Building cache for {len(files)} lakes")
    print(f"  bands : {args.bands}")
    print(f"  masks : {args.masks}")
    print(f"  out   : {out_root}")
    print(f"  codec : LZ4 + bitshuffle, clevel 5\n")

    t0 = time.time()
    totals, src_bytes, ids = {}, 0, []
    for i, fp in enumerate(files, 1):
        src_bytes += fp.stat().st_size
        lake_id, w = build_one(fp, out_root, args.bands, args.masks,
                               overwrite=args.overwrite)
        ids.append(lake_id)
        for k, v in w.items():
            totals[k] = totals.get(k, 0) + v
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  {i}/{len(files)}  {el:6.1f}s  "
                  f"({el / i:.2f}s/lake, eta {(len(files) - i) * el / i / 60:.1f} min)")

    out = sum(totals.values())
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min")
    print(f"  source .nc : {src_bytes / 1e9:7.2f} GB")
    print(f"  cache      : {out / 1e9:7.2f} GB  ({src_bytes / max(out,1):.2f}x smaller)")
    print(f"  per lake   : {out / len(files) / 1e6:7.1f} MB")
    for k in sorted(totals):
        print(f"    {k:20s} {totals[k] / 1e6:8.1f} MB")

    manifest = {
        "lake_ids": ids,
        "bands": args.bands,
        "masks": args.masks,
        "nodata_u16": NODATA_U16,
        "dn_scale": DN_SCALE,
        "cparams": {k: str(v) for k, v in CPARAMS.items()},
        "note": ("reflectance stored as round(DN * dn_scale); "
                 "surface_reflectance = (stored/dn_scale + boa_add_offset)/10000"),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
