#!/usr/bin/env python3
"""
Find out WHY the cached pipeline is still dataloader-bound.

Job 39234460 (32 cores, 24 workers) was no faster than 39080585 (16 cores, 12
workers) -- 35.55s vs 32.94s loader time for 200 lakes. Doubling the cores did
nothing, so the loader is not limited by parallel per-worker work. Aggregate
throughput is ~1.3 GB/s on a node with 1007 GB RAM and the cache resident in
node-local page cache, which is far below what the hardware should do.

Four hypotheses survive that measurement, and they imply DIFFERENT fixes:

  H1 worker spawn      persistent_workers=False respawns every worker every
                       epoch. 24 workers costs twice the spawn tax of 12, which
                       alone could explain why more workers got slower.
                       Fix: persistent_workers=True (needs care -- see the
                       comment in run_training.py about train/val loaders).

  H2 pin_memory        Pinning runs in ONE thread in the parent process. Every
                       batch is 8 x 230 MB = 1.84 GB. Does not scale with
                       workers. Fix: is on the transfer path, not the cache.

  H3 IPC / shm         Workers ship each 230 MB sample through shared memory;
                       the parent maps and copies it. Also does not scale.
                       Fix: fewer/larger transfers, or fewer workers.

  H4 memory bandwidth  The 4+ full-sample copies per __getitem__ (np.stack
                       building [T,C,H,W] from three [T,H,W] planes, then
                       ascontiguousarray) saturate the bus, so extra workers
                       only contend. Fix: re-layout the cache as one
                       [T,C,H,W] array per lake -- a ~2.3 h rebuild.

H1-H3 all predict "more workers doesn't help" and are fixed WITHOUT touching the
cache. H4 predicts the same and needs the rebuild. That is why this script
exists: to avoid paying for a rebuild that may fix nothing.

WHAT EACH PHASE TELLS YOU
  1 raw floor      blosc2 decode rate with no DataLoader at all. The ceiling.
  2 stage split    where __getitem__'s time actually goes. Big np.stack => H4.
  3 dataset direct one process, no IPC, no pinning, no spawn. Compare to phase 4.
  4 loader grid    workers x pin_memory x persistent_workers, with first-batch
                   (spawn) reported separately from steady state.

READING THE RESULT
  phase 3 >> phase 4 steady        the DataLoader machinery is the cost (H2/H3)
  pin=False much faster            H2
  persistent=True much faster      H1
  flat in workers AND phase 2 is
    dominated by stack/contiguous  H4 -- the rebuild is justified
  phase 1 ~ phase 3                decode is not the problem (expected)

Usage (inside a GPU allocation, after the cache is built):
    python diagnose_loader.py --cache_root $L_SCRATCH/cache
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.data.cached_dataset import CachedLakeDataset, worker_init

try:
    import blosc2
except ImportError:
    sys.exit("blosc2 not installed")


def mbps(nbytes, seconds):
    return nbytes / 1024**2 / seconds if seconds > 0 else float("nan")


# ---------------------------------------------------------------- phase 1
def phase_raw(root, bands, lake_ids):
    """Decode straight from blosc2. No dataset, no torch, no copies past the read."""
    blosc2.set_nthreads(1)
    total = 0
    t0 = time.perf_counter()
    for lid in lake_ids:
        for b in bands:
            a = blosc2.open(str(Path(root) / b / f"{lid}.b2nd"))[:]
            total += a.nbytes
            del a
    dt = time.perf_counter() - t0
    return dict(seconds=dt, gb=total / 1024**3, mb_s=mbps(total, dt),
                s_per_lake=dt / len(lake_ids))


# ---------------------------------------------------------------- phase 2
def phase_stages(root, bands, lake_ids, seq_len):
    """Time each step inside __getitem__ separately."""
    blosc2.set_nthreads(1)
    acc = dict(read=0.0, stack=0.0, contig_view=0.0, torch=0.0)
    root = Path(root)
    for lid in lake_ids:
        t = time.perf_counter()
        planes = [blosc2.open(str(root / b / f"{lid}.b2nd"))[:] for b in bands]
        acc["read"] += time.perf_counter() - t

        t = time.perf_counter()
        img = np.stack(planes, axis=1)
        acc["stack"] += time.perf_counter() - t
        del planes

        t = time.perf_counter()
        arr = np.ascontiguousarray(img).view(np.int16)
        acc["contig_view"] += time.perf_counter() - t

        t = time.perf_counter()
        _ = torch.from_numpy(arr)
        acc["torch"] += time.perf_counter() - t
        del img, arr

    n = len(lake_ids)
    return {k: v / n for k, v in acc.items()}


# ---------------------------------------------------------------- phase 3
def phase_dataset(ds, n):
    """__getitem__ in this process: no IPC, no pinning, no worker spawn."""
    t0 = time.perf_counter()
    nbytes = 0
    for i in range(n):
        img = ds[i][0]
        nbytes += img.numel() * img.element_size()
        del img
    dt = time.perf_counter() - t0
    return dict(seconds=dt, s_per_lake=dt / n, mb_s=mbps(nbytes, dt))


# ---------------------------------------------------------------- phase 4
def phase_loader(ds, batch_size, workers, pin, persistent, epochs=2):
    """Full DataLoader. Separates first-batch latency (spawn) from steady state."""
    kw = {}
    if workers > 0:
        kw.update(prefetch_factor=1, persistent_workers=persistent,
                  worker_init_fn=worker_init)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=pin, drop_last=False, **kw)

    first_batch_s, steady_s, nbytes = [], [], 0
    for _ in range(epochs):
        t_epoch = time.perf_counter()
        t_first = None
        for i, batch in enumerate(loader):
            if i == 0:
                t_first = time.perf_counter() - t_epoch
                t_steady = time.perf_counter()
            img = batch[0]
            nbytes += img.numel() * img.element_size()
            del batch, img
        first_batch_s.append(t_first)
        steady_s.append(time.perf_counter() - t_steady)
    del loader

    tot = float(np.mean(first_batch_s)) + float(np.mean(steady_s))
    return dict(
        total_s=tot,
        first_batch_s=float(np.mean(first_batch_s)),
        steady_s=float(np.mean(steady_s)),
        mb_s=mbps(nbytes / epochs, tot),
        s_per_lake=tot / len(ds),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache_root", required=True)
    p.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"])
    p.add_argument("--mask", default=None)
    p.add_argument("--seq_len", type=int, default=153)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--n_lakes", type=int, default=96,
                   help="Lakes per configuration. Keep small; every cell repeats it.")
    p.add_argument("--n_raw", type=int, default=16,
                   help="Lakes for the single-threaded phases 1-3 (slow per lake).")
    p.add_argument("--workers", nargs="+", type=int, default=[0, 4, 12, 24])
    p.add_argument("--out", default=None)
    a = p.parse_args()

    ds_all = CachedLakeDataset(a.cache_root, bands=a.bands, mask=a.mask,
                               seq_len=a.seq_len)
    ids = ds_all.lake_ids[: a.n_lakes]
    ds = CachedLakeDataset(a.cache_root, lake_ids=ids, bands=a.bands,
                           mask=a.mask, seq_len=a.seq_len)
    sample_mb = a.seq_len * ds.n_channels * 512 * 512 * 2 / 1024**2

    print("=" * 78)
    print("LOADER DIAGNOSIS")
    print("=" * 78)
    print(f"cache      : {a.cache_root}")
    print(f"bands      : {a.bands}   mask: {a.mask}")
    print(f"lakes      : {len(ds)} (raw phases use {a.n_raw})")
    print(f"sample     : {sample_mb:.0f} MB   batch_size: {a.batch_size}")
    print(f"cpus       : {torch.get_num_threads()} torch threads\n")

    results = {"sample_mb": sample_mb, "n_lakes": len(ds)}

    print("-" * 78)
    print("PHASE 1  raw blosc2 decode (no DataLoader, 1 thread) -- the ceiling")
    print("-" * 78)
    r1 = phase_raw(a.cache_root, a.bands, ids[: a.n_raw])
    results["raw"] = r1
    print(f"  {r1['gb']:.2f} GB in {r1['seconds']:.2f}s   "
          f"{r1['mb_s']:.0f} MB/s   {r1['s_per_lake']:.3f} s/lake\n")

    print("-" * 78)
    print("PHASE 2  where __getitem__ time goes (per lake, 1 thread)")
    print("-" * 78)
    r2 = phase_stages(a.cache_root, a.bands, ids[: a.n_raw], a.seq_len)
    results["stages"] = r2
    tot2 = sum(r2.values())
    for k, v in r2.items():
        print(f"  {k:<14} {v*1000:8.1f} ms   {100*v/tot2:5.1f}%")
    print(f"  {'TOTAL':<14} {tot2*1000:8.1f} ms\n")

    print("-" * 78)
    print("PHASE 3  dataset __getitem__ in-process (no IPC, no pin, no spawn)")
    print("-" * 78)
    r3 = phase_dataset(ds, min(a.n_raw, len(ds)))
    results["dataset"] = r3
    print(f"  {r3['s_per_lake']:.3f} s/lake   {r3['mb_s']:.0f} MB/s\n")

    print("-" * 78)
    print("PHASE 4  DataLoader grid  (prefetch_factor=1 throughout)")
    print("-" * 78)
    print(f"{'workers':>7} {'pin':>6} {'persist':>8} {'total':>8} {'spawn':>8} "
          f"{'steady':>8} {'MB/s':>8} {'s/lake':>8}")
    print("-" * 78)
    grid = []
    for w in a.workers:
        combos = ([(True, False), (False, False), (True, True)] if w > 0
                  else [(True, False), (False, False)])
        for pin, persist in combos:
            try:
                r = phase_loader(ds, a.batch_size, w, pin, persist)
            except Exception as e:                      # OOM, worker death, ...
                print(f"{w:>7} {str(pin):>6} {str(persist):>8}   FAILED: "
                      f"{type(e).__name__}: {str(e)[:40]}")
                continue
            r.update(workers=w, pin_memory=pin, persistent=persist)
            grid.append(r)
            print(f"{w:>7} {str(pin):>6} {str(persist):>8} "
                  f"{r['total_s']:>7.2f}s {r['first_batch_s']:>7.2f}s "
                  f"{r['steady_s']:>7.2f}s {r['mb_s']:>8.0f} {r['s_per_lake']:>8.3f}")
    results["grid"] = grid

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not grid:
        sys.exit("no loader configuration completed")

    def ratio(num, den):
        return f"{num/den:.2f}x" if den > 1e-9 else "n/a"

    best = max(grid, key=lambda r: r["mb_s"])
    base = next((r for r in grid if r["workers"] == max(a.workers)
                 and r["pin_memory"] and not r["persistent"]), best)
    print(f"current production config ~ workers={base['workers']}, pin=True, "
          f"persistent=False: {base['mb_s']:.0f} MB/s")
    print(f"best measured: workers={best['workers']}, pin={best['pin_memory']}, "
          f"persistent={best['persistent']}: {best['mb_s']:.0f} MB/s "
          f"({ratio(best['mb_s'], base['mb_s'])})")

    # Same worker count, pin on vs off -- isolates H2 from everything else.
    for w in sorted({r["workers"] for r in grid}):
        on = next((r for r in grid if r["workers"] == w and r["pin_memory"]
                   and not r["persistent"]), None)
        off = next((r for r in grid if r["workers"] == w and not r["pin_memory"]
                    and not r["persistent"]), None)
        if on and off:
            print(f"H2 check (workers={w}): pin=False is "
                  f"{ratio(off['mb_s'], on['mb_s'])} vs pin=True")

    stack_frac = (r2["stack"] + r2["contig_view"]) / tot2
    print(f"\nH4 check: stack+contiguous is {100*stack_frac:.0f}% of __getitem__ "
          f"-> cache re-layout would remove ~{100*stack_frac:.0f}% of phase-3 time")
    print(f"H1 check: spawn is {100*base['first_batch_s']/base['total_s']:.0f}% "
          f"of the current config's epoch (paid EVERY epoch when persistent=False)")
    print(f"in-process ceiling (phase 3): {r3['mb_s']:.0f} MB/s vs "
          f"loader {base['mb_s']:.0f} MB/s "
          f"-> DataLoader machinery costs {ratio(r3['mb_s'], base['mb_s'])}")

    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=2))
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
