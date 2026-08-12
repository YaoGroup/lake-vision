#!/usr/bin/env python3
"""
Measure the cached pipeline end-to-end. This is the go/no-go gate before any
full-scale cache build.

Every throughput number in the JSTARS plan is a *component* estimate -- the
assembled pipeline (collate + pin_memory + H2D + GPU overlap) has never been
measured. This script separates the two things that matter:

    dataloader-only : iterate batches, touch nothing else
    full step       : dataloader + normalize + forward + backward

If dataloader-only >> full-step-minus-dataloader, we are still I/O bound and the
cache did not deliver. If they are comparable, the workload is balanced and
scaling to 5000 lakes is a matter of nodes, not redesign.

Usage (inside a GPU allocation):
    python bench_pipeline.py --cache_root $L_SCRATCH/cache \
        --batch_sizes 8 32 64 --epochs 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.data.cached_dataset import (
    CachedLakeDataset, normalize_batch, worker_init,
)
from lakevision.data.loader_plan import plan_loader_workers
from lakevision.models.classifier import LakeDrainageClassifier


def human(sec):
    return f"{sec:6.2f}s" if sec < 90 else f"{sec / 60:6.2f}m"


def bench(cache_root, bands, mask, batch_size, workers, epochs,
          seq_len, device, do_backward, grad_ckpt):
    ds = CachedLakeDataset(cache_root, bands=bands, mask=mask, seq_len=seq_len)
    n_ch = ds.n_channels

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(), prefetch_factor=2 if workers else None,
        persistent_workers=False, worker_init_fn=worker_init, drop_last=False,
    )

    # ---- 1. dataloader only -------------------------------------------------
    t0 = time.perf_counter()
    nb = 0
    for _ in range(epochs):
        for batch in loader:
            nb += 1
            del batch
    t_load = (time.perf_counter() - t0) / epochs

    # ---- 2. full training step ---------------------------------------------
    model = LakeDrainageClassifier(
        use_imgseq=True, use_areaseq=True, use_cloudyseq=False,
        num_classes=5, seq_len=seq_len,
        use_nir="B08" in bands, use_swir16="B11" in bands,
        frontcnn_out_hw=None,          # no upsample: CLSTM sees 32x32
        gradient_checkpointing=grad_ckpt,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(epochs):
        for img, area, cloudy, label, _lid, boa in loader:
            img = img.to(device, non_blocking=True)
            area = area.to(device, non_blocking=True)
            cloudy = cloudy.to(device, non_blocking=True)
            boa = boa.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True).clamp_min(0)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                x = normalize_batch(img, boa)
                logits = model(x, area, cloudy)
                loss = crit(logits, label)
            if do_backward:
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_step = (time.perf_counter() - t0) / epochs

    peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    sample_mb = seq_len * n_ch * 512 * 512 * 2 / 1024**2   # uint16 handoff
    return dict(
        batch_size=batch_size, workers=workers, n_lakes=len(ds), n_channels=n_ch,
        batches=nb // epochs, loader_s=t_load, step_s=t_step,
        compute_s=max(t_step - t_load, 0.0),
        samples_per_s=len(ds) / t_step, peak_vram_gb=peak,
        queue_gb=workers * 2 * batch_size * sample_mb / 1024,
        sample_mb=sample_mb,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", required=True)
    p.add_argument("--bands", nargs="+", default=["B04", "B03", "B02"])
    p.add_argument("--mask", default=None,
                   choices=[None, "lake_boundary", "water_mask_ndwi"])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[8, 32, 64])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seq_len", type=int, default=153)
    p.add_argument("--host_mem_budget_gb", type=float, default=80.0)
    p.add_argument("--max_workers", type=int, default=12)
    p.add_argument("--no_backward", action="store_true")
    p.add_argument("--no_grad_ckpt", action="store_true")
    p.add_argument("--out", default=None, help="write results as JSON")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("CACHED PIPELINE BENCHMARK")
    print("=" * 78)
    print(f"device      : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    print(f"cache       : {a.cache_root}")
    print(f"bands       : {a.bands}   mask: {a.mask}")
    print(f"grad ckpt   : {not a.no_grad_ckpt}")
    print(f"epochs/pt   : {a.epochs}\n")

    n_ch = len(a.bands) + (1 if a.mask else 0)
    sample_mb = a.seq_len * n_ch * 512 * 512 * 2 / 1024**2

    rows = []
    for bs in a.batch_sizes:
        workers, _, q = plan_loader_workers(
            batch_size=bs, sample_mb=sample_mb,
            host_mem_budget_gb=a.host_mem_budget_gb, max_workers=a.max_workers)
        print(f"--- bs={bs}  workers={workers}  projected queue {q:.0f} GB ---")
        try:
            r = bench(a.cache_root, a.bands, a.mask, bs, workers, a.epochs,
                      a.seq_len, device, not a.no_backward, not a.no_grad_ckpt)
        except torch.cuda.OutOfMemoryError:
            print("    CUDA OOM -- skipping\n")
            torch.cuda.empty_cache()
            continue
        rows.append(r)
        print(f"    loader only {human(r['loader_s'])}   full step {human(r['step_s'])}"
              f"   compute {human(r['compute_s'])}")
        print(f"    {r['samples_per_s']:.1f} samples/s   peak VRAM {r['peak_vram_gb']:.1f} GB\n")

    if not rows:
        sys.exit("No configuration completed.")

    print("=" * 78)
    print(f"{'bs':>4} {'wk':>3} {'loader':>9} {'step':>9} {'compute':>9} "
          f"{'smp/s':>7} {'VRAM':>7} {'bound by':>10}")
    print("-" * 78)
    for r in rows:
        bound = "dataloader" if r["loader_s"] > r["compute_s"] else "GPU"
        print(f"{r['batch_size']:>4} {r['workers']:>3} {r['loader_s']:>8.1f}s "
              f"{r['step_s']:>8.1f}s {r['compute_s']:>8.1f}s "
              f"{r['samples_per_s']:>7.1f} {r['peak_vram_gb']:>6.1f}G {bound:>10}")

    best = min(rows, key=lambda r: r["step_s"] / r["n_lakes"])
    per_lake = best["step_s"] / best["n_lakes"]
    print("\nExtrapolated epoch time (linear in lakes, same node):")
    for n in (1175, 1679, 5000):
        print(f"  N={n:5d}  {human(per_lake * n)}")
    print(f"\nBaseline for comparison: ~14.5 min/epoch at N=1175 (old pipeline).")
    print(f"GO/NO-GO: cache is worth it if N=1175 extrapolates well under that.")

    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {a.out}")


if __name__ == "__main__":
    main()
