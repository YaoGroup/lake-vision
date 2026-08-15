#!/bin/bash
#SBATCH --job-name=lv_cache_smoke
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# CACHED PIPELINE — end-to-end validation on the REAL training path
# =============================================================================
#
# Supersedes engine/benchmarks/run_prototype.sh, which exercised a benchmark
# harness that merely resembled training. This runs run_training.py itself with
# --use_cache, so what gets measured is what will actually train.
#
# MEASURED so far (job 38671174, sh03-18n11, died in pip before step 1):
#   L_SCRATCH  3.0 TB      node RAM 503 GB      A100-SXM4-80GB      16 cores
#   3.0 TB is 3x the 1 TB we assumed, and it removes capacity as a constraint:
#   uncompressed uint16 is 240.6 MB/lake for RGB (153 x 512 x 512 x 2 x 3) plus
#   40.1 MB for a per-timestep mask, so even 6 bands + mask at N=5000 is 2.6 TB.
#   Compression now buys page-cache residency, not disk headroom.
#
# STEPS
#   0. print L_SCRATCH capacity + node specs (the number that gates the
#      full-scale plan; $L_SCRATCH only exists inside an allocation)
#   1. build a 200-lake blosc2 cache on node-local SSD
#   2. benchmark dataloader-vs-compute at bs=8/32/64
#   3. run real training for a few epochs at bs=8
#
# GO/NO-GO
#   Extrapolated epoch time at N=1175 must land far below the ~14.5 min/epoch
#   the old NetCDF pipeline took. If it does not, stop and re-plan before
#   committing a multi-terabyte cache build.
#
# RESOURCES — 32 cores + 256 GB. Job 39080585 measured loader 32.9s vs compute
# 14.9s at bs=8 with 12 workers on 16 cores (dataloader-bound 2.2:1), and the
# node had 1007 GB RAM against our 128 GB request. The bottleneck is CPU-side
# memory traffic, so cores are the resource that matters; doubling them should
# put the loader at ~crossover with compute. prefetch_factor=1 because the
# queue-size model undercounts real worker RSS by ~2x (collated batch, pinned
# copy, IPC duplication) — that undercount is what OOM-killed bs=32 at 128 GB.
#
# Step 3 runs bs=8, not 32: without the chunked FrontCNN, bs=32 needs ~97 GB of
# VRAM (FrontCNN pushes all B*T=4896 images through one conv call), which no
# A100 has. bs=8 measured 28.0 GB. It is also apples-to-apples with the
# ~14.5 min/epoch legacy baseline, which ran bs=8. Revisit after chunking lands.
#
# USAGE
#   sbatch engine/training/run_training_cached_smoke.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
RESULTS_DIR="$SHERLOCK_DIR/benchmarks"
MODELS_DIR="$SHERLOCK_DIR/models/cache_smoke"

CACHE_DIR="$L_SCRATCH/cache"
N_PER_YEAR=100        # 200 lakes total

mkdir -p "$SHERLOCK_DIR/logs" "$RESULTS_DIR" "$MODELS_DIR"

echo "=============================================="
echo "CACHED PIPELINE SMOKE TEST"
echo "Node:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"
echo "=============================================="

echo ""
echo "--- STEP 0: L_SCRATCH capacity (gates the 5-band plan at N=5000) ---"
df -h "$L_SCRATCH"
echo ""
echo "--- node resources ---"
echo "CPUs: $(nproc)    RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 \
   py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 \
   py-scikit-learn/1.5.1_py312

# Version pins are load-bearing, not tidiness. Sherlock's compute nodes are
# glibc 2.17 (binutils 2.27), so only manylinux2014 / manylinux_2_17 wheels
# install. Newer releases have moved past that:
#   blosc2  >=4.8 ships manylinux_2_28 only; 4.11 has no cp312 wheel at all
#   netCDF4 >=1.7.3 ships no cp312 x86_64 linux wheel
# Without an upper bound pip falls back to the sdist, which pulls numpy>=2.1 as
# a build dep, which is also sdist-only here, which needs GCC >= 10.3 -- but
# `ml py-pytorch` reloads gcc 12.4.0 => 10.1.0. That is what killed job
# 38671174 before it ran a single line of our code.
# blosc2 4.7.0 is the newest manylinux_2_17 build; it needs numpy>=1.26, which
# py-numpy/1.26.3_py312 satisfies, so numpy is NOT upgraded out from under
# torch 2.2.1 (which cannot run on numpy 2.x).
# --only-binary=:all: makes a missing wheel fail loudly in seconds instead of
# silently starting a doomed source build.
pip install --user --quiet --only-binary=:all: \
    "blosc2>=3.3,<4.8" "netCDF4<1.7.3" xarray

# Preflight: exercise the exact blosc2 API build_cache.py uses. A version that
# imports but lacks CParams (any blosc2 2.x) would otherwise fail 40 minutes
# into the cache build.
python3 - <<'PREFLIGHT'
import tempfile, os
import numpy as np, blosc2, netCDF4, torch
print(f"blosc2 {blosc2.__version__} | netCDF4 {netCDF4.__version__} | "
      f"numpy {np.__version__} | torch {torch.__version__}")
a = (np.arange(4 * 8 * 8, dtype=np.uint16).reshape(4, 8, 8) % 65535)
p = os.path.join(tempfile.mkdtemp(), "preflight.b2nd")
blosc2.asarray(a, urlpath=p, mode="w", chunks=(2, 8, 8),
               cparams=blosc2.CParams(codec=blosc2.Codec.LZ4,
                                      filters=[blosc2.Filter.BITSHUFFLE],
                                      clevel=5))
assert (blosc2.open(p)[:] == a).all(), "blosc2 roundtrip mismatch"
blosc2.set_nthreads(1)
print("preflight OK: blosc2 write/read/set_nthreads round-trips")
PREFLIGHT

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
cd "$REPO_DIR"

# Records the commit in this log and refuses to run from a dirty checkout.
# See engine/sherlock_preflight.sh for the reasoning and the escape hatch.
source "$REPO_DIR/engine/sherlock_preflight.sh"
lv_preflight "$REPO_DIR"

# --- 1. build cache ----------------------------------------------------------
echo ""
echo "=============================================="
echo "STEP 1  build cache ($((N_PER_YEAR * 2)) lakes)"
echo "=============================================="
T0=$(date +%s)
python3 -u engine/preprocessing/build_cache.py \
    --stacks_root "$STACKS_ROOT" \
    --out_root    "$CACHE_DIR" \
    --years CW_2018 CW_2019 \
    --bands B04 B03 B02 \
    --masks lake_boundary water_mask_ndwi \
    --limit "$N_PER_YEAR"
echo "cache build: $(( $(date +%s) - T0 ))s, $(du -sh "$CACHE_DIR" | cut -f1) on disk"

# --- 2. dataloader vs compute -----------------------------------------------
echo ""
echo "=============================================="
echo "STEP 2  benchmark: dataloader vs compute"
echo "=============================================="
python3 -u engine/benchmarks/bench_pipeline.py \
    --cache_root "$CACHE_DIR" \
    --bands B04 B03 B02 \
    --batch_sizes 8 32 64 \
    --epochs 3 \
    --host_mem_budget_gb 100 \
    --max_workers 24 \
    --prefetch_factor 1 \
    --out "$RESULTS_DIR/cache_bench_${SLURM_JOB_ID}.json"

# --- 3. the real training path ----------------------------------------------
echo ""
echo "=============================================="
echo "STEP 3  run_training.py --use_cache (5 epochs)"
echo "=============================================="
T0=$(date +%s)
python3 -u engine/training/run_training.py \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
    --use_cache --cache_root "$CACHE_DIR" \
    --cache_bands B04 B03 B02 \
    --scalar_var p_water \
    --batch_size 8 \
    --epochs 5 \
    --gradient_checkpointing \
    --host_mem_budget_gb 100 \
    --num_workers 24 \
    --prefetch_factor 1 \
    --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
    --wandb_name "cache_smoke_${SLURM_JOB_ID}" \
    --save_path "$MODELS_DIR/cache_smoke.pth"
echo "training (5 epochs): $(( $(date +%s) - T0 ))s"

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "Bench JSON: $RESULTS_DIR/cache_bench_${SLURM_JOB_ID}.json"
echo "=============================================="
