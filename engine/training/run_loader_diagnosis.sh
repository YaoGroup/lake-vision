#!/bin/bash
#SBATCH --job-name=lv_loader_diag
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=02:00:00
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
# WHY IS THE LOADER STILL THE BOTTLENECK?  (+ does the VRAM fix work?)
# =============================================================================
#
# Job 39234460 answered the go/no-go gate RED, and killed the simplest
# explanation along the way:
#
#     16 cores / 12 workers -> loader 32.94s
#     32 cores / 24 workers -> loader 35.55s      (SLOWER)
#
# Doubling the cores did nothing, so per-worker parallel work is not the limit.
# Aggregate throughput was ~1.3 GB/s with the cache resident in node-local page
# cache, which is far below what this hardware should manage.
#
# Four hypotheses survive, and they imply different fixes -- three of them free,
# one of them a 2.3-hour cache rebuild:
#
#   H1 worker spawn      persistent_workers=False respawns 24 workers EVERY
#                        epoch. Free to fix.
#   H2 pin_memory        single pinning thread in the parent, 1.84 GB/batch. Free.
#   H3 IPC / shared mem  230 MB per sample across a process boundary. Free-ish.
#   H4 memory bandwidth  the 4+ full-sample copies in __getitem__. Needs the
#                        cache re-layout ([T,C,H,W] in one array) = REBUILD.
#
# The point of this job is to avoid paying for the rebuild if H1-H3 explain it.
# Runs in ~30 min, no cache rebuild, reuses whatever this job builds.
#
# STEP 3 is unrelated but free to piggyback: it re-measures peak VRAM now that
# chunking is paired with PER-CHUNK gradient checkpointing. Job 39234460 showed
# 28.0 GB at bs=8 with grad-checkpointing already on, which confirmed that
# checkpointing a single big segment saves nothing. Expect ~8.5 GB at bs=8 and
# bs=32 to fit at all -- neither has been seen on hardware yet.
#
# USAGE
#   sbatch engine/training/run_loader_diagnosis.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
RESULTS_DIR="$SHERLOCK_DIR/benchmarks"

CACHE_DIR="$L_SCRATCH/cache"
N_PER_YEAR=60          # 120 lakes -- plenty for throughput, ~9 min to build

mkdir -p "$SHERLOCK_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "LOADER DIAGNOSIS"
echo "Node:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"
echo "=============================================="
echo "CPUs: $(nproc)    RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 \
   py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 \
   py-scikit-learn/1.5.1_py312

pip install --user --quiet --only-binary=:all: \
    "blosc2>=3.3,<4.8" "netCDF4<1.7.3" xarray

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"
source "$REPO_DIR/engine/sherlock_preflight.sh"
lv_preflight "$REPO_DIR"

# --- 1. small cache -----------------------------------------------------------
echo ""
echo "=============================================="
echo "STEP 1  build a $((N_PER_YEAR * 2))-lake cache"
echo "=============================================="
python3 -u engine/preprocessing/build_cache.py \
    --stacks_root "$STACKS_ROOT" \
    --out_root    "$CACHE_DIR" \
    --years CW_2018 CW_2019 \
    --bands B04 B03 B02 \
    --masks lake_boundary \
    --limit "$N_PER_YEAR"
echo "cache: $(du -sh "$CACHE_DIR" | cut -f1)"

# Make sure the cache is warm in page cache before timing anything, so we are
# measuring the pipeline and not first-touch disk reads.
echo "warming page cache..."
cat "$CACHE_DIR"/B0*/*.b2nd > /dev/null 2>&1 || true

# --- 2. the diagnosis ---------------------------------------------------------
echo ""
echo "=============================================="
echo "STEP 2  loader diagnosis"
echo "=============================================="
python3 -u engine/benchmarks/diagnose_loader.py \
    --cache_root "$CACHE_DIR" \
    --bands B04 B03 B02 \
    --batch_size 8 \
    --n_lakes 96 \
    --n_raw 16 \
    --workers 0 4 12 24 \
    --out "$RESULTS_DIR/loader_diag_${SLURM_JOB_ID}.json"

# --- 3. does the VRAM fix actually work? --------------------------------------
# Chunking + PER-CHUNK checkpointing. bs=32 has never fitted on hardware.
echo ""
echo "=============================================="
echo "STEP 3  peak VRAM with chunk + per-chunk checkpointing"
echo "=============================================="
python3 -u engine/benchmarks/bench_pipeline.py \
    --cache_root "$CACHE_DIR" \
    --bands B04 B03 B02 \
    --batch_sizes 8 32 \
    --epochs 2 \
    --host_mem_budget_gb 100 \
    --max_workers 12 \
    --prefetch_factor 1 \
    --frontcnn_chunk_size 153 \
    --out "$RESULTS_DIR/vram_check_${SLURM_JOB_ID}.json"

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "  diagnosis: $RESULTS_DIR/loader_diag_${SLURM_JOB_ID}.json"
echo "  vram:      $RESULTS_DIR/vram_check_${SLURM_JOB_ID}.json"
echo "=============================================="
