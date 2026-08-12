#!/bin/bash
#SBATCH --job-name=lv_proto
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# 50-LAKE PROTOTYPE — go/no-go gate for the blosc2 cache
# =============================================================================
#
# WHAT THIS ANSWERS
#   Every throughput number in the JSTARS plan is a component estimate. Nobody
#   has measured the assembled pipeline. This job builds a 50-lake cache and
#   benchmarks it at bs=8/32/64, separating dataloader time from compute time.
#
#   Decision rule: extrapolated epoch time at N=1175 should land far below the
#   ~14.5 min/epoch the old pipeline took. If it does not, stop and re-plan
#   before committing a multi-terabyte cache build.
#
# RESOURCE REQUEST — note what changed vs the ESSD scripts
#   Those asked for 1 GPU + 320 GB. The A100 nodes run 128 GB of RAM per GPU
#   (SH3_G4TF64: 64c/512G/4 GPUs; SH3_G8TF64: 128c/1T/8 GPUs), so a 320 GB
#   request for one GPU forces SLURM to strand 2 other A100s — probably a large
#   part of past queue times. 16 cores + 128 GB is exactly one GPU's share and
#   should schedule quickly.
#
#   Also: no GPU_MEM:80GB constraint. With gradient checkpointing and the
#   FrontCNN upsample removed, bs=64 needs ~17 GB, so any 40 GB A100 works.
#
# USAGE
#   sbatch engine/benchmarks/run_prototype.sh
#   # then:  tail -f .../logs/lv_proto_<jobid>.out
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
RESULTS_DIR="$SHERLOCK_DIR/benchmarks"

CACHE_DIR="$L_SCRATCH/cache"
N_LAKES=25          # per year -> 50 total

mkdir -p "$SHERLOCK_DIR/logs" "$RESULTS_DIR"

echo "=============================================="
echo "50-lake prototype: cache build + benchmark"
echo "=============================================="
echo "Node:       $(hostname)"
echo "Stacks:     $STACKS_ROOT"
echo "Cache:      $CACHE_DIR"
echo "Started:    $(date)"
echo "=============================================="

# --- the number that gates the full-scale plan -------------------------------
echo ""
echo "--- L_SCRATCH capacity (decides whether 5-band can stage locally) ---"
df -h "$L_SCRATCH"
echo ""
echo "--- node resources ---"
echo "CPUs: $(nproc)   Mem: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo ""

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 \
   py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 \
   py-scikit-learn/1.5.1_py312

pip install --user --quiet blosc2 netcdf4 xarray
python3 -c "import blosc2; print(f'blosc2 {blosc2.__version__}')"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$REPO_DIR"

# --- 1. build the cache ------------------------------------------------------
echo ""
echo "=============================================="
echo "STEP 1  build cache for $((N_LAKES * 2)) lakes"
echo "=============================================="
BUILD_START=$(date +%s)

python3 -u engine/preprocessing/build_cache.py \
    --stacks_root "$STACKS_ROOT" \
    --out_root    "$CACHE_DIR" \
    --years CW_2018 CW_2019 \
    --bands B04 B03 B02 \
    --masks lake_boundary water_mask_ndwi \
    --limit "$N_LAKES"

echo "Cache build took $(( $(date +%s) - BUILD_START ))s"
echo "Cache on disk: $(du -sh "$CACHE_DIR" | cut -f1)"

# --- 2. benchmark ------------------------------------------------------------
echo ""
echo "=============================================="
echo "STEP 2  benchmark (RGB, no mask)"
echo "=============================================="
python3 -u engine/benchmarks/bench_pipeline.py \
    --cache_root "$CACHE_DIR" \
    --bands B04 B03 B02 \
    --batch_sizes 8 32 64 \
    --epochs 3 \
    --host_mem_budget_gb 80 \
    --max_workers 12 \
    --out "$RESULTS_DIR/proto_rgb_${SLURM_JOB_ID}.json"

echo ""
echo "=============================================="
echo "STEP 3  benchmark (RGB + NDWI mask)"
echo "=============================================="
python3 -u engine/benchmarks/bench_pipeline.py \
    --cache_root "$CACHE_DIR" \
    --bands B04 B03 B02 \
    --mask water_mask_ndwi \
    --batch_sizes 8 32 \
    --epochs 3 \
    --host_mem_budget_gb 80 \
    --max_workers 12 \
    --out "$RESULTS_DIR/proto_ndwi_${SLURM_JOB_ID}.json"

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "Results:  $RESULTS_DIR/proto_*_${SLURM_JOB_ID}.json"
echo "=============================================="
