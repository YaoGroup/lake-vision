#!/bin/bash
#SBATCH --job-name=lv_cv_dryrun
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=04:00:00
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
# CV DRY RUN — one config, every grid flag set, bs=32
# =============================================================================
#
# PURPOSE. This is a *plumbing* test, not a science run. It exists to prove the
# repo is ready to hand to a colleague for the CV grid: every axis flag is
# accepted, the model builds, bs=32 fits in VRAM, an epoch completes, and the
# checkpoint carries enough provenance to identify the run afterwards.
#
# It deliberately sets flags that are NOT the ESSD baseline, so a silently
# ignored flag shows up as "the model summary looks like the baseline" rather
# than passing unnoticed. Do not read the F1 numbers as meaningful: 5 epochs on
# 300 lakes is noise.
#
# WHY bs=32 IS THE POINT. FrontCNN flattens [B,T,C,H,W] -> [B*T,C,H,W], so at
# bs=32 it pushes 4,896 images through the first conv (job 39080585 measured
# 28.0 GB at bs=8 and host-OOMed at bs=32). Two flags together fix that, and
# BOTH are needed:
#
#   --frontcnn_chunk_size 153   slices the B*T axis
#   --gradient_checkpointing    checkpoints EACH chunk separately
#
# Chunking alone saves NO training memory — autograd stores every chunk's
# activations, so the total is identical to the unchunked pass (measured:
# 29.09 MB either way on a B*T=64 probe). Checkpointing the whole FrontCNN as
# one segment does not help either: backward rebuilds the entire segment at
# once. Only per-chunk checkpointing makes peak backward activation one chunk's
# worth regardless of batch size. Measured: 28.0 -> 9.9 GB at bs=8 (job 39254647).
#
# That alone was NOT enough. Job 39254653 still OOMed at bs=32 (37.88 GB of a
# 39.38 GB card), because flattening FrontCNN merely exposed the next term: the
# float32 input itself is 20.5 GB at bs=32/T=153/4ch and lives through forward
# AND backward. Hence the third flag:
#
#   --normalize_in_chunks       convert int16 -> float32 one chunk at a time,
#                               inside the checkpointed region, so only the
#                               int16 input persists (saves ~10.3 GB at bs=32)
#
# Projected with all three: ~27 GB at bs=32. Verified bit-identical to
# normalizing up front, so this is a memory change only, not a model change.
#
# WHAT TO CHECK, IN ORDER
#   1. "--- provenance ---" block names the commit (preflight).
#   2. MODEL SUMMARY shows GroupNorm layers in FrontCNN, and a parameter count
#      ABOVE the 139,925 baseline (pool_type=both + clstm_hidden=64).
#   3. "Optimizer: adamw" — proves the optimizer axis is live, not hardcoded.
#   4. Peak VRAM well under 40 GB at bs=32. This is the chunking gate — and it
#      only holds if BOTH --frontcnn_chunk_size and --gradient_checkpointing are
#      in the command below.
#   5. Five epochs complete; per-epoch wall-clock is the number to size the grid.
#   6. The saved .pth loads as a dict with config/fold/git_sha (see TO VERIFY).
#
# TO VERIFY THE CHECKPOINT AFTERWARDS
#   python3 -c "
#   import torch; c=torch.load('<save_path>', map_location='cpu')
#   print(sorted(c)); print(c['fold'], c['git_sha'], c['class_names'])"
#
# NOT SET HERE, ON PURPOSE
#   --frontcnn_chunk_size is FIXED at 153 and must never become a grid axis:
#   two runs differing only in chunk size diverge into different weights through
#   float non-associativity. It is a memory knob, not a scientific variable.
#
# USAGE
#   sbatch engine/training/run_training_cv_dryrun.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
COMPOSITES_ROOT="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
MODELS_DIR="$SHERLOCK_DIR/models/cv_dryrun"

CACHE_DIR="$L_SCRATCH/cache"
N_PER_YEAR=150        # 300 lakes — enough for a real epoch, ~25 min to cache

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

echo "=============================================="
echo "CV DRY RUN — one config, all grid flags, bs=32"
echo "Node:    $(hostname)"
echo "Job:     ${SLURM_JOB_ID:-none}"
echo "Started: $(date)"
echo "=============================================="
echo "CPUs: $(nproc)    RAM: $(free -g | awk '/^Mem:/{print $2}') GB"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
df -h "$L_SCRATCH" | tail -1

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 \
   py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 \
   py-scikit-learn/1.5.1_py312

# Pins are load-bearing — see run_training_cached_smoke.sh for the full reasoning
# (glibc 2.17 => manylinux2014 wheels only; an unpinned blosc2 falls back to an
# sdist that needs a GCC newer than the pytorch module provides).
pip install --user --quiet --only-binary=:all: \
    "blosc2>=3.3,<4.8" "netCDF4<1.7.3" xarray

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

source "$REPO_DIR/engine/sherlock_preflight.sh"
lv_preflight "$REPO_DIR"

# --- 1. cache, with the cloudy-tile flag transplanted from the composites -----
# The published v2 stacks carry no cloudy_seq_* variable; cloudy-tile ran on the
# composites in April 2026. Both trees hold bit-identical pixels on the same
# daily grid (verified 2026-08-15), and build_cache re-asserts per-lake time
# alignment, so this needs no GPU re-inference.
echo ""
echo "=============================================="
echo "STEP 1  build cache ($((N_PER_YEAR * 2)) lakes) + cloudy_seq_rgb"
echo "=============================================="
T0=$(date +%s)
python3 -u engine/preprocessing/build_cache.py \
    --stacks_root "$STACKS_ROOT" \
    --out_root    "$CACHE_DIR" \
    --years CW_2018 CW_2019 \
    --bands B04 B03 B02 \
    --masks lake_boundary water_mask_ndwi \
    --composites_root "$COMPOSITES_ROOT" \
    --cloudy_seq_var cloudy_seq_rgb \
    --limit "$N_PER_YEAR"
echo "cache build: $(( $(date +%s) - T0 ))s, $(du -sh "$CACHE_DIR" | cut -f1) on disk"

# --- 2. one training run with every CV axis explicitly set --------------------
echo ""
echo "=============================================="
echo "STEP 2  training: bs=32, all CV flags set"
echo "=============================================="
T0=$(date +%s)
python3 -u engine/training/run_training.py \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
    --use_cache --cache_root "$CACHE_DIR" \
    --cache_bands B04 B03 B02 \
    --cache_mask lake_boundary \
    --cache_cloudy_seq_var cloudy_seq_rgb \
    --scalar_var p_water \
    `# --- axis 2: which sequences feed the model ---` \
    --use_imgseq --use_areaseq --use_cloudyseq \
    `# --- axis 1: attention + how the mask enters ---` \
    --attention_type spatial \
    --mask_as_channel \
    `# --- axis 5: LSTM readout and gating ---` \
    --temporal_readout mean \
    --clstm_forget_bias 1.0 \
    --clstm_hidden 64 \
    --pool_type both \
    `# --- axis 4: FrontCNN shape + normalization ---` \
    --frontcnn_base_channels 8 \
    --frontcnn_num_layers 4 \
    --frontcnn_norm group \
    `# --- memory knobs: FIXED, never grid axes. BOTH are required: ---` \
    `# chunking alone saves nothing (autograd keeps every chunk's activations); ---` \
    `# checkpointing makes FrontCNN recompute ONE chunk at a time in backward. ---` \
    --frontcnn_chunk_size 153 \
    --gradient_checkpointing \
    --normalize_in_chunks \
    `# --- axis 6: optimization hyperparameters ---` \
    --optimizer adamw \
    --lr 3e-4 \
    --weight_decay 1e-5 \
    --batch_size 32 \
    --epochs 5 \
    --classhead_dropout 0.3 \
    `# --- throughput: seeds stay fixed, only kernel choice varies ---` \
    --no_deterministic \
    --amp \
    --host_mem_budget_gb 100 \
    --num_workers 24 \
    --prefetch_factor 1 \
    `# --- provenance: these land inside the checkpoint ---` \
    --fold_idx 0 \
    --wandb_group cv_dryrun \
    --wandb_tags dryrun plumbing \
    --wandb_name "cv_dryrun_${SLURM_JOB_ID}" \
    --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
    --save_path "$MODELS_DIR/cv_dryrun_${SLURM_JOB_ID}.pth"
echo "training (5 epochs): $(( $(date +%s) - T0 ))s"

# --- 3. prove the checkpoint is self-describing -------------------------------
echo ""
echo "=============================================="
echo "STEP 3  checkpoint provenance"
echo "=============================================="
python3 - "$MODELS_DIR/cv_dryrun_${SLURM_JOB_ID}.pth" <<'CKPT'
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
cand = p if p.exists() else next(iter(sorted(p.parent.glob(p.stem + "*.pth"))), None)
if cand is None:
    sys.exit(f"no checkpoint found at {p}")
c = torch.load(cand, map_location="cpu")
assert isinstance(c, dict) and "state_dict" in c, "checkpoint is not a provenance dict"
print(f"file        : {cand.name}")
print(f"keys        : {sorted(c)}")
print(f"epoch       : {c['epoch']}   fold: {c['fold']}   seed: {c['seed']}")
print(f"git_sha     : {c['git_sha']}")
print(f"class_names : {c['class_names']}")
for k in ("temporal_readout", "frontcnn_norm", "optimizer", "batch_size",
          "mask_as_channel", "frontcnn_chunk_size", "cache_cloudy_seq_var"):
    print(f"  config[{k}] = {c['config'].get(k)!r}")
print("checkpoint provenance OK")
CKPT

echo ""
echo "=============================================="
echo "Finished: $(date)"
echo "=============================================="
