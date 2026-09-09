#!/bin/bash
#SBATCH --job-name=lv_eleven
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=72:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# THE ELEVEN RUNS (docs/eleven_runs/ELEVEN_RUNS.html)
#
# One array task per run, cross-year split (train 800 / val 200 CW2019, test
# 679 CW2018), trained on the deposit stacks (stacks_v2). Flags come from
# eleven_runs_matrix.sh; nothing else differs between tasks.
#
# Array index -> run. The eight 40 GB runs are 0-7, the three that need an
# 80 GB A100 (R2 GroupNorm, R9 capacity, R10 the stack) are 8-10, so the two
# GPU classes are two submissions of this one script:
#
#   40 GB cards (defaults in the header):
#     sbatch engine/training/run_eleven_runs.sh
#
#   80 GB A100s. GPU_MEM:80GB ALONE matches the H100 node, which the
#   py-pytorch/2.2.1 module cannot drive (job 22429909); keep both features:
#     sbatch --array=8-10 -C "GPU_SKU:A100_SXM4&GPU_MEM:80GB" --mem=320GB \
#            --export=ALL,MEM_BUDGET=208 engine/training/run_eleven_runs.sh
#
#   Smoke test first (50 lakes per split, 5 epochs, both classes at once):
#     sbatch --array=0-10 --time=04:00:00 -C "GPU_SKU:A100_SXM4&GPU_MEM:80GB" \
#            --mem=320GB --export=ALL,SMOKE=1,MEM_BUDGET=208 engine/training/run_eleven_runs.sh
#
# PREREQUISITES
#   - stacks_v2/CW_2018 (679) and CW_2019 (1000) on Oak
#   - band_stats for R3/R10: sbatch engine/training/run_band_stats_crossyear.sh
#   - the repo clone on Oak checked out at the commit you mean to run
# =============================================================================

set -euo pipefail

ARRAY_RUNS=(R0 R1 R3 R4 R5 R6 R7 R8 R2 R9 R10)
RUN="${ARRAY_RUNS[$SLURM_ARRAY_TASK_ID]}"
SMOKE="${SMOKE:-0}"
MEM_BUDGET="${MEM_BUDGET:-166}"     # ~65% of --mem, the loader queue's share
EPOCHS="${EPOCHS:-400}"             # override per submission, e.g. --export=ALL,EPOCHS=350
NUM_WORKERS="${NUM_WORKERS:-12}"    # raise together with --cpus-per-task

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"
SPLITS_DIR="$REPO_DIR/splits/essd_CW_crossyear"
export BAND_STATS="$SHERLOCK_DIR/band_stats/band_stats_crossyear_train.json"

source "$REPO_DIR/engine/training/eleven_runs_matrix.sh"

if [ "$SMOKE" = "1" ]; then
    TAG="eleven_smoke"; EXTRA="--epochs 5 --max_lakes 50"
else
    TAG="eleven"; EXTRA="--epochs $EPOCHS"
fi
MODELS_DIR="$SHERLOCK_DIR/models/$TAG"
PRED_DIR="$SHERLOCK_DIR/inference_essd/$TAG"
SAVE_PATH="$MODELS_DIR/lakevision_${TAG}_${RUN}.pth"
PRED_CSV="$PRED_DIR/${RUN}_test_predictions_bestf1.csv"
mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR" "$PRED_DIR"

for f in "$LABELS_2018" "$LABELS_2019" "$SPLITS_DIR/train_ids.json" "$SPLITS_DIR/val_ids.json" "$SPLITS_DIR/test_ids.json"; do
    [ -f "$f" ] || { echo "ERROR: missing file $f"; exit 1; }
done
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing stacks directory $d"; exit 1; }
done
case "$RUN" in R3|R10)
    [ -f "$BAND_STATS" ] || { echo "ERROR: $RUN needs $BAND_STATS; run run_band_stats_crossyear.sh first"; exit 1; } ;;
esac
if [ "$SMOKE" != "1" ] && [ -f "$SAVE_PATH" ]; then
    echo "ERROR: $SAVE_PATH exists; refusing to overwrite a finished run. Move it or change TAG."; exit 1
fi

FLAGS="$COMMON_FLAGS $(run_flags "$RUN") $EXTRA"
GIT_SHA="$(cd "$REPO_DIR" && git rev-parse HEAD)"
export LV_GIT_SHA="$GIT_SHA"

echo "=============================================="
echo "Eleven runs: $RUN (array task $SLURM_ARRAY_TASK_ID, job $SLURM_JOB_ID)"
echo "=============================================="
echo "Node:        $(hostname)   GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Commit:      $GIT_SHA"
echo "Flags:       $FLAGS"
echo "Smoke:       $SMOKE"
echo "Model save:  $SAVE_PATH"
echo "Predictions: $PRED_CSV"
echo "=============================================="

NC_DIR="$L_SCRATCH/nc_data"
echo "Copying deposit stacks to node-local SSD ($(date))..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
if [ "$SMOKE" = "1" ]; then
    # Stage only the 150 lakes the capped splits use.
    python3 - "$SPLITS_DIR" "$STACKS_ROOT" "$NC_DIR" <<'PYEOF'
import json, shutil, sys, pathlib
splits, root, out = map(pathlib.Path, sys.argv[1:])
for name, year in [("train_ids", "2019"), ("val_ids", "2019"), ("test_ids", "2018")]:
    for lid in json.load(open(splits / f"{name}.json"))[:50]:
        shutil.copy(root / f"CW_{year}" / f"{lid}.nc", out / f"{lid}.nc")
PYEOF
else
    rsync -a "$STACKS_ROOT/CW_2018/" "$NC_DIR/"
    rsync -a "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
fi
echo "  $(ls "$NC_DIR"/*.nc | wc -l) files, $(du -sh "$NC_DIR" | cut -f1), $(( $(date +%s) - COPY_START ))s"

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 py-scikit-learn/1.5.1_py312
pip install --user xarray netcdf4

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="lake-vision"
export WANDB_RUN_GROUP="$TAG"
export WANDB_TAGS="$TAG,$RUN"

cd "$SHERLOCK_DIR"
echo "Start time: $(date)"
START_TIME=$(date +%s)

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2019" "$LABELS_2018" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$SPLITS_DIR/train_ids.json" \
    --val_ids_file "$SPLITS_DIR/val_ids.json" \
    --test_ids_file "$SPLITS_DIR/test_ids.json" \
    --host_mem_budget_gb "$MEM_BUDGET" --num_workers "$NUM_WORKERS" \
    --wandb_name "${TAG}_${RUN}" \
    --save_path "$SAVE_PATH" \
    --test_predictions_csv "$PRED_CSV" \
    $FLAGS
EXIT_CODE=$?

DUR=$(( $(date +%s) - START_TIME ))
echo "=============================================="
echo "End time: $(date)   Duration: $((DUR / 3600))h $(( (DUR % 3600) / 60 ))m   Exit code: $EXIT_CODE"
[ -f "${SAVE_PATH%.pth}_bestf1.pth" ] && ls -lh "${SAVE_PATH%.pth}_bestf1.pth"
[ -f "$PRED_CSV" ] && echo "$(($(wc -l < "$PRED_CSV") - 1)) test predictions in $PRED_CSV"
echo "=============================================="
exit $EXIT_CODE
