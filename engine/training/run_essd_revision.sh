#!/bin/bash
#SBATCH --job-name=lv_essd_rev
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=96:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --array=0-2
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD REVISION RUNS
#
# Scope is deliberately narrow. The reviewers asked for LESS model content, so
# this is the published architecture plus (a) input standardisation, which is
# preprocessing hygiene rather than a modelling change, (b) the one ablation a
# reviewer explicitly requested -- does the p_water area stream contribute
# (R3-G1) -- and (c) random D4 augmentation. Soft labels, the stride-1 FrontCNN
# and the stacked configuration belong to the JSTARS follow-on
# (docs/eleven_runs/ELEVEN_RUNS.html), not here.
#
# THE PROTOCOL IS 2019-ONLY (600/200/200 of the 1,000 CW2019 lakes). The
# published cross-year and combined protocols leak: a CW2018 lake's nearest
# CW2019 lake is a median 110 m away and 47% are within 100 m, i.e. the same
# basin refilling, so a model trained on one year has seen the other year's
# test sites. Within CW2019 no two lakes are within 500 m of each other (the
# closest test-to-train pair in this split is 600 m, median 3.4 km), so the
# single-year protocol removes that leakage entirely with no spatial blocking
# and no buffered subsetting. Cross-year generalisation, the melt-season shift
# and the cloud-observability diagnosis move to the discussion. Tasks 3-6 still
# reproduce the published protocols if they are wanted for reference.
#
# Everything else matches the published runs exactly, because the revision's
# tables have to stay comparable to the frozen essd-2026-submission tag:
#   - COMPOSITES, not the stacks_v2 deposits the JSTARS runs use. The published
#     baselines rsynced $SHERLOCK_DIR/composites and their statistics, channel
#     layout (RGB + mask) and NaN pattern all differ from the deposits.
#   - frontcnn_out_hw 64,64, i.e. the adaptive-max-pool upsample Appendix B
#     documents. Changing it is a JSTARS experiment, not a revision.
#   - imagery + area streams, no cloudy stream; batch 8, Adam 1e-4, wd 1e-5,
#     400 epochs, seed 42; test scored on the best-val-macro-F1 checkpoint.
#
#   Band statistics first (composites, per protocol), then this:
#     sbatch engine/training/run_band_stats_essd.sh
#     sbatch --dependency=afterok:<BAND_STATS_JOBID> engine/training/run_essd_revision.sh
#   Tasks: 0 baseline, 1 no p_water (R3-G1), 2 augmented.
#
#   Score a saved best-F1 checkpoint without retraining (the combined runs are
#   close to the 96 h wall; if one is killed, this writes its test table):
#     sbatch --array=<task> --time=03:00:00 --export=ALL,EVAL_ONLY=1 engine/training/run_essd_revision.sh
# =============================================================================
set -euo pipefail

RUNS=(y2019_base y2019_noarea y2019_augment crossyear_base crossyear_noarea combined_base combined_noarea)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
EPOCHS="${EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MEM_BUDGET="${MEM_BUDGET:-166}"
EVAL_ONLY="${EVAL_ONLY:-0}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
COMPOSITES="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"

case "$RUN" in
  y2019_*)     SPLIT="essd_CW_2019only";  STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_2019only_composites.json" ;;
  crossyear_*) SPLIT="essd_CW_crossyear"; STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_crossyear_composites.json" ;;
  combined_*)  SPLIT="essd_CW";           STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_combined_composites.json" ;;
esac
case "$RUN" in
  *_noarea)  ABLATION="--no_areaseq" ;;
  *_augment) ABLATION="--augment --augment_mode random" ;;
  *)         ABLATION="" ;;
esac

SPLITS_DIR="$REPO_DIR/splits/$SPLIT"
MODELS_DIR="$SHERLOCK_DIR/models/essd_revision"
PRED_DIR="$SHERLOCK_DIR/inference_essd/essd_revision"
SAVE_PATH="$MODELS_DIR/lakevision_essd_rev_${RUN}.pth"
PRED_CSV="$PRED_DIR/${RUN}_test_predictions_bestf1.csv"
mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR" "$PRED_DIR"

for f in "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
         "$SPLITS_DIR/train_ids.json" "$SPLITS_DIR/val_ids.json" "$SPLITS_DIR/test_ids.json" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
for d in "$COMPOSITES/CW_2018" "$COMPOSITES/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing composites directory $d"; exit 1; }
done
if [ "$EVAL_ONLY" = "1" ]; then
    [ -f "${SAVE_PATH%.pth}_bestf1.pth" ] || { echo "ERROR: EVAL_ONLY needs ${SAVE_PATH%.pth}_bestf1.pth"; exit 1; }
    EXTRA="--epochs 0 --no_wandb"
elif [ -f "$SAVE_PATH" ]; then
    echo "ERROR: $SAVE_PATH exists; refusing to overwrite a finished run."; exit 1
else
    EXTRA="--epochs $EPOCHS"
fi

GIT_SHA="$(cd "$REPO_DIR" && git rev-parse HEAD)"
export LV_GIT_SHA="$GIT_SHA"
echo "=============================================="
echo "ESSD revision: $RUN (array task $SLURM_ARRAY_TASK_ID, job $SLURM_JOB_ID)"
echo "=============================================="
echo "Node:        $(hostname)   GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Commit:      $GIT_SHA"
echo "Split:       $SPLITS_DIR"
echo "Imagery:     $COMPOSITES  (composites, as published)"
echo "Band stats:  $STATS"
echo "Ablation:    ${ABLATION:-none}   Eval-only: $EVAL_ONLY"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

NC_DIR="$L_SCRATCH/nc_data"
echo "Copying composites to node-local SSD ($(date))..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$COMPOSITES/CW_2018/" "$NC_DIR/"
rsync -a "$COMPOSITES/CW_2019/" "$NC_DIR/"
echo "  $(ls "$NC_DIR"/*.nc | wc -l) files, $(du -sh "$NC_DIR" | cut -f1), $(( $(date +%s) - COPY_START ))s"

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 py-scikit-learn/1.5.1_py312
pip install --user xarray netcdf4

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="lake-vision"
export WANDB_RUN_GROUP="essd_revision"
export WANDB_TAGS="essd_revision,$RUN"

cd "$SHERLOCK_DIR"
echo "Start time: $(date)"
START_TIME=$(date +%s)

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_ROOT/labels_CW_2019.csv" "$LABELS_ROOT/labels_CW_2018.csv" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$SPLITS_DIR/train_ids.json" \
    --val_ids_file "$SPLITS_DIR/val_ids.json" \
    --test_ids_file "$SPLITS_DIR/test_ids.json" \
    --frontcnn_out_hw 64,64 \
    --band_stats "$STATS" --fill mean \
    --test_checkpoint f1 \
    --host_mem_budget_gb "$MEM_BUDGET" --num_workers "$NUM_WORKERS" \
    --wandb_name "essd_rev_${RUN}" \
    --save_path "$SAVE_PATH" \
    --test_predictions_csv "$PRED_CSV" \
    $ABLATION $EXTRA && EXIT_CODE=0 || EXIT_CODE=$?

DUR=$(( $(date +%s) - START_TIME ))
echo "=============================================="
echo "End time: $(date)   Duration: $((DUR / 3600))h $(( (DUR % 3600) / 60 ))m   Exit code: $EXIT_CODE"
[ -f "${SAVE_PATH%.pth}_bestf1.pth" ] && ls -lh "${SAVE_PATH%.pth}_bestf1.pth"
[ -f "$PRED_CSV" ] && echo "$(($(wc -l < "$PRED_CSV") - 1)) test predictions in $PRED_CSV"
echo "=============================================="
exit $EXIT_CODE
