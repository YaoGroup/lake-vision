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
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD REVISION RUNS  (two runs: with and without p_water)
#
# Scope is deliberately narrow. The reviewers asked for LESS model content, so
# the architecture is the published one and the only additions are input
# hygiene and augmentation. Soft labels, the extra spectral bands, the dynamic
# NDWI water mask, the cloud channel, attention/CBAM and every cross-year
# generalisation result belong to the JSTARS follow-on
# (docs/eleven_runs/ELEVEN_RUNS.html), not here.
#
#   task 0  y2019_base_s42    hygiene + augmentation, p_water in
#   task 1  y2019_noarea_s42  the same, p_water dropped (reviewer request R3-G1)
#   task 2  y2019_base_s43    seed replicate
#   task 3  y2019_noarea_s43  seed replicate
#
# TWO SEEDS PER ARM. R3-G1 asks whether p_water contributes, and a single run
# against a single run cannot answer that: if the arms land within the 0.03-0.05
# tie band the matrix uses, an n=1 difference is not reportable. The replicates
# cost no extra wall-clock (separate array tasks, separate nodes) and turn
# "decide when we see the numbers" into a result. The split files are fixed, so
# --seed varies only weight init, shuffling and the augmentation draw.
#
# TRAINED ON THE DEPOSIT, not the internal composites. The stacks_v2 deposit is
# what a reader gets from DOI 10.25740/sf350xp4038, so a baseline trained on it
# is reproducible from the published artefact; the composites are not in the
# deposit at all. The published inputs are all present under deposit names:
# bands 0-2 = B04/B03/B02, `lake_boundary` = the static Dunmire polygon that was
# the composites' `mask` channel, and `p_water` = the composites' `water_area`.
#
# THE PROTOCOL IS 2019-ONLY (600/200/200 of the 1,000 CW2019 lakes). The
# published cross-year and combined protocols leak: a CW2018 lake's nearest
# CW2019 lake is a median 110 m away and 47% are within 100 m, i.e. the same
# basin refilling, so a model trained on one year has seen the other year's
# test sites. Within CW2019 no two lakes are within 500 m of each other, so the
# single-year protocol removes that leakage entirely with no spatial blocking
# and no buffered subsetting. Cross-year generalisation, the melt-season shift
# and the cloud-observability diagnosis move to the discussion. Tasks 2-5 still
# reproduce the published protocols if they are wanted for reference.
#
# Changes from the published runs, all deliberate and all disclosable:
#   - band standardisation on the TRAIN split + train-mean NaN fill + a
#     per-pixel validity channel (published: raw 0-1, zero fill, no validity).
#     The single biggest evidence-backed gain in the nineteen-run matrix.
#   - random D4 augmentation (published: none).
#   - no frontcnn_out_hw, i.e. the CLSTM sees the conv stack's natural 32x32
#     instead of the adaptive-max-pool upsample to 64x64 that Appendix B
#     documents. 4x cheaper; every run in the matrix used 32x32.
#   - the deposit rather than the composites (above).
# NOTE the third item is an architecture change, not just hygiene: it alters the
# feature-map resolution entering the recurrent layer by 4x. Appendix B of the
# manuscript currently documents the 64x64 upsample and MUST be rewritten to
# match. Do not describe this baseline as architecturally unchanged.
# Otherwise unchanged: 4-layer FrontCNN base 8, CLSTM hidden 32, last-step
# readout, hard labels, batch 8, Adam 1e-4, wd 1e-5, 400 epochs, bf16 AMP; test
# scored on the best-val-macro-F1 checkpoint.
#
#   Band statistics first (deposit, 2019-only train split), then this:
#     sbatch engine/training/run_band_stats_essd.sh
#     sbatch --dependency=afterok:<BAND_STATS_JOBID> engine/training/run_essd_revision.sh
#
#   ~5 input channels over 800 lakes/epoch puts 400 epochs near the 96 h wall.
#   The best-F1 checkpoint is written continuously, so hitting the wall costs
#   only the test table; recover it without retraining:
#     sbatch --array=<task> --time=03:00:00 --export=ALL,EVAL_ONLY=1 engine/training/run_essd_revision.sh
# =============================================================================
set -euo pipefail

RUNS=(y2019_base_s42 y2019_noarea_s42 y2019_base_s43 y2019_noarea_s43
      crossyear_base_s42 crossyear_noarea_s42 combined_base_s42 combined_noarea_s42)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
SEED="${RUN##*_s}"
EPOCHS="${EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MEM_BUDGET="${MEM_BUDGET:-166}"
EVAL_ONLY="${EVAL_ONLY:-0}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"

case "$RUN" in
  y2019_*)     SPLIT="essd_CW_2019only";  STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_2019only_deposit.json" ;;
  crossyear_*) SPLIT="essd_CW_crossyear"; STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_crossyear_deposit.json" ;;
  combined_*)  SPLIT="essd_CW";           STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_combined_deposit.json" ;;
esac
case "$RUN" in
  *_noarea_*)  ABLATION="--no_areaseq" ;;
  *)           ABLATION="" ;;
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
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing deposit directory $d"; exit 1; }
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
echo "Imagery:     $STACKS_ROOT  (stacks_v2 deposit, as published to the SDR)"
echo "Band stats:  $STATS"
echo "Ablation:    ${ABLATION:-none}   Seed: $SEED   Eval-only: $EVAL_ONLY"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

NC_DIR="$L_SCRATCH/nc_data"
echo "Copying deposit stacks to node-local SSD ($(date))..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
# The 2019-only protocol never touches a CW2018 lake; skip half the copy.
case "$RUN" in
  y2019_*) : ;;
  *)       rsync -a "$STACKS_ROOT/CW_2018/" "$NC_DIR/" ;;
esac
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
    --no_mask --mask_source static \
    --band_stats "$STATS" --fill mean --validity_channel \
    --augment --augment_mode random \
    --seed "$SEED" \
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
