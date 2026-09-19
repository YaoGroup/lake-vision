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
#SBATCH --mem=320GB
#SBATCH -C "GPU_SKU:A100_SXM4&GPU_MEM:80GB"
#SBATCH --array=0-7
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD REVISION -- a 2x2x2 on the 2019 season (branch essd/revision01)
#
#   task 0  y2019_5c_pw_aug       5-class, p_water in,  augmentation on
#   task 1  y2019_5c_pw_noaug     5-class, p_water in,  augmentation off
#   task 2  y2019_5c_nopw_aug     5-class, p_water out, augmentation on
#   task 3  y2019_5c_nopw_noaug   5-class, p_water out, augmentation off
#   task 4  y2019_4c_pw_aug       4-class (HF+MD merged), p_water in,  aug on
#   task 5  y2019_4c_pw_noaug     4-class, p_water in,  aug off
#   task 6  y2019_4c_nopw_aug     4-class, p_water out, aug on
#   task 7  y2019_4c_nopw_noaug   4-class, p_water out, aug off
#
# Three axes, fully crossed, so each main effect is estimated from four
# independent pairs:
#   class scheme  5-class is the deposit's schema and stays the baseline; the
#                 4-class run merges HF into MD's partner (--merge_classes HF MD
#                 -> 'HFMD'). Both drain water into the ice, so they share their
#                 ice-dynamics implication, and a human can separate them
#                 afterwards. The model cannot: 29-43% of MD lakes are called HF
#                 across every run, while annotators assign them p_MD = 0.92
#                 (91% of the 521 HF/MD lakes are unanimous within the pair).
#                 Post-hoc merging of existing predictions gains +0.07..+0.13
#                 macro-F1; retraining lets the model spend that capacity.
#   p_water       in/out answers reviewer request R3-G1.
#   augmentation  on/off is the data question (see below).
#
# WHY THIS CONFIGURATION (the R2 + R3 pairing), decided 2026-09-19:
# the published baseline cannot fit its own training data. Final-epoch train
# macro-F1 across the nineteen-run matrix:
#
#     R0 (published architecture)  train 0.665   val 0.536
#     R8  soft labels              train 0.639   val 0.517
#     R3  input hygiene            train 0.744   val 0.592
#     R2  optimisation fixes       train 0.978   val 0.604
#     R10 R2 + R3 + everything     train 0.985   val 0.579
#
# R0 is UNDERFITTING, not overfitting. R2 -- GroupNorm, forget-gate bias 1.0,
# warmup-cosine -- is the only configuration that fits, and it does so on the
# published preprocessing (no standardisation), so the gain is optimisation
# alone. R10 shows the fit survives being stacked with R3's hygiene.
#
# This also invalidates the April learning curve as evidence about data volume:
# N=200..1000 saturating at 0.567 was measured on the config that never fits its
# training set, so it recorded an optimisation ceiling, not a data ceiling. With
# a model that provably fits, the train/val gap is a real generalization gap and
# augmentation on/off tests whether added samples close it -- augmentation adds
# quantity without adding diversity, so if it does not close the gap, DIVERSITY
# is the limit and the dataset needs more seasons and basins, not more lakes.
# That is a dataset-descriptor argument, which is why it belongs in ESSD.
#
# TERMINAL-BENCH: all 40 bench lakes are CW2018, so they are absent from this
# split by construction (verified: 0 of 40 appear in train, val or test). No
# hold-out is needed and none is done. To score TBS-40 later, run the saved
# checkpoint over those 40 CW2018 lakes as a separate cross-year inference pass
# and feed the table to engine/eval/score_tbs40.py.
#
# TRAINED ON THE DEPOSIT, not the internal composites. The stacks_v2 deposit is
# what a reader gets from DOI 10.25740/sf350xp4038, so a baseline trained on it
# is reproducible from the published artefact; the composites are not in the
# deposit at all. The published inputs are all present under deposit names:
# bands 0-2 = B04/B03/B02, `lake_boundary` = the static Dunmire polygon that was
# the composites' `mask` channel, and `p_water` = the composites' `water_area`.
#
# THE PROTOCOL IS 2019-ONLY (600/200/200 of the 1,000 CW2019 lakes). This
# baseline is a usability demonstration -- it shows the deposited dataset
# ingests cleanly and supports the task it was built for -- not a modelling
# contribution. A single-season within-year split is the simplest thing that
# demonstrates it, and scoping it that way is why the section can be short,
# which is what R1-M2 asked for. Whether a model trained on one melt season
# transfers to another is a research question for the JSTARS follow-on.
#
# R3-G1's conditional follow-up -- if p_water contributes, test the NDWI water
# mask as an input in its place -- is DEFERRED, and the response letter has to
# say so with the reason. They are not the same quantity. The NDWI formula is
# shared (Dunmire et al. 2025, Eq. 1: (Blue - Red)/(Blue + Red)), and that is
# the only thing that carries over:
#
#                Dunmire p_water              deposit water_mask_ndwi
#   threshold    NDWI > 0.18                  NDWI > 0.3
#   radiometry   top-of-atmosphere            bottom-of-atmosphere (L2A)
#   footprint    inside the lake outline      the whole 512x512 tile
#   cloud mask   Moussavi SWIR/Cirrus         Sen2Cor SCL classes 3/8/9/10
#
#   Dunmire et al. (2025), Earth and Space Science, 10.1029/2024EA003793, p. 3.
# A like-for-like swap needs threshold, radiometry and footprint reconciled
# first; substituting one for the other as-is would produce a misleading
# ablation. This belongs to JSTARS, where the dynamic mask is already a variable.
#
# Changes from the published runs, all deliberate and all disclosable:
#   - R2's three optimisation changes: GroupNorm after each FrontCNN conv,
#     ConvLSTM forget-gate bias 1.0, ten epochs of warmup then cosine decay to
#     1e-6 (published: no normalisation, bias 0.0, flat 1e-4 for 400 epochs).
#   - R3's input hygiene: band standardisation on the TRAIN split, train-mean
#     NaN fill, a per-pixel validity channel (published: raw 0-1, zero fill,
#     no validity channel).
#   - random D4 augmentation in two of the four cells (published: none).
#   - no frontcnn_out_hw, so the CLSTM sees the conv stack's natural 32x32
#     instead of the adaptive-max-pool upsample to 64x64.
#   - the deposit rather than the composites (above).
#
# APPENDIX B of the manuscript documents the 64x64 upsample and no
# normalisation, and MUST be rewritten to match items 1 and 4. Do not describe
# this baseline as architecturally unchanged -- it is not.
#
# Otherwise unchanged: 4-layer FrontCNN base 8, CLSTM hidden 32, last-step
# readout, hard labels (--soft_labels is incompatible with --merge_classes), batch 8, Adam 1e-4, weight decay 1e-5, 400 epochs,
# seed 42, bf16 AMP; test scored on the best-val-macro-F1 checkpoint.
# Do NOT substitute AdamW at 1e-2: that decay is the main reason R14 reaches
# only train 0.794 despite containing all of R2's stack.
#
#   Band statistics first (deposit, 2019-only train split), then this:
#     sbatch engine/training/run_band_stats_essd.sh
#     sbatch --dependency=afterok:<BAND_STATS_JOBID> engine/training/run_essd_revision.sh
#
#   R2 peaked at 54 GB on 3 channels and R10 at 69 GB on 8, so 5 channels needs
#   the 80 GB pool. ~12 min/epoch over 800 lakes puts 400 epochs near the 96 h
#   wall; the best-F1 checkpoint is written continuously, so hitting the wall
#   costs only the test table. Recover it without retraining:
#     sbatch --array=<task> --time=03:00:00 --export=ALL,EVAL_ONLY=1 engine/training/run_essd_revision.sh
# =============================================================================
set -euo pipefail

RUNS=(y2019_5c_pw_aug  y2019_5c_pw_noaug  y2019_5c_nopw_aug  y2019_5c_nopw_noaug
      y2019_4c_pw_aug  y2019_4c_pw_noaug  y2019_4c_nopw_aug  y2019_4c_nopw_noaug)
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
EPOCHS="${EPOCHS:-400}"
NUM_WORKERS="${NUM_WORKERS:-12}"
MEM_BUDGET="${MEM_BUDGET:-208}"
EVAL_ONLY="${EVAL_ONLY:-0}"
SEED="${SEED:-42}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
SPLIT="essd_CW_2019only"
STATS="$SHERLOCK_DIR/band_stats/band_stats_essd_2019only_deposit.json"

# R2: optimisation.  R3: input hygiene.  Both on the published architecture.
R2_FLAGS="--frontcnn_norm group --clstm_forget_bias 1.0 --lr_schedule warmup_cosine --warmup_epochs 10"
R3_FLAGS="--band_stats $STATS --fill mean --validity_channel"

case "$RUN" in
  *_nopw_*)  AREA="--no_areaseq" ;;
  *)         AREA="" ;;
esac
case "$RUN" in
  *_noaug)   AUG="" ;;
  *)         AUG="--augment --augment_mode random" ;;
esac
case "$RUN" in
  *_4c_*)    CLS="--merge_classes HF MD" ;;   # num_classes becomes 4 automatically
  *)         CLS="" ;;
esac

SPLITS_DIR="$REPO_DIR/splits/$SPLIT"
MODELS_DIR="$SHERLOCK_DIR/models/essd_revision"
PRED_DIR="$SHERLOCK_DIR/inference_essd/essd_revision"
SAVE_PATH="$MODELS_DIR/lakevision_essd_rev_${RUN}.pth"
PRED_CSV="$PRED_DIR/${RUN}_test_predictions_bestf1.csv"
mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR" "$PRED_DIR"

for f in "$LABELS_ROOT/labels_CW_2019.csv" \
         "$SPLITS_DIR/train_ids.json" "$SPLITS_DIR/val_ids.json" "$SPLITS_DIR/test_ids.json" "$STATS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -d "$STACKS_ROOT/CW_2019" ] || { echo "ERROR: missing deposit directory $STACKS_ROOT/CW_2019"; exit 1; }
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
echo "Imagery:     $STACKS_ROOT/CW_2019  (stacks_v2 deposit, as published to the SDR)"
echo "Band stats:  $STATS"
echo "Classes:     ${CLS:-5-class}   p_water: ${AREA:-in}   Augment: ${AUG:-off}   Seed: $SEED   Eval-only: $EVAL_ONLY"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

NC_DIR="$L_SCRATCH/nc_data"
echo "Copying CW_2019 deposit stacks to node-local SSD ($(date))..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
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
    --labels_csv "$LABELS_ROOT/labels_CW_2019.csv" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$SPLITS_DIR/train_ids.json" \
    --val_ids_file "$SPLITS_DIR/val_ids.json" \
    --test_ids_file "$SPLITS_DIR/test_ids.json" \
    --no_mask --mask_source static \
    $R2_FLAGS $R3_FLAGS \
    --test_checkpoint f1 --seed "$SEED" \
    --host_mem_budget_gb "$MEM_BUDGET" --num_workers "$NUM_WORKERS" \
    --wandb_name "essd_rev_${RUN}" \
    --save_path "$SAVE_PATH" \
    --test_predictions_csv "$PRED_CSV" \
    $CLS $AREA $AUG $EXTRA && EXIT_CODE=0 || EXIT_CODE=$?

DUR=$(( $(date +%s) - START_TIME ))
echo "=============================================="
echo "End time: $(date)   Duration: $((DUR / 3600))h $(( (DUR % 3600) / 60 ))m   Exit code: $EXIT_CODE"
[ -f "${SAVE_PATH%.pth}_bestf1.pth" ] && ls -lh "${SAVE_PATH%.pth}_bestf1.pth"
[ -f "$PRED_CSV" ] && echo "$(($(wc -l < "$PRED_CSV") - 1)) test predictions in $PRED_CSV"
echo "=============================================="
exit $EXIT_CODE
