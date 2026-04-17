#!/bin/bash
#SBATCH --job-name=lv_essd_lcurve_io
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
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu
#SBATCH --array=0-4

# =============================================================================
# ESSD LEARNING CURVE — IMAGERY-ONLY ABLATION
# =============================================================================
#
# Ablation version of run_training_essd_learning_curve.sh: imagery-only
# input (no water_area, cloudy_seq_rgb, or lake polygon mask). Reads from
# raw sat-tile-stack output. Same nested stratified train subsets at
# N in {200, 400, 600, 800, 1000}, same val/test sets.
#
# USAGE:
#   sbatch run_training_essd_learning_curve_imageryonly.sh
# =============================================================================

set -euo pipefail

N_VALUES=(200 400 600 800 1000)
N_TRAIN="${N_VALUES[$SLURM_ARRAY_TASK_ID]}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/essd/lcurve_imageryonly"

STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"

SPLITS_DIR="$REPO_DIR/splits/essd_CW"
TRAIN_IDS="$SPLITS_DIR/train_ids.json"
VAL_IDS="$SPLITS_DIR/val_ids.json"
TEST_IDS="$SPLITS_DIR/test_ids.json"

SAVE_PATH="$MODELS_DIR/lakevision_essd_lcurve_imageryonly_N${N_TRAIN}.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

for f in "$LABELS_2018" "$LABELS_2019" "$TRAIN_IDS" "$VAL_IDS" "$TEST_IDS"; do
    [ -f "$f" ] || { echo "ERROR: missing file $f"; exit 1; }
done
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing stacks directory $d"; exit 1; }
done

NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD lcurve (imagery-only) — N_train=$N_TRAIN"
echo "=============================================="
echo "Array task:  $SLURM_ARRAY_TASK_ID / ${#N_VALUES[@]}"
echo "Stacks:      $STACKS_ROOT/CW_{2018,2019}"
echo "Local SSD:   $NC_DIR"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

echo ""
echo "Copying raw stacks to node-local SSD..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$STACKS_ROOT/CW_2018/" "$NC_DIR/"
rsync -a "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
COPY_END=$(date +%s)
NC_COUNT=$(ls "$NC_DIR/"*.nc 2>/dev/null | wc -l)
echo "  Copied $NC_COUNT files in $((COPY_END - COPY_START))s"
echo "=============================================="

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 py-scikit-learn/1.5.1_py312
pip install --user xarray netcdf4

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="lake-vision"
export WANDB_RUN_GROUP="essd_lcurve_imageryonly"

cd "$SHERLOCK_DIR"

echo ""
echo "Start time: $(date)"

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" "$LABELS_2019" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$TRAIN_IDS" \
    --val_ids_file "$VAL_IDS" \
    --test_ids_file "$TEST_IDS" \
    --max_train_lakes "$N_TRAIN" \
    --no_areaseq --no_cloudyseq --no_mask \
    --wandb_name "essd_lcurve_imageryonly_N${N_TRAIN}" \
    --save_path "$SAVE_PATH"

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
[ $EXIT_CODE -eq 0 ] && [ -f "$SAVE_PATH" ] && ls -lh "$SAVE_PATH"
echo "=============================================="

exit $EXIT_CODE
