#!/bin/bash
#SBATCH --job-name=lv_essd_comb_aug
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=144:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD BASELINE + 8x AUGMENTATION — COMBINED 2018+2019
# =============================================================================
#
# Same baseline architecture as run_training_essd_combined.sh (140k params)
# but with 8x D4 data augmentation (rotations + flips).
# Effectively 1175 x 8 = 9400 training samples per epoch.
# 144h walltime (6 days), bs=8, 400 epochs.
#
# USAGE:
#   sbatch run_training_essd_combined_augment.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/essd/combined_augment"

COMPOSITES_ROOT="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"

SAVE_PATH="$MODELS_DIR/lakevision_essd_combined_augment.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

for f in "$LABELS_2018" "$LABELS_2019"; do
    [ -f "$f" ] || { echo "ERROR: missing file $f"; exit 1; }
done
for d in "$COMPOSITES_ROOT/CW_2018" "$COMPOSITES_ROOT/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing directory $d"; exit 1; }
done

NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD Baseline + 8x Augmentation"
echo "=============================================="
echo "Composites: $COMPOSITES_ROOT/CW_{2018,2019}"
echo "Local SSD:  $NC_DIR"
echo "Model save: $SAVE_PATH"
echo "=============================================="

echo ""
echo "Copying composites to node-local SSD..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$COMPOSITES_ROOT/CW_2018/" "$NC_DIR/"
rsync -a "$COMPOSITES_ROOT/CW_2019/" "$NC_DIR/"
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
export WANDB_RUN_GROUP="essd_combined_augment"

cd "$SHERLOCK_DIR"

echo ""
echo "Start time: $(date)"

START_TIME=$(date +%s)

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" "$LABELS_2019" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$REPO_DIR/splits/essd_CW/train_ids.json" \
    --val_ids_file "$REPO_DIR/splits/essd_CW/val_ids.json" \
    --test_ids_file "$REPO_DIR/splits/essd_CW/test_ids.json" \
    --augment \
    --epochs 400 \
    --wandb_name "essd_combined_augment" \
    --save_path "$SAVE_PATH"

EXIT_CODE=$?

END_TIME=$(date +%s)
DUR=$((END_TIME - START_TIME))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: $((DUR / 3600))h $(( (DUR % 3600) / 60 ))m"
echo "Exit code: $EXIT_CODE"
[ $EXIT_CODE -eq 0 ] && [ -f "$SAVE_PATH" ] && ls -lh "$SAVE_PATH"
echo "=============================================="

exit $EXIT_CODE
