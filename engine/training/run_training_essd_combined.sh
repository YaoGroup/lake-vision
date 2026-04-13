#!/bin/bash
#SBATCH --job-name=lv_essd_combined
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD BASELINE 1 — COMBINED 2018+2019
# =============================================================================
#
# Trains the 5-class LakeDrainageClassifier on the union of CW 2018 and
# CW 2019 labels, with a stratified 70/20/10 train/val/test split.
#
# PREREQUISITES:
#   - CW 2018 stacks at $STACKS_ROOT/CW_2018/
#   - CW 2019 stacks at $STACKS_ROOT/CW_2019/
#   - GUI label CSVs at $LABELS_ROOT/
#     (copy from laptop: sat-tile-stack/labeling/CW_{2018,2019}/labels_CW_{2018,2019}.csv)
#
# USAGE:
#   sbatch run_training_essd_combined.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models"

# Data paths (source on OAK)
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"

SAVE_PATH="$MODELS_DIR/lakevision_essd_combined.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

# --- Sanity checks on source data ---
for f in "$LABELS_2018" "$LABELS_2019"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing labels file $f"
        echo "Copy it from laptop sat-tile-stack/labeling/CW_XXXX/labels_CW_XXXX.csv"
        exit 1
    fi
done
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: missing stack directory $d"
        exit 1
    fi
done

# -------------------------------------------------------------------------
# Copy training data to node-local SSD ($L_SCRATCH) for fast I/O.
# Both CW_2018 and CW_2019 stacks copy into a single merged directory —
# lake IDs (CW2018_*, CW2019_*) don't collide so this is safe.
# -------------------------------------------------------------------------
NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD Baseline 1: Combined 2018+2019 (70/20/10)"
echo "=============================================="
echo "Labels 2018: $LABELS_2018"
echo "Labels 2019: $LABELS_2019"
echo "Stacks:      $STACKS_ROOT/CW_{2018,2019}"
echo "Local SSD:   $NC_DIR"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

echo ""
echo "Copying stacks to node-local SSD..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a "$STACKS_ROOT/CW_2018/" "$NC_DIR/"
rsync -a "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
COPY_END=$(date +%s)
COPY_SEC=$((COPY_END - COPY_START))
NC_COUNT=$(ls "$NC_DIR/"*.nc 2>/dev/null | wc -l)
echo "  Copied $NC_COUNT files in ${COPY_SEC}s"
echo "=============================================="

# --- Modules ---
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

pip install --user xarray netcdf4

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="lake-vision"
export WANDB_RUN_GROUP="essd_combined"

cd "$SHERLOCK_DIR"

echo ""
echo "Starting training..."
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" "$LABELS_2019" \
    --nc_dir "$NC_DIR" \
    --label_mode "essd_5class" \
    --id_col "lake_id" \
    --label_col "label" \
    --num_classes 5 \
    --wandb_name "essd_combined" \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --epochs 50 \
    --batch_size 8 \
    --amp \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --use_scheduler \
    --seq_len 153 \
    --no_mask \
    --no_areaseq \
    --attention_type "none" \
    --frontcnn_base_channels 8 \
    --frontcnn_num_layers 3 \
    --clstm_hidden 32 \
    --slstm_hidden 16 \
    --classhead_hidden 64 \
    --classhead_dropout 0.3 \
    --save_path "$SAVE_PATH" \
    --num_workers 7 \
    --seed 42

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ] && [ -f "$SAVE_PATH" ]; then
    echo "Model saved to: $SAVE_PATH"
    ls -lh "$SAVE_PATH"
fi
echo "=============================================="

exit $EXIT_CODE
