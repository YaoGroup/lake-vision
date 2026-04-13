#!/bin/bash
#SBATCH --job-name=lv_essd_pilot18
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD PILOT (CW 2018 only) — smoke test while CW 2019 build runs overnight.
# =============================================================================
#
# Identical to run_training_essd_pilot.sh but trains only on CW 2018 lakes
# (679 total). Useful for validating end-to-end plumbing right away without
# waiting for the CW 2019 stack build to finish.
#
# Verifies:
#   - LakeDataset auto-detects raw sat-tile-stack `.nc` format
#   - essd_5class label loader handles a single CSV
#   - 5-class model head initializes correctly
#   - bs=16 + bf16 AMP fits in VRAM
#   - Forward/backward pass completes for 3 epochs
#   - Checkpoint saves
#
# Runs 3 epochs on 50 lakes per split. Copy step ~2-5 min for 679 files.
#
# USAGE:
#   sbatch run_training_essd_pilot_2018.sh
# =============================================================================

set -uo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models"

STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"

SAVE_PATH="$MODELS_DIR/lakevision_essd_pilot_2018.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

if [ ! -f "$LABELS_2018" ]; then
    echo "ERROR: missing labels file $LABELS_2018"
    exit 1
fi
if [ ! -d "$STACKS_ROOT/CW_2018" ]; then
    echo "ERROR: missing stack directory $STACKS_ROOT/CW_2018"
    exit 1
fi

NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD Pilot (CW 2018 only) — smoke test"
echo "=============================================="
echo "  3 epochs, 50 lakes per split"
echo "  5-class, imagery-only, no mask, bs=16 + bf16 AMP"
echo "=============================================="

echo ""
echo "Copying CW 2018 stacks to node-local SSD..."
COPY_START=$(date +%s)
mkdir -p "$NC_DIR"
rsync -a --info=progress2 "$STACKS_ROOT/CW_2018/" "$NC_DIR/"
COPY_END=$(date +%s)
COPY_SEC=$((COPY_END - COPY_START))
NC_COUNT=$(ls "$NC_DIR/"*.nc 2>/dev/null | wc -l)
echo "  Copied $NC_COUNT files in ${COPY_SEC}s"
echo "=============================================="

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
export WANDB_RUN_GROUP="essd_pilot_2018"

cd "$SHERLOCK_DIR"

echo ""
echo "Start time: $(date)"
echo ""

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" \
    --nc_dir "$NC_DIR" \
    --label_mode "essd_5class" \
    --id_col "lake_id" \
    --label_col "label" \
    --num_classes 5 \
    --train_ratio 0.7 \
    --val_ratio 0.2 \
    --test_ratio 0.1 \
    --max_lakes 50 \
    --epochs 3 \
    --batch_size 16 \
    --amp \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --seq_len 153 \
    --no_mask \
    --attention_type "none" \
    --frontcnn_base_channels 8 \
    --frontcnn_num_layers 4 \
    --clstm_hidden 32 \
    --slstm_hidden 16 \
    --classhead_hidden 64 \
    --classhead_dropout 0.3 \
    --save_path "$SAVE_PATH" \
    --num_workers 4 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "CW 2018 pilot succeeded. Once CW 2019 stacks are built,"
    echo "run the full pilot and then the real baselines:"
    echo "  sbatch run_training_essd_pilot.sh           (dual-year pilot)"
    echo "  sbatch run_training_essd_combined.sh        (baseline 1)"
    echo "  sbatch run_training_essd_crossyear.sh       (baseline 2)"
    echo "  sbatch run_training_essd_learning_curve.sh  (array)"
fi
echo "=============================================="

exit $EXIT_CODE
