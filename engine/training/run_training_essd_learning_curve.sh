#!/bin/bash
#SBATCH --job-name=lv_essd_lcurve
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
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
#SBATCH --array=0-4

# =============================================================================
# ESSD LEARNING CURVE — nested stratified train subsets
# =============================================================================
#
# SLURM array job. Each task trains the baseline 5-class model on the
# first N lake IDs from splits/essd_CW/train_ids.json (nested stratified
# ordering, so N=400 is a superset of N=200). Validation and test sets
# are fixed across all N via splits/essd_CW/{val,test}_ids.json.
#
# N values:   200  400  600  800  1000
# Task IDs:     0    1    2    3     4
#
# The N=1175 point (= full train set) is already covered by
# run_training_essd_combined.sh, so it's omitted here.
#
# PREREQUISITES:
#   1. splits/essd_CW/{train,val,test}_ids.json — build with:
#        python engine/training/split_fixed.py \
#            --labels_csv /oak/.../essd_labels/labels_CW_2018.csv \
#                         /oak/.../essd_labels/labels_CW_2019.csv \
#            --out_dir /oak/.../repos/lake-vision/splits/essd_CW
#   2. CW 2018 + CW 2019 stacks at $STACKS_ROOT/CW_{2018,2019}/
#   3. Label CSVs at $LABELS_ROOT/labels_CW_{2018,2019}.csv
#
# USAGE:
#   sbatch run_training_essd_learning_curve.sh
# =============================================================================

set -euo pipefail

# --- Per-task N value ---
N_VALUES=(200 400 600 800 1000)
N_TRAIN="${N_VALUES[$SLURM_ARRAY_TASK_ID]}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/essd/lcurve"

STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"

SPLITS_DIR="$REPO_DIR/splits/essd_CW"
TRAIN_IDS="$SPLITS_DIR/train_ids.json"
VAL_IDS="$SPLITS_DIR/val_ids.json"
TEST_IDS="$SPLITS_DIR/test_ids.json"

SAVE_PATH="$MODELS_DIR/lakevision_essd_lcurve_N${N_TRAIN}.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

# --- Sanity checks ---
for f in "$LABELS_2018" "$LABELS_2019" "$TRAIN_IDS" "$VAL_IDS" "$TEST_IDS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing file $f"
        exit 1
    fi
done
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: missing stack directory $d"
        exit 1
    fi
done

NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD Learning Curve — N_train=$N_TRAIN"
echo "=============================================="
echo "Array task:   $SLURM_ARRAY_TASK_ID / ${#N_VALUES[@]}"
echo "Splits dir:   $SPLITS_DIR"
echo "Model save:   $SAVE_PATH"
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
export WANDB_RUN_GROUP="essd_lcurve"
export WANDB_NAME="essd_lcurve_N${N_TRAIN}"

cd "$SHERLOCK_DIR"

echo ""
echo "Start time: $(date)"
echo ""

python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" "$LABELS_2019" \
    --nc_dir "$NC_DIR" \
    --label_mode "essd_5class" \
    --id_col "lake_id" \
    --label_col "label" \
    --num_classes 5 \
    --wandb_name "essd_lcurve_N${N_TRAIN}" \
    --train_ids_file "$TRAIN_IDS" \
    --val_ids_file "$VAL_IDS" \
    --test_ids_file "$TEST_IDS" \
    --max_train_lakes "$N_TRAIN" \
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
    --frontcnn_num_layers 4 \
    --clstm_hidden 32 \
    --slstm_hidden 16 \
    --classhead_hidden 64 \
    --classhead_dropout 0.3 \
    --save_path "$SAVE_PATH" \
    --num_workers 7 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
if [ $EXIT_CODE -eq 0 ] && [ -f "$SAVE_PATH" ]; then
    echo "Model saved to: $SAVE_PATH"
fi
echo "=============================================="

exit $EXIT_CODE
