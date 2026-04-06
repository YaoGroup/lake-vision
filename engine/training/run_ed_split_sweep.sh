#!/bin/bash
#SBATCH --job-name=lv_edsplit
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu
#SBATCH --array=0-15%6

# =============================================================================
# ED_SPLIT SWEEP: ATTENTION x AREASEQ x OPTIMIZER (SLURM array)
# =============================================================================
#
# 16 experiments = 4 attention x 2 areaseq x 2 optimizer
# Runs up to 6 simultaneously (%6 throttle).
# Uses ed_split label mode: ND=0, LD+MD=1, HF=2, CD=3
# Weighted CrossEntropyLoss (inverse frequency).
#
# USAGE:
#   sbatch run_ed_split_sweep.sh
#
# =============================================================================

# Set paths
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/ed_split_sweep"

# Data paths
LABELS_CSV="/oak/stanford/groups/cyaolai/JoshRines/data/labels_2019_volumes_CW.csv"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_with_cloudyseq"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"

# Create directories
mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$MODELS_DIR"

# -------------------------------------------------------------------------
# Define the 16 experiment configs
# Grid: attention_type x use_areaseq x optimizer
#   attention: none(0), spatial(1), full(2), arch(3)
#   areaseq:   false(0), true(1)
#   optimizer:  Adam(0), SGD(1)
#
#   task_id = attn*4 + areaseq*2 + optim
# -------------------------------------------------------------------------

ATTENTION_TYPES=("none" "spatial" "full" "arch")
AREASEQ_OPTS=("" "--use_areaseq")
AREASEQ_NAMES=("noarea" "area")
OPTIMIZERS=("Adam" "SGD")

TASK_ID=$SLURM_ARRAY_TASK_ID

ATTN_IDX=$((TASK_ID / 4))
REMAINDER=$((TASK_ID % 4))
AREA_IDX=$((REMAINDER / 2))
OPT_IDX=$((REMAINDER % 2))

ATTENTION=${ATTENTION_TYPES[$ATTN_IDX]}
AREASEQ=${AREASEQ_OPTS[$AREA_IDX]}
AREASEQ_NAME=${AREASEQ_NAMES[$AREA_IDX]}
OPTIMIZER=${OPTIMIZERS[$OPT_IDX]}

EXP_NAME="edsplit_${ATTENTION}_${AREASEQ_NAME}_${OPTIMIZER}"
SAVE_PATH="$MODELS_DIR/lakevision_${EXP_NAME}.pth"

echo "=============================================="
echo "Lake Vision ED-Split Sweep"
echo "=============================================="
echo "Experiment:  $EXP_NAME (array task $TASK_ID)"
echo "Attention:   $ATTENTION"
echo "Area seq:    $AREASEQ_NAME"
echo "Optimizer:   $OPTIMIZER"
echo "Label mode:  ed_split"
echo "Model save:  $SAVE_PATH"
echo "=============================================="

# Load modules
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

# Install additional dependencies
pip install --user xarray netcdf4

# Add repo to PYTHONPATH
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Wandb offline mode
export WANDB_MODE=offline
export WANDB_DIR="$SHERLOCK_DIR"
export WANDB_PROJECT="lake-vision"
export WANDB_RUN_GROUP="ed_split_sweep"

cd $SHERLOCK_DIR

echo ""
echo "Starting training..."
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Set optimizer-specific args
if [ "$OPTIMIZER" == "Adam" ]; then
    LR=0.001
else
    LR=0.001
fi

# Training configuration
python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_CSV" \
    --nc_dir "$NC_DIR" \
    --id_col "new_id" \
    --label_col "label_rines" \
    --label_mode "ed_split" \
    --epochs 250 \
    --batch_size 4 \
    --lr $LR \
    --seq_len 153 \
    --band_stats "$BAND_STATS" \
    --use_imgseq \
    $AREASEQ \
    --attention_type "$ATTENTION" \
    --num_classes 4 \
    --frontcnn_base_channels 8 \
    --frontcnn_num_layers 3 \
    --clstm_hidden 32 \
    --slstm_hidden 16 \
    --classhead_hidden 64 \
    --classhead_dropout 0.3 \
    --save_path "$SAVE_PATH" \
    --num_workers 2 \
    --seed 42 \
    --wandb_name "$EXP_NAME"

EXIT_CODE=$?

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
echo "Experiment: $EXP_NAME"
echo "End time: $(date)"
echo "Duration: ${DURATION_HR}h ${DURATION_MIN_REM}m (${DURATION_SEC}s total)"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    if [ -f "$SAVE_PATH" ]; then
        echo "Model saved to: $SAVE_PATH"
        ls -lh "$SAVE_PATH"
    else
        echo "WARNING: Model file not found at $SAVE_PATH"
    fi
else
    echo "Training FAILED with exit code $EXIT_CODE"
fi
echo "=============================================="
echo ""
echo "To sync wandb runs, run:"
echo "  cd $SHERLOCK_DIR && wandb sync wandb/offline-run-*"
echo ""

exit $EXIT_CODE
