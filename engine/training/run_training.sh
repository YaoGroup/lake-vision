#!/bin/bash
#SBATCH --job-name=lakevision
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=24:00:00
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
# TRAIN LAKE-VISION CLASSIFIER
# =============================================================================
#
# Trains the LakeDrainageClassifier model on lake NC files.
#
# PREREQUISITES:
#   - Lake NC files with cloudy_seq variables (from cloudy-tile inference)
#   - Labels CSV file with lake IDs and labels
#
# USAGE:
#   sbatch run_training.sh
#
# OUTPUT:
#   Model weights saved to:
#     /oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/models/
#
# =============================================================================

# Set paths
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models"

# Data paths
LABELS_CSV="/oak/stanford/groups/cyaolai/JoshRines/data/labels_2019_volumes_CW/labels_2019_volumes_CW.csv"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_processed"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"

# Model save path
SAVE_PATH="$MODELS_DIR/lakevision_baseline.pth"

# Create directories
mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$MODELS_DIR"

echo "=============================================="
echo "Lake Vision Training"
echo "=============================================="
echo "Labels CSV: $LABELS_CSV"
echo "NC Directory: $NC_DIR"
echo "Model save path: $SAVE_PATH"
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
export WANDB_RUN_GROUP="baseline"

cd $SHERLOCK_DIR

echo ""
echo "Starting training..."
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Training configuration
# Adjust these hyperparameters as needed
python3 "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_CSV" \
    --nc_dir "$NC_DIR" \
    --id_col "new_id" \
    --label_col "label_rines" \
    --epochs 50 \
    --batch_size 4 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --use_scheduler \
    --seq_len 153 \
    --band_stats "$BAND_STATS" \
    --cloudy_seq_var "cloudy_seq_rgb" \
    --use_areaseq \
    --attention_type "none" \
    --num_classes 4 \
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

END_TIME=$(date +%s)
DURATION_SEC=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION_SEC / 60))
DURATION_HR=$((DURATION_MIN / 60))
DURATION_MIN_REM=$((DURATION_MIN % 60))

echo ""
echo "=============================================="
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

exit $EXIT_CODE
