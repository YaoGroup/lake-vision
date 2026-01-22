#!/bin/bash
#SBATCH --job-name=lv_highcap
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=120:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=900GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu
#SBATCH --array=0-1

# =============================================================================
# HIGH-CAPACITY MODEL: 200 EPOCHS - ATTENTION COMPARISON
# =============================================================================
#
# Runs 2 experiments with different attention mechanisms:
#   0: Full CBAM attention (channel + spatial)
#   1: Architectural attention (mask pathway)
#
# Larger model with:
#   - frontcnn_base_channels: 16 (vs 8)
#   - clstm_hidden: 64 (vs 32)
#   - slstm_hidden: 32 (vs 16)
#   - classhead_hidden: 128 (vs 64)
#   - Higher dropout: 0.4 (vs 0.3)
#   - 200 epochs (vs 50)
#   - 800GB RAM with training data preloaded to memory (eliminates I/O bottleneck)
#
# USAGE:
#   sbatch run_highcap.sh
#
# =============================================================================

# Set paths
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/highcap"

# Data paths
LABELS_CSV="/oak/stanford/groups/cyaolai/JoshRines/data/labels_2019_volumes_CW.csv"
NC_DIR="/oak/stanford/groups/cyaolai/JoshRines/data/tstacks/CW2019_tstacks_with_cloudyseq"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"

# Create directories
mkdir -p "$SHERLOCK_DIR/logs"
mkdir -p "$MODELS_DIR"

# Define experiment configurations based on array task ID
case $SLURM_ARRAY_TASK_ID in
    0)
        EXP_NAME="cbam"
        ATTENTION_TYPE="full"
        ;;
    1)
        EXP_NAME="arch"
        ATTENTION_TYPE="arch"
        ;;
esac

SAVE_PATH="$MODELS_DIR/lakevision_highcap_${EXP_NAME}.pth"

echo "=============================================="
echo "Lake Vision Training - High Capacity Model"
echo "=============================================="
echo "Experiment: $EXP_NAME (array task $SLURM_ARRAY_TASK_ID)"
echo "Attention type: $ATTENTION_TYPE"
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
export WANDB_RUN_GROUP="highcap"

cd $SHERLOCK_DIR

echo ""
echo "Starting training..."
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# High-capacity training configuration
python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_CSV" \
    --nc_dir "$NC_DIR" \
    --id_col "new_id" \
    --label_col "label_rines" \
    --epochs 200 \
    --batch_size 2 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --use_scheduler \
    --seq_len 153 \
    --band_stats "$BAND_STATS" \
    --cloudy_seq_var "cloudy_seq_rgb" \
    --use_imgseq \
    --use_areaseq \
    --use_cloudyseq \
    --attention_type "$ATTENTION_TYPE" \
    --num_classes 4 \
    --frontcnn_base_channels 16 \
    --frontcnn_num_layers 4 \
    --clstm_hidden 64 \
    --slstm_hidden 32 \
    --classhead_hidden 128 \
    --classhead_dropout 0.4 \
    --gradient_checkpointing \
    --accumulation_steps 8 \
    --preload_to_ram \
    --save_path "$SAVE_PATH" \
    --num_workers 0 \
    --seed 42

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
