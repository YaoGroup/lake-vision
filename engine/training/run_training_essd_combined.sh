#!/bin/bash
#SBATCH --job-name=lv_essd_combined
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=96:00:00
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
# ESSD BASELINE 1 — COMBINED 2018+2019
# =============================================================================
#
# This IS the canonical ESSD baseline (imagery + water_area + static lake
# polygon mask, no attention, 5-class ND/HF/MD/LD/CD, no cloudy_seq).
# Most configuration is pulled from run_training.py's argparse defaults,
# which are hardcoded to the ESSD baseline for simplicity + paper-citation.
#
# PREREQUISITES:
#   - Composite .nc files at $COMPOSITES_ROOT/CW_{2018,2019}/ from
#     sbatch engine/preprocessing/run_synthesize_region.sh CW {2018, 2019}
#   - Label CSVs at $LABELS_ROOT/.
#
# USAGE:
#   sbatch run_training_essd_combined.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
MODELS_DIR="$SHERLOCK_DIR/models/essd/combined"

COMPOSITES_ROOT="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"
LABELS_2018="$LABELS_ROOT/labels_CW_2018.csv"
LABELS_2019="$LABELS_ROOT/labels_CW_2019.csv"

SAVE_PATH="$MODELS_DIR/lakevision_essd_combined.pth"

mkdir -p "$SHERLOCK_DIR/logs" "$MODELS_DIR"

for f in "$LABELS_2018" "$LABELS_2019"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing labels file $f"
        exit 1
    fi
done
for d in "$COMPOSITES_ROOT/CW_2018" "$COMPOSITES_ROOT/CW_2019"; do
    if [ ! -d "$d" ]; then
        echo "ERROR: missing composites directory $d"
        exit 1
    fi
done

NC_DIR="$L_SCRATCH/nc_data"

echo "=============================================="
echo "ESSD Baseline 1: Combined 2018+2019 (70/20/10)"
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
export WANDB_RUN_GROUP="essd_combined"

cd "$SHERLOCK_DIR"

echo ""
echo "Start time: $(date)"

START_TIME=$(date +%s)

# All other flags (num_classes=5, label_mode=essd_5class, epochs=400, bs=8,
# amp, no cloudyseq, attention_type=none, etc.) pulled from run_training.py
# argparse defaults.
python3 -u "$REPO_DIR/engine/training/run_training.py" \
    --labels_csv "$LABELS_2018" "$LABELS_2019" \
    --nc_dir "$NC_DIR" \
    --train_ids_file "$REPO_DIR/splits/essd_CW/train_ids.json" \
    --val_ids_file "$REPO_DIR/splits/essd_CW/val_ids.json" \
    --test_ids_file "$REPO_DIR/splits/essd_CW/test_ids.json" \
    --wandb_name "essd_combined" \
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
