#!/bin/bash
#SBATCH --job-name=lv_essd_inference_crossyear_bestf1
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# Inference: cross-year best-F1 checkpoint on train (CW 2019, n=800)
#                                          + val   (CW 2019, n=200)
#                                          + test  (CW 2018, n=679)
#
# Writes three CSVs to $OUT_DIR. Each row carries lake_id, true_label,
# pred_label, and per-class softmax probabilities (p_ND, p_HF, p_MD, p_LD, p_CD).
#
# USAGE:
#   sbatch run_inference_essd_crossyear_bestf1.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
COMPOSITES_ROOT="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"

CHECKPOINT="$SHERLOCK_DIR/models/essd/crossyear/lakevision_essd_crossyear_bestf1.pth"
SPLITS_DIR="$REPO_DIR/splits/essd_CW_crossyear"
OUT_DIR="$SHERLOCK_DIR/inference_essd/crossyear"

# Refuse to clobber: this run must go to a fresh directory. The legacy
# results under $SHERLOCK_DIR/inference/essd_crossyear_bestf1 are left intact.
if [ -e "$OUT_DIR" ] && [ -n "$(ls -A "$OUT_DIR" 2>/dev/null)" ]; then
    echo "ERROR: $OUT_DIR already exists and is non-empty — refusing to overwrite."
    echo "       Move/rename it or change OUT_DIR, then resubmit."
    exit 1
fi
mkdir -p "$OUT_DIR" "$SHERLOCK_DIR/logs"

for f in "$CHECKPOINT" \
         "$SPLITS_DIR/train_ids.json" \
         "$SPLITS_DIR/val_ids.json" \
         "$SPLITS_DIR/test_ids.json" \
         "$LABELS_ROOT/labels_CW_2018.csv" \
         "$LABELS_ROOT/labels_CW_2019.csv"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done

echo "=============================================="
echo "ESSD inference: cross-year bestf1"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Out dir:    $OUT_DIR"
echo "Start time: $(date)"
echo "=============================================="

# Stage ONLY the .nc files we'll predict on (train + val + test) — selective
# staging. Concurrent training jobs on the same node also use $L_SCRATCH, so
# copying the entire composites tree (~1679 files × ~30 MB ≈ 50 GB) competes
# with their rsyncs and can exhaust node-local SSD ("No space left on device"
# on rsync). The 1679 files we actually need fit at ~50 GB; bump --mem-per-cpu
# or move to a node with more $L_SCRATCH headroom if you observe rsync errors.
NC_DIR="$L_SCRATCH/nc_data"
mkdir -p "$NC_DIR"

LIST="$L_SCRATCH/files_to_stage.txt"
python3 -c "
import json
for path, year in [('$SPLITS_DIR/train_ids.json', '2019'),
                   ('$SPLITS_DIR/val_ids.json',   '2019'),
                   ('$SPLITS_DIR/test_ids.json',  '2018')]:
    for lid in json.load(open(path)):
        print(f'CW_{year}/{lid}.nc')
" > "$LIST"
N_NEEDED=$(wc -l < "$LIST")
echo "Staging $N_NEEDED selected .nc files (train + val + test) to $NC_DIR ..."

COPY_START=$(date +%s)
rsync -a --files-from="$LIST" "$COMPOSITES_ROOT/" "$NC_DIR/"
COPY_END=$(date +%s)
N_STAGED=$(find "$NC_DIR" -name '*.nc' | wc -l)
echo "Staged $N_STAGED .nc files in $((COPY_END - COPY_START))s"
echo "Local-scratch usage:"
du -sh "$NC_DIR"
df -h "$L_SCRATCH" | tail -1

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 \
    py-scipy/1.12.0_py312 py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 \
    py-scikit-learn/1.5.1_py312
pip install --user xarray netcdf4

export PYTHONPATH="$REPO_DIR:$PYTHONPATH"
cd "$SHERLOCK_DIR"

START=$(date +%s)

# --- TRAIN: 800 lakes from CW 2019 ---
echo ""
echo "--- train 2019 (800 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/train_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2019.csv" \
    --nc_dir     "$NC_DIR" \
    --output_csv "$OUT_DIR/train_predictions_bestf1.csv"

# --- VAL: 200 lakes from CW 2019 ---
echo ""
echo "--- val 2019 (200 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/val_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2019.csv" \
    --nc_dir     "$NC_DIR" \
    --output_csv "$OUT_DIR/val_predictions_bestf1.csv"

# --- TEST: 679 lakes from CW 2018 ---
echo ""
echo "--- test 2018 (679 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/test_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" \
    --nc_dir     "$NC_DIR" \
    --output_csv "$OUT_DIR/test_predictions_bestf1.csv"

END=$(date +%s)
DUR=$((END - START))

echo ""
echo "=============================================="
echo "End time: $(date)"
echo "Duration: $((DUR / 60))m $((DUR % 60))s"
ls -lh "$OUT_DIR"
echo "=============================================="
