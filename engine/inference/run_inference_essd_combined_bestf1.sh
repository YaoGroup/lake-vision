#!/bin/bash
#SBATCH --job-name=lv_essd_inference_combined_bestf1
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
# Inference: COMBINED best-val-F1 checkpoint (saved @ epoch 334, val F1 0.607)
# on the combined 70/20/10 stratified split:
#     train (n=1175)  +  val (n=336)  +  test (n=168)   — all year-mixed
#
# Combined splits draw from BOTH CW 2018 and CW 2019, so (unlike the
# cross-year script) the year is derived per-lake from the lake_id prefix
# (CW2018_* -> 2018, CW2019_* -> 2019) and BOTH label CSVs are passed to
# run_inference.py (which accepts --labels_csv as nargs="+").
#
# Writes three CSVs to $OUT_DIR. Each row carries lake_id, true_label,
# pred_label, and per-class softmax probabilities (p_ND..p_CD).
#
# USAGE:
#   sbatch run_inference_essd_combined_bestf1.sh
# =============================================================================

set -euo pipefail

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
COMPOSITES_ROOT="$SHERLOCK_DIR/composites"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"

CHECKPOINT="$SHERLOCK_DIR/models/essd/combined/lakevision_essd_combined_bestf1.pth"
SPLITS_DIR="$REPO_DIR/splits/essd_CW"
OUT_DIR="$SHERLOCK_DIR/inference_essd/combined"

# Refuse to clobber: this run must go to a fresh directory.
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
echo "ESSD inference: combined bestf1 (ep 334, val F1 0.607)"
echo "=============================================="
echo "Checkpoint: $CHECKPOINT"
echo "Splits:     $SPLITS_DIR  (combined 70/20/10, year-mixed)"
echo "Out dir:    $OUT_DIR"
echo "Start time: $(date)"
echo "=============================================="

# Stage ONLY the .nc files we predict on. Combined splits are year-mixed, so
# derive the year from each lake_id prefix and preserve the CW_{year}/ subdir
# in $NC_DIR (run_inference.py's path resolver looks for nc_dir/CW_{year}/{id}.nc
# when a flat nc_dir/{id}.nc is absent). train+val+test == all 1679 lakes.
NC_DIR="$L_SCRATCH/nc_data"
mkdir -p "$NC_DIR"

LIST="$L_SCRATCH/files_to_stage.txt"
python3 -c "
import json
seen = set()
for path in ['$SPLITS_DIR/train_ids.json',
             '$SPLITS_DIR/val_ids.json',
             '$SPLITS_DIR/test_ids.json']:
    for lid in json.load(open(path)):
        if lid in seen:
            continue
        seen.add(lid)
        year = '2018' if lid.startswith('CW2018') else '2019'
        print(f'CW_{year}/{lid}.nc')
" | sort -u > "$LIST"
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

# Both label CSVs passed every call; run_inference.py merges them and resolves
# each lake by id (the splits, not the CSV, define which lakes are scored).

# --- TRAIN: 1175 lakes (year-mixed) ---
echo ""
echo "--- combined train (1175 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/train_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
    --nc_dir     "$NC_DIR" \
    --output_csv "$OUT_DIR/train_predictions_bestf1.csv"

# --- VAL: 336 lakes (year-mixed) ---
echo ""
echo "--- combined val (336 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/val_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
    --nc_dir     "$NC_DIR" \
    --output_csv "$OUT_DIR/val_predictions_bestf1.csv"

# --- TEST: 168 lakes (year-mixed) — the row dropped in the 20260507R session ---
echo ""
echo "--- combined test (168 lakes) ---"
python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
    --checkpoint "$CHECKPOINT" \
    --ids_file   "$SPLITS_DIR/test_ids.json" \
    --labels_csv "$LABELS_ROOT/labels_CW_2018.csv" "$LABELS_ROOT/labels_CW_2019.csv" \
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
