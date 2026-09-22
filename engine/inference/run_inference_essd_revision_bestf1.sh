#!/bin/bash
#SBATCH --job-name=lv_essd_rev_inference
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH -p serc
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH -C GPU_SKU:A100_SXM4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jrines@stanford.edu

# =============================================================================
# ESSD revision: score one cell's best-val-F1 checkpoint on its TRAIN and VAL
# splits (the training job already wrote the TEST table with this same
# checkpoint). Feeds Appendix B's spatial map (figB1_map: train / val / test).
#
#   sbatch engine/inference/run_inference_essd_revision_bestf1.sh
#   sbatch --export=ALL,RUN=y2019_4c_nopw_noaug engine/inference/run_inference_essd_revision_bestf1.sh
#   sbatch --export=ALL,WITH_TEST=1 ...        # also re-score test (should reproduce the training CSV)
#
# The model and the dataset are rebuilt from the checkpoint's stored config by
# run_inference.py, so the inputs are exactly what training used: the
# stacks_v2 DEPOSIT, train-split band standardisation, mean fill, validity
# channel, static mask. Nothing here needs to repeat those flags.
#
# Outputs, alongside the training job's test CSV:
#   $SHERLOCK_DIR/inference_essd/essd_revision/${RUN}_{train,val}_predictions_bestf1.csv
# =============================================================================
set -euo pipefail

RUN="${RUN:-y2019_5c_pw_noaug}"
WITH_TEST="${WITH_TEST:-0}"

SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
LABELS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels"

CHECKPOINT="$SHERLOCK_DIR/models/essd_revision/lakevision_essd_rev_${RUN}_bestf1.pth"
SPLITS_DIR="$REPO_DIR/splits/essd_CW_2019only"
OUT_DIR="$SHERLOCK_DIR/inference_essd/essd_revision"
LABELS="$LABELS_ROOT/labels_CW_2019.csv"

for f in "$CHECKPOINT" "$SPLITS_DIR/train_ids.json" "$SPLITS_DIR/val_ids.json" \
         "$SPLITS_DIR/test_ids.json" "$LABELS"; do
    [ -f "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -d "$STACKS_ROOT/CW_2019" ] || { echo "ERROR: missing deposit directory $STACKS_ROOT/CW_2019"; exit 1; }
# Refuse to clobber a finished table. Delete it deliberately to redo.
for split in train val; do
    out="$OUT_DIR/${RUN}_${split}_predictions_bestf1.csv"
    [ -f "$out" ] && { echo "ERROR: $out exists; delete it to recompute"; exit 1; }
done
mkdir -p "$OUT_DIR" "$SHERLOCK_DIR/logs"

echo "=============================================="
echo "ESSD revision inference: $RUN  (job $SLURM_JOB_ID)"
echo "=============================================="
echo "Node:       $(hostname)   GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
echo "Checkpoint: $CHECKPOINT"
echo "Splits:     $SPLITS_DIR"
echo "Imagery:    $STACKS_ROOT/CW_2019  (stacks_v2 deposit)"
echo "Out dir:    $OUT_DIR"
echo "Start:      $(date)"
echo "=============================================="

# Stage only the files we will score (800, or 1000 with WITH_TEST) to node-local
# SSD. The deposit is ~136 MB/lake, so this is ~110-136 GB and ~15 min.
NC_DIR="$L_SCRATCH/nc_data"
mkdir -p "$NC_DIR"
LIST="$L_SCRATCH/files_to_stage.txt"
SPLITS="train val"; [ "$WITH_TEST" = "1" ] && SPLITS="train val test"
python3 - "$SPLITS_DIR" $SPLITS > "$LIST" <<'PY'
import json, sys
d = sys.argv[1]
for split in sys.argv[2:]:
    for lid in json.load(open(f"{d}/{split}_ids.json")):
        print(f"{lid}.nc")
PY
echo "Staging $(wc -l < "$LIST") deposit files to $NC_DIR ($(date))..."
COPY_START=$(date +%s)
rsync -a --files-from="$LIST" "$STACKS_ROOT/CW_2019/" "$NC_DIR/"
echo "  staged $(ls "$NC_DIR"/*.nc | wc -l) files, $(du -sh "$NC_DIR" | cut -f1), $(( $(date +%s) - COPY_START ))s"

ml system python/3.12.1 py-numpy/1.26.3_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-pytorch/2.2.1_py312 py-torchvision/0.17.1_py312 py-scikit-learn/1.5.1_py312
pip install --user xarray netcdf4
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
cd "$SHERLOCK_DIR"

START=$(date +%s)
for split in $SPLITS; do
    echo ""; echo "--- $split ($(python3 -c "import json;print(len(json.load(open('$SPLITS_DIR/${split}_ids.json'))))") lakes) ---"
    out="$OUT_DIR/${RUN}_${split}_predictions_bestf1.csv"
    [ "$split" = "test" ] && out="$OUT_DIR/${RUN}_test_predictions_bestf1_rescored.csv"
    python3 -u "$REPO_DIR/engine/inference/run_inference.py" \
        --checkpoint "$CHECKPOINT" \
        --ids_file   "$SPLITS_DIR/${split}_ids.json" \
        --labels_csv "$LABELS" \
        --nc_dir     "$NC_DIR" \
        --output_csv "$out" \
        --batch_size 4 --num_workers 8
done

DUR=$(( $(date +%s) - START ))
echo ""; echo "=============================================="
echo "End: $(date)   inference $((DUR / 60))m $((DUR % 60))s"
ls -lh "$OUT_DIR"/${RUN}_*_predictions_bestf1*.csv
echo "=============================================="
