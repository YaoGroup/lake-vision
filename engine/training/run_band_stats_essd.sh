#!/bin/bash
#SBATCH --job-name=lv_band_stats_essd
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --array=0

# Per-band mean/std over the TRAINING lakes of each ESSD protocol, computed on
# the stacks_v2 DEPOSITS -- the data readers actually get from the SDR DOI, and
# therefore what the revision baseline trains on. (The published baselines used
# the internal composites, which are not in the deposit; their statistics are a
# different quantity and are not interchangeable.) Feeds --band_stats and
# --fill mean for the revision runs.
#   sbatch engine/training/run_band_stats_essd.sh              # task 0 only
#   sbatch --array=0-2 engine/training/run_band_stats_essd.sh  # all three
#     task 0: 2019-only train  (600 CW2019)      <- the revision protocol
#     task 1: cross-year train (800 CW2019)      <- published, kept for reference
#     task 2: combined  train  (1175 both years) <- published, kept for reference
set -euo pipefail
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS_ROOT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"

case "$SLURM_ARRAY_TASK_ID" in
  0) SPLIT="essd_CW_2019only";  OUT="$SHERLOCK_DIR/band_stats/band_stats_essd_2019only_deposit.json" ;;
  1) SPLIT="essd_CW_crossyear"; OUT="$SHERLOCK_DIR/band_stats/band_stats_essd_crossyear_deposit.json" ;;
  2) SPLIT="essd_CW";           OUT="$SHERLOCK_DIR/band_stats/band_stats_essd_combined_deposit.json" ;;
  *) echo "ERROR: array task must be 0, 1 or 2"; exit 1 ;;
esac
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] && { echo "ERROR: $OUT exists; delete it to recompute"; exit 1; }
for d in "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019"; do
    [ -d "$d" ] || { echo "ERROR: missing deposit directory $d"; exit 1; }
done

ml system python/3.12.1 py-numpy/1.26.3_py312
pip install --user netcdf4
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
echo "Split: $SPLIT -> $OUT"
echo "Start: $(date)"
python3 -u "$REPO_DIR/engine/preprocessing/compute_band_stats.py" \
    --nc_dir "$STACKS_ROOT/CW_2018" "$STACKS_ROOT/CW_2019" \
    --ids_file "$REPO_DIR/splits/$SPLIT/train_ids.json" \
    --out "$OUT" --stride 4 --tstride 3
echo "End: $(date)"
