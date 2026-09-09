#!/bin/bash
#SBATCH --job-name=lv_band_stats
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p serc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB

# Per-band mean/std over the 800 cross-year TRAINING lakes (never val/test),
# for R3 and R10 (--band_stats / --fill mean). CPU only, reads Oak directly.
#   sbatch engine/training/run_band_stats_crossyear.sh
set -euo pipefail
SHERLOCK_DIR="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
REPO_DIR="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
STACKS="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2/CW_2019"
OUT="$SHERLOCK_DIR/band_stats/band_stats_crossyear_train.json"
mkdir -p "$(dirname "$OUT")"
[ -f "$OUT" ] && { echo "ERROR: $OUT exists; delete it to recompute"; exit 1; }

ml system python/3.12.1 py-numpy/1.26.3_py312
pip install --user netcdf4
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
echo "Start: $(date)"
python3 -u "$REPO_DIR/engine/preprocessing/compute_band_stats.py" \
    --nc_dir "$STACKS" --ids_file "$REPO_DIR/splits/essd_CW_crossyear/train_ids.json" \
    --out "$OUT" --stride 4 --tstride 3
echo "End: $(date)"
