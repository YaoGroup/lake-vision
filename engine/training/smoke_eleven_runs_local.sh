#!/bin/bash
# Local gate for the eleven runs: every configuration trains for 2 epochs on
# the twelve local deposit stacks (CPU, batch 1) and must finish with a finite
# loss and a predictions table. Catches flag, shape and NaN errors in minutes,
# before anything touches Sherlock.
#
#   bash engine/training/smoke_eleven_runs_local.sh            # all eleven
#   bash engine/training/smoke_eleven_runs_local.sh R3 R10     # a subset
#
# Memory: the Mac has 17 GB, so the conv stack runs in checkpointed chunks of
# 16 frames. That is numerically exact (see FrontCNN) and not part of any
# run's science flags.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/eleven_runs_matrix.sh"

PY="${PY:-$HOME/anaconda3/envs/lakevision/bin/python}"
STACKS="${STACKS:-$HOME/stanford_gp/dissertation/defense/cinematic/source_data/stacks/CW_2019}"
LABELS="${LABELS:-$HOME/stanford_gp/research/lakes/2026/essd/labels/labels_CW_2019.csv}"
OUT="${OUT:-${TMPDIR:-/tmp}/eleven_runs_smoke}"
EPOCHS="${EPOCHS:-2}"
mkdir -p "$OUT"

# 12 local lakes -> 8 train / 2 val / 2 test, as the trainer's id files
$PY - "$STACKS" "$OUT" <<'PYEOF'
import json, sys, pathlib
ids = sorted(p.stem for p in pathlib.Path(sys.argv[1]).glob("*.nc"))
out = pathlib.Path(sys.argv[2])
json.dump(ids[:8], open(out / "train_ids.json", "w"))
json.dump(ids[8:10], open(out / "val_ids.json", "w"))
json.dump(ids[10:12], open(out / "test_ids.json", "w"))
print(f"{len(ids)} local stacks -> train {ids[:8]} val {ids[8:10]} test {ids[10:12]}")
PYEOF

export BAND_STATS="$OUT/band_stats_train.json"
if [ ! -f "$BAND_STATS" ]; then
    $PY "$REPO/engine/preprocessing/compute_band_stats.py" \
        --nc_dir "$STACKS" --ids_file "$OUT/train_ids.json" --out "$BAND_STATS" --stride 8 --tstride 6
fi

RUNS=("$@"); [ ${#RUNS[@]} -eq 0 ] && RUNS=("${ELEVEN_RUNS[@]}")
status=0
for run in "${RUNS[@]}"; do
    log="$OUT/$run.log"
    echo "=== $run: $(run_flags "$run") ==="
    if $PY "$REPO/engine/training/run_training.py" \
        --labels_csv "$LABELS" --nc_dir "$STACKS" \
        --train_ids_file "$OUT/train_ids.json" --val_ids_file "$OUT/val_ids.json" --test_ids_file "$OUT/test_ids.json" \
        --epochs "$EPOCHS" --batch_size 1 --num_workers 2 --no_amp --no_wandb \
        --frontcnn_chunk_size 16 --gradient_checkpointing \
        --save_path "$OUT/$run.pth" --test_predictions_csv "$OUT/${run}_test_predictions.csv" \
        $COMMON_FLAGS $(run_flags "$run") > "$log" 2>&1; then
        loss=$(grep -E "^  Train " "$log" | tail -1 | awk '{print $2}')
        f1=$(grep -E "F1 \(macro\):" "$log" | tail -1 | awk '{print $NF}')
        peak=$(grep -E "^Model input:" "$log" | head -1)
        if [ -n "$loss" ] && [ "$loss" != "nan" ] && [ -f "$OUT/${run}_test_predictions.csv" ]; then
            echo "    OK   final train loss $loss, test macro-F1 $f1 | $peak"
        else
            echo "    FAIL (loss='$loss'); see $log"; status=1
        fi
    else
        echo "    FAIL (exit code); see $log"; tail -5 "$log"; status=1
    fi
done
echo; echo "gate exit status $status (logs in $OUT)"; exit $status
