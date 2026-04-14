#!/bin/bash
#SBATCH --job-name=lv_synthesize
#SBATCH --output=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.out
#SBATCH --error=/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/logs/%x_%j.err
#SBATCH --time=12:00:00
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
# SYNTHESIZE ESSD COMPOSITE .NC FILES (CPU) + CLOUDY-TILE INFERENCE (GPU)
# =============================================================================
#
# Produces the final per-lake composite NetCDFs for the ESSD benchmark dataset
# by combining:
#   - raw sat-tile-stack output          (imagery, 5 bands + SCL-cloudmask)
#   - Dunmire static lake polygons       (→ static mask channel)
#   - Dunmire S2_water time series       (→ water_area[T])
#   - GUI 5-class labels                 (→ global attrs)
#   - cloudy-tile RGB classifier         (→ cloudy_seq_rgb[T])
#
# USAGE:
#   sbatch run_synthesize_region.sh <REGION> <YEAR>
#
#   REGION: CW, NW, NO, NE, SW, SE
#   YEAR:   2018 or 2019
#
# EXAMPLE:
#   sbatch run_synthesize_region.sh CW 2018
#
# OUTPUT:
#   /oak/.../sherlock_lakevision/composites/<REGION>_<YEAR>/
#
# =============================================================================

set -uo pipefail
# No -e: same SIGPIPE-on-skip caveat as sat-tile-stack's build script.

# --- Parse args ---
REGION="${1:?Usage: sbatch run_synthesize_region.sh <REGION> <YEAR>}"
YEAR="${2:?Usage: sbatch run_synthesize_region.sh <REGION> <YEAR>}"

VALID_REGIONS="CW NW NO NE SW SE"
if ! echo "$VALID_REGIONS" | grep -qw "$REGION"; then
    echo "ERROR: Invalid region '$REGION'. Must be one of: $VALID_REGIONS"
    exit 1
fi
if [[ "$YEAR" != "2018" && "$YEAR" != "2019" ]]; then
    echo "ERROR: Invalid year '$YEAR'. Must be 2018 or 2019."
    exit 1
fi

# --- Paths ---
SHERLOCK_LV="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision"
SHERLOCK_STS="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack"
SHERLOCK_CT="/oak/stanford/groups/cyaolai/JoshRines/sherlock/sherlock_cloudytile"
REPO_LV="/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision"
REPO_STS="/oak/stanford/groups/cyaolai/JoshRines/repos/sat-tile-stack"

RAW_DIR="$SHERLOCK_STS/stacks/${REGION}_${YEAR}"
OUTPUT_DIR="$SHERLOCK_LV/composites/${REGION}_${YEAR}"

DUNMIRE_AREA="/oak/stanford/groups/cyaolai/JoshRines/data/dunmire/all_lakes_${YEAR}.nc"
DUNMIRE_POLYGONS="$REPO_STS/labeling/dunmire/labels_${YEAR}_volumes.geojson"
LABELS_CSV="/oak/stanford/groups/cyaolai/JoshRines/data/essd_labels/labels_${REGION}_${YEAR}.csv"

# Cloudy-tile weights (only RGB variant for MVP)
CLOUDYTILE_RGB_WEIGHTS="$SHERLOCK_CT/models/cloudytile_rgb.pth"
BAND_STATS="/oak/stanford/groups/cyaolai/JoshRines/data/cloudytile/band_stats.json"

mkdir -p "$SHERLOCK_LV/logs" "$OUTPUT_DIR"

# --- Sanity checks ---
echo "=============================================="
echo "ESSD Composite Synthesis: ${REGION} ${YEAR}"
echo "=============================================="
echo "Raw stacks:       $RAW_DIR"
echo "Dunmire area:     $DUNMIRE_AREA"
echo "Dunmire polygons: $DUNMIRE_POLYGONS"
echo "Labels CSV:       $LABELS_CSV"
echo "Output:           $OUTPUT_DIR"
echo "Cloudy-tile RGB:  $CLOUDYTILE_RGB_WEIGHTS"
echo "=============================================="

for f in "$DUNMIRE_AREA" "$DUNMIRE_POLYGONS" "$CLOUDYTILE_RGB_WEIGHTS" "$BAND_STATS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing file $f"
        exit 1
    fi
done
if [ ! -d "$RAW_DIR" ]; then
    echo "ERROR: missing raw stacks directory $RAW_DIR"
    exit 1
fi
if [ ! -f "$LABELS_CSV" ]; then
    echo "NOTE: labels CSV not found at $LABELS_CSV — proceeding without label attrs"
    LABELS_CSV=""
fi

# --- Environment ---
ml system
ml python/3.12.1
ml py-numpy/1.26.3_py312
ml py-pandas/2.2.1_py312
ml py-scipy/1.12.0_py312
ml py-pytorch/2.2.1_py312
ml py-torchvision/0.17.1_py312
ml py-scikit-learn/1.5.1_py312

pip install --user xarray netcdf4 rasterio shapely geopandas affine

export PYTHONPATH="$REPO_LV:$REPO_STS:$PYTHONPATH"

# =============================================================================
# PHASE 1 — synthesize composites (CPU only; GPU idle this phase)
# =============================================================================

echo ""
echo "=============================================="
echo "PHASE 1: synthesize composites (CPU)"
echo "=============================================="
P1_START=$(date +%s)

LABELS_ARG=""
if [ -n "$LABELS_CSV" ]; then
    LABELS_ARG="--labels_csv $LABELS_CSV"
fi

python3 -u "$REPO_LV/engine/preprocessing/synthesize_region.py" \
    --region "$REGION" \
    --year "$YEAR" \
    --raw_dir "$RAW_DIR" \
    --dunmire_area "$DUNMIRE_AREA" \
    --dunmire_polygons "$DUNMIRE_POLYGONS" \
    $LABELS_ARG \
    --output_dir "$OUTPUT_DIR" \
    --workers 8

P1_EXIT=$?
P1_END=$(date +%s)
echo ""
echo "Phase 1 duration: $((P1_END - P1_START))s  (exit=$P1_EXIT)"

if [ $P1_EXIT -ne 0 ]; then
    echo "ERROR: Phase 1 failed; skipping phase 2."
    exit $P1_EXIT
fi

# =============================================================================
# PHASE 2 — cloudy-tile RGB inference (GPU)
# =============================================================================

echo ""
echo "=============================================="
echo "PHASE 2: cloudy-tile RGB inference (GPU)"
echo "=============================================="
P2_START=$(date +%s)

# Call process_directory as a Python one-liner, in-place on the composites.
# The cloudy-tile library already knows how to open ds["imagery"] and query
# ds.coords["channel"] — this matches our composite schema exactly.
python3 -u <<PY
import sys
sys.path.insert(0, '$REPO_LV')
sys.path.insert(0, '/oak/stanford/groups/cyaolai/JoshRines/repos/cloudy-tile')
from cloudytile.inference import process_directory

process_directory(
    nc_dir='$OUTPUT_DIR',
    model_path='$CLOUDYTILE_RGB_WEIGHTS',
    nc_channels=['red', 'green', 'blue'],
    band_stats_path='$BAND_STATS',
    var_name='cloudy_seq_rgb',
    output_dir=None,  # in-place: adds cloudy_seq_rgb variable to each composite
)
PY

P2_EXIT=$?
P2_END=$(date +%s)
echo ""
echo "Phase 2 duration: $((P2_END - P2_START))s  (exit=$P2_EXIT)"

# --- Summary ---
N_COMPOSITES=$(ls "$OUTPUT_DIR"/*.nc 2>/dev/null | wc -l)
TOTAL=$((P2_END - P1_START))
echo ""
echo "=============================================="
echo "SYNTHESIS COMPLETE — ${REGION} ${YEAR}"
echo "=============================================="
echo "Composites written: $N_COMPOSITES"
echo "Total duration:     ${TOTAL}s (~$((TOTAL / 60)) min)"
echo "Output directory:   $OUTPUT_DIR"
echo "=============================================="

exit $P2_EXIT
