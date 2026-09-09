#!/bin/bash
# The eleven-runs flag matrix (docs/eleven_runs/ELEVEN_RUNS.html). Sourced by
# the local gate and both sbatch arrays so the flags live in exactly one place.
#
#   run_flags R3        -> the flags beyond the template for that run
#   run_gpu_class R9    -> "80" for the runs that need an 80 GB A100, else "40"
#
# Every run is trained on the deposit stacks (no 'mask' band: --no_mask), is
# scored on the test set with the best-val-F1 checkpoint (the ESSD selection
# rule) and writes a per-lake predictions table. $BAND_STATS must point at the
# training-split band_stats.json when R3 or R10 is used.

ELEVEN_RUNS=(R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10)
COMMON_FLAGS="--no_mask --test_checkpoint f1"

run_flags() {
    case "$1" in
        R0)  echo "" ;;
        R1)  echo "--temporal_readout attn --pool_type both" ;;
        R2)  echo "--frontcnn_norm group --clstm_forget_bias 1.0 --lr_schedule warmup_cosine --warmup_epochs 10" ;;
        R3)  echo "--band_stats ${BAND_STATS:?set BAND_STATS} --fill mean --validity_channel" ;;
        R4)  echo "--use_cloudyseq --cloudy_seq_var observed" ;;
        R5)  echo "--mask_source both" ;;
        R6)  echo "--attention_type spatial" ;;
        R7)  echo "--use_nir --use_swir16" ;;
        R8)  echo "--soft_labels" ;;
        R9)  echo "--frontcnn_base_channels 16 --clstm_hidden 64" ;;
        R10) echo "$(run_flags R1) $(run_flags R2) $(run_flags R3) $(run_flags R4) $(run_flags R5) $(run_flags R6) $(run_flags R7) $(run_flags R8)" ;;
        *)   echo "unknown run $1" >&2; return 1 ;;
    esac
}

run_gpu_class() {
    case "$1" in
        R2|R9|R10) echo 80 ;;
        *)         echo 40 ;;
    esac
}
