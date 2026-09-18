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
# Added 2026-09-09 after R1 proved unstable (its attention stayed near-uniform, so
# the readout is effectively a mean; suspect: gradient into all 153 steps with no
# clipping). R11 isolates the readout; R12 is R1 with gradient clipping.
EXTRA_RUNS=(R11 R12)
# Added 2026-09-15 from the results: only R3 (standardised inputs) and R8 (soft
# labels) beat R0 on the cross-year test. A three-step ladder, each step one change:
#   R13 = R3 + R8 + the dynamic water mask + AdamW 1e-2, reference architecture
#   R14 = R13 + R2's optimiser stack (fits to 0.98 train; does decay convert that?)
#   R15 = R14 + random rot/flip augmentation (never used by any run so far)
# All three train on the COMBINED 2018+2019 split with the 40 bench lakes held
# out in test (splits/essd_CW_benchout), so they answer "what does a two-season
# model do" and still go on the bench ladder. $BAND_STATS = the benchout stats.
# R16 added 2026-09-18: R10 (the stack) is the best cross-year model (test 0.471,
# TBS 22), but R13-R15 are NOT R10 — they drop its attention readout, observed
# flag, static mask, CBAM and extra bands on the evidence of the single-change
# runs. Since R10 as a whole beat all of its parts, that pruning is a hypothesis,
# not a result. R16 is R10's exact flags on the benchout split, so the combined
# ladder has an apples-to-apples cell for the current best configuration.
# R17 added 2026-09-18. Measured over all 515 LD deposits: both years have the
# same number of days with no image (63 vs 62 of 153), but 2018 has 67 days where
# an image exists and is too cloudy to use against 2019's 42. --validity_channel
# (isfinite red) only ever saw the first number, so nothing the network has been
# given marks the days where the years actually differ. R17 = R16 + the per-pixel
# cloud mask as an aux channel.
COMBINED_RUNS=(R13 R14 R15 R16 R17)
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
        R11) echo "--temporal_readout mean" ;;
        R12) echo "$(run_flags R1) --grad_clip 1.0" ;;
        R13) echo "$(run_flags R3) $(run_flags R8) --mask_source dynamic --optimizer adamw --weight_decay 1e-2" ;;
        R14) echo "$(run_flags R13) $(run_flags R2)" ;;
        R15) echo "$(run_flags R14) --augment" ;;
        R16) echo "$(run_flags R10)" ;;
        R17) echo "$(run_flags R16) --cloud_channel" ;;
        *)   echo "unknown run $1" >&2; return 1 ;;
    esac
}

run_gpu_class() {
    case "$1" in
        R2|R9|R10|R14|R15|R16|R17) echo 80 ;;
        *)                         echo 40 ;;
    esac
}

# Which split directory (under splits/) a run trains on.
run_split() {
    case "$1" in
        R13|R14|R15|R16|R17) echo "essd_CW_benchout" ;;
        *)       echo "essd_CW_crossyear" ;;
    esac
}

# Which band_stats file a run's --band_stats must point at (basename under band_stats/).
run_band_stats_name() {
    case "$1" in
        R13|R14|R15|R16|R17) echo "band_stats_benchout_train.json" ;;
        *)       echo "band_stats_crossyear_train.json" ;;
    esac
}
