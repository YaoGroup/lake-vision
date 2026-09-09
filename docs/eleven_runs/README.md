# Eleven Runs for lake-vision

The 2026-09-08 proposal for eleven Sherlock training runs on the ESSD baseline
classifier: one change each against a common reference, plus the stack. Status:
**proposal**. Nothing here has been coded into the model or submitted.

Start by opening `ELEVEN_RUNS.html` in a browser (it is self-contained: no
build, no server; the fonts come from Google Fonts and fall back cleanly
offline). Read it top to bottom. The last section, *For the next session*, is the
hand-off brief: paths, rules, reading order, the deposit-reader gap, per-piece
specifications, the eleven-run flag matrix and the smoke-test gate.

The same page is published as a Claude artifact at
https://claude.ai/code/artifact/1ce47cc8-92e4-4be1-8839-302d482b99da
(Josh's account; the file here is the copy of record).

## What is in this folder

| file | what |
|---|---|
| `ELEVEN_RUNS.html` | the proposal, diagnosis and hand-off brief |
| `bench_variants.py` | measures parameters and forward FLOPs per lake for every run's architecture, using this repo's own modules (`torch.utils.flop_counter`); run from the repo root with the `lakevision` env |
| `bench_costs.json` | the bench output the page quotes (ESSD as published 225 GFLOP/lake; R0 121; R10 150; R9 451) |
| `mem_bench.py` | saved-for-backward bytes per run under bf16 autocast, scaled to batch 8 × 153 frames; run from the repo root with the `lakevision` env |
| `mem_costs.json` | the bench output the page quotes (R0 28 GB, R2 38, R10 48, R9 51 at batch 8; halve for batch 4) |
| `../../engine/training/eleven_runs_matrix.sh` | the flag matrix, one place, sourced by the gate and the sbatch |
| `../../engine/training/smoke_eleven_runs_local.sh` | local gate: all eleven configs, 2 epochs, twelve local stacks, CPU |
| `../../engine/training/run_eleven_runs.sh` | the sbatch array (two submissions: 40 GB and 80 GB A100s; `SMOKE=1` for the 50-lake smoke job) |
| `../../engine/training/run_band_stats_crossyear.sh` | CPU job that writes the training-split band statistics R3 and R10 need |
| `../../engine/eval/score_tbs40.py` | scores a predictions table on the 40 Terminal-Bench lakes |
| `diagnosis_numbers.py` | reproduces the diagnosis numbers from published tables and local copies: the 679-lake confusion, the Terminal-Bench 40 (unanimity 27/40, labeler and model scores), the combined model's split leakage, missing-day statistics |

## The runs, in one line each

| run | change |
|---|---|
| R0 | reference: the ESSD model at 32×32 (upsample removed, main's default) |
| R1 | attention over time, max beside mean over space |
| R2 | GroupNorm, forget-gate bias 1, warm-up + cosine |
| R3 | pixels: per-band standardisation (train split), mean-fill, validity channel |
| R4 | `p_water` observed flag as a second scalar stream |
| R5 | static outline + daily NDWI mask as input channels |
| R6 | spatial CBAM on the feature map |
| R7 | + NIR + SWIR 1610 nm |
| R8 | soft targets from `label_probability` |
| R9 | capacity: base 16, hidden 64 (expected null; the run that died twice) |
| R10 | the stack: R1–R8 together |

Memory: at batch 8 R9 and R10 exceed a 40 GB A100 and R2 is marginal, so those three
go up as a second array on 80 GB A100s with both features in the constraint,
`-C "GPU_SKU:A100_SXM4&GPU_MEM:80GB"` (`GPU_MEM:80GB` alone lands on an H100 that
`py-pytorch/2.2.1` cannot drive). serc has 40 such GPUs on six nodes (sh03-17n01/03/05/07, sh03-18n11/16) and
one H100 node (sh04-09n01). Fallback if none are free:
`--batch_size 4 --accumulation_steps 2` (no BatchNorm, so the gradient is
identical). Every run passes `--host_mem_budget_gb 200`.

Protocol: the committed cross-year split, selection on val macro-F1, test scored
once, the 40 Terminal-Bench lakes scored too, ±0.03–0.05 treated as a tie, train
macro-F1 logged every epoch (the gate question: does any run clear 0.85?).

## Where the reasoning came from

`docs/PROVENANCE_ESSD.md`, `docs/CV_GRID.md`, and the August diaries
(`claudiary/20260811T.md` §0b/§6, `20260814F.md` §10–§12, `20260817M.md`,
`20260830U.md`). The proposal builds on those; it does not re-derive them.

## Rules

- Nothing runs on Sherlock without Josh's explicit OK for that submission.
- Every change flag-gated so the `essd-2026-submission` and `essd-training-runs`
  tags reproduce bit for bit.
- Train on the deposit stacks (`stacks_v2`), not the legacy composites.
