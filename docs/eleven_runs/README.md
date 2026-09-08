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
