# Cross-validation grid (planned)

The workload the training pipeline should be optimized for: ~75–150 runs
sharing one dataset. Design agreed August 2026, not yet launched.

Context: the model underfits (train macro-F1 ~0.71 after 400 epochs). Axis
priority below follows the August 2026 audit; Stage A exists to test that
ordering, not assume it.

## Protocol

- 5-fold stratified CV on the combined CW 2018+2019 pool (1,679 lakes,
  5 classes). Folds are JSON ID lists via `--train_ids_file/--val_ids_file/
  --test_ids_file` (already supported; a script writes fold files once).
- Model selection on val macro-F1. Report mean ± std across folds.
  Single-split noise is ±0.03–0.05 macro-F1 — that is why k-fold.
- Class weights recompute per training set (per-fold reweighting is free).
- Published ESSD splits (`splits/essd_CW*`) untouched; used only for headline
  runs.
- Staged sweep, not full crossing (~540 runs). Stage A: ~5-cell pilot on one
  split to confirm axes move the number and measure wall-clock. Stage B:
  surviving axes × 5 folds.
- Per-run epoch budget TBD (ESSD ran 400 and was still improving at walltime;
  sweep likely 100–200, long runs for finalists).
- Sizing datum: legacy pipeline ≈14.5 min/epoch at N≈1,175, ~99% of wall-clock
  in the dataloader.

## Axes

Status: `main` = usable now. `branch` = implemented on `feature/cache-data`
only (re-land on main before use). **`NO CODE`** = not implemented anywhere.

| # | axis | values | status |
|---|------|--------|--------|
| 1 | input representation | raw ÷10000 / per-band standardized + mean-fill + validity channel (1=valid, 0=NaN) | **NO CODE** — `--band_stats` plumbing exists on main but no stats file has ever been computed; validity channel unimplemented |
| 2 | temporal readout | last / mean / max over T | `branch` (`--temporal_readout`) |
| 3 | spatial pooling | avg / both (avg‖max) | model on `main`; CLI flag `branch` (`--pool_type`) |
| 4 | FrontCNN normalization | none / GroupNorm / BatchNorm | `branch` (`--frontcnn_norm`). Model currently has no normalization anywhere |
| 5 | mask as input channel | none / static `lake_boundary` / dynamic `water_mask_ndwi` | `branch` (`--mask_as_channel`). On main the mask channel is loaded then discarded unless `attention_type=arch` (measured: zero logit effect) — dead dataloader weight in default configs |
| 6 | attention | none / spatial CBAM / full CBAM / arch | `main` (`--attention_type`) |
| 7 | sequences | imagery / p_water / both | `main` (`--use_imgseq/--use_areaseq`) |
| 8 | cloudy flag | off / `cloudy_seq_rgb` (current CNN) / re-trained CNN | `main` has `--use_cloudyseq`; **re-labeling and re-training of the cloudy-tile CNN is in progress**; the flag must first be added to the canonical data (see Data) |
| 9 | spectral bands | RGB / +NIR / +SWIR16 / +SWIR22 | `--use_nir` on `main`; SWIR16/22 not wired. `branch` derives band flags from the band list + asserts channel count (extra channels used to be silently dropped — re-land before running this axis) |
| 10 | CLSTM forget-gate bias | 0.0 / 1.0 | `branch` (`--clstm_forget_bias`) |
| 11 | capacity | `clstm_hidden` 32/64; `frontcnn_base_channels`; `frontcnn_num_layers` | `main`. Last in priority: symptom is underfitting; guardrail is the train–val gap |
| 12 | batch size | 8 / 16 / 32 | `main` (`--batch_size`). bs=8 ≈28 GB VRAM on main's code; bs=32 OOMs a 40 GB A100 without the branch memory work — pipeline optimization decides what is reachable. Re-tune lr per batch size (effective-lr coupling) |
| 13 | hyperparameters | lr, weight_decay, dropout, epochs / optimizer adam-adamw-sgd | lr/wd/dropout/epochs `main`; optimizer choice `branch` (main hardcodes Adam) |

## Fixed knobs (not axes)

- `num_workers` / prefetch / `pin_memory` — pipeline's choice. Branch
  measurements if useful: throughput saturated at 12 workers (24 slower);
  `pin_memory=False` 1.6–1.9× loader-only, full-step never compared.
- `--deterministic` off during the sweep (seeds fixed, kernel selection varies);
  on for headline runs.
- Augmentation: random-D4 default; `--augment_mode expand` only to reproduce
  ESSD.

## Data

**Canonical source: the ESSD submission data = SDR deposit
(DOI 10.25740/sf350xp4038), on OAK at
`/oak/.../sherlock_sattilestack/stacks_v2/`.** Anything the grid needs that it
lacks gets added to it, not sourced elsewhere.

- Has: 6-band `reflectance`, `lake_boundary`, `water_mask_ndwi`, `p_water`,
  labels.
- Missing: `cloudy_seq_*`. Add it — either transplant from `composites/`
  (verified bit-identical pixels/time grid; working code on the branch) or
  re-infer; the re-trained CNN replaces it when ready.
- `eo_cloud_cover` and `pct_nans` are NOT substitutes for the cloudy flag:
  scene-level, and inverted polarity (`cloudy_seq` is 1=useful).
- `composites/` is a legacy intermediate (what the ESSD baselines trained on).
  Do not build new work on it.

## Sherlock

- Compute nodes have no outbound network: `WANDB_MODE=offline` + `wandb sync`.
  Drive the grid as a SLURM array over a deterministic config list.
- `--wandb_group/--wandb_tags/--fold_idx` and provenance checkpoints
  (`{state_dict, config, fold, seed, git_sha, ...}` + dual-format loader):
  `branch` only. Re-land before the grid, or runs are unattributable.
- Stage shared data once to a shared filesystem; per-task copy to node-local.

## Branch

`feature/cache-data` is a frozen record — do not merge wholesale. Every
`branch` item above is implemented there, flag-gated, tested (~100 tests),
with defaults that reproduce the ESSD tags bit-for-bit. Re-land as clean
commits decoupled from the abandoned cache pipeline.
