# The planned cross-validation grid

This is the workload the training pipeline is being optimized for. It exists so
that pipeline work (data staging, caching, job layout) can be designed against
the *grid's* shape — roughly a hundred short-ish runs sharing one dataset —
rather than against a single long training run, which is a different problem.

Status: design agreed August 2026, not yet launched. The scientific motivation
in one line: the model **underfits** (train macro-F1 ceilings at ~0.71 after 400
epochs — it cannot fit data it has already seen), and the leading suspects are
the input representation and the readout, not model capacity.

---

## 1. What one run is

- Train `LakeDrainageClassifier` (`engine/training/run_training.py`) on a fold
  of the combined CW 2018+2019 pool: **1,679 lakes**, 5 classes (ND/HF/MD/LD/CD),
  one sample = a full melt season `[T=153, C, 512, 512]` plus scalar sequences.
- **5-fold stratified CV**, stratified by the 5-class label. Folds are supplied
  as JSON ID lists via `--train_ids_file/--val_ids_file/--test_ids_file`, which
  already exist — no splitting logic needs to change, a small script just writes
  the fold files once.
- Model selection on **val macro-F1**; report mean ± std across folds. Fold
  spread is the entire reason for k-fold: single-split noise was estimated at
  **±0.03–0.05 macro-F1**, larger than many effects we want to detect.
- Class weights recompute from the training set automatically, so per-fold
  reweighting is free.
- The published ESSD splits (`splits/essd_CW*`) stay untouched and are used only
  for the final headline numbers, so results remain comparable to the paper.

## 2. Scale

Full crossing of every axis below is ~540 runs — deliberately **not** the plan.
The plan is staged, one-factor-at-a-time from a baseline, promoting winners:

1. **Stage A — pilot, ~5 cells × 1 split.** Confirms each axis actually moves
   the number and measures real per-run wall-clock for sizing Stage B.
2. **Stage B — surviving axes × 5 folds.** Expect **~15–30 configs × 5 folds ≈
   75–150 runs**.

Per-run epoch budget is an open decision. The ESSD baselines ran 400 epochs and
were still improving at walltime death; the sweep will likely run 100–200 epochs
per trial and reserve long runs for finalists. For sizing: the legacy pipeline
measured **~14.5 min/epoch** at N≈1,175 (~99% of wall-clock in the dataloader) —
at that speed even 100 runs × 150 epochs is ~15 GPU-weeks, which is the number
the pipeline optimization needs to attack.

## 3. The axes

Ordered by expected effect size (from the August 2026 audit, where each claim
was probed by measurement). **Status** says where the knob exists today:
`main` = usable now; `branch` = implemented on `feature/cache-data` (entangled
with the abandoned cache pipeline, to be re-landed on main as clean commits);
`todo` = not implemented anywhere.

| # | axis | values | status | notes |
|---|------|--------|--------|-------|
| 1 | **input representation** | raw ÷10000 (today) / per-band standardized + mean-fill + validity channel | `todo` (`--band_stats` plumbing exists on main; no stats file ever computed; validity channel unimplemented) | Believed highest-leverage. Valid pixels sit at mean≈0.87, std≈0.12 with ~41% zero-filled NaNs, so the strongest edge in every image is the data/nodata boundary, not water/ice. Validity convention: **1 = valid, 0 = NaN**, per-timestep. |
| 2 | **temporal readout** | last / mean / max over T | `branch` | 'last' (today) makes a mid-season drainage signal survive ~100 recurrent steps to reach the classifier. Costs no parameters. Also the cheap in-family test of the "transformer would help" hypothesis. |
| 3 | **spatial pooling** | avg / both (avg‖max) | model supports both on `main`; CLI flag `branch` | All image evidence for a lake currently collapses to 32 numbers via global *average* — near-worst-case for a localized transient feature. 'both' costs ~1% more parameters. |
| 4 | **normalization** | none / GroupNorm / BatchNorm in FrontCNN | `branch` | There is currently **no normalization anywhere in the model**. Genuinely undecided which is right: BN pools statistics across B·T mixed frames; GN erases per-image brightness which may itself be signal. Hence an axis, not a decision. |
| 5 | **mask** | none / static (Dunmire `lake_boundary`) / dynamic (`water_mask_ndwi`) — as an input channel | `branch` (`--mask_as_channel`) | ⚠️ On main the mask channel is loaded, carried through the dataloader, and **discarded** unless `attention_type='arch'` (measured: inverting it changes logits by exactly 0.0). The ESSD baseline never used it. Without the branch wiring this axis is inert in 3 of 4 attention cells — and the dead 4th channel is pure dataloader cost in default configs, which matters for profiling. |
| 6 | **attention** | none / spatial CBAM / full CBAM / architectural | `main` (`--attention_type`) | Interacts with the mask axis (arch consumes the mask its own way); scheduled late for that reason. |
| 7 | **sequences** | imagery only / p_water only / both / + cloudy flag | `main` (`--use_imgseq/--use_areaseq/--use_cloudyseq`) | See §5 for the cloudy-flag data trap. |
| 8 | **spectral bands** | RGB / +NIR / +SWIR | `main` has `--use_nir`; SWIR16/22 need wiring | On the branch, band flags are *derived* from the requested band list with a channel-count assert, because extra channels used to be silently sliced off (a "6-band" run could quietly train on 3). Worth re-landing before any band axis runs. |
| 9 | **CLSTM forget-gate bias** | 0.0 / 1.0 | `branch` | init 0.0 opens the gate at 0.5, decaying memory ~0.5/step over T=153 early in training. Standard fix (Jozefowicz 2015), one line. |
| 10 | **capacity** | `clstm_hidden` 32 / 64; FrontCNN `base_channels`, `num_layers` | `main` | **Deliberately last**: the symptom is underfitting, and at 140k params / 1,679 lakes the model is nowhere near the overfitting regime. Guardrail: watch the train–val *gap*, not val alone. |
| 11 | **optimization hyperparams** | lr; weight_decay; dropout; optimizer adam/adamw/sgd; epochs | lr/wd/dropout/epochs `main`; optimizer choice `branch` (main hardcodes Adam) | Tune lr *at* the chosen fixed batch size (see §4). |

## 4. Explicitly NOT axes

These are engineering knobs. Fix them per-machine and never grid over them:

- **batch_size** — changes effective learning rate, so it confounds any axis it
  rides along with. The pipeline optimizer should pick whatever is fastest,
  **fix it for the whole grid**, and lr gets tuned at that batch size.
  (Reference point from the abandoned branch: bs=32 fit a 40 GB A100 only with
  chunked+checkpointed FrontCNN and chunk-wise normalization; on main's code
  bs=8 is the known-safe setting at ~28 GB.)
- **num_workers / prefetch / pinning** — free for the optimizer. Measured on the
  branch, for whatever it is worth here: worker throughput saturated at ~12
  workers (24 was *slower*), and `pin_memory=False` was 1.6–1.9× on loader-only
  (full-step comparison never run).
- **`--deterministic` / cudnn benchmark** — off during the sweep (seeds still
  fixed; only kernel selection varies), on for final headline runs.
- **augmentation mode** — random-D4 per epoch is the default; `--augment_mode
  expand` exists only to reproduce ESSD's deterministic 8× variant.

## 5. Data-source matrix — the one decision that shapes staging

Two on-disk sources exist, and **neither currently has everything the grid
needs**. This is probably the first thing to resolve, because it decides what
gets staged/cached:

| | `composites/` (v1-derived) | `stacks_v2/` (published ESSD deposit, DOI 10.25740/sf350xp4038) |
|---|---|---|
| imagery | 7-channel `imagery` (RGB, NIR, SWIR16, SCL cloudmask, static mask) | 6-band `reflectance` (adds SWIR22/B12) |
| static lake mask | ✓ (baked in as channel 7) | ✓ (`lake_boundary`) |
| dynamic NDWI mask | ✗ **dropped** | ✓ (`water_mask_ndwi`) |
| cloudy flag `cloudy_seq_rgb` | ✓ | ✗ **absent** |
| area scalar | `water_area` (km²) | `p_water` (fraction) — different semantics, an explicitly open comparison |
| provenance | internal intermediate | **reproducible from the public DOI** |
| ESSD baselines trained on | **this** | — |

⚠️ **Cloudy-flag trap:** `cloudy_seq_rgb` is the cloudy-tile CNN's output,
**1 = useful, 0 = cloudy-or-nodata**, per tile per timestep. The v2 stacks
instead carry `eo_cloud_cover` — a *scene-level* ESA percentage with **inverted
polarity** — and `pct_nans`. Neither is an acceptable substitute for the cloudy
axis; substituting one silently was a live bug on the branch and is guarded
against there. If v2 is the staging source, `cloudy_seq_rgb` must be brought
over from the composites (verified bit-identical pixels and time grids, so a
transplant is legitimate — working code for it is on the branch) or re-inferred.

## 6. Sherlock execution constraints (as designed, for whatever they're worth)

- Compute nodes have **no outbound network**: `WANDB_MODE=offline` + `wandb
  sync` afterwards. `wandb agent`-style sweeps cannot work; drive the grid as a
  **SLURM array over a deterministic config list**, one cell per task.
- Grouping needs `--wandb_group/--wandb_tags/--fold_idx` so folds aggregate —
  currently branch-only.
- Checkpoint provenance: the branch saves `{state_dict, config, fold, seed,
  git_sha, ...}` instead of bare state_dicts, because a grid otherwise produces
  dozens of anonymous `.pth` files. Worth re-landing (a dual-format loader on
  the branch keeps old ESSD checkpoints readable).
- Shared data staged **once** to a shared filesystem, per-task copy to
  node-local — per-job `$L_SCRATCH` rebuilds don't amortize across an array.
  Co-locating folds of the same config on one node shares page cache.

## 7. Where the branch work lives

`feature/cache-data` (frozen record, do not merge wholesale) contains working,
tested implementations of every `branch`-status item above, entangled with the
abandoned blosc2 cache pipeline. The intended path is re-landing them on main as
clean commits decoupled from the cache. All are flag-gated with defaults that
reproduce the ESSD tags bit-for-bit, with ~100 tests covering them.
