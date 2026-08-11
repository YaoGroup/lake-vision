# ESSD paper provenance

Reproducibility record for the ESSD dataset paper (under review as of 2026-08).
Written during the 2026-08-11 branch consolidation, when `feature/data-augment`
and `feature/reclass` were merged into `main` and deleted.

**Nothing was lost in that consolidation.** Every commit behind every ESSD result
remains an ancestor of `main`. The tags below exist so you can get back to the
exact state in one command without archaeology.

## Tags

| Tag | Commit | What it is |
|-----|--------|------------|
| `essd-2026-submission` | `c50bbba` | Full repo state behind the submitted paper, including the inference runs that produced the figures. |
| `essd-training-runs` | `6bb8962` | The commit that produced the trained checkpoints (2026-04-23). |

```bash
git checkout essd-2026-submission     # exact submission state
git diff essd-2026-submission main    # everything that changed since
```

Two files exist at `essd-2026-submission` but not on `main`:

- `CLAUDE.md` — described the superseded 4-class ND/ED/LD/CD scheme, not the
  5-class ESSD schema. Removed as stale.
- `engine/training/run_training_essd_{combined,crossyear,learning_curve}_imageryonly.sh`
  — an imagery-only ablation (`--no_areaseq --no_cloudyseq --no_mask`). Removed
  because they used `--train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1` rather
  than the committed split files, so their numbers are **not** comparable to any
  other ESSD run. `6bb8962` fixed this for the combined scripts but not these.
  Recover with `git show essd-2026-submission:engine/training/<name>.sh`.

## Data deposit

Stanford Digital Repository, DOI **10.25740/sf350xp4038**. Local source of truth
at `essd/essd_sdr/`; OAK copy at `data/essd_sdr/`. Pre-flight verification of the
deposit (file counts, label/ID coverage, variable schema) is recorded in
`essd/claudiary/20260522F.md`.

## Dataset

1679 labeled lakes: 679 in CW2018, 1000 in CW2019. 5-class schema
ND / HF / MD / LD / CD (indices 0-4).

Canonical splits are committed at `splits/` and are also in the deposit. Both use
`seed=42` and nested stratified train ordering (so the N=400 learning-curve train
set is a strict superset of N=200).

| Split | Train | Val | Test | Notes |
|-------|-------|-----|------|-------|
| `splits/essd_CW/` | 1175 | 336 | 168 | Combined 2018+2019, 70/20/10 |
| `splits/essd_CW_crossyear/` | 800 | 200 | 679 | Train+val CW2019, test = all CW2018 |

Parent class distribution (combined): ND 236, HF 279, MD 242, LD 580, CD 342.

## Results and the commit behind each

All runs: 5-class, 400 epochs, bs=8, bf16 AMP, lr=1e-4 fixed (no scheduler),
imagery + water_area + static Dunmire polygon mask, no cloudy_seq, attention off,
4-layer FrontCNN, seed 42. Defaults are hardcoded in `engine/training/run_training.py`
argparse — see the header comment there.

### Cross-year baseline — `run_training_essd_crossyear.sh`

Best val F1 **0.6203**, test F1 **0.4454**. Per-class test F1: ND 0.459, HF 0.471,
MD 0.059, LD 0.536, CD 0.703. Completed before 2026-04-23; the script reached its
final form in the `74151af`..`cf2a399` range (2026-04-17 to 04-19).

### Learning curve — `run_training_essd_learning_curve.sh`

Per-epoch curves preserved at `essd/inference/training_metrics/metrics_N*.json`.

| N train | best val F1 | at epoch | epochs run |
|---------|-------------|----------|------------|
| 200 | 0.498 | 389 | 400 |
| 400 | 0.529 | 399 | 400 |
| 600 | 0.567 | 327 | 400 |
| 800 | 0.568 | 357 | 400 |
| 1000 | 0.564 | 173 | 334 (walltime) |
| 1175 | 0.607 | 334 | 398 |

### SLURM jobs fired 2026-04-23 (from `claudiary/20260423R.md`)

| Job ID | Experiment | Script |
|--------|-----------|--------|
| 22429907 | Combined baseline N=1175 | `run_training_essd_combined.sh` |
| 22429908 | Combined + 8x D4 augment | `run_training_essd_combined_augment.sh` |
| 22429909 | Combined highcap + 8x augment | `run_training_essd_combined_highcap_augment.sh` |
| 22430000_4 | Learning curve N=1000 | `run_training_essd_learning_curve.sh` |

`6bb8962` switched all combined scripts from ratio-based splitting to the
committed split files, making every run from that commit onward mutually
comparable and comparable to the learning curve. Runs before it are not.

### Dunmire et al. (2025) comparison

Mapping B (ND/ED/LD, CD excluded, n=162 val lakes): macro F1 0.658 vs 0.611,
accuracy 0.673 vs 0.611, Cohen's kappa 0.486 vs 0.423. Label mapping
`{0: refreeze, 1: rapid, 2: slow, 3: buried}`, verified against drain_date
patterns in the source GeoJSON. Notebook: `essd/notebooks/fig_talk_dunmire_examples.ipynb`.

## Known caveat, recorded for review

The learning-curve metrics show train macro-F1 plateauing at 0.62-0.71 after 400
epochs, with best val F1 arriving at epoch 330-400 in most runs — i.e. the models
were still improving when walltime ended. **The ESSD baselines are compute-limited,
not converged.** This does not affect the dataset paper's claims (the baselines are
presented as reference points for a dataset paper, not as tuned models), but it
does mean the reported numbers are a floor rather than a ceiling. Follow-on tuning
is JSTARS work.

One documentation caveat inherited from the ESSD state: the docstring in
`lakevision/data/datasets.py:_load_from_disk` asserts the imagery variable is
HDF5-chunked along the channel axis with chunks `[51, 1, 171, 171]`. The sample
file at `datasets/processed/CW2019_1579.nc` is in fact contiguous and
uncompressed, while `lakevision/data/synthesis.py` writes composites with
`zlib complevel=4` and no explicit `chunksizes`. The on-OAK layout has not been
verified either way; treat that docstring as unconfirmed pending measurement.
