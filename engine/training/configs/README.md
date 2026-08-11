# ESSD training configs

**Status (2026-04-12): STAGED, NOT YET WIRED.**

This directory holds YAML files that define the canonical ESSD training
configurations. As of the ESSD planning session on 2026-04-12, the SLURM
scripts (`run_training_essd_*.sh`) still pass every flag on the command
line. This design is intentional for today — we want to get a passing
pilot run on Sherlock before introducing a new config-loading variable
into the debugging surface.

## Why YAML at all?

For ESSD paper reproducibility, a committed `essd_baseline.yaml` is a
cleaner citation than "see line 127 of the SLURM script." It also makes
the train/val/test ratios and model hyperparameters a machine-readable
artifact that travels with the paper.

## Intended hierarchy

```
essd_baseline.yaml              # canonical: label scheme, model, optimizer
├── essd_combined.yaml          # 70/20/10 split
├── essd_crossyear.yaml         # 80/20 train/val, held-out test
├── essd_learning_curve.yaml    # nested stratified subsets
└── essd_pilot.yaml             # smoke test (3 epochs, 50 lakes)
```

Experiment configs set `base_config: essd_baseline.yaml` and add the
fields that vary per experiment. Paths (`labels_csv`, `nc_dir`,
`test_labels_csv`, `*_ids_file`, `save_path`) stay on the CLI because
they are environment-specific (laptop vs Sherlock) and should not be
baked into a committed YAML.

## Wiring work (deferred until pilot passes)

When we're ready to wire this in:

1. Add `pyyaml` to dependencies if not already present.
2. Add a small `load_config(path)` helper to `run_training.py` that:
   - Reads the YAML
   - Recursively resolves `base_config:` references (so
     `essd_combined.yaml` pulls in `essd_baseline.yaml`)
   - Returns a dict that argparse can merge with explicit CLI values
     (CLI should override YAML so per-job overrides still work)
3. Add `--config PATH` to the argparser.
4. Update the four `run_training_essd_*.sh` scripts to pass
   `--config configs/essd_<variant>.yaml` and drop the
   hyperparameter-defining flags.
5. Keep every CLI flag functional so ad-hoc runs without a config
   still work.

This is a ~30-minute refactor once the pilot validates end-to-end. Until
then, the YAMLs here are a **design document**, not a runtime artifact.

## Canonical config quick reference

For the current period before wiring, if you need to know what the
canonical config *should* look like, read [essd_baseline.yaml](essd_baseline.yaml).
Anything different in the SLURM scripts is a bug and should be brought
into line with the YAML.
