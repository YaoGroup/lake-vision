# Archived Sherlock stashes

Two `git stash` entries lived on the Sherlock clone
(`/oak/stanford/groups/cyaolai/JoshRines/repos/lake-vision`) and existed
**nowhere else** — not on `main`, not on GitHub, not on any laptop. They are
preserved here as patches so that clone can be reset to a pristine checkout
without losing anything.

Recovered 2026-08-14. The Sherlock clone is now a run-target only; see
`engine/sherlock_preflight.sh`, which refuses to launch a job from a dirty
checkout.

These are **archived, not applied.** They are historical artifacts. Read them
before reviving anything — both contain changes that are wrong against the
current data.

---

## `2026-08-11_sherlock-local-edits.patch`

Stash commit `6818ac9`, "On feature/data-augment: sherlock local edits pre
cache-data". Touches `CLAUDE.md`, `README.md`, `lakevision/data/datasets.py`,
`lakevision/models/classifier.py`.

Three unrelated changes tangled into one stash:

1. **Copy elision in `LakeDataset.__getitem__`** — in-place `nan_to_num` and
   `torch.from_numpy` instead of `torch.tensor`, removing two full-sample
   (~640 MB) copies per read, plus a defensive `.copy()` so the `preload_to_ram`
   cache is not written through.
   → **Salvaged.** Landed on `feature/cache-data`, with the dtype hole fixed
   (a bare `from_numpy` returns float64 for a float64 source, where
   `torch.tensor(dtype=...)` had cast it) and four regression tests in
   `TestPreloadCacheIsNotMutated`.

2. **`swir22` → `cloudmask_scl` in channel slot 5**, and deletion of
   `'B12': 'swir22'` from `BAND_TO_CHANNEL`.
   → **Deliberately discarded — it is wrong against the current data.** The v2
   stacks carry `band_name = [B04, B03, B02, B08, B11, B12]`; B12 is present.
   Applying this would silently remove B12 access from the legacy path, which
   matters if the JSTARS sweep goes multi-spectral.
   The underlying observation is still true and worth keeping: the *older*
   preprocessed NC files (from `combine_lake_data`) wrote an SCL-derived cloud
   flag into slot 5 where the docs claimed SWIR22. That is a fact about legacy
   files, not about `stacks_v2`.

3. Doc edits describing (2), and removal of a per-dataset print.
   → Discarded with (2).

## `2026-03_preprocess-array-chunking.patch`

Stash commit `bb898fd`, "WIP on main: edc059f bug fix". Touches
`engine/preprocessing/append_labels.sh`, `engine/preprocessing/preprocess_tstacks.py`,
`engine/preprocessing/preprocess_tstacks.sh`.

Adds, none of which is on `main`:

- `--compress` on `append_labels_and_save` (zlib `complevel=4`, ~8× smaller)
- `--chunk_idx` / `--num_chunks` for SLURM array parallelism
- `#SBATCH --array=0-9` wiring, `NUM_CHUNKS=10`

**This code ran.** The OAK logs show a 10-way array job,
`append_labels_17689234_[0-9]`, on 2026-03-03 — matching `--array=0-9` and
`NUM_CHUNKS=10` exactly. So label-appended `.nc` files were produced by code
that was never committed.

Label columns are `label_rines, rinesID, p_nd, p_ed, p_ld, p_cd` — the **4-class**
scheme, predating the 5-class relabel. That is why it is archived rather than
merged: reviving it as-is would drag 4-class assumptions into a 5-class
codebase.

Whether any of this is on the ESSD critical path is **not established** — the
ESSD stacks come from `sat-tile-stack`, and the canonical 5-class label CSVs
live in `2026/essd/labels/`. Flagged rather than resolved, because the paper is
under review and the cost of keeping this is a 10 KB file.

---

## Applying one, if it ever comes to that

```
git apply --check docs/archive/sherlock-stashes/<file>.patch   # dry run first
```

Expect conflicts. Both predate the cache work; the first predates the deletion
of `feature/data-augment`.
