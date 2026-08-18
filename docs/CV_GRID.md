# Cross-validation grid (planned)

## Axes

Status: `main` = usable now. `branch` = implemented on `feature/cache-data`
only (re-land on main before use). **`NO CODE`** = not implemented anywhere.

| # | axis | values | status |
|---|------|--------|--------|
| 1 | input representation | raw ÷10000 / per-band standardized + mean-fill + validity channel (1=valid, 0=NaN) | **NO CODE** — `--band_stats` plumbing exists on main but no stats file has ever been computed; validity channel unimplemented |
| 2 | temporal readout | last / mean / max over T | `branch` (`--temporal_readout`) |
| 3 | spatial pooling | avg / both (avg‖max) | model on `main`; CLI flag `branch` (`--pool_type`) |
| 4 | FrontCNN normalization | none / GroupNorm / BatchNorm | `branch` (`--frontcnn_norm`). Model currently has no normalization anywhere |
| 5 | mask as input channel | none / static `lake_boundary` / dynamic `water_mask_ndwi` | `branch` (`--mask_as_channel`). On main the mask channel is loaded then discarded unless `attention_type=arch` (measured: zero logit effect) |
| 6 | attention | none / spatial CBAM / full CBAM / arch | `main` (`--attention_type`) |
| 7 | sequences | imagery / p_water / both | `main` (`--use_imgseq/--use_areaseq`) |
| 8 | cloudy flag | off / `cloudy_seq_rgb` (current CNN) / re-trained CNN | `main` has `--use_cloudyseq`; **re-labeling and re-training of the cloudy-tile CNN is in progress**; the flag must first be added to the canonical data (see Data) |
| 9 | spectral bands | RGB / +NIR / +SWIR16 / +SWIR22 | `--use_nir` on `main`; SWIR16/22 not wired. `branch` derives band flags from the band list + asserts channel count (extra channels used to be silently dropped — re-land before running this axis) |
| 10 | CLSTM forget-gate bias | 0.0 / 1.0 | `branch` (`--clstm_forget_bias`) |
| 11 | capacity | `clstm_hidden` 32/64; `frontcnn_base_channels`; `frontcnn_num_layers` | `main` |
| 12 | batch size | 8 / 16 / 32 | `main` (`--batch_size`). bs=8 ≈28 GB VRAM on main's code; bs=32 OOMs a 40 GB A100 without the branch memory work. Re-tune lr per batch size |
| 13 | hyperparameters | lr, weight_decay, dropout, epochs / optimizer adam-adamw-sgd | lr/wd/dropout/epochs `main`; optimizer choice `branch` (main hardcodes Adam) |

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
