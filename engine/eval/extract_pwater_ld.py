"""Pull p_water (and cloud/nan context) for the LD lakes that matter.

Groups: 2019 training LD lakes (what the model learned), 2018 test LD lakes
split by whether R10 called them LD or ND. Writes one tidy CSV.
"""
import json, sys, numpy as np, pandas as pd, netCDF4, time

ROOT = "/Users/jrines/stanford_gp/research/lakes/2026"
STACKS = "/Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_sattilestack/stacks_v2"
PRED = "/Volumes/groups/cyaolai/JoshRines/sherlock/sherlock_lakevision/inference_essd/eleven/R10_test_predictions_bestf1.csv"
OUT = sys.argv[1]

lab = {y: pd.read_csv(f"{ROOT}/essd/labels/labels_CW_{y}.csv", dtype=str).set_index("lake_id")["label"]
       for y in (2018, 2019)}
train = json.load(open(f"{ROOT}/lake-vision/splits/essd_CW_crossyear/train_ids.json"))
pred = pd.read_csv(PRED, dtype=str).set_index("lake_id")

jobs = []
for lid in train:
    if lab[2019].get(lid) == "LD":
        jobs.append((lid, 2019, "train2019_LD"))
for lid, row in pred.iterrows():
    if row.true_label == "LD":
        jobs.append((lid, 2018, f"test2018_LD_as_{row.pred_label}"))

print(f"{len(jobs)} lakes to read", flush=True)
rows, t0 = [], time.time()
for i, (lid, year, grp) in enumerate(jobs):
    try:
        with netCDF4.Dataset(f"{STACKS}/CW_{year}/{lid}.nc") as nc:
            nc.set_auto_mask(False)
            p = np.asarray(nc.variables["p_water"][:], float)
            cc = np.asarray(nc.variables["eo_cloud_cover"][:], float)
            pn = np.asarray(nc.variables["pct_nans"][:], float)
    except Exception as e:
        print(f"  skip {lid}: {e}", flush=True); continue
    rows.append(dict(lake_id=lid, year=year, group=grp,
                     p_water=",".join(f"{v:.5f}" for v in p),
                     cloud=",".join(f"{v:.1f}" for v in cc),
                     nans=",".join(f"{v:.1f}" for v in pn)))
    if (i + 1) % 50 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(jobs)}, {el:.0f}s, ~{el/(i+1)*(len(jobs)-i-1):.0f}s left", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"wrote {len(rows)} rows to {OUT}", flush=True)
