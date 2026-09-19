"""Cross-year spatial leakage between the CW2018 and CW2019 lake populations.

The published cross-year protocol trains on CW2019 and tests on all of CW2018.
This quantifies how far that is from a held-out test set: Greenland supraglacial
lakes reform in the same bedrock-controlled basins year after year, so a CW2018
"test" lake is frequently the same site as a CW2019 training lake.

Numbers quoted in the ESSD response letter come from here. Run it rather than
recomputing by hand -- an earlier ad hoc pass reported label agreement as 49%,
which is wrong (it is 62%).

    python3 engine/eval/crossyear_leakage.py --root /path/to/lakes/2026

Geometry is the Dunmire et al. maximum-extent polygon set (CRS84); centroids are
area-weighted (shoelace), and distances use a local equirectangular projection
at the population mean latitude, which is accurate to well under a metre over
the few-kilometre separations that matter here.
"""
import argparse, json, pathlib
import numpy as np
import pandas as pd

YEARS = (2018, 2019)


def poly_centroid(ring):
    p = np.asarray(ring, float)[:, :2]
    if not np.allclose(p[0], p[-1]):
        p = np.vstack([p, p[0]])
    x, y = p[:, 0], p[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2.0
    if abs(area) < 1e-15:                       # degenerate sliver
        return p[:-1].mean(axis=0)
    return np.array([((x[:-1] + x[1:]) * cross).sum(),
                     ((y[:-1] + y[1:]) * cross).sum()]) / (6.0 * area)


def centroids(root, year, region="CW"):
    d = json.loads((root / f"labels/dunmire/labels_{year}_volumes.geojson").read_text())
    out = {}
    for f in d["features"]:
        props = f["properties"]
        if props["region"] != region:
            continue
        geom = f["geometry"]
        ring = (geom["coordinates"][0] if geom["type"] == "Polygon"
                else max((poly[0] for poly in geom["coordinates"]), key=len))
        out[props["new_id"]] = poly_centroid(ring)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[3],
                    help="the 2026/ directory holding labels/, essd/ and lake-vision/")
    ap.add_argument("--train_ids", type=pathlib.Path, default=None,
                    help="cross-year train_ids.json, for the train-restricted variant")
    args = ap.parse_args()
    root = args.root
    train_ids = args.train_ids or root / "lake-vision/splits/essd_CW_crossyear/train_ids.json"

    labels = {y: pd.read_csv(root / f"essd/labels/labels_CW_{y}.csv")
                  .set_index("lake_id")["label"].to_dict() for y in YEARS}
    cen = {y: centroids(root, y) for y in YEARS}
    ids = {y: [i for i in labels[y] if i in cen[y]] for y in YEARS}
    for y in YEARS:
        print(f"CW{y}: {len(labels[y])} labelled, {len(cen[y])} geometries, {len(ids[y])} joined")

    ll = {y: np.array([cen[y][i] for i in ids[y]]) for y in YEARS}
    lat0 = np.deg2rad(np.concatenate([ll[2018][:, 1], ll[2019][:, 1]]).mean())
    to_m = lambda a: np.c_[a[:, 0] * 111320 * np.cos(lat0), a[:, 1] * 110540]
    D = np.linalg.norm(to_m(ll[2018])[:, None, :] - to_m(ll[2019])[None, :, :], axis=2)

    def report(D_sub, pool, title):
        near = D_sub.argmin(axis=1)
        dist = D_sub[np.arange(D_sub.shape[0]), near]
        same = np.array([labels[2018][ids[2018][k]] == labels[2019][pool[near[k]]]
                         for k in range(len(dist))])
        print(f"\n{title}")
        print(f"  median distance    {np.median(dist):6.0f} m")
        for cut in (100, 500):
            m = dist < cut
            print(f"  within {cut:3d} m       {100 * m.mean():6.1f} %  (n={m.sum()})"
                  f"   label agreement {100 * same[m].mean():.1f} %")
        print(f"  label agreement    {100 * same.mean():6.1f} %  (all pairs)")
        return dist, same

    report(D, ids[2019], f"Nearest of all {len(ids[2019])} CW2019 lakes "
                         f"to each of the {len(ids[2018])} CW2018 lakes")

    p18 = pd.Series([labels[2018][i] for i in ids[2018]]).value_counts(normalize=True)
    p19 = pd.Series([labels[2019][i] for i in ids[2019]]).value_counts(normalize=True)
    chance = sum(p18.get(c, 0) * p19.get(c, 0) for c in set(p18.index) | set(p19.index))
    print(f"  chance agreement   {100 * chance:6.1f} %  (independent draws from the two priors)")

    tr = set(json.loads(train_ids.read_text()))
    keep = [n for n, i in enumerate(ids[2019]) if i in tr]
    report(D[:, keep], [ids[2019][k] for k in keep],
           f"Restricted to the {len(keep)} cross-year TRAIN lakes")


if __name__ == "__main__":
    main()
