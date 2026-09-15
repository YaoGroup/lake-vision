"""
Combined-year split with the 40 Terminal-Bench lakes held out.

Starts from the paper's committed combined split (splits/essd_CW: seed 42,
stratified 70/20/10 over all 1,679 CW2018+CW2019 lakes) and moves every
bench lake found in train or val into test. Nothing else changes, so the
non-bench lakes keep the split the paper used, and a model trained on this
split can still be scored on the bench with engine/eval/score_tbs40.py.

    python engine/training/split_bench_heldout.py \
        --src splits/essd_CW --out splits/essd_CW_benchout \
        --tbs ~/stanford_gp/research/lakes/2026/terminal-bench/terminal-bench-science/tasks/earth-sciences/geosciences/supraglacial-lake-classification

Writes train_ids.json, val_ids.json, test_ids.json and split_meta.json
(which lists the lakes that moved). Deterministic; commit the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEF_TBS = Path.home() / "stanford_gp/research/lakes/2026/terminal-bench/terminal-bench-science/tasks/earth-sciences/geosciences/supraglacial-lake-classification"


def bench_lake_ids(tbs: Path) -> list[str]:
    mapping = pd.read_csv(tbs / "authoring/provenance/id_mapping.csv", dtype=str).set_index("anon_id")["original_lake_id"]
    key = pd.read_csv(tbs / "tests/labels_key.csv", dtype=str)
    ids = key.iloc[:, 0].map(mapping)
    if ids.isna().any():
        raise SystemExit("id mapping incomplete")
    return sorted(set(ids))


def hold_out(splits: dict[str, list[str]], held: list[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Move every id in `held` from train/val into test. Returns (new_splits, moved)."""
    held_set = set(held)
    moved = {"train": [], "val": []}
    out = {}
    for name in ("train", "val"):
        keep, gone = [], []
        for lid in splits[name]:
            (gone if lid in held_set else keep).append(lid)
        out[name] = keep
        moved[name] = gone
    out["test"] = list(splits["test"]) + moved["train"] + moved["val"]
    missing = held_set - set(out["test"])
    if missing:
        raise SystemExit(f"{len(missing)} held-out ids are not in any split: {sorted(missing)[:5]}")
    assert len(set(out["test"])) == len(out["test"]), "duplicate in test"
    assert not (set(out["train"]) | set(out["val"])) & held_set
    assert sum(map(len, out.values())) == sum(map(len, splits.values())), "a lake was lost"
    return out, moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="splits/essd_CW")
    ap.add_argument("--out", default="splits/essd_CW_benchout")
    ap.add_argument("--tbs", default=str(DEF_TBS))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    splits = {n: json.load(open(src / f"{n}_ids.json")) for n in ("train", "val", "test")}
    held = bench_lake_ids(Path(a.tbs))
    new, moved = hold_out(splits, held)
    out.mkdir(parents=True, exist_ok=True)
    for n in ("train", "val", "test"):
        json.dump(new[n], open(out / f"{n}_ids.json", "w"))
    src_meta = json.load(open(src / "split_meta.json")) if (src / "split_meta.json").exists() else {}
    meta = {
        "derived_from": str(src), "source_meta": src_meta,
        "held_out": "the 40 Terminal-Bench lakes, all placed in test",
        "held_out_ids": held,
        "moved_from_train": moved["train"], "moved_from_val": moved["val"],
        "already_in_test": sorted(set(held) & set(splits["test"])),
        "sizes": {n: len(new[n]) for n in ("train", "val", "test")},
    }
    json.dump(meta, open(out / "split_meta.json", "w"), indent=2)
    print(f"{src} -> {out}: train {len(new['train'])} val {len(new['val'])} test {len(new['test'])}; "
          f"moved {len(moved['train'])} from train, {len(moved['val'])} from val, "
          f"{len(meta['already_in_test'])} already in test")


if __name__ == "__main__":
    main()
