"""
One-shot stratified split generator for the ESSD learning-curve study.

Builds three JSON artifacts from one or more 5-class GUI labels CSVs:

  - val_ids.json     : frozen validation set (fraction of total)
  - test_ids.json    : frozen test set        (fraction of total)
  - train_ids.json   : ordered train set such that train_ids[:N] is a
                       stratified subset of the parent train pool for
                       every N. Used for nested learning-curve runs.

The stratified ordering uses a "water-filling" algorithm: at each step,
pick the class that is currently most underrepresented relative to its
target proportion and append one of its (shuffled) members. This produces
a single ordering that is as-close-to-stratified-as-possible at every
cutoff N simultaneously, so that N=200 ⊂ N=400 ⊂ ... ⊂ N_max in a
monotone, nested fashion.

Usage (locally):

    python engine/training/split_fixed.py \
        --labels_csv ../labels/CW_2018/labels_CW_2018.csv \
                     ../labels/CW_2019/labels_CW_2019.csv \
        --out_dir lake-vision/splits/essd_CW \
        --train_ratio 0.7 --val_ratio 0.2 --test_ratio 0.1 \
        --seed 42

Output files are small JSON lists of lake_id strings. Commit them to the
repo — they become the canonical splits for the paper.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Matches CLASS_NAMES_ESSD_5CLASS in run_training.py
ESSD_5CLASS_MAP = {'ND': 0, 'HF': 1, 'MD': 2, 'LD': 3, 'CD': 4}


def load_labels(csv_paths, id_col='lake_id', label_col='label'):
    """Load 5-class string labels from one or more GUI CSVs and map to ints."""
    combined = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[id_col, label_col])
        df = df[df[label_col].astype(str).str.strip() != ""]
        for lid, lbl in zip(df[id_col].astype(str), df[label_col].astype(str)):
            lbl = lbl.strip()
            if lbl not in ESSD_5CLASS_MAP:
                raise ValueError(
                    f"Unknown label '{lbl}' in {csv_path}. "
                    f"Expected one of {list(ESSD_5CLASS_MAP)}."
                )
            if lid in combined and combined[lid] != ESSD_5CLASS_MAP[lbl]:
                print(f"  WARNING: duplicate ID {lid} with conflicting labels; "
                      f"keeping first occurrence")
                continue
            combined[lid] = ESSD_5CLASS_MAP[lbl]
    return combined


def stratified_nested_order(ids, labels, seed=42):
    """Return a reordering of (ids, labels) such that ids[:N] is
    stratified for every N, built via water-filling.

    Args:
        ids: list of lake_id strings
        labels: list of int labels (same length as ids)
        seed: random seed for per-class shuffling

    Returns:
        ordered_ids: list of lake_id strings, same length as ids
    """
    rng = np.random.default_rng(seed)

    # Group IDs by class and shuffle each group
    by_class = defaultdict(list)
    for lid, lbl in zip(ids, labels):
        by_class[lbl].append(lid)
    for c in by_class:
        rng.shuffle(by_class[c])

    # Target proportions from the parent distribution
    total = len(ids)
    counts = {c: len(v) for c, v in by_class.items()}
    targets = {c: n / total for c, n in counts.items()}

    # Water-fill: at each step, pick the class whose (target * k - placed)
    # residual is largest, pop one member. Ties broken by class index for
    # determinism.
    placed = {c: 0 for c in by_class}
    ordered = []
    for k in range(1, total + 1):
        # Desired count for each class after placing the k-th item
        best_c, best_residual = None, -np.inf
        for c in by_class:
            if placed[c] >= counts[c]:
                continue  # this class is exhausted
            # residual = how many more items this class should have by step k
            residual = targets[c] * k - placed[c]
            if residual > best_residual or (
                residual == best_residual and (best_c is None or c < best_c)
            ):
                best_residual = residual
                best_c = c
        if best_c is None:
            raise RuntimeError("All classes exhausted before placing all ids")
        ordered.append(by_class[best_c].pop())
        placed[best_c] += 1

    return ordered


def verify_stratification(ordered_ids, id_to_label, n_values, parent_dist):
    """Print class proportions at each N to verify stratification quality."""
    print("\n--- Stratification check at various N ---")
    header = f"  {'N':>6}  " + "  ".join(
        f"{name:>6}" for name in ESSD_5CLASS_MAP
    )
    print(header)
    print(f"  {'parent':>6}  " + "  ".join(
        f"{parent_dist.get(i, 0.0):>6.1%}" for i in range(5)
    ))
    for n in n_values:
        if n > len(ordered_ids):
            continue
        subset = ordered_ids[:n]
        counts = Counter(id_to_label[lid] for lid in subset)
        print(f"  {n:>6}  " + "  ".join(
            f"{counts.get(i, 0) / n:>6.1%}" for i in range(5)
        ))


def main():
    ap = argparse.ArgumentParser(description="Generate fixed stratified splits for ESSD")
    ap.add_argument("--labels_csv", nargs='+', required=True,
                    help="One or more GUI labels CSVs (5-class schema). For the "
                         "combined mode, this is the full labeled pool. For the "
                         "crossyear mode, this is the train+val pool.")
    ap.add_argument("--test_labels_csv", nargs='+', default=None,
                    help="If set, enables crossyear mode: every lake in "
                         "these CSVs becomes the fixed test set, and "
                         "--labels_csv is split 80/20 into train/val.")
    ap.add_argument("--out_dir", required=True,
                    help="Output directory for train_ids.json, val_ids.json, test_ids.json")
    ap.add_argument("--id_col", default="lake_id")
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--train_ratio", type=float, default=0.7,
                    help="Train fraction. In crossyear mode this is the fraction "
                         "of --labels_csv used for train (default flips to 0.8).")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1,
                    help="Ignored in crossyear mode (test is the entirety of "
                         "--test_labels_csv).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check_n", nargs='+', type=int,
                    default=[200, 400, 600, 800, 1000],
                    help="Report stratification quality at these train-set sizes")
    args = ap.parse_args()

    crossyear = args.test_labels_csv is not None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # In crossyear mode, default to 80/20 if user didn't override
    if crossyear:
        # If user left the combined defaults (0.7/0.2/0.1), switch to 0.8/0.2
        if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 1e-6 \
                and args.test_ratio > 0:
            args.train_ratio = 0.8
            args.val_ratio = 0.2
            args.test_ratio = 0.0
        if abs(args.train_ratio + args.val_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"In crossyear mode train_ratio + val_ratio must equal 1.0 "
                f"(got {args.train_ratio} + {args.val_ratio})"
            )
    else:
        if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
            raise ValueError("ratios must sum to 1.0")

    # --- Load labels ---
    labels_dict = load_labels(args.labels_csv, args.id_col, args.label_col)
    print(f"Loaded {len(labels_dict)} labeled lakes from "
          f"{len(args.labels_csv)} CSV(s) [train+val pool].")
    total = len(labels_dict)
    parent_counts = Counter(labels_dict.values())
    parent_dist = {c: n / total for c, n in parent_counts.items()}
    print("\nParent distribution (train+val pool):")
    for i, name in enumerate(ESSD_5CLASS_MAP):
        print(f"  {name} ({i}): {parent_counts.get(i, 0):>5d}  ({parent_dist.get(i, 0.0):.1%})")

    test_labels_dict = None
    if crossyear:
        test_labels_dict = load_labels(args.test_labels_csv, args.id_col, args.label_col)
        # Drop any overlap between train+val and test (shouldn't happen for
        # disjoint-by-year CSVs, but defensive)
        overlap = set(labels_dict) & set(test_labels_dict)
        if overlap:
            print(f"\n  WARNING: {len(overlap)} lakes appear in both train+val "
                  f"and test pools; removing from train+val.")
            labels_dict = {k: v for k, v in labels_dict.items() if k not in overlap}
        test_counts = Counter(test_labels_dict.values())
        print(f"\nTest pool: {len(test_labels_dict)} lakes from "
              f"{len(args.test_labels_csv)} CSV(s).")
        print("Test distribution:")
        for i, name in enumerate(ESSD_5CLASS_MAP):
            n = test_counts.get(i, 0)
            frac = n / len(test_labels_dict) if test_labels_dict else 0.0
            print(f"  {name} ({i}): {n:>5d}  ({frac:.1%})")

    # --- Stratified split of the train+val pool ---
    ids = list(labels_dict.keys())
    labels = [labels_dict[lid] for lid in ids]

    if crossyear:
        # Single split: train vs val. No test carved out here.
        train_ids, val_ids, train_labels, _ = train_test_split(
            ids, labels,
            test_size=args.val_ratio,
            random_state=args.seed,
            stratify=labels,
        )
        test_ids = list(test_labels_dict.keys())
    else:
        # Two-stage split: train vs (val+test), then val vs test.
        train_ids, temp_ids, train_labels, temp_labels = train_test_split(
            ids, labels,
            test_size=args.val_ratio + args.test_ratio,
            random_state=args.seed,
            stratify=labels,
        )
        val_test_ratio = args.test_ratio / (args.val_ratio + args.test_ratio)
        val_ids, test_ids, _, _ = train_test_split(
            temp_ids, temp_labels,
            test_size=val_test_ratio,
            random_state=args.seed,
            stratify=temp_labels,
        )

    print(f"\nSplit sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    # --- Nested stratified ordering of the train set ---
    ordered_train_ids = stratified_nested_order(train_ids, train_labels, seed=args.seed)
    verify_stratification(
        ordered_train_ids, labels_dict,
        n_values=args.check_n + [len(ordered_train_ids)],
        parent_dist=parent_dist,
    )

    # --- Write outputs ---
    def _write(name, data):
        path = out_dir / name
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  Wrote {path} ({len(data)} ids)")

    print("\n--- Writing outputs ---")
    _write("train_ids.json", ordered_train_ids)
    _write("val_ids.json", sorted(val_ids))
    _write("test_ids.json", sorted(test_ids))

    meta = {
        "mode": "crossyear" if crossyear else "combined",
        "labels_csv": [str(p) for p in args.labels_csv],
        "test_labels_csv": [str(p) for p in args.test_labels_csv] if crossyear else None,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio if not crossyear else None,
        "train_size": len(ordered_train_ids),
        "val_size": len(val_ids),
        "test_size": len(test_ids),
        "train_val_pool_size": total,
        "parent_distribution_trainval": {
            name: parent_counts.get(i, 0) for i, name in enumerate(ESSD_5CLASS_MAP)
        },
        "class_map": ESSD_5CLASS_MAP,
        "train_ids_are_nested_stratified": True,
    }
    if crossyear:
        meta["test_distribution"] = {
            name: test_counts.get(i, 0) for i, name in enumerate(ESSD_5CLASS_MAP)
        }
    with open(out_dir / "split_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  Wrote {out_dir / 'split_meta.json'}")


if __name__ == "__main__":
    main()
