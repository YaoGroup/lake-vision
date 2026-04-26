#!/usr/bin/env python3
"""Run a trained LakeDrainageClassifier on a list of lake_ids and dump
predictions to CSV.

Standalone CLI (no notebook). Mirrors the inference logic in
`essd/notebooks/essd_fig7.ipynb::fig7-inference-helper`, plus probabilities
in the output. Uses GPU when available.

Example (from the SLURM submission script):
    python3 run_inference.py \\
        --checkpoint $MODELS_DIR/lakevision_essd_crossyear_bestf1.pth \\
        --ids_file   $SPLITS_DIR/val_ids.json \\
        --labels_csv $LABELS_ROOT/labels_CW_2019.csv \\
        --nc_dir     $NC_DIR \\
        --output_csv $OUT_DIR/val_predictions_bestf1.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import netCDF4 as nc4

from lakevision.models.classifier import LakeDrainageClassifier

CLASS_NAMES = ["ND", "HF", "MD", "LD", "CD"]


def load_labels(csv_paths):
    out = {}
    for p in csv_paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("label", "").strip():
                    out[row["lake_id"]] = row["label"].strip()
    return out


def build_model(device):
    # Same hyperparameters as the cross-year + combined training runs
    # (see engine/training/run_training.py argparse defaults).
    model = LakeDrainageClassifier(
        num_classes=5, seq_len=153,
        use_imgseq=True, use_areaseq=True,
        use_cloudyseq=False, use_nir=False, use_swir16=False,
        attention_type="none",
        frontcnn_base_channels=8, frontcnn_num_layers=4,
        clstm_hidden=32, slstm_hidden=16,
        classhead_hidden=64, classhead_dropout=0.3,
    ).to(device)
    return model


def predict_one(model, device, nc_path):
    with nc4.Dataset(str(nc_path)) as nc:
        nc.set_auto_mask(False)
        ch_names = [str(c) for c in nc.variables["channel"][:]]
        ch_idxs = [ch_names.index(c) for c in ("red", "green", "blue", "mask")]
        imagery = np.asarray(nc.variables["imagery"][:, ch_idxs, :, :], dtype=np.float32)
        wa = np.asarray(nc.variables["water_area"][:], dtype=np.float32)
    for ci in range(3):
        imagery[:, ci, :, :] /= 10000.0
    imagery = np.nan_to_num(imagery, nan=0.0)
    wa = np.nan_to_num(wa, nan=0.0)
    img_seq = torch.from_numpy(imagery).unsqueeze(0).to(device)
    area_seq = torch.from_numpy(wa).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        logits = model(img_seq, area_seq, None)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return CLASS_NAMES[int(probs.argmax())], probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to .pth weights")
    ap.add_argument("--ids_file", required=True, help="JSON file: list of lake_ids")
    ap.add_argument("--labels_csv", nargs="+", required=True,
                    help="One or more label CSVs (lake_id,label,...)")
    ap.add_argument("--nc_dir", required=True,
                    help="Flat directory containing {lake_id}.nc composite files")
    ap.add_argument("--output_csv", required=True, help="Where to write predictions")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    labels = load_labels(args.labels_csv)
    print(f"Loaded {len(labels)} labels from {args.labels_csv}")

    ids = json.load(open(args.ids_file))
    print(f"Loaded {len(ids)} ids from {args.ids_file}")

    model = build_model(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Resume support: read any predictions already in output_csv and skip them. ---
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lake_id", "true_label", "pred_label",
                  "p_ND", "p_HF", "p_MD", "p_LD", "p_CD"]
    already_done = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            for row in csv.DictReader(f):
                already_done.add(row["lake_id"])
        print(f"Resuming: found {len(already_done)} cached predictions in {out_path.name}")

    # Open in append mode and flush after each row so a SLURM kill leaves a
    # valid partial CSV on disk that the next run can resume from.
    file_existed = out_path.exists() and out_path.stat().st_size > 0
    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not file_existed:
        writer.writeheader()
        fout.flush()

    nc_dir = Path(args.nc_dir)
    t0 = time.time()
    n_done_this_run = n_skip = 0
    todo = [lid for lid in ids if lid not in already_done]
    for idx, lid in enumerate(todo):
        nc_path = nc_dir / f"{lid}.nc"
        if not nc_path.exists():
            print(f"  SKIP {lid}: file not found")
            n_skip += 1
            continue
        try:
            pred, probs = predict_one(model, device, nc_path)
        except Exception as e:
            print(f"  SKIP {lid}: {e}")
            n_skip += 1
            continue
        writer.writerow({
            "lake_id": lid,
            "true_label": labels.get(lid, ""),
            "pred_label": pred,
            "p_ND": float(probs[0]), "p_HF": float(probs[1]), "p_MD": float(probs[2]),
            "p_LD": float(probs[3]), "p_CD": float(probs[4]),
        })
        fout.flush()
        n_done_this_run += 1
        if (idx + 1) % 50 == 0 or idx == 0:
            dt = time.time() - t0
            eta = dt / (idx + 1) * (len(todo) - idx - 1) / 60
            print(f"  [{idx+1:4d}/{len(todo)}]  done_this_run={n_done_this_run}  "
                  f"skipped={n_skip}  elapsed={dt/60:.1f}m  ETA={eta:.1f}m")

    fout.close()
    total_in_csv = len(already_done) + n_done_this_run
    print(f"Wrote {n_done_this_run} new rows to {out_path} "
          f"(total now {total_in_csv}/{len(ids)})")
    print(f"This run: {(time.time()-t0)/60:.1f}m  done_this_run={n_done_this_run}  skipped={n_skip}")


if __name__ == "__main__":
    main()
