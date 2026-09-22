#!/usr/bin/env python3
"""Run a trained LakeDrainageClassifier on a list of lake_ids and dump
predictions to CSV.

Two checkpoint formats exist and both are handled (see
lakevision/models/checkpoint.py):

  provenance  {state_dict, config, class_names, ...}  -- eleven-runs and the
              ESSD revision. The model AND the dataset are rebuilt from the
              stored config, mirroring engine/training/run_training.py, so the
              inputs (band standardisation, fill policy, validity channel,
              static mask, deposit vs composite layout) are exactly what the
              network was trained on. Nothing about preprocessing is
              hard-coded here; it cannot drift from training.

  bare        a plain state_dict -- the frozen ESSD-submission tags. Kept on
              the original code path (published architecture, composite files,
              /10000 + clip, zero fill) so those runs stay reproducible.

Output columns are the same the training script writes for the test split
(lake_id, true_label, pred_label, p_<class>...), so the appendix tables, the
map figures and the Terminal-Bench scorer read either without special-casing.

Example (from a SLURM submission script):
    python3 run_inference.py \\
        --checkpoint $MODELS_DIR/lakevision_essd_rev_y2019_5c_pw_noaug_bestf1.pth \\
        --ids_file   $SPLITS_DIR/train_ids.json \\
        --labels_csv $LABELS_ROOT/labels_CW_2019.csv \\
        --nc_dir     $NC_DIR \\
        --output_csv $OUT_DIR/y2019_5c_pw_noaug_train_predictions_bestf1.csv
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
from torch.utils.data import DataLoader

from lakevision.data import LakeDataset
from lakevision.models.classifier import LakeDrainageClassifier
from lakevision.models.checkpoint import load_checkpoint, describe_checkpoint

CLASS_NAMES_5 = ["ND", "HF", "MD", "LD", "CD"]


def load_labels(csv_paths):
    out = {}
    for p in csv_paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("label", "").strip():
                    out[row["lake_id"]] = row["label"].strip()
    return out


def find_nc(nc_dir: Path, lid: str) -> Path:
    """Flat layout first (a staged dir), then CW_YYYY/ subdirs (OAK layout)."""
    flat = nc_dir / f"{lid}.nc"
    if flat.exists():
        return flat
    year = lid.split("_", 1)[0][2:]   # 'CW2019_1234' -> '2019'
    return nc_dir / f"CW_{year}" / f"{lid}.nc"


# ---------------------------------------------------------------------------
# Provenance checkpoints: rebuild everything from the stored config
# ---------------------------------------------------------------------------

def class_index_map(class_names, merge_classes):
    """5-class label name -> index in this checkpoint's class list.

    Mirrors the --merge_classes remap in run_training.py: the two merged
    classes both land on the merged name (class_a + class_b, e.g. 'HFMD').
    """
    m = {}
    merged = "".join(merge_classes) if merge_classes else None
    for name in CLASS_NAMES_5:
        target = merged if (merge_classes and name in merge_classes) else name
        if target not in class_names:
            raise ValueError(f"class {name!r} -> {target!r} not in checkpoint classes {class_names}")
        m[name] = class_names.index(target)
    return m


def build_from_config(config, class_names, labels_dict, nc_paths, device):
    """LakeDataset + LakeDrainageClassifier exactly as run_training.py builds them."""
    dataset_kwargs = {
        'seq_len': config.get("seq_len", 153),
        'use_nir': config.get("use_nir", False),
        'use_swir16': config.get("use_swir16", False),
        'use_swir22': config.get("use_swir22", False),
        'use_mask': not config.get("no_mask", False),
        'band_stats': config.get("band_stats"),
        'cloudy_seq_var': config.get("cloudy_seq_var", "cloudy_seq_rgb"),
        'validity_channel': config.get("validity_channel", False),
        'cloud_channel': config.get("cloud_channel"),
        'fill': config.get("fill", "zero"),
        'mask_source': config.get("mask_source"),
        'labels_dict': labels_dict,
    }
    dataset = LakeDataset(nc_paths, preload_to_ram=False, **dataset_kwargs)
    n_input_channels = dataset.n_channels
    n_aux_channels = dataset.n_aux_channels

    model = LakeDrainageClassifier(
        use_imgseq=config.get("use_imgseq", True),
        use_areaseq=config.get("use_areaseq", True),
        use_cloudyseq=config.get("use_cloudyseq", False),
        learn_area_weights=config.get("learn_area_weights", False),
        learn_cloudy_weights=config.get("learn_cloudy_weights", False),
        seq_len=config.get("seq_len", 153),
        use_nir=config.get("use_nir", False),
        use_swir16=config.get("use_swir16", False),
        use_swir22=config.get("use_swir22", False),
        attention_type=config.get("attention_type", "none"),
        n_aux_channels=n_aux_channels,
        expect_channels=n_input_channels,
        num_classes=len(class_names),
        frontcnn_base_channels=config.get("frontcnn_base_channels", 8),
        frontcnn_num_layers=config.get("frontcnn_num_layers", 4),
        frontcnn_out_hw=config.get("frontcnn_out_hw"),
        frontcnn_norm=config.get("frontcnn_norm", "none"),
        frontcnn_norm_groups=config.get("frontcnn_norm_groups", 8),
        frontcnn_chunk_size=config.get("frontcnn_chunk_size"),
        clstm_hidden=config.get("clstm_hidden", 32),
        clstm_kernel=config.get("clstm_kernel", 3),
        clstm_forget_bias=config.get("clstm_forget_bias", 0.0),
        temporal_readout=config.get("temporal_readout", "last"),
        slstm_hidden=config.get("slstm_hidden", 16),
        slstm_num_layers=config.get("slstm_num_layers", 1),
        slstm_dropout=config.get("slstm_dropout", 0.0),
        classhead_hidden=config.get("classhead_hidden", 64),
        classhead_dropout=config.get("classhead_dropout", 0.0),
        pool_type=config.get("pool_type", "avg"),
        gradient_checkpointing=False,
    ).to(device)
    return dataset, model


def predict_provenance(state, meta, args, device):
    config = meta["config"]
    class_names = list(meta.get("class_names") or CLASS_NAMES_5[: meta.get("num_classes", 5)])
    idx_of = class_index_map(class_names, config.get("merge_classes"))

    labels_txt = load_labels(args.labels_csv)
    ids = json.load(open(args.ids_file))
    nc_dir = Path(args.nc_dir)

    labels_dict, nc_paths, missing = {}, [], []
    for lid in ids:
        p = find_nc(nc_dir, lid)
        if not p.exists():
            missing.append(lid); continue
        if lid not in labels_txt:
            missing.append(lid); continue
        labels_dict[lid] = idx_of[labels_txt[lid]]
        nc_paths.append(p)
    if missing:
        print(f"  WARNING: {len(missing)} ids skipped (no file or no label), e.g. {missing[:3]}")
    print(f"Predicting {len(nc_paths)} lakes; classes {class_names}")

    dataset, model = build_from_config(config, class_names, labels_dict, nc_paths, device)
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    # Match the training-time evaluate(): bf16 autocast iff the run used AMP.
    use_amp = bool(config.get("amp", False)) and device.type == "cuda"

    all_ids, all_y, all_p, all_probs = [], [], [], []
    t0 = time.time()
    with torch.no_grad():
        for bi, (img_seq, area_seq, cloudy_seq, y, lake_ids) in enumerate(loader):
            img_seq, area_seq, cloudy_seq = img_seq.to(device), area_seq.to(device), cloudy_seq.to(device)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(img_seq, area_seq, cloudy_seq)
            else:
                logits = model(img_seq, area_seq, cloudy_seq)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_ids.extend(list(lake_ids)); all_y.extend(y.numpy().tolist())
            all_p.extend(probs.argmax(1).tolist()); all_probs.append(probs)
            if bi % 10 == 0:
                done = len(all_ids)
                print(f"  [{done:4d}/{len(nc_paths)}]  {(time.time()-t0)/60:.1f}m", flush=True)
    probs = np.concatenate(all_probs) if all_probs else np.zeros((0, len(class_names)))

    out = Path(args.output_csv); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lake_id", "true_label", "pred_label"] + [f"p_{c}" for c in class_names])
        for lid, y, p, pr in zip(all_ids, all_y, all_p, probs):
            w.writerow([lid, class_names[int(y)] if int(y) >= 0 else "", class_names[int(p)]]
                       + [f"{float(v):.6f}" for v in pr])
    acc = float(np.mean(np.asarray(all_y) == np.asarray(all_p))) if all_y else float("nan")
    print(f"Wrote {len(all_ids)} predictions to {out}  (accuracy {acc:.4f}, {(time.time()-t0)/60:.1f}m)")


# ---------------------------------------------------------------------------
# Bare checkpoints: the frozen ESSD-submission path, unchanged
# ---------------------------------------------------------------------------

def build_model_legacy(device):
    # Same hyperparameters as the cross-year + combined training runs
    # (see engine/training/run_training.py argparse defaults at the tag).
    return LakeDrainageClassifier(
        num_classes=5, seq_len=153,
        use_imgseq=True, use_areaseq=True,
        use_cloudyseq=False, use_nir=False, use_swir16=False,
        attention_type="none",
        frontcnn_base_channels=8, frontcnn_num_layers=4,
        frontcnn_out_hw=(64, 64),
        clstm_hidden=32, slstm_hidden=16,
        classhead_hidden=64, classhead_dropout=0.3,
    ).to(device)


def predict_one_legacy(model, device, nc_path):
    import netCDF4 as nc4
    with nc4.Dataset(str(nc_path)) as nc:
        nc.set_auto_mask(False)
        ch_names = [str(c) for c in nc.variables["channel"][:]]
        ch_idxs = [ch_names.index(c) for c in ("red", "green", "blue", "mask")]
        imagery = np.asarray(nc.variables["imagery"][:, ch_idxs, :, :], dtype=np.float32)
        wa = np.asarray(nc.variables["water_area"][:], dtype=np.float32)
    # Mirror the tag-era LakeDataset exactly: reflectance clipped to [0,1],
    # NaN -> 0, area min-max normalised per sample.
    for ci in range(3):
        imagery[:, ci, :, :] = np.clip(imagery[:, ci, :, :] / 10000.0, 0.0, 1.0)
    imagery = np.nan_to_num(imagery, nan=0.0)
    if int(np.isnan(wa).sum()):
        raise ValueError("water_area contains NaN; composites should be NaN-free")
    wa = (wa - wa.min()) / (wa.max() - wa.min() + 1e-8)
    img_seq = torch.from_numpy(imagery).unsqueeze(0).to(device)
    area_seq = torch.from_numpy(wa).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(img_seq, area_seq, None), dim=1).cpu().numpy()[0]
    return CLASS_NAMES_5[int(probs.argmax())], probs


def predict_legacy(state, args, device):
    labels = load_labels(args.labels_csv)
    ids = json.load(open(args.ids_file))
    model = build_model_legacy(device)
    model.load_state_dict(state)
    model.eval()

    out_path = Path(args.output_csv); out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lake_id", "true_label", "pred_label", "p_ND", "p_HF", "p_MD", "p_LD", "p_CD"]
    already = set()
    if out_path.exists() and out_path.stat().st_size > 0:
        with open(out_path) as f:
            already = {r["lake_id"] for r in csv.DictReader(f)}
        print(f"Resuming: {len(already)} cached predictions in {out_path.name}")
    fout = open(out_path, "a", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    if not already:
        writer.writeheader(); fout.flush()

    nc_dir = Path(args.nc_dir); t0 = time.time(); n_done = n_skip = 0
    todo = [lid for lid in ids if lid not in already]
    for idx, lid in enumerate(todo):
        p = find_nc(nc_dir, lid)
        if not p.exists():
            print(f"  SKIP {lid}: file not found at {p}"); n_skip += 1; continue
        try:
            pred, probs = predict_one_legacy(model, device, p)
        except Exception as e:
            print(f"  SKIP {lid}: {e}"); n_skip += 1; continue
        writer.writerow({"lake_id": lid, "true_label": labels.get(lid, ""), "pred_label": pred,
                         **{f"p_{c}": float(v) for c, v in zip(CLASS_NAMES_5, probs)}})
        fout.flush(); n_done += 1
        if (idx + 1) % 50 == 0 or idx == 0:
            dt = time.time() - t0
            print(f"  [{idx+1:4d}/{len(todo)}]  done={n_done} skipped={n_skip}  "
                  f"elapsed={dt/60:.1f}m  ETA={dt/(idx+1)*(len(todo)-idx-1)/60:.1f}m")
    fout.close()
    print(f"Wrote {n_done} new rows to {out_path} (total {len(already)+n_done}/{len(ids)}); skipped={n_skip}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth (provenance or bare)")
    ap.add_argument("--ids_file", required=True, help="JSON file: list of lake_ids")
    ap.add_argument("--labels_csv", nargs="+", required=True, help="One or more label CSVs (lake_id,label,...)")
    ap.add_argument("--nc_dir", required=True, help="Dir of {lake_id}.nc files (flat, or CW_YYYY/ subdirs)")
    ap.add_argument("--output_csv", required=True, help="Where to write predictions")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    state, meta = load_checkpoint(args.checkpoint, map_location=device)
    print(f"Checkpoint: {describe_checkpoint(args.checkpoint, meta)}")

    if meta and meta.get("config"):
        predict_provenance(state, meta, args, device)
    else:
        print("Bare state_dict: using the ESSD-submission architecture and composite preprocessing.")
        predict_legacy(state, args, device)


if __name__ == "__main__":
    main()
