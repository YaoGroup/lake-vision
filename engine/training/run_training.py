"""
Training script for LakeDrainageClassifier with wandb integration.

Usage:
    python run_training.py --labels_csv /path/to/labels.csv --nc_dir /path/to/nc/files

For wandb sweeps:
    wandb sweep sweep.yaml
    wandb agent <sweep_id>
"""
import argparse
import json
import random
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.data import LakeDataset
from lakevision.data.loader_plan import plan_loader_workers, sample_size_mb
from lakevision.data.transforms import (
    AugmentedDatasetWrapper,
    RandomD4Dataset,
    AUGMENTATIONS,
)
from lakevision.models.classifier import LakeDrainageClassifier


def set_seed(seed: int):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed, logging disabled")


# Class names for lake drainage types
CLASS_NAMES_ORIGINAL = ['ND', 'ED', 'LD', 'CD']  # 0: No Drainage, 1: Englacial, 2: Lateral, 3: Crevasse
CLASS_NAMES_ED_SPLIT = ['ND', 'LD_MD', 'HF', 'CD']  # 0: No Drainage, 1: Lateral+Moulin, 2: Hydrofracture, 3: Crevasse
CLASS_NAMES_ESSD_5CLASS = ['ND', 'HF', 'MD', 'LD', 'CD']  # 0: No Drainage, 1: Hydrofracture, 2: Moulin Drainage, 3: Lateral, 4: Crevasse

# String-to-int mapping for essd_5class (matches the GUI schema)
ESSD_5CLASS_MAP = {name: i for i, name in enumerate(CLASS_NAMES_ESSD_5CLASS)}

# Default (overridden by label_mode in train())
CLASS_NAMES = CLASS_NAMES_ORIGINAL


def remap_labels_ed_split(labels_csv, id_col='new_id', label_col='label_rines', edm_edf_col='edm_edf'):
    """
    Remap labels for the ed_split scheme: split ED into moulin (merged with LD) and hydrofracture.

    Original: ND=0, ED=1, LD=2, CD=3
    New:      ND=0, LD+MD=1, HF=2, CD=3

    ED lakes with edm_edf='m' (moulin) -> 1 (merged with LD)
    ED lakes with edm_edf='f' (hydrofracture) or '?' -> 2 (HF)
    LD lakes -> 1
    ND lakes -> 0 (unchanged)
    CD lakes -> 3 (unchanged)

    Args:
        labels_csv: Path to CSV with labels
        id_col: Column name for lake IDs
        label_col: Column name for original labels
        edm_edf_col: Column name for moulin/hydrofracture indicator

    Returns:
        dict: mapping lake_id -> remapped label (int)
    """
    df = pd.read_csv(labels_csv)
    df = df.dropna(subset=[id_col, label_col])

    remapped = {}
    for _, row in df.iterrows():
        lake_id = row[id_col]
        orig_label = int(row[label_col])

        if orig_label == 0:  # ND -> 0
            remapped[lake_id] = 0
        elif orig_label == 1:  # ED -> split based on edm_edf
            edm_edf = str(row.get(edm_edf_col, '?')).strip().lower()
            if edm_edf == 'm':
                remapped[lake_id] = 1  # moulin -> LD+MD
            elif edm_edf == 'f':
                remapped[lake_id] = 2  # hydrofracture -> HF
            else:  # '?' or unknown -> drop from dataset
                continue
        elif orig_label == 2:  # LD -> 1 (LD+MD)
            remapped[lake_id] = 1
        elif orig_label == 3:  # CD -> 3
            remapped[lake_id] = 3

    # Print distribution
    counts = Counter(remapped.values())
    print(f"Remapped label distribution (ed_split):")
    for i, name in enumerate(CLASS_NAMES_ED_SPLIT):
        print(f"  {name} ({i}): {counts.get(i, 0)}")

    return remapped


def load_labels_essd_5class(csv_paths, id_col='lake_id', label_col='label'):
    """Load labels from one or more GUI CSVs with the 5-class ESSD schema.

    The sat-tile-stack labeling GUI writes string labels (ND/HF/MD/LD/CD)
    in the `label` column. This function reads any number of such CSVs,
    filters out rows with empty/missing labels (e.g. flagged-only rows),
    and maps strings to integers 0-4.

    Flagged lakes are kept (the `flagged` column is ignored here).

    Args:
        csv_paths: single path or list of paths to labels CSVs
        id_col: lake ID column (default: 'lake_id')
        label_col: label string column (default: 'label')

    Returns:
        dict mapping lake_id -> int label in [0, 4]
    """
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]

    combined = {}
    per_file_counts = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=[id_col, label_col])
        # Drop rows with empty-string labels (can happen when `flagged=True`
        # but no label assigned yet)
        df = df[df[label_col].astype(str).str.strip() != ""]

        n = 0
        for lid, lbl_str in zip(df[id_col].astype(str), df[label_col].astype(str)):
            lbl_str = lbl_str.strip()
            if lbl_str not in ESSD_5CLASS_MAP:
                raise ValueError(
                    f"Unknown label '{lbl_str}' in {csv_path}. "
                    f"Expected one of {list(ESSD_5CLASS_MAP.keys())}."
                )
            if lid in combined and combined[lid] != ESSD_5CLASS_MAP[lbl_str]:
                print(f"  WARNING: duplicate ID {lid} with conflicting labels; "
                      f"keeping first occurrence")
                continue
            combined[lid] = ESSD_5CLASS_MAP[lbl_str]
            n += 1
        per_file_counts[str(csv_path)] = n

    # Print per-file + combined distribution
    print(f"\nLoaded essd_5class labels:")
    for path, n in per_file_counts.items():
        print(f"  {Path(path).name}: {n} labels")
    counts = Counter(combined.values())
    print(f"  Combined distribution:")
    for i, name in enumerate(CLASS_NAMES_ESSD_5CLASS):
        print(f"    {name} ({i}): {counts.get(i, 0)}")
    print(f"  Total: {len(combined)} lakes")

    return combined


def create_splits(
    labels_csv: str,
    id_col: str = 'new_id',
    label_col: str = 'label_rines',
    train_ratio: float = 0.7,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    stratify: bool = True,
    labels_dict: dict = None,
):
    """
    Create train/val/test splits from labels CSV.

    Args:
        labels_csv: Path to CSV with labels
        id_col: Column name for lake IDs
        label_col: Column name for labels
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed
        stratify: Whether to stratify splits by label
        labels_dict: Optional pre-remapped labels dict (lake_id -> int).
            If provided, uses these labels for stratification instead of label_col.

    Returns:
        tuple: (train_ids, val_ids, test_ids) as lists of lake IDs
    """
    if labels_dict is not None:
        ids = list(labels_dict.keys())
        labels = [labels_dict[lid] for lid in ids]
    else:
        df = pd.read_csv(labels_csv)
        df = df.dropna(subset=[id_col, label_col])
        ids = df[id_col].tolist()
        labels = df[label_col].astype(int).tolist()

    # First split: train vs (val+test)
    stratify_labels = labels if stratify else None
    train_ids, temp_ids, train_labels, temp_labels = train_test_split(
        ids, labels,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=stratify_labels,
    )

    # Second split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    stratify_temp = temp_labels if stratify else None
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_labels,
        test_size=val_test_ratio,
        random_state=seed,
        stratify=stratify_temp,
    )

    print(f"Split sizes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    return train_ids, val_ids, test_ids


def create_splits_fixed_test(
    labels_dict: dict,
    test_labels_dict: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
):
    """Create train/val from labels_dict and use test_labels_dict as the full test set.

    Used for the cross-year ESSD baseline: train+val from one year (e.g. 2019),
    held-out test from another year (e.g. 2018).

    Args:
        labels_dict: {lake_id: int} for the train+val pool.
        test_labels_dict: {lake_id: int} for the held-out test set.
        train_ratio, val_ratio: fractions within the train+val pool (must sum to 1.0).

    Returns:
        (train_ids, val_ids, test_ids)
    """
    if abs((train_ratio + val_ratio) - 1.0) > 1e-6:
        raise ValueError(f"train_ratio + val_ratio must equal 1.0 "
                         f"(got {train_ratio} + {val_ratio})")

    # Warn if any overlap between train+val pool and test pool
    overlap = set(labels_dict) & set(test_labels_dict)
    if overlap:
        print(f"  WARNING: {len(overlap)} lakes appear in both train+val and "
              f"test pools; removing from train+val pool.")
        labels_dict = {k: v for k, v in labels_dict.items() if k not in overlap}

    ids = list(labels_dict.keys())
    labels = [labels_dict[lid] for lid in ids]

    stratify_labels = labels if stratify else None
    train_ids, val_ids, _, _ = train_test_split(
        ids, labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=stratify_labels,
    )

    test_ids = list(test_labels_dict.keys())
    print(f"Split sizes (fixed test): train={len(train_ids)}, "
          f"val={len(val_ids)}, test={len(test_ids)}")

    return train_ids, val_ids, test_ids


def unpack_batch(batch):
    """Split a batch from either dataset into a common shape.

    LakeDataset yields 5 elements (already float32, already normalized).
    CachedLakeDataset yields 6, appending the per-timestep BOA offset, because
    its imagery is uint16 raw DN that still needs
    (stored/DN_SCALE + boa)/10000 applied -- on the GPU, which is what keeps the
    dataloader queue at 2 bytes/value instead of 4.

    Returns (img, area_seq, cloudy_seq, labels, boa_or_None).
    """
    img, area_seq, cloudy_seq, labels = batch[0], batch[1], batch[2], batch[3]
    boa = batch[5] if len(batch) > 5 else None
    return img, area_seq, cloudy_seq, labels, boa


def train_one_epoch(model, loader, optimizer, criterion, device, num_classes=4,
                    accumulation_steps=1, amp=False, normalize_fn=None):
    """Train for one epoch and return loss + metrics.

    Args:
        accumulation_steps: Number of mini-batches to accumulate gradients over.
                           Effective batch size = batch_size * accumulation_steps.
        amp: If True, run forward/backward in bf16 autocast. Halves activation
             memory on A100, enables larger batch sizes. No GradScaler needed
             because bf16 has the same dynamic range as fp32.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    optimizer.zero_grad()  # Zero gradients once at start

    amp_dtype = torch.bfloat16 if amp else None

    for batch_idx, batch in enumerate(loader):
        img_seq, area_seq, cloudy_seq, labels, boa = unpack_batch(batch)

        img_seq = img_seq.to(device, non_blocking=True)
        area_seq = area_seq.to(device, non_blocking=True)
        cloudy_seq = cloudy_seq.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if boa is not None:
            boa = boa.to(device, non_blocking=True)

        if amp:
            with torch.autocast(device_type='cuda', dtype=amp_dtype):
                x = normalize_fn(img_seq, boa) if normalize_fn else img_seq
                logits = model(x, area_seq, cloudy_seq)
                loss = criterion(logits, labels)
        else:
            x = normalize_fn(img_seq, boa) if normalize_fn else img_seq
            logits = model(x, area_seq, cloudy_seq)
            loss = criterion(logits, labels)

        # Scale loss by accumulation steps to maintain proper gradient magnitude
        loss = loss / accumulation_steps
        loss.backward()

        # Step optimizer every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        n_batches += 1

        # Collect predictions for metrics
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Handle remaining gradients if batches not divisible by accumulation_steps
    if n_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / n_batches

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
    }

    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES[:num_classes]):
        binary_labels = [1 if l == i else 0 for l in all_labels]
        binary_preds = [1 if p == i else 0 for p in all_preds]
        metrics[f'precision_{class_name}'] = precision_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'recall_{class_name}'] = recall_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'f1_{class_name}'] = f1_score(binary_labels, binary_preds, zero_division=0)

    return avg_loss, metrics


def evaluate(model, loader, criterion, device, num_classes=4, amp=False,
             normalize_fn=None):
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    amp_dtype = torch.bfloat16 if amp else None

    with torch.no_grad():
        for batch in loader:
            img_seq, area_seq, cloudy_seq, labels, boa = unpack_batch(batch)

            img_seq = img_seq.to(device, non_blocking=True)
            area_seq = area_seq.to(device, non_blocking=True)
            cloudy_seq = cloudy_seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if boa is not None:
                boa = boa.to(device, non_blocking=True)

            if amp:
                with torch.autocast(device_type='cuda', dtype=amp_dtype):
                    x = normalize_fn(img_seq, boa) if normalize_fn else img_seq
                    logits = model(x, area_seq, cloudy_seq)
                    loss = criterion(logits, labels)
            else:
                x = normalize_fn(img_seq, boa) if normalize_fn else img_seq
                logits = model(x, area_seq, cloudy_seq)
                loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / n_batches

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=list(range(num_classes))),
    }

    # Per-class metrics
    for i, class_name in enumerate(CLASS_NAMES[:num_classes]):
        binary_labels = [1 if l == i else 0 for l in all_labels]
        binary_preds = [1 if p == i else 0 for p in all_preds]
        metrics[f'precision_{class_name}'] = precision_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'recall_{class_name}'] = recall_score(binary_labels, binary_preds, zero_division=0)
        metrics[f'f1_{class_name}'] = f1_score(binary_labels, binary_preds, zero_division=0)

    return avg_loss, metrics


def count_parameters(model):
    """Count trainable and total parameters in a model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def train(config: dict):
    """Main training function."""
    # Normalize labels_csv to a list (argparse uses nargs='+').
    # For non-essd_5class modes, collapse back to a single string for
    # backward compatibility with create_splits() and pd.read_csv().
    if isinstance(config.get("labels_csv"), list):
        if config.get("label_mode", "original") != "essd_5class":
            if len(config["labels_csv"]) > 1:
                print(f"  WARNING: {len(config['labels_csv'])} CSVs passed but "
                      f"label_mode is '{config.get('label_mode', 'original')}'. "
                      f"Only the first will be used.")
            config["labels_csv"] = config["labels_csv"][0]

    # Print header
    print("\n" + "=" * 70)
    print("LAKE-VISION TRAINING")
    print("=" * 70)

    # Seed and device
    seed = config.get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print configuration summary
    print("\n--- CONFIGURATION ---")
    print(f"Random seed:    {seed}")
    print(f"Device:         {device}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        print(f"CUDA version:   {torch.version.cuda}")

    print("\n--- DATA PATHS ---")
    print(f"Labels CSV:     {config['labels_csv']}")
    print(f"NC directory:   {config['nc_dir']}")
    print(f"Band stats:     {config.get('band_stats', 'None')}")
    print(f"Save path:      {config.get('save_path', 'None')}")

    # Defaults below MUST match the argparse defaults in main(); when called
    # from CLI, every key is already populated by argparse so the .get fallback
    # is dead code, but keeping them in sync prevents future drift.
    print("\n--- TRAINING HYPERPARAMETERS ---")
    print(f"Epochs:         {config.get('epochs', 400)}")
    print(f"Batch size:     {config.get('batch_size', 8)}")
    print(f"Learning rate:  {config.get('lr', 1e-4)}")
    print(f"Weight decay:   {config.get('weight_decay', 1e-5)}")
    print(f"AMP (bf16):     {config.get('amp', True)}")
    print(f"Scheduler:      {config.get('use_scheduler', False)}")
    print(f"Num workers:    {config.get('num_workers', 12)}")

    print("\n--- INPUT STREAMS ---")
    print(f"use_imgseq:     {config.get('use_imgseq', True)}")
    print(f"use_areaseq:    {config.get('use_areaseq', True)}")
    print(f"use_cloudyseq:  {config.get('use_cloudyseq', False)}")
    print(f"cloudy_seq_var: {config.get('cloudy_seq_var', 'cloudy_seq_rgb')}")

    print("\n--- SPECTRAL BANDS ---")
    print(f"use_nir:        {config.get('use_nir', False)}")
    print(f"use_swir16:     {config.get('use_swir16', False)}")

    print("\n--- MODEL ARCHITECTURE ---")
    print(f"seq_len:              {config.get('seq_len', 153)}")
    print(f"num_classes:          {config.get('num_classes', 5)}")
    print(f"attention_type:       {config.get('attention_type', 'none')}")
    print(f"frontcnn_base_ch:     {config.get('frontcnn_base_channels', 8)}")
    print(f"frontcnn_num_layers:  {config.get('frontcnn_num_layers', 4)}")
    print(f"clstm_hidden:         {config.get('clstm_hidden', 32)}")
    print(f"slstm_hidden:         {config.get('slstm_hidden', 16)}")
    print(f"classhead_hidden:     {config.get('classhead_hidden', 64)}")
    print(f"classhead_dropout:    {config.get('classhead_dropout', 0.3)}")
    print(f"learn_area_weights:   {config.get('learn_area_weights', False)}")
    print(f"learn_cloudy_weights: {config.get('learn_cloudy_weights', False)}")
    print(f"grad_checkpointing:   {config.get('gradient_checkpointing', False)}")
    print(f"accumulation_steps:   {config.get('accumulation_steps', 1)}")
    print(f"preload_to_ram:       {config.get('preload_to_ram', False)}")

    # Label mode: remap labels if using ed_split
    label_mode = config.get("label_mode", "original")
    print(f"\n--- LABEL MODE ---")
    print(f"label_mode:     {label_mode}")

    global CLASS_NAMES
    labels_dict = None
    test_labels_dict = None  # Only set for essd_5class cross-year mode

    if label_mode == "ed_split":
        CLASS_NAMES = CLASS_NAMES_ED_SPLIT
        labels_dict = remap_labels_ed_split(
            config["labels_csv"],
            id_col=config.get("id_col", "new_id"),
            label_col=config.get("label_col", "label_rines"),
            edm_edf_col=config.get("edm_edf_col", "edm_edf"),
        )
    elif label_mode == "essd_5class":
        CLASS_NAMES = CLASS_NAMES_ESSD_5CLASS
        # labels_csv may be a list (merged for train+val+test) or single path.
        # test_labels_csv, if set, holds the full held-out test set (cross-year).
        labels_dict = load_labels_essd_5class(
            config["labels_csv"],
            id_col=config.get("id_col", "lake_id"),
            label_col=config.get("label_col", "label"),
        )
        test_csv = config.get("test_labels_csv")
        if test_csv:
            test_labels_dict = load_labels_essd_5class(
                test_csv,
                id_col=config.get("id_col", "lake_id"),
                label_col=config.get("label_col", "label"),
            )
    else:
        CLASS_NAMES = CLASS_NAMES_ORIGINAL

    # ------------------------------------------------------------------
    # Optional class merge: fold CLASS_B into CLASS_A, producing a merged
    # class named "CLASS_A+CLASS_B" (e.g. MDLD). Indices are renumbered
    # so they stay contiguous 0..num_classes-1.
    # ------------------------------------------------------------------
    merge_classes = config.get("merge_classes")
    if merge_classes is not None:
        class_a, class_b = merge_classes
        if class_a not in CLASS_NAMES:
            raise ValueError(f"--merge_classes: '{class_a}' not in {CLASS_NAMES}")
        if class_b not in CLASS_NAMES:
            raise ValueError(f"--merge_classes: '{class_b}' not in {CLASS_NAMES}")
        if class_a == class_b:
            raise ValueError(f"--merge_classes: cannot merge a class with itself")

        idx_a = CLASS_NAMES.index(class_a)
        idx_b = CLASS_NAMES.index(class_b)
        merged_name = class_a + class_b

        # Build old→new index mapping: class_b → idx_a, then compress
        new_names = [n for n in CLASS_NAMES if n != class_b]
        new_names[new_names.index(class_a)] = merged_name
        old_to_new = {}
        for old_idx, name in enumerate(CLASS_NAMES):
            if old_idx == idx_b:
                old_to_new[old_idx] = new_names.index(merged_name)
            elif name == class_a:
                old_to_new[old_idx] = new_names.index(merged_name)
            else:
                old_to_new[old_idx] = new_names.index(name)

        # Remap labels_dict
        if labels_dict is not None:
            labels_dict = {lid: old_to_new[lbl] for lid, lbl in labels_dict.items()}
        if test_labels_dict is not None:
            labels_dict_test = {lid: old_to_new[lbl] for lid, lbl in test_labels_dict.items()}
            test_labels_dict = labels_dict_test

        CLASS_NAMES = new_names
        config["num_classes"] = len(CLASS_NAMES)

        print(f"\n--- CLASS MERGE ---")
        print(f"  Merged {class_b} into {class_a} → '{merged_name}'")
        print(f"  New classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
        print(f"  Index remap: {old_to_new}")
        if labels_dict is not None:
            counts = Counter(labels_dict.values())
            for i, name in enumerate(CLASS_NAMES):
                print(f"    {name} ({i}): {counts.get(i, 0)}")

    # Create data splits — three paths:
    #   1. Pre-computed ID files (learning-curve runs — fixed across N)
    #   2. Cross-year fixed test set
    #   3. Fresh stratified 70/20/10 split
    train_ids_file = config.get("train_ids_file")
    val_ids_file = config.get("val_ids_file")
    test_ids_file = config.get("test_ids_file")

    if train_ids_file or val_ids_file or test_ids_file:
        if not (train_ids_file and val_ids_file and test_ids_file):
            raise ValueError(
                "When using pre-computed split files, all three "
                "(--train_ids_file, --val_ids_file, --test_ids_file) must be set."
            )

        def _load_ids(path):
            with open(path) as f:
                return json.load(f)

        train_ids = _load_ids(train_ids_file)
        val_ids = _load_ids(val_ids_file)
        test_ids = _load_ids(test_ids_file)
        print(f"\nLoaded split IDs from files:")
        print(f"  train={len(train_ids)} from {train_ids_file}")
        print(f"  val  ={len(val_ids)} from {val_ids_file}")
        print(f"  test ={len(test_ids)} from {test_ids_file}")

        # Drop IDs not present in labels_dict (shouldn't happen if split
        # was built from same CSVs, but be defensive)
        known = set(labels_dict)
        dropped = [lid for lid in (train_ids + val_ids + test_ids) if lid not in known]
        if dropped:
            print(f"  WARNING: {len(dropped)} IDs from split files not in labels; dropping.")
            train_ids = [lid for lid in train_ids if lid in known]
            val_ids = [lid for lid in val_ids if lid in known]
            test_ids = [lid for lid in test_ids if lid in known]
    elif test_labels_dict is not None:
        # Cross-year: fixed held-out test set, 80/20 train/val from labels_dict
        train_ids, val_ids, test_ids = create_splits_fixed_test(
            labels_dict,
            test_labels_dict,
            train_ratio=config.get("train_ratio", 0.8),
            val_ratio=config.get("val_ratio", 0.2),
            seed=seed,
            stratify=config.get("stratify", True),
        )
        # Merge the two dicts so dataset_kwargs['labels_dict'] covers every ID
        labels_dict = {**labels_dict, **test_labels_dict}
    else:
        # Standard 70/20/10 stratified split
        train_ids, val_ids, test_ids = create_splits(
            config["labels_csv"],
            id_col=config.get("id_col", "new_id"),
            label_col=config.get("label_col", "label_rines"),
            train_ratio=config.get("train_ratio", 0.7),
            val_ratio=config.get("val_ratio", 0.20),
            test_ratio=config.get("test_ratio", 0.10),
            seed=seed,
            stratify=config.get("stratify", True),
            labels_dict=labels_dict,
        )

    # Learning-curve knob: truncate the train set to the first N IDs.
    # Used with train_ids_file (nested stratified order) so that N=400
    # is a superset of N=200, etc.
    max_train_lakes = config.get("max_train_lakes")
    if max_train_lakes is not None and max_train_lakes > 0:
        print(f"\n--- LEARNING CURVE: capping TRAIN at {max_train_lakes} lakes ---")
        train_ids = train_ids[:max_train_lakes]

    # Pilot cap: truncate all three splits (for end-to-end smoke tests).
    max_lakes = config.get("max_lakes")
    if max_lakes is not None and max_lakes > 0:
        print(f"\n--- PILOT MODE: capping each split at {max_lakes} lakes ---")
        train_ids = train_ids[:max_lakes]
        val_ids = val_ids[:max_lakes]
        test_ids = test_ids[:max_lakes]

    # Resolve which lakes actually have data on disk. The two backends store
    # things differently, so `_have` is what the rest of the function trusts.
    if config.get("use_cache"):
        cache_root = Path(config["cache_root"])
        probe_band = (config.get("cache_bands") or ["B04"])[0]
        def _have(lid):
            return (cache_root / probe_band / f"{lid}.b2nd").exists()
        train_paths = val_paths = test_paths = []      # unused on the cache path
    else:
        nc_dir = Path(config["nc_dir"])
        def _have(lid):
            return (nc_dir / f"{lid}.nc").exists()
        train_paths = [nc_dir / f"{lid}.nc" for lid in train_ids if _have(lid)]
        val_paths = [nc_dir / f"{lid}.nc" for lid in val_ids if _have(lid)]
        test_paths = [nc_dir / f"{lid}.nc" for lid in test_ids if _have(lid)]

    train_ids = [lid for lid in train_ids if _have(lid)]
    val_ids = [lid for lid in val_ids if _have(lid)]
    test_ids = [lid for lid in test_ids if _have(lid)]

    print(f"Found data: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    if not train_ids:
        raise ValueError(
            "No training lakes found on disk. Check --cache_root/--nc_dir and "
            "that the split IDs match what was built.")

    # Compute class weights from training set for weighted CrossEntropyLoss
    if labels_dict is not None:
        train_labels = [labels_dict[lid] for lid in train_ids]
    else:
        df_labels = pd.read_csv(config["labels_csv"])
        df_labels = df_labels.dropna(subset=[config.get("id_col", "new_id"), config.get("label_col", "label_rines")])
        label_map = dict(zip(df_labels[config.get("id_col", "new_id")],
                             df_labels[config.get("label_col", "label_rines")].astype(int)))
        train_labels = [label_map[lid] for lid in train_ids if lid in label_map]

    label_counts = Counter(train_labels)
    num_classes = config.get("num_classes", 5)
    total_train = len(train_labels)
    class_weights = torch.tensor([
        total_train / (num_classes * label_counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=torch.float32)

    print(f"\n--- CLASS WEIGHTS (inverse frequency) ---")
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        print(f"  {name} ({i}): count={label_counts.get(i, 0)}, weight={class_weights[i]:.3f}")

    # Dataset configuration
    dataset_kwargs = {
        'seq_len': config.get("seq_len", 153),
        'use_nir': config.get("use_nir", False),
        'use_swir16': config.get("use_swir16", False),
        'use_mask': not config.get("no_mask", False),
        'band_stats': config.get("band_stats"),
        'cloudy_seq_var': config.get("cloudy_seq_var", "cloudy_seq_rgb"),
    }

    # Print the channel list once (was printed 3x by LakeDataset.__init__).
    channels_to_load = ['red', 'green', 'blue']
    if dataset_kwargs['use_nir']:
        channels_to_load.append('nir')
    if dataset_kwargs['use_swir16']:
        channels_to_load.append('swir16')
    if dataset_kwargs['use_mask']:
        channels_to_load.append('mask')
    print(f"\nLoading {len(channels_to_load)} channels from NC files: {channels_to_load}")

    # Pass labels: use remapped dict if available, otherwise read from CSV
    if labels_dict is not None:
        dataset_kwargs['labels_dict'] = labels_dict
    else:
        dataset_kwargs['labels_file'] = config["labels_csv"]
        dataset_kwargs['id_col'] = config.get("id_col", "new_id")
        dataset_kwargs['label_col'] = config.get("label_col", "label_rines")

    # Create datasets. Two paths:
    #   cache  — blosc2 uint16, normalized on the GPU (fast; see build_cache.py)
    #   legacy — float32 NetCDF decompressed per read (the ESSD path)
    use_cache = bool(config.get("use_cache", False))
    normalize_fn = None

    if use_cache:
        cache_root = config.get("cache_root")
        if not cache_root:
            raise ValueError("--use_cache requires --cache_root")
        from lakevision.data.cached_dataset import (
            CachedLakeDataset, normalize_batch, worker_init as _worker_init,
        )

        cache_kwargs = dict(
            bands=config.get("cache_bands") or ["B04", "B03", "B02"],
            mask=config.get("cache_mask"),
            labels_dict=labels_dict,
            seq_len=config.get("seq_len", 153),
            scalar_var=config.get("scalar_var", "p_water"),
            normalize_scalar=config.get("normalize_scalar", False),
        )
        print(f"\n--- CACHED DATASET ---")
        print(f"cache_root:       {cache_root}")
        print(f"bands:            {cache_kwargs['bands']}")
        print(f"mask:             {cache_kwargs['mask']}")
        print(f"scalar_var:       {cache_kwargs['scalar_var']} "
              f"(min-max: {cache_kwargs['normalize_scalar']})")

        train_dataset = CachedLakeDataset(cache_root, lake_ids=train_ids, **cache_kwargs)
        val_dataset = CachedLakeDataset(cache_root, lake_ids=val_ids, **cache_kwargs)
        test_dataset = CachedLakeDataset(cache_root, lake_ids=test_ids, **cache_kwargs)
        channels_to_load = list(cache_kwargs["bands"]) + (
            [cache_kwargs["mask"]] if cache_kwargs["mask"] else [])

        # Bind n_refl so the BOA offset is applied to the reflectance channels
        # only, never to an appended 0/1 mask channel.
        _n_refl = train_dataset.n_refl
        normalize_fn = partial(normalize_batch, n_refl=_n_refl)
    else:
        _worker_init = None
        preload_to_ram = config.get("preload_to_ram", False)
        train_dataset = LakeDataset(train_paths, preload_to_ram=preload_to_ram, **dataset_kwargs)
        val_dataset = LakeDataset(val_paths, preload_to_ram=False, **dataset_kwargs)
        test_dataset = LakeDataset(test_paths, preload_to_ram=False, **dataset_kwargs)

    # Wrap training set with augmentations if requested.
    # Default is random-per-epoch (one disk read per lake per epoch). The
    # deterministic 8x expansion re-reads each lake once per symmetry, which is
    # 8x the I/O for the same augmentation distribution; it is kept only to
    # reproduce the ESSD augmented runs via --augment_mode expand.
    if config.get("augment", False):
        base_size = len(train_dataset)
        augment_mode = config.get("augment_mode", "random")
        print(f"\n--- AUGMENTATION ---")
        print(f"Augmentations: {list(AUGMENTATIONS.keys())}")
        if augment_mode == "expand":
            train_dataset = AugmentedDatasetWrapper(train_dataset)
            print(f"Mode: expand (deterministic 8x — ESSD-compatible)")
            print(f"Training set: {base_size} -> {len(train_dataset)} samples "
                  f"({len(train_dataset) // base_size}x)")
            print(f"NOTE: this reads every lake {len(train_dataset) // base_size}x per epoch.")
        else:
            train_dataset = RandomD4Dataset(train_dataset, seed=seed)
            print(f"Mode: random (one random D4 symmetry per sample per epoch)")
            print(f"Training set: {base_size} samples, 1 read per lake per epoch")

    # Create loaders.
    # Per-sample img_seq is ~640 MB (153 timesteps × 4 channels × 512² × float32),
    # so the in-flight queue is workers × prefetch_factor × batch_size × 640 MB.
    # That scales with batch_size: 12 workers × 2 prefetch is ~123 GB at bs=8 but
    # ~984 GB at bs=64. Pass --host_mem_budget_gb to size workers against the
    # budget instead of hardcoding them.
    # pin_memory=True enables async host→GPU transfer (overlap with compute).
    batch_size = config.get("batch_size", 8)
    prefetch_factor = 2

    # Approximate per-sample MB as handed to the collate function:
    # seq_len * channels * H * W * 4 bytes (float32).
    # The cache hands off uint16 (2 bytes); the legacy path hands off float32
    # (4 bytes). That factor of two is what decides how many workers fit.
    n_ch = len(channels_to_load)
    bytes_per_elem = 2 if use_cache else 4
    sample_mb = sample_size_mb(config.get("seq_len", 153), n_ch,
                               bytes_per_elem=bytes_per_elem)

    host_mem_budget_gb = config.get("host_mem_budget_gb")
    if host_mem_budget_gb:
        num_workers, prefetch_factor, projected_gb = plan_loader_workers(
            batch_size=batch_size,
            sample_mb=sample_mb,
            host_mem_budget_gb=host_mem_budget_gb,
            max_workers=config.get("num_workers", 16),
            prefetch_factor=prefetch_factor,
        )
        print(f"\n--- DATALOADER MEMORY PLAN ---")
        print(f"Per-sample:      {sample_mb:.0f} MB ({n_ch} channels x {config.get('seq_len', 153)} timesteps)")
        print(f"Budget:          {host_mem_budget_gb:.0f} GB")
        print(f"Chosen:          {num_workers} workers x {prefetch_factor} prefetch x bs {batch_size}")
        print(f"Projected queue: {projected_gb:.0f} GB")
    else:
        # Legacy path: fixed worker count. Safe at bs=8, OOMs the node at bs>=32.
        num_workers = config.get("num_workers", 12)
        projected_gb = num_workers * prefetch_factor * batch_size * sample_mb / 1024
        print(f"\n--- DATALOADER ---")
        print(f"{num_workers} workers x {prefetch_factor} prefetch x bs {batch_size} "
              f"-> ~{projected_gb:.0f} GB queued")
        if projected_gb > 200:
            print(f"  WARNING: projected queue is {projected_gb:.0f} GB. Pass "
                  f"--host_mem_budget_gb to size workers against your --mem "
                  f"instead of guessing.")

    # persistent_workers MUST be False here: with separate train/val loaders,
    # persistence keeps train workers alive while val workers spawn — doubling
    # worker processes and OOM-killing the job at the train->val transition.
    # The ~5-10s/epoch worker spin-up tax is the cost of safety.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        worker_init_fn=_worker_init,   # pins blosc2 to 1 thread per worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,
        worker_init_fn=_worker_init,
    )

    # Create model
    num_classes = config.get("num_classes", 5)
    seq_len = config.get("seq_len", 153)
    model = LakeDrainageClassifier(
        use_imgseq=config.get("use_imgseq", True),
        use_areaseq=config.get("use_areaseq", True),
        use_cloudyseq=config.get("use_cloudyseq", False),
        learn_area_weights=config.get("learn_area_weights", False),
        learn_cloudy_weights=config.get("learn_cloudy_weights", False),
        seq_len=seq_len,
        use_nir=config.get("use_nir", False),
        use_swir16=config.get("use_swir16", False),
        attention_type=config.get("attention_type", "none"),
        num_classes=num_classes,
        frontcnn_base_channels=config.get("frontcnn_base_channels", 8),
        frontcnn_num_layers=config.get("frontcnn_num_layers", 4),
        frontcnn_out_hw=config.get("frontcnn_out_hw"),
        clstm_hidden=config.get("clstm_hidden", 32),
        clstm_kernel=config.get("clstm_kernel", 3),
        slstm_hidden=config.get("slstm_hidden", 16),
        slstm_num_layers=config.get("slstm_num_layers", 1),
        slstm_dropout=config.get("slstm_dropout", 0.0),
        classhead_hidden=config.get("classhead_hidden", 64),
        classhead_dropout=config.get("classhead_dropout", 0.0),
        pool_type=config.get("pool_type", "avg"),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    ).to(device)

    # Print model summary
    trainable_params, total_params = count_parameters(model)
    print("\n--- MODEL SUMMARY ---")
    print(model)
    print(f"\nTrainable parameters: {trainable_params:,}")
    print(f"Total parameters:     {total_params:,}")
    print("=" * 70)

    # Loss and optimizer (weighted by inverse class frequency)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.0),
    )

    # ESSD baseline uses a fixed learning rate (no scheduler) by default.
    # The --use_scheduler flag remains available for ablation experiments.
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    # Training loop
    best_val_loss = float("inf")
    best_val_f1 = -float("inf")

    # Checkpoint paths derived from --save_path.
    #   <stem>.pth            → best val loss (original behavior)
    #   <stem>_bestf1.pth     → best val macro F1
    #   <stem>_epoch{E:03d}.pth → periodic every N epochs
    _save_path = config.get("save_path")
    if _save_path:
        _sp = Path(_save_path)
        best_loss_path = _sp
        best_f1_path = _sp.with_name(f"{_sp.stem}_bestf1{_sp.suffix}")
        def _periodic_path(epoch_1indexed):
            return _sp.with_name(f"{_sp.stem}_epoch{epoch_1indexed:03d}{_sp.suffix}")
    else:
        best_loss_path = best_f1_path = None
        _periodic_path = lambda e: None

    periodic_every = 5  # save every N epochs
    epochs = config.get("epochs", 50)
    epoch_times = []
    training_start_time = time.time()

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        accumulation_steps = config.get("accumulation_steps", 1)
        amp = config.get("amp", False)
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, num_classes,
            accumulation_steps, amp=amp, normalize_fn=normalize_fn)
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device, num_classes, amp=amp,
            normalize_fn=normalize_fn)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if scheduler:
            scheduler.step(val_loss)

        # Logging
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_loss,
            "val_acc": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_sec": epoch_time,
        }

        # Per-class metrics
        for class_name in CLASS_NAMES[:num_classes]:
            log_dict[f"train_f1_{class_name}"] = train_metrics[f"f1_{class_name}"]
            log_dict[f"val_f1_{class_name}"] = val_metrics[f"f1_{class_name}"]

        # Calculate estimated time remaining
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs} | Time: {epoch_time:.1f}s | ETA: {eta_str}")
        print(f"  {'':12} {'Loss':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print(f"  {'Train':12} {train_loss:>8.4f} {train_metrics['accuracy']:>8.3f} {train_metrics['precision_macro']:>8.3f} {train_metrics['recall_macro']:>8.3f} {train_metrics['f1_macro']:>8.3f}")
        print(f"  {'Val':12} {val_loss:>8.4f} {val_metrics['accuracy']:>8.3f} {val_metrics['precision_macro']:>8.3f} {val_metrics['recall_macro']:>8.3f} {val_metrics['f1_macro']:>8.3f}")

        # Print confusion matrix and per-class metrics every 10 epochs (and first epoch)
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0:
            cm = val_metrics['confusion_matrix']
            print(f"\n  Validation Confusion Matrix (epoch {epoch+1}):")
            print(f"  Classes: {CLASS_NAMES[:num_classes]}")
            print(f"  (rows=true, cols=pred)")
            for i, row in enumerate(cm):
                print(f"    {CLASS_NAMES[i]}: {row}")
            # Per-class metrics table
            print(f"\n  Per-class metrics:")
            print(f"  {'Class':8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
            for i, class_name in enumerate(CLASS_NAMES[:num_classes]):
                print(f"  {class_name:8} {val_metrics[f'precision_{class_name}']:>8.3f} {val_metrics[f'recall_{class_name}']:>8.3f} {val_metrics[f'f1_{class_name}']:>8.3f}")

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(log_dict)

        # --- Checkpointing ---
        # 1. Best val loss (canonical checkpoint — kept as the path the user passed)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if best_loss_path is not None:
                torch.save(model.state_dict(), best_loss_path)
                print(f"  Saved best-val-loss model (val_loss={val_loss:.4f})")

        # 2. Best val macro F1 (separate file)
        val_f1 = val_metrics["f1_macro"]
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if best_f1_path is not None:
                torch.save(model.state_dict(), best_f1_path)
                print(f"  Saved best-val-F1 model (val_f1_macro={val_f1:.4f})")

        # 3. Periodic snapshot every N epochs (for post-hoc analysis)
        epoch_1indexed = epoch + 1
        if epoch_1indexed % periodic_every == 0 and best_loss_path is not None:
            periodic = _periodic_path(epoch_1indexed)
            torch.save(model.state_dict(), periodic)
            print(f"  Saved periodic snapshot: {periodic.name}")

    # Training timing summary
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Total training time: {total_training_time / 3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"  Average epoch time:  {avg_epoch_time:.1f} seconds")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print(f"  Best val F1 (macro): {best_val_f1:.4f}")
    print("=" * 70)

    # Final test evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=False,  # one-shot eval; no benefit from persistence
        worker_init_fn=_worker_init,
    )

    # Load best model for test evaluation
    if config.get("save_path") and Path(config["save_path"]).exists():
        model.load_state_dict(torch.load(config["save_path"], map_location=device))
        print("Loaded best model for test evaluation")

    test_loss, test_metrics = evaluate(
        model, test_loader, criterion, device, num_classes,
        amp=config.get("amp", False), normalize_fn=normalize_fn)

    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):{test_metrics['f1_macro']:.4f}")
    print(f"\nPer-class F1:")
    for class_name in CLASS_NAMES[:num_classes]:
        print(f"  {class_name}: {test_metrics[f'f1_{class_name}']:.4f}")
    print(f"\nConfusion Matrix (rows=true, cols=pred):")
    print(f"Classes: {CLASS_NAMES[:num_classes]}")
    print(test_metrics['confusion_matrix'])

    # Print final save path confirmation
    if config.get("save_path"):
        save_path = Path(config["save_path"])
        if save_path.exists():
            file_size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"\nModel saved to: {save_path}")
            print(f"Model file size: {file_size_mb:.2f} MB")
        else:
            print(f"\nWARNING: Expected model file not found at: {save_path}")

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
        })
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["best_val_f1_macro"] = best_val_f1
        wandb.summary["test_acc"] = test_metrics["accuracy"]
        wandb.summary["test_f1_macro"] = test_metrics["f1_macro"]

    return best_val_loss, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LakeDrainageClassifier")
    # Required paths
    parser.add_argument("--labels_csv", type=str, required=True, nargs='+',
                        help="Path(s) to labels CSV file(s). For essd_5class "
                             "mode, multiple files can be merged (union of lakes).")
    parser.add_argument("--test_labels_csv", type=str, default=None, nargs='+',
                        help="Optional path(s) to CSV(s) whose lakes form a "
                             "held-out test set (cross-year baseline). When "
                             "set, labels_csv is split 80/20 into train/val.")
    parser.add_argument("--nc_dir", type=str, default=None,
                        help="Directory containing lake NC files (legacy path; "
                             "not needed with --use_cache)")

    # ------------------------- blosc2 cache backend -------------------------
    # Built by engine/preprocessing/build_cache.py. Stores uint16 raw DN and
    # defers (DN/DN_SCALE + boa)/10000 to the GPU, which halves the dataloader
    # queue and is what makes bs=32-64 fit in host RAM.
    parser.add_argument("--use_cache", action="store_true", default=False,
                        help="Read from the blosc2 cache instead of NetCDF.")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Cache directory (required with --use_cache).")
    parser.add_argument("--cache_bands", nargs="+", default=None,
                        help="Bands to load from the cache, in model channel "
                             "order. Default: B04 B03 B02 (red, green, blue).")
    parser.add_argument("--cache_mask", type=str, default=None,
                        choices=["lake_boundary", "water_mask_ndwi"],
                        help="Append a mask as a trailing channel. "
                             "'lake_boundary' is the static Dunmire polygon; "
                             "'water_mask_ndwi' is the per-timestep NDWI mask.")
    parser.add_argument("--scalar_var", type=str, default="p_water",
                        help="Per-timestep scalar feeding area_seq. 'p_water' "
                             "is the Dunmire 2025 S2_water fractional extent.")
    parser.add_argument("--normalize_scalar", action="store_true", default=False,
                        help="Min-max the scalar per sample. OFF by default: "
                             "p_water is already a physical [0,1] fraction, and "
                             "per-sample min-max would erase cross-lake "
                             "amplitude (a lake that only half-fills would look "
                             "like one that fills completely).")

    # -------------------------------------------------------------------
    # ESSD baseline defaults
    #
    # Defaults below are hard-coded to the canonical ESSD baseline so the
    # SLURM scripts can stay short and the paper can cite "default values"
    # without a long qualifier. Override explicitly via CLI for ablations.
    #
    # 5-class schema (ND/HF/MD/LD/CD), 400 epochs, bs=8, bf16 AMP,
    # lr=1e-4 fixed (no scheduler), imgseq+areaseq+mask active (no cloudyseq),
    # attention off, 4-layer FrontCNN. Seed 42 throughout.
    # -------------------------------------------------------------------

    # Label configuration
    parser.add_argument("--id_col", type=str, default="lake_id",
                        help="Column name for lake IDs in CSV")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Column name for labels in CSV")
    parser.add_argument("--label_mode", type=str, default="essd_5class",
                        choices=["original", "ed_split", "essd_5class"],
                        help="Label scheme: 'original' (ND/ED/LD/CD), "
                             "'ed_split' (ND/LD+MD/HF/CD), or "
                             "'essd_5class' (ND/HF/MD/LD/CD — reads string labels "
                             "from the sat-tile-stack GUI CSV; ESSD default)")
    parser.add_argument("--edm_edf_col", type=str, default="edm_edf",
                        help="Column name for moulin/hydrofracture indicator (used with --label_mode ed_split)")
    parser.add_argument("--merge_classes", type=str, nargs=2, default=None,
                        metavar=("CLASS_A", "CLASS_B"),
                        help="Merge two classes into one. CLASS_B is folded into CLASS_A "
                             "and the merged class is named CLASS_A+CLASS_B (e.g. "
                             "--merge_classes MD LD produces a 4-class model with MDLD). "
                             "Class names must match the label_mode scheme.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Fixed learning rate (no scheduler in ESSD baseline)")
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use bf16 mixed-precision autocast for forward/backward "
                             "(A100+). Default True for ESSD baseline.")
    parser.add_argument("--no_amp", action="store_false", dest="amp",
                        help="Disable bf16 AMP (forces fp32 training)")
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Enable ReduceLROnPlateau LR scheduler "
                             "(off by default in ESSD baseline).")
    parser.add_argument("--host_mem_budget_gb", type=float, default=None,
                        help="RAM the DataLoader queue may occupy, in GB. When "
                             "set, num_workers is chosen so that "
                             "workers x prefetch x batch_size x sample_size fits "
                             "the budget — which is what keeps large batches from "
                             "OOM-killing the node. Use ~65%% of --mem, leaving "
                             "room for page cache and the parent process "
                             "(e.g. --mem=500G -> --host_mem_budget_gb 325). "
                             "When unset, --num_workers is used verbatim.")
    parser.add_argument("--num_workers", type=int, default=12,
                        help="DataLoader workers. Each worker buffers prefetch_factor "
                             "× batch_size samples (~640 MB each), so RAM grows as "
                             "workers × prefetch × batch × 640MB. Default 12 paired "
                             "with --cpus-per-task=16 and --mem=256GB; gives ~35%% "
                             "headroom and saturates the GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_lakes", type=int, default=None,
                        help="Cap each split (train/val/test) at this many lakes. "
                             "Useful for pilot/smoke-test runs.")
    parser.add_argument("--max_train_lakes", type=int, default=None,
                        help="Cap only the train set at this many lakes "
                             "(learning-curve studies).")
    parser.add_argument("--train_ids_file", type=str, default=None,
                        help="Path to JSON list of train lake IDs. Overrides "
                             "--labels_csv splitting. Must be used with "
                             "--val_ids_file and --test_ids_file.")
    parser.add_argument("--val_ids_file", type=str, default=None,
                        help="Path to JSON list of val lake IDs.")
    parser.add_argument("--test_ids_file", type=str, default=None,
                        help="Path to JSON list of test lake IDs.")

    # Split configuration (ignored when --train_ids_file etc. are set)
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Fraction of labeled lakes used for training "
                             "(crossyear mode defaults to 0.8).")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="Ignored in crossyear mode.")
    parser.add_argument("--no_stratify", action="store_true", default=False,
                        help="Disable class-stratified splitting.")

    # Data configuration
    parser.add_argument("--seq_len", type=int, default=153,
                        help="Sequence length for temporal window")
    parser.add_argument("--band_stats", type=str, default=None,
                        help="Path to band statistics JSON for normalization")
    parser.add_argument("--cloudy_seq_var", type=str, default="cloudy_seq_rgb",
                        help="Name of cloudy_seq variable in NC files")
    parser.add_argument("--augment", action="store_true", default=False,
                        help="Apply spatial augmentations (random flips + 90-degree rotations) to training data")
    parser.add_argument("--augment_mode", type=str, default="random",
                        choices=["random", "expand"],
                        help="'random' (default): one random D4 symmetry per "
                             "sample per epoch — 1 disk read per lake per epoch. "
                             "'expand': deterministic 8x dataset expansion, which "
                             "re-reads every lake 8x per epoch (8x the I/O for the "
                             "same augmentation distribution). Use 'expand' only "
                             "to reproduce the ESSD augmented runs.")

    # Model feature flags
    parser.add_argument("--use_imgseq", action="store_true", default=True,
                        help="Use image sequence processing")
    parser.add_argument("--no_imgseq", action="store_false", dest="use_imgseq",
                        help="Disable image sequence processing")
    parser.add_argument("--use_areaseq", action="store_true", default=True,
                        help="Use water area sequence")
    parser.add_argument("--no_areaseq", action="store_false", dest="use_areaseq",
                        help="Disable water area sequence")
    parser.add_argument("--use_cloudyseq", action="store_true", default=False,
                        help="Use cloudy sequence (default off for ESSD baseline)")
    parser.add_argument("--no_cloudyseq", action="store_false", dest="use_cloudyseq",
                        help="Disable cloudy sequence")
    parser.add_argument("--learn_area_weights", action="store_true", default=False,
                        help="Learn per-timestep weights for area_seq")
    parser.add_argument("--learn_cloudy_weights", action="store_true", default=False,
                        help="Learn per-timestep weights for cloudy_seq (requires --use_cloudyseq)")
    parser.add_argument("--use_nir", action="store_true", default=False,
                        help="Include NIR band")
    parser.add_argument("--use_swir16", action="store_true", default=False,
                        help="Include SWIR16 band")
    parser.add_argument("--no_mask", action="store_true", default=False,
                        help="Disable mask band (required for raw sat-tile-stack NC files)")

    # Model architecture
    parser.add_argument("--attention_type", type=str, default="none",
                        choices=["none", "spatial", "full", "arch"],
                        help="Attention mechanism type")
    parser.add_argument("--num_classes", type=int, default=5,
                        help="Number of output classes (default 5 for ESSD baseline)")
    parser.add_argument("--frontcnn_base_channels", type=int, default=8)
    parser.add_argument("--frontcnn_num_layers", type=int, default=4)
    parser.add_argument("--frontcnn_out_hw", type=str, default=None,
                        help="Spatial size fed to the CLSTM, as 'H,W'. Default "
                             "(unset) keeps the conv stack's own output — 32x32 "
                             "for 512x512 input with 4 layers. Pooling DOWN is "
                             "allowed; requesting a size larger than the conv "
                             "output now raises instead of silently upsampling. "
                             "The ESSD baselines ran '64,64' with 4 layers, which "
                             "upsampled 32->64 and cost 4x in the CLSTM for no "
                             "gain; pass '64,64' only to reproduce those runs.")
    parser.add_argument("--clstm_hidden", type=int, default=32)
    parser.add_argument("--slstm_hidden", type=int, default=16)
    parser.add_argument("--classhead_hidden", type=int, default=64)
    parser.add_argument("--classhead_dropout", type=float, default=0.3)

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False,
                        help="Use gradient checkpointing to reduce GPU memory (trades compute for memory)")
    parser.add_argument("--accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * accumulation_steps)")
    parser.add_argument("--preload_to_ram", action="store_true", default=False,
                        help="Preload training data to RAM (requires ~1GB per lake, ~700GB for full training set)")

    # Output
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save best model weights")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="lake-vision")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()
    if args.use_cache and not args.cache_root:
        parser.error("--use_cache requires --cache_root")
    if not args.use_cache and not args.nc_dir:
        parser.error("--nc_dir is required unless --use_cache is set")
    config = vars(args)
    # Translate --no_stratify (action flag) to config["stratify"] so existing
    # code that reads config.get("stratify", True) keeps working.
    config["stratify"] = not config.pop("no_stratify", False)

    # "64,64" -> (64, 64); unset stays None (= keep the conv stack's own output)
    if config.get("frontcnn_out_hw"):
        try:
            h, w = (int(v) for v in str(config["frontcnn_out_hw"]).split(","))
        except ValueError:
            parser.error("--frontcnn_out_hw must be 'H,W' (e.g. '32,32')")
        config["frontcnn_out_hw"] = (h, w)

    # Initialize wandb
    if WANDB_AVAILABLE and not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
        )
        # Allow sweep to override config
        config = dict(wandb.config)
        # Ensure required paths are set
        config["labels_csv"] = args.labels_csv
        config["nc_dir"] = args.nc_dir
        config["use_cache"] = args.use_cache
        config["cache_root"] = args.cache_root
        config["cache_bands"] = args.cache_bands
        config["cache_mask"] = args.cache_mask
        config["test_labels_csv"] = args.test_labels_csv
        config["train_ids_file"] = args.train_ids_file
        config["val_ids_file"] = args.val_ids_file
        config["test_ids_file"] = args.test_ids_file

    train(config)

    if WANDB_AVAILABLE and wandb.run is not None:
        run_dir = Path(wandb.run.dir).parent
        print(f"\n{'='*60}")
        print(f"WANDB RUN COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {wandb.run.id}")
        print(f"Run directory: {run_dir}")
        print(f"\nTo sync this run to wandb.ai, use:")
        print(f"  wandb sync {run_dir}")
        print(f"{'='*60}\n")
        wandb.finish()


if __name__ == "__main__":
    main()
