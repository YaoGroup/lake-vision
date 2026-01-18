"""
Training script for LakeDrainageClassifier with wandb integration.

Usage:
    python run_training.py --labels_csv /path/to/labels.csv --nc_dir /path/to/nc/files

For wandb sweeps:
    wandb sweep sweep.yaml
    wandb agent <sweep_id>
"""
import argparse
import random
import sys
import time
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
CLASS_NAMES = ['ND', 'ED', 'LD', 'CD']  # 0: No Drainage, 1: Englacial, 2: Lateral, 3: Crevasse


def create_splits(
    labels_csv: str,
    id_col: str = 'new_id',
    label_col: str = 'label_rines',
    train_ratio: float = 0.7,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
    stratify: bool = True,
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

    Returns:
        tuple: (train_ids, val_ids, test_ids) as lists of lake IDs
    """
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


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        img_seq, area_seq, cloudy_seq, labels, _ = batch

        img_seq = img_seq.to(device)
        area_seq = area_seq.to(device)
        cloudy_seq = cloudy_seq.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(img_seq, area_seq, cloudy_seq)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, loader, criterion, device, num_classes=4):
    """Evaluate model and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            img_seq, area_seq, cloudy_seq, labels, _ = batch

            img_seq = img_seq.to(device)
            area_seq = area_seq.to(device)
            cloudy_seq = cloudy_seq.to(device)
            labels = labels.to(device)

            logits = model(img_seq, area_seq, cloudy_seq)
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


def train(config: dict):
    """Main training function."""
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"Random seed: {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create data splits
    train_ids, val_ids, test_ids = create_splits(
        config["labels_csv"],
        id_col=config.get("id_col", "new_id"),
        label_col=config.get("label_col", "label_rines"),
        train_ratio=config.get("train_ratio", 0.7),
        val_ratio=config.get("val_ratio", 0.20),
        test_ratio=config.get("test_ratio", 0.10),
        seed=seed,
        stratify=config.get("stratify", True),
    )

    # Build file paths from IDs
    nc_dir = Path(config["nc_dir"])
    train_paths = [nc_dir / f"{lid}.nc" for lid in train_ids]
    val_paths = [nc_dir / f"{lid}.nc" for lid in val_ids]
    test_paths = [nc_dir / f"{lid}.nc" for lid in test_ids]

    # Filter to existing files
    train_paths = [p for p in train_paths if p.exists()]
    val_paths = [p for p in val_paths if p.exists()]
    test_paths = [p for p in test_paths if p.exists()]

    print(f"Found files: train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}")

    # Dataset configuration
    dataset_kwargs = {
        'seq_len': config.get("seq_len", 153),
        'labels_file': config["labels_csv"],
        'id_col': config.get("id_col", "new_id"),
        'label_col': config.get("label_col", "label_rines"),
        'use_nir': config.get("use_nir", False),
        'use_swir16': config.get("use_swir16", False),
        'use_swir22': config.get("use_swir22", False),
        'band_stats': config.get("band_stats"),
        'cloudy_seq_var': config.get("cloudy_seq_var", "cloudy_seq_rgb"),
    }

    # Create datasets
    train_dataset = LakeDataset(train_paths, **dataset_kwargs)
    val_dataset = LakeDataset(val_paths, **dataset_kwargs)
    test_dataset = LakeDataset(test_paths, **dataset_kwargs)

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=True,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Create model
    num_classes = config.get("num_classes", 4)
    model = LakeDrainageClassifier(
        use_imgseq=config.get("use_imgseq", True),
        use_areaseq=config.get("use_areaseq", True),
        use_cloudyseq=config.get("use_cloudyseq", False),
        use_nir=config.get("use_nir", False),
        use_swir16=config.get("use_swir16", False),
        use_swir22=config.get("use_swir22", False),
        attention_type=config.get("attention_type", "none"),
        num_classes=num_classes,
        frontcnn_base_channels=config.get("frontcnn_base_channels", 8),
        frontcnn_num_layers=config.get("frontcnn_num_layers", 4),
        clstm_hidden=config.get("clstm_hidden", 32),
        clstm_kernel=config.get("clstm_kernel", 3),
        slstm_hidden=config.get("slstm_hidden", 16),
        slstm_num_layers=config.get("slstm_num_layers", 1),
        slstm_dropout=config.get("slstm_dropout", 0.0),
        classhead_hidden=config.get("classhead_hidden", 64),
        classhead_dropout=config.get("classhead_dropout", 0.0),
        pool_type=config.get("pool_type", "avg"),
    ).to(device)

    print(model)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.get("lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.0),
    )

    # Learning rate scheduler
    scheduler = None
    if config.get("use_scheduler", False):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

    # Training loop
    best_val_loss = float("inf")
    epochs = config.get("epochs", 50)
    epoch_times = []
    training_start_time = time.time()

    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, num_classes)

        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)

        if scheduler:
            scheduler.step(val_loss)

        # Logging
        log_dict = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
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
            log_dict[f"val_f1_{class_name}"] = val_metrics[f"f1_{class_name}"]

        # Calculate estimated time remaining
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        eta_seconds = avg_epoch_time * remaining_epochs
        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m"

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"Acc: {val_metrics['accuracy']:.3f} | "
            f"F1: {val_metrics['f1_macro']:.3f} | "
            f"Time: {epoch_time:.1f}s | ETA: {eta_str}"
        )

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(log_dict)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if config.get("save_path"):
                torch.save(model.state_dict(), config["save_path"])
                print(f"  Saved best model (val_loss={val_loss:.4f})")

    # Training timing summary
    total_training_time = time.time() - training_start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    print("\n" + "=" * 70)
    print("Training Complete!")
    print(f"  Total training time: {total_training_time / 3600:.2f} hours ({total_training_time:.1f} seconds)")
    print(f"  Average epoch time:  {avg_epoch_time:.1f} seconds")
    print(f"  Best val loss:       {best_val_loss:.4f}")
    print("=" * 70)

    # Final test evaluation
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=config.get("num_workers", 4),
    )

    # Load best model for test evaluation
    if config.get("save_path") and Path(config["save_path"]).exists():
        model.load_state_dict(torch.load(config["save_path"], map_location=device))
        print("Loaded best model for test evaluation")

    test_loss, test_metrics = evaluate(model, test_loader, criterion, device, num_classes)

    print(f"\nTest Results:")
    print(f"  Loss:      {test_loss:.4f}")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision_macro']:.4f}")
    print(f"  Recall:    {test_metrics['recall_macro']:.4f}")
    print(f"  F1 (macro):{test_metrics['f1_macro']:.4f}")
    print(f"\nPer-class F1:")
    for class_name in CLASS_NAMES[:num_classes]:
        print(f"  {class_name}: {test_metrics[f'f1_{class_name}']:.4f}")
    print(f"\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            "test_loss": test_loss,
            "test_acc": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
        })
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.summary["test_acc"] = test_metrics["accuracy"]
        wandb.summary["test_f1_macro"] = test_metrics["f1_macro"]

    return best_val_loss, test_metrics


def main():
    parser = argparse.ArgumentParser(description="Train LakeDrainageClassifier")
    # Required paths
    parser.add_argument("--labels_csv", type=str, required=True,
                        help="Path to labels CSV file")
    parser.add_argument("--nc_dir", type=str, required=True,
                        help="Directory containing lake NC files")

    # Label configuration
    parser.add_argument("--id_col", type=str, default="new_id",
                        help="Column name for lake IDs in CSV")
    parser.add_argument("--label_col", type=str, default="label_rines",
                        help="Column name for labels in CSV")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--use_scheduler", action="store_true",
                        help="Use learning rate scheduler")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Data configuration
    parser.add_argument("--seq_len", type=int, default=153,
                        help="Sequence length for temporal window")
    parser.add_argument("--band_stats", type=str, default=None,
                        help="Path to band statistics JSON for normalization")
    parser.add_argument("--cloudy_seq_var", type=str, default="cloudy_seq_rgb",
                        help="Name of cloudy_seq variable in NC files")

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
                        help="Use cloudy sequence")
    parser.add_argument("--use_nir", action="store_true", default=False,
                        help="Include NIR band")
    parser.add_argument("--use_swir16", action="store_true", default=False,
                        help="Include SWIR16 band")
    parser.add_argument("--use_swir22", action="store_true", default=False,
                        help="Include SWIR22 band")

    # Model architecture
    parser.add_argument("--attention_type", type=str, default="none",
                        choices=["none", "spatial", "full", "arch"],
                        help="Attention mechanism type")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes")
    parser.add_argument("--frontcnn_base_channels", type=int, default=8)
    parser.add_argument("--frontcnn_num_layers", type=int, default=4)
    parser.add_argument("--clstm_hidden", type=int, default=32)
    parser.add_argument("--slstm_hidden", type=int, default=16)
    parser.add_argument("--classhead_hidden", type=int, default=64)
    parser.add_argument("--classhead_dropout", type=float, default=0.0)

    # Output
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save best model weights")

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="lake-vision")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")

    args = parser.parse_args()
    config = vars(args)

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
