"""
Utility functions for lake data processing and splitting.
"""
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List
from sklearn.model_selection import train_test_split


def create_stratified_split(
    data_dir: Union[str, Path],
    labels_file: Union[str, Path],
    id_col: str = 'lake_id',
    label_col: str = 'label',
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[Path]]:
    """
    Create stratified train/val/test splits for lake data.

    Ensures each split has representative class proportions using
    stratified sampling based on labels.

    Args:
        data_dir: Directory containing processed .nc files
        labels_file: Path to labels CSV
        id_col: Column name for lake IDs in CSV
        label_col: Column name for labels in CSV
        val_size: Fraction for validation (default: 0.2)
        test_size: Fraction for test (default: 0.1)
        random_state: Random seed for reproducibility

    Returns:
        dict with 'train', 'val', 'test' keys, each containing list of file paths

    Example:
        >>> splits = create_stratified_split(
        ...     data_dir="datasets/processed/",
        ...     labels_file="labels/labels.csv",
        ...     id_col='new_id',
        ...     label_col='label_rines',
        ... )
        >>> print(f"Train: {len(splits['train'])} lakes")
        >>> print(f"Val: {len(splits['val'])} lakes")
        >>> print(f"Test: {len(splits['test'])} lakes")
    """
    # Load labels
    df = pd.read_csv(labels_file)
    df = df.dropna(subset=[id_col, label_col])

    # Get available files
    data_dir = Path(data_dir)
    available_files = {f.stem: f for f in data_dir.glob("*.nc")}

    # Filter to lakes that have both data and labels
    df = df[df[id_col].isin(available_files.keys())]

    if len(df) == 0:
        raise ValueError("No matching lakes found between labels and data files")

    lake_ids = df[id_col].values
    labels = df[label_col].values

    # Check if we have enough samples for stratification
    unique_labels, label_counts = pd.unique(labels, return_counts=True)
    min_count = min(label_counts)

    if min_count < 2:
        raise ValueError(
            f"Not enough samples for stratified split. "
            f"Class distribution: {dict(zip(unique_labels, label_counts))}"
        )

    # First split: train+val vs test
    if test_size > 0:
        train_val_ids, test_ids, train_val_labels, _ = train_test_split(
            lake_ids, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )
    else:
        train_val_ids = lake_ids
        train_val_labels = labels
        test_ids = []

    # Second split: train vs val
    if val_size > 0:
        val_fraction = val_size / (1 - test_size)  # Adjust for remaining data
        train_ids, val_ids, _, _ = train_test_split(
            train_val_ids, train_val_labels,
            test_size=val_fraction,
            stratify=train_val_labels,
            random_state=random_state
        )
    else:
        train_ids = train_val_ids
        val_ids = []

    # Convert to file paths
    splits = {
        'train': [available_files[lid] for lid in train_ids],
        'val': [available_files[lid] for lid in val_ids],
        'test': [available_files[lid] for lid in test_ids],
    }

    # Print summary
    print(f"Split summary:")
    print(f"  Train: {len(splits['train'])} lakes")
    print(f"  Val:   {len(splits['val'])} lakes")
    print(f"  Test:  {len(splits['test'])} lakes")

    # Print class distribution per split
    for split_name, file_list in splits.items():
        if len(file_list) > 0:
            split_ids = [f.stem for f in file_list]
            split_labels = df[df[id_col].isin(split_ids)][label_col].value_counts().sort_index()
            print(f"  {split_name} class distribution: {dict(split_labels)}")

    return splits
