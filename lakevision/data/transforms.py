"""
Spatial augmentations for multi-temporal image sequences.

All transforms apply the same spatial operation consistently across
all timesteps so that temporal coherence is preserved.
"""
import torch


# Each augmentation is a function: img_seq [T, C, H, W] -> img_seq [T, C, H, W]
AUGMENTATIONS = {
    'rot90':   lambda x: x.rot90(1, [-2, -1]),   # 90 degrees clockwise
    'mirror':  lambda x: x.flip(-1),              # horizontal mirror
}


class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps an existing dataset to expand it with deterministic augmentations.

    Each sample in the base dataset is repeated len(augmentations) + 1 times:
    once as the original, and once per augmentation. The augmentation for each
    index is fixed (not random), so the dataset is fully deterministic.

    Args:
        base_dataset: The original dataset (e.g., LakeDataset)
        augmentations: Dict of {name: transform_fn}. Each transform_fn takes
            and returns a tensor of shape [T, C, H, W].
            Defaults to AUGMENTATIONS (rot90 + mirror).

    Example:
        >>> dataset = LakeDataset(...)          # 667 samples
        >>> aug_dataset = AugmentedDatasetWrapper(dataset)  # 667 x 3 = 2001 samples
    """

    def __init__(self, base_dataset, augmentations=None):
        self.base_dataset = base_dataset
        self.augmentations = augmentations or AUGMENTATIONS
        # [identity, aug1, aug2, ...]
        self.transforms = [None] + list(self.augmentations.values())
        self.n_versions = len(self.transforms)

    def __len__(self):
        return len(self.base_dataset) * self.n_versions

    def __getitem__(self, idx):
        base_idx = idx // self.n_versions
        aug_idx = idx % self.n_versions

        img_seq, area_seq, cloudy_seq, label, lake_id = self.base_dataset[base_idx]

        if self.transforms[aug_idx] is not None:
            img_seq = self.transforms[aug_idx](img_seq)

        return img_seq, area_seq, cloudy_seq, label, lake_id
