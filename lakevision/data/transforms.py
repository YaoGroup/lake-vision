"""
Spatial augmentations for multi-temporal image sequences.

All transforms apply the same spatial operation consistently across
all timesteps so that temporal coherence is preserved.
"""
import torch


# Each augmentation is a function: img_seq [T, C, H, W] -> img_seq [T, C, H, W]
# All 7 non-identity elements of the dihedral group D4 (symmetries of a square):
# 4 rotations (0°, 90°, 180°, 270°) × 2 reflections = 8 total, minus identity = 7
AUGMENTATIONS = {
    'rot90':       lambda x: x.rot90(1, [-2, -1]),               # 90° clockwise
    'rot180':      lambda x: x.rot90(2, [-2, -1]),               # 180°
    'rot270':      lambda x: x.rot90(3, [-2, -1]),               # 270° clockwise
    'flip_h':      lambda x: x.flip(-1),                          # horizontal flip
    'flip_v':      lambda x: x.flip(-2),                          # vertical flip
    'rot90_flip':  lambda x: x.rot90(1, [-2, -1]).flip(-1),      # 90° + horizontal flip
    'rot270_flip': lambda x: x.rot90(3, [-2, -1]).flip(-1),      # 270° + horizontal flip
}


class AugmentedDatasetWrapper(torch.utils.data.Dataset):
    """
    Wraps an existing dataset to expand it with deterministic augmentations.

    Each sample in the base dataset is repeated len(augmentations) + 1 times:
    once as the original, and once per augmentation. The augmentation for each
    index is fixed (not random), so the dataset is fully deterministic.

    .. warning::
        This multiplies **disk reads** by ``n_versions``, not just sample count:
        each of the 8 D4 variants calls ``base_dataset[base_idx]`` separately, so
        every lake is decoded 8x per epoch. At ~640 MB/sample that dominates
        epoch time. Prefer :class:`RandomD4Dataset`, which reads once per lake
        per epoch and picks a random symmetry — same expected augmentation
        distribution, 1/8 the I/O.

        Kept for reproducing the ESSD augmented runs, which used this class.

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

        # Tuple length varies by dataset: LakeDataset returns 5 elements,
        # CachedLakeDataset returns 6 (it appends the BOA offset). Transform
        # element 0 and pass the rest through untouched.
        sample = self.base_dataset[base_idx]
        if self.transforms[aug_idx] is not None:
            sample = (self.transforms[aug_idx](sample[0]),) + tuple(sample[1:])
        return sample


class RandomD4Dataset(torch.utils.data.Dataset):
    """
    Applies one uniformly-random D4 symmetry per sample, per access.

    Same length as the base dataset, so each lake is read from disk **once** per
    epoch instead of once per symmetry. Over many epochs a sample is seen under
    every symmetry with equal probability, which is the standard formulation of
    random spatial augmentation and gives the same regularization as the 8x
    deterministic expansion at 1/8 the I/O.

    The transform is applied identically across all timesteps, preserving
    temporal coherence.

    Args:
        base_dataset: The original dataset (e.g., LakeDataset)
        augmentations: Dict of {name: transform_fn}. Defaults to AUGMENTATIONS.
            The identity is included implicitly, so a sample has a
            1/(len+1) chance of being returned untransformed.
        seed: Optional base seed. When set, the symmetry chosen for a given
            (epoch, index) is reproducible; call :meth:`set_epoch` each epoch.
            When None, uses global torch RNG (which DataLoader workers seed
            per-epoch already).

    Example:
        >>> train = RandomD4Dataset(LakeDataset(...))   # same length, 8x fewer reads
    """

    def __init__(self, base_dataset, augmentations=None, seed=None):
        self.base_dataset = base_dataset
        self.augmentations = augmentations or AUGMENTATIONS
        self.transforms = [None] + list(self.augmentations.values())
        self.n_versions = len(self.transforms)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        """Advance the epoch so seeded runs draw fresh symmetries."""
        self.epoch = epoch

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Tuple length varies by dataset (LakeDataset 5, CachedLakeDataset 6);
        # transform element 0 and pass the rest through.
        sample = self.base_dataset[idx]

        if self.seed is None:
            aug_idx = int(torch.randint(self.n_versions, (1,)).item())
        else:
            g = torch.Generator().manual_seed(
                (self.seed * 1_000_003 + self.epoch) * 1_000_003 + idx
            )
            aug_idx = int(torch.randint(self.n_versions, (1,), generator=g).item())

        if self.transforms[aug_idx] is not None:
            sample = (self.transforms[aug_idx](sample[0]),) + tuple(sample[1:])
        return sample
