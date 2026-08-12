"""
Tests for spatial augmentation wrappers and DataLoader memory planning.

Covers two I/O regressions found in the ESSD pipeline:
  - AugmentedDatasetWrapper reads every lake once per D4 symmetry (8x the I/O).
  - The DataLoader queue was sized with a hardcoded worker count, so host RAM
    scaled with batch_size until the node OOMed.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lakevision.data.transforms import (
    AUGMENTATIONS,
    AugmentedDatasetWrapper,
    RandomD4Dataset,
)


class CountingDataset(torch.utils.data.Dataset):
    """Minimal stand-in for LakeDataset that records how often it is read."""

    def __init__(self, n=5, T=3, C=2, H=4, W=4):
        self.n = n
        self.shape = (T, C, H, W)
        self.reads = Counter()

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self.reads[idx] += 1
        img = torch.arange(torch.tensor(self.shape).prod()).float().reshape(self.shape)
        return img, torch.zeros(self.shape[0], 1), torch.zeros(self.shape[0], 1), 0, f"L{idx}"


class TestRandomD4Dataset:
    def test_length_matches_base(self):
        """Random mode does not inflate dataset length."""
        base = CountingDataset(n=5)
        assert len(RandomD4Dataset(base)) == 5

    def test_one_read_per_sample_per_epoch(self):
        """The whole point: 1 disk read per lake per epoch, not 8."""
        base = CountingDataset(n=5)
        ds = RandomD4Dataset(base)
        for i in range(len(ds)):
            ds[i]
        assert set(base.reads.values()) == {1}, f"expected 1 read each, got {base.reads}"

    def test_expand_mode_reads_eight_times(self):
        """Documents the behavior random mode replaces."""
        base = CountingDataset(n=5)
        ds = AugmentedDatasetWrapper(base)
        assert len(ds) == 5 * (len(AUGMENTATIONS) + 1)
        for i in range(len(ds)):
            ds[i]
        assert set(base.reads.values()) == {len(AUGMENTATIONS) + 1}

    def test_shape_preserved(self):
        """D4 symmetries keep [T, C, H, W] on square inputs."""
        ds = RandomD4Dataset(CountingDataset(n=3, H=6, W=6))
        img = ds[0][0]
        assert img.shape == (3, 2, 6, 6)

    def test_covers_all_symmetries_over_epochs(self):
        """Every symmetry is reachable, so regularization matches expand mode."""
        ds = RandomD4Dataset(CountingDataset(n=1), seed=0)
        seen = set()
        for epoch in range(400):
            ds.set_epoch(epoch)
            seen.add(ds[0][0].numpy().tobytes())
        assert len(seen) == len(AUGMENTATIONS) + 1, (
            f"expected {len(AUGMENTATIONS) + 1} distinct variants, saw {len(seen)}"
        )

    def test_seeded_is_reproducible(self):
        """Same seed + epoch + index yields the same symmetry."""
        a, b = RandomD4Dataset(CountingDataset(n=4), seed=7), RandomD4Dataset(CountingDataset(n=4), seed=7)
        a.set_epoch(3)
        b.set_epoch(3)
        assert torch.equal(a[2][0], b[2][0])


class TestLoaderMemoryPlan:
    """The queue is workers x prefetch x batch x sample; it must fit a budget."""

    @staticmethod
    def _plan(**kw):
        from lakevision.data.loader_plan import plan_loader_workers
        return plan_loader_workers(**kw)

    def test_queue_stays_within_budget(self):
        for batch in (8, 16, 32, 64):
            w, pf, gb = self._plan(batch_size=batch, sample_mb=640,
                                   host_mem_budget_gb=325)
            assert gb <= 325, f"bs={batch} projected {gb:.0f} GB over a 325 GB budget"

    def test_workers_shrink_as_batch_grows(self):
        small = self._plan(batch_size=8, sample_mb=640, host_mem_budget_gb=325)[0]
        large = self._plan(batch_size=64, sample_mb=640, host_mem_budget_gb=325)[0]
        assert large < small, "worker count must fall as batch size rises"

    def test_hardcoded_12_workers_would_have_oomed(self):
        """Regression: the old fixed 12x2 queue at bs=64 exceeds any real node."""
        legacy_gb = 12 * 2 * 64 * 640 / 1024
        assert legacy_gb > 500, "sanity: legacy config should blow past 500 GB"
        _, _, planned = self._plan(batch_size=64, sample_mb=640,
                                   host_mem_budget_gb=325)
        assert planned < legacy_gb

    def test_at_least_one_worker(self):
        """Absurd budgets degrade to 1 worker rather than 0 or negative."""
        w, _, _ = self._plan(batch_size=64, sample_mb=640, host_mem_budget_gb=1)
        assert w == 1

    def test_respects_max_workers(self):
        w, _, _ = self._plan(batch_size=1, sample_mb=1,
                             host_mem_budget_gb=10_000, max_workers=16)
        assert w == 16
