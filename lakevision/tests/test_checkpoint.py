"""
Tests for checkpoint loading across both formats.

Written after job 39284581: training ran to completion at bs=32, then the run's
own final test evaluation crashed because the save format had been changed to a
provenance dict without updating the load path.

    RuntimeError: Error(s) in loading state_dict for LakeDrainageClassifier:
        Unexpected key(s) in state_dict: "state_dict", "config", "epoch", ...

The regression that matters most here is the OLD format: the frozen ESSD tags
saved bare state_dicts, and those must keep loading forever.
"""
import pytest
import torch
import torch.nn as nn

from lakevision.models.checkpoint import describe_checkpoint, load_checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        return self.fc(x)


def provenance(model, **over):
    d = {
        "state_dict": model.state_dict(),
        "config": {"batch_size": 32, "temporal_readout": "mean", "lr": 3e-4},
        "epoch": 4, "metrics": {"f1_macro": 0.25}, "fold": 2, "seed": 42,
        "git_sha": "8d6113a", "class_names": ["ND", "HF", "MD", "LD", "CD"],
        "num_classes": 5, "wandb_run_id": "abc123",
    }
    d.update(over)
    return d


class TestLoadCheckpoint:

    def test_bare_state_dict_still_loads(self, tmp_path):
        """The ESSD tags' format. Breaking this breaks paper reproduction."""
        m = Tiny()
        p = tmp_path / "bare.pth"
        torch.save(m.state_dict(), p)

        state, meta = load_checkpoint(p)
        assert meta is None
        assert Tiny().load_state_dict(state) is not None

    def test_provenance_dict_loads_and_returns_metadata(self, tmp_path):
        m = Tiny()
        p = tmp_path / "prov.pth"
        torch.save(provenance(m), p)

        state, meta = load_checkpoint(p)
        assert meta is not None
        assert meta["git_sha"] == "8d6113a" and meta["fold"] == 2
        assert "state_dict" not in meta
        Tiny().load_state_dict(state)          # must not raise

    def test_reproduces_the_39284581_crash_without_the_helper(self, tmp_path):
        """Lock in WHY this module exists: the naive path really does fail."""
        m = Tiny()
        p = tmp_path / "prov.pth"
        torch.save(provenance(m), p)
        with pytest.raises(RuntimeError, match="Unexpected key"):
            Tiny().load_state_dict(torch.load(p))

    def test_round_trip_preserves_weights(self, tmp_path):
        m = Tiny()
        with torch.no_grad():
            m.fc.weight.fill_(0.5)
        p = tmp_path / "prov.pth"
        torch.save(provenance(m), p)

        loaded = Tiny()
        state, _ = load_checkpoint(p)
        loaded.load_state_dict(state)
        assert torch.equal(loaded.fc.weight, m.fc.weight)

    def test_weights_only_true_falls_back_for_provenance_dicts(self, tmp_path):
        """run_inference.py passes weights_only=True. A provenance dict carries a
        config, which a strict load can reject — it must not hard-fail."""
        m = Tiny()
        p = tmp_path / "prov.pth"
        torch.save(provenance(m), p)
        state, meta = load_checkpoint(p, weights_only=True)
        assert meta["git_sha"] == "8d6113a"
        Tiny().load_state_dict(state)

    def test_weights_only_true_still_works_on_bare(self, tmp_path):
        m = Tiny()
        p = tmp_path / "bare.pth"
        torch.save(m.state_dict(), p)
        state, meta = load_checkpoint(p, weights_only=True)
        assert meta is None
        Tiny().load_state_dict(state)


class TestDescribe:

    def test_bare_is_flagged_as_lacking_provenance(self, tmp_path):
        assert "no provenance" in describe_checkpoint(tmp_path / "x.pth", None)

    def test_summary_includes_sha_and_fold(self, tmp_path):
        s = describe_checkpoint(tmp_path / "x.pth", provenance(Tiny()))
        assert "git_sha=8d6113a" in s and "fold=2" in s

    def test_missing_fields_are_skipped_not_printed_as_none(self, tmp_path):
        s = describe_checkpoint(tmp_path / "x.pth",
                                provenance(Tiny(), git_sha=None, fold=None))
        assert "None" not in s
