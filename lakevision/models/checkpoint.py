"""
Checkpoint I/O that reads both formats this project has produced.

Two formats exist and BOTH must keep loading:

  bare      torch.save(model.state_dict(), path)
            Everything up to 2026-08-15, including the frozen ESSD tags. Those
            runs must stay reproducible, so this is not a format to migrate away
            from -- it is one to keep reading forever.

  provenance {state_dict, config, epoch, metrics, fold, seed, git_sha,
              class_names, num_classes, wandb_run_id}
            From 2026-08-15 on. A CV grid produces dozens of .pth files and a
            bare state_dict leaves no way to tell which config, fold, or commit
            made which file.

Job 39284581 found out the hard way that changing the save format without
changing the load path breaks the run's own final test evaluation:

    RuntimeError: Error(s) in loading state_dict for LakeDrainageClassifier:
        Unexpected key(s) in state_dict: "state_dict", "config", "epoch", ...

Every caller should go through :func:`load_checkpoint` rather than calling
``torch.load`` directly.
"""
from pathlib import Path
from typing import Optional, Tuple

import torch

# Metadata keys worth echoing when a checkpoint carries them. Order is the order
# they print in.
_SUMMARY_KEYS = ("git_sha", "epoch", "fold", "seed", "num_classes", "wandb_run_id")


def load_checkpoint(path, map_location=None, weights_only: Optional[bool] = None
                    ) -> Tuple[dict, Optional[dict]]:
    """Load either checkpoint format.

    Args:
        path: .pth file.
        map_location: passed to torch.load.
        weights_only: passed to torch.load when not None. A provenance
            checkpoint stores a config dict, so a strict weights_only=True load
            can reject it; on failure this retries with weights_only=False,
            which is safe for files this project wrote itself.

    Returns:
        (state_dict, metadata) where metadata is None for a bare checkpoint.
    """
    kw = {"map_location": map_location}
    if weights_only is not None:
        kw["weights_only"] = weights_only
    try:
        obj = torch.load(path, **kw)
    except Exception:
        if not weights_only:
            raise
        obj = torch.load(path, map_location=map_location, weights_only=False)

    # A bare state_dict is also a dict, so distinguish by the marker key. No
    # parameter is ever literally named "state_dict" -- real keys look like
    # "frontcnn.conv_block.0.weight".
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], {k: v for k, v in obj.items() if k != "state_dict"}
    return obj, None


def describe_checkpoint(path, metadata: Optional[dict]) -> str:
    """One-line provenance summary for logs."""
    name = Path(path).name
    if not metadata:
        return f"{name}: bare state_dict (pre-2026-08-15 format, no provenance)"
    bits = [f"{k}={metadata[k]}" for k in _SUMMARY_KEYS
            if metadata.get(k) is not None]
    return f"{name}: " + ("  ".join(bits) if bits else "provenance dict (empty)")
