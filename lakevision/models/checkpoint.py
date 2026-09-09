"""
Checkpoint I/O that reads both formats this project has produced.

Two formats exist and BOTH must keep loading:

  bare        torch.save(model.state_dict(), path)
              Everything up to 2026-09, including the frozen ESSD tags. Those
              runs must stay reproducible, so this is a format to keep reading
              forever, not one to migrate away from.

  provenance  {state_dict, config, epoch, metrics, git_sha, wandb_run_id,
               class_names, num_classes}
              From the eleven-runs work on. An array of runs produces dozens
              of .pth files and a bare state_dict leaves no way to tell which
              config or commit made which file (the last high-capacity run
              became unidentifiable exactly this way).

Every caller should go through :func:`load_checkpoint` rather than
``torch.load`` directly: a provenance checkpoint stores a config dict, so a
strict ``weights_only=True`` load rejects it.
"""
from pathlib import Path
from typing import Optional, Tuple

import torch

# Metadata keys worth echoing when a checkpoint carries them, in print order.
_SUMMARY_KEYS = ("git_sha", "epoch", "seed", "num_classes", "wandb_run_id")


def save_checkpoint(path, model, metadata: Optional[dict] = None):
    """Write a provenance checkpoint: the state_dict plus whatever metadata is given."""
    obj = {"state_dict": model.state_dict()}
    if metadata:
        obj.update({k: v for k, v in metadata.items() if k != "state_dict"})
    torch.save(obj, path)


def load_checkpoint(path, map_location=None, weights_only: Optional[bool] = None
                    ) -> Tuple[dict, Optional[dict]]:
    """Load either checkpoint format.

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
        # weights_only=True refuses the config dict; the file is our own.
        obj = torch.load(path, map_location=map_location, weights_only=False)

    # A bare state_dict is also a dict, so distinguish by the marker key. No
    # parameter is ever literally named "state_dict"; real keys look like
    # "frontcnn.conv_block.0.weight".
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], {k: v for k, v in obj.items() if k != "state_dict"}
    return obj, None


def describe_checkpoint(path, metadata: Optional[dict]) -> str:
    """One-line provenance summary for logs."""
    name = Path(path).name
    if not metadata:
        return f"{name}: bare state_dict (no provenance)"
    bits = [f"{k}={metadata[k]}" for k in _SUMMARY_KEYS
            if metadata.get(k) is not None]
    return f"{name}: " + ("  ".join(bits) if bits else "provenance dict (empty)")
