"""
DataLoader memory planning.

Deliberately dependency-free (no torch, numpy, or sklearn) so it can be imported
and unit-tested without pulling in the training stack.
"""


def sample_size_mb(seq_len, n_channels, height=512, width=512, bytes_per_elem=4):
    """MB of one img_seq sample as handed from a worker to the collate function.

    bytes_per_elem=4 reflects the current float32 path. A uint16 cache that
    defers normalization to the GPU halves this, which is what makes large
    batch sizes fit.
    """
    return seq_len * n_channels * height * width * bytes_per_elem / (1024 ** 2)


def plan_loader_workers(batch_size, sample_mb, host_mem_budget_gb,
                        max_workers=16, prefetch_factor=2):
    """Pick num_workers so the in-flight queue fits a memory budget.

    A DataLoader holds roughly ``num_workers * prefetch_factor * batch_size``
    samples in RAM at once. Because that scales with batch_size, a worker count
    that is safe at bs=8 will OOM the node at bs=32 or 64 — the failure behind
    commit b896d26 ("Fix OOM in dataloader"). Hardcoding num_workers is the bug;
    sizing it against a budget is the fix.

    Args:
        batch_size: samples per batch (per rank, under DDP).
        sample_mb: approximate MB per sample as handed off by the worker.
        host_mem_budget_gb: RAM the queue may occupy. Should be well under
            --mem, leaving room for page cache, the model, and the parent
            process. ~65% of --mem is a reasonable starting point.
        max_workers: never exceed this (keep cores for the main process).
        prefetch_factor: batches prefetched per worker.

    Returns:
        (num_workers, prefetch_factor, projected_queue_gb)
    """
    per_worker_gb = prefetch_factor * batch_size * sample_mb / 1024.0
    if per_worker_gb <= 0:
        return max_workers, prefetch_factor, 0.0

    affordable = int(host_mem_budget_gb // per_worker_gb)
    num_workers = max(1, min(max_workers, affordable))
    projected = num_workers * per_worker_gb

    if affordable < 1:
        print(f"  WARNING: one worker alone needs {per_worker_gb:.0f} GB at "
              f"batch_size={batch_size} but the budget is {host_mem_budget_gb:.0f} GB. "
              f"Falling back to 1 worker; expect swapping or an OOM. "
              f"Lower --batch_size or raise --mem/--host_mem_budget_gb.")

    return num_workers, prefetch_factor, projected
