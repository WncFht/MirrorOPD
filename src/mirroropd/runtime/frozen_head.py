"""Run-scoped immutable anchor heads shared by actor and reference workers.

Only rank zero publishes, after FSDP has materialized the linear inside its
forward. The driver waits for the complete anchor pass before teacher dispatch.
Readers retain one host head, never one copy per sample. No automatic resume.
"""

import os
from functools import lru_cache
from pathlib import Path

import torch


@torch.no_grad()
def publish_head(context, weight, bias, hidden):
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank != 0:
        return
    path = Path(context.anchor_head_path)
    if path.exists():
        return  # Later microbatches of the same frozen anchor pass.
    dtype = (
        torch.get_autocast_dtype(hidden.device.type)
        if torch.is_autocast_enabled(hidden.device.type)
        else hidden.dtype
    )
    payload = {
        "context": context.to_dict(),
        "weight": weight.detach().to(device="cpu", dtype=dtype, copy=True),
        "bias": bias.detach().to(device="cpu", dtype=dtype, copy=True)
        if bias is not None
        else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def _read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def load_head(context, device):
    payload = _read(context.anchor_head_path)
    if payload["context"] != context.to_dict():
        raise ValueError("Frozen head identity differs from the anchor context")
    weight, bias = payload["weight"], payload["bias"]
    if weight.ndim != 2 or len(weight) != context.vocab_size or weight.requires_grad:
        raise ValueError("Frozen head does not cover the declared legal vocabulary")
    return weight.to(device), bias.to(device) if bias is not None else None
