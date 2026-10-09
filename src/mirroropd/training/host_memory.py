"""Record process/host/pinned memory at GPU phase boundaries."""

import ctypes
import gc
import json
import os
import time
from pathlib import Path


def phase_memory(stage, *, trim=False, engine=None):
    import psutil
    import torch

    path = os.environ.get("MIRROR_INNER_METRICS_PATH")
    if not path:
        return
    process = psutil.Process()

    def snapshot():
        host = psutil.virtual_memory()
        pinned = torch.cuda.memory.host_memory_stats()
        return {
            "rss_bytes": process.memory_info().rss,
            "pss_bytes": process.memory_full_info().pss,
            "uss_bytes": process.memory_full_info().uss,
            "system_available_bytes": host.available,
            "gpu_allocated": torch.cuda.memory_allocated(),
            "gpu_reserved": torch.cuda.memory_reserved(),
            "pinned": {k: v for k, v in pinned.items() if k.endswith((".current", ".peak"))},
        }

    before = snapshot()
    if trim and os.environ.get("MIRROR_HOST_TRIM", "1") == "1":
        torch.cuda.synchronize()
        gc.collect()
        # PyTorch 2.8 exposes the caching host allocator through this binding.
        # It frees only unused blocks; live CPU-offloaded weights remain valid.
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)  # noqa: SLF001 -- pinned PyTorch 2.8 binding
        if empty_host_cache is not None:
            empty_host_cache()
        libc = ctypes.CDLL(None)
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    buffers = {}
    if engine is not None:
        seen = set()
        for module in engine.module.modules():
            handle = getattr(module, "_handle", None)
            if handle is None:
                continue
            for name in ("_local_shard", "_cpu_grad", "_mp_shard", "_full_param_padded"):
                value = getattr(handle.flat_param, name, None)
                if value is None or value.untyped_storage().data_ptr() in seen:
                    continue
                seen.add(value.untyped_storage().data_ptr())
                key = f"{name}:{value.device}:{value.dtype}:pinned={value.is_pinned()}"
                buffers[key] = buffers.get(key, 0) + value.untyped_storage().nbytes()
    record = {
        "time": time.time(),
        "stage": stage,
        "pid": os.getpid(),
        "before": before,
        "after": snapshot(),
        "fsdp_buffers": buffers,
    }
    with (Path(path).parent / "host-memory.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
