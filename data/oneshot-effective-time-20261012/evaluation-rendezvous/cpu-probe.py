import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def probe(index):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="oneshot-cpu-filestore-", dir="/tmp") as directory:
        store = Path(directory) / "store"
        assert not store.exists()
        dist.init_process_group("gloo", init_method=store.as_uri(), rank=0, world_size=1)
        value = torch.tensor([index + 1])
        dist.all_reduce(value)
        assert value.item() == index + 1
        dist.destroy_process_group()
    return {
        "replica": index,
        "world_size": 1,
        "reduction": value.item(),
        "temporary_directory_removed": not Path(directory).exists(),
    }


if __name__ == "__main__":
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=4, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        results = list(executor.map(probe, range(4)))
    Path("results/oneshot-effective-time/file-store-cpu-probe.json").write_text(
        json.dumps({"backend": "gloo", "gpu_validation": "pending", "replicas": results}, indent=2)
        + "\n"
    )
    print(json.dumps(results))
