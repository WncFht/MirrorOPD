"""Avoid the pinned vLLM executor's check-then-close TCP rendezvous race."""

import uuid
from pathlib import Path

from mirroropd.runtime.events import environment_writer


def configure_singleton_rendezvous(directory, *, world_size, data_parallel_size):
    """Install FileStore initialization inside the executor's EngineCore process.

    Training DP is separate from this executor group. Each replica needs its
    own store; sharing the training store would join unrelated rank-zero
    groups. This hook must run before worker spawn in vLLM 0.10's executor;
    patching only the frontend actor is lost across EngineCore spawn. Larger
    executor groups are unsupported.
    """
    if (world_size, data_parallel_size) != (1, 1):
        raise ValueError("Mirror vLLM FileStore requires a singleton executor and DP=1")
    from vllm.v1.executor import multiproc_executor

    directory = Path(directory).resolve()
    if any(character in str(directory) for character in "?#"):
        raise ValueError("FileStore directory cannot contain URI query or fragment delimiters")
    directory.mkdir(parents=True, exist_ok=True)

    def init_method(_ip, _port):
        # torch 2.7's file rendezvous consumes the raw POSIX path rather than
        # URL-decoding it, so Path.as_uri() would break directories with spaces.
        method = "file://" + str(directory / ("vllm-" + uuid.uuid4().hex + ".store"))
        writer = environment_writer("vllm-bootstrap")
        if writer:
            writer.emit(
                "runtime.rendezvous",
                component="vllm_executor",
                backend="file",
                world_size=world_size,
                data_parallel_size=data_parallel_size,
                init_method=method,
            )
        return method

    multiproc_executor.get_distributed_init_method = init_method
