"""Singleton executor groups remain independent when the suggested TCP port is busy."""

import socket
import sys
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from mirroropd.integrations.verl.rendezvous import configure_singleton_rendezvous


@pytest.fixture
def executor_module(monkeypatch):
    executor = ModuleType("vllm.v1.executor.multiproc_executor")
    executor.get_distributed_init_method = lambda ip, port: f"tcp://{ip}:{port}"

    class MultiprocExecutor:
        def __init__(self, parallel_config):
            self.parallel_config = parallel_config
            self._init_executor()

        def _init_executor(self):
            self.used_init_method = executor.get_distributed_init_method("127.0.0.1", 12345)

    executor.MultiprocExecutor = MultiprocExecutor
    for name in ("vllm", "vllm.v1", "vllm.v1.executor"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys.modules["vllm.v1.executor"], "multiproc_executor", executor, raising=False
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.executor.multiproc_executor", executor)
    return executor


def test_file_groups_ignore_busy_tcp_port_and_use_distinct_stores(
    tmp_path, monkeypatch, executor_module
):
    monkeypatch.delenv("MIRROR_EVENT_DIR", raising=False)
    directory = tmp_path / "with spaces"
    configure_singleton_rendezvous(directory, world_size=1, data_parallel_size=1)
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        methods = [
            executor_module.get_distributed_init_method(*occupied.getsockname()) for _ in range(2)
        ]
        assert len(set(methods)) == 2
        assert all(method.startswith("file://" + str(directory)) for method in methods)
        for method in methods:
            try:
                dist.init_process_group(
                    "gloo", init_method=method, rank=0, world_size=1, timeout=timedelta(seconds=10)
                )
                value = torch.tensor(7.0)
                dist.all_reduce(value)
                assert value.item() == 7
            finally:
                if dist.is_initialized():
                    dist.destroy_process_group()


def test_executor_installs_method_at_its_own_initialization(tmp_path, monkeypatch, executor_module):
    assert executor_module.get_distributed_init_method("localhost", 1).startswith("tcp://")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.delenv("MIRROR_EVENT_DIR", raising=False)
    path = Path(sys.modules[configure_singleton_rendezvous.__module__].__file__).with_name(
        "vllm_executor.py"
    )
    spec = spec_from_file_location("mirror_executor_under_test", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    executor = module.SingletonFileStoreExecutor(
        SimpleNamespace(world_size=1, data_parallel_size=1)
    )
    assert executor.used_init_method.startswith("file://" + str(tmp_path))


@pytest.mark.parametrize(("world_size", "data_parallel_size"), [(2, 1), (1, 2)])
def test_file_rendezvous_rejects_larger_groups(tmp_path, world_size, data_parallel_size):
    with pytest.raises(ValueError, match="singleton"):
        configure_singleton_rendezvous(
            tmp_path, world_size=world_size, data_parallel_size=data_parallel_size
        )
