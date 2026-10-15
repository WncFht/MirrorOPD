"""GPU-only executor selected explicitly by the pinned native vLLM backend."""

import os

from vllm.v1.executor.multiproc_executor import MultiprocExecutor

from mirroropd.integrations.verl.rendezvous import configure_singleton_rendezvous


class SingletonFileStoreExecutor(MultiprocExecutor):
    def _init_executor(self):
        # EngineCore is spawned independently of the frontend actor. Install
        # the method here so it is active where worker bootstrap is prepared.
        configure_singleton_rendezvous(
            os.environ["TMPDIR"],
            world_size=self.parallel_config.world_size,
            data_parallel_size=self.parallel_config.data_parallel_size,
        )
        super()._init_executor()
