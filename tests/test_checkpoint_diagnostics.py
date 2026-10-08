"""Validate checkpoint loading and exact fixed-prefix scoring without downloaded models."""

import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


def main():
    path = (
        Path(__file__).resolve().parents[1] / "experiments/scoring/pythia_checkpoint_diagnostics.py"
    )
    spec = importlib.util.spec_from_file_location("checkpoint_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p = torch.tensor([[0.6, 0.3, 0.1]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64)
    values = module.distribution_metrics(p.log(), q.log())
    torch.testing.assert_close(values["student_teacher_kl"], (p * (p / q).log()).sum(-1))
    torch.testing.assert_close(values["teacher_student_kl"], (q * (q / p).log()).sum(-1))
    assert values["student_teacher_kl"] != values["teacher_student_kl"]
    assert module.selected_positions(
        [0, 1, 2], [3, 4], all_prompt_prefixes=False, max_prompt_prefixes=64
    ) == [2, 3]
    assert module.selected_positions(
        [0, 1, 2], [3, 4], all_prompt_prefixes=True, max_prompt_prefixes=64
    ) == [0, 1, 2, 3]

    class TinyCausalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(5, 5)

        def forward(self, input_ids, *, use_cache):
            assert use_cache is False
            return SimpleNamespace(logits=self.embedding(input_ids))

    torch.manual_seed(123)
    source = TinyCausalModel()
    target = TinyCausalModel()
    with tempfile.TemporaryDirectory() as directory:
        torch.save(source.state_dict(), Path(directory) / "model_world_size_1_rank_0.pt")
        module.load_actor_checkpoint(target, directory, directory)
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, target.state_dict()[name])
        assert not torch.distributed.is_initialized()
        # Exercise actual FSDP-style ShardedTensor serialization on one CPU rank.
        from torch.distributed._shard.sharded_tensor import Shard, init_from_local_shards

        torch.distributed.init_process_group(
            "gloo", init_method=f"file://{directory}/shard-group", rank=0, world_size=1
        )
        try:
            sharded = {
                name: init_from_local_shards(
                    [Shard.from_tensor_and_offsets(tensor, [0, 0], 0)], *tensor.shape
                )
                for name, tensor in source.state_dict().items()
            }
            torch.save(sharded, Path(directory) / "model_world_size_1_rank_0.pt")
        finally:
            torch.distributed.destroy_process_group()
        module.load_actor_checkpoint(target, directory, directory)
        for name, tensor in source.state_dict().items():
            torch.testing.assert_close(tensor, target.state_dict()[name])
    args = SimpleNamespace(
        max_continuation_tokens=64,
        prompt_prefixes="all",
        max_prompt_prefixes=64,
        prefix_chunk_size=1,
    )
    rows = module.score_records(
        source, target, [{"prompt_ids": [0, 1], "continuation_ids": [2, 3]}], args
    )
    assert len(rows) == 3
    assert [row["prefix_length"] for row in rows] == [1, 2, 3]
    assert all(abs(row["student_teacher_kl"]) < 1e-12 for row in rows)
    summary = module.summarize(rows)
    assert summary["student_continuation_nll"]["count"] == 2
    expected = -source.embedding.weight[[1, 2]].double().log_softmax(-1)[[0, 1], [2, 3]].mean()
    assert abs(summary["student_continuation_nll"]["mean"] - expected.item()) < 1e-12
    print(
        "checkpoint diagnostics: PASS (ordinary + one-rank ShardedTensor, exact KL, NLL alignment)"
    )


if __name__ == "__main__":
    main()
