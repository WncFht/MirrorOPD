from dataclasses import replace

import pytest
import torch

from mirroropd.core.config import CacheContext, Estimator, Objective, Signal
from mirroropd.runtime.frozen_head import load_head, publish_head
from mirroropd.runtime.memory import Memory, check_cache_capacity


def test_head_snapshot_owns_its_values_and_rejects_wrong_version(tmp_path):
    context = CacheContext(
        0,
        "t",
        "v",
        7,
        0.7,
        Signal(),
        Estimator("full", "sampled", 2),
        anchor_storage="hidden",
        anchor_head_path=str(tmp_path / "0.pt"),
    )
    weight, bias, hidden = torch.randn(7, 4), torch.randn(7), torch.randn(3, 4)
    expected = weight.clone()
    publish_head(context, weight, bias, hidden)
    weight.add_(1)
    publish_head(context, weight, bias, hidden)
    actual, _ = load_head(context, "cpu")
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="identity"):
        load_head(replace(context, anchor_version=1), "cpu")


def test_budget_counts_hidden_scalars_and_shared_head():
    spec = Objective(estimator=Estimator("full", "sampled", 16), diagnostics="none")
    memory = Memory(anchor_storage="hidden")
    estimate = check_cache_capacity(
        spec,
        memory,
        prefixes=64 * 7168,
        vocab_size=151665,
        sequence_slots=64 * 9216,
        hidden_size=1536,
    )
    assert estimate["full_bytes"] == 0
    assert estimate["hidden_bytes"] == 64 * 7168 * (2 * 1536 + 4)
    assert estimate["head_bytes"] == 151665 * 1537 * 2
    with pytest.raises(MemoryError):
        check_cache_capacity(
            spec,
            replace(memory, cache_bytes=estimate["total_bytes"] - 1),
            prefixes=64 * 7168,
            vocab_size=151665,
            sequence_slots=64 * 9216,
            hidden_size=1536,
        )


def test_nonzero_rank_never_publishes_head(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 2)
    context = CacheContext(
        0,
        "t",
        "v",
        7,
        1.0,
        Signal(),
        Estimator("full", "sampled", 2),
        anchor_storage="hidden",
        anchor_head_path=str(tmp_path / "0.pt"),
    )
    publish_head(context, torch.randn(7, 4), None, torch.randn(3, 4))
    assert not (tmp_path / "0.pt").exists()


def test_failed_publication_leaves_no_partial_head(tmp_path, monkeypatch):
    context = CacheContext(
        0,
        "t",
        "v",
        7,
        1.0,
        Signal(),
        Estimator("full", "sampled", 2),
        anchor_storage="hidden",
        anchor_head_path=str(tmp_path / "0.pt"),
    )

    def broken_save(_value, path):
        path.write_bytes(b"incomplete")
        raise OSError("test disk failure")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError, match="disk failure"):
        publish_head(context, torch.randn(7, 4), None, torch.randn(3, 4))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "override",
    [
        {"algorithm": "mdpo"},
        {"estimator": {"center": "topj_tail", "integration": "sampled"}},
        {"estimator": {"center": "full", "integration": "full"}},
    ],
)
def test_hidden_storage_rejects_incompatible_job(override):
    from mirroropd.runtime.config import Job

    objective = {
        "algorithm": "bpo",
        "estimator": {"center": "full", "integration": "sampled"},
        **override,
    }
    with pytest.raises(ValueError, match="Hidden anchor"):
        Job.from_dict(
            {
                "name": "reject",
                "models": {"student": "s", "teacher": "t"},
                "data": {"train": ["a"], "validation": ["b"]},
                "objective": objective,
                "memory": {"anchor_storage": "hidden"},
            }
        )
