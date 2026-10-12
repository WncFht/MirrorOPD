"""Full-pool coverage, state-preserving forks, and lease release-race checks."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import torch

from training.scalar_full_pool_study import LRS, STAGES, base, forks, pilot_configs
from training.scalar_gpu_lease import GpuGroupLease
from training.scalar_large_worker import Worker


def test_full_pool_plan_and_stage_sources():
    configs = pilot_configs()
    assert len(configs) == 8
    assert [c["lr"] for _, c in configs] == list(LRS)
    assert all(
        c["full_batch"] and c["batch_sequences"] == c["train_sequences"] == 64 for _, c in configs
    )
    assert all(c["keep_checkpoints"] == list(STAGES) for _, c in configs)
    trials = forks({"baselines": [{"id": "selected", "lr": 5e-7}]})
    assert len(trials) == 24
    for stage in STAGES:
        paths = {c["fork_checkpoint"] for _, c in trials if f"parent{stage}-" in _}
        assert len(paths) == 1


def test_release_between_owner_lookup_and_read_is_not_fatal(tmp_path):
    first = GpuGroupLease(tmp_path / "lease")
    assert first.acquire()
    second = GpuGroupLease(first.directory)
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        assert not second.acquire()
    first.release()


def test_fork_preserves_adam_moments_and_update_count(tmp_path):
    config = base()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7, weight_decay=0)
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.ones(2, 2)).square().mean().backward()
        optimizer.step()
    sampler = torch.Generator().manual_seed(1042)
    record = {
        "config": config,
        "parameters": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sampler": sampler.get_state(),
        "update": 120,
    }
    workers = []
    for lr in (1e-6, 2e-6):
        worker = object.__new__(Worker)
        worker.model = copy.deepcopy(model)
        worker.optimizer = torch.optim.AdamW(worker.model.parameters(), lr=lr)
        worker.config = config | {"lr": lr, "fork_checkpoint": "checkpoint-120.pt"}
        worker.pool = SimpleNamespace(sampler=torch.Generator())
        worker.output = tmp_path / str(lr)
        worker.output.mkdir()
        worker.load_fork(copy.deepcopy(record))
        assert worker.optimizer.param_groups[0]["lr"] == lr
        for state in worker.optimizer.state.values():
            assert int(state["step"]) == 3
        worker.optimizer.zero_grad()
        worker.model(torch.ones(2, 2)).square().mean().backward()
        worker.optimizer.step()
        workers.append(worker)
    for a, b in zip(
        workers[0].optimizer.state.values(), workers[1].optimizer.state.values(), strict=True
    ):
        assert int(a["step"]) == int(b["step"]) == 4
        torch.testing.assert_close(a["exp_avg"], b["exp_avg"], rtol=0, atol=0)
        torch.testing.assert_close(a["exp_avg_sq"], b["exp_avg_sq"], rtol=0, atol=0)
    assert any(
        not torch.equal(a, b)
        for a, b in zip(workers[0].model.parameters(), workers[1].model.parameters(), strict=True)
    )


def test_fresh_ablation_entrypoint_honors_long_check_priority(tmp_path, monkeypatch):
    import json
    import sys

    import training.scalar_full_pool_study as study_module
    import training.scalar_large_suite as suite_module

    study = tmp_path / "results/scalar-large/fullpool-lr-study"
    study.mkdir(parents=True)
    (study / "plan.json").write_text("{}")
    (study / "selection.json").write_text(json.dumps({"baselines": [{"lr": 1e-6}]}))
    old_root = tmp_path / "runs/scalar-large"
    old_root.mkdir(parents=True)
    (old_root / "selection.json").write_text(
        json.dumps(
            {
                "baseline_configs": [{"lr": 5e-7}],
                "primary_scalar_config": {
                    "method": "feedback",
                    "search_every": 1,
                    "target_rate": 0.01,
                },
            }
        )
    )
    calls = []
    monkeypatch.setattr(suite_module, "ROOT", tmp_path)
    monkeypatch.setattr(study_module, "long_check", lambda _selection: calls.append("long"))
    monkeypatch.setattr(suite_module, "launch_phase", lambda *_args: calls.append("ablate"))
    monkeypatch.setattr(suite_module, "report", lambda *_args: None)
    monkeypatch.setattr(sys, "argv", ["suite", "--phase", "ablate"])
    suite_module.main()
    assert calls == ["long", "ablate"]
