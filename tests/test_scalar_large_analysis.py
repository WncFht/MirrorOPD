"""Calibration ranking, failure retention, and confirmation statistics."""

import json
import math

from experiments.analysis.scalar_large_report import exact_sign_test, read_run, select_calibration
from training.scalar_large_suite import base_config, confirmation_configs, tuning_configs


def fake_run(root, name, method, scale, *, status="complete"):
    directory = root / name
    directory.mkdir()
    config = base_config() | {"method": method, "updates": 20, "lr": scale * 1e-6}
    summary = {
        "config": config,
        "status": status,
        "seconds": 2,
        "counts": {"backwards": 4, "forwards": 12},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    points = [
        {
            "update": update,
            "seconds": update / 10,
            "metrics": {
                "train": {"bpo": value},
                "validation": {"bpo": 0.2, "teacher_reverse_kl": 0.3},
            },
        }
        for update, value in ((0, 1), (10, 0.5**scale), (20, 0.25**scale))
    ]
    (directory / "audits.jsonl").write_text("".join(json.dumps(point) + "\n" for point in points))
    return directory


def test_log_integral_normalizes_to_update_budget(tmp_path):
    directory = fake_run(tmp_path, "linear-log-decay", "constant", 1)
    run = read_run(directory)
    assert abs(run["row"]["log_auc"] + math.log(2)) < 1e-12


def test_failed_candidate_is_retained_and_not_selected(tmp_path):
    fake_run(tmp_path, "slow", "constant", 1)
    fake_run(tmp_path, "fast", "constant", 2)
    fake_run(tmp_path, "bad", "constant", 9, status="nonfinite")
    fake_run(tmp_path, "feedback", "feedback", 1.5)
    output = tmp_path / "selection.json"
    result = select_calibration(tmp_path, output)
    assert result["baseline_configs"][0]["lr"] == 2e-6
    assert len(result["rankings"]) == 4
    assert result["rankings"][-1]["status"] == "nonfinite"
    assert result["rankings"][-1]["log_auc"] is None
    assert "Infinity" not in output.read_text()


def test_sign_test_uses_independent_pairs():
    assert exact_sign_test([1, 2, 3, 4, 5]) == 1 / 32
    assert exact_sign_test([1, 2, 3, -4, -5]) == 0.5
    assert exact_sign_test([0, 0]) == 1


def test_confirmation_does_not_retune_learning_rates():
    tuning = tuning_configs()
    selection = {"baseline_configs": [tuning[0][1], tuning[1][1]], "feedback_target": 0.03}
    configurations = confirmation_configs(selection)
    assert len(configurations) == 25
    constants = [config for _, config in configurations if config["method"] == "constant"]
    assert {config["lr"] for config in constants} == {2.5e-7, 5e-7}
    assert {config["seed"] for _, config in configurations} == {42, 142, 242, 342, 442}
    assert all(config["updates"] == 600 for _, config in configurations)


def test_checkpoint_restores_moments_rng_and_discards_uncommitted_logs(tmp_path):
    import collections
    import time
    from types import SimpleNamespace

    import torch

    from training.scalar_large_worker import Worker

    original = object.__new__(Worker)
    original.model = torch.nn.Linear(2, 1)
    original.optimizer = torch.optim.AdamW(original.model.parameters(), lr=1e-3)
    original.pool = SimpleNamespace(sampler=torch.Generator().manual_seed(42))
    original.output = tmp_path
    original.config = {"test": "resume"}
    original.lr, original.previous, original.initial_proxy = 1e-3, 0.4, 0.5
    original.counts = collections.Counter(forwards=3, backwards=1)
    original.elapsed = 0
    original.begin = time.perf_counter()
    x = torch.tensor([[1.0, 2.0]])
    original.model(x).square().mean().backward()
    original.optimizer.step()
    original.append("updates.jsonl", {"update_after": 1})
    original.append("audits.jsonl", {"update": 1})
    original.save(1)
    original.append("updates.jsonl", {"update_after": 2})
    expected_draw = torch.randint(1000, (10,), generator=original.pool.sampler)
    original.optimizer.zero_grad()
    original.model(x).square().mean().backward()
    original.optimizer.step()

    restored = object.__new__(Worker)
    restored.model = torch.nn.Linear(2, 1)
    restored.optimizer = torch.optim.AdamW(restored.model.parameters(), lr=1e-3)
    restored.pool = SimpleNamespace(sampler=torch.Generator().manual_seed(0))
    restored.output, restored.config = tmp_path, original.config
    restored.restore()
    actual_draw = torch.randint(1000, (10,), generator=restored.pool.sampler)
    assert torch.equal(expected_draw, actual_draw)
    restored.optimizer.zero_grad()
    restored.model(x).square().mean().backward()
    restored.optimizer.step()
    for left, right in zip(original.model.parameters(), restored.model.parameters(), strict=True):
        assert torch.equal(left, right)
    assert [
        json.loads(line)["update_after"]
        for line in (tmp_path / "updates.jsonl").read_text().splitlines()
    ] == [1]
    assert restored.start_update == 1


def test_single_final_crossing_does_not_count_as_persistent(tmp_path):
    run = read_run(fake_run(tmp_path, "late-crossing", "constant", 2))
    assert run["row"]["two_audits_0.1"] is None
    assert run["row"]["persistent_0.1"] is None
    assert run["row"]["persistent_0.5"] == 10


def test_duplicate_audit_replay_is_rejected(tmp_path):
    import pytest

    directory = fake_run(tmp_path, "duplicate", "constant", 1)
    with (directory / "audits.jsonl").open("a") as stream:
        stream.write((directory / "audits.jsonl").read_text().splitlines()[-1] + "\n")
    with pytest.raises(ValueError, match="duplicate replay"):
        read_run(directory)


def test_optional_proxy_logging_does_not_change_constant_adam_updates(tmp_path, monkeypatch):
    import collections
    import copy
    import time
    from types import SimpleNamespace

    import torch

    import training.scalar_large_worker as module

    initial = torch.nn.Linear(2, 1)
    x = torch.tensor([[1.0, 2.0]])
    monkeypatch.setattr(module, "field", lambda logits, _record, _config: logits.square().mean())
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    workers = []
    for minimal in (False, True):
        worker = object.__new__(module.Worker)
        worker.model = copy.deepcopy(initial)
        worker.parameters = list(worker.model.parameters())
        worker.optimizer = torch.optim.AdamW(worker.parameters, lr=1e-3, weight_decay=0)
        worker.pool = SimpleNamespace(batch=lambda _count: [({"response_tokens": 1}, 1.0)])
        worker.config = {
            "method": "constant",
            "minimal_monitoring": minimal,
            "search_every": 1,
            "batch_sequences": 1,
        }
        worker.output = tmp_path / str(minimal)
        worker.output.mkdir()
        worker.lr = 1e-3
        worker.previous = worker.initial_proxy = None
        worker.elapsed = 0
        worker.begin = time.perf_counter()
        worker.counts = collections.Counter()
        worker.forward = lambda _record, worker=worker: worker.model(x)

        def proxy(worker=worker):
            worker.counts["proxy_calls"] += 1
            return float(worker.model(x).square().mean().detach())

        worker.proxy_value = proxy
        for index in range(3):
            worker.update(index)
        workers.append(worker)
    assert workers[0].counts["proxy_calls"] == 3
    assert workers[1].counts["proxy_calls"] == 0
    for left, right in zip(workers[0].parameters, workers[1].parameters, strict=True):
        assert torch.equal(left, right)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(
                workers[0].optimizer.state[left][key], workers[1].optimizer.state[right][key]
            )


def test_frozen_input_restore_verifies_bytes_and_preserves_existing_data(tmp_path):
    import gzip
    import hashlib

    import pytest

    from experiments.rollout.scalar_large_cache import restore_frozen_tokens

    folder = tmp_path / "data/scalar-large"
    folder.mkdir(parents=True)
    original = b"fixed rollout IDs; test fixture, not a torch archive"
    archive = folder / "seed-42-tokens.pt.gz"
    archive.write_bytes(gzip.compress(original))
    (tmp_path / "data-manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "archive": "data/scalar-large/seed-42-tokens.pt.gz",
                        "bytes": len(original),
                        "sha256": hashlib.sha256(original).hexdigest(),
                    }
                ]
            }
        )
    )
    destination = tmp_path / "runs/cache/tokens.pt"
    assert restore_frozen_tokens(tmp_path, 42, destination)
    assert destination.read_bytes() == original
    destination.write_bytes(b"existing data is preserved")
    assert not restore_frozen_tokens(tmp_path, 42, destination)
    assert destination.read_bytes() == b"existing data is preserved"
    destination.unlink()
    archive.write_bytes(gzip.compress(b"corrupt"))
    with pytest.raises(ValueError, match="checksum"):
        restore_frozen_tokens(tmp_path, 42, destination)
    assert not destination.exists()


def test_directory_gpu_lease_excludes_competing_owner(tmp_path):
    from training.scalar_gpu_lease import GpuGroupLease

    first = GpuGroupLease(tmp_path / "physical-gpu-0-3.lease")
    second = GpuGroupLease(first.directory)
    assert first.acquire()
    assert not second.acquire()
    second.release()
    assert first.directory.exists()
    first.release()
    assert second.acquire()
    second.release()


def test_directory_lease_reclaims_only_provably_dead_local_identity(tmp_path):
    from training.scalar_gpu_lease import GpuGroupLease

    lease = GpuGroupLease(tmp_path / "lease")
    lease.directory.mkdir()
    foreign = dict(lease.owner, pid_namespace="another namespace", process_identity="dead")
    path = lease.directory / "owner.json"
    path.write_text(json.dumps(foreign))
    assert not lease.acquire()
    assert path.exists()
    foreign["pid_namespace"] = lease.owner["pid_namespace"]
    path.write_text(json.dumps(foreign))
    assert not lease.acquire()
    assert not lease.directory.exists()
    assert lease.acquire()
    lease.release()


def test_resource_conflict_retries_without_changing_experiment_plan(tmp_path, monkeypatch):
    import training.scalar_large_suite as module

    calls = []
    commands = [("run", ["training/scalar_large_worker.py", "--config", "locked.json"])]

    def attempt(name, arguments, output):
        calls.append((name, arguments, output))
        if len(calls) == 1:
            raise module.GPUResourceConflictError("competing process")
        return [{"job": "run", "exit_code": 0}]

    monkeypatch.setattr(module, "launch_phase_once", attempt)
    monkeypatch.setattr(module.time, "sleep", lambda _seconds: None)
    result = module.launch_phase("confirm", commands, tmp_path)
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert result[0]["exit_code"] == 0


def test_feedback_winner_keeps_all_named_monitor_ablations():
    from training.scalar_large_suite import ablation_configs

    base = base_config()
    selection = {"baseline_configs": [base], "primary_scalar_config": base | {"method": "feedback"}}
    configurations = ablation_configs(selection)
    names = {name for name, _ in configurations}
    assert len(configurations) == 17
    assert {
        "ablate-monitor2048-scalar",
        "ablate-monitor-sampled-scalar",
        "ablate-monitor-two-clusters-scalar",
    } <= names
