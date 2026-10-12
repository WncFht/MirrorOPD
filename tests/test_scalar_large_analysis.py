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
