"""Scientific metric identities, effective-time plans, and GPU conflict guards."""

import json
from unittest.mock import patch

import numpy as np
import torch

from experiments.analysis.oneshot_effective_time import collapse_statistics
from training.gpu_exclusive import (
    GpuGroupLease,
    foreign_gpu_processes,
    gpu_processes,
    register_gpu_owner,
)
from training.oneshot_suite import experiment_plan
from training.oneshot_worker import distribution_sums


def test_full_distribution_targets():
    """An exact mirror target has zero target error and positive teacher distance."""
    generator = torch.Generator().manual_seed(42)
    qlog = torch.randn(9, 16, generator=generator).log_softmax(-1)
    tlog = torch.randn(9, 16, generator=generator).log_softmax(-1)
    target = (0.7 * qlog + 0.3 * tlog).log_softmax(-1)
    actions = torch.arange(9)
    at_anchor = distribution_sums(qlog, qlog, tlog, actions, 0.3)
    at_target = distribution_sums(target, qlog, tlog, actions, 0.3)
    assert abs(at_anchor["anchor_student_full_kl"]) < 1e-6
    assert at_anchor["student_teacher_full_kl"] > 0
    assert abs(at_target["mirror_target_full_kl"]) < 1e-6
    assert at_target["bpo_target_variance"] < 1e-10
    assert at_target["student_teacher_full_kl"] > 0
    expected = (target.exp() * (target - tlog)).sum()
    torch.testing.assert_close(torch.tensor(at_target["student_teacher_full_kl"]), expected)


def test_global_token_gradient_weighting():
    """Unequal micro-batch lengths must reproduce one global token mean."""
    logits = torch.linspace(-1, 1, 16, requires_grad=True)
    teacher = torch.linspace(1, -1, 16).log_softmax(-1)
    actions = (torch.tensor([0, 1]), torch.tensor([2, 3, 4, 5, 6]))
    all_actions = torch.cat(actions)
    logp = logits.log_softmax(-1)
    advantage = (teacher - logp).detach()
    losses = [-advantage[index] * logp[index] for index in actions]
    # A DP average of world/global_count-scaled local sums is the global mean.
    simulated_dp = sum(loss.sum() * (2 / len(all_actions)) for loss in losses) / 2
    reference = (-advantage[all_actions] * logp[all_actions]).mean()
    torch.testing.assert_close(
        torch.autograd.grad(simulated_dp, logits, retain_graph=True)[0],
        torch.autograd.grad(reference, logits)[0],
    )


def test_preregistered_effective_time_matrix():
    """Every learning-rate triple reaches the same effective-time endpoint."""
    plan = experiment_plan()
    assert len(plan) == 45
    assert sum(config["updates"] for config in plan) == 16074
    for seed in (42, 142, 242):
        for rollout, precision in (("frozen", "bf16"), ("frozen", "fp32"), ("online", "bf16")):
            runs = [
                config
                for config in plan
                if config["seed"] == seed
                and config["rollout"] == rollout
                and config["precision"] == precision
                and config["method"] == "opd"
                and config["lr"] > 0
            ]
            assert len(runs) == 3
            for config in runs:
                assert config["lr"] / 1e-6 * config["updates"] == 300
                assert [config["lr"] / 1e-6 * step for step in config["checkpoint_steps"]] == [
                    50,
                    100,
                    200,
                    300,
                ]


def test_driver_pid_guard_and_gpu_restriction(tmp_path, monkeypatch):
    """GPU driver IDs remain authoritative across different PID namespaces."""
    inventory = "\n".join(f"{index}, GPU-{index}, 1" for index in range(8))
    with patch(
        "training.gpu_exclusive.subprocess.check_output",
        side_effect=[inventory, "GPU-0, 987\nGPU-4, 999\n"],
    ):
        devices = gpu_processes()
    assert set(devices) == {0, 1, 2, 3}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    with patch("training.gpu_exclusive.gpu_processes", return_value=devices):
        register_gpu_owner(tmp_path, 0)
        assert not foreign_gpu_processes(tmp_path)
    owner = json.loads((tmp_path / "rank-0.json").read_text())
    assert owner["driver_pid"] == 987
    devices[0]["pids"].append(1234)
    with patch("training.gpu_exclusive.gpu_processes", return_value=devices):
        assert foreign_gpu_processes(tmp_path) == {0: {1234}}


def test_directory_lease_is_exclusive_and_releasable(tmp_path):
    """Two controllers cannot hold the same atomic-directory reservation."""
    first = GpuGroupLease(tmp_path / "gpu-group.lease")
    second = GpuGroupLease(tmp_path / "gpu-group.lease")
    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    first.release()
    assert (tmp_path / "gpu-group.lease/owner.json").exists()
    second.release()
    assert not (tmp_path / "gpu-group.lease").exists()


def test_collapse_statistic_detects_known_time_rescaling():
    """An exact effective-time curve must have zero effective dispersion."""
    arms = []
    for factor, count in ((0.5, 600), (1.0, 300), (2.0, 150)):
        steps = np.arange(count + 1, dtype=float)
        arms.append(
            {
                "steps": steps,
                "effective_time": factor * steps,
                "distance": np.exp(-0.01 * factor * steps),
            }
        )
    result = collapse_statistics(arms)
    assert result["raw_log_dispersion"] > 0.1
    assert result["effective_log_dispersion"] < 1e-12
    assert result["preregistered_30_percent_reduction_passed"]


def test_directory_lease_owner_disappearance_is_retryable(tmp_path):
    """A lease released during owner lookup must keep the waiter alive."""
    first = GpuGroupLease(tmp_path / "gpu-group.lease")
    second = GpuGroupLease(first.directory)
    assert first.acquire()
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()
