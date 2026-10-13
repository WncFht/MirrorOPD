"""CPU checks for the fixed-prefix protocol and GPU ownership handoff guards."""

import json

import pytest
import torch

from training.oneshot_priority_handoff import remove_retired_lease
from training.power_mean_suite import canary_plan, study_plan
from training.power_mean_worker import panels


def test_power_plan_separates_target_fit_and_sampling_choices():
    plan = study_plan({"bf16": {"matched_strengths": {"0.5": 0.10445, "1.0": 0.08613}}})
    assert len(plan) == 63
    assert len({c["id"] for c in plan}) == len(plan)
    assert all(c["updates"] == 300 and c["batch_size"] == 64 and c["seed"] == 42 for c in plan)
    assert sum(c["updates"] for c in plan) == 18900
    canaries = canary_plan(plan)
    assert len(canaries) == 5
    assert sum(c["updates"] for c in canaries) == 10
    assert {c["kappa"] for c in canaries[:3]} == {0, 0.5, 1}
    assert all(c["batch_size"] == 64 and c["gradient_probe_draws"] == 8 for c in canaries)
    assert all(c["sampler"] == "anchor" for c in plan if c["fit"] == "bpo")
    decoupled = [c for c in plan if c["comparison"] == "decoupled_inner_shape"]
    assert {c["kappa"] for c in decoupled} == {0, 1}
    assert all(c["fit_kappa"] == 0.5 and c["sampler"] == "student" for c in decoupled)


def test_training_and_heldout_responses_are_disjoint_without_cuda():
    sequences = [
        {
            "ids": torch.arange(21) + index * 100,
            "start": 2,
            "response_tokens": 18,
            "selected": torch.arange(18),
            "q_logp_bf16": torch.randn(18, 7).log_softmax(-1),
            "teacher_logp": torch.randn(18, 7).log_softmax(-1),
        }
        for index in range(8)
    ]
    result = panels({"sequences": sequences}, {"kappa": 0.5, "strength": 0.1}, device="cpu")
    assert len(result["train"]) == len(result["heldout"]) == 4
    assert {int(s["ids"][0]) for s in result["train"]}.isdisjoint(
        {int(s["ids"][0]) for s in result["heldout"]}
    )
    assert all(s["qlog"].device.type == "cpu" for values in result.values() for s in values)
    assert sum(len(s["selected"]) for s in result["train"]) == 64


def test_handoff_never_releases_a_live_or_different_controller(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    directory = tmp_path / "runs/gpu-locks/physical-gpu-0-3.lease"
    directory.mkdir(parents=True)
    owner = {"pid": 7, "process_identity": "alive", "nonce": "owned"}
    (directory / "owner.json").write_text(json.dumps(owner))
    monkeypatch.setattr("training.oneshot_priority_handoff.process_identity", lambda _: "alive")
    with pytest.raises(RuntimeError, match="live controller"):
        remove_retired_lease(repo, owner)
    monkeypatch.setattr("training.oneshot_priority_handoff.process_identity", lambda _: None)
    with pytest.raises(RuntimeError, match="different controller"):
        remove_retired_lease(repo, owner | {"nonce": "foreign"})
    assert (directory / "owner.json").exists()
    (directory / "owner.tmp").write_text("old heartbeat")
    remove_retired_lease(repo, owner)
    assert not directory.exists()


def test_batch64_uses_every_response_with_disjoint_positions():
    sequences = [
        {
            "ids": torch.arange(40) + i * 100,
            "start": 2,
            "response_tokens": 38,
            "selected": torch.arange(38),
            "q_logp_bf16": torch.randn(38, 7).log_softmax(-1),
            "teacher_logp": torch.randn(38, 7).log_softmax(-1),
        }
        for i in range(16)
    ]
    result = panels(
        {"sequences": sequences}, {"kappa": 0.5, "strength": 0.1, "batch_size": 64}, device="cpu"
    )
    assert len(result["train"]) == len(result["heldout"]) == 16
    for train, heldout in zip(result["train"], result["heldout"], strict=True):
        assert len(train["selected"]) == len(heldout["selected"]) == 16
        assert set(train["selected"].tolist()).isdisjoint(heldout["selected"].tolist())
    plan = study_plan({"bf16": {"matched_strengths": {"0.5": 0.1, "1.0": 0.1}}})
    primary = [c for c in plan if c["comparison"] == "fixed_strength"]
    assert len(primary) == 50
    assert len({(c["kappa"], c["strength"]) for c in primary if c["strength"] < 1}) == 49
    assert len(primary[-1]["equivalent_kappas"]) == 7
