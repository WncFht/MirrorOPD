"""Cache provenance, padding, frozen tensors and strict schema boundaries."""

from dataclasses import replace

import pytest
import torch

from mirroropd.core.cache import FrozenCache, build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective, Signal
from mirroropd.core.objectives import compute_loss
from mirroropd.core.signal import power_reward


@pytest.mark.parametrize(
    ("center", "integration"), [("topj_tail", "sampled"), ("full", "sampled"), ("full", "full")]
)
@pytest.mark.parametrize("empty", [False, True])
def test_frozen_cache_padding_and_wire_roundtrip(center, integration, empty):
    torch.manual_seed(41)
    objective = Objective(estimator=Estimator(center, integration, 3))
    context = CacheContext(
        2, "teacher", "vocabulary", 7, 1.0, objective.signal, objective.estimator
    )
    q = torch.randn(2, 3, 7, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(q, requires_grad=True)
    valid = (
        torch.zeros(2, 3, dtype=torch.bool)
        if empty
        else torch.tensor([[True, False, True], [False, True, True]])
    )
    with torch.no_grad():
        q[~valid] = teacher[~valid] = float("nan")
    actions = torch.tensor([[2, -99, 5], [-99, 1, 6]])
    cache = build_cache(q, teacher, actions, valid, context)
    saved = {key: value.clone() for key, value in cache.tensors().items()}
    reconstructed = FrozenCache.from_wire(
        cache.to_wire(), CacheContext.from_dict(context.to_dict())
    )
    z = q.detach().clone().requires_grad_()
    result = compute_loss(z, reconstructed, objective, anchor_version=2)
    result.loss.sum().backward()
    assert q.grad is None
    assert teacher.grad is None
    assert torch.isfinite(result.loss).all()
    assert torch.isfinite(z.grad).all()
    assert not z.grad[~valid].any()
    assert all(torch.equal(cache.tensors()[key], value) for key, value in saved.items())
    for invalid in (
        replace(objective, signal=Signal(0.5, 0.3)),
        replace(
            objective,
            estimator=Estimator("full" if center == "topj_tail" else "topj_tail", "sampled", 3),
        ),
    ):
        with pytest.raises(ValueError, match="frozen cache"):
            compute_loss(z, cache, invalid, anchor_version=2)
    with pytest.raises(ValueError, match="Stale"):
        compute_loss(z, cache, objective, anchor_version=3)


@pytest.mark.parametrize(
    "bad",
    [
        {"algorithm": "opd_is"},
        {"signal": {"eta": float("inf")}},
        {"signal": {"eta": 2.0}},
        {"mirror_eta": 0.2},
        {"estimator": {"center": "full_vocab"}},
        {"estimator": {"integration": "full"}},
    ],
)
def test_legacy_modes_and_unknown_fields_are_rejected(bad):
    with pytest.raises((TypeError, ValueError)):
        Objective.from_dict(bad)


def test_schema_and_context_identity_are_explicit():
    objective = Objective()
    with pytest.raises(ValueError, match="schema"):
        CacheContext(
            0,
            "teacher",
            "tokenizer",
            20,
            1.0,
            objective.signal,
            objective.estimator,
            schema_version=1,
        )
    with pytest.raises(ValueError, match="identities"):
        CacheContext(0, "", "tokenizer", 20, 1.0, objective.signal, objective.estimator)
    assert Objective.from_dict(objective.to_dict()) == objective


def test_reward_extremes_and_continuous_endpoints():
    r = torch.tensor([-10000.0, -0.1, 0, 0.1, 10000.0], dtype=torch.float64)
    for kappa in (0.0, 1e-12, 0.5, 1.0):
        for eta in (1e-12, 0.2, 1.0):
            assert torch.isfinite(power_reward(r, Signal(kappa, eta))).all()
        torch.testing.assert_close(power_reward(r, Signal(kappa, 1.0)), r)
    torch.testing.assert_close(power_reward(r, Signal(1e-12, 0.2)), r, atol=1e-8, rtol=1e-8)
