"""Independent normalized-target derivatives, not production score identities."""

from dataclasses import replace

import pytest
import torch

from mirroropd.core.cache import build_cache
from mirroropd.core.config import CacheContext, Estimator, Objective, Signal
from mirroropd.core.distributions import Distribution
from mirroropd.core.objectives import compute_loss


def target(q, teacher, signal):
    if signal.kappa == 0:
        raw = (1 - signal.eta) * q.log() + signal.eta * teacher.log()
    else:
        raw = (
            (1 - signal.eta) * q.pow(signal.kappa) + signal.eta * teacher.pow(signal.kappa)
        ).log() / signal.kappa
    return raw - raw.logsumexp(-1, keepdim=True)


def objective_loss(z, q, teacher, objective):
    logp = z.log_softmax(-1)
    logm = target(q, teacher, objective.signal)
    if objective.algorithm in {"mdpo", "on_policy"}:
        return (logp.exp() * (logp - logm)).sum(-1) / objective.signal.eta
    residual = logp - logm
    residual = residual - (q * residual).sum(-1, keepdim=True)
    return (q * residual.square()).sum(-1) / (2 * objective.signal.eta)


def context(objective, vocab=3):
    return CacheContext(
        3, "teacher-sha", "tokenizer-sha", vocab, 1.0, objective.signal, objective.estimator
    )


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
@pytest.mark.parametrize("kappa", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("eta", [0.03, 0.2, 1.0])
@pytest.mark.parametrize("integration", ["sampled", "full"])
def test_independent_objective_gradient(algorithm, kappa, eta, integration):
    objective = Objective(algorithm, Signal(kappa, eta), Estimator("full", integration, 1))
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    z = (
        q.log() if algorithm == "on_policy" else torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64)
    ).requires_grad_()
    expected = objective_loss(z, q, teacher, objective)
    expected_gradient = torch.autograd.grad(expected, z)[0]
    # Every possible action shares the same fixed prefix.
    cache = build_cache(
        q.log().expand(3, -1),
        teacher.log().expand(3, -1),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        context(objective),
    )
    result = compute_loss(z.expand(3, -1), cache, objective, anchor_version=3)
    actual = (q * result.loss).sum()
    torch.testing.assert_close(
        torch.autograd.grad(actual, z)[0], expected_gradient, atol=2e-13, rtol=3e-12
    )
    if result.kind == "conditional_objective":
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=3e-12)
    else:
        assert result.kind == "score_surrogate"
    assert all(not value.requires_grad for value in result.diagnostics.values())


@pytest.mark.parametrize("topj", [0, 1, 2, 3])
def test_fresh_gradients_agree_and_partial_center_bias_is_explicit(topj):
    q = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    teacher = torch.tensor([0.6, 0.35, 0.05], dtype=torch.float64)
    base = Objective(estimator=Estimator(topj=topj))
    cache = build_cache(
        q.log().expand(3, -1),
        teacher.log().expand(3, -1),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        context(base),
    )
    for method in ("on_policy", "mdpo", "bpo"):
        objective = replace(base, algorithm=method)
        z = q.log().requires_grad_()
        result = compute_loss(z.expand(3, -1), cache, objective, anchor_version=3)
        actual = torch.autograd.grad((q * result.loss).sum(), z)[0]
        expected = torch.autograd.grad(objective_loss(z, q, teacher, objective), z)[0]
        torch.testing.assert_close(actual, expected, atol=2e-13, rtol=3e-12)
    exact_spec = replace(base, estimator=Estimator("full", "sampled", topj))
    exact_cache = build_cache(
        q.log().expand(3, -1),
        teacher.log().expand(3, -1),
        torch.arange(3),
        torch.ones(3, dtype=torch.bool),
        context(exact_spec),
    )
    z = torch.tensor([-0.7, 0.2, 0.8], dtype=torch.float64, requires_grad=True)
    grouped = compute_loss(z.expand(3, -1), cache, base, anchor_version=3)
    exact = compute_loss(z.expand(3, -1), exact_cache, exact_spec, anchor_version=3)
    delta = (
        grouped.diagnostics["student_anchor_kl"][0] - exact.diagnostics["student_anchor_kl"][0]
    ) / base.signal.eta
    delta -= cache.reward_center[0] - exact_cache.reward_center[0]
    actual = torch.autograd.grad((q * (grouped.loss - exact.loss)).sum(), z)[0]
    torch.testing.assert_close(actual, delta * (q - z.detach().softmax(-1)), atol=2e-13, rtol=3e-12)


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo", "bpo"])
@pytest.mark.parametrize(
    ("center", "integration"), [("topj_tail", "sampled"), ("full", "sampled"), ("full", "full")]
)
def test_diagnostics_do_not_change_gradients(algorithm, center, integration):
    torch.manual_seed(19)
    q, teacher = torch.randn(4, 7), torch.randn(4, 7)
    objective = Objective(algorithm, estimator=Estimator(center, integration, 3))
    cache = build_cache(
        q,
        teacher,
        torch.tensor([1, 2, 3, 4]),
        torch.ones(4, dtype=torch.bool),
        context(objective, 7),
    )
    gradients = []
    for level in ("none", "basic", "full"):
        z = q.clone().requires_grad_()
        result = compute_loss(z, cache, replace(objective, diagnostics=level), anchor_version=3)
        gradients.append(torch.autograd.grad(result.loss.mean(), z)[0])
        if level == "none":
            assert result.diagnostics == {}
        elif level == "basic":
            assert "student_entropy" not in result.diagnostics
            assert "student_anchor_kl" in result.diagnostics
        else:
            assert "student_entropy" in result.diagnostics
        if algorithm != "mdpo" or integration == "full":
            assert "importance_ratio" not in result.diagnostics
    for gradient in gradients[1:]:
        torch.testing.assert_close(gradient, gradients[0])


@pytest.mark.parametrize("algorithm", ["on_policy", "mdpo"])
def test_compact_minimal_mode_skips_full_probabilities_and_tail_diagnostics(monkeypatch, algorithm):
    objective = Objective(algorithm, diagnostics="none", estimator=Estimator(topj=1))
    q = torch.tensor([[0.2, 0.3, 0.5]]).log()
    cache = build_cache(
        q,
        q.flip(-1),
        torch.tensor([1]),
        torch.tensor([True]),
        context(objective),
        diagnostics="none",
    )

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Unneeded full-vocabulary probability/diagnostic computation")

    monkeypatch.setattr(Distribution, "full", unexpected)
    monkeypatch.setattr(Distribution, "tail", unexpected)
    monkeypatch.setattr(Distribution, "entropy", unexpected)
    z = q.clone().requires_grad_()
    compute_loss(z, cache, objective, anchor_version=3).loss.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_full_integration_avoids_unused_overflowing_importance_ratios():
    objective = Objective("mdpo", estimator=Estimator("full", "full", 1))
    q = torch.tensor([[-1000.0, 0.0]])
    cache = build_cache(
        q, torch.zeros_like(q), torch.tensor([0]), torch.tensor([True]), context(objective, 2)
    )
    z = torch.zeros_like(q, requires_grad=True)
    result = compute_loss(z, cache, objective, anchor_version=3)
    result.loss.sum().backward()
    assert torch.isfinite(z.grad).all()
    assert "importance_ratio" not in result.diagnostics
