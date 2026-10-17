"""Three production algorithms with explicit score and conditional-loss semantics."""

from dataclasses import dataclass

import torch

from mirroropd.core.cache import FULL_FIELDS
from mirroropd.core.diagnostics import diagnostic_names, full_diagnostics, student_statistics
from mirroropd.core.distributions import Distribution, probability_dtype
from mirroropd.core.signal import power_reward


@dataclass(slots=True)
class LossResult:
    loss: torch.Tensor
    diagnostics: dict[str, torch.Tensor]
    kind: str


def score_coefficient(algorithm, u, reward, reward_center, student_kl, eta):
    """Only the sampled log probability receives gradients in the surrogate."""
    if algorithm == "on_policy":
        return -reward - reward_center
    if algorithm == "bpo":
        return (u + student_kl) / eta - reward - reward_center
    if algorithm == "mdpo":
        return u.exp() * (u / eta - reward)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def integrated_loss(logp, anchor_logp, target_logp, algorithm, eta):
    """Normalized target objectives; on-policy enumerates a frozen score surrogate."""
    if algorithm == "mdpo":
        return (logp.exp() * (logp - target_logp)).sum(-1) / eta
    q = anchor_logp.exp()
    if algorithm == "bpo":
        residual = logp - target_logp
        residual = residual - (q * residual).sum(-1, keepdim=True)
        return (q * residual.square()).sum(-1) / (2 * eta)
    if algorithm == "on_policy":
        delta = target_logp - anchor_logp
        coefficient = -(delta - (q * delta).sum(-1, keepdim=True)) / eta
        return (q * coefficient.detach() * logp).sum(-1)
    raise ValueError(f"Unknown algorithm: {algorithm}")


def compute_loss(student_logits, cache, objective, *, anchor_version, vocab_chunk_size=4096):
    """One loss per prefix. Prefix sampling remains the frozen rollout measure.

    On-policy is one full-batch update per fresh window; the runtime enforces
    this scheduling contract. Every algorithm uses the same frozen reward.
    Padding never enters probability arithmetic, including nonfinite padding.
    """
    cache.context.validate(objective, anchor_version=anchor_version)
    if student_logits.shape != (*cache.valid.shape, cache.context.vocab_size):
        raise ValueError("Student logits do not match the cache prefix/vocabulary shape")
    if student_logits.device != cache.valid.device:
        raise ValueError("Student and cache must share a device")
    dtype = probability_dtype(student_logits)
    flat_valid = cache.valid.reshape(-1)
    all_valid = bool(flat_valid.all())
    flat_logits = student_logits.reshape(-1, cache.context.vocab_size)
    logits = flat_logits if all_valid else flat_logits[flat_valid]
    kind = objective.loss_kind
    if len(logits) == 0:
        zero = student_logits[..., 0].to(dtype).nan_to_num().mul(0)
        return LossResult(
            zero, {name: zero.detach().clone() for name in diagnostic_names(objective)}, kind
        )
    values = {}
    for name, tensor in cache.tensors().items():
        if name in FULL_FIELDS:
            values[name] = tensor
            continue
        trailing = tensor.shape[cache.valid.ndim :]
        value = tensor.reshape(flat_valid.numel(), *trailing)
        values[name] = value if all_valid else value[flat_valid]
    raw_bpo = objective.algorithm == "bpo" and objective.estimator.integration == "full"
    if raw_bpo and objective.diagnostics == "none":
        from mirroropd.core.kernels import full_bpo

        loss = full_bpo(
            logits,
            values["anchor_full"],
            values["target_full"],
            objective.signal.eta,
            token_chunk=objective.estimator.chunk_size,
            vocab_chunk=vocab_chunk_size,
        )
        if not torch.isfinite(loss).all():
            raise FloatingPointError("Nonfinite conditional BPO objective")
        output = (
            loss.reshape(cache.valid.shape)
            if all_valid
            else loss.new_zeros(flat_valid.shape)
            .index_copy(0, flat_valid.nonzero().squeeze(-1), loss)
            .reshape(cache.valid.shape)
        )
        return LossResult(output, {}, kind)
    distribution = Distribution.from_logits(
        logits,
        materialize=objective.estimator.integration == "full" and not raw_bpo,
    )
    logp = (
        distribution.full() if objective.estimator.integration == "full" and not raw_bpo else None
    )
    center = head = tail = None
    if objective.diagnostics != "none" or (
        objective.algorithm == "bpo" and objective.estimator.integration == "sampled"
    ):
        center, head, tail = student_statistics(
            distribution,
            values,
            cache.context,
            logp=logp,
            include_support=objective.diagnostics == "full",
        )
    coefficient = reward = u = None
    if raw_bpo:
        from mirroropd.core.kernels import full_bpo

        loss = full_bpo(
            logits,
            values["anchor_full"],
            values["target_full"],
            objective.signal.eta,
            token_chunk=objective.estimator.chunk_size,
            vocab_chunk=vocab_chunk_size,
        )
    elif logp is not None:
        loss = integrated_loss(
            logp,
            values["anchor_full"],
            values["target_full"],
            objective.algorithm,
            objective.signal.eta,
        )
    else:
        action_lp = distribution.action(values["actions"])
        with torch.no_grad():
            u = action_lp.detach() - values["anchor_action"]
            reward = power_reward(
                values["teacher_action"] - values["anchor_action"], objective.signal
            )
            coefficient = score_coefficient(
                objective.algorithm,
                u,
                reward,
                values.get("reward_center"),
                center,
                objective.signal.eta,
            )
        loss = coefficient * action_lp
    with torch.no_grad():
        finite = torch.isfinite(loss).all()
        if coefficient is not None:
            finite = finite & torch.isfinite(coefficient).all()
    if not finite:
        raise FloatingPointError("Nonfinite distillation loss/score; ratios are never clipped")
    diagnostics = {}
    if objective.diagnostics != "none":
        with torch.no_grad():
            action_lp = (
                distribution.action(values["actions"])
                if logp is None
                else logp.gather(-1, values["actions"].unsqueeze(-1)).squeeze(-1)
            )
            diagnostics = {
                "action_nll": -action_lp.detach(),
                "frozen_reward_center": values["reward_center"],
                "student_anchor_kl": center,
            }
            if objective.algorithm == "on_policy":
                diagnostics["on_policy_baseline"] = values["reward_center"]
            if coefficient is not None:
                diagnostics.update(reward=reward, score_coefficient=coefficient, log_anchor_ratio=u)
                if objective.algorithm == "mdpo":
                    diagnostics["importance_ratio"] = u.exp()
            if kind == "conditional_objective":
                diagnostics["conditional_objective"] = loss.detach()
            if objective.diagnostics == "full":
                diagnostics.update(
                    full_diagnostics(distribution, values, cache.context, head, tail)
                )

    def restore(value):
        if all_valid:
            return value.reshape(cache.valid.shape)
        output = value.new_zeros(flat_valid.shape)
        return output.index_copy(0, flat_valid.nonzero().squeeze(-1), value).reshape(
            cache.valid.shape
        )

    return LossResult(
        restore(loss), {name: restore(value) for name, value in diagnostics.items()}, kind
    )
