"""Core categorical implementations of PMD-Part, PMD-Mean, BPO and MDPO.

All methods use one frozen anchor per outer step.  The implementation is deliberately
small: it exposes the exact sampled scalar/vector estimators used in the derivations,
so the toy experiment can be audited against the equations in the TeX note.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits, axis=-1, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=-1, keepdims=True))


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def sample_actions(probs: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Return [states, n] actions from every categorical row."""
    cdf = np.cumsum(probs, axis=-1)
    u = rng.random((probs.shape[0], n, 1))
    return np.sum(u > cdf[:, None, :], axis=-1).clip(0, probs.shape[1] - 1)


def score_vectors(probs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Categorical score e_a - p, shape [states, samples, actions]."""
    eye = np.eye(probs.shape[-1])
    return eye[actions] - probs[:, None, :]


def mirror_target(anchor: np.ndarray, teacher: np.ndarray, eta: float) -> np.ndarray:
    """Exact token-level Gibbs target p+ proportional to anchor^(1-eta) teacher^eta."""
    z = (1.0 - eta) * np.log(anchor) + eta * np.log(teacher)
    return softmax(z)


def inner_update(
    name: str,
    logits: np.ndarray,
    anchor: np.ndarray,
    teacher: np.ndarray,
    actions: np.ndarray,
    eta: float,
    lr: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """One stochastic inner update for a frozen anchor and teacher.

    The returned diagnostics expose the norm of the sampled update and the estimator
    coefficient variance.  ``actions`` are sampled once from the frozen anchor and
    can be reused for several inner updates, matching replay-style inner training.
    """
    p = softmax(logits)
    logp, logb, logt = log_softmax(logits), np.log(anchor), np.log(teacher)
    s, k = actions.shape
    scores = score_vectors(p, actions)
    gathered_p = np.take_along_axis(p, actions, axis=1)
    gathered_b = np.take_along_axis(anchor, actions, axis=1)
    gathered_r = np.take_along_axis(logt - logb, actions, axis=1)
    ratio_log = np.log(gathered_p) - np.log(gathered_b)
    u = ratio_log

    if name == "pmd_part":
        z = np.sum(anchor * np.exp(eta * (logt - logb)), axis=1)
        h = eta * gathered_r - np.log(z)[:, None]
        coeff = h - u
        grad = np.mean(coeff[..., None] * scores, axis=1)
    elif name == "pmd_mean":
        center = np.mean(logt - logb, axis=1)
        coeff = eta * (gathered_r - center[:, None]) - u
        grad = np.mean(coeff[..., None] * scores, axis=1)
    elif name == "bpo":
        # BPO's practical, binary-KL form from arXiv:2609.15987.  In the
        # OPD instantiation the exact centered teacher signal is
        # A=log(T/B)+KL(B||T), and the complement ratio replaces PPO's IS
        # ratio.  The cap and mask are deliberately exposed as arguments in
        # the sequence experiment; this low-level routine keeps the no-clip
        # algebraic form used by the earlier regression comparison.
        d = eta * gathered_r - u
        d_center = d - np.mean(d, axis=1, keepdims=True)
        score_center = scores - np.mean(scores, axis=1, keepdims=True)
        coeff = d_center
        grad = np.mean(d_center[..., None] * score_center, axis=1)
    elif name in ("mdpo", "mdpo_is"):
        rho = np.exp(ratio_log)
        # ``mdpo`` is the sampled-token implementation in the official
        # MD-OPD repository: samples come from the frozen anchor and the KL
        # penalty is evaluated directly on those samples.  ``mdpo_is`` is the
        # population-correct action-IS estimator, useful as an oracle control.
        coeff = (gathered_r - ratio_log / eta) if name == "mdpo" else rho * (gathered_r - ratio_log / eta)
        grad = np.mean(coeff[..., None] * scores, axis=1)
    else:
        raise ValueError(name)

    new_logits = logits + lr * grad
    diagnostics = {
        "update_norm": float(np.linalg.norm(lr * grad)),
        "coefficient_var": float(np.var(coeff)),
        "kl_to_anchor": float(np.mean(kl_rows(p, anchor))),
    }
    return new_logits, diagnostics
