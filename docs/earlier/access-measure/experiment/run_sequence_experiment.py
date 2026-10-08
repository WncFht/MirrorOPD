#!/usr/bin/env python3
"""Sequence-level OPD experiment with explicit prefix occupancy.

Unlike the independent-row toy, this experiment samples complete trajectories
from a frozen anchor on a prefix tree.  Inner updates only see visited prefixes,
so the report separates token fitting error from the occupancy error induced by
off-policy replay.  The BPO branch implements the paper's practical
complement-probability weight (without inventing an official code repository),
and MDPO is evaluated both in the official sampled-KL form and with an oracle
action-IS correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from mirroropd_algorithms import kl_rows, log_softmax, mirror_target, softmax


def build_tree(vocab: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Return parent and depth for all prefixes, root included."""
    parents = [-1]
    depths = [0]
    frontier = [0]
    for depth in range(horizon):
        nxt = []
        for state in frontier:
            for _ in range(vocab):
                parents.append(state)
                depths.append(depth + 1)
                nxt.append(len(parents) - 1)
        frontier = nxt
    return np.asarray(parents, dtype=np.int64), np.asarray(depths, dtype=np.int64)


def children_of(state: int, vocab: int) -> tuple[int, int]:
    first = 1 + state * vocab
    return first, first + vocab


def sample_trajectories(
    probs: np.ndarray, vocab: int, horizon: int, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((n, horizon), dtype=np.int64)
    actions = np.zeros((n, horizon), dtype=np.int64)
    for i in range(n):
        state = 0
        for t in range(horizon):
            states[i, t] = state
            cdf = np.cumsum(probs[state])
            action = int(np.searchsorted(cdf, rng.random(), side="right"))
            action = min(action, vocab - 1)
            actions[i, t] = action
            state = children_of(state, vocab)[0] + action
    return states, actions


def grouped_update(
    method: str,
    logits: np.ndarray,
    anchor: np.ndarray,
    teacher: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    eta: float,
    lr: float,
    vocab: int,
    bpo_eps: float = 0.05,
    bpo_cap: float = 3.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """One inner update, grouped by visited prefix."""
    p = softmax(logits)
    logp, logb, logt = log_softmax(logits), np.log(anchor), np.log(teacher)
    grad = np.zeros_like(logits)
    coeff_values = []
    touched = np.unique(states)
    for state in touched:
        mask = states == state
        aa = actions[mask]
        pp, bb, tt = p[state], anchor[state], teacher[state]
        score = np.eye(vocab)[aa] - pp[None, :]
        r = logt[state, aa] - logb[state, aa]
        u = logp[state, aa] - logb[state, aa]
        if method == "pmd_part":
            z = np.sum(bb * np.exp(eta * (logt[state] - logb[state])))
            coeff = eta * r - np.log(z) - u
        elif method == "pmd_mean":
            coeff = eta * (r - np.mean(r)) - u
        elif method == "bpo":
            # Exact OPD analogue of the BPO centered advantage, followed by
            # the paper's binary-KL complement ratio and mask.
            adv = eta * (r + kl_rows(bb[None, :], tt[None, :])[0])
            omega = (1.0 + bpo_eps - bb[aa]) / np.maximum(1.0 + bpo_eps - pp[aa], 1e-8)
            keep = ~(((adv > 0) & (omega > 1.0 + 0.2)) | ((adv < 0) & (omega < 1.0 - 0.2)))
            # The paper writes a minimization loss ``-2 eta A w log pi``;
            # converting its gradient to a logit *ascent* direction removes
            # the leading minus sign.
            coeff = 2.0 * adv * np.minimum(omega, bpo_cap) * keep
        elif method in ("mdpo", "mdpo_is"):
            rho = np.exp(u)
            coeff = r - u / eta
            if method == "mdpo_is":
                coeff = rho * coeff
        else:
            raise ValueError(method)
        grad[state] += np.mean(coeff[:, None] * score, axis=0)
        coeff_values.extend(np.asarray(coeff).ravel().tolist())
    logits = logits + lr * grad
    return logits, {
        "update_norm": float(np.linalg.norm(lr * grad)),
        "coefficient_var": float(np.var(coeff_values)) if coeff_values else 0.0,
        "visited_prefixes": float(len(touched)),
    }


def occupancy(probs: np.ndarray, vocab: int, horizon: int) -> np.ndarray:  # noqa: ARG001 — retained public tree-depth argument
    d = np.zeros(probs.shape[0])
    d[0] = 1.0
    for state in range(probs.shape[0]):
        if d[state] == 0 or state == probs.shape[0] - 1:
            continue
        lo, hi = children_of(state, vocab)
        if hi > len(d):
            continue
        d[lo:hi] += d[state] * probs[state]
    return d


def run(
    method: str,
    seed: int,
    vocab: int,
    horizon: int,
    trajectories: int,
    outer_steps: int,
    inner_steps: int,
    eta: float,
    lr: float,
) -> dict:
    rng = np.random.default_rng(seed)
    parents, _depths = build_tree(vocab, horizon)
    n_states = len(parents)
    teacher = softmax(rng.normal(0.0, 1.25, size=(n_states, vocab)))
    logits = log_softmax(teacher) + rng.normal(0.0, 1.5, size=(n_states, vocab))
    history = []
    for outer in range(outer_steps):
        anchor = softmax(logits)
        target = mirror_target(anchor, teacher, eta)
        states, actions = sample_trajectories(anchor, vocab, horizon, trajectories, rng)
        for _ in range(inner_steps):
            logits, diag = grouped_update(
                method, logits, anchor, teacher, states, actions, eta, lr, vocab
            )
        student = softmax(logits)
        d_anchor = occupancy(anchor, vocab, horizon)
        d_student = occupancy(student, vocab, horizon)
        history.append(
            {
                "outer": outer,
                "token_kl_student_teacher_anchor_occ": float(
                    np.sum(d_anchor * kl_rows(student, teacher)) / np.sum(d_anchor)
                ),
                "token_kl_student_target_anchor_occ": float(
                    np.sum(d_anchor * kl_rows(student, target)) / np.sum(d_anchor)
                ),
                "token_kl_student_teacher_student_occ": float(
                    np.sum(d_student * kl_rows(student, teacher)) / np.sum(d_student)
                ),
                "occupancy_l1": float(np.sum(np.abs(d_student - d_anchor))),
                "inner_update_norm": diag["update_norm"],
                "inner_coefficient_var": diag["coefficient_var"],
                "visited_prefixes": diag["visited_prefixes"],
            }
        )
    return {
        "method": method,
        "seed": seed,
        "vocab": vocab,
        "horizon": horizon,
        "trajectories": trajectories,
        "outer_steps": outer_steps,
        "inner_steps": inner_steps,
        "eta": eta,
        "lr": lr,
        "history": history,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="sequence_results.json")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--vocab", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--trajectories", type=int, default=512)
    ap.add_argument("--outer-steps", type=int, default=10)
    ap.add_argument("--inner-steps", type=int, default=32)
    ap.add_argument("--eta", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=0.18)
    args = ap.parse_args()
    methods = ("pmd_part", "pmd_mean", "bpo", "mdpo", "mdpo_is")
    records = [
        run(
            m,
            args.seed,
            args.vocab,
            args.horizon,
            args.trajectories,
            args.outer_steps,
            args.inner_steps,
            args.eta,
            args.lr,
        )
        for m in methods
    ]
    Path(args.out).write_text(
        json.dumps({"config": vars(args), "methods": records}, indent=2),
        encoding="utf-8",
    )
    for rec in records:
        h = rec["history"][-1]
        print(
            f"{rec['method']:9s} anchor-occ KL={h['token_kl_student_teacher_anchor_occ']:.6f} "
            f"target KL={h['token_kl_student_target_anchor_occ']:.6f} "
            f"occupancy L1={h['occupancy_l1']:.6f}"
        )


if __name__ == "__main__":
    main()
