"""Scalar-only, forward-verified search along an already computed direction.

The caller owns parameters and AdamW state. ``evaluate(a)`` evaluates the
same frozen residual metric at theta - a * direction without advancing Adam.
No claim that this metric's gradient equals a sampled score surrogate is made.
"""

import math
from dataclasses import dataclass


@dataclass
class SearchResult:
    step: float
    value: float
    evaluations: int
    curvature: float
    slope: float


def residual_line_search(evaluate, value, probe, maximum, *, growth=2.0, minimum=1e-12):
    """Fit Q(a)=Q(0)-A*a+B*a*a/2, then verify candidates by forward.

    The current point is always a candidate; rejection returns step zero.
    Bounds constrain the scalar only. Nonfinite trials are rejected. If a
    local fit fails, geometric backtracking tests at most eight smaller steps.
    A rejection still consumes the caller's one moment update for this gradient.
    """
    if not (math.isfinite(value) and value >= 0 and 0 < probe <= maximum):
        raise ValueError("finite nonnegative metric and 0 < probe <= maximum required")
    first = min(probe, maximum / 2)
    second = 2 * first
    candidates = [(0.0, value)]
    calls = 0

    def measure(step):
        nonlocal calls
        result = float(evaluate(step))
        calls += 1
        if math.isfinite(result):
            candidates.append((step, result))
        return result

    one, two = measure(first), measure(second)
    curvature = (two - 2 * one + value) / (first * first)
    slope = (value - one) / first + 0.5 * curvature * first
    if math.isfinite(curvature) and math.isfinite(slope) and curvature > 0 and slope > 0:
        fitted = min(slope / curvature, maximum, growth * probe)
        if fitted >= minimum and all(abs(fitted - x[0]) > minimum for x in candidates):
            measure(fitted)
    if min(x[1] for x in candidates) >= value:
        smaller = first
        for _ in range(8):
            smaller *= 0.5
            if smaller < minimum or measure(smaller) < value:
                break
    step, best = min(candidates, key=lambda item: item[1])
    return SearchResult(step, best, calls, curvature, slope)


def absorption_feedback(lr, previous, current, target, maximum, *, floor=0.0):
    """First-order log-contraction feedback; bounded, not a stability proof."""
    if current >= previous or current <= floor:
        return max(lr / 2, maximum * 1e-6)
    observed = math.log((previous - floor) / (current - floor))
    factor = min(1.25, max(0.5, target / max(observed, 1e-12)))
    return min(maximum, lr * factor)
