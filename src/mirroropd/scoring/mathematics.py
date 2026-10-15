"""Preserve the validated boxed grader and final-2000-character Math-Verify fallback."""

from mirroropd.scoring.vendor import math_verify
from mirroropd.scoring.vendor.ttrl_math import compute_score as boxed_score


def score_math(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    *,
    fallback_characters=2000,
    fallback_timeout_seconds=30.0,
):
    del data_source, extra_info
    if (
        type(fallback_characters) is not int
        or fallback_characters < 1
        or fallback_timeout_seconds <= 0
    ):
        raise ValueError("Math fallback budgets must be positive")
    strict = boxed_score(solution_str, str(ground_truth))
    if strict["format_score"] == 1.0:
        return strict
    correct = float(
        math_verify.compute_score(
            solution_str[-fallback_characters:], str(ground_truth), timeout=fallback_timeout_seconds
        )
    )
    return {
        "score": correct,
        "format_score": 0.0,
        "acc": bool(correct),
        "extracted_gt": ground_truth,
        "pred": "",
    }
