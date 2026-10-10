"""One-Shot-OPD-compatible math reward for DAPO and MATH-500.

The primary score is the upstream ``ttrl_math`` boxed-answer check.  Outputs
without a boxed answer receive a Math-Verify fallback score on their final
2000 characters, matching ``ttrl_mathverify.py`` from the pinned upstream
checkout.  This keeps training reward and standalone evaluation identical.
"""

from verl.utils.reward_score.ttrl_mathverify import reward_func

from mirroropd.training.gsm8k import score_response


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source == "openai/gsm8k":
        return score_response(solution_str, ground_truth)
    return reward_func(data_source, solution_str, ground_truth, extra_info, **kwargs)
