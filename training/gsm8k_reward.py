"""verl reward callback; GSM8K task scores monitor MDPO, not its gradient."""

from mirroropd.training.gsm8k import score_response


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    return score_response(solution_str, ground_truth)
