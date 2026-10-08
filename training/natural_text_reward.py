"""Task reward is disabled in MDPO; this callback keeps upstream schema valid."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, solution_str, ground_truth, extra_info, kwargs
    return 0.0
