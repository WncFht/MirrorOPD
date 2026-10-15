"""GSM8K numeric answers with explicit-format and flexible correctness fields."""

import re
from decimal import Decimal, InvalidOperation

NUMBER = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"


def numeric_answer(text):
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    if not re.fullmatch(NUMBER, value):
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def score_numeric(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info
    hash_answers = re.findall(r"####\s*(" + NUMBER + r")", solution_str)
    boxed_answers = re.findall(r"\\boxed\s*\{([^{}]+)\}", solution_str)
    strict = numeric_answer(hash_answers[-1]) if hash_answers else None
    boxed = numeric_answer(boxed_answers[-1]) if boxed_answers else None
    explicit = strict if strict is not None else boxed
    numbers = re.findall(NUMBER, solution_str)
    flexible = (
        explicit if explicit is not None else numeric_answer(numbers[-1]) if numbers else None
    )
    expected = numeric_answer(str(ground_truth))
    if expected is None:
        raise ValueError(f"Non-numeric GSM8K ground truth: {ground_truth}")
    return {
        "score": float(flexible == expected),
        "strict_accuracy": float(strict == expected),
        "explicit_accuracy": float(explicit == expected),
        "parse_failure": float(flexible is None),
    }
