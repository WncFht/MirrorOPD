"""Numeric GSM8K scoring with separate explicit-format and flexible metrics."""

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


def score_response(solution, ground_truth):
    hash_answers = re.findall(r"####\s*(" + NUMBER + r")", solution)
    boxed_answers = re.findall(r"\\boxed\s*\{([^{}]+)\}", solution)
    strict = numeric_answer(hash_answers[-1]) if hash_answers else None
    boxed = numeric_answer(boxed_answers[-1]) if boxed_answers else None
    explicit = strict if strict is not None else boxed
    numbers = re.findall(NUMBER, solution)
    flexible = (
        explicit if explicit is not None else (numeric_answer(numbers[-1]) if numbers else None)
    )
    expected = numeric_answer(str(ground_truth))
    if expected is None:
        raise ValueError(f"Non-numeric GSM8K ground truth: {ground_truth}")
    acc = float(flexible == expected)
    return {
        "score": acc,
        "acc": acc,
        "acc_strict": float(strict == expected),
        "acc_explicit": float(explicit == expected),
        "format_hash": float(strict is not None),
        "format_boxed": float(boxed is not None),
        "parse_failure": float(flexible is None),
    }


def response_diagnostics(token_ids, mask, eos_token_id, budget):
    """Use actual response masks, including EOS; padding is never counted."""
    valid_ids = [token for token, valid in zip(token_ids, mask, strict=True) if valid]
    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    ended = bool(valid_ids and valid_ids[-1] in eos_ids)
    return {
        "response_tokens": len(valid_ids),
        "eos_terminated": float(ended),
        "budget_exhausted": float(len(valid_ids) >= budget and not ended),
    }
