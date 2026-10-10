"""Exercise the pinned upstream math verifier without starting CUDA inference."""

from training.math_reward import compute_score


def main():
    cases = [
        (r"The answer is \boxed{37}", "37", 1, 1),
        (r"The answer is \boxed{38}", "37", 0, 1),
        (r"The answer is \boxed{\frac{1}{2}}", "0.5", 1, 1),
        ("The final answer is 37.", "37", 1, 0),
        ("I cannot solve this problem.", "37", 0, 0),
    ]
    for text, truth, score, format_score in cases:
        result = compute_score("DAPO", text, truth)
        assert result["score"] == score, (text, result)
        assert result["format_score"] == format_score, (text, result)
    print("All five official-verifier protocol cases passed")


if __name__ == "__main__":
    main()
