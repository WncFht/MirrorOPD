"""Verify scoring, censoring and template safety rather than model accuracy."""

# Dependency-free unittest runner is also exercised by make test.
# ruff: noqa: PT009, PT027

import unittest

from mirroropd.training.gsm8k import response_diagnostics, score_response
from mirroropd.training.tokenization import assert_shared_tokenization


class Tokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 0
    chat_template = "native"

    def get_vocab(self):
        return {"hello": 1, "eos": self.eos_token_id}

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [0, 1]


class EvaluationTest(unittest.TestCase):
    def test_final_answer_precedence_and_format(self):
        result = score_response("Reasoning: 900. #### 1,200.0", "1200")
        self.assertEqual(result["acc_strict"], 1)
        self.assertEqual(result["acc_explicit"], 1)
        result = score_response(r"Wrong intermediate 88, final \boxed{42}", "42")
        self.assertEqual(result["acc"], 1)
        self.assertEqual(result["acc_strict"], 0)
        self.assertEqual(result["format_boxed"], 1)
        result = score_response("#### 41 then a stray 42", "42")
        self.assertEqual(result["acc"], 0)

    def test_flexible_is_separately_reported(self):
        result = score_response("The answer is 42", "42")
        self.assertEqual(result["acc"], 1)
        self.assertEqual(result["acc_explicit"], 0)
        self.assertEqual(score_response("No answer", "42")["parse_failure"], 1)

    def test_eos_and_budget_are_distinct(self):
        result = response_diagnostics([10, 2, 0], [1, 1, 0], 2, 3)
        self.assertEqual(result["response_tokens"], 2)
        self.assertEqual(result["eos_terminated"], 1)
        self.assertEqual(result["budget_exhausted"], 0)
        self.assertEqual(response_diagnostics([10, 11], [1, 1], 2, 2)["budget_exhausted"], 1)
        self.assertEqual(response_diagnostics([10, 2], [1, 1], 2, 2)["budget_exhausted"], 0)

    def test_shared_prefix_rejects_semantic_mismatch(self):
        a, b = Tokenizer(), Tokenizer()
        assert_shared_tokenization(a, b, require_chat=True)
        b.eos_token_id = 3
        with self.assertRaisesRegex(ValueError, "token-to-ID"):
            assert_shared_tokenization(a, b, require_chat=True)
        b.eos_token_id = 2
        b.chat_template = "different"
        with self.assertRaisesRegex(ValueError, "chat templates"):
            assert_shared_tokenization(a, b, require_chat=True)


if __name__ == "__main__":
    unittest.main()
