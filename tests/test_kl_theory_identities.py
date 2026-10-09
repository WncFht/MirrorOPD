"""Enumerate finite vocabularies to audit the KL chapter's estimator identities."""

from __future__ import annotations

import math
import random
import unittest


def softmax(logits):
    weights = [math.exp(x - max(logits)) for x in logits]
    return [x / sum(weights) for x in weights]


def mean(prob, values):
    return [sum(p * v[k] for p, v in zip(prob, values, strict=True)) for k in range(len(values[0]))]


def variance(prob, values):
    center = mean(prob, values)
    return sum(
        p * sum((x - m) ** 2 for x, m in zip(v, center, strict=True))
        for p, v in zip(prob, values, strict=True)
    )


class KLTheoryIdentities(unittest.TestCase):
    def assert_vector_close(self, actual, expected, places=11):
        for x, y in zip(actual, expected, strict=True):
            assert math.isclose(x, y, rel_tol=10 ** (-places), abs_tol=10 ** (-places)), (x, y)

    def test_finite_enumeration(self):
        rng = random.Random(42)
        for _ in range(100):
            size = 7
            p, q, rho = [softmax([rng.uniform(-3, 3) for _ in range(size)]) for _ in range(3)]
            r = [math.log(a / b) for a, b in zip(p, q, strict=True)]
            support = set(sorted(range(size), key=lambda i: q[i])[-3:])
            tail = [i for i in range(size) if i not in support]
            tau = sum(p[i] for i in tail)
            b0 = math.log(tau / sum(q[i] for i in tail))
            score = [[float(i == k) - p[k] for k in range(size)] for i in range(size)]
            z = [[r[i] * x for x in score[i]] for i in range(size)]
            target = mean(p, z)
            head = [sum(p[i] * z[i][k] for i in support) for k in range(size)]
            dp = [sum(p[i] * score[i][k] for i in support) for k in range(size)]
            mu = [sum(p[i] * z[i][k] for i in tail) / tau for k in range(size)]
            tail_score = [sum(p[i] * score[i][k] for i in tail) / tau for k in range(size)]
            bucket = [b0 * x for x in tail_score]
            fluct = [[b0 * (score[i][k] - tail_score[k]) for k in range(size)] for i in range(size)]
            h = sum(p[i] * r[i] for i in support)
            kl = sum(p[i] * r[i] for i in range(size))

            # Independently enumerate outcomes, then compare with conditional-moment formulas.
            for beta in (0.0, b0, 0.73):
                values = [[h + tau * beta + (r[i] - beta if i in tail else 0)] for i in range(size)]
                assert math.isclose(mean(p, values)[0], kl, rel_tol=1e-11, abs_tol=1e-11)
                tail_mean = sum(p[i] * r[i] for i in tail) / tau
                expected_var = sum(p[i] * (r[i] - tail_mean) ** 2 for i in tail)
                expected_var += (1 - tau) * tau * (tail_mean - beta) ** 2
                assert math.isclose(variance(p, values), expected_var, rel_tol=1e-11, abs_tol=1e-11)

            for control in ([0.0] * size, bucket, mu):
                for gamma in (0.0, 1.0, -0.4):
                    values = [
                        [
                            head[k]
                            + tau * control[k]
                            + (z[i][k] - control[k] - gamma * fluct[i][k] if i in tail else 0)
                            for k in range(size)
                        ]
                        for i in range(size)
                    ]
                    self.assert_vector_close(mean(p, values), target)
                    expected_var = sum(
                        p[i]
                        * sum((z[i][k] - gamma * fluct[i][k] - mu[k]) ** 2 for k in range(size))
                        for i in tail
                    )
                    expected_var += (
                        (1 - tau) * tau * sum((mu[k] - control[k]) ** 2 for k in range(size))
                    )
                    assert math.isclose(
                        variance(p, values), expected_var, rel_tol=1e-11, abs_tol=1e-11
                    )

                    if gamma == 0:
                        weighted = [
                            [
                                head[k]
                                + tau * control[k]
                                + (p[i] / rho[i] * (z[i][k] - control[k]) if i in tail else 0)
                                for k in range(size)
                            ]
                            for i in range(size)
                        ]
                        self.assert_vector_close(mean(rho, weighted), target)
                        expected_var = sum(
                            p[i] ** 2
                            / rho[i]
                            * sum((z[i][k] - control[k]) ** 2 for k in range(size))
                            for i in tail
                        ) - tau**2 * sum((mu[k] - control[k]) ** 2 for k in range(size))
                        assert math.isclose(
                            variance(rho, weighted), expected_var, rel_tol=1e-11, abs_tol=1e-11
                        )

            for alpha in (0.0, 1.0, -0.6):
                values = [
                    [
                        head[k] + alpha * dp[k] + ((r[i] + alpha) * score[i][k] if i in tail else 0)
                        for k in range(size)
                    ]
                    for i in range(size)
                ]
                self.assert_vector_close(mean(p, values), target)
            mismatched = [
                [head[k] + dp[k] + (z[i][k] if i in tail else 0) for k in range(size)]
                for i in range(size)
            ]
            self.assert_vector_close(mean(p, mismatched), [target[k] + dp[k] for k in range(size)])

    def test_gradient_against_finite_differences(self):
        logits = [0.7, -0.4, 1.2, -0.8]
        q = softmax([-0.6, 0.9, 0.2, -0.1])
        p = softmax(logits)
        r = [math.log(a / b) for a, b in zip(p, q, strict=True)]
        score = [[float(i == k) - p[k] for k in range(4)] for i in range(4)]
        g = mean(p, [[r[i] * x for x in score[i]] for i in range(4)])

        def objective(z):
            prob = softmax(z)
            return sum(a * math.log(a / b) for a, b in zip(prob, q, strict=True))

        finite_diff = []
        for k in range(4):
            plus, minus = logits.copy(), logits.copy()
            plus[k] += 1e-5
            minus[k] -= 1e-5
            finite_diff.append((objective(plus) - objective(minus)) / 2e-5)
        self.assert_vector_close(g, finite_diff, places=9)
        k3pp_grad = [
            [(math.exp(-r[i]) + r[i] - 1 + 1 - math.exp(-r[i])) * x for x in score[i]]
            for i in range(4)
        ]
        self.assert_vector_close(mean(p, k3pp_grad), finite_diff, places=9)
        live_grad = [[(r[i] + 1) * x for x in score[i]] for i in range(4)]
        self.assert_vector_close(mean(p, live_grad), finite_diff, places=9)

    def test_independent_and_same_action_baselines(self):
        p = [0.2, 0.3, 0.5]
        q = [0.5, 0.1, 0.4]
        r = [math.log(a / b) for a, b in zip(p, q, strict=True)]
        score = [[float(i == k) - p[k] for k in range(3)] for i in range(3)]
        target = mean(p, [[r[i] * x for x in score[i]] for i in range(3)])
        baseline_prob, baseline = [0.2, 0.5, 0.3], [-0.8, 1.7, 3.4]
        center = sum(a * b for a, b in zip(baseline_prob, baseline, strict=True))
        baseline_var = sum(
            a * (b - center) ** 2 for a, b in zip(baseline_prob, baseline, strict=True)
        )
        joint_prob = [p[i] * baseline_prob[j] for i in range(3) for j in range(3)]
        joint_value = [
            [(r[i] - baseline[j]) * x for x in score[i]] for i in range(3) for j in range(3)
        ]
        fixed_value = [[(r[i] - center) * x for x in score[i]] for i in range(3)]
        score_second_moment = sum(p[i] * sum(x * x for x in score[i]) for i in range(3))
        self.assert_vector_close(mean(joint_prob, joint_value), target)
        assert math.isclose(
            variance(joint_prob, joint_value),
            variance(p, fixed_value) + score_second_moment * baseline_var,
            abs_tol=1e-12,
        )
        same_action = [[(r[i] - r[i]) * x for x in score[i]] for i in range(3)]
        self.assert_vector_close(mean(p, same_action), [0.0] * 3)
        assert sum(x * x for x in target) > 1e-3

    def test_mean_outer_invariant(self):
        for eta in (0.1, 0.5, 1.0):
            for error in (0.0, 0.05, 0.4):
                steady = 2 * error / (eta * (1 - 2 * error))
                for initial in (0.0, 1.0, 100.0):
                    bound = max(initial, steady)
                    span = initial
                    for _ in range(100):
                        span = span - eta * span / (1 + eta * span) + 2 * error
                        assert span <= bound + 1e-12


if __name__ == "__main__":
    unittest.main()
