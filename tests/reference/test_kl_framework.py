"""Enumerate the unified residual-integral, partition, and control identities."""

from __future__ import annotations

import math
import random
import unittest


def softmax(logits):
    weights = [math.exp(value - max(logits)) for value in logits]
    return [value / sum(weights) for value in weights]


def dot(left, right):
    return sum(x * y for x, y in zip(left, right, strict=True))


def subtract(left, right):
    return [x - y for x, y in zip(left, right, strict=True)]


def mean(prob, values):
    return [sum(prob[i] * values[i][j] for i in range(len(prob))) for j in range(len(values[0]))]


def covariance(prob, values):
    center = mean(prob, values)
    residual = [subtract(value, center) for value in values]
    return [
        [
            sum(prob[i] * residual[i][j] * residual[i][k] for i in range(len(prob)))
            for k in range(len(center))
        ]
        for j in range(len(center))
    ]


def variance(prob, values):
    matrix = covariance(prob, values)
    return sum(matrix[j][j] for j in range(len(matrix)))


def estimate(nu, rho, function, proxy):
    exact = mean(nu, proxy)
    return [
        [exact[j] + nu[i] / rho[i] * (function[i][j] - proxy[i][j]) for j in range(len(exact))]
        for i in range(len(rho))
    ]


class UnifiedKLFramework(unittest.TestCase):
    def assert_vector_close(self, actual, expected, tolerance=1e-11):
        for left, right in zip(actual, expected, strict=True):
            assert math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)

    def test_arbitrary_integrands_covariance_and_mse(self):
        rng = random.Random(128)
        for _ in range(32):
            p, q, rho = [softmax([rng.uniform(-2, 2) for _ in range(6)]) for _ in range(3)]
            function, proxy = [
                [[rng.uniform(-3, 3) for _ in range(3)] for _ in range(6)] for _ in range(2)
            ]
            for nu in (p, q):
                values = estimate(nu, rho, function, proxy)
                residual = [subtract(f, a) for f, a in zip(function, proxy, strict=True)]
                residual_mean = mean(nu, residual)
                self.assert_vector_close(mean(rho, values), mean(nu, function))
                actual = covariance(rho, values)
                for j in range(3):
                    expected = [
                        sum(nu[i] ** 2 / rho[i] * residual[i][j] * residual[i][k] for i in range(6))
                        - residual_mean[j] * residual_mean[k]
                        for k in range(3)
                    ]
                    self.assert_vector_close(actual[j], expected)
                target = mean(nu, function)
                mse = sum(
                    rho[i] * dot(subtract(values[i], target), subtract(values[i], target))
                    for i in range(6)
                )
                assert math.isclose(mse, variance(rho, values), abs_tol=1e-10)

    def test_combined_integration_weight_and_support_errors(self):
        nu, rho = [0.2, 0.3, 0.5], [0.4, 0.6, 0.0]
        function, proxy = [[1.0, -2.0], [3.0, 1.0], [-1.0, 4.0]], [[0.2, 0.5]] * 3
        delta, weights = [0.1, -0.3], [0.8, 0.4]
        exact, target = mean(nu, proxy), mean(nu, function)
        residual = [subtract(f, a) for f, a in zip(function, proxy, strict=True)]
        values = [
            [exact[j] + delta[j] + weights[i] * residual[i][j] for j in range(2)] for i in range(2)
        ]
        expected_bias = [
            delta[j]
            + sum((rho[i] * weights[i] - nu[i]) * residual[i][j] for i in range(2))
            - nu[2] * residual[2][j]
            for j in range(2)
        ]
        bias = subtract(mean(rho[:2], values), target)
        self.assert_vector_close(bias, expected_bias)
        mse = sum(
            rho[i] * dot(subtract(values[i], target), subtract(values[i], target)) for i in range(2)
        )
        assert math.isclose(mse, variance(rho[:2], values) + dot(bias, bias), abs_tol=1e-11)

    def test_partition_reverse_and_forward_gradients(self):
        logits, q, cells = [0.3, -0.5, 1.1, 0.7], [0.1, 0.4, 0.2, 0.3], [[0], [1, 2], [3]]
        p = softmax(logits)
        score = [[float(i == j) - p[j] for j in range(4)] for i in range(4)]
        for direction in ("reverse", "forward"):
            gradient = [0.0] * 4
            for cell in cells:
                mass_p, mass_q = sum(p[i] for i in cell), sum(q[i] for i in cell)
                coefficient = (
                    math.log(mass_p / mass_q) + 1 if direction == "reverse" else -mass_q / mass_p
                )
                for j in range(4):
                    gradient[j] += coefficient * sum(p[i] * score[i][j] for i in cell)

            def objective(parameters, direction=direction):
                prob = softmax(parameters)
                masses = [(sum(prob[i] for i in cell), sum(q[i] for i in cell)) for cell in cells]
                return sum(
                    a * math.log(a / b) if direction == "reverse" else b * math.log(b / a)
                    for a, b in masses
                )

            finite_difference = []
            for j in range(4):
                plus, minus = logits.copy(), logits.copy()
                plus[j] += 1e-5
                minus[j] -= 1e-5
                finite_difference.append((objective(plus) - objective(minus)) / 2e-5)
            self.assert_vector_close(gradient, finite_difference, tolerance=1e-9)

    def test_equal_integral_proxies_zero_shape_counterexample(self):
        p, q = [0.4, 0.2, 0.4], [0.7, 0.1, 0.2]
        score = [[float(i == j) - p[j] for j in range(3)] for i in range(3)]
        ratio = [math.log(p[i] / q[i]) for i in range(3)]
        function = [[ratio[i] * value for value in score[i]] for i in range(3)]
        tail_score = [(p[1] * score[1][j] + p[2] * score[2][j]) / 0.6 for j in range(3)]
        constant = [math.log(2) * value for value in tail_score]
        res_proxy = [function[0], constant, constant]
        scta_proxy = [function[0], *[[math.log(2) * value for value in score[i]] for i in (1, 2)]]
        self.assert_vector_close(mean(p, res_proxy), mean(p, scta_proxy))
        res_values, scta_values = [
            estimate(p, p, function, proxy) for proxy in (res_proxy, scta_proxy)
        ]
        self.assert_vector_close(mean(p, res_values), mean(p, function))
        assert math.isclose(variance(p, res_values), 4 / 15 * math.log(2) ** 2, abs_tol=1e-11)
        assert variance(p, scta_values) < 1e-25
        scalar = [[value] for value in ratio]
        scalar_proxy = [[ratio[0]], [math.log(2)], [math.log(2)]]
        assert variance(p, estimate(p, p, scalar, scalar_proxy)) < 1e-25

    def test_optimal_proposal_and_zero_residual_support(self):
        nu = [0.1, 0.2, 0.3, 0.4]
        function, proxy = [[1.0, 2.0], [2.0, -1.0], [-2.0, 3.0], [0.0, 0.0]], [[0.0, 0.0]] * 4
        norms = [math.sqrt(dot(value, value)) for value in function]
        total = dot(nu, norms)
        rho = [nu[i] * norms[i] / total for i in range(4)]
        # The zero-residual token needs no coverage; enumerate only sampled outcomes.
        values = estimate(nu[:3], rho[:3], function[:3], proxy[:3])
        center = mean(nu, function)
        expected = total**2 - dot(center, center)
        assert math.isclose(variance(rho[:3], values), expected, abs_tol=1e-11)
        assert variance(nu, estimate(nu, nu, function, proxy)) > expected

    def test_partition_covariance_and_optimal_tail_constant(self):
        p, function, tail = (
            [0.1, 0.2, 0.3, 0.4],
            [[1.0, 2.0], [0.0, 3.0], [-1.0, 0.0], [2.0, -2.0]],
            [1, 2, 3],
        )
        tau = sum(p[i] for i in tail)
        tail_prob = [p[i] / tau for i in tail]
        tail_function = [function[i] for i in tail]
        tail_mean = mean(tail_prob, tail_function)
        for constant in ([0.0, 0.0], tail_mean, [1.5, -0.4]):
            proxy = [function[0], constant, constant, constant]
            actual = covariance(p, estimate(p, p, function, proxy))
            within = covariance(tail_prob, tail_function)
            difference = subtract(tail_mean, constant)
            for j in range(2):
                self.assert_vector_close(
                    actual[j],
                    [
                        tau * within[j][k] + (1 - tau) * tau * difference[j] * difference[k]
                        for k in range(2)
                    ],
                )
        assert math.isclose(
            variance(p, estimate(p, p, function, [function[0], tail_mean, tail_mean, tail_mean])),
            tau * variance(tail_prob, tail_function),
            abs_tol=1e-11,
        )

    def test_single_and_nonorthogonal_double_control_regret(self):
        nu, rho = [0.2, 0.3, 0.5], [0.4, 0.35, 0.25]
        function = [[1.0, -2.0], [3.0, 0.5], [-1.0, 4.0]]
        controls = [[[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]], [[2.0, 1.0], [1.0, 2.0], [3.0, 0.0]]]
        weighted = [[nu[i] / rho[i] * value for value in function[i]] for i in range(3)]
        target = mean(nu, function)
        centered = [subtract(value, target) for value in weighted]
        innovations = []
        for control in controls:
            integral = mean(nu, control)
            innovations.append(
                [[nu[i] / rho[i] * control[i][j] - integral[j] for j in range(2)] for i in range(3)]
            )
        gram = [
            [sum(rho[i] * dot(left[i], right[i]) for i in range(3)) for right in innovations]
            for left in innovations
        ]
        correlation = [
            sum(rho[i] * dot(control[i], centered[i]) for i in range(3)) for control in innovations
        ]
        assert abs(gram[0][1]) > 0.1
        determinant = gram[0][0] * gram[1][1] - gram[0][1] ** 2
        double_optimum = [
            (gram[1][1] * correlation[0] - gram[0][1] * correlation[1]) / determinant,
            (gram[0][0] * correlation[1] - gram[0][1] * correlation[0]) / determinant,
        ]
        for optimum in ([correlation[0] / gram[0][0]], double_optimum):
            count = len(optimum)

            def outcomes(coefficients, count=count):
                return [
                    [
                        weighted[i][j]
                        - sum(coefficients[k] * innovations[k][i][j] for k in range(count))
                        for j in range(2)
                    ]
                    for i in range(3)
                ]

            minimum = variance(rho, outcomes(optimum))
            for displacement in ([0.0] * count, [0.6] * count, [-0.4] * count):
                coefficients = [optimum[k] + displacement[k] for k in range(count)]
                regret = sum(
                    displacement[j] * gram[j][k] * displacement[k]
                    for j in range(count)
                    for k in range(count)
                )
                assert math.isclose(
                    variance(rho, outcomes(coefficients)) - minimum, regret, abs_tol=1e-11
                )
                self.assert_vector_close(mean(rho, outcomes(coefficients)), target)

    def test_conditional_tail_sampling_cancels_constant_control(self):
        p = [0.2, 0.3, 0.5]
        function = [[1.0, -0.5], [2.0, 3.0], [-1.0, 0.7]]
        tau = 0.8
        rho = [p[1] / tau, p[2] / tau]
        target = mean(p, function)
        for control in ([0.0, 0.0], [2.0, -0.4]):
            values = [
                [
                    p[0] * function[0][j] + tau * control[j] + tau * (function[i][j] - control[j])
                    for j in range(2)
                ]
                for i in (1, 2)
            ]
            self.assert_vector_close(mean(rho, values), target)
            assert math.isclose(
                variance(rho, values), tau**2 * variance(rho, function[1:]), abs_tol=1e-12
            )

    def test_autodiff_transform_and_off_policy_gram_coupling(self):
        p, q, rho = [0.2, 0.3, 0.5], [0.5, 0.1, 0.4], [0.5, 0.4, 0.1]
        score = [[float(i == j) - p[j] for j in range(3)] for i in range(3)]
        r = [math.log(p[i] / q[i]) for i in range(3)]
        expected = mean(p, [[r[i] * s for s in score[i]] for i in range(3)])
        transformed = [
            [(math.exp(-r[i]) + r[i] - 1 + 1 - math.exp(-r[i])) * s for s in score[i]]
            for i in range(3)
        ]
        self.assert_vector_close(mean(p, transformed), expected)
        tau = 0.8
        tail_score = [(p[1] * score[1][j] + p[2] * score[2][j]) / tau for j in range(3)]
        fluctuation = [[s - tail_score[j] for j, s in enumerate(row)] for row in score]
        for proposal in (p, rho):
            cross = [
                sum(p[i] ** 2 / proposal[i] * fluctuation[i][j] for i in (1, 2)) for j in range(3)
            ]
            if proposal == p:
                self.assert_vector_close(cross, [0.0] * 3)
            else:
                assert dot(cross, cross) > 0.1

    def test_off_policy_score_baseline_optimum(self):
        p, q, rho = [0.2, 0.3, 0.5], [0.5, 0.1, 0.4], [0.5, 0.4, 0.1]
        score = [[float(i == j) - p[j] for j in range(3)] for i in range(3)]
        ratio = [math.log(p[i] / q[i]) for i in range(3)]
        weights = [p[i] / rho[i] for i in range(3)]
        second_moment = sum(rho[i] * weights[i] ** 2 * dot(score[i], score[i]) for i in range(3))
        optimum = (
            sum(rho[i] * weights[i] ** 2 * ratio[i] * dot(score[i], score[i]) for i in range(3))
            / second_moment
        )

        def values(baseline):
            return [[weights[i] * (ratio[i] - baseline) * s for s in score[i]] for i in range(3)]

        for displacement in (-0.7, 0.0, 1.1):
            assert math.isclose(
                variance(rho, values(optimum + displacement)) - variance(rho, values(optimum)),
                second_moment * displacement**2,
                abs_tol=1e-11,
            )
            self.assert_vector_close(
                mean(rho, values(optimum + displacement)), mean(rho, values(0))
            )


if __name__ == "__main__":
    unittest.main()
