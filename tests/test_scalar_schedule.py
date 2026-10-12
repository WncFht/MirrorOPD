"""Numerical contracts for scalar-only residual line search."""

import math

from mirroropd.training.scalar_schedule import absorption_feedback, residual_line_search


def test_recovers_quadratic_optimum():
    # E(a) = 5 - 6a + 2a²; optimum is 1.5 and E*=0.5.
    result = residual_line_search(lambda a: 5 - 6 * a + 2 * a * a, 5, 1, 4)
    assert abs(result.step - 1.5) < 1e-12
    assert abs(result.value - 0.5) < 1e-12
    assert result.evaluations == 3
    assert abs(result.slope - 6) < 1e-12
    assert abs(result.curvature - 4) < 1e-12


def test_non_descent_direction_is_rejected():
    result = residual_line_search(lambda a: 1 + a + a * a, 1, 0.1, 1)
    assert result.step == 0
    assert result.value == 1
    assert result.evaluations == 10


def test_nonfinite_trials_backtrack():
    result = residual_line_search(lambda a: math.inf if a > 0.04 else (1 - a) ** 2, 1, 0.1, 1)
    assert result.step == 0.025
    assert result.value < 1
    assert result.evaluations == 4


def test_feedback_cannot_provide_unbounded_growth():
    assert absorption_feedback(1, 1, 0.999, 0.1, 10) == 1.25
    assert absorption_feedback(1, 1, 2, 0.1, 10) == 0.5
    assert absorption_feedback(9, 1, 0.999, 0.1, 10) == 10


def test_constant_loss_scale_cancels_in_adam_but_lr_does_not():
    import torch

    original = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    scaled = torch.nn.Parameter(original.detach().clone())
    larger_lr = torch.nn.Parameter(original.detach().clone())
    opts = [
        torch.optim.AdamW([p], lr=lr, eps=0, weight_decay=0)
        for p, lr in ((original, 0.01), (scaled, 0.01), (larger_lr, 0.08))
    ]
    for gradient in ([1.0, 3.0], [-0.2, 0.8]):
        for parameter, optimizer, scale in zip(
            (original, scaled, larger_lr), opts, (1, 8, 1), strict=True
        ):
            parameter.grad = torch.tensor(gradient, dtype=torch.float64) * scale
            optimizer.step()
    assert torch.allclose(original, scaled, atol=1e-14, rtol=0)
    initial = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert torch.allclose(initial - larger_lr, 8 * (initial - original), atol=1e-14, rtol=0)


def test_variable_loss_scale_changes_adam_direction():
    import torch

    params = [torch.nn.Parameter(torch.zeros(2, dtype=torch.float64)) for _ in range(2)]
    opts = [torch.optim.AdamW([p], lr=1, eps=0, weight_decay=0) for p in params]
    for iteration, gradient in enumerate(([1.0, 3.0], [-0.2, 0.8])):
        old = [p.detach().clone() for p in params]
        for index, (parameter, optimizer) in enumerate(zip(params, opts, strict=True)):
            scale = 8 if index == 1 and iteration == 1 else 1
            parameter.grad = torch.tensor(gradient, dtype=torch.float64) * scale
            optimizer.step()
    directions = [before - p.detach() for before, p in zip(old, params, strict=True)]
    determinant = directions[0][0] * directions[1][1] - directions[0][1] * directions[1][0]
    assert abs(float(determinant)) > 0.01


def test_fixed_sample_square_gradient_has_center_derivative():
    import torch

    z = torch.tensor([0.8, -0.3, 0.2], dtype=torch.float64, requires_grad=True)
    qlog = torch.tensor([0.4, -0.1, 0.6], dtype=torch.float64).log_softmax(-1)
    tlog = torch.tensor([-0.2, 0.4, 0.9], dtype=torch.float64).log_softmax(-1)
    logp = z.log_softmax(-1)
    center = (qlog.exp() * (qlog - logp)).sum()
    teacher_center = (qlog.exp() * (qlog - tlog)).sum()
    residual = logp[0] - qlog[0] + center - 0.3 * (tlog[0] - qlog[0] + teacher_center)
    square_gradient = torch.autograd.grad(residual.square() / 0.6, z, retain_graph=True)[0]
    score_gradient = torch.autograd.grad(residual.detach() * logp[0] / 0.3, z, retain_graph=True)[0]
    center_gradient = torch.autograd.grad(center, z)[0]
    correction = residual.detach() * center_gradient / 0.3
    assert torch.allclose(square_gradient - score_gradient, correction, atol=1e-14)
    assert float(correction.norm()) > 0.01


def test_topk_expected_field_matches_predicted_bias():
    import torch

    from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
    from mirroropd.training.losses import compute_mirror_opd_token_loss

    z = torch.tensor([0.8, -0.3, 0.2, -0.5], requires_grad=True)
    anchor = torch.tensor([0.4, -0.1, 0.6, 0.2])
    teacher = torch.tensor([-0.2, 0.4, 0.9, -0.7])
    actions = torch.arange(4)
    cache = build_mirror_opd_cache_from_full_logits(
        anchor.repeat(4, 1), teacher.repeat(4, 1), actions, torch.ones(4, dtype=torch.bool), 1
    )
    out = compute_mirror_opd_token_loss(
        z.repeat(4, 1),
        actions,
        cache["mirror_anchor_ids"],
        cache["mirror_anchor_head_log_probs"],
        cache["mirror_anchor_tail_log_prob"],
        cache["mirror_anchor_action_log_prob"],
        cache["mirror_teacher_action_log_prob"],
        cache["mirror_teacher_center"],
        "bpo",
        0.3,
    )
    qlog, tlog, logp = anchor.log_softmax(-1), teacher.log_softmax(-1), z.log_softmax(-1)
    q = qlog.exp()
    sampled_field = torch.autograd.grad(
        (q * out["distillation_losses"]).sum(), z, retain_graph=True
    )[0]
    residual = z - 0.7 * anchor - 0.3 * teacher
    residual = residual - (q * residual).sum()
    exact = (q * residual.square()).sum() / 0.6
    exact_gradient = torch.autograd.grad(exact, z)[0]
    center = (q * (qlog - logp.detach())).sum()
    teacher_center = (q * (qlog - tlog)).sum()
    bias = (
        out["mirror_student_center"][0]
        - center
        - 0.3 * (out["mirror_teacher_center"][0] - teacher_center)
    )
    assert torch.allclose(
        sampled_field - exact_gradient, bias * (q - logp.detach().exp()) / 0.3, atol=2e-7
    )
