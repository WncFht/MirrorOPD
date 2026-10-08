"""Check on-policy and token-IS OPD score surrogates on a full categorical support."""

import torch

from mirroropd.training.cache import build_mirror_anchor_cache, complete_mirror_teacher_cache
from mirroropd.training.losses import compute_mirror_opd_token_loss


def _loss(
    method: str, logits: torch.Tensor, q: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    actions = torch.arange(q.numel())
    anchor = build_mirror_anchor_cache(
        q.log().expand(q.numel(), -1), actions, torch.ones(q.numel(), dtype=torch.bool), q.numel()
    )
    teacher_cache = complete_mirror_teacher_cache(teacher.log().expand(q.numel(), -1), anchor)
    result = compute_mirror_opd_token_loss(
        logits.expand(q.numel(), -1),
        actions,
        teacher_cache["mirror_anchor_ids"],
        teacher_cache["mirror_anchor_head_log_probs"],
        teacher_cache["mirror_anchor_tail_log_prob"],
        teacher_cache["mirror_anchor_action_log_prob"],
        teacher_cache["mirror_teacher_action_log_prob"],
        teacher_cache["mirror_teacher_center"],
        method,
        1.0,
        teacher_head_log_probs=teacher_cache["mirror_teacher_head_log_probs"],
        teacher_tail_log_prob=teacher_cache["mirror_teacher_tail_log_prob"],
    )
    return (q * result["distillation_losses"]).sum()


def main() -> None:
    q = torch.tensor([0.2, 0.5, 0.3])
    teacher = torch.tensor([0.6, 0.1, 0.3])

    on_logits = q.log().requires_grad_()
    on_grad = torch.autograd.grad(_loss("opd_onpolicy", on_logits, q, teacher), on_logits)[0]
    p = on_logits.softmax(-1)
    exact_on = (p * (p.log() - teacher.log())).sum()
    exact_on_grad = torch.autograd.grad(exact_on, on_logits)[0]
    torch.testing.assert_close(on_grad, exact_on_grad, atol=2e-7, rtol=2e-6)

    is_logits = torch.tensor([-0.4, 0.2, 0.8], requires_grad=True)
    is_grad = torch.autograd.grad(_loss("opd_is", is_logits, q, teacher), is_logits)[0]
    p = is_logits.softmax(-1)
    exact_is = (p * (p.log() - teacher.log())).sum()
    exact_is_grad = torch.autograd.grad(exact_is, is_logits)[0]
    torch.testing.assert_close(is_grad, exact_is_grad, atol=2e-7, rtol=2e-6)
    print("PASS: OPD on-policy and token-IS score gradients match reverse KL")


if __name__ == "__main__":
    main()
