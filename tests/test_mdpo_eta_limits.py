"""Enumerate frozen-anchor actions to verify the production MDPO eta endpoints."""

import math

import torch

from mirroropd.training.cache import build_mirror_anchor_cache, complete_mirror_teacher_cache
from mirroropd.training.losses import compute_mirror_opd_token_loss


def main():
    torch.manual_seed(42)
    for eta in (0.03, 0.1, 0.3, 1.0, math.inf):
        q = torch.tensor([0.2, 0.5, 0.3])
        teacher = torch.tensor([0.6, 0.1, 0.3])
        logits = torch.tensor([-0.4, 0.2, 0.8], requires_grad=True)
        actions = torch.arange(3)
        cache = build_mirror_anchor_cache(q.log().expand(3, -1), actions, torch.ones(3).bool(), 2)
        cache = complete_mirror_teacher_cache(teacher.log().expand(3, -1), cache)
        result = compute_mirror_opd_token_loss(
            logits.expand(3, -1),
            actions,
            cache["mirror_anchor_ids"],
            cache["mirror_anchor_head_log_probs"],
            cache["mirror_anchor_tail_log_prob"],
            cache["mirror_anchor_action_log_prob"],
            cache["mirror_teacher_action_log_prob"],
            cache["mirror_teacher_center"],
            "mdpo",
            eta,
        )
        estimated = torch.autograd.grad((q * result["distillation_losses"]).sum(), logits)[0]
        p = logits.softmax(-1)
        objective = (p * (logits.log_softmax(-1) - q.log())).sum() / eta
        objective = objective - (p * (teacher.log() - q.log())).sum()
        exact = torch.autograd.grad(objective, logits)[0]
        torch.testing.assert_close(estimated, exact, atol=2e-7, rtol=2e-6)
        if eta == 1:
            kl = (p * (logits.log_softmax(-1) - teacher.log())).sum()
            torch.testing.assert_close(objective, kl)
        if math.isinf(eta):
            torch.testing.assert_close(objective, -(p * (teacher.log() - q.log())).sum())
    print("PASS: MDPO eta scan exact gradients; eta=1 teacher KL; infinity frozen reward")


if __name__ == "__main__":
    main()
