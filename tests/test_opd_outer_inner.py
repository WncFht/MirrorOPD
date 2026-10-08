"""Oracle validation of the Tail-aware outer/inner score-function runner."""

import torch

from mirroropd.opd_categorical import CategoricalOPDConfig, run_categorical_opd, tail_aware_kl
from mirroropd.opd_pg import coefficients, sampled_pg_loss
from mirroropd.token_inner import bpo_loss, mdpo_loss, pmd_mean_loss


def main():
    torch.manual_seed(20261008)
    dtype = torch.float64
    for _ in range(20):
        anchor = torch.randn(8, dtype=dtype).softmax(-1)
        teacher = torch.randn(8, dtype=dtype).softmax(-1)
        logits = torch.randn(8, dtype=dtype).requires_grad_(True)
        logp = logits.log_softmax(-1)
        reward = teacher.log() - anchor.log()
        u = logp - anchor.log()
        kt = tail_aware_kl(anchor.log(), teacher.log(), 8)
        ks = tail_aware_kl(anchor.log(), logp, 8)
        eta = 0.5
        for method, coefficient, weights, exact in [
            ("pmd_mean", u - eta * (reward + kt), anchor, pmd_mean_loss),
            ("bpo", (u + ks) / eta - reward - kt, anchor, bpo_loss),
            ("mdpo", u / eta - reward, logp.exp().detach(), mdpo_loss),
        ]:
            surrogate = (weights * coefficient.detach() * logp).sum()
            actual = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
            expected = torch.autograd.grad(
                exact(logits, anchor, reward, eta), logits, retain_graph=True
            )[0]
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
            # Enumerate the expectation of one rollout action, including
            # MDPO's frozen importance ratio. Repeat at partial support.
            for support in [1, 3, 8]:
                kt_partial = tail_aware_kl(anchor.log(), teacher.log(), support)
                ks_partial = tail_aware_kl(anchor.log(), logp, support)
                c = coefficients(
                    method,
                    logp,
                    anchor.log(),
                    teacher.log(),
                    kt_partial,
                    ks_partial,
                    eta,
                )
                w = logp.exp().detach() if method == "mdpo" else anchor
                oracle = (w * c.detach() * logp).sum()
                enumerated = sum(
                    anchor[a]
                    * sampled_pg_loss(
                        method,
                        logp[a : a + 1],
                        anchor.log()[a : a + 1],
                        teacher.log()[a : a + 1],
                        kt_partial,
                        ks_partial,
                        eta,
                    )
                    for a in range(8)
                )
                torch.testing.assert_close(
                    torch.autograd.grad(oracle, logits, retain_graph=True)[0],
                    torch.autograd.grad(enumerated, logits, retain_graph=True)[0],
                    atol=1e-12,
                    rtol=1e-10,
                )
    anchor = torch.tensor([0.6, 0.25, 0.1, 0.05], dtype=dtype)
    teacher = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=dtype)
    for method in ["pmd_mean", "bpo", "mdpo"]:
        for rho in [None, 0.5]:
            for topk in [1, 2, 4]:
                cfg = CategoricalOPDConfig(method=method, soft_sync=rho, top_k=topk)
                history = run_categorical_opd(anchor, teacher, cfg)
                assert len(history) == cfg.outer_steps
                assert all(
                    torch.isfinite(torch.tensor(list(row.values()))).all() for row in history
                )
                assert history[-1]["inner_to_teacher_exact"] < history[0]["teacher_exact"]
                torch.manual_seed(20261008)
                cfg.sampled_inner = True
                cfg.rollout_batch = 4096
                sampled = run_categorical_opd(anchor, teacher, cfg)
                assert sampled[-1]["inner_to_teacher_exact"] < sampled[0]["teacher_exact"]
    print("PASS: 60 exact gradients; 180 enumerated MC gradients; 36 outer/inner runs")


if __name__ == "__main__":
    main()
