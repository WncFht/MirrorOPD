"""Local Transformers integration using a tiny random causal LM, no download."""

import copy

import torch
from opd_lm import freeze_cache, log_probs, train_trajectory
from transformers import GPT2Config, GPT2LMHeadModel


def main():
    torch.manual_seed(20261008)
    cfg = GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    initial = GPT2LMHeadModel(cfg).eval()
    teacher = GPT2LMHeadModel(cfg).eval()
    for method in ["pmd_mean", "bpo", "mdpo"]:
        student = copy.deepcopy(initial)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        for _outer in range(3):
            anchor = copy.deepcopy(student).eval()
            prompt = [1, 2]
            generated = []
            with torch.no_grad():
                for _ in range(4):
                    ids = torch.tensor([prompt + generated])
                    p = anchor(ids).logits[0, -1].softmax(-1)
                    generated.append(int(torch.multinomial(p, 1)))
            cache = freeze_cache(anchor, teacher, prompt, generated, 4)
            frozen = [x.clone() for x in anchor.parameters()]
            for _ in range(4):
                optimizer.zero_grad()
                loss = train_trajectory(student, cache, method, 0.5)
                assert torch.isfinite(loss)
                loss.backward()
                assert all(
                    p.grad is None or torch.isfinite(p.grad).all() for p in student.parameters()
                )
                optimizer.step()
            assert all(torch.equal(a, b) for a, b in zip(frozen, anchor.parameters(), strict=True))
            assert any(
                not torch.equal(a, b) for a, b in zip(frozen, student.parameters(), strict=True)
            )
        print(f"PASS {method}: 3 outer rollouts x 4 inner steps; frozen anchor, finite gradients")
    # Full support must equal full-vocabulary anchor || teacher KL.
    cache = freeze_cache(initial, teacher, [1, 2], [3, 4], 16)
    a = log_probs(initial, cache.input_ids, 2)
    t = log_probs(teacher, cache.input_ids, 2)
    torch.testing.assert_close(
        cache.teacher_center, (a.exp() * (a - t)).sum(-1), atol=1e-6, rtol=1e-5
    )
    print("PASS full-support cache KL matches oracle")


if __name__ == "__main__":
    main()
