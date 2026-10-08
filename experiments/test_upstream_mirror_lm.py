"""Causal-LM outer/inner integration for the actual upstream loss adapter.

Runs upstream tensor functions without Ray/CUDA service imports. Generation
is a CPU oracle, not a substitute for the production vLLM synchronization test.
"""

import copy
from pathlib import Path
from types import SimpleNamespace

import torch
from test_upstream_opd_semantics import load_functions
from transformers import GPT2Config, GPT2LMHeadModel


def main():
    torch.manual_seed(20261008)
    torch.set_num_threads(1)
    source = (
        Path(__file__).resolve().parents[1]
        / "upstream/TA-OPD/verl/verl"
        / "trainer/distillation/fsdp/losses.py"
    )
    functions = load_functions(
        source,
        [
            "build_mirror_opd_cache_from_full_logits",
            "build_mirror_anchor_cache",
            "complete_mirror_teacher_cache",
            "compute_mirror_opd_token_loss",
            "compute_mirror_opd_from_cache",
        ],
        {"torch": torch, "get_ulysses_sequence_parallel_world_size": lambda: 1},
    )
    cfg = GPT2Config(
        vocab_size=16,
        n_positions=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        bos_token_id=1,
        eos_token_id=15,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    initial = GPT2LMHeadModel(cfg).eval()
    teacher = GPT2LMHeadModel(cfg).eval().requires_grad_(False)
    for method in ("pmd_mean", "bpo", "mdpo"):
        student = copy.deepcopy(initial)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)
        settings = SimpleNamespace(
            distillation_loss=SimpleNamespace(
                loss_mode="mirror_" + method, mirror_eta=0.5
            )
        )
        for outer in range(3):
            # Hard synchronization: next anchor includes every inner update.
            anchor = copy.deepcopy(student).eval().requires_grad_(False)
            frozen_params = [p.clone() for p in anchor.parameters()]
            prompt = [1, 2]
            tokens = prompt.copy()
            with torch.no_grad():
                for _ in range(4):
                    rollout_ids = torch.tensor([tokens])
                    p = (
                        anchor(rollout_ids, attention_mask=torch.ones_like(rollout_ids))
                        .logits[0, -1]
                        .softmax(-1)
                    )
                    tokens.append(int(torch.multinomial(p, 1)))
                ids = torch.tensor([tokens])
                actions = ids.roll(-1, dims=1)
                valid = torch.zeros_like(ids, dtype=torch.bool)
                valid[:, len(prompt) - 1 : -1] = True
                anchor_cache = functions["build_mirror_anchor_cache"](
                    anchor(ids, attention_mask=torch.ones_like(ids)).logits,
                    actions,
                    valid,
                    topk=4,
                )
                cache = functions["complete_mirror_teacher_cache"](
                    teacher(ids, attention_mask=torch.ones_like(ids)).logits,
                    anchor_cache,
                )
            frozen_cache = {key: value.clone() for key, value in cache.items()}
            centers = []
            for inner in range(4):
                optimizer.zero_grad()
                outputs = functions["compute_mirror_opd_from_cache"](
                    student(ids, attention_mask=torch.ones_like(ids)).logits,
                    cache,
                    settings,
                    "bshd",
                )
                loss = outputs["distillation_losses"][valid].mean()
                assert torch.isfinite(loss)
                centers.append(outputs["mirror_student_center"][valid].clone())
                loss.backward()
                assert all(
                    p.grad is None or torch.isfinite(p.grad).all()
                    for p in student.parameters()
                )
                optimizer.step()
                assert all(
                    torch.equal(value, frozen_cache[key])
                    for key, value in cache.items()
                )
            torch.testing.assert_close(
                centers[0], torch.zeros_like(centers[0]), atol=2e-7, rtol=0
            )
            assert (centers[-1] - centers[0]).abs().max() > 1e-8
            assert all(
                torch.equal(p, saved)
                for p, saved in zip(anchor.parameters(), frozen_params)
            )
            assert any(
                not torch.equal(p, saved)
                for p, saved in zip(student.parameters(), frozen_params)
            )
        print(
            f"PASS upstream {method}: 3 outer x 4 inner, frozen caches, recomputed centers, hard sync"
        )


if __name__ == "__main__":
    main()
