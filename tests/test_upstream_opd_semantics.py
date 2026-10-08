"""Execute upstream pure tensor functions without loading GPU/Ray services.

AST extraction retains original function bodies. It is a mathematical audit,
not a substitute for importing and testing the complete production worker.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import torch


def load_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in found} == set(names)
    exec(compile(ast.Module(body=found, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    root = Path(__file__).resolve().parents[1] / "upstream"
    ns = load_functions(
        root / "One-Shot-OPD/verl/verl/workers/fsdp_teacher_worker.py",
        ["compute_reward_weights", "compute_top_k_reward"],
        {"torch": torch},
    )
    p = torch.tensor([[0.6, 0.3]], dtype=torch.float64)
    q = torch.tensor([[0.3, 0.2]], dtype=torch.float64)
    overlap = torch.ones_like(p, dtype=torch.bool)
    reward = ns["compute_top_k_reward"](
        p.log(), q.log(), overlap, overlap, "only_stu", "student_p"
    ).sum()
    norm = ((p / p.sum()) * ((p / p.sum()).log() - (q / q.sum()).log())).sum()
    # Reward uses normalized weights but raw full-vocabulary log-ratios.
    torch.testing.assert_close(reward, -norm - torch.log(p.sum() / q.sum()))
    assert not torch.isclose(reward, -norm)
    print(f"One-Shot sum reward={reward.item():.9f}; negative conditional KL={-norm.item():.9f}")
    ns = load_functions(
        root / "TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py",
        [
            "log1mexp",
            "sample_corrected_ta_opd",
            "compute_mirror_opd_token_loss",
            "build_mirror_opd_cache_from_full_logits",
            "build_mirror_anchor_cache",
            "complete_mirror_teacher_cache",
            "compute_mirror_cache_from_logits",
        ],
        {"torch": torch, "_LOG2": math.log(2)},
    )
    tails = torch.tensor([1e-3, 1e-6, 1e-8, 1e-12], dtype=torch.float64)
    actual = ns["log1mexp"](torch.log1p(-tails)).exp()
    print("TA-OPD true tails:", tails.tolist())
    print("TA-OPD returned tails:", actual.tolist())
    torch.testing.assert_close(actual[0], tails[0])
    assert (actual[2:] > 99 * tails[2:]).all()
    # Enumerate reused behavior actions: correction must have correct value
    # AND gradient, even when behavior differs substantially from student.
    logits = torch.tensor([0.5, -0.1, -0.7, 0.2], dtype=torch.float64, requires_grad=True)
    lp = logits.log_softmax(-1)
    p = lp.exp()
    q = torch.tensor([0.2, 0.4, 0.1, 0.3], dtype=torch.float64)
    behavior = torch.tensor([0.5, 0.1, 0.3, 0.1], dtype=torch.float64)
    inside = torch.tensor([False, True, False, True])
    pt = p[~inside].sum()
    qt = q[~inside].sum()
    pa = torch.cat([p[inside], pt[None]])
    qa = torch.cat([q[inside], qt[None]])
    ta = (pa * (pa.log() - qa.log()).detach()).sum()
    corrected = ns["sample_corrected_ta_opd"](
        ta.expand(4),
        lp,
        q.log(),
        pt.log().expand(4),
        qt.log().expand(4),
        inside,
        behavior.log(),
    )
    expected = (behavior * corrected).sum()
    full = (p * (lp - q.log())).sum()
    # Upstream casts to float32, so use tolerances reflecting that operation.
    torch.testing.assert_close(expected, full, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(
        torch.autograd.grad(expected, logits, retain_graph=True)[0],
        torch.autograd.grad(full, logits)[0],
        atol=2e-7,
        rtol=2e-6,
    )
    print("PASS: upstream sample-corrected TA value/gradient under reused behavior")
    from mirroropd.token_inner import bpo_loss, mdpo_loss, pmd_mean_loss

    for method, original in [
        ("pmd_mean", pmd_mean_loss),
        ("bpo", bpo_loss),
        ("mdpo", mdpo_loss),
    ]:
        for k in [0, 2, 4]:
            x = logits.detach().clone().requires_grad_(True)
            anchor = behavior
            ids = anchor.topk(k).indices
            outside = torch.ones(4, dtype=torch.bool)
            outside[ids] = False
            qt = anchor[outside].sum()
            tail_lp = qt.log()
            kt = (anchor * (anchor.log() - q.log())).sum()
            fields = ns["compute_mirror_opd_token_loss"](
                x.expand(4, -1),
                torch.arange(4),
                ids.expand(4, -1),
                anchor[ids].log().expand(4, -1),
                tail_lp.expand(4),
                anchor.log(),
                q.log(),
                kt.expand(4),
                method,
                0.5,
            )
            estimated = (anchor * fields["distillation_losses"]).sum()
            gradient = torch.autograd.grad(estimated, x, retain_graph=True)[0]
            assert torch.isfinite(gradient).all()
            assert not fields["mirror_student_center"].requires_grad
            if k == 4:
                expected_loss = original(x, anchor, q.log() - anchor.log(), 0.5)
                torch.testing.assert_close(
                    gradient,
                    torch.autograd.grad(expected_loss, x)[0],
                    atol=3e-7,
                    rtol=3e-6,
                )
    print("PASS: TA-OPD fork MirrorOPD primitive: 9 support/method cases, exact gradient endpoints")
    load_functions(
        root / "TA-OPD/verl/verl/trainer/distillation/fsdp/losses.py",
        ["compute_mirror_opd_from_cache"],
        ns,
    )
    ns["get_ulysses_sequence_parallel_world_size"] = lambda: 1
    offsets = torch.tensor([0, 4, 7])
    valid = torch.tensor([False, True, True, False, False, True, False])
    anchor = torch.tensor([0.4, 0.3, 0.2, 0.1])
    teacher = torch.tensor([0.1, 0.2, 0.3, 0.4])
    kt = (anchor * (anchor.log() - teacher.log())).sum()
    cache = {
        "mirror_valid": valid,
        "mirror_actions": torch.tensor([-99, 0, 1, -99, -99, 2, -99]),
        "mirror_anchor_ids": torch.tensor([[0, 1]]).expand(7, -1).clone(),
        "mirror_anchor_head_log_probs": anchor[:2].log().expand(7, -1).clone(),
        "mirror_anchor_tail_log_prob": torch.full((7,), 0.3).log(),
        "mirror_anchor_action_log_prob": anchor[torch.tensor([0, 0, 1, 0, 0, 2, 0])].log(),
        "mirror_teacher_action_log_prob": teacher[torch.tensor([0, 0, 1, 0, 0, 2, 0])].log(),
        "mirror_teacher_center": kt.expand(7).clone(),
    }
    # Invalid positions deliberately contain unusable ids and NaN caches.
    cache["mirror_anchor_ids"][~valid] = -99
    for value in cache.values():
        if value.is_floating_point():
            value[~valid] = float("nan")
    nested = {
        key: torch.nested.nested_tensor_from_jagged(value, offsets) for key, value in cache.items()
    }
    for method in ("pmd_mean", "bpo", "mdpo"):
        x = torch.randn(1, 7, 4, requires_grad=True)
        cfg = SimpleNamespace(
            distillation_loss=SimpleNamespace(loss_mode="mirror_" + method, mirror_eta=0.5)
        )
        fields = ns["compute_mirror_opd_from_cache"](x, nested, cfg, "thd")
        direct = ns["compute_mirror_opd_token_loss"](
            x[0, valid],
            cache["mirror_actions"][valid],
            cache["mirror_anchor_ids"][valid],
            cache["mirror_anchor_head_log_probs"][valid],
            cache["mirror_anchor_tail_log_prob"][valid],
            cache["mirror_anchor_action_log_prob"][valid],
            cache["mirror_teacher_action_log_prob"][valid],
            cache["mirror_teacher_center"][valid],
            method,
            0.5,
        )
        for key, values in fields.items():
            assert torch.isfinite(values).all()
            assert (values[0, ~valid] == 0).all()
            torch.testing.assert_close(values[0, valid], direct[key])
        grad = torch.autograd.grad(fields["distillation_losses"].sum(), x)[0]
        assert (grad[0, ~valid] == 0).all()
        assert grad[0, valid].abs().sum() > 0
    print("PASS: jagged MirrorOPD adapter: 3 methods, invalid-position masking and live gradients")
    # Partial-support centers change BPO's score update and its regression
    # gradient in different ways; check both derived error identities.
    q = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float64)
    teacher = torch.tensor([0.05, 0.15, 0.3, 0.5], dtype=torch.float64)
    x = torch.tensor([-0.3, 0.2, 0.7, -0.1], dtype=torch.float64, requires_grad=True)
    lp = x.log_softmax(-1)
    eta = 0.3
    reward = teacher.log() - q.log()
    kt = (q * (q.log() - teacher.log())).sum()
    ks = (q * (q.log() - lp)).sum()
    head = slice(0, 2)
    qt = q[2:].sum()
    khat_t = (q[head] * (q[head].log() - teacher[head].log())).sum() + qt * (
        qt.log() - teacher[2:].sum().log()
    )
    khat_s = (q[head] * (q[head].log() - lp[head])).sum() + qt * (qt.log() - lp[2:].logsumexp(-1))
    d = lp - q.log() + ks - eta * (reward + kt)
    dhat = lp - q.log() + khat_s - eta * (reward + khat_t)
    exact = (q * d.square()).sum() / (2 * eta)
    surrogate = (q * (dhat / eta).detach() * lp).sum()
    regression = (q * dhat.square()).sum() / (2 * eta)
    ge = torch.autograd.grad(exact, x, retain_graph=True)[0]
    gs = torch.autograd.grad(surrogate, x, retain_graph=True)[0]
    gr = torch.autograd.grad(regression, x, retain_graph=True)[0]
    score_mean = torch.autograd.grad((q * lp).sum(), x, retain_graph=True)[0]
    center_gradient = torch.autograd.grad(khat_s, x, retain_graph=True)[0]
    torch.testing.assert_close(gs - ge, ((kt - khat_t) - (ks - khat_s) / eta) * score_mean)
    torch.testing.assert_close(gr - gs, (q * dhat).sum() / eta * center_gradient)
    assert (gr - gs).abs().max() > 1e-3
    mean_regression = 0.5 * (q * (lp - q.log() - eta * (reward + khat_t)).square()).sum()
    mean_surrogate = (q * (lp - q.log() - eta * (reward + khat_t)).detach() * lp).sum()
    torch.testing.assert_close(
        torch.autograd.grad(mean_regression, x, retain_graph=True)[0],
        torch.autograd.grad(mean_surrogate, x)[0],
    )
    print("PASS: Tail-aware BPO bias and regression-gradient discrepancy; Mean equivalence")
    anchor_logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.1, 0.2, 0.3, 0.4]])
    teacher_logits = torch.tensor([[0.1, 0.7, 1.2, -0.4], [0.4, 0.3, 0.2, 0.1]])
    actions = torch.tensor([2, -99])
    valid = torch.tensor([True, False])
    cache = ns["build_mirror_opd_cache_from_full_logits"](
        anchor_logits, teacher_logits, actions, valid, topk=2
    )
    anchor_cache = ns["build_mirror_anchor_cache"](anchor_logits, actions, valid, topk=2)
    saved_anchor = {key: value.clone() for key, value in anchor_cache.items()}
    compact = ns["complete_mirror_teacher_cache"](teacher_logits, anchor_cache)
    assert compact.keys() == cache.keys()
    for key in cache:
        torch.testing.assert_close(compact[key], cache[key])
    assert all(torch.equal(value, saved_anchor[key]) for key, value in anchor_cache.items())
    try:
        ns["complete_mirror_teacher_cache"](torch.zeros(2, 5), anchor_cache)
    except ValueError as error:
        if "vocabulary sizes" not in str(error):
            raise AssertionError("unexpected vocabulary validation error") from error
    else:
        raise AssertionError("mismatched teacher vocabulary was accepted")
    q = anchor_logits.log_softmax(-1).exp()
    t = teacher_logits.log_softmax(-1).exp()
    ids = cache["mirror_anchor_ids"][0]
    outside = torch.ones(4, dtype=torch.bool)
    outside[ids] = False
    ta = (q[0, ids] * (q[0, ids].log() - t[0, ids].log())).sum()
    ta += q[0, outside].sum() * (q[0, outside].sum().log() - t[0, outside].sum().log())
    torch.testing.assert_close(cache["mirror_teacher_center"][0], ta)
    torch.testing.assert_close(
        cache["mirror_anchor_action_log_prob"][0], anchor_logits[0].log_softmax(-1)[2]
    )
    assert (cache["mirror_anchor_ids"][1] == 0).all()
    assert cache["mirror_teacher_center"].requires_grad is False
    cache_full = ns["build_mirror_opd_cache_from_full_logits"](
        anchor_logits[:1], teacher_logits[:1], actions[:1], valid[:1], topk=4
    )
    assert torch.isneginf(cache_full["mirror_anchor_tail_log_prob"]).all()
    torch.testing.assert_close(
        cache_full["mirror_teacher_center"][0], (q[0] * (q[0].log() - t[0].log())).sum()
    )
    cache_empty = ns["build_mirror_opd_cache_from_full_logits"](
        anchor_logits[:1], teacher_logits[:1], actions[:1], valid[:1], topk=0
    )
    torch.testing.assert_close(
        cache_empty["mirror_teacher_center"], torch.zeros(1), atol=2e-7, rtol=0
    )
    concentrated = torch.tensor([[0.0, -40.0, -41.0, -42.0]])
    tiny_tail = ns["build_mirror_opd_cache_from_full_logits"](
        concentrated, concentrated, actions[:1], valid[:1], topk=1
    )
    torch.testing.assert_close(
        tiny_tail["mirror_anchor_tail_log_prob"],
        concentrated.log_softmax(-1)[:, 1:].logsumexp(-1),
    )
    assert tiny_tail["mirror_anchor_tail_log_prob"].exp().item() < 1e-16
    anchor_logits[1] = float("nan")
    teacher_logits[1] = float("nan")
    invalid = ns["build_mirror_opd_cache_from_full_logits"](
        anchor_logits, teacher_logits, actions, valid, topk=2
    )
    assert all(torch.isfinite(value).all() for value in invalid.values())
    print(
        "PASS: full-logit cache producer: anchor support, Tail-aware teacher center, full-tail case"
    )
    print("PASS: upstream normalized-weight identity and small-tail floor verified")
    padding = load_functions(
        root / "TA-OPD/verl/verl/workers/utils/padding.py",
        [
            "build_mirror_prediction_fields",
            "no_padding_2_full_padding",
            "no_padding_2_padding",
        ],
        {
            "torch": torch,
            "TensorDict": dict,
            "F": torch.nn.functional,
            "tu": SimpleNamespace(
                get_non_tensor_data=lambda data, key, default=None: data.get(key, default)
            ),
        },
    )
    ids = torch.tensor([[0, 1, 2, 3, 4, 0, 0], [5, 6, 7, 8, 9, 10, 11]])
    attention = torch.tensor([[0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]])
    response_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    indices = attention.flatten().nonzero().flatten()
    data = {
        "input_ids": ids,
        "attention_mask": attention,
        "response_mask": response_mask,
        "prompts": ids[:, :3],
        "responses": ids[:, 3:],
        "indices": indices,
    }
    prediction = padding["build_mirror_prediction_fields"](data)
    assert prediction["mirror_valid"].sum().item() == 6
    lengths = attention.sum(-1)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])
    nested = {
        key: torch.nested.nested_tensor_from_jagged(value.flatten()[indices], offsets)
        for key, value in prediction.items()
    }
    torch.manual_seed(37)
    qlogits = torch.randn(len(indices), 16)
    tlogits = torch.randn(len(indices), 16)
    anchor = ns["compute_mirror_cache_from_logits"](qlogits, nested, "anchor", topk=3)
    nested_anchor = {
        key: torch.nested.nested_tensor_from_jagged(value, offsets) for key, value in anchor.items()
    }
    teacher = ns["compute_mirror_cache_from_logits"](tlogits, nested_anchor, "teacher")
    oracle = ns["build_mirror_opd_cache_from_full_logits"](
        qlogits,
        tlogits,
        nested["mirror_actions"].values(),
        nested["mirror_valid"].values(),
        3,
    )
    for key, value in teacher.items():
        torch.testing.assert_close(value, oracle[key])
    for value in {**anchor, **teacher}.values():
        restored = padding["no_padding_2_full_padding"](value, data)
        torch.testing.assert_close(restored.flatten(0, 1)[indices], value)
        assert restored[~attention.bool()].eq(0).all()
    restored_actions = padding["no_padding_2_padding"](anchor["mirror_actions"], data)
    torch.testing.assert_close(restored_actions, torch.where(response_mask.bool(), ids[:, 3:], 0))
    print(
        "PASS: worker cache inference and padding: ragged prompts/responses, predecessor alignment, vector fields"
    )


if __name__ == "__main__":
    main()
