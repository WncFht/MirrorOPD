"""Three native Top-J comparisons through frozen caches and masked FSDP loss."""

from types import SimpleNamespace

import torch

from mirroropd.integrations.verl.fsdp import compute_mirror_opd_from_cache
from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.metrics import finalize_token_moments, token_moments
from mirroropd.training.support import topk_pair_overlap
from mirroropd.training.tracking import metric_name


def main():
    """Check pair semantics, unequal response masks, ties and support boundaries."""
    left, right = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[3, 4, 5, 6]])
    expected = topk_pair_overlap(left, right, "student_teacher")
    assert expected["mirror_overlap_student_teacher_intersection_count"].item() == 2
    assert expected["mirror_overlap_student_teacher_union_count"].item() == 6
    assert expected["mirror_overlap_student_teacher_intersection_fraction"].item() == 0.5
    torch.testing.assert_close(
        expected["mirror_overlap_student_teacher_jaccard"], torch.tensor([1 / 3])
    )
    for key, value in topk_pair_overlap(left.flip(-1), right.flip(-1), "student_teacher").items():
        torch.testing.assert_close(value, expected[key])
    # One identical and one disjoint prefix: mean Jaccard=.5, pooled=1/3.
    pair = topk_pair_overlap(
        left.repeat(2, 1), torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), "student_teacher"
    )
    raw = token_moments(pair, torch.ones(2, dtype=torch.bool))
    pooled = finalize_token_moments({key: value for key, (_, value) in raw.items()})
    assert pooled["mirror/overlap_student_teacher_jaccard/mean"] == 0.5
    assert abs(pooled["mirror/overlap_student_teacher_pooled_jaccard"] - 1 / 3) < 1e-12

    torch.manual_seed(73)
    anchor, teacher, student = (torch.randn(2, 5, 6) for _ in range(3))
    valid = torch.tensor([[False, True, True, False, False], [True, True, True, True, False]])
    actions = torch.randint(0, 6, valid.shape)
    config = SimpleNamespace(
        distillation_loss=SimpleNamespace(loss_mode="mirror_mdpo", mirror_eta=0.3)
    )
    cache = build_mirror_opd_cache_from_full_logits(anchor, teacher, actions, valid, 3)
    frozen = {key: value.clone() for key, value in cache.items()}
    assert not cache["mirror_teacher_topk_ids"][~valid].any()
    for current in (student, student + torch.randn_like(student)):
        logits = current.clone().requires_grad_()
        output = compute_mirror_opd_from_cache(logits, cache, config, "bshd")
        for pair, a, b in (
            ("student_teacher", logits.detach().topk(3, -1).indices, teacher.topk(3, -1).indices),
            ("student_anchor", logits.detach().topk(3, -1).indices, anchor.topk(3, -1).indices),
            ("anchor_teacher", anchor.topk(3, -1).indices, teacher.topk(3, -1).indices),
        ):
            wanted = topk_pair_overlap(a, b, pair)
            for key, value in wanted.items():
                torch.testing.assert_close(output[key][valid], value[valid])
                assert not output[key][~valid].any()
                assert not output[key].requires_grad
                assert metric_name(
                    key.replace("mirror_", "mirror/", 1) + "/mean", "inner"
                ).startswith("06_support_mass/")
        torch.testing.assert_close(
            output["mirror_support_overlap"],
            output["mirror_overlap_student_anchor_intersection_fraction"],
        )
        loss = output["distillation_losses"][valid].mean()
        gradient = torch.autograd.grad(loss, logits)[0]
        legacy_cache = {
            key: value for key, value in cache.items() if key != "mirror_teacher_topk_ids"
        }
        legacy = compute_mirror_opd_from_cache(logits, legacy_cache, config, "bshd")
        assert "mirror_overlap_student_teacher_jaccard" not in legacy
        torch.testing.assert_close(output["distillation_losses"], legacy["distillation_losses"])
        torch.testing.assert_close(
            gradient, torch.autograd.grad(legacy["distillation_losses"][valid].mean(), logits)[0]
        )
        assert all(torch.equal(value, frozen[key]) for key, value in cache.items())
        # Pool sufficient statistics across unequal masks, not rank means.
        moments = {}
        for rank in range(2):
            for key, (reduction, value) in token_moments(
                {k: v[rank] for k, v in output.items()}, valid[rank]
            ).items():
                moments[key] = (
                    moments.get(key, 0) + value
                    if reduction == "sum"
                    else (
                        min(moments.get(key, value), value)
                        if reduction == "min"
                        else max(moments.get(key, value), value)
                    )
                )
        final = finalize_token_moments(moments)
        key = "mirror_overlap_student_teacher_intersection_fraction"
        assert final["mirror/valid_tokens"] == 6
        assert (
            abs(
                final["mirror/overlap_student_teacher_intersection_fraction/mean"]
                - output[key][valid].double().mean().item()
            )
            < 1e-12
        )

    # Native Top-J tie membership follows torch.topk, while pair comparisons
    # remain invariant to ordering. Identical tied distributions match fully.
    tied = torch.zeros(2, 5, 6)
    for j in (0, 1, 6):
        cache = build_mirror_opd_cache_from_full_logits(tied, tied, actions, valid, j)
        output = compute_mirror_opd_from_cache(tied.requires_grad_(), cache, config, "bshd")
        for pair in ("student_teacher", "student_anchor", "anchor_teacher"):
            assert torch.all(output[f"mirror_overlap_{pair}_intersection_count"][valid] == j)
            assert torch.all(output[f"mirror_overlap_{pair}_union_count"][valid] == j)
            for stat in ("intersection_fraction", "jaccard"):
                assert torch.all(output[f"mirror_overlap_{pair}_{stat}"][valid] == (1 if j else 0))
        assert torch.isfinite(output["distillation_losses"]).all()
    empty_valid = torch.zeros_like(valid)
    cache = build_mirror_opd_cache_from_full_logits(anchor, teacher, actions, empty_valid, 3)
    output = compute_mirror_opd_from_cache(student.requires_grad_(), cache, config, "bshd")
    assert all(not value.any() for value in output.values())
    assert token_moments(output, empty_valid) == {"mirror/raw/count": ("sum", 0.0)}
    print(
        "native Top-J pairs, frozen teacher cache, masks, gradient invariance, ties and J=0/V passed"
    )


if __name__ == "__main__":
    main()
