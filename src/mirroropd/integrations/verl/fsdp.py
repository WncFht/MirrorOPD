"""Prediction-position and nested-tensor adapters for verl FSDP workers.

The caller supplies the sequence-parallel world size; no Ray/CUDA/verl
imports are needed to test this cache contract.
"""

import torch

from mirroropd.training.cache import (
    ANCHOR_CACHE_FIELDS,
    TEACHER_CACHE_FIELDS,
    build_mirror_anchor_cache,
    complete_mirror_teacher_cache,
)
from mirroropd.training.losses import compute_mirror_opd_token_loss


def compute_mirror_cache_from_logits(
    student_logits,
    data,
    stage,
    topk=None,
    *,
    sequence_parallel_world_size=1,
    center_mode="topj_tail",
    kappa=0.0,
    eta=0.1,
):
    """Compact cache scoring for a no-padding, full-vocabulary inference pass.

    The engine supplies temperature-scaled logits of shape (total_tokens, V).
    Cache fields remain prediction-position aligned; they are not rolled here.
    """
    if sequence_parallel_world_size != 1:
        raise NotImplementedError("MirrorOPD cache scoring requires sequence parallel size 1")
    if student_logits.ndim != 2:
        raise ValueError("cache inference expects (total_tokens, vocab) logits")

    def values(key):
        tensor = data[key]
        return tensor.values() if tensor.is_nested else tensor

    if stage == "anchor":
        if topk is None:
            raise ValueError("anchor cache inference requires mirror_cache_topk")
        return build_mirror_anchor_cache(
            student_logits,
            values("mirror_actions"),
            values("mirror_valid"),
            topk,
            center_mode=center_mode,
        )
    if stage == "teacher":
        cache = {key: values(key) for key in ANCHOR_CACHE_FIELDS if key in data}
        if (center_mode == "full_vocab") != ("mirror_full_anchor_log_probs" in cache):
            raise ValueError("teacher center mode must match the frozen anchor cache")
        completed = complete_mirror_teacher_cache(student_logits, cache, kappa=kappa, eta=eta)
        return {key: completed[key] for key in TEACHER_CACHE_FIELDS if key in completed}
    raise ValueError(f"unknown mirror_cache_stage: {stage}")


def compute_mirror_opd_from_cache(
    student_logits, data, config, data_format, *, sequence_parallel_world_size=1
):
    """Consume prediction-position-aligned caches after engine unpadding.

    Cache producers must put response token t at its predecessor position.
    Prompt and terminal positions have mirror_valid=False and are excluded
    before any probability arithmetic. Sequence parallel is not enabled yet.
    """
    if sequence_parallel_world_size != 1:
        raise NotImplementedError("MirrorOPD cache adapter requires sequence parallel size 1")
    if data_format not in ("thd", "bshd"):
        raise ValueError("unsupported MirrorOPD data format")
    fields = (
        "mirror_valid",
        "mirror_actions",
        "mirror_anchor_ids",
        "mirror_anchor_head_log_probs",
        "mirror_anchor_tail_log_prob",
        "mirror_anchor_action_log_prob",
        "mirror_teacher_action_log_prob",
        "mirror_teacher_center",
    )
    cache = {}
    for key in fields:
        value = data[key]
        value = value.values().unsqueeze(0) if value.is_nested else value
        if value.shape[:2] != student_logits.shape[:2]:
            raise ValueError(f"{key} must align with full prediction positions")
        cache[key] = value
    valid = cache["mirror_valid"].bool()
    for key in (
        "mirror_teacher_head_log_probs",
        "mirror_teacher_tail_log_prob",
        "mirror_teacher_topk_ids",
        "mirror_full_anchor_log_probs",
        "mirror_full_teacher_log_probs",
        "mirror_signal_kappa",
        "mirror_signal_eta",
    ):
        if key in data:
            value = data[key]
            cache[key] = value.values().unsqueeze(0) if value.is_nested else value
    overlap_pairs = ("student_anchor",)
    if "mirror_teacher_topk_ids" in cache:
        overlap_pairs += ("student_teacher", "anchor_teacher")
    action_mode = getattr(config.distillation_loss, "mirror_action_mode", "sampled")
    result = {
        key: student_logits.new_zeros(
            valid.shape,
            dtype=torch.float64 if student_logits.dtype == torch.float64 else torch.float32,
        )
        for key in (
            "distillation_losses",
            "mirror_student_center",
            "mirror_teacher_center",
            "mirror_importance_ratio",
            "mirror_score_coefficient",
            "mirror_token_advantage",
            "mirror_full_loss",
            "mirror_support_overlap",
            "mirror_action_nll",
            "mirror_sampled_reward",
            "mirror_sampled_log_ratio",
            "mirror_student_entropy",
            "mirror_anchor_head_mass",
            "mirror_student_head_mass",
            "mirror_anchor_tail_mass",
            "mirror_student_tail_mass",
            *(
                f"mirror_overlap_{pair}_{stat}"
                for pair in overlap_pairs
                for stat in (
                    "intersection_count",
                    "union_count",
                    "intersection_fraction",
                    "jaccard",
                )
            ),
        )
    }
    for key in (
        ("mirror_importance_ratio", "mirror_score_coefficient", "mirror_token_advantage")
        if action_mode == "full_vocab"
        else ("mirror_full_loss",)
    ):
        del result[key]
    if valid.any():
        settings = config.distillation_loss
        kappa = getattr(settings, "mirror_kappa", 0.0)
        center_mode = getattr(settings, "mirror_center_mode", "topj_tail")
        action_mode = getattr(settings, "mirror_action_mode", "sampled")
        if (center_mode == "full_vocab") != ("mirror_full_anchor_log_probs" in cache):
            raise ValueError("inner center mode must match the frozen anchor cache")
        if "mirror_signal_kappa" not in cache and kappa != 0:
            raise ValueError("power-shaped inner loss requires a matching teacher cache")
        for key, target in (
            ("mirror_signal_kappa", kappa),
            ("mirror_signal_eta", settings.mirror_eta),
        ):
            if key in cache and (key == "mirror_signal_kappa" or kappa != 0):
                observed = cache[key][valid]
                if not torch.equal(observed, torch.full_like(observed, target)):
                    raise ValueError("inner kappa/eta must match the frozen teacher signal")
        outputs = compute_mirror_opd_token_loss(
            student_logits=student_logits[valid],
            actions=cache["mirror_actions"][valid],
            anchor_ids=cache["mirror_anchor_ids"][valid],
            anchor_head_log_probs=cache["mirror_anchor_head_log_probs"][valid],
            anchor_tail_log_prob=cache["mirror_anchor_tail_log_prob"][valid],
            anchor_action_log_prob=cache["mirror_anchor_action_log_prob"][valid],
            teacher_action_log_prob=cache["mirror_teacher_action_log_prob"][valid],
            teacher_center=cache["mirror_teacher_center"][valid],
            method=config.distillation_loss.loss_mode.removeprefix("mirror_"),
            eta=config.distillation_loss.mirror_eta,
            kappa=kappa,
            action_mode=action_mode,
            full_anchor_log_probs=cache["mirror_full_anchor_log_probs"][valid]
            if "mirror_full_anchor_log_probs" in cache
            else None,
            full_teacher_log_probs=cache["mirror_full_teacher_log_probs"][valid]
            if "mirror_full_teacher_log_probs" in cache
            else None,
            teacher_head_log_probs=cache["mirror_teacher_head_log_probs"][valid]
            if "mirror_teacher_head_log_probs" in cache
            else None,
            teacher_topk_ids=cache["mirror_teacher_topk_ids"][valid]
            if "mirror_teacher_topk_ids" in cache
            else None,
            teacher_tail_log_prob=cache["mirror_teacher_tail_log_prob"][valid]
            if "mirror_teacher_tail_log_prob" in cache
            else None,
        )
        for key, value in outputs.items():
            result[key][valid] = value.to(result[key].dtype)
    else:
        result["distillation_losses"] = student_logits.sum(-1) * 0
    return result
