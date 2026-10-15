"""Layout adaptation only; algorithms and cache validation live in the core."""

import torch

from mirroropd.core.cache import (
    ANCHOR_FIELDS,
    TEACHER_FIELDS,
    WIRE_PREFIX,
    AnchorCache,
    FrozenCache,
    build_anchor,
    score_teacher,
)
from mirroropd.core.objectives import compute_loss

DIAGNOSTIC_PREFIX = "mirror_metric_"


def cache_outputs(logits, data, stage, context, *, diagnostics, sequence_parallel_size):
    if sequence_parallel_size != 1:
        raise NotImplementedError("Cache inference requires an unsharded vocabulary and SP=1")
    if logits.ndim != 2:
        raise ValueError("Cache inference requires flat prediction-position logits")

    def value(name):
        tensor = data[WIRE_PREFIX + name]
        return tensor.values() if tensor.is_nested else tensor

    if stage == "anchor":
        return build_anchor(logits, value("actions"), value("valid"), context).to_wire()
    if stage == "teacher":
        fields = {
            WIRE_PREFIX + name: value(name) for name in ANCHOR_FIELDS if WIRE_PREFIX + name in data
        }
        result = score_teacher(
            logits, AnchorCache.from_wire(fields, context), diagnostics=diagnostics
        ).to_wire()
        return {
            WIRE_PREFIX + name: result[WIRE_PREFIX + name]
            for name in TEACHER_FIELDS
            if WIRE_PREFIX + name in result
        }
    raise ValueError("Cache stage must be anchor or teacher")


def loss_outputs(
    logits, data, objective, context, *, anchor_version, data_format, sequence_parallel_size
):
    if sequence_parallel_size != 1:
        raise NotImplementedError("Loss adaptation requires an unsharded vocabulary and SP=1")
    if data_format not in {"thd", "bshd"} or logits.ndim != 3:
        raise ValueError("Loss adaptation requires thd/bshd prediction-position logits")
    fields = {}
    for name in (*ANCHOR_FIELDS, *TEACHER_FIELDS):
        key = WIRE_PREFIX + name
        if key in data:
            value = data[key]
            fields[key] = value.values().unsqueeze(0) if value.is_nested else value
    cache = FrozenCache.from_wire(fields, context)
    result = compute_loss(logits, cache, objective, anchor_version=anchor_version)
    return {
        "distillation_losses": result.loss,
        **{DIAGNOSTIC_PREFIX + name: value for name, value in result.diagnostics.items()},
    }


def prediction_fields(input_ids, attention_mask, response_mask):
    attention = attention_mask.bool()
    response = response_mask.bool()
    if input_ids.ndim != 2 or input_ids.shape != attention.shape or response.ndim != 2:
        raise ValueError("Prediction fields require padded two-dimensional IDs and masks")
    if response.shape[0] != input_ids.shape[0] or response.shape[1] >= input_ids.shape[1]:
        raise ValueError("Every response needs a nonempty preceding prompt")
    actions_mask = torch.zeros_like(attention)
    actions_mask[:, -response.shape[1] :] = response
    valid = torch.zeros_like(attention)
    valid[:, :-1] = actions_mask[:, 1:] & attention[:, :-1] & attention[:, 1:]
    if valid.sum() != response.sum():
        raise ValueError("Response mask contains actions without valid attended predictors")
    actions = torch.where(valid, input_ids.roll(-1, 1), 0)
    return {WIRE_PREFIX + "valid": valid, WIRE_PREFIX + "actions": actions}


def restore_predictions(tensor, attention_shape, indices):
    values = tensor.values() if tensor.is_nested else tensor
    if indices is None or values.shape[0] != indices.numel():
        raise ValueError("Prediction restoration requires matching unpadding indices")
    count = 1
    for dimension in attention_shape:
        count *= dimension
    result = values.new_zeros((count, *values.shape[1:]))
    result.index_copy_(0, indices.to(values.device), values)
    return result.reshape(*attention_shape, *values.shape[1:])


def unpad_cache_fields(data, indices, offsets):
    """First-party cache tensors carry all trailing head/vocabulary dimensions."""
    output = {}
    for name in (*ANCHOR_FIELDS, *TEACHER_FIELDS):
        key = WIRE_PREFIX + name
        if key not in data:
            continue
        tensor = data[key]
        if tensor.is_nested:
            raise ValueError("Cache fields were already unpadded")
        flattened = tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])
        values = flattened.index_select(0, indices.to(tensor.device))
        output[key] = torch.nested.nested_tensor_from_jagged(values, offsets.to(tensor.device))
    return output
