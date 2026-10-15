"""Typed frozen cache, shared across algorithms and explicit on the wire."""

from dataclasses import dataclass, fields

import torch

from mirroropd.core.config import CacheContext
from mirroropd.core.distributions import Distribution
from mirroropd.core.signal import power_reward

ANCHOR_FIELDS = (
    "valid",
    "actions",
    "head_ids",
    "anchor_head",
    "anchor_tail",
    "anchor_action",
    "anchor_full",
)
TEACHER_FIELDS = ("teacher_action", "reward_center", "teacher_ids", "target_full")
WIRE_PREFIX = "mirror_cache_"


def scatter_valid(value, valid, trailing=()):
    result = value.new_zeros((*valid.shape, *trailing))
    result[valid] = value
    return result


@dataclass(frozen=True, slots=True)
class AnchorCache:
    context: CacheContext
    valid: torch.Tensor
    actions: torch.Tensor
    head_ids: torch.Tensor
    anchor_head: torch.Tensor
    anchor_tail: torch.Tensor
    anchor_action: torch.Tensor
    anchor_full: torch.Tensor | None = None

    def __post_init__(self):
        shape = self.valid.shape
        if self.valid.dtype != torch.bool or self.actions.dtype != torch.long:
            raise TypeError("Cache valid/actions must be bool/int64 tensors")
        if self.head_ids.dtype != torch.long:
            raise TypeError("Cache head_ids must be int64")
        for name in ("actions", "anchor_tail", "anchor_action"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"Cache {name} does not align with prefixes")
        expected = (*shape, self.context.estimator.topj)
        if self.head_ids.shape != expected or self.anchor_head.shape != expected:
            raise ValueError("Cache head fields do not align with declared topj")
        full = self.context.estimator.center == "full"
        if full != (self.anchor_full is not None):
            raise ValueError("Cache full anchor does not match estimator")
        if full and self.anchor_full.shape != (*shape, self.context.vocab_size):
            raise ValueError("Full anchor vocabulary does not match context")
        self._validate_tensors()

    def _validate_tensors(self):
        devices = set()
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, torch.Tensor):
                if value.requires_grad:
                    raise ValueError(f"Frozen cache field {item.name} carries gradients")
                devices.add(value.device)
        if len(devices) != 1:
            raise ValueError("Cache tensors must share a device")

    def tensors(self):
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if isinstance(getattr(self, item.name), torch.Tensor)
        }

    def to_wire(self):
        return {WIRE_PREFIX + key: value for key, value in self.tensors().items()}

    @classmethod
    def from_wire(cls, data, context):
        values = {
            name: data[WIRE_PREFIX + name] for name in ANCHOR_FIELDS if WIRE_PREFIX + name in data
        }
        return cls(context=context, **values)


@dataclass(frozen=True, slots=True)
class FrozenCache(AnchorCache):
    teacher_action: torch.Tensor | None = None
    reward_center: torch.Tensor | None = None
    teacher_ids: torch.Tensor | None = None
    target_full: torch.Tensor | None = None

    def __post_init__(self):
        super(FrozenCache, self).__post_init__()
        for name in ("teacher_action", "reward_center"):
            value = getattr(self, name)
            if value is None or value.shape != self.valid.shape:
                raise ValueError(f"Frozen cache requires aligned {name}")
        if self.teacher_ids is not None and self.teacher_ids.shape != self.head_ids.shape:
            raise ValueError("Teacher support must align with anchor support")
        full = self.context.estimator.integration == "full"
        if full != (self.target_full is not None):
            raise ValueError("Cache target does not match action integration")
        if full and self.target_full.shape != self.anchor_full.shape:
            raise ValueError("Full target must align with anchor vocabulary")

    @classmethod
    def from_wire(cls, data, context):
        values = {
            name: data[WIRE_PREFIX + name]
            for name in (*ANCHOR_FIELDS, *TEACHER_FIELDS)
            if WIRE_PREFIX + name in data
        }
        return cls(context=context, **values)


@torch.no_grad()
def build_anchor(logits, actions, valid, context):
    if logits.shape != (*valid.shape, context.vocab_size) or actions.shape != valid.shape:
        raise ValueError("Anchor logits/actions must align with the declared context")
    if actions.dtype != torch.long or valid.dtype != torch.bool:
        raise TypeError("Anchor actions/valid must be int64/bool")
    if ((actions[valid] < 0) | (actions[valid] >= context.vocab_size)).any():
        raise ValueError("Anchor action is outside the legal vocabulary")
    distribution = Distribution.from_logits(logits[valid].detach())
    estimator = context.estimator
    ids = distribution.logits.topk(estimator.topj, -1).indices
    return AnchorCache(
        context=context,
        valid=valid.detach().clone(),
        actions=actions.detach().clone(),
        head_ids=scatter_valid(ids, valid, (estimator.topj,)),
        anchor_head=scatter_valid(distribution.selected(ids), valid, (estimator.topj,)),
        anchor_tail=scatter_valid(distribution.tail(ids, estimator.chunk_size), valid),
        anchor_action=scatter_valid(distribution.action(actions[valid]), valid),
        anchor_full=scatter_valid(distribution.full(), valid, (context.vocab_size,))
        if estimator.center == "full"
        else None,
    )


@torch.no_grad()
def score_teacher(logits, anchor, *, diagnostics="full"):
    context, valid = anchor.context, anchor.valid
    if logits.shape != (*valid.shape, context.vocab_size):
        raise ValueError("Teacher logits do not match the frozen anchor context")
    distribution = Distribution.from_logits(logits[valid].detach())
    ids = anchor.head_ids[valid]
    qhead, qtail = anchor.anchor_head[valid], anchor.anchor_tail[valid]
    center = -(qhead.exp() * power_reward(distribution.selected(ids) - qhead, context.signal)).sum(
        -1
    )
    ttail = distribution.tail(ids, context.estimator.chunk_size)
    nonempty = torch.isfinite(qtail)
    center[nonempty] -= qtail[nonempty].exp() * power_reward(
        ttail[nonempty] - qtail[nonempty], context.signal
    )
    target = None
    if context.estimator.center == "full":
        qlog = anchor.anchor_full[valid]
        reward = power_reward(distribution.full() - qlog, context.signal)
        center = -(qlog.exp() * reward).sum(-1)
        if context.estimator.integration == "full":
            raw = qlog + context.signal.eta * reward
            target = scatter_valid(
                raw - raw.logsumexp(-1, keepdim=True), valid, (context.vocab_size,)
            )
    values = anchor.tensors()
    return FrozenCache(
        context=context,
        **values,
        teacher_action=scatter_valid(distribution.action(anchor.actions[valid]), valid),
        reward_center=scatter_valid(center, valid),
        teacher_ids=scatter_valid(
            distribution.logits.topk(context.estimator.topj, -1).indices,
            valid,
            (context.estimator.topj,),
        )
        if diagnostics == "full"
        else None,
        target_full=target,
    )


def build_cache(anchor_logits, teacher_logits, actions, valid, context, *, diagnostics="full"):
    anchor = build_anchor(anchor_logits, actions, valid, context)
    return score_teacher(teacher_logits, anchor, diagnostics=diagnostics)
