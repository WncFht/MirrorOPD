"""Stable normalized power reward shared by every production algorithm."""

import math

import torch


def power_reward(log_ratio, signal):
    value = log_ratio.double() if log_ratio.dtype == torch.float64 else log_ratio.float()
    if signal.kappa == 0 or signal.eta == 1:
        return value
    scaled = signal.kappa * value
    local = torch.log1p(signal.eta * torch.expm1(scaled.clamp(-0.5, 0.5)))
    global_value = torch.logaddexp(
        torch.full_like(scaled, math.log1p(-signal.eta)), math.log(signal.eta) + scaled
    )
    return torch.where(scaled.abs() <= 0.5, local, global_value) / (signal.kappa * signal.eta)
