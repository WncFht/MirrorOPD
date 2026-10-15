"""Shared validation for mirror target shape, cache and action integration."""

import math


def validate_mirror_configuration(eta, kappa, center_mode, action_mode, method):
    if math.isnan(eta) or eta <= 0:
        raise ValueError("mirror_eta must be positive")
    if not math.isfinite(kappa) or not 0 <= kappa <= 1:
        raise ValueError("mirror_kappa must be finite and in [0,1]")
    if kappa > 0 and eta > 1:
        raise ValueError("positive power shape requires eta at most one")
    if center_mode not in ("topj_tail", "full_vocab"):
        raise ValueError("mirror_center_mode must be topj_tail or full_vocab")
    if action_mode not in ("sampled", "full_vocab"):
        raise ValueError("mirror_action_mode must be sampled or full_vocab")
    if action_mode == "full_vocab" and center_mode != "full_vocab":
        raise ValueError("full-vocabulary actions require full-vocabulary caches")
    if method not in ("bpo", "mdpo", "pmd_mean", "opd_onpolicy", "opd_is", "opd_is_no_baseline"):
        raise ValueError("unknown mirror loss method")
    if method.startswith("opd_") and action_mode != "sampled":
        raise ValueError("OPD baselines require sampled actions")
    if math.isinf(eta) and method not in ("mdpo", "opd_onpolicy", "opd_is", "opd_is_no_baseline"):
        raise ValueError("infinite eta is supported only for MDPO and OPD baselines")
