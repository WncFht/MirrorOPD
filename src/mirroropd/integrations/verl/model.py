"""Keep the sampled policy and gradient policy deterministic at equal weights."""

import torch


def prepare_policy_model(model):
    changed = []
    for name, module in model.named_modules():
        if (
            isinstance(
                module,
                (
                    torch.nn.Dropout,
                    torch.nn.Dropout1d,
                    torch.nn.Dropout2d,
                    torch.nn.Dropout3d,
                    torch.nn.AlphaDropout,
                    torch.nn.FeatureAlphaDropout,
                ),
            )
            and module.p != 0
        ):
            changed.append({"module": name, "field": "p", "previous": module.p})
            module.p = 0.0
        for owner in (module, getattr(module, "config", None)):
            if owner is None or not hasattr(owner, "__dict__"):
                continue
            for key, value in list(vars(owner).items()):
                if type(value) not in {int, float} or not 0 < value <= 1:
                    continue
                if "dropout" in key.lower() or key.endswith(("_pdrop", "layerdrop", "drop_prob")):
                    changed.append({"module": name, "field": key, "previous": value})
                    setattr(owner, key, 0.0)
    return changed
