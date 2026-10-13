"""Calibrate on the exact training prefixes; keep held-out responses out of tuning."""

import json
from pathlib import Path

import torch

from mirroropd.proximal import power_log_target
from training.power_mean_worker import panels


def main():
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[2]
    inputs = []
    for rank in range(4):
        artifact = torch.load(
            repo / f"data/oneshot-effective-time-20261012/seed-42/rank-{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        selected = panels(artifact, {"kappa": 0, "strength": 0.1, "batch_size": 64}, device="cpu")[
            "train"
        ]
        inputs.extend((s["qlog"], s["tlog"]) for s in selected)
    qlog, tlog = (torch.cat([pair[i] for pair in inputs]) for i in (0, 1))

    def radius(kappa, strength):
        mlog = power_log_target(qlog, tlog, kappa, strength)
        return float((4 * torch.exp((qlog + mlog) / 2).sum(-1).clamp(0, 1).acos().square()).mean())

    reference = radius(0, 0.1)
    strengths = {"0.0": 0.1}
    measured = {"0.0": reference}
    for kappa in (0.5, 1.0):
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = (low + high) / 2
            if radius(kappa, middle) < reference:
                low = middle
            else:
                high = middle
        strengths[str(kappa)] = (low + high) / 2
        measured[str(kappa)] = radius(kappa, strengths[str(kappa)])
        assert abs(measured[str(kappa)] / reference - 1) < 1e-5
    result = {
        "bf16": {
            "matched_strengths": strengths,
            "training_prefixes": len(qlog),
            "training_responses": 64,
            "mean_squared_fisher_rao_radius": reference,
            "measured_radius_by_shape": measured,
        },
        "amendment": "The 128-prefix diagnostic calibration did not transfer to the 4096-prefix panel. GPU radius controls now use the exact training panel, excluding heldout positions and all trained-model outcomes.",
    }
    (repo / "results/power-mean-20261013/training-radius-calibration.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
