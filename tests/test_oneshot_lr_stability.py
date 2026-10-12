"""Scientific checks for time-window comparability and held-out LR isolation."""

import numpy as np
import pandas as pd

from experiments.analysis.oneshot_lr_stability import RUN_IDS, sustained_hit, velocity_models


def test_threshold_window_has_common_effective_time_lag():
    hits = []
    for scale in (0.5, 1, 2):
        steps = np.arange(1, 401)
        time = (steps - 1) * scale
        distance = 0.001 + 0.1 * np.exp(-time / 50)
        hit = sustained_hit(steps, time, distance, 0.02)
        assert hit["status"] == "qualified"
        hits.append((hit["step"] - 1) * scale)
    assert max(hits) - min(hits) <= 2


def test_velocity_calibration_does_not_use_held_out_rates():
    frames = {}
    for run_id, scale in zip(RUN_IDS, (0.5, 1, 2, 5, 10, 100), strict=True):
        steps = np.arange(1, 280)
        frames[run_id] = pd.DataFrame(
            {
                "step": steps,
                "lr": scale * 1e-6,
                "distance": 0.1 * np.exp(-0.01 * scale * (steps - 1)),
                "grad_norm": np.ones(len(steps)),
            }
        )
    before = [row for row in velocity_models(frames) if row["phase"] == "held_out_1e-5_and_1e-4"]
    for run_id in RUN_IDS[-2:]:
        frames[run_id]["distance"] = 0.1
    after = [row for row in velocity_models(frames) if row["phase"] == "held_out_1e-5_and_1e-4"]
    assert [(row["a"], row["b"]) for row in before] == [(row["a"], row["b"]) for row in after]
