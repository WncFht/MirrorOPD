"""Fit low-LR effective-time dynamics and audit held-out high-LR histories."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

RUN_IDS = ("ybi7q89v", "mdxt04lm", "yj2w13fq", "ml6mz58f", "l0a6rsc1", "vmk256kl")
FIT_IDS = RUN_IDS[:2]


def curve(parameters, time):
    floor, fast_amplitude, slow_amplitude, fast_rate, slow_rate = np.exp(parameters)
    return (
        floor
        + fast_amplitude * np.exp(-fast_rate * (time - 30))
        + slow_amplitude * np.exp(-slow_rate * (time - 30))
    )


def fit_reference(frames, offset):
    times, values = [], []
    for run_id in FIT_IDS:
        frame = frames[run_id]
        time = (frame["step"].to_numpy() - offset) * frame["lr"].iloc[0] / 1e-6
        mask = time >= 30
        times.extend(time[mask])
        values.extend(frame["distance"].to_numpy()[mask])
    times, values = np.array(times), np.array(values)
    fitted = []
    for fast in (0.01, 0.02, 0.05, 0.1):
        for slow in (0.001, 0.003, 0.005, 0.01):
            fitted.append(  # noqa: PERF401 -- explicit nonlinear multistart fits
                least_squares(
                    lambda p: np.log(curve(p, times)) - np.log(values),
                    np.log([0.0005, 0.06, 0.03, fast, slow]),
                    bounds=(np.log([1e-8, 1e-7, 1e-7, 1e-5, 1e-5]), np.log([0.02, 1, 1, 1, 1])),
                    max_nfev=3000,
                )
            )
    best = min(fitted, key=lambda result: result.cost)
    parameters = np.exp(best.x)
    return best.x, {
        "clock_offset": offset,
        "fit_run_ids": list(FIT_IDS),
        "fit_min_u": 30,
        "fit_max_u": float(times.max()),
        "floor": float(parameters[0]),
        "amplitudes": parameters[1:3].tolist(),
        "rates_per_u": parameters[3:5].tolist(),
        "training_log_rmse": float(np.sqrt(2 * best.cost / len(times))),
        "hypothetical_two_mode_GD_upper_lr": float(4e-6 / max(parameters[3:5])),
        "upper_lr_semantics": "Not an AdamW stability guarantee; omits invisible fast modes and changing geometry.",
    }


def effective_median(effective_time, values, window=30):
    """Smooth with a fixed effective-time width, retaining insufficient resolution as NaN."""
    smoothed = np.full(len(values), np.nan)
    for i, u in enumerate(effective_time):
        first = np.searchsorted(effective_time, u - window, side="left")
        if i - first + 1 >= 3:
            smoothed[i] = np.median(values[first : i + 1])
    return smoothed


def sustained_hit(time, effective_time, values, threshold):
    """Distinguish a persistent crossing from right censoring and unresolved sampling."""
    smoothed = effective_median(effective_time, values)
    crossing = np.flatnonzero(np.isfinite(smoothed) & (smoothed <= threshold))
    eligible = effective_time <= effective_time[-1] - 30
    index = next(
        (
            i
            for i in range(len(values))
            if eligible[i]
            and np.isfinite(smoothed[i])
            and np.isfinite(smoothed[i:]).all()
            and np.all(smoothed[i:] <= threshold)
        ),
        None,
    )
    if index is not None:
        return {
            "status": "qualified",
            "step": int(time[index]),
            "smoothed_distance": float(smoothed[index]),
        }
    if len(crossing):
        return {
            "status": "right_censored_or_not_persistent",
            "step": None,
            "first_smoothed_step": int(time[crossing[0]]),
        }
    return {
        "status": "not_reached"
        if np.isfinite(smoothed).any()
        else "insufficient_effective_time_resolution",
        "step": None,
    }


def velocity_models(frames):
    """Fit stage-average absorption as a quadratic in LR; hold out large rates first."""
    results = []
    for window in (30, 60, 90):
        for upper, lower in ((0.05, 0.02), (0.02, 0.01)):
            measurements = []
            for run_id, frame in frames.items():
                lr = float(frame["lr"].iloc[0])
                steps = frame["step"].to_numpy() - 1
                u = steps * lr / 1e-6
                smooth = effective_median(u, frame["distance"].to_numpy(), window)
                hits = [
                    np.flatnonzero(np.isfinite(smooth) & (smooth <= threshold))
                    for threshold in (upper, lower)
                ]
                if all(len(hit) for hit in hits):
                    elapsed = int(steps[hits[1][0]] - steps[hits[0][0]])
                    if elapsed > 0:
                        measurements.append(
                            {
                                "id": run_id,
                                "lr": lr,
                                "elapsed_updates": elapsed,
                                "rate": float((upper - lower) / elapsed),
                                "log_rate": float(np.log(upper / lower) / elapsed),
                                "grad_norm_median_in_band": float(
                                    np.median(
                                        frame["grad_norm"].to_numpy()[hits[0][0] : hits[1][0] + 1]
                                    )
                                ),
                            }
                        )
            for phase, cutoff in (
                ("held_out_1e-5_and_1e-4", 5e-6),
                ("refit_all_usable_rates", 1e-5),
            ):
                training = [r for r in measurements if r["lr"] <= cutoff * (1 + 1e-12)]
                x = np.array([r["lr"] / 1e-6 for r in training])
                v = np.array([r["rate"] for r in training])
                a, b = np.linalg.lstsq(np.column_stack((x, -x * x)), v, rcond=None)[0]
                results.append(
                    {
                        "phase": phase,
                        "effective_window": window,
                        "distance_band": [upper, lower],
                        "fit_run_ids": [r["id"] for r in training],
                        "model": "v(x)=a*x-b*x^2, x=lr/1e-6; v is absolute proxy decrease per update",
                        "a": float(a),
                        "b": float(b),
                        "candidate_zero_rate_lr": float(1e-6 * a / b) if a > 0 and b > 0 else None,
                        "candidate_best_rate_lr": float(0.5e-6 * a / b)
                        if a > 0 and b > 0
                        else None,
                        "fit_relative_rmse": float(
                            np.sqrt(np.mean(((a * x - b * x * x - v) / v) ** 2))
                        ),
                        "measurements": measurements,
                        "held_out_1e-5_predicted_rate": float(a * 10 - b * 100),
                        "held_out_1e-4_predicted_rate": float(a * 100 - b * 10000),
                        "validation_limit": "1e-4 never reaches either lower band endpoint; its global failure is compatible with negative-rate prediction but is not a same-state local velocity test.",
                    }
                )
    return results


def analyze(source, inputs, output):
    inputs.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    frames, settings, signatures = {}, {}, {}
    columns = {
        "_step": "step",
        "actor/lr": "lr",
        "opd/token_reward_mean": "reward",
        "actor/grad_norm": "grad_norm",
        "opd/top_k_reward_mean": "topk_reward",
        "actor/entropy": "entropy",
    }
    keys = (
        "data.train_files",
        "data.train_batch_size",
        "data.max_response_length",
        "actor_rollout_ref.model.path",
        "distillation.fsdp_teacher.model_path",
        "actor_rollout_ref.actor.optim.lr",
        "actor_rollout_ref.actor.optim.betas",
        "actor_rollout_ref.actor.optim.weight_decay",
        "actor_rollout_ref.actor.grad_clip",
        "actor_rollout_ref.actor.ppo_epochs",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.fsdp_config.dtype",
        "actor_rollout_ref.actor.fsdp_config.model_dtype",
        "actor_rollout_ref.actor.loss_agg_mode",
        "actor_rollout_ref.actor.fsdp_config.seed",
        "actor_rollout_ref.rollout.n",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.rollout.top_p",
        "actor_rollout_ref.rollout.top_k",
        "algorithm.frozen_rollout",
        "trainer.n_gpus_per_node",
        "distillation.fsdp_teacher.kl_estimator",
        "distillation.distillation_loss.policy_loss_mode",
    )
    for run_id in RUN_IDS:
        csv_path, meta_path = source / f"{run_id}__history.csv", source / f"{run_id}__meta.json"
        frame = pd.read_csv(csv_path).sort_values("_step").reset_index(drop=True)
        if frame["_step"].duplicated().any() or not np.array_equal(
            frame["_step"], np.arange(1, 280)
        ):
            raise ValueError("expected one complete 279-row history per run")
        frame = frame[list(columns)].rename(columns=columns)
        frame["distance"] = -frame.pop("reward")
        frame["topk_distance"] = -frame.pop("topk_reward")
        if not frame["lr"].eq(frame["lr"].iloc[0]).all() or (frame["distance"] <= 0).any():
            raise ValueError("expected positive proxy and constant logged LR")
        frame.to_csv(inputs / f"{run_id}.csv", index=False)
        frames[run_id] = frame
        meta = json.loads(meta_path.read_text())
        settings[run_id] = {key: meta["config_flat"].get(key) for key in keys}
        signatures[run_id] = {
            "url": meta["url"],
            "source_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "source_meta_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            "extracted_csv_sha256": hashlib.sha256(
                (inputs / f"{run_id}.csv").read_bytes()
            ).hexdigest(),
        }
    (inputs / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (inputs / "source-signatures.json").write_text(json.dumps(signatures, indent=2) + "\n")
    parameters, fitted = fit_reference(frames, offset=1)
    historical_parameters, historical_fit = fit_reference(frames, offset=0)
    fitted["predicted_threshold_u"] = {
        str(threshold): float(
            brentq(
                lambda u, threshold=threshold: curve(parameters, np.array([u]))[0] - threshold,
                30,
                1e6,
            )
        )
        for threshold in (0.02, 0.01, 0.005, 0.002)
    }
    results = []
    for run_id, frame in frames.items():
        lr = float(frame["lr"].iloc[0])
        time = frame["step"].to_numpy()
        u = (time - 1) * lr / 1e-6
        d = frame["distance"].to_numpy()
        grad = frame["grad_norm"].to_numpy()
        smooth = pd.Series(d).rolling(20, min_periods=20).median().to_numpy()
        minimum = float(np.nanmin(smooth))
        tail = float(np.median(d[-30:]))
        errors = np.log(d[u >= 30] / curve(parameters, u[u >= 30]))
        common = (u >= 30) & (u <= 139)
        common_errors = np.log(d[common] / curve(parameters, u[common]))
        before = float(np.median(d[-60:-30]))
        stats = {
            "id": run_id,
            "lr": lr,
            "fit_role": "fit" if run_id in FIT_IDS else "held-out",
            "distance_initial": float(d[0]),
            "distance_final": float(d[-1]),
            "distance_tail_median30": tail,
            "best_rolling20_median": minimum,
            "best_rolling20_step": int(time[np.nanargmin(smooth)]),
            "tail_over_best_rolling20": tail / minimum,
            "tail_over_previous30": tail / before,
            "grad_norm_tail_median30": float(np.median(grad[-30:])),
            "fraction_norm_above_clip1": float(np.mean(grad > 1)),
            "all_time_model_log_rmse": float(np.sqrt(np.mean(errors**2))),
            "common_time_model_log_rmse": float(np.sqrt(np.mean(common_errors**2))),
            "common_time_median_observed_over_model": float(np.exp(np.median(common_errors))),
            "end_observed_over_model": float(d[-1] / curve(parameters, np.array([u[-1]]))[0]),
            "historical_clock_common_log_rmse": float(
                np.sqrt(
                    np.mean(
                        np.log(
                            d[(time * lr / 1e-6 >= 30) & (time * lr / 1e-6 <= 139)]
                            / curve(
                                historical_parameters,
                                time[(time * lr / 1e-6 >= 30) & (time * lr / 1e-6 <= 139)]
                                * lr
                                / 1e-6,
                            )
                        )
                        ** 2
                    )
                )
            ),
            "reference_u_max": float(u[-1]),
            "observed_rebound_failure_25pct": tail / minimum > 1.25,
            "threshold_hits": {},
        }
        for threshold in (0.02, 0.01, 0.005, 0.002):
            hit = sustained_hit(time, u, d, threshold)
            if hit["step"] is not None:
                hit["u_pre_update"] = (hit["step"] - 1) * lr / 1e-6
                hit["u_historical"] = hit["step"] * lr / 1e-6
            stats["threshold_hits"][str(threshold)] = hit
        tolerance = 0.05 * (d[0] - tail)
        index = next(
            (
                i
                for i in range(len(smooth) - 19)
                if np.isfinite(smooth[i]) and np.all(np.abs(smooth[i:] - tail) <= tolerance)
            ),
            None,
        )
        stats["observed_95pct_proximity"] = (
            None
            if index is None
            else {
                "step": int(time[index]),
                "u_pre_update": float(u[index]),
                "tail_reference": tail,
                "definition": "retrospective tail reference; persistent rolling20 median; not an asymptotic floor",
            }
        )
        results.append(stats)
    stage_models = velocity_models(frames)
    (output / "sensitivity-velocity-models.json").write_text(
        json.dumps([model for model in stage_models if model["effective_window"] != 30], indent=2)
        + "\n"
    )
    result = {
        "distance_semantics": "negative signed sampled-token reward mean; on-policy reverse-KL Monte Carlo proxy, not full-vocabulary KL",
        "primary_clock": "(logged_step - 1) * lr / 1e-6, update-before rollout distance",
        "threshold_smoothing": "trailing u-window 30 with >=3 points; require all later smoothed values below threshold and >=30 u of observed persistence. LR=1e-4 too sparse for this smoother; its raw distances all exceed 0.02.",
        "fit_protocol": "two positive exponentials plus floor, fitted only to LR 5e-7 and 1e-6 at u>=30; four other rates held out. All three diagnostics and both clocks retained.",
        "stage_velocity_models": [
            model for model in stage_models if model["effective_window"] == 30
        ],
        "stage_velocity_sensitivity_file": "sensitivity-velocity-models.json",
        "reference_fit": fitted,
        "historical_clock_fit": historical_fit,
        "runs": results,
        "tested_usable_lr": 1e-5,
        "tested_rebound_failure_lr": 1e-4,
        "unknown_interval": [1e-5, 1e-4],
        "interval_semantics": "No test between these points; empirical usefulness endpoints, not a certified global maximum or numerical divergence bound.",
    }
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in ("threshold_hits", "observed_95pct_proximity")
            }
            for row in results
        ]
    )
    summary.to_csv(output / "runs.csv", index=False)
    print(
        json.dumps(
            {
                "fit": fitted,
                "held_out": [
                    {
                        "lr": r["lr"],
                        "shared_error": r["common_time_model_log_rmse"],
                        "end_over_model": r["end_observed_over_model"],
                    }
                    for r in results
                    if r["fit_role"] == "held-out"
                ],
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.source, args.inputs, args.output)


if __name__ == "__main__":
    main()
