"""Full-pool scalar BPO follow-up using the existing frozen 64-rollout inputs."""

import argparse
import json

from training.scalar_large_suite import ROOT, base_config, launch_phase, training_commands


def configurations(seeds=(42, 142, 242), updates=600):
    result = []
    for seed in seeds:
        base = base_config(seed) | {
            "full_batch": True,
            "batch_sequences": 64,
            "updates": updates,
            "minimal_monitoring": True,
        }
        result += [
            (f"fullpool-constant-{lr:.3g}-s{seed}", base | {"lr": lr})
            for lr in (2.5e-7, 5e-7, 1e-6, 2e-6, 4e-6)
        ]
        result += [
            (f"fullpool-search-every4-s{seed}", base | {"method": "search", "search_every": 4}),
            (f"fullpool-feedback-s{seed}", base | {"method": "feedback", "target_rate": 0.01}),
        ]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    root = ROOT / "runs/scalar-large"
    output = ROOT / "results/scalar-large/fullpool"
    output.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    plan = {
        "scope": "single-factor batch correction; same 64 frozen 256-token rollouts per seed",
        "full_batch_semantics": "all valid response tokens; sum length-weighted trajectory gradients; one AdamW step",
        "seeds": [42, 142, 242],
        "gpu_indices": [0, 1, 2, 3],
        "jobs": configs,
        "total_optimizer_updates": sum(config["updates"] for _, config in configs),
        "expected_backward_calls": sum(
            config["updates"] * config["train_sequences"] for _, config in configs
        ),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.plan_only:
        print(json.dumps({"jobs": len(configs), "output": str(output)}))
        return
    canary = [
        (name.replace("fullpool-", "fullpool-canary-"), config | {"updates": 4, "audit_every": 2})
        for name, config in configurations(seeds=(42,))
        if config["method"] != "constant" or config["lr"] == 5e-7
    ]
    launch_phase(
        "fullpool-canary",
        training_commands("fullpool-canary", canary),
        root / "fullpool-canary-controller",
    )
    for name, _ in canary:
        directory = root / "fullpool-canary" / name
        summary = json.loads((directory / "summary.json").read_text())
        rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()]
        if summary["status"] != "complete" or summary["counts"]["backwards"] != 256:
            raise RuntimeError("full-pool canary failed; formal jobs not started")
        if [row["update_after"] for row in rows] != [1, 2, 3, 4]:
            raise RuntimeError("full-pool canary update sequence is invalid")
    launch_phase("fullpool", training_commands("fullpool", configs), root / "fullpool-controller")
    from experiments.analysis.scalar_large_report import summarize
    from experiments.plotting.plot_scalar_large import plot_phase

    summarize(root / "fullpool", output, None)
    plot_phase(root / "fullpool", output / "figures")
    from experiments.plotting.plot_scalar_kl_lr import plot

    for seed in (42, 142, 242):
        plot(root / "fullpool", output / "figures", seed)


if __name__ == "__main__":
    main()
