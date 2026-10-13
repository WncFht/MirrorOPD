"""Sequence-consistent arithmetic posterior weights on recorded real model trajectories."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    torch.set_num_threads(4)
    repo = Path(__file__).resolve().parents[2]
    output = repo / "results/power-mean-20261013"
    records = []
    depths = {s: [] for s in (0.1, 0.3)}
    for rank in range(4):
        artifact = torch.load(
            repo / f"data/oneshot-effective-time-20261012/seed-42/rank-{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        for index, sequence in enumerate(artifact["sequences"]):
            cache = sequence["cache_bf16"]
            r = (
                cache["mirror_teacher_action_log_prob"] - cache["mirror_anchor_action_log_prob"]
            ).double()
            before = torch.cat((torch.zeros(1), r.cumsum(0)[:-1]))
            for strength in (0.1, 0.3):
                logit = np.log(strength / (1 - strength)) + before
                log_posterior = -torch.nn.functional.softplus(-logit)
                total = r.sum()
                zero = torch.tensor(np.log1p(-strength), dtype=torch.float64)
                prior = torch.tensor(np.log(strength), dtype=torch.float64)
                increment = torch.logaddexp(zero, prior + before + r) - torch.logaddexp(
                    zero, prior + before
                )
                full = torch.logaddexp(zero, prior + total)
                assert abs(float(increment.sum() - full)) < 1e-8
                records.append(
                    {
                        "response": rank * 16 + index,
                        "strength": strength,
                        "tokens": len(r),
                        "sequence_log_teacher_anchor_ratio": float(total),
                        "mean_posterior_teacher_weight": float(log_posterior.exp().mean()),
                        "fraction_posterior_below_1em6": float(
                            (log_posterior < np.log(1e-6)).double().mean()
                        ),
                    }
                )
                for start in range(0, len(r), 256):
                    depths[strength].append(
                        (start, float((log_posterior[start : start + 256] / np.log(10)).mean()))
                    )
    summary = {
        "responses": 64,
        "prompts": 1,
        "scope": "Recorded 1.5B model trajectories; prefix posterior of a sequence-level arithmetic mixture, not constant token interpolation",
        "records": records,
    }
    for strength in (0.1, 0.3):
        rows = [r for r in records if r["strength"] == strength]
        summary[str(strength)] = {
            "mean_sequence_log_ratio": float(
                np.mean([r["sequence_log_teacher_anchor_ratio"] for r in rows])
            ),
            "mean_response_posterior": float(
                np.mean([r["mean_posterior_teacher_weight"] for r in rows])
            ),
            "token_weighted_fraction_below_1em6": sum(
                r["tokens"] * r["fraction_posterior_below_1em6"] for r in rows
            )
            / sum(r["tokens"] for r in rows),
        }
    (output / "sequence-posterior-audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for strength in (0.1, 0.3):
        bins = sorted({d for d, _ in depths[strength]})
        values = [np.mean([v for d, v in depths[strength] if d == b]) for b in bins]
        ax.plot(bins, values, label=f"Sequence mixture strength={strength:g}")
        ax.axhline(np.log10(strength), ls=":", alpha=0.5)
    ax.set(
        xlabel="Response depth",
        ylabel="Mean log10 posterior teacher weight",
        title="Sequence and token arithmetic mixtures differ on real trajectories",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    for extension in ("png", "pdf"):
        fig.savefig(
            output / f"sequence-posterior.{extension}",
            dpi=180,
            metadata={"CreationDate": None, "ModDate": None} if extension == "pdf" else None,
        )
    plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("0.1", "0.3")}, indent=2))


if __name__ == "__main__":
    main()
