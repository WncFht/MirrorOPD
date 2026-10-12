"""Full-vocabulary center bias at the exact Gibbs target on large frozen pools."""

import argparse
import json
from pathlib import Path

import torch


def sequence_sums(record, eta):
    totals = {"count": 0, "initial_bpo": 0.0}
    for k in (16, 64):
        for name in (
            "head_mass",
            "teacher_kl_missing",
            "target_residual_bias_sq",
            "target_logit_field_norm",
        ):
            totals[f"k{k}_{name}"] = 0.0
    for start in range(0, len(record["qlog"]), 32):
        stop = start + 32
        qlog = record["qlog"][start:stop].double()
        tlog = record["tlog"][start:stop].double()
        q = qlog.exp()
        pgl = ((1 - eta) * qlog + eta * tlog).log_softmax(-1)
        pg = pgl.exp()
        kt = (q * (qlog - tlog)).sum(-1)
        ks = (q * (qlog - pgl)).sum(-1)
        signal = tlog - qlog + kt.unsqueeze(-1)
        totals["initial_bpo"] += float((q * signal.square()).sum() * eta / 2)
        distance = (q - pg).square().sum(-1).sqrt()
        count = len(qlog)
        totals["count"] += count
        for k in (16, 64):
            cache = record["caches"][str(k)]
            ids = cache["mirror_anchor_ids"][start:stop]
            qh = qlog.gather(-1, ids)
            ph = pgl.gather(-1, ids)
            qt = cache["mirror_anchor_tail_log_prob"][start:stop].double()
            outside = pgl.clone()
            outside.scatter_(-1, ids, float("-inf"))
            pt = outside.logsumexp(-1)
            coarse_ks = (qh.exp() * (qh - ph)).sum(-1) + qt.exp() * (qt - pt)
            coarse_kt = cache["mirror_teacher_center"][start:stop].double()
            bias = coarse_ks - ks - eta * (coarse_kt - kt)
            totals[f"k{k}_head_mass"] += float(qh.exp().sum())
            totals[f"k{k}_teacher_kl_missing"] += float((kt - coarse_kt).sum())
            totals[f"k{k}_target_residual_bias_sq"] += float(bias.square().sum())
            totals[f"k{k}_target_logit_field_norm"] += float((bias.abs() * distance / eta).sum())
    return totals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    manifest = json.loads((args.cache / "manifest.json").read_text())
    accumulated = {}
    for index, entry in enumerate(manifest["entries"]):
        record = torch.load(args.cache / entry["file"], weights_only=False, mmap=True)
        sums = sequence_sums(record, 0.3)
        for key, value in sums.items():
            accumulated[key] = accumulated.get(key, 0.0) + value
        if index % 16 == 0:
            print(json.dumps({"seed": manifest["seed"], "sequences": index + 1}), flush=True)
    count = accumulated.pop("count")
    result = {
        "seed": manifest["seed"],
        "prefixes": count,
        "eta": 0.3,
        "reference": "exact Gibbs target; logit field bias, not neural parameter gradient",
        "means": {key: value / count for key, value in accumulated.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
