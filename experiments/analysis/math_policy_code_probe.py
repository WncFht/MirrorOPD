"""CPU complete-action BPO gradient and W&B payload audit.

Run from MirrorOPD: PYTHONPATH=src CUDA_VISIBLE_DEVICES='' python <this file>
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch

from mirroropd.training.cache import build_mirror_opd_cache_from_full_logits
from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.tracking import WandbLogger

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--raw", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
target = args.output
target.mkdir(parents=True, exist_ok=True)
q = torch.tensor([0.4, 0.3, 0.2, 0.1])
t = torch.tensor([0.05, 0.15, 0.3, 0.5])
z = torch.tensor([-0.3, 0.2, 0.7, -0.1], requires_grad=True)
p = z.softmax(0)
eta = 0.3
u, r = p.log() - q.log(), t.log() - q.log()
ks = (q * (q.log() - p.log())).sum()
kt = (q * (q.log() - t.log())).sum()
residual = u + ks - eta * (r + kt)
exact = (q * residual.square()).sum() / (2 * eta)
gfull = torch.autograd.grad(exact, z, retain_graph=True)[0]
mdpo_exact = (p * (u / eta - r)).sum()
mdpo_gradient = torch.autograd.grad(mdpo_exact, z, retain_graph=True)[0]
mdpo_rows = []
rows = []
for k in [0, 1, 2, 4]:
    cache = build_mirror_opd_cache_from_full_logits(
        q.log().repeat(4, 1),
        t.log().repeat(4, 1),
        torch.arange(4),
        torch.ones(4, dtype=torch.bool),
        k,
    )
    out = compute_mirror_opd_token_loss(
        z.repeat(4, 1),
        cache["mirror_actions"],
        cache["mirror_anchor_ids"],
        cache["mirror_anchor_head_log_probs"],
        cache["mirror_anchor_tail_log_prob"],
        cache["mirror_anchor_action_log_prob"],
        cache["mirror_teacher_action_log_prob"],
        cache["mirror_teacher_center"],
        "bpo",
        eta,
    )
    g = torch.autograd.grad((q * out["distillation_losses"]).sum(), z, retain_graph=True)[0]
    hatks, hatkt = out["mirror_student_center"][0], cache["mirror_teacher_center"][0]
    bias = ((hatks - ks) / eta - (hatkt - kt)) * (q - p)
    rows.append(
        {
            "k": k,
            "gradient": g.tolist(),
            "exact_gradient": gfull.tolist(),
            "max_error": float((g - gfull).abs().max()),
            "bias_formula_max_error": float((g - gfull - bias).abs().max()),
            "KS": float(ks),
            "KT": float(kt),
            "hatKS": float(hatks),
            "hatKT": float(hatkt),
        }
    )
    assert float((g - gfull - bias).abs().max()) < 2e-6
    mdpo_out = compute_mirror_opd_token_loss(
        z.repeat(4, 1),
        cache["mirror_actions"],
        cache["mirror_anchor_ids"],
        cache["mirror_anchor_head_log_probs"],
        cache["mirror_anchor_tail_log_prob"],
        cache["mirror_anchor_action_log_prob"],
        cache["mirror_teacher_action_log_prob"],
        cache["mirror_teacher_center"],
        "mdpo",
        eta,
    )
    mdpo_g = torch.autograd.grad((q * mdpo_out["distillation_losses"]).sum(), z, retain_graph=True)[
        0
    ]
    error = float((mdpo_g - mdpo_gradient).abs().max())
    assert error < 2e-6
    mdpo_rows.append({"J": k, "gradient": mdpo_g.tolist(), "max_error": error})
assert rows[-1]["max_error"] < 2e-6
assert rows[2]["max_error"] > 0.01
(target / "bpo-coarse-center-gradient.json").write_text(json.dumps(rows, indent=2) + "\n")
(target / "mdpo-complete-action-gradient.json").write_text(json.dumps(mdpo_rows, indent=2) + "\n")


class FakeRun:
    step = 0

    def __init__(self):
        self.summary = {}

    def define_metric(self, *args, **kwargs):
        pass


class FakeWandb:
    run = FakeRun()

    def __init__(self):
        self.payloads = []

    def log(self, payload, step):
        self.payloads.append({"sdk_step": step, "payload": payload})


logger_backend = FakeWandb()
logger = WandbLogger(logger_backend, {})
run = args.raw / "dapo-deepseek-mdpo-o50-j16"
outer_lines = (run / "outer.jsonl").read_text().splitlines()
initial_outer = json.loads(outer_lines[0])
logger.log(initial_outer["data"], initial_outer["step"])
with tempfile.TemporaryDirectory() as scratch:
    first_batch = Path(scratch) / "inner.jsonl"
    first_batch.write_text("\n".join((run / "inner.jsonl").read_text().splitlines()[:16]) + "\n")
    logger.log_inner_records(first_batch)
first_outer = json.loads(outer_lines[1])
logger.log(first_outer["data"], first_outer["step"])
snapshot_key = "04_kl_centers/rollout_student_center/mean"
aggregate_key = "04_kl_centers/aggregate_student_center/mean"
snapshots = [item for item in logger_backend.payloads if snapshot_key in item["payload"]]
aggregates = [item for item in logger_backend.payloads if aggregate_key in item["payload"]]
assert len(snapshots) == len(aggregates) == 1
assert snapshots[0]["payload"]["00_axes/outer_rollout_step"] == 0
assert "00_axes/outer_rollout_step" not in aggregates[0]["payload"]
assert abs(snapshots[0]["payload"][snapshot_key]) < 1e-6
assert aggregates[0]["payload"][aggregate_key] > 1e-4
evidence = {"snapshot": snapshots[0], "aggregate": aggregates[0], "same_metric_name": False}
(target / "logger-current-axis-audit.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps({"BPO": rows, "current_logger_scopes": evidence}, indent=2))
