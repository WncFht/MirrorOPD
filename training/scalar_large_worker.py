"""Streaming fixed-pool BPO benchmark with paired minibatches and held-out audits."""

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from mirroropd.training.losses import compute_mirror_opd_token_loss
from mirroropd.training.scalar_schedule import absorption_feedback, residual_line_search
from training.scalar_large_gpu import register_owner

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def cache_helpers():
    sys.path.insert(0, str(ROOT / "experiments/rollout"))
    from scalar_large_cache import load_student_helper, response_logits

    return load_student_helper(), response_logits


class FrozenPool:
    def __init__(self, directory, train_sequences, seed):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.entries = self.manifest["entries"]
        self.records = [
            torch.load(self.directory / entry["file"], mmap=True, weights_only=False)
            for entry in self.entries
        ]
        self.training = self.records[:train_sequences]
        self.validation = self.records[self.manifest["train_sequences"] :]
        self.lengths = torch.tensor(
            [r["response_tokens"] for r in self.training], dtype=torch.float64
        )
        self.sampler = torch.Generator().manual_seed(seed + 100000)
        self.audit_train = self.uniform_positions(self.training, 4096, seed + 200000)
        self.audit_validation = self.uniform_positions(self.validation, 4096, seed + 300000)

    @staticmethod
    def uniform_positions(records, count, seed):
        total = sum(r["response_tokens"] for r in records)
        indices = (
            torch.randperm(total, generator=torch.Generator().manual_seed(seed))[
                : min(count, total)
            ]
            .sort()
            .values
        )
        result = []
        offset = 0
        for record in records:
            chosen = (
                indices[(indices >= offset) & (indices < offset + record["response_tokens"])]
                - offset
            )
            if len(chosen):
                result.append((record, chosen))
            offset += record["response_tokens"]
        return result

    def proxy(self, count, seed, clusters=None):
        # A fixed cluster sample is cheap enough for repeated ray evaluations.
        generator = torch.Generator().manual_seed(seed + 400000)
        clusters = min(len(self.training), clusters or max(2, math.ceil(count / 256)))
        selected = torch.multinomial(self.lengths, clusters, replacement=False, generator=generator)
        records = [self.training[i] for i in selected.tolist()]
        return self.uniform_positions(records, count, seed + 500000)

    def batch(self, count):
        selected = torch.multinomial(
            self.lengths, count, replacement=True, generator=self.sampler
        ).tolist()
        return [
            (self.training[index], multiplicity / count)
            for index, multiplicity in collections.Counter(selected).items()
        ]


def field(logits, record, config, positions=None):
    index = positions if positions is not None else slice(None)
    cache = record["caches"][str(config["topk"])]
    qlog = (
        record["qlog"][index].cuda() if config["objective"] in ("exact", "sampled_full") else None
    )
    if config["objective"] == "exact":
        tlog = record["tlog"][index].cuda()
        logp = logits.log_softmax(-1)
        delta = logp - (1 - config["eta"]) * qlog - config["eta"] * tlog
        residual = delta - (qlog.exp() * delta).sum(-1, keepdim=True)
        return (qlog.exp() * residual.square()).sum(-1).mean() / (2 * config["eta"])
    kt = (
        record["teacher_full_center"][index].cuda()
        if config["objective"] == "sampled_full"
        else cache["mirror_teacher_center"][index].cuda()
    )
    return compute_mirror_opd_token_loss(
        logits,
        cache["mirror_actions"][index].cuda(),
        cache["mirror_anchor_ids"][index].cuda(),
        cache["mirror_anchor_head_log_probs"][index].cuda(),
        cache["mirror_anchor_tail_log_prob"][index].cuda(),
        cache["mirror_anchor_action_log_prob"][index].cuda(),
        cache["mirror_teacher_action_log_prob"][index].cuda(),
        kt,
        "bpo",
        config["eta"],
        full_anchor_log_probs=qlog,
    )


def distribution_metrics(logits, record, positions, eta):
    logp = logits.log_softmax(-1)
    qlog, tlog = record["qlog"][positions].cuda(), record["tlog"][positions].cuda()
    q = qlog.exp()
    delta = logp - (1 - eta) * qlog - eta * tlog
    residual = delta - (q * delta).sum(-1, keepdim=True)
    actions = record["ids"][record["start"] + 1 :][positions].cuda()
    gap = (logp - tlog).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return {
        "bpo": float((q * residual.square()).sum(-1).double().sum() / (2 * eta)),
        "teacher_reverse_kl": float((logp.exp() * (logp - tlog)).sum(-1).double().sum()),
        "anchor_forward_kl": float((q * (qlog - logp)).sum(-1).double().sum()),
        "sampled_abs_gap": float(gap.abs().double().sum()),
        "count": len(positions),
    }


class Worker:
    def __init__(self, config, output, device):
        if device not in range(4) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(device):
            raise RuntimeError("this worker only runs on assigned physical GPU 0--3")
        torch.cuda.set_device(0)
        torch.set_num_threads(4)
        self.config = config
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.pool = FrozenPool(config["cache"], config["train_sequences"], config["seed"])
        load_model, self.logits = cache_helpers()
        self.model = load_model(self.pool.manifest["student"])
        register_owner(device)
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
        self.parameters = list(self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=config["lr"],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
            foreach=False,
        )
        self.proxy = self.pool.proxy(
            config["monitor_positions"], config["seed"], config.get("monitor_clusters")
        )
        self.counts = {
            "forwards": 0,
            "backwards": 0,
            "forward_tokens": 0,
            "backward_tokens": 0,
            "search_forwards": 0,
            "rejected": 0,
        }
        self.lr = config["lr"]
        self.previous = None
        self.initial_proxy = None
        self.elapsed = 0
        self.begin = time.perf_counter()
        self.start_update = 0
        self.restore()

    def forward(self, record, positions=None, precision=None):
        self.counts["forwards"] += 1
        self.counts["forward_tokens"] += len(record["ids"])
        return self.logits(
            self.model, record, positions, precision=precision or self.config["precision"]
        )

    def proxy_value(self):
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for record, positions in self.proxy:
                logits = self.forward(record, positions, "fp32")
                if self.config["monitor"] == "full":
                    value = distribution_metrics(logits, record, positions, self.config["eta"])[
                        "bpo"
                    ]
                else:
                    result = field(logits, record, self.config, positions)
                    if isinstance(result, torch.Tensor):
                        value = float(result) * len(positions)
                    else:
                        value = float(
                            result["mirror_score_coefficient"].square().double().sum()
                        ) * (self.config["eta"] / 2)
                total += value
                count += len(positions)
        return total / count

    def audit(self, update):
        self.model.eval()
        result = {}
        with torch.no_grad():
            for label, records in (
                ("train", self.pool.audit_train),
                ("validation", self.pool.audit_validation),
            ):
                accumulated = collections.defaultdict(float)
                for record, positions in records:
                    values = distribution_metrics(
                        self.forward(record, positions, "fp32"),
                        record,
                        positions,
                        self.config["eta"],
                    )
                    for key, value in values.items():
                        accumulated[key] += value
                count = accumulated.pop("count")
                result[label] = {key: value / count for key, value in accumulated.items()} | {
                    "positions": count
                }
        self.append(
            "audits.jsonl",
            {
                "update": update,
                "seconds": self.elapsed + time.perf_counter() - self.begin,
                "counts": dict(self.counts),
                "metrics": result,
            },
        )
        return result

    def append(self, name, record):
        with (self.output / name).open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    def save(self, update):
        snapshot = {
            "update": update,
            "parameters": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "optimizer": self.optimizer.state_dict(),
            "sampler": self.pool.sampler.get_state(),
            "lr": self.lr,
            "previous": self.previous,
            "initial_proxy": self.initial_proxy,
            "counts": self.counts,
            "elapsed": self.elapsed + time.perf_counter() - self.begin,
            "config": self.config,
        }
        temporary = self.output / "resume.tmp"
        torch.save(snapshot, temporary)
        temporary.replace(self.output / "resume.pt")
        self.append("checkpoints.jsonl", {"update": update, "checkpoint": "resume.pt"})

    def restore(self):
        checkpoint = self.output / "resume.pt"
        if not checkpoint.exists():
            return
        record = torch.load(checkpoint, weights_only=False, map_location="cpu")
        if record["config"] != self.config:
            raise ValueError("checkpoint configuration differs; preserved")
        self.model.load_state_dict(record["parameters"])
        self.optimizer.load_state_dict(record["optimizer"])
        self.pool.sampler.set_state(record["sampler"])
        self.start_update = record["update"]
        for name, key in (
            ("updates.jsonl", "update_after"),
            ("audits.jsonl", "update"),
            ("checkpoints.jsonl", "update"),
        ):
            path = self.output / name
            if path.exists():
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                rows = [row for row in rows if row[key] <= self.start_update]
                path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))

        self.lr = record["lr"]
        self.previous = record["previous"]
        self.initial_proxy = record["initial_proxy"]
        self.counts = record["counts"]
        self.elapsed = record["elapsed"]
        self.begin = time.perf_counter()

    def direction(self):
        result = []
        for parameter in self.parameters:
            state = self.optimizer.state[parameter]
            step = int(state["step"])
            first = state["exp_avg"] / (1 - 0.9**step)
            denominator = (state["exp_avg_sq"] / (1 - 0.999**step)).sqrt().add_(1e-8)
            result.append(first / denominator)
        return result

    def update(self, index):
        config = self.config
        search = config["method"] == "search" and index % config["search_every"] == 0
        needs_proxy = (
            not config.get("minimal_monitoring", True)
            or config["method"] in ("feedback", "logloss")
            or search
        )
        value = self.proxy_value() if needs_proxy else None
        if value is not None:
            if self.initial_proxy is None:
                self.initial_proxy = value
            if not math.isfinite(value) or value > self.initial_proxy * 1e6:
                raise FloatingPointError("residual exceeded the preregistered stability limit")
        if config["method"] == "feedback" and self.previous is not None:
            self.lr = absorption_feedback(
                self.lr, self.previous, value, config["target_rate"], config["maximum"]
            )
        self.optimizer.zero_grad(set_to_none=True)
        self.model.train()
        loss_value = 0.0
        batch = (
            [(r, r["response_tokens"] / float(self.pool.lengths.sum())) for r in self.pool.training]
            if config.get("full_batch")
            else self.pool.batch(config["batch_sequences"])
        )
        for record, weight in batch:
            logits = self.forward(record)
            result = field(logits, record, config)
            loss = (
                result if isinstance(result, torch.Tensor) else result["distillation_losses"].mean()
            )
            scale = (
                max(self.initial_proxy, 1e-30) / max(value, self.initial_proxy * 1e-6, 1e-30)
                if config["method"] == "logloss"
                else 1
            )
            (loss * weight * scale).backward()
            self.counts["backwards"] += 1
            self.counts["backward_tokens"] += record["response_tokens"]
            loss_value += float(loss.detach()) * weight
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, 2))
        if not math.isfinite(grad_norm):
            raise FloatingPointError("nonfinite gradient norm")
        self.optimizer.param_groups[0]["lr"] = self.lr
        selected = self.lr
        fitted = None
        if search:
            old = [p.detach().clone() for p in self.parameters]
            self.optimizer.step()
            direction = self.direction()

            def install(alpha):
                with torch.no_grad():
                    for parameter, before, d in zip(self.parameters, old, direction, strict=True):
                        parameter.copy_(before).add_(d, alpha=-alpha)

            def evaluate(alpha):
                install(alpha)
                before = self.counts["forwards"]
                measured = self.proxy_value()
                self.counts["search_forwards"] += self.counts["forwards"] - before
                return measured

            fitted = residual_line_search(
                evaluate, value, self.lr, config["maximum"], minimum=1e-10
            )
            selected = fitted.step
            install(selected)
            self.counts["rejected"] += selected == 0
            self.lr = max(config["lr"] * 0.01, self.lr / 2) if selected == 0 else selected
            del old, direction
        else:
            self.optimizer.step()
        if value is not None:
            self.previous = value
        record = {
            "update_before": index,
            "update_after": index + 1,
            "proxy_before": value,
            "surrogate_value": loss_value,
            "applied_lr": selected,
            "next_probe": self.lr,
            "gradient_norm_before_clip": grad_norm,
            "clipped": grad_norm > 2,
            "seconds": self.elapsed + time.perf_counter() - self.begin,
            "counts": dict(self.counts),
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        if fitted:
            record.update(
                ray_slope=fitted.slope,
                ray_curvature=fitted.curvature,
                ray_evaluations=fitted.evaluations,
                proxy_trial=fitted.value,
            )
        self.append("updates.jsonl", record)
        return record

    def run(self):
        if (self.output / "summary.json").exists():
            return
        if self.start_update == 0:
            initial = self.audit(0)
            if initial["train"]["anchor_forward_kl"] > 1e-5:
                raise ValueError("zero-step FP32 anchor alignment failed")
        for update in range(self.start_update, self.config["updates"]):
            try:
                record = self.update(update)
            except FloatingPointError as error:
                self.finish("nonfinite", str(error))
                return
            if update % 10 == 0:
                print(json.dumps(record), flush=True)
            if (update + 1) % self.config["audit_every"] == 0 or update + 1 == self.config[
                "updates"
            ]:
                self.audit(update + 1)
            if (update + 1) % 100 == 0 or update + 1 == self.config["updates"]:
                self.save(update + 1)
        self.finish("complete")

    def finish(self, status, reason=None):
        summary = {
            "config": self.config,
            "counts": self.counts,
            "seconds": self.elapsed + time.perf_counter() - self.begin,
            "source_sha256": SOURCE_SHA256,
            "manifest_sha256": hashlib.sha256(
                (self.pool.directory / "manifest.json").read_bytes()
            ).hexdigest(),
            "status": status,
            "monitoring_policy": "algorithm-required"
            if self.config.get("minimal_monitoring", True)
            else "every-step",
            "reason": reason,
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    Worker(config, args.output, args.device).run()


if __name__ == "__main__":
    main()
