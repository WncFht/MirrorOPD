"""Unequal token counts, DP aggregation and stopped diagnostics contracts."""

import ast
import tempfile
from pathlib import Path

import torch

from mirroropd.training.metrics import append_inner_record, finalize_token_moments, token_moments


def main():
    # Two DP ranks with different valid lengths and two micros per rank.
    values = [
        [torch.tensor([1.0]), torch.tensor([2.0, 3.0])],
        [torch.tensor([4.0, 5.0, 6.0]), torch.tensor([7.0])],
    ]
    root = Path(__file__).resolve().parents[1]
    metric_path = root / "upstream/TA-OPD/verl/verl/utils/metric/utils.py"
    ns = {}
    exec(compile(ast.parse(metric_path.read_text()), str(metric_path), "exec"), ns)
    metric_class = ns["Metric"]
    ranks = []
    for micros in values:
        result = {}
        for value in micros:
            fields = {"mirror_importance_ratio": value, "mirror_sampled_reward": -value}
            fields = {k: torch.cat([v, torch.tensor([float("nan")])]) for k, v in fields.items()}
            mask = torch.arange(len(value) + 1) < len(value)
            for key, (reduction, scalar) in token_moments(fields, mask).items():
                result.setdefault(key, metric_class(reduction)).append(scalar)
        ranks.append(result)
    aggregated = {k: metric_class.aggregate_dp([r[k] for r in ranks]) for k in ranks[0]}
    final = finalize_token_moments(aggregated, dp_size=2)
    all_values = torch.arange(1, 8, dtype=torch.float64)
    assert final["mirror/valid_tokens"] == 7
    assert final["mirror/importance_ratio/mean"] == 4
    assert final["mirror/importance_ratio/std"] == 2
    assert final["mirror/importance_ratio/min"] == 1
    assert final["mirror/importance_ratio/max"] == 7
    assert (
        abs(
            final["mirror/importance_ess"]
            - float(all_values.sum() ** 2 / all_values.square().sum())
        )
        < 1e-12
    )
    assert not any(k.startswith("mirror/raw") for k in final)
    assert token_moments({}, torch.zeros(3, dtype=torch.bool)) == {"mirror/raw/count": ("sum", 0.0)}
    assert finalize_token_moments({"mirror/raw/count": 0}) == {"mirror/valid_tokens": 0}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "inner.jsonl"
        append_inner_record(path, final, outer_call=2, inner_step=3)
        import json

        record = json.loads(path.read_text())
        assert record["outer_call"] == 2
        assert record["inner_step"] == 3
        assert record["measurement"] == "pre-update"
    # Execute actual patched trainer metric callback with identity layout.
    path = root / "upstream/TA-OPD/verl/verl/trainer/distillation/losses.py"
    tree = ast.parse(path.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "compute_mirror_opd_loss"
    )
    node.decorator_list = []
    env = {"Metric": metric_class, "no_padding_2_padding": lambda x, _data: x}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    from types import SimpleNamespace

    settings = SimpleNamespace(
        use_policy_gradient=False,
        use_task_rewards=False,
        loss_max_clamp=None,
        log_prob_min_clamp=None,
    )
    x = torch.tensor([[1.0, 2.0, 9.0]], requires_grad=True)
    loss, metrics = env["compute_mirror_opd_loss"](
        None,
        SimpleNamespace(distillation_loss=settings),
        {"distillation_losses": x, "mirror_sampled_reward": -x},
        {"response_mask": torch.tensor([[True, True, False]])},
    )
    assert loss is x
    assert loss.requires_grad
    assert metrics["mirror/raw/sampled_reward/sum"].aggregate() == -3
    print("PASS: token-weighted micro/DP moments, ESS, masking, JSONL and trainer callback")


if __name__ == "__main__":
    main()
