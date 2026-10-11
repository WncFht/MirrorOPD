"""Global inner diagnostics keep DP reductions and have one file writer."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from mirroropd.training.metrics import append_inner_record, finalize_token_moments, token_moments


def four_rank_metrics():
    """Unequal rank token counts expose accidental rank-zero-only reduction."""
    ranks = []
    offset = 1
    for count in range(1, 5):
        values = torch.arange(offset, offset + count, dtype=torch.float64)
        ranks.append(token_moments({"mirror_importance_ratio": values}, torch.ones(count)))
        offset += count
    # verl Metric.aggregate_dp averages each SUM over DP ranks and takes
    # global MIN/MAX. Reproduce its reduced input without loading Ray or CUDA.
    reduced = {}
    for key in ranks[0]:
        reduction = ranks[0][key][0]
        values = [rank[key][1] for rank in ranks]
        if reduction == "sum":
            reduced[key] = sum(values) / len(ranks)
        elif reduction == "min":
            reduced[key] = min(values)
        else:
            reduced[key] = max(values)
    reduced["grad_norm"] = 2.0
    reduced["loss"] = 0.25
    return reduced


class GlobalInnerWriterTests(unittest.TestCase):
    def test_four_ranks_two_outer_calls_sixteen_updates_write_32_rows(self):
        reduced = four_rank_metrics()
        reduced_before = copy.deepcopy(reduced)
        final = finalize_token_moments(reduced, dp_size=4)
        final_before = copy.deepcopy(final)
        assert reduced == reduced_before
        assert final["mirror/valid_tokens"] == 10
        assert final["mirror/importance_ratio/mean"] == 5.5
        assert math.isclose(final["mirror/importance_ratio/std"], math.sqrt(8.25))
        assert final["mirror/importance_ratio/min"] == 1
        assert final["mirror/importance_ratio/max"] == 10
        assert math.isclose(final["mirror/importance_ess"], 55**2 / 385)
        assert math.isclose(final["mirror/importance_ess_fraction"], 55**2 / (10 * 385))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inner.jsonl"
            metadata = {"epochs": 16, "mini_batch_size": 32, "clip_grad": 1.0}
            metadata_before = copy.deepcopy(metadata)
            returned = [[] for _ in range(4)]
            for outer in range(2):
                for inner in range(16):
                    # Nonzero ranks enter first, as is possible during live
                    # distributed execution; every rank keeps trainer metrics.
                    for rank in (3, 1, 2, 0):
                        caller_metrics = {**final, "perf/rank_marker": rank}
                        caller_before = copy.deepcopy(caller_metrics)
                        append_inner_record(
                            path,
                            caller_metrics,
                            outer_call=outer,
                            inner_step=inner,
                            metadata=metadata,
                            global_rank=rank,
                        )
                        assert caller_metrics == caller_before
                        returned[rank].append(caller_metrics)
            assert metadata == metadata_before
            assert final == final_before
            assert [len(rows) for rows in returned] == [32] * 4
            for rank, rows in enumerate(returned):
                assert all(row["perf/rank_marker"] == rank for row in rows)
                assert all(row["loss"] == 0.25 for row in rows)
                assert all(row["mirror/valid_tokens"] == 10 for row in rows)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            expected = [(outer, inner) for outer in range(2) for inner in range(16)]
            assert len(records) == 32
            assert [(r["outer_call"], r["inner_step"]) for r in records] == expected
            assert len({(r["outer_call"], r["inner_step"]) for r in records}) == 32
            for record in records:
                assert record["measurement"] == "pre-update"
                assert record["metrics"]["mirror/valid_tokens"] == 10
                assert record["metrics"]["mirror/importance_ratio/mean"] == 5.5
                assert record["metrics"]["grad_clip_scale"] == 0.5
                assert record["metrics"]["grad_was_clipped"] == 1.0
                assert "perf/rank_marker" not in record["metrics"]

    def test_nonzero_ranks_do_not_touch_path_or_serialize_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "must-not-be-created" / "inner.jsonl"
            # A nonwriter should return before path creation or serialization;
            # this value cannot be serialized by a writing worker.
            metrics = {"mirror/not_serializable": object()}
            for rank in (1, 2, 3):
                append_inner_record(path, metrics, outer_call=0, inner_step=0, global_rank=rank)
            assert not path.parent.exists()
            assert "mirror/not_serializable" in metrics

    def test_explicit_rank_zero_and_default_single_process_write(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inner.jsonl"
            metrics = {"mirror/valid_tokens": 7}
            append_inner_record(path, metrics, outer_call=0, inner_step=0, global_rank=0)
            append_inner_record(path, metrics, outer_call=0, inner_step=1)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert [row["inner_step"] for row in records] == [0, 1]
            assert metrics == {"mirror/valid_tokens": 7}


if __name__ == "__main__":
    unittest.main()
