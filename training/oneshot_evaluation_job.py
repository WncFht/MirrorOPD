"""Launch four math replicas as one serialized, owned evaluation job."""

import argparse
import json
import os
import socket
import subprocess
from pathlib import Path


def main():
    """Wait for every replica and require all 570 unique question outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owners", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    processes = []
    streams = []
    reservations = []
    ports = [0] * 4
    for rank in range(4):
        reservation = socket.socket()
        reservation.bind(("127.0.0.1", 0))
        reservations.append(reservation)
        ports[rank] = reservation.getsockname()[1]
    for reservation in reservations:
        reservation.close()
    for rank in range(4):
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(rank),
            VLLM_USE_V1="1",
            VLLM_ENABLE_V1_MULTIPROCESSING="0",
            VLLM_WORKER_MULTIPROC_METHOD="spawn",
            VLLM_PORT=str(ports[rank]),
            PYTHONPATH=str(args.source)
            + ":"
            + str(args.source / "src")
            + ":"
            + str(args.repo.parent / "backends/mirroropd-taopd/verl"),
        )
        for variable in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE"):
            environment.pop(variable, None)
        stream = (args.output / f"rank-{rank}.log").open("w")
        streams.append(stream)
        processes.append(
            subprocess.Popen(  # noqa: S603 -- trusted evaluator argv
                [
                    str(args.repo.parent / "envs/opd/bin/python"),
                    str(args.source / "experiments/scoring/oneshot_math_evaluate.py"),
                    "--model",
                    str(args.model),
                    "--repo",
                    str(args.repo),
                    "--output",
                    str(args.output),
                    "--owners",
                    str(args.owners),
                    "--rank",
                    str(rank),
                ],
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        )
    try:
        for process in processes:
            code = process.wait()
            if code:
                raise RuntimeError(f"Math replica exited {code}; inspect rank logs")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for stream in streams:
            stream.close()
    rows = [
        json.loads(line)
        for rank in range(4)
        for line in (args.output / f"rank-{rank}.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 570
    assert len({row["uid"] for row in rows}) == 570
    summary = {}
    for task, expected in (("MATH-500", 500), ("AMC23", 40), ("AIME25", 30)):
        selected = [row for row in rows if row["task"] == task]
        assert len(selected) == expected
        summary[task] = {
            "questions": expected,
            "correct": sum(row["correct"] for row in selected),
            "accuracy": sum(row["correct"] for row in selected) / expected,
            "budget_exhausted_fraction": sum(row["budget_exhausted"] for row in selected)
            / expected,
        }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "completion.json").write_text(
        json.dumps({"state": "completed", "unique_questions": 570, "model": str(args.model)}) + "\n"
    )


if __name__ == "__main__":
    main()
