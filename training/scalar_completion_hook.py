"""Wait for the scalar queue's exit event, then audit artifacts and generate CPU reports."""

import argparse
import ctypes
import json
import os
import platform
import select
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

from training.scalar_gpu_lease import process_identity


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def expected_jobs(root):
    from training.scalar_full_pool_study import LRS, base, confirmation, forks, pilot_configs
    from training.scalar_large_suite import ablation_configs, confirmation_configs

    selection = json.loads(
        (root / "results/scalar-large/fullpool-lr-study/selection.json").read_text()
    )
    old = json.loads((root / "runs/scalar-large/selection.json").read_text())
    pilots = pilot_configs()
    if (
        any(item["lr"] > max(LRS) for item in selection["baselines"])
        or (root / "runs/scalar-large/configs/fullpilot-lr3.2e-05.json").exists()
    ):
        pilots += pilot_configs((3.2e-5, 6.4e-5))
    rates = sorted({item["lr"] for item in selection["baselines"]} | {1e-6, 2e-6, 4e-6})
    long = base(42) | {"cache": "runs/scalar-large/cache-long-s42", "updates": 80}
    return {
        "fullpilot-canary": [("fullpilot-canary", base() | {"updates": 4, "lr": 5e-7})],
        "fullpilot": pilots,
        "fullstage": forks(selection),
        "fullconfirm": confirmation(selection),
        "confirm": confirmation_configs(old),
        "fulllong-canary": [("fulllong-canary", long | {"updates": 1, "lr": 5e-7})],
        "fulllong": [(f"fulllong-lr{lr:.3g}", long | {"lr": lr}) for lr in rates],
        "ablate": ablation_configs(old),
    }


def audit_summary(summary, expected):
    if summary["status"] != "complete":
        return summary["status"]
    if summary["config"]["updates"] != expected["updates"]:
        return "wrong_budget"
    for key in (
        "seed",
        "lr",
        "eta",
        "precision",
        "objective",
        "train_sequences",
        "batch_sequences",
        "topk",
        "method",
        "search_every",
        "target_rate",
    ):
        if key in expected and summary["config"].get(key) != expected[key]:
            return f"wrong_setting: {key}"
    if expected.get("full_batch"):
        if not summary["config"].get("full_batch"):
            return "wrong_batch_mode"
        if summary["counts"]["backwards"] != expected["train_sequences"] * expected["updates"]:
            return "wrong_full_pool_backward_count"
    return "complete"


def audit_job(directory, expected):
    path = directory / "summary.json"
    if not path.exists():
        return "missing_summary"
    status = audit_summary(json.loads(path.read_text()), expected)
    if status != "complete":
        return status
    points = [json.loads(line) for line in (directory / "audits.jsonl").read_text().splitlines()]
    if not points or points[0]["update"] != 0 or points[-1]["update"] != expected["updates"]:
        return "incomplete_audits"
    if any(b["update"] <= a["update"] for a, b in pairwise(points)):
        return "invalid_audit_order"
    return "complete"


def finish(root, output):
    failures = []
    rows = []
    try:
        for phase, configurations in expected_jobs(root).items():
            for name, config in configurations:
                try:
                    status = audit_job(root / "runs/scalar-large" / phase / name, config)
                except (KeyError, ValueError, OSError) as error:
                    status = f"invalid_record: {error}"
                row = {"phase": phase, "run": name, "status": status}
                rows.append(row)
                if status != "complete":
                    failures.append(row)
        marker = root / "results/scalar-large/fullpool-lr-study/long-transfer-complete.json"
        if not marker.exists() or json.loads(marker.read_text()).get("status") != "finished":
            failures.append({"phase": "fulllong", "status": "missing_long_transfer_completion"})
    except (KeyError, ValueError, OSError) as error:
        failures.append({"phase": "queue", "status": f"missing_or_invalid_plan: {error}"})

    from experiments.analysis.scalar_full_pool_report import report
    from experiments.analysis.scalar_large_report import summarize

    reports = root / "results/scalar-large/completion-reports"
    for phase in ("fullpilot", "fullstage", "fullconfirm", "confirm", "fulllong", "ablate"):
        try:
            selection_path = (
                root / "runs/scalar-large/selection.json" if phase == "confirm" else None
            )
            summarize(root / "runs/scalar-large" / phase, reports / phase, selection_path)
            if phase == "fullconfirm":
                selection = json.loads(
                    (root / "results/scalar-large/fullpool-lr-study/selection.json").read_text()
                )
                report(
                    root / "runs/scalar-large/fullconfirm",
                    reports / phase / "comparisons",
                    selection,
                )
        except (KeyError, ValueError, OSError) as error:
            failures.append({"phase": phase, "status": f"report_error: {error}"})
    record = {
        "status": "needs_review" if failures else "experiments_complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runs": rows,
        "issues": failures,
        "reports": str(reports),
        "scope": "experiment queue artifacts; research goal still requires final analysis",
    }
    write_json(output / "completion.json", record)
    (output / "ready.md").write_text(
        "# BPO queue finished\n\n"
        f"Status: {record['status']}.\n\n"
        f"Checked {len(rows)} runs; {len(failures)} issues require review.\n\n"
        f"CPU reports: {reports}.\n\n"
        "See completion.json for the complete inventory. No GPU job was launched or restarted.\n"
    )
    return record


def wait_for_exit(pidfd):
    select.select([pidfd], [], [])


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    # Linux x86_64 and asm-generic headers define __NR_pidfd_open as 434.
    # The current Python/glibc build lacks the wrapper; the running kernel supports it.
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "aarch64"):
        raise RuntimeError("pidfd requires a supported Linux architecture")
    syscall = ctypes.CDLL(None, use_errno=True).syscall
    syscall.argtypes = (ctypes.c_long, ctypes.c_int, ctypes.c_uint)
    syscall.restype = ctypes.c_long
    descriptor = syscall(434, pid, 0)
    if descriptor < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return descriptor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    identity = process_identity(args.pid)
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes()
    if identity is None or b"training.scalar_queue_handoff" not in command:
        raise RuntimeError("hook requires the scalar queue supervisor")
    descriptor = open_pidfd(args.pid)
    try:
        if process_identity(args.pid) != identity:
            raise RuntimeError("supervisor identity changed while attaching")
        write_json(
            args.output / "armed.json",
            {
                "status": "waiting_for_exit_event",
                "supervisor_pid": args.pid,
                "supervisor_identity": identity,
                "hook_pid": os.getpid(),
                "waiting": "Linux pidfd; no periodic process or GPU polling",
            },
        )
        wait_for_exit(descriptor)
    finally:
        os.close(descriptor)
    print(json.dumps(finish(args.root, args.output)), flush=True)


if __name__ == "__main__":
    main()
