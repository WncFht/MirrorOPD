"""Wait for the normalization-only run; conditionally launch FP32-storage ablation.

The condition is a diagnostic screen, not a statistical significance test.
Runs share all algorithm and data settings except actor parameter storage.
Never change a running launch script or checkpoint weights.
"""

import argparse
import json
import os
import statistics
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    first = args.repo / "runs/gsm8k-smollm2-mdpo-o50-global-token-fixed"
    status_path = first / "followup-decision.json"
    while not (first / "exit-code.txt").exists():
        active = subprocess.run(
            ["/usr/bin/systemctl", "--user", "is-active", "mirroropd-gsm8k-normalized-o50"],
            capture_output=True,
            text=True,
            check=False,
        )
        if active.stdout.strip() not in ("active", "activating"):
            status_path.write_text(
                json.dumps({"state": "first_run_interrupted", "second_started": False}, indent=2)
                + "\n"
            )
            return
        time.sleep(30)
    code = int((first / "exit-code.txt").read_text())
    rows = (
        [json.loads(x) for x in (first / "outer.jsonl").read_text().splitlines()]
        if (first / "outer.jsonl").exists()
        else []
    )
    tests = [r for r in rows if "evaluation/generated_examples" in r["data"]]
    train = [r for r in rows if "timing_s/step" in r["data"]]
    result = {
        "first_exit_code": code,
        "second_started": False,
        "criterion": "Run FP32 if execution incomplete, accuracy does not improve, or teacher center does not decrease; diagnostic only.",
    }
    if code != 0 or len(train) != 50 or len(tests) != 11:
        result.update(
            state="first_incomplete", reason="Fix infrastructure before a long precision ablation"
        )
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        return
    key = "val-core/openai/gsm8k/acc/mean@1"
    center = "actor/mirror/teacher_center/mean"
    result.update(
        initial_accuracy=tests[0]["data"][key],
        final_accuracy=tests[-1]["data"][key],
        first10_teacher_center=statistics.mean(r["data"][center] for r in train[:10]),
        last10_teacher_center=statistics.mean(r["data"][center] for r in train[-10:]),
    )
    issue = (
        result["final_accuracy"] <= result["initial_accuracy"]
        or result["last10_teacher_center"] >= result["first10_teacher_center"]
    )
    if not issue:
        result.update(
            state="first_improved", reason="Both diagnostics improved; no automatic second run"
        )
        status_path.write_text(json.dumps(result, indent=2) + "\n")
        return
    second = args.repo / "runs/gsm8k-smollm2-mdpo-o50-global-token-fp32"
    if second.exists():
        raise FileExistsError(second)
    result.update(state="second_running", second_started=True, second_run=second.name)
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    env = dict(
        os.environ,
        WANDB_RUN_GROUP="gsm8k-normalization-ablation",
        WANDB_TAGS="mdpo,o50,j16,clip2,b64,fp32,global-token-fixed",
    )
    proc = subprocess.run(  # noqa: S603 -- fixed launcher, validated local repository path
        [
            "/usr/bin/bash",
            str(args.repo / "training/run_gsm8k_mdpo.sh"),
            second.name,
            "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
        ],
        env=env,
        check=False,
    )
    result.update(
        state="second_finished" if proc.returncode == 0 else "second_failed",
        second_exit_code=proc.returncode,
    )
    status_path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
