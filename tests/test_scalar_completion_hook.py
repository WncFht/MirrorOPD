"""Exit-event waiting and honest completion markers without launching GPU work."""

import json
import os
import subprocess
import sys

from training.scalar_completion_hook import audit_job, finish, open_pidfd, wait_for_exit


def test_exit_event_is_ready_after_child_finishes():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    descriptor = open_pidfd(child.pid)
    try:
        wait_for_exit(descriptor)
        assert child.wait(timeout=5) == 0
    finally:
        os.close(descriptor)


def test_summary_alone_does_not_prove_full_pool_completion(tmp_path):
    summary = {
        "status": "complete",
        "config": {"updates": 4, "full_batch": True, "train_sequences": 64},
        "counts": {"backwards": 255},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "audits.jsonl").write_text("\n".join(json.dumps({"update": x}) for x in (0, 4)))
    assert (
        audit_job(tmp_path, {"updates": 4, "full_batch": True, "train_sequences": 64})
        == "wrong_full_pool_backward_count"
    )
    summary["counts"]["backwards"] = 256
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert (
        audit_job(tmp_path, {"updates": 4, "full_batch": True, "train_sequences": 64}) == "complete"
    )


def test_queue_exit_with_missing_artifacts_requires_review(tmp_path):
    output = tmp_path / "hook"
    result = finish(tmp_path, output)
    assert result["status"] == "needs_review"
    assert result["issues"]
    assert (output / "completion.json").exists()
    assert (output / "ready.md").exists()


def test_user_cancelled_confirmations_are_not_expected(tmp_path):
    from training.scalar_completion_hook import expected_jobs
    from training.scalar_large_suite import base_config

    study = tmp_path / "results/scalar-large/fullpool-lr-study"
    study.mkdir(parents=True)
    (study / "selection.json").write_text(
        json.dumps({"baselines": [{"id": "chosen", "lr": 1e-6}, {"id": "second", "lr": 5e-7}]})
    )
    old = tmp_path / "runs/scalar-large"
    old.mkdir(parents=True)
    config = base_config()
    (old / "selection.json").write_text(
        json.dumps(
            {
                "baseline_configs": [config, config | {"lr": 5e-7}],
                "feedback_target": 0.01,
                "primary_scalar_config": config | {"method": "feedback"},
            }
        )
    )
    assert {"fullconfirm", "confirm"} <= expected_jobs(tmp_path).keys()
    (study / "user-amendment.json").write_text(
        json.dumps({"skip_fullconfirm": True, "skip_remaining_old_confirm": True})
    )
    jobs = expected_jobs(tmp_path)
    assert "fullconfirm" not in jobs
    assert "confirm" not in jobs
    assert {"fullpilot", "fullstage", "fulllong", "ablate"} <= jobs.keys()
    assert len(jobs["ablate"]) == 17
