"""Exit-event waiting and honest completion markers without launching GPU work."""

import json
import os
import subprocess
import sys

from training.scalar_completion_hook import audit_job, finish, open_pidfd, wait_for_exit


def test_exit_event_is_ready_after_child_finishes():
    child = subprocess.Popen([sys.executable, "-c", "pass"])  # noqa: S603 -- fixed local Python child
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
