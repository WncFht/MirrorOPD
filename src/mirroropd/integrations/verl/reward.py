"""Single backend scorer entry; the job selects the dataset-specific implementation."""

import json
import os
from functools import lru_cache
from pathlib import Path

from mirroropd.runtime.config import Job
from mirroropd.runtime.events import environment_writer
from mirroropd.scoring import score


@lru_cache(maxsize=1)
def configured_job():
    path = os.environ.get("MIRROR_JOB_PATH")
    if not path:
        raise ValueError("Scoring requires an explicit MIRROR_JOB_PATH")
    return Job.from_dict(json.loads(Path(path).read_text()))


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del kwargs
    extra = extra_info or {}
    if extra.get("split") == "train" and not extra.get("has_reference", True):
        return {"score": 0.0, "reference_available": 0.0}
    try:
        result = score(
            configured_job().data.scorer, data_source, solution_str, ground_truth, extra_info=extra
        )
    except Exception as error:
        writer = environment_writer("scorer")
        if writer:
            writer.emit(
                "scoring.failed",
                data_source=data_source,
                question_uid=extra.get("row_uid"),
                error_type=type(error).__name__,
                error=str(error),
            )
        raise
    return result | {"reference_available": float(extra.get("has_reference", True))}
