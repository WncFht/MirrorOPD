"""Portable scorer registry; importing this module starts no CUDA or framework."""

import importlib
import importlib.metadata
import inspect
import math
from pathlib import Path

from mirroropd.identity import content_id
from mirroropd.runtime.artifacts import file_digest


def load_scorer(spec):
    if spec.name == "math":
        from mirroropd.scoring.mathematics import score_math

        function = score_math
    elif spec.name == "gsm8k":
        from mirroropd.scoring.numeric import score_numeric

        function = score_numeric
    elif spec.name == "exact":
        function = score_exact
    else:
        module_name, function_name = spec.name.split(":", 1)
        function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError("Configured scorer must be callable")
    inspect.signature(function).bind(
        "source", "response", "truth", extra_info=None, **spec.parameters
    )
    return function


def score_exact(data_source, solution_str, ground_truth, extra_info=None):
    del data_source, extra_info
    return {"score": float(solution_str.strip() == str(ground_truth).strip())}


def score(spec, data_source, solution_str, ground_truth, extra_info=None):
    result = load_scorer(spec)(
        data_source, solution_str, ground_truth, extra_info=extra_info, **spec.parameters
    )
    if not isinstance(result, dict):
        result = {"score": result}
    value = float(result["score"])
    if not math.isfinite(value) or (spec.kind == "binary" and value not in {0.0, 1.0}):
        raise ValueError("Scorer returned a value outside its declared domain")
    return result | {"score": value}


def scorer_identity(spec):
    root = Path(__file__).parent
    if spec.name in {"math", "gsm8k", "exact"}:
        files = [root / "__init__.py"]
        if spec.name == "math":
            files += [root / "mathematics.py", *sorted((root / "vendor").rglob("*.py"))]
        elif spec.name == "gsm8k":
            files.append(root / "numeric.py")
        records = [
            {"path": str(path.relative_to(root)), "sha256": file_digest(path)} for path in files
        ]
    else:
        function = load_scorer(spec)
        path = Path(importlib.import_module(function.__module__).__file__)
        records = [{"path": path.name, "sha256": file_digest(path)}]
    versions = {}
    if spec.name == "math":
        for name in ("math-verify", "sympy", "latex2sympy2-extended", "pylatexenc"):
            versions[name] = importlib.metadata.version(name)
    return content_id(
        {
            "name": spec.name,
            "kind": spec.kind,
            "parameters": spec.parameters,
            "files": records,
            "dependencies": versions,
        }
    )
