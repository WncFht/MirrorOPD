"""Smoke-test an installed wheel from outside the source checkout."""

import importlib.util
import json
import subprocess
import sys
from importlib.resources import files
from pathlib import Path

import mirroropd


def main():
    origin = Path(mirroropd.__file__).resolve()
    if "site-packages" not in origin.parts:
        raise SystemExit(f"Expected an installed wheel, imported {origin}")
    package = files("mirroropd")
    for category in ("models", "datasets", "recipes", "evaluations", "tasks", "studies"):
        profiles = list((package / "profiles" / category).iterdir())
        if not profiles:
            raise SystemExit(f"Missing packaged profiles: {category}")
        for profile in profiles:
            if profile.name.endswith(".json"):
                json.loads(profile.read_text())
    for name in (
        "integrations/verl/backend.json",
        "integrations/verl/backend.patch",
        "scoring/vendor/LICENSE",
        "scoring/vendor/NOTICE",
        "scoring/vendor/origin.json",
    ):
        if not (package / name).read_bytes():
            raise SystemExit(f"Empty packaged resource: {name}")
    if importlib.util.find_spec("mirroropd.training") is not None:
        raise SystemExit("Retired training package is present in the wheel")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mirroropd",
            "plan",
            "--recipe",
            "math-four-gpu",
            "--models",
            "deepseek-justrl",
            "--dataset",
            "dapo-math",
            "--var",
            "model_root=/models",
            "--var",
            "data_root=/data",
            "--algorithm",
            "on_policy",
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mirroropd",
            "study",
            "plan",
            "--config",
            "smoke-grid",
            "--var",
            "model_root=/models",
            "--var",
            "data_root=/data",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"Installed wheel resources and CLI verified: {origin}")


if __name__ == "__main__":
    main()
