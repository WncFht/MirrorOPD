"""Install the exact MD-OPD math verifier into the deployed Mirror backend."""

import hashlib
import json
import os
import shutil
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo.parent / "MD-OPD/verl/verl/utils/reward_score"
    backend = Path(os.environ.get("MIRROR_UPSTREAM_ROOT", repo.parent / "backends/mirroropd-taopd"))
    target = backend / "verl/verl/utils/reward_score"
    files = [source / "ttrl_mathverify.py", source / "_timeout.py"]
    files.extend(sorted((source / "ttrl_math").glob("*.py")))
    # Validate the entire set before modifying the backend.
    for file in files:
        destination = target / file.relative_to(source)
        if destination.exists() and destination.read_bytes() != file.read_bytes():
            raise ValueError(f"Existing verifier differs; refusing replacement: {destination}")
    if (source / "math_verify.py").read_bytes() != (target / "math_verify.py").read_bytes():
        raise ValueError("Math-Verify fallback differs from MD-OPD")
    records = []
    for file in files:
        destination = target / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(file, destination)
        records.append(
            {
                "source": str(file),
                "target": str(destination),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    output = repo / "results/training/math-4gpu/reward-integration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Verified {len(records)} math verifier modules")


if __name__ == "__main__":
    main()
