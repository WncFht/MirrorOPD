"""Independent eager KL derivatives audit the frozen external research source."""

import hashlib
import importlib.util
import json
import tarfile
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def research_reference(tmp_path_factory):
    root = Path(__file__).resolve().parents[2]
    evidence = root / "data/deepseek-opd-research-20261018"
    manifest = json.loads((evidence / "evidence-files.json").read_text())
    identities = {item["path"]: item["sha256"] for item in manifest["files"]}
    destination = tmp_path_factory.mktemp("nitrobrew-reference")
    with tarfile.open(root / manifest["archive"], "r:gz") as archive:
        for name in ("check-nitrobrew.py", "sources/nitrobrew/nitrobrew.py"):
            content = archive.extractfile(name).read()
            assert hashlib.sha256(content).hexdigest() == identities[name]
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    spec = importlib.util.spec_from_file_location(
        "nitrobrew_research_check", destination / "check-nitrobrew.py"
    )
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    reference = checker.load_reference(destination)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield checker, reference
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("temperature", [0.7, 1.0, 2.0])
@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_frozen_source_matches_independent_objective(
    research_reference, direction, temperature, chunk
):
    checker, reference = research_reference
    with torch.random.fork_rng(devices=[]):
        checker.check_case(reference, direction, temperature, chunk)
