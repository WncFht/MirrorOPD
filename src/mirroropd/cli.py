"""One command line for model/data profiles, job execution, status and verification."""

import argparse
import json
import sys
from pathlib import Path

from mirroropd import __version__
from mirroropd.core.config import ALGORITHMS
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.audit import audit_run
from mirroropd.runtime.config import load_job, with_algorithm
from mirroropd.runtime.reservation import process_identity


def profile(value, category):
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    path = Path(__file__).parent / "profiles" / category / (value + ".json")
    if not path.is_file():
        raise ValueError(f"Unknown {category} profile: {value}")
    return path


def variables(items):
    result = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name:
            raise ValueError("Variables use name=value")
        if name in result:
            raise ValueError(f"Duplicate explicit variable: {name}")
        result[name] = value
    return result


def job_arguments(parser):
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--models")
    parser.add_argument("--dataset")
    parser.add_argument("--var", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--name")


def load_arguments(args):
    job = load_job(
        profile(args.recipe, "recipes"),
        models=profile(args.models, "models") if args.models else None,
        dataset=profile(args.dataset, "datasets") if args.dataset else None,
        variables=variables(args.var),
        overrides=args.overrides,
    )
    if args.algorithm:
        job = with_algorithm(job, args.algorithm)
    if args.name:
        value = job.to_dict()
        value["name"] = args.name
        job = type(job).from_dict(value)
    return job


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mirroropd", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    backend_command = commands.add_parser("backend")
    backend_command.add_argument("--output", type=Path, required=True)
    plan = commands.add_parser("plan")
    job_arguments(plan)
    plan.add_argument("--output", type=Path)
    for name in ("run", "suite"):
        run = commands.add_parser(name)
        job_arguments(run)
        run.add_argument("--backend", type=Path, required=True)
        run.add_argument("--output", type=Path, required=True)
        run.add_argument("--python", default=sys.executable)
        run.add_argument("--lease-root", type=Path)
        run.add_argument("--borrowed-slot", type=Path)
        run.add_argument("--prepare-only", action="store_true")
        if name == "suite":
            run.add_argument(
                "--algorithms",
                nargs="+",
                choices=sorted(ALGORITHMS),
                default=["on_policy", "mdpo", "bpo"],
            )
    for name in ("status", "audit"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "backend":
            from mirroropd.integrations.verl.bootstrap import prepare_backend

            print(json.dumps(prepare_backend(args.output), indent=2))
            return 0
        if args.command == "status":
            value = json.loads((args.directory / "state.json").read_text())
            if (
                value["state"] in {"running", "preparing", "verifying"}
                and process_identity(value["controller_pid"]) != value["process_identity"]
            ):
                value["state"] = "orphaned"
            print(json.dumps(value, indent=2))
            return 0
        if args.command == "audit":
            result = audit_run(args.directory)
            atomic_json(args.directory / "verification.json", result)
            print(json.dumps(result, indent=2))
            return 0
        job = load_arguments(args)
        if args.command == "plan":
            value = {"config_id": job.config_id, "job": job.to_dict()}
            if args.output:
                atomic_json(args.output, value)
            print(json.dumps(value, indent=2))
            return 0
        from mirroropd.runtime.runner import run_job

        if args.command == "suite":
            args.output.mkdir(parents=True, exist_ok=False)
            for algorithm in args.algorithms:
                run_job(
                    with_algorithm(job, algorithm),
                    backend=args.backend,
                    directory=args.output / algorithm,
                    python=args.python,
                    lease_root=args.lease_root,
                    borrowed_slot=args.borrowed_slot,
                    prepare_only=args.prepare_only,
                )
        else:
            run_job(
                job,
                backend=args.backend,
                directory=args.output,
                python=args.python,
                lease_root=args.lease_root,
                borrowed_slot=args.borrowed_slot,
                prepare_only=args.prepare_only,
            )
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"mirroropd: {error}", file=sys.stderr)
        return 1
    return 0
