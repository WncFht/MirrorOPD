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
from mirroropd.runtime.control import run_status, start_run, stop_run, wait_run


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
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"])


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
    if args.wandb:
        value = job.to_dict()
        value["logging"]["wandb"]["enabled"] = args.wandb != "disabled"
        if args.wandb != "disabled":
            value["logging"]["wandb"]["mode"] = args.wandb
        job = type(job).from_dict(value)
    return job


def main(argv=None):  # noqa: PLR0911 -- each CLI subcommand returns its own exit status
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
        run.add_argument(
            "--detach", action="store_true", help="Run in background; write controller.log"
        )
        if name == "suite":
            run.add_argument(
                "--algorithms",
                nargs="+",
                choices=sorted(ALGORITHMS),
                default=["on_policy", "mdpo", "bpo"],
            )
    for name in ("status", "audit", "stop"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        if name == "stop":
            command.add_argument("--timeout", type=float, default=30)
    pack = commands.add_parser(
        "pack", help="Create a checksummed, optionally split offline package"
    )
    pack.add_argument("kind", choices=["code", "results", "wandb"])
    pack.add_argument("--backend", type=Path)
    pack.add_argument("--run", type=Path)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--chunk-bytes", type=int, default=10 * 1024**2)
    unpack = commands.add_parser("unpack")
    unpack.add_argument("manifest", type=Path)
    unpack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "backend":
            from mirroropd.integrations.verl.bootstrap import prepare_backend

            print(json.dumps(prepare_backend(args.output), indent=2))
            return 0
        if args.command == "status":
            print(json.dumps(run_status(args.directory), indent=2))
            return 0
        if args.command == "stop":
            print(json.dumps(stop_run(args.directory, timeout=args.timeout), indent=2))
            return 0
        if args.command in {"pack", "unpack"}:
            from mirroropd.runtime.transfer import pack_transfer, unpack_transfer

            result = (
                pack_transfer(
                    args.kind,
                    output=args.output,
                    backend=args.backend,
                    run=args.run,
                    chunk_bytes=args.chunk_bytes,
                )
                if args.command == "pack"
                else unpack_transfer(args.manifest, args.output)
            )
            print(json.dumps(result, indent=2))
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
        child, record = start_run(
            job,
            backend=args.backend,
            directory=args.output,
            python=args.python,
            algorithms=args.algorithms if args.command == "suite" else None,
            lease_root=args.lease_root,
            borrowed_slot=args.borrowed_slot,
            prepare_only=args.prepare_only,
        )
        if args.detach:
            print(json.dumps(record, indent=2))
        else:
            return wait_run(child, args.output)
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"mirroropd: {error}", file=sys.stderr)
        return 1
    return 0
