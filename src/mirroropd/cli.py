"""One command line for model/data profiles, job execution, status and verification."""

import argparse
import json
import sys
from pathlib import Path

from mirroropd import __version__
from mirroropd.core.config import ALGORITHMS
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.audit import audit_run
from mirroropd.runtime.config import Job
from mirroropd.runtime.control import run_status, start_run, stop_run, wait_run
from mirroropd.runtime.experiments import run_budget
from mirroropd.runtime.profiles import Selection, profile

__all__ = ["main", "profile"]


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
    parser.add_argument("--task")
    parser.add_argument("--recipe")
    parser.add_argument("--models")
    parser.add_argument("--dataset")
    parser.add_argument("--evaluation")
    parser.add_argument("--var", action="append", default=[])
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--name")
    parser.add_argument("--wandb", choices=["online", "offline", "disabled"])


def load_arguments(args):
    job = Selection(
        task=args.task,
        recipe=args.recipe,
        models=args.models,
        dataset=args.dataset,
        evaluation=args.evaluation,
        variables=variables(args.var),
        overrides=args.overrides,
        algorithm=args.algorithm,
    ).load()
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


def launch_arguments(parser):
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--lease-root", type=Path)
    parser.add_argument("--borrowed-slot", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--detach", action="store_true", help="Run in background; write controller.log"
    )


def launch_job(args, job, *, study=None):
    child, record = start_run(
        job,
        backend=args.backend,
        directory=args.output,
        python=args.python,
        algorithms=args.algorithms if args.command == "suite" else None,
        lease_root=args.lease_root,
        borrowed_slot=args.borrowed_slot,
        prepare_only=args.prepare_only,
        study=study,
    )
    if args.detach:
        print(json.dumps(record, indent=2))
        return 0
    print(json.dumps({"directory": record["directory"], "run_id": record["run_id"]}), flush=True)
    return wait_run(child, record["directory"])


def study_command(args):
    from mirroropd.runtime.study import plan_study, retry_study, summarize_study

    if args.study_command == "status":
        print(json.dumps(summarize_study(args.directory), indent=2))
        return 0
    plan = (
        retry_study(args.directory)
        if args.study_command == "retry"
        else plan_study(args.config, variables=variables(args.var), overrides=args.overrides)
    )
    if args.study_command == "plan":
        if args.output:
            atomic_json(args.output, plan)
        print(json.dumps(plan, indent=2))
        return 0
    return launch_job(args, Job.from_dict(plan["trials"][0]["job"]), study=plan)


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
        launch_arguments(run)
        if name == "suite":
            run.add_argument(
                "--algorithms",
                nargs="+",
                choices=sorted(ALGORITHMS),
                default=["on_policy", "mdpo", "bpo"],
            )
    study = commands.add_parser(
        "study", help="Plan, execute, summarize or retry a small parameter grid"
    )
    study_commands = study.add_subparsers(dest="study_command", required=True)
    for name in ("plan", "run", "retry", "status"):
        command = study_commands.add_parser(name)
        if name in {"plan", "run"}:
            command.add_argument("--config", required=True)
            command.add_argument("--var", action="append", default=[])
            command.add_argument("--set", dest="overrides", action="append", default=[])
        else:
            command.add_argument("directory", type=Path)
        if name in {"run", "retry"}:
            launch_arguments(command)
        elif name == "plan":
            command.add_argument("--output", type=Path)
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
        if args.command == "study":
            return study_command(args)
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
            value = {"config_id": job.config_id, "job": job.to_dict(), "budget": run_budget(job)}
            if args.output:
                atomic_json(args.output, value)
            print(json.dumps(value, indent=2))
            return 0
        return launch_job(args, job)
    except (ValueError, TypeError, RuntimeError, OSError) as error:
        print(f"mirroropd: {error}", file=sys.stderr)
        return 1
    return 0
