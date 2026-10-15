"""One portable lifecycle for configuration, source freeze, resources, training and audit."""

import json
import os
import signal
import sys
import time
import uuid
from pathlib import Path

from mirroropd.integrations.verl.compiler import compile_backend
from mirroropd.runtime.artifacts import atomic_json
from mirroropd.runtime.audit import audit_run
from mirroropd.runtime.control import controller_identity
from mirroropd.runtime.events import EventReader, EventWriter
from mirroropd.runtime.experiments import execution_identity
from mirroropd.runtime.preflight import prepare_inputs
from mirroropd.runtime.processes import launch
from mirroropd.runtime.provenance import snapshot_sources, verify_backend
from mirroropd.runtime.reservation import GpuReservation


def public_environment(job, directory, scratch):
    directory, scratch = Path(directory), Path(scratch)
    identity = execution_identity(directory, job)
    return {
        "MIRROR_RUN_DIR": str(directory),
        "MIRROR_JOB_PATH": str(directory / "job.json"),
        "MIRROR_RUN_ID": identity["run_id"],
        "MIRROR_CONFIG_ID": job.config_id,
        "MIRROR_EVENT_DIR": str(directory / "events"),
        "VLLM_USE_V1": "1",
        "USE_LIBUV": "0",
        "DIST_INIT_METHOD": "file://" + str(scratch / "c10d.store"),
        "RAY_ADDRESS": "local",
        "RAY_TMPDIR": str(scratch / "r"),
        "TMPDIR": str(scratch / "t"),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(directory / "source"),
        "VERL_FILE_LOGGER_PATH": str(directory / "backend-metrics.jsonl"),
        "WANDB_MODE": job.logging.wandb.effective_mode,
        "WANDB_RUN_ID": identity["run_id"],
        "WANDB_NAME": identity["name"],
        "WANDB_RUN_GROUP": identity["group"],
        "WANDB_TAGS": ",".join(
            dict.fromkeys(
                [job.objective.algorithm, f"seed:{job.training.seed}", *job.logging.wandb.tags]
            )
        ),
        "WANDB_RESUME": "never",
        "WANDB_DIR": str(directory / "wandb"),
        "WANDB_CACHE_DIR": str(directory / "wandb-cache"),
        "WANDB_DATA_DIR": str(directory / "wandb-data"),
    }


def _show(events, console):
    if not console:
        return
    for event in events:
        value = event["payload"]
        if event["event"] == "phase.started":
            print(f"[{event['source']}] {value['phase']} started", flush=True)
        elif event["event"] == "phase.completed":
            print(
                f"[{event['source']}] {value['phase']} {value['duration_seconds']:.2f}s", flush=True
            )
        elif event["event"] == "optimization.step":
            norm = value["metrics"].get("optimization/gradient_norm")
            print(
                f"update={value['update'] + 1} outer={value['outer']} inner={value['inner']} grad_norm={norm}",
                flush=True,
            )


def _check_exit(tree, directory):
    if tree.process.returncode:
        raise RuntimeError(
            f"Native training exited with {tree.process.returncode}; see {directory / 'backend.log'}"
        )


def run_job(
    job,
    *,
    backend,
    directory,
    python=sys.executable,
    lease_root=None,
    borrowed_slot=None,
    prepare_only=False,
    directory_created=False,
):
    directory = Path(directory).expanduser().resolve()
    if not directory_created:
        directory.mkdir(parents=True, exist_ok=False)
    identity = execution_identity(directory, job)
    run_id = identity["run_id"]
    requested_config_id = job.config_id
    atomic_json(directory / "requested-job.json", job.to_dict())
    backend = Path(backend).expanduser().resolve()
    package = Path(__file__).resolve().parents[1]
    integration = package / "integrations/verl"
    upstream = json.loads((integration / "backend.json").read_text())
    writer = EventWriter(directory / "events", run_id, "controller")
    state_path = directory / "state.json"
    source = None
    tree = None
    previous_handlers = {}

    def state(name, **details):
        atomic_json(
            state_path,
            {
                "schema_version": 2,
                "run_id": run_id,
                "state": name,
                **controller_identity(),
                "updated_unix": time.time(),
                **details,
            },
        )

    def interrupted(signum, _frame):
        raise InterruptedError(f"Controller received signal {signum}")

    try:
        state("preparing")
        writer.emit("run.started", configuration=job.to_dict())
        with writer.phase("backend.verify"):
            backend_record = verify_backend(
                backend, integration / upstream["patch"], upstream["base_commit"]
            )
        with writer.phase("inputs.prepare"):
            job, inputs = prepare_inputs(
                job, directory, fingerprint_cache=Path.home() / ".cache/mirroropd/fingerprints.json"
            )
        with writer.phase("source.snapshot"):
            source = snapshot_sources(package, backend, directory / "source")
        scratch = Path.home() / ".cache/mopd" / ("j" + uuid.uuid4().hex[:8])
        scratch.mkdir(parents=True, exist_ok=False)
        for name in ("r", "t"):
            (scratch / name).mkdir()
        environment = public_environment(job, directory, scratch)
        for name in ("wandb", "wandb-cache", "wandb-data"):
            (directory / name).mkdir()
        with writer.phase("backend.compile"):
            config = compile_backend(job, inputs, directory, directory / "source", environment)
        atomic_json(
            directory / "run.json",
            {
                "schema_version": 2,
                "run_id": run_id,
                "identity": identity,
                "requested_config_id": requested_config_id,
                "config_id": job.config_id,
                "source_id": source["source_id"],
                "backend": backend_record,
                "scratch": str(scratch),
            },
        )
        if prepare_only:
            state("prepared", config_id=job.config_id, source_id=source["source_id"])
            writer.emit("run.prepared", config_id=job.config_id)
            return None
        lease_root = Path(lease_root) if lease_root else Path.home() / ".cache/mirroropd/gpu-leases"
        child_environment = (
            os.environ
            | environment
            | {
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, job.resources.gpu_ids)),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            }
        )
        if job.logging.wandb.entity:
            child_environment["WANDB_ENTITY"] = job.logging.wandb.entity
        if job.logging.wandb.group:
            child_environment["WANDB_RUN_GROUP"] = job.logging.wandb.group
        command = [
            str(python),
            "-m",
            "verl.trainer.main_ppo",
            "--config-path",
            str(directory),
            "--config-name",
            config.stem,
        ]
        atomic_json(directory / "command.json", command)
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        reader = EventReader(directory / "events")
        with GpuReservation(
            job.resources.gpu_ids, lease_root, run_id, borrowed_slot=borrowed_slot
        ) as reservation:
            if reservation.borrowed_parent:
                atomic_json(directory / "borrowed-reservation.json", reservation.borrowed_parent)
            with (
                (directory / "backend.log").open("w") as log,
                (directory / "gpu.jsonl").open("w") as gpu_log,
            ):
                tree = launch(
                    command, cwd=directory / "source", environment=child_environment, log=log
                )
                state(
                    "running",
                    child_pid=tree.process.pid,
                    config_id=job.config_id,
                    source_id=source["source_id"],
                )
                while tree.process.poll() is None:
                    tree.observe()
                    reservation.heartbeat()
                    from mirroropd.runtime.reservation import gpu_snapshot

                    gpu_log.write(
                        json.dumps(
                            {
                                "time_unix": time.time(),
                                "devices": gpu_snapshot(job.resources.gpu_ids),
                            }
                        )
                        + "\n"
                    )
                    gpu_log.flush()
                    _show(reader.read(), job.logging.console)
                    time.sleep(1)
                atomic_json(
                    directory / "backend-exit.json", {"return_code": tree.process.returncode}
                )
                tree.terminate(timeout=5)
                _check_exit(tree, directory)
                _show(reader.read(), job.logging.console)
                state("verifying", config_id=job.config_id)
                with writer.phase("artifacts.audit"):
                    result = audit_run(directory)
                atomic_json(directory / "verification.json", result)
                reservation.assert_idle()
        state("completed", optimizer_updates=result["optimizer_updates"], config_id=job.config_id)
        writer.emit(
            "run.completed",
            verification="verification.json",
            optimizer_updates=result["optimizer_updates"],
        )
    except BaseException as error:
        if tree is not None:
            tree.terminate(timeout=5)
        state(
            "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        writer.emit("run.failed", error_type=type(error).__name__, error=str(error))
        raise
    else:
        return result
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
