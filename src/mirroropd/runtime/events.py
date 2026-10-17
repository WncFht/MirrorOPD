"""Versioned append-only event streams, phase spans and incremental readers."""

import contextlib
import json
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from mirroropd.core.config import API_VERSION
from mirroropd.identity import canonical_json
from mirroropd.runtime.artifacts import atomic_json

_PRIVATE_FIELDS = frozenset(
    {"api_key", "access_token", "refresh_token", "password", "secret", "authorization"}
)


def public_fields(value):
    if isinstance(value, dict):
        return {
            key: "[redacted]" if key.lower() in _PRIVATE_FIELDS else public_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [public_fields(item) for item in value]
    return value


def process_rss_bytes():
    path = Path("/proc/self/status")
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


class EventWriter:
    """One producer owns one stream; producer IDs separate restarts and workers."""

    def __init__(self, directory, run_id, source, *, context=None):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", source):
            raise ValueError("Event source must be a simple name")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.source = source
        self.context = public_fields(context or {})
        self.producer_id = f"{source}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.path = self.directory / (self.producer_id + ".jsonl")
        self.sequence = 0
        self._lock = threading.Lock()

    def emit(self, event, **payload):
        with self._lock:
            return self._emit(event, payload)

    def _emit(self, event, payload):
        sequence = self.sequence + 1
        record = {
            "schema_version": API_VERSION,
            "run_id": self.run_id,
            "source": self.source,
            "producer_id": self.producer_id,
            "sequence": sequence,
            "event_id": f"{self.producer_id}:{sequence}",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "context": self.context,
            "payload": public_fields(payload),
        }
        encoded = (canonical_json(record) + "\n").encode()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            view = memoryview(encoded)
            while view:
                view = view[os.write(descriptor, view) :]
        finally:
            os.close(descriptor)
        self.sequence = sequence
        return record

    @contextlib.contextmanager
    def phase(self, name, *, profile_memory=False, **coordinates):
        from mirroropd.runtime.memory import memory_snapshot

        # Profiling is opt-in and applies only to this producer's initialized
        # CUDA context. Normal execution neither resets peaks nor synchronizes.
        torch = sys.modules.get("torch")
        profiling = bool(profile_memory and torch is not None and torch.cuda.is_initialized())
        if profiling:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        initial = memory_snapshot()
        self.emit(
            "phase.started",
            phase=name,
            memory=initial,
            rss_bytes=process_rss_bytes(),
            **coordinates,
        )
        try:
            yield
        except BaseException as error:
            self.emit(
                "phase.failed",
                phase=name,
                duration_seconds=time.monotonic() - start,
                error_type=type(error).__name__,
                error=str(error),
                **coordinates,
            )
            raise
        else:
            if profiling:
                torch.cuda.synchronize()
            final = memory_snapshot()
            self.emit(
                "phase.completed",
                phase=name,
                duration_seconds=time.monotonic() - start,
                rss_bytes=process_rss_bytes(),
                memory=final,
                memory_scope="phase_peak" if profiling else "process_peak",
                allocated_delta_bytes=(
                    final.get("allocated_bytes", 0) - initial.get("allocated_bytes", 0)
                ),
                **coordinates,
            )


def environment_writer(source):
    directory = os.environ.get("MIRROR_EVENT_DIR")
    run_id = os.environ.get("MIRROR_RUN_ID")
    if not directory or not run_id:
        return None
    return _environment_writer(directory, run_id, source, os.environ.get("MIRROR_CONFIG_ID"))


@lru_cache(maxsize=16)
def _environment_writer(directory, run_id, source, config_id):
    return EventWriter(directory, run_id, source, context={"config_id": config_id})


@contextlib.contextmanager
def phase(name, *, source="driver", profile_memory=False, **coordinates):
    writer = environment_writer(source)
    if writer is None:
        yield
    else:
        with writer.phase(name, profile_memory=profile_memory, **coordinates):
            yield


class EventReader:
    """Read only complete new rows, preserving byte cursors between drains."""

    def __init__(self, directory, *, cursor_path=None):
        self.directory = Path(directory)
        self.cursor_path = Path(cursor_path) if cursor_path else None
        self.offsets = (
            json.loads(self.cursor_path.read_text())
            if self.cursor_path and self.cursor_path.exists()
            else {}
        )

    def read(self):
        records = []
        for path in sorted(self.directory.glob("*.jsonl")):
            offset = self.offsets.get(path.name, 0)
            if path.stat().st_size < offset:
                raise ValueError(f"Append-only event stream was truncated: {path}")
            with path.open("rb") as stream:
                stream.seek(offset)
                while row := stream.readline():
                    if not row.endswith(b"\n"):
                        break
                    record = json.loads(row)
                    if record.get("schema_version") != API_VERSION:
                        raise ValueError("Unsupported event schema")
                    records.append(record)
                    offset = stream.tell()
            self.offsets[path.name] = offset
        if self.cursor_path:
            atomic_json(self.cursor_path, self.offsets)
        return sorted(records, key=lambda record: (record["monotonic_ns"], record["event_id"]))
