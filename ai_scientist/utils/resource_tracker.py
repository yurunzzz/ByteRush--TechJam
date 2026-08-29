"""Process-safe competition resource accounting for AI Scientist runs.

The JSONL ledger is shared by the parent and ProcessPool workers.  Every event
is appended while holding an OS file lock, so LLM calls and executed candidates
are counted even when they happen in different Python processes.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


LEDGER_ENV = "AI_SCIENTIST_RESOURCE_LEDGER"
ITERATION_LIMIT_ENV = "AI_SCIENTIST_ITERATION_LIMIT"
DEFAULT_ITERATION_LIMIT = 50


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ledger_path() -> Path | None:
    value = os.environ.get(LEDGER_ENV)
    return Path(value) if value else None


def _append_event(event: dict[str, Any]) -> None:
    path = _ledger_path()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp_utc": _utc_now(), "pid": os.getpid(), **event}
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def record_llm_call(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    request_seconds: float | None = None,
    backend: str | None = None,
) -> None:
    """Record one completed LLM request without storing prompt contents."""
    _append_event(
        {
            "event": "llm_call",
            "model": str(model),
            "backend": backend,
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "total_tokens": int(prompt_tokens or 0) + int(completion_tokens or 0),
            "request_seconds": float(request_seconds or 0.0),
        }
    )


def reserve_iteration(
    *,
    stage: str | None,
    node_id: str | None,
    seed_evaluation: bool,
) -> int | None:
    """Atomically reserve a counted candidate execution.

    Seed replications are recorded but deliberately excluded from the search
    iteration count.  This makes the competition field unambiguous while still
    exposing the additional work in ``seed_evaluations``.
    """
    path = _ledger_path()
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    limit = int(os.environ.get(ITERATION_LIMIT_ENV, DEFAULT_ITERATION_LIMIT))
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        count = 0
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "candidate_execution" and not event.get(
                "seed_evaluation", False
            ):
                count += 1
        if not seed_evaluation and count >= limit:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            raise RuntimeError(
                f"competition iteration limit reached: {count}/{limit}"
            )
        iteration = None if seed_evaluation else count + 1
        payload = {
            "timestamp_utc": _utc_now(),
            "pid": os.getpid(),
            "event": "candidate_execution",
            "iteration": iteration,
            "iteration_limit": limit,
            "stage": stage,
            "node_id": node_id,
            "seed_evaluation": bool(seed_evaluation),
        }
        handle.seek(0, os.SEEK_END)
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return iteration


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
        for line in handle:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return events


def _query_gpu_models() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


class GPUUsageMonitor:
    """Sample GPU compute processes belonging to this run's process tree."""

    def __init__(self, root_pid: int, interval_seconds: float = 1.0):
        self.root_pid = root_pid
        self.interval_seconds = interval_seconds
        self.gpu_models = _query_gpu_models()
        self.sample_count = 0
        self.active_sample_count = 0
        self.active_seconds = 0.0
        self.peak_used_memory_mib = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_sample = time.monotonic()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name="gpu-usage-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._sample()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _run_pids(self) -> set[int]:
        try:
            root = psutil.Process(self.root_pid)
            return {root.pid, *(child.pid for child in root.children(recursive=True))}
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {self.root_pid}

    def _sample(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_sample
        self._last_sample = now
        self.sample_count += 1
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return
        run_pids = self._run_pids()
        active = False
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 2:
                continue
            try:
                pid, memory_mib = int(fields[0]), float(fields[1])
            except ValueError:
                continue
            if pid in run_pids:
                active = True
                self.peak_used_memory_mib = max(
                    self.peak_used_memory_mib, memory_mib
                )
        if active:
            self.active_sample_count += 1
            self.active_seconds += elapsed


class CompetitionResourceTracker:
    """Own the run wall clock, GPU monitor, and final summary artifact."""

    def __init__(
        self,
        run_dir: str | Path,
        iteration_limit: int = 50,
        started_at_utc: str | None = None,
        started_monotonic: float | None = None,
    ):
        self.run_dir = Path(run_dir).resolve()
        self.ledger_path = self.run_dir / "resource_events.jsonl"
        self.summary_path = self.run_dir / "resource_summary.json"
        self.started_at_utc = started_at_utc or _utc_now()
        self.started_monotonic = started_monotonic or time.monotonic()
        self.iteration_limit = int(iteration_limit)
        self.monitor = GPUUsageMonitor(os.getpid())
        self._finalized = False
        self._lock = threading.Lock()

    def start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text("", encoding="utf-8")
        os.environ[LEDGER_ENV] = str(self.ledger_path)
        os.environ[ITERATION_LIMIT_ENV] = str(self.iteration_limit)
        _append_event(
            {
                "event": "run_started",
                "python_executable": sys.executable,
                "iteration_limit": self.iteration_limit,
            }
        )
        self.monitor.start()
        atexit.register(self.finalize)

    def finalize(self) -> dict[str, Any]:
        with self._lock:
            if self._finalized and self.summary_path.exists():
                return json.loads(self.summary_path.read_text(encoding="utf-8"))
            self.monitor.stop()
            finished_at_utc = _utc_now()
            wall_clock_seconds = time.monotonic() - self.started_monotonic
            events = _read_events(self.ledger_path)
            llm_events = [e for e in events if e.get("event") == "llm_call"]
            execution_events = [
                e for e in events if e.get("event") == "candidate_execution"
            ]
            counted = [e for e in execution_events if not e.get("seed_evaluation")]
            seed_evals = [e for e in execution_events if e.get("seed_evaluation")]
            by_model: dict[str, dict[str, int]] = {}
            for event in llm_events:
                model = str(event.get("model", "unknown"))
                totals = by_model.setdefault(
                    model, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}
                )
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    totals[key] += int(event.get(key, 0))
                totals["calls"] += 1
            prompt_tokens = sum(int(e.get("prompt_tokens", 0)) for e in llm_events)
            completion_tokens = sum(
                int(e.get("completion_tokens", 0)) for e in llm_events
            )
            summary = {
                "accounting_version": "1.0",
                "run_started_at_utc": self.started_at_utc,
                "run_finished_at_utc": finished_at_utc,
                "wall_clock_seconds": wall_clock_seconds,
                "llm_prompt_tokens": prompt_tokens,
                "llm_completion_tokens": completion_tokens,
                "llm_total_tokens": prompt_tokens + completion_tokens,
                "llm_calls": len(llm_events),
                "llm_by_model": by_model,
                "executed_iterations": len(counted),
                "seed_evaluations": len(seed_evals),
                "iteration_limit": self.iteration_limit,
                "gpu_used": self.monitor.active_sample_count > 0,
                "gpu_active_seconds": self.monitor.active_seconds,
                "gpu_models": self.monitor.gpu_models,
                "gpu_peak_used_memory_mib": self.monitor.peak_used_memory_mib,
                "gpu_sample_interval_seconds": self.monitor.interval_seconds,
                "gpu_samples": self.monitor.sample_count,
                "gpu_active_samples": self.monitor.active_sample_count,
                "resource_event_file": self.ledger_path.name,
            }
            temporary = self.summary_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self.summary_path)
            self._finalized = True
            return summary
