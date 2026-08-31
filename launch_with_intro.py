"""Launch ByteRush with normal logs and a concurrently animated cyber footer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import TextIO

from terminal_intro import LiveCyberDisplay, show_intro


MILESTONES = (
    ("Set AI_SCIENTIST_ROOT", 0.10, "PROJECT ROOT READY"),
    ("Using GPUs:", 0.20, "COMPUTE DEVICES SCANNED"),
    ("pregenerated ideas from", 0.30, "RESEARCH IDEAS LOADED"),
    ("Results will be saved in", 0.42, "EXPERIMENT WORKSPACE CREATED"),
    ("Resource accounting:", 0.52, "RESOURCE GUARDS ONLINE"),
    ("Search iteration limit:", 0.60, "SEARCH BUDGET VERIFIED"),
    ('Starting run "', 0.72, "AGENT MANAGER ONLINE"),
    ("Starting main stage:", 0.85, "RESEARCH STAGE INITIALIZED"),
    ("MinimalAgent: Getting plan and code", 0.94, "AGENT GENERATING CANDIDATE"),
    ("Running code", 1.00, "RESEARCH LOOP ACTIVE"),
)


def milestone_for(line: str, current: float) -> tuple[float, str] | None:
    """Map existing ByteRush log messages to honest startup progress."""
    match = None
    for marker, progress, status in MILESTONES:
        if marker in line and progress > current:
            match = (progress, status)
    return match


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        add_help=False,
        description="ByteRush live terminal launcher; unknown options pass to launch_scientist_bfts.py",
    )
    parser.add_argument("--intro-help", action="store_true")
    parser.add_argument("--intro-fps", type=float, default=3.0)
    parser.add_argument("--intro-log-file", type=Path)
    parser.add_argument("--intro-ascii", action="store_true")
    parser.add_argument("--no-live-intro", action="store_true")
    args, child_args = parser.parse_known_args()
    if args.intro_help:
        parser.print_help()
        print("\nAll other arguments are forwarded to launch_scientist_bfts.py.")
        raise SystemExit(0)
    return args, child_args


def main() -> int:
    args, child_args = parse_wrapper_args()
    root = Path(__file__).resolve().parent
    command = [sys.executable, "-u", str(root / "launch_scientist_bfts.py"), *child_args]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    display = LiveCyberDisplay(fps=args.intro_fps, ascii_only=args.intro_ascii)
    live = display.enabled and not args.no_live_intro
    if live:
        display.start()
    else:
        show_intro(animate=False, ascii_only=args.intro_ascii)

    raw_log: TextIO | None = None
    if args.intro_log_file:
        args.intro_log_file.parent.mkdir(parents=True, exist_ok=True)
        raw_log = args.intro_log_file.open("a", encoding="utf-8")

    child = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def forward_signal(signum: int, _frame: object) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    previous_sigint = signal.signal(signal.SIGINT, forward_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, forward_signal)
    progress = 0.02
    try:
        assert child.stdout is not None
        for line in child.stdout:
            if raw_log is not None:
                raw_log.write(line)
                raw_log.flush()
            if live:
                display.write(line)
                milestone = milestone_for(line, progress)
                if milestone is not None:
                    progress, status = milestone
                    display.update(progress, status)
            else:
                sys.stdout.write(line)
                sys.stdout.flush()
        return_code = child.wait()
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if raw_log is not None:
            raw_log.close()

    if live:
        display.stop(success=return_code == 0)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
