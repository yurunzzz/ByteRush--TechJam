"""Zero-dependency terminal intro samples for ByteRush.

Preview the design with::

    python terminal_intro.py --force-animate

Import ``show_intro`` from a real entry point once a design has been selected.
The animation automatically becomes a single static frame in CI, redirected
logs, dumb terminals, or when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import Iterable, TextIO


ESC = "\033["
RESET = f"{ESC}0m"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"
CYAN = f"{ESC}38;5;51m"
BLUE = f"{ESC}38;5;39m"
VIOLET = f"{ESC}38;5;141m"
GREEN = f"{ESC}38;5;84m"
AMBER = f"{ESC}38;5;220m"
WHITE = f"{ESC}38;5;255m"

LOGO = (
    "██████╗ ██╗   ██╗████████╗███████╗██████╗ ██╗   ██╗███████╗██╗  ██╗",
    "██╔══██╗╚██╗ ██╔╝╚══██╔══╝██╔════╝██╔══██╗██║   ██║██╔════╝██║  ██║",
    "██████╔╝ ╚████╔╝    ██║   █████╗  ██████╔╝██║   ██║███████╗███████║",
    "██╔══██╗  ╚██╔╝     ██║   ██╔══╝  ██╔══██╗██║   ██║╚════██║██╔══██║",
    "██████╔╝   ██║      ██║   ███████╗██║  ██║╚██████╔╝███████║██║  ██║",
    "╚═════╝    ╚═╝      ╚═╝   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝",
)

MINI_LOGO = (
    "╔╗ ╦ ╦╔╦╗╔═╗╦═╗╦ ╦╔═╗╦ ╦",
    "╠╩╗╚╦╝ ║ ║╣ ╠╦╝║ ║╚═╗╠═╣",
    "╚═╝ ╩  ╩ ╚═╝╩╚═╚═╝╚═╝╩ ╩",
)


@dataclass(frozen=True)
class RenderContext:
    color: bool
    unicode: bool
    width: int

    def paint(self, text: str, *codes: str) -> str:
        return "".join(codes) + text + RESET if self.color else text


def _terminal_context(stream: TextIO, ascii_only: bool = False) -> RenderContext:
    color = bool(
        stream.isatty()
        and os.getenv("TERM") != "dumb"
        and "NO_COLOR" not in os.environ
    )
    return RenderContext(
        color=color,
        unicode=not ascii_only,
        width=max(54, min(shutil.get_terminal_size((88, 24)).columns, 100)),
    )


def _center(lines: Iterable[str], width: int) -> list[str]:
    ansi = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
    centered = []
    for line in lines:
        visible_width = len(ansi.sub("", line))
        padding = max(0, width - visible_width)
        centered.append(" " * (padding // 2) + line + " " * (padding - padding // 2))
    return centered


def _cyber_scan(ctx: RenderContext, frame: int, frames: int) -> list[str]:
    """Neon scan line: strongest team/competition identity."""
    logo = LOGO if ctx.width >= 78 else MINI_LOGO
    scan = round((len(logo) - 1) * frame / max(1, frames - 1))
    lines = _center([ctx.paint("BYTE // RUSH", BOLD, CYAN)], ctx.width)
    for index, line in enumerate(logo):
        tone = (WHITE, BOLD) if index == scan else (BLUE if index < scan else VIOLET,)
        lines.extend(_center([ctx.paint(line, *tone)], ctx.width))

    ratio = min(1.0, (frame + 1) / frames)
    progress_width = min(42, ctx.width - 16)
    completed = round(progress_width * ratio)
    empty = progress_width - completed
    bar = (
        ctx.paint("█" * completed, CYAN, BOLD)
        + ctx.paint("░" * empty, DIM, BLUE)
    )
    stages = (
        "BOOTING RESEARCH CORE",
        "LOADING EXPERIMENT GUARDS",
        "CONNECTING AGENT PIPELINE",
        "SYSTEM READY",
    )
    stage_index = len(stages) - 1 if ratio >= 1 else min(2, int(ratio * 3))
    stage = stages[stage_index]
    progress_line = f"[ {bar} ] {ratio:>4.0%}"
    lines.extend(
        _center(
            [
                "",
                progress_line,
                ctx.paint(stage, GREEN if ratio >= 1 else AMBER, BOLD),
                ctx.paint("AUTONOMOUS RECOMMENDER RESEARCH SYSTEM", DIM, WHITE),
            ],
            ctx.width,
        )
    )
    return lines


def live_cyber_frame(
    progress: float,
    status: str,
    tick: int,
    *,
    stream: TextIO = sys.stdout,
    ascii_only: bool = False,
) -> list[str]:
    """Build a cyber frame whose scan animation is independent of progress."""
    ctx = _terminal_context(stream, ascii_only=ascii_only)
    logo = LOGO if ctx.width >= 78 else MINI_LOGO
    cycle = max(1, len(logo) * 2 - 2)
    scan = tick % cycle
    if scan >= len(logo):
        scan = cycle - scan

    lines = _center([ctx.paint("BYTE // RUSH", BOLD, CYAN)], ctx.width)
    for index, line in enumerate(logo):
        distance = abs(index - scan)
        if distance == 0:
            tone = (WHITE, BOLD)
        elif distance == 1:
            tone = (CYAN,)
        else:
            tone = (VIOLET,)
        lines.extend(_center([ctx.paint(line, *tone)], ctx.width))

    progress = max(0.0, min(1.0, progress))
    progress_width = min(42, ctx.width - 16)
    completed = round(progress_width * progress)
    bar = (
        ctx.paint("█" * completed, CYAN, BOLD)
        + ctx.paint("░" * (progress_width - completed), DIM, BLUE)
    )
    pulse = ("◆" if tick % 2 == 0 else "◇") if ctx.unicode else ("*" if tick % 2 == 0 else "+")
    lines.extend(
        _center(
            [
                f"[ {bar} ] {progress:>4.0%}",
                ctx.paint(f"{pulse} {status}", GREEN if progress >= 1 else AMBER, BOLD),
            ],
            ctx.width,
        )
    )
    return lines


class LiveCyberDisplay:
    """Keep an animated footer below normal terminal output.

    Calls to :meth:`write` temporarily erase the footer, append durable log
    output, and redraw the footer. A background thread keeps the scan moving
    even while the child process is silent.
    """

    def __init__(
        self,
        *,
        stream: TextIO = sys.stdout,
        fps: float = 4.0,
        ascii_only: bool = False,
    ) -> None:
        self.stream = stream
        self.fps = max(1.0, min(12.0, fps))
        self.ascii_only = ascii_only
        self.progress = 0.02
        self.status = "LAUNCHING BYTERUSH"
        self.tick = 0
        self._height = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.stream.isatty() and os.getenv("TERM") != "dumb")

    def _erase_locked(self) -> None:
        if not self._height:
            return
        self.stream.write(f"{ESC}{self._height}A\r")
        for index in range(self._height):
            self.stream.write(f"{ESC}2K")
            if index < self._height - 1:
                self.stream.write(f"{ESC}1B\r")
        if self._height > 1:
            self.stream.write(f"{ESC}{self._height - 1}A\r")
        self._height = 0

    def _draw_locked(self) -> None:
        lines = live_cyber_frame(
            self.progress,
            self.status,
            self.tick,
            stream=self.stream,
            ascii_only=self.ascii_only,
        )
        self.stream.write("\n".join(f"{ESC}2K{line}" for line in lines) + "\n")
        self.stream.flush()
        self._height = len(lines)

    def _animate(self) -> None:
        while not self._stop.wait(1.0 / self.fps):
            with self._lock:
                self._erase_locked()
                self.tick += 1
                self._draw_locked()

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.stream.write(f"{ESC}?25l")
            self._draw_locked()
        self._thread = threading.Thread(target=self._animate, name="byterush-intro", daemon=True)
        self._thread.start()

    def update(self, progress: float, status: str) -> None:
        with self._lock:
            self.progress = max(self.progress, min(1.0, progress))
            self.status = status

    def write(self, text: str) -> None:
        if not text:
            return
        if not self.enabled:
            self.stream.write(text)
            self.stream.flush()
            return
        with self._lock:
            self._erase_locked()
            self.stream.write(text)
            if not text.endswith("\n"):
                self.stream.write("\n")
            self._draw_locked()

    def stop(self, *, success: bool) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.enabled:
            return
        with self._lock:
            self._erase_locked()
            self.progress = 1.0
            self.status = "RUN COMPLETE" if success else "RUN STOPPED WITH ERRORS"
            self.tick += 1
            self._draw_locked()
            self.stream.write(f"{ESC}?25h{RESET}")
            self.stream.flush()


def show_intro(
    *,
    duration: float = 0.9,
    animate: bool | None = None,
    ascii_only: bool = False,
    stream: TextIO = sys.stdout,
) -> None:
    """Render the Cyber intro, falling back to a static final frame."""
    renderer = _cyber_scan
    total_frames = 10
    ctx = _terminal_context(stream, ascii_only=ascii_only)
    if animate is None:
        animate = ctx.color and not os.getenv("CI") and not os.getenv("BYTERUSH_NO_INTRO")
    frames_to_render = total_frames if animate else 1

    if animate:
        stream.write(f"{ESC}?25l")
    try:
        previous_height = 0
        for frame in range(frames_to_render):
            render_frame = frame if animate else total_frames - 1
            lines = renderer(ctx, render_frame, total_frames)
            if previous_height:
                stream.write(f"{ESC}{previous_height}A")
            prefix = f"{ESC}2K" if animate else ""
            stream.write("\n".join(f"{prefix}{line}" for line in lines) + "\n")
            stream.flush()
            previous_height = len(lines)
            if animate:
                time.sleep(max(0.01, duration / total_frames))
    finally:
        if animate:
            stream.write(f"{ESC}?25h{RESET}")
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview the ByteRush Cyber terminal intro")
    parser.add_argument("--duration", type=float, default=0.9, help="animation duration in seconds")
    parser.add_argument("--force-animate", action="store_true", help="animate even when output is redirected")
    parser.add_argument("--no-animate", action="store_true", help="print one static frame")
    parser.add_argument("--ascii", action="store_true", help="avoid Unicode-only symbols")
    args = parser.parse_args()

    requested_animation = True if args.force_animate else False if args.no_animate else None
    show_intro(
        duration=max(0.0, args.duration),
        animate=requested_animation,
        ascii_only=args.ascii,
    )


if __name__ == "__main__":
    main()
