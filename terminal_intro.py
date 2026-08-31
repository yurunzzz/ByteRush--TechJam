"""Zero-dependency terminal intro samples for ByteRush.

Preview every design with::

    python terminal_intro.py --all --force-animate

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
from typing import Callable, Iterable, TextIO


ESC = "\033["
RESET = f"{ESC}0m"
BOLD = f"{ESC}1m"
DIM = f"{ESC}2m"
CYAN = f"{ESC}38;5;51m"
BLUE = f"{ESC}38;5;39m"
VIOLET = f"{ESC}38;5;141m"
PINK = f"{ESC}38;5;213m"
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


def _neural_boot(ctx: RenderContext, frame: int, frames: int) -> list[str]:
    """Research-console boot sequence with an animated signal graph."""
    del frames
    nodes = ["IDEA", "CODE", "TRAIN", "EVAL", "SELECT"]
    active = min(frame // 2, len(nodes) - 1)
    rendered = []
    for index, node in enumerate(nodes):
        if index < active:
            rendered.append(ctx.paint(f"● {node}", GREEN, BOLD))
        elif index == active:
            rendered.append(ctx.paint(f"◉ {node}", CYAN, BOLD))
        else:
            rendered.append(ctx.paint(f"○ {node}", DIM, WHITE))
    connector = "  ━━━  "
    title = ctx.paint("NEURAL RESEARCH CORE", BOLD, WHITE)
    pulse = "▁▂▃▅▇█▇▅▃▂"[(frame * 2) % 10 :] + "▁▂▃▅▇█▇▅▃▂"[: (frame * 2) % 10]
    inner = min(72, ctx.width - 4)
    graph = connector.join(re.sub(r"\x1b\[[0-9;]*m", "", item) for item in rendered)
    graph = graph[:inner].center(inner)
    return _center(
        [
            "┌" + "─" * inner + "┐",
            "│" + _center([title], inner)[0] + "│",
            "│" + " " * inner + "│",
            "│" + ctx.paint(graph, CYAN) + "│",
            "│" + ctx.paint(pulse.center(inner), VIOLET, PINK) + "│",
            "└" + "─" * inner + "┘",
            ctx.paint("Turning experiments into evidence.", DIM, WHITE),
        ],
        ctx.width,
    )


def _warp_drive(ctx: RenderContext, frame: int, frames: int) -> list[str]:
    """Compact speed/launch motif, best for frequent local starts."""
    ratio = min(1.0, (frame + 1) / frames)
    speed = int(ratio * 100)
    streak = "═" * max(1, int(ratio * 21))
    bolt = "⚡" if ctx.unicode else ">>"
    return _center(
        [
            ctx.paint(f"{streak} {bolt} BYTERUSH {bolt} {streak}", BOLD, AMBER),
            ctx.paint("RESEARCH ENGINE / IGNITION", BOLD, WHITE),
            "",
            ctx.paint(f"CORE VELOCITY  {speed:>3}%  " + "█" * int(ratio * 18), CYAN),
            ctx.paint("KuaiRand · GAUC · nDCG@5 · autonomous search", DIM, WHITE),
        ],
        ctx.width,
    )


def _minimal(ctx: RenderContext, frame: int, frames: int) -> list[str]:
    """Quiet professional variant for production logs and demos."""
    del frames
    spinner = ("◐", "◓", "◑", "◒") if ctx.unicode else ("|", "/", "-", "\\")
    glyph = spinner[frame % len(spinner)]
    return [
        "",
        "  " + ctx.paint("BYTERUSH", BOLD, CYAN) + ctx.paint(" / research agent", DIM, WHITE),
        "  " + ctx.paint("━" * 46, VIOLET),
        "  " + ctx.paint(glyph, CYAN) + " Initializing autonomous experiment pipeline",
        "  " + ctx.paint("READY", GREEN, BOLD) + ctx.paint("  validation-only · reproducible · guarded", DIM, WHITE),
        "",
    ]


STYLES: dict[str, tuple[str, Callable[[RenderContext, int, int], list[str]], int]] = {
    "cyber": ("霓虹扫描 / 适合比赛演示", _cyber_scan, 10),
    "neural": ("神经研究核心 / 适合展示 Agent 流程", _neural_boot, 10),
    "warp": ("曲速点火 / 紧凑有冲击力", _warp_drive, 12),
    "minimal": ("专业极简 / 适合日常启动", _minimal, 8),
}


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
    style: str = "cyber",
    *,
    duration: float = 0.9,
    animate: bool | None = None,
    ascii_only: bool = False,
    stream: TextIO = sys.stdout,
) -> None:
    """Render one intro, automatically falling back to a static final frame."""
    if style not in STYLES:
        raise ValueError(f"Unknown intro style: {style!r}")
    _, renderer, total_frames = STYLES[style]
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
    parser = argparse.ArgumentParser(description="Preview ByteRush terminal intro designs")
    parser.add_argument("--style", choices=STYLES, default="cyber")
    parser.add_argument("--all", action="store_true", help="preview all designs")
    parser.add_argument("--list", action="store_true", help="list available designs")
    parser.add_argument("--duration", type=float, default=0.9, help="seconds per design")
    parser.add_argument("--force-animate", action="store_true", help="animate even when output is redirected")
    parser.add_argument("--no-animate", action="store_true", help="print one static frame")
    parser.add_argument("--ascii", action="store_true", help="avoid Unicode-only symbols")
    args = parser.parse_args()

    if args.list:
        for name, (description, _, _) in STYLES.items():
            print(f"{name:<8} {description}")
        return

    styles = list(STYLES) if args.all else [args.style]
    requested_animation = True if args.force_animate else False if args.no_animate else None
    for index, style in enumerate(styles):
        if len(styles) > 1:
            print(f"\n[{style}] {STYLES[style][0]}")
        show_intro(
            style,
            duration=max(0.0, args.duration),
            animate=requested_animation,
            ascii_only=args.ascii,
        )
        if index < len(styles) - 1 and requested_animation:
            time.sleep(0.25)


if __name__ == "__main__":
    main()
