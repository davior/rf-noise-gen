"""Live run-status reporting.

The engine emits a :class:`HopStatus` for every hop; a reporter renders it. In a
terminal :class:`LiveStatusReporter` keeps a single line updated in place;
otherwise :class:`LogStatusReporter` prints one line per hop. Reporters are
device-agnostic, so every device gets a status display for free.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Optional, TextIO

from .freq import format_freq


@dataclass
class HopStatus:
    """A snapshot of one hop, passed to reporters."""

    index: int          # 1-based hop number
    start_hz: int
    stop_hz: int
    power_dbm: Optional[float]
    dwell_s: float
    elapsed_s: float

    @property
    def center_hz(self) -> int:
        return (self.start_hz + self.stop_hz) // 2

    @property
    def width_hz(self) -> int:
        return self.stop_hz - self.start_hz

    def line(self) -> str:
        power = "--" if self.power_dbm is None else f"{self.power_dbm:+.1f} dBm"
        # Rate over hops already completed (this hop hasn't dwelt yet), so the
        # first hop reads 0.0 rather than a spike from near-zero elapsed time.
        completed = max(0, self.index - 1)
        rate = completed / self.elapsed_s if self.elapsed_s > 0 else 0.0
        return (
            f"⟳ {format_freq(self.center_hz):>10}  "
            f"band {format_freq(self.start_hz)}-{format_freq(self.stop_hz)}  "
            f"{power:>10}  hop {self.index}  "
            f"{_fmt_clock(self.elapsed_s)}  {rate:.1f} hop/s"
        )


def _fmt_clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class StatusReporter:
    """Base reporter: no-op unless overridden (also serves as NullReporter)."""

    def start(self) -> None:
        pass

    def update(self, status: HopStatus) -> None:
        pass

    def finish(self, hops: int, elapsed_s: float) -> None:
        pass


NullReporter = StatusReporter


class LogStatusReporter(StatusReporter):
    """Print one line per hop -- good for non-TTY output and logs."""

    def __init__(self, stream: Optional[TextIO] = None):
        self.stream = stream or sys.stdout

    def update(self, status: HopStatus) -> None:
        self.stream.write(status.line() + "\n")
        self.stream.flush()

    def finish(self, hops: int, elapsed_s: float) -> None:
        self.stream.write(
            f"stopped after {hops} hops in {_fmt_clock(elapsed_s)}\n"
        )
        self.stream.flush()


class LiveStatusReporter(StatusReporter):
    """Keep a single status line updated in place using a carriage return."""

    def __init__(self, stream: Optional[TextIO] = None):
        self.stream = stream or sys.stdout
        self._width = 0

    def update(self, status: HopStatus) -> None:
        line = status.line()
        pad = max(0, self._width - len(line))
        self._width = len(line)
        self.stream.write("\r" + line + " " * pad)
        self.stream.flush()

    def finish(self, hops: int, elapsed_s: float) -> None:
        self.stream.write(
            f"\ndone: {hops} hops in {_fmt_clock(elapsed_s)}\n"
        )
        self.stream.flush()


def make_reporter(mode: str = "auto", stream: Optional[TextIO] = None) -> StatusReporter:
    """Create a reporter.

    ``mode``: ``"auto"`` (live if the stream is a TTY, else log), ``"live"``,
    ``"log"``, or ``"quiet"`` (no output).
    """
    stream = stream or sys.stdout
    if mode == "quiet":
        return NullReporter()
    if mode == "log":
        return LogStatusReporter(stream)
    if mode == "live":
        return LiveStatusReporter(stream)
    # auto
    if hasattr(stream, "isatty") and stream.isatty():
        return LiveStatusReporter(stream)
    return LogStatusReporter(stream)
