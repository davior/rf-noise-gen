"""Core data model: frequency ranges and a runnable session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .devices.base import Modulation, ModSource, Traversal
from .freq import format_freq


@dataclass
class FrequencyRange:
    """A frequency range the generator may hop within.

    ``max_bandwidth_hz`` is optional: leave it ``None`` to let the selected
    device supply its own maximum broadcast bandwidth automatically. Setting it
    only *narrows* the bands further than the device would.
    """

    lower_hz: int
    upper_hz: int
    max_bandwidth_hz: Optional[int] = None

    def __post_init__(self) -> None:
        self.lower_hz = int(self.lower_hz)
        self.upper_hz = int(self.upper_hz)
        if self.upper_hz <= self.lower_hz:
            raise ValueError(
                f"range upper bound ({format_freq(self.upper_hz)}) must exceed "
                f"lower bound ({format_freq(self.lower_hz)})"
            )
        if self.max_bandwidth_hz is not None:
            self.max_bandwidth_hz = int(self.max_bandwidth_hz)
            if self.max_bandwidth_hz <= 0:
                raise ValueError("max_bandwidth_hz must be positive")

    @property
    def width_hz(self) -> int:
        return self.upper_hz - self.lower_hz

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lower_hz": self.lower_hz,
            "upper_hz": self.upper_hz,
            "max_bandwidth_hz": self.max_bandwidth_hz,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrequencyRange":
        return cls(
            lower_hz=data["lower_hz"],
            upper_hz=data["upper_hz"],
            max_bandwidth_hz=data.get("max_bandwidth_hz"),
        )

    def __str__(self) -> str:
        bw = (
            "device default"
            if self.max_bandwidth_hz is None
            else format_freq(self.max_bandwidth_hz)
        )
        return (
            f"{format_freq(self.lower_hz)} - {format_freq(self.upper_hz)} "
            f"(max bw: {bw})"
        )


@dataclass
class Session:
    """A saved, reloadable generator configuration."""

    name: str = "untitled"
    device: str = "mock"
    device_options: Dict[str, Any] = field(default_factory=dict)
    ranges: List[FrequencyRange] = field(default_factory=list)
    dwell_seconds: float = 0.5
    overlap: float = 0.0  # 0 = sequential bands; 0<overlap<1 = fractional overlap
    # How the centre frequency moves over time. ``RANDOM_HOP`` (default) picks a
    # random band each hop; ``SEQUENTIAL`` sweeps every band low-to-high in order.
    traversal: Traversal = Traversal.RANDOM_HOP
    # Modulation axis (Phase 3): what rides on the carrier and what drives it.
    # ``NONE`` (default) is today's plain CW/noise output and needs no numpy.
    # ``mod_source`` only applies when modulation is AM/FM; ``depth`` (AM,
    # 0..1) and ``deviation_hz`` (FM peak) fall back to DSP defaults when None.
    modulation: Modulation = Modulation.NONE
    mod_source: Optional[ModSource] = None
    depth: Optional[float] = None
    deviation_hz: Optional[float] = None
    tone_hz: Optional[float] = None
    seed: Optional[int] = None
    # Optional periodic pause: hold transmission for ``pause_seconds`` after
    # every ``pause_every_hops`` hops. Both must be > 0 to take effect
    # (``pause_every_hops`` = 0 disables it).
    pause_seconds: float = 0.0
    pause_every_hops: int = 0
    # Optional random output-level range in dBm; when both are set, each hop
    # broadcasts at a random level drawn uniformly from [min, max].
    power_min_dbm: Optional[float] = None
    power_max_dbm: Optional[float] = None

    def __post_init__(self) -> None:
        # Accept a plain string (e.g. from a session file or CLI flag) as well
        # as a Traversal enum, so callers never have to import the enum.
        if isinstance(self.traversal, str):
            self.traversal = Traversal(self.traversal)
        # Accept plain strings for the modulation axis too (session files / CLI).
        if isinstance(self.modulation, str):
            self.modulation = Modulation(self.modulation)
        if isinstance(self.mod_source, str):
            self.mod_source = ModSource(self.mod_source)
        if self.depth is not None:
            self.depth = float(self.depth)
            if not 0.0 <= self.depth <= 1.0:
                raise ValueError("depth must be between 0 and 1")
        if self.deviation_hz is not None:
            self.deviation_hz = float(self.deviation_hz)
            if self.deviation_hz <= 0:
                raise ValueError("deviation_hz must be positive")
        if self.tone_hz is not None:
            self.tone_hz = float(self.tone_hz)
            if self.tone_hz <= 0:
                raise ValueError("tone_hz must be positive")
        if self.power_min_dbm is not None and self.power_max_dbm is not None:
            if self.power_max_dbm < self.power_min_dbm:
                raise ValueError("power_max_dbm must be >= power_min_dbm")
        self.pause_seconds = float(self.pause_seconds)
        self.pause_every_hops = int(self.pause_every_hops)
        if self.pause_seconds < 0:
            raise ValueError("pause_seconds must be >= 0")
        if self.pause_every_hops < 0:
            raise ValueError("pause_every_hops must be >= 0")

    @property
    def has_power_range(self) -> bool:
        return self.power_min_dbm is not None and self.power_max_dbm is not None

    @property
    def has_pause(self) -> bool:
        return self.pause_every_hops > 0 and self.pause_seconds > 0

    @property
    def has_modulation(self) -> bool:
        return self.modulation != Modulation.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "device_options": self.device_options,
            "ranges": [r.to_dict() for r in self.ranges],
            "dwell_seconds": self.dwell_seconds,
            "overlap": self.overlap,
            "traversal": self.traversal.value,
            "modulation": self.modulation.value,
            "mod_source": None if self.mod_source is None else self.mod_source.value,
            "depth": self.depth,
            "deviation_hz": self.deviation_hz,
            "tone_hz": self.tone_hz,
            "seed": self.seed,
            "pause_seconds": self.pause_seconds,
            "pause_every_hops": self.pause_every_hops,
            "power_min_dbm": self.power_min_dbm,
            "power_max_dbm": self.power_max_dbm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            name=data.get("name", "untitled"),
            device=data.get("device", "mock"),
            device_options=dict(data.get("device_options", {})),
            ranges=[FrequencyRange.from_dict(r) for r in data.get("ranges", [])],
            dwell_seconds=float(data.get("dwell_seconds", 0.5)),
            overlap=float(data.get("overlap", 0.0)),
            traversal=data.get("traversal", Traversal.RANDOM_HOP.value),
            modulation=data.get("modulation", Modulation.NONE.value),
            mod_source=data.get("mod_source"),
            depth=data.get("depth"),
            deviation_hz=data.get("deviation_hz"),
            tone_hz=data.get("tone_hz"),
            seed=data.get("seed"),
            pause_seconds=float(data.get("pause_seconds", 0.0)),
            pause_every_hops=int(data.get("pause_every_hops", 0)),
            power_min_dbm=data.get("power_min_dbm"),
            power_max_dbm=data.get("power_max_dbm"),
        )
