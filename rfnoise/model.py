"""Core data model: frequency ranges and a runnable session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
            "device_options": self.device_options,
            "ranges": [r.to_dict() for r in self.ranges],
            "dwell_seconds": self.dwell_seconds,
            "overlap": self.overlap,
            "seed": self.seed,
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
            seed=data.get("seed"),
        )
