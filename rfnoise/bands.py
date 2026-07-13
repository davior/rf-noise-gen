"""Band splitting math.

A range is divided into consecutive slices ("bands") no wider than the effective
maximum bandwidth. *Selecting* which band to emit next is a tuning strategy and
lives in :mod:`rfnoise.tuning`; the original ``RandomBandSelector`` name is still
importable from here for backwards compatibility (see :func:`__getattr__`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .model import FrequencyRange


@dataclass(frozen=True)
class Band:
    """A concrete slice the device will broadcast on."""

    start_hz: int
    stop_hz: int

    @property
    def width_hz(self) -> int:
        return self.stop_hz - self.start_hz

    @property
    def center_hz(self) -> int:
        return (self.start_hz + self.stop_hz) // 2


def split_range(rng: FrequencyRange, effective_bw: int, overlap: float = 0.0) -> List[Band]:
    """Split ``rng`` into bands at most ``effective_bw`` wide.

    ``overlap`` in ``[0, 1)`` shifts each successive band's start by
    ``(1 - overlap) * effective_bw`` so adjacent bands overlap. The final band
    is clipped to the range's upper bound.
    """
    if effective_bw <= 0:
        raise ValueError("effective bandwidth must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    # A band never exceeds the range width.
    width = min(effective_bw, rng.width_hz)
    step = max(1, int(round(width * (1.0 - overlap))))

    bands: List[Band] = []
    start = rng.lower_hz
    while start < rng.upper_hz:
        stop = min(start + width, rng.upper_hz)
        bands.append(Band(start, stop))
        if stop >= rng.upper_hz:
            break
        start += step
    return bands


def effective_bandwidth(rng: FrequencyRange, device_max: Optional[int],
                        device_default: int) -> int:
    """Resolve the band width to use for ``rng`` on a device.

    Precedence: the range's own override (if set), then the device's hardware
    cap (if any), otherwise the device's default band width. The result is
    never wider than the range itself.
    """
    candidates = [rng.width_hz]
    if rng.max_bandwidth_hz is not None:
        candidates.append(rng.max_bandwidth_hz)
    if device_max is not None:
        candidates.append(device_max)
    if rng.max_bandwidth_hz is None and device_max is None:
        candidates.append(device_default)
    return max(1, min(candidates))


def build_bands(ranges: Sequence[FrequencyRange], device_max: Optional[int],
                device_default: int, overlap: float = 0.0) -> List[Band]:
    """Build the pooled list of bands across every range."""
    pool: List[Band] = []
    for rng in ranges:
        bw = effective_bandwidth(rng, device_max, device_default)
        pool.extend(split_range(rng, bw, overlap))
    return pool


def __getattr__(name: str) -> Any:
    """Lazily re-export ``RandomBandSelector`` from :mod:`rfnoise.tuning`.

    The selector moved to ``tuning.py`` (which imports :class:`Band` from here),
    so a top-level import would be circular. Resolving it on attribute access
    keeps ``from rfnoise.bands import RandomBandSelector`` working without the
    cycle.
    """
    if name == "RandomBandSelector":
        from .tuning import RandomPooledStrategy
        return RandomPooledStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
