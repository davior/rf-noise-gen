"""Band splitting math.

A range is divided into consecutive slices ("bands") no wider than the effective
maximum bandwidth. *Selecting* which band to emit next is a tuning strategy and
lives in :mod:`rfnoise.tuning`; the original ``RandomBandSelector`` name is still
importable from here for backwards compatibility (see :func:`__getattr__`).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from .model import FrequencyRange


@dataclass(frozen=True)
class Band:
    """A concrete slice the device will broadcast on.

    ``range_lower_hz``/``range_upper_hz`` remember the bounds of the parent
    :class:`~rfnoise.model.FrequencyRange` the band was cut from. They are
    optional (``None`` when a band is built in isolation, e.g. in tests) and are
    used to keep per-hop drift from pushing an emission outside its range.
    """

    start_hz: int
    stop_hz: int
    range_lower_hz: Optional[int] = None
    range_upper_hz: Optional[int] = None

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
        bands.append(Band(start, stop, rng.lower_hz, rng.upper_hz))
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


def coverage_bandwidth(rng: FrequencyRange) -> int:
    """Width of one *coverage* chunk for sweep-in-band mode.

    Unlike :func:`effective_bandwidth`, this is **not** capped by the device: it
    is the amount of spectrum to cover in a single dwell, taken from the range's
    ``max_bandwidth_hz`` override if set, else the whole range. The engine then
    sweeps across a chunk wider than the device can emit in one burst.
    """
    if rng.max_bandwidth_hz is not None:
        return max(1, min(rng.max_bandwidth_hz, rng.width_hz))
    return rng.width_hz


def drift_offset(band: Band, fraction: float, rand: random.Random) -> int:
    """Random frequency offset (Hz) to shift ``band`` by on a single hop.

    The offset is drawn uniformly from ``[-reach, +reach]`` where
    ``reach = fraction * band.width_hz``, then clamped so the shifted band stays
    fully inside its parent range: it can never move ``start`` below
    ``range_lower_hz`` nor ``stop`` above ``range_upper_hz``. Interior bands get
    the full ``±reach``; bands touching a range edge drift only inward; a band
    that already fills its range cannot move. Returns ``0`` when ``fraction`` is
    non-positive, the band has no width, or the clamped interval is empty.

    When the parent-range bounds are ``None`` (a band built in isolation), only
    the ``±reach`` cap applies.
    """
    if fraction <= 0 or band.width_hz <= 0:
        return 0
    reach = fraction * band.width_hz
    lo, hi = -reach, reach
    if band.range_lower_hz is not None:
        lo = max(lo, band.range_lower_hz - band.start_hz)
    if band.range_upper_hz is not None:
        hi = min(hi, band.range_upper_hz - band.stop_hz)
    if hi <= lo:
        return 0
    return int(round(rand.uniform(lo, hi)))


def build_coverage_bands(ranges: Sequence[FrequencyRange],
                         overlap: float = 0.0) -> List[Band]:
    """Build the pooled coverage chunks across every range (device-uncapped)."""
    pool: List[Band] = []
    for rng in ranges:
        pool.extend(split_range(rng, coverage_bandwidth(rng), overlap))
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
