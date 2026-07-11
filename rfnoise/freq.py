"""Frequency parsing and formatting helpers.

Frequencies are represented internally as an integer number of hertz. These
helpers let the interactive UI and config files accept human-friendly strings
like ``"100kHz"``, ``"5.3 GHz"`` or ``"2.4M"`` and render them back nicely.
"""

from __future__ import annotations

import re

_SUFFIXES = {
    "": 1,
    "hz": 1,
    "k": 1_000,
    "khz": 1_000,
    "m": 1_000_000,
    "mhz": 1_000_000,
    "g": 1_000_000_000,
    "ghz": 1_000_000_000,
}

_PATTERN = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([a-zA-Z]*)\s*$")


def parse_freq(value) -> int:
    """Parse a frequency string or number into an integer count of hertz.

    Accepts plain numbers (interpreted as hertz) and suffixed strings such as
    ``"100k"``, ``"100kHz"``, ``"5.3GHz"``. Raises ``ValueError`` on anything
    that is not a recognisable, non-negative frequency.
    """
    if isinstance(value, (int, float)):
        if value < 0:
            raise ValueError("frequency must be non-negative")
        return int(round(value))

    if not isinstance(value, str):
        raise ValueError(f"cannot parse frequency from {value!r}")

    match = _PATTERN.match(value)
    if not match:
        raise ValueError(f"invalid frequency: {value!r}")

    number, suffix = match.groups()
    suffix = suffix.lower()
    if suffix not in _SUFFIXES:
        raise ValueError(f"unknown frequency unit {suffix!r} in {value!r}")

    hz = float(number) * _SUFFIXES[suffix]
    if hz < 0:
        raise ValueError("frequency must be non-negative")
    return int(round(hz))


def format_freq(hz: int) -> str:
    """Render an integer hertz value as a compact, human-readable string."""
    hz = int(hz)
    if abs(hz) >= 1_000_000_000:
        return f"{hz / 1_000_000_000:.6g} GHz"
    if abs(hz) >= 1_000_000:
        return f"{hz / 1_000_000:.6g} MHz"
    if abs(hz) >= 1_000:
        return f"{hz / 1_000:.6g} kHz"
    return f"{hz} Hz"
