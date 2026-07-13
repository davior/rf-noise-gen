"""Tuning strategies: *how the centre frequency moves over time*.

This is the first of the three signal-generator axes (see the implementation
plan). A :class:`TuningStrategy` decides which :class:`~rfnoise.bands.Band` the
engine emits on next, independently of *what* is emitted (modulation) or *what
drives it* (source).

Two strategies exist today:

* :class:`RandomPooledStrategy` reproduces the original ``RandomBandSelector``
  behavior exactly: pool the bands from all ranges and pick one uniformly at
  random each hop. Because wider ranges split into more slices, they are
  proportionally more likely -- this width-weighting is a deliberate property
  and must be preserved.
* :class:`SequentialSweepStrategy` plays every band once, in ascending frequency
  order, then wraps -- a deterministic low-to-high sweep across the ranges.

The sweep-within-band strategy is added in a later phase.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .bands import Band


class TuningStrategy(ABC):
    """Yields the next :class:`Band` to emit on."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of bands in the pool this strategy draws from."""

    @abstractmethod
    def next(self) -> Band:
        """Return the band for the next hop."""


class RandomPooledStrategy(TuningStrategy):
    """Uniformly selects a band from a fixed pool (the original behavior).

    Pass ``rng`` to share a single :class:`random.Random` with the engine (so
    band and power draws come from the same reproducible stream), or ``seed`` to
    let the strategy own its RNG.
    """

    def __init__(self, bands: Sequence[Band], seed: Optional[int] = None,
                 rng: Optional[random.Random] = None):
        if not bands:
            raise ValueError("cannot select from an empty band pool")
        self._bands = list(bands)
        self._rng = rng if rng is not None else random.Random(seed)

    def __len__(self) -> int:
        return len(self._bands)

    def next(self) -> Band:
        return self._rng.choice(self._bands)


class SequentialSweepStrategy(TuningStrategy):
    """Play every band once in ascending frequency order, then wrap.

    A deterministic low-to-high sweep across all configured ranges: the pooled
    bands are sorted by ``(start_hz, stop_hz)`` and yielded in turn, wrapping to
    the lowest band after the highest. Draws **no** randomness, so enabling it
    never perturbs the seeded power stream shared by the engine.
    """

    def __init__(self, bands: Sequence[Band]):
        if not bands:
            raise ValueError("cannot sweep an empty band pool")
        self._bands = sorted(bands, key=lambda b: (b.start_hz, b.stop_hz))
        self._index = 0

    def __len__(self) -> int:
        return len(self._bands)

    def next(self) -> Band:
        band = self._bands[self._index]
        self._index = (self._index + 1) % len(self._bands)
        return band


#: Backwards-compatible alias for the pre-refactor class name.
RandomBandSelector = RandomPooledStrategy
