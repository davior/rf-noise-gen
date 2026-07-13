"""Tuning strategies: *how the centre frequency moves over time*.

This is the first of the three signal-generator axes (see the implementation
plan). A :class:`TuningStrategy` decides which :class:`~rfnoise.bands.Band` the
engine emits on next, independently of *what* is emitted (modulation) or *what
drives it* (source).

Today there is one strategy, :class:`RandomPooledStrategy`, which reproduces the
original ``RandomBandSelector`` behavior exactly: pool the bands from all ranges
and pick one uniformly at random each hop. Because wider ranges split into more
slices, they are proportionally more likely -- this width-weighting is a
deliberate property and must be preserved. Sequential and sweep-within-band
strategies are added in later phases.
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


#: Backwards-compatible alias for the pre-refactor class name.
RandomBandSelector = RandomPooledStrategy
