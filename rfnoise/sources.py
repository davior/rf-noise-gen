"""Modulating sources: *what drives AM/FM modulation* (Phase 3).

This is the third signal-generator axis (see ``IMPLEMENTATION_PLAN.md``). A
source produces a baseband message signal ``m(t)`` normalised to ``[-1, 1]``,
which :mod:`rfnoise.modulation` then imposes on the carrier as amplitude (AM)
or frequency (FM) variation.

numpy is imported lazily (via :mod:`rfnoise.modulation`) so the stdlib core
stays dependency-free; sampling a source therefore requires the optional
``[dsp]`` extra. The ``noise`` source draws from its **own** independent RNG and
never touches the engine's schedule stream -- per Invariant 1 and the
schedule-level-only reproducibility decision, the raw sample stream does not
need to be reproducible.
"""

from __future__ import annotations

from typing import Optional

from .devices.base import ModSource
from .modulation import require_numpy

#: Default message-tone frequency when a TONE source leaves ``tone_hz`` unset.
DEFAULT_TONE_HZ = 1_000.0


def sample(source: ModSource, n_samples: int, sample_rate: float, *,
           tone_hz: Optional[float] = None, noise_seed: Optional[int] = None):
    """Return ``n_samples`` of the message signal ``m(t)`` in ``[-1, 1]``.

    ``source`` selects the waveform:

    * :attr:`~rfnoise.devices.base.ModSource.TONE` -- a pure sine at ``tone_hz``
      (defaulting to :data:`DEFAULT_TONE_HZ`).
    * :attr:`~rfnoise.devices.base.ModSource.NOISE` -- broadband uniform noise
      from an **independent** generator (``noise_seed`` only for test
      determinism; production leaves it ``None``).

    Returns a real ``float64`` numpy array of length ``n_samples``.
    """
    np = require_numpy()
    n = max(0, int(n_samples))
    if source == ModSource.TONE:
        f = DEFAULT_TONE_HZ if tone_hz is None else float(tone_hz)
        t = np.arange(n) / float(sample_rate)
        return np.sin(2.0 * np.pi * f * t)
    if source == ModSource.NOISE:
        # Independent RNG: never the schedule stream (Invariant 1).
        rng = np.random.default_rng(noise_seed)
        return rng.uniform(-1.0, 1.0, size=n)
    raise ValueError(f"unknown modulating source: {source!r}")
