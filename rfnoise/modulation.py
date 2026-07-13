"""DSP core: AM / FM / continuous-chirp modulation (Phase 3, numpy ``[dsp]``).

Pure, side-effect-free numpy. Everything here is a plain function of its inputs
so it can be unit-tested with no hardware and no engine. numpy is the *only*
new dependency and is imported lazily via :func:`require_numpy`; the stdlib
core (noise, CW, sequential + stepped sweep) keeps working without it, and any
modulated emission raises a clear "install ``.[dsp]``" error instead.

Signal model (all produced as complex baseband IQ centred at DC; the device
tunes the physical carrier):

* **AM** -- envelope ``(1 + depth * m(t))`` with zero phase.
* **FM** -- ``exp(j * 2*pi * deviation * integral(m(t) dt))``.
* **Chirp** (linear FM) -- phase is the integral of a linear frequency ramp
  from ``start_hz`` to ``stop_hz`` over the buffer.

``m(t)`` comes from :mod:`rfnoise.sources` and is normalised to ``[-1, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .devices.base import Modulation, ModSource

#: Shown when numpy is missing; keep the install hint actionable.
_DSP_HINT = (
    "numpy is required for AM/FM/chirp modulation. Install the DSP extra:\n"
    "    pip install -e .[dsp]"
)

#: Defaults for modulation parameters left unset on a session/emission.
DEFAULT_AM_DEPTH = 0.5
DEFAULT_FM_DEVIATION_HZ = 5_000.0


def require_numpy():
    """Return the numpy module, or raise a clear ImportError with a fix hint."""
    try:
        import numpy as np  # noqa: WPS433 (lazy import is intentional)
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_DSP_HINT) from exc
    return np


# -- IQ generators ----------------------------------------------------------

def am_iq(message, depth: float):
    """Amplitude-modulate: complex envelope ``(1 + depth * m(t))``.

    ``depth`` in ``[0, 1]`` is the modulation index; ``message`` is ``m(t)`` in
    ``[-1, 1]``. Returns complex IQ (imaginary part zero at baseband).
    """
    np = require_numpy()
    m = np.asarray(message, dtype=float)
    return (1.0 + float(depth) * m).astype(np.complex128)


def fm_iq(message, deviation_hz: float, sample_rate: float):
    """Frequency-modulate: ``exp(j 2*pi * deviation * integral(m))``.

    The instantaneous frequency offset is ``deviation_hz * m(t)``; the phase is
    its running integral (a cumulative sum divided by ``sample_rate``).
    """
    np = require_numpy()
    m = np.asarray(message, dtype=float)
    phase = 2.0 * np.pi * float(deviation_hz) * np.cumsum(m) / float(sample_rate)
    return np.exp(1j * phase)


def chirp_iq(n_samples: int, sample_rate: float, start_hz: float, stop_hz: float):
    """Linear-FM chirp: instantaneous frequency ramps ``start_hz`` -> ``stop_hz``.

    Phase is ``2*pi*(start*t + 0.5*k*t^2)`` with ``k`` the linear frequency
    rate over the buffer, so :func:`instantaneous_freq` is linear in time.
    """
    np = require_numpy()
    n = max(0, int(n_samples))
    t = np.arange(n) / float(sample_rate)
    duration = n / float(sample_rate) if sample_rate else 0.0
    rate = (float(stop_hz) - float(start_hz)) / duration if duration else 0.0
    phase = 2.0 * np.pi * (float(start_hz) * t + 0.5 * rate * t * t)
    return np.exp(1j * phase)


# -- measurements (used by mock summaries and the DSP tests) ----------------

def instantaneous_freq(iq, sample_rate: float):
    """Instantaneous frequency (Hz) from the derivative of the unwrapped phase.

    Returns an array one shorter than ``iq`` (finite differences).
    """
    np = require_numpy()
    phase = np.unwrap(np.angle(iq))
    return np.diff(phase) / (2.0 * np.pi) * float(sample_rate)


def measure_fm_deviation(iq, sample_rate: float) -> float:
    """Peak absolute instantaneous-frequency deviation of an FM signal (Hz)."""
    np = require_numpy()
    if len(iq) < 2:
        return 0.0
    return float(np.max(np.abs(instantaneous_freq(iq, sample_rate))))


def measure_am_depth(iq) -> float:
    """AM depth from the magnitude envelope: ``(max - min) / (max + min)``."""
    np = require_numpy()
    env = np.abs(iq)
    if len(env) == 0:
        return 0.0
    hi, lo = float(np.max(env)), float(np.min(env))
    total = hi + lo
    return (hi - lo) / total if total else 0.0


# -- high-level composition -------------------------------------------------

@dataclass(frozen=True)
class IQSummary:
    """A cheap measured summary of a generated IQ buffer (for mock/logging)."""

    modulation: Modulation
    source: Optional[ModSource]
    n_samples: int
    sample_rate: float
    depth: Optional[float] = None            # measured AM depth
    deviation_hz: Optional[float] = None     # measured FM peak deviation


def generate_iq(modulation: Modulation, n_samples: int, sample_rate: float, *,
                source: Optional[ModSource] = None,
                depth: Optional[float] = None,
                deviation_hz: Optional[float] = None,
                tone_hz: Optional[float] = None,
                chirp_start_hz: Optional[float] = None,
                chirp_stop_hz: Optional[float] = None,
                noise_seed: Optional[int] = None):
    """Generate the complex IQ buffer for a modulated emission.

    Dispatches on ``modulation``; AM/FM pull a message signal from
    :mod:`rfnoise.sources`. Chirp needs no source. Raises for
    :attr:`~rfnoise.devices.base.Modulation.NONE` (the caller should take the
    plain broadcast path instead).
    """
    # Imported here to avoid an import cycle at module load (sources imports us).
    from . import sources

    if modulation == Modulation.AM:
        m = sources.sample(source or ModSource.TONE, n_samples, sample_rate,
                            tone_hz=tone_hz, noise_seed=noise_seed)
        return am_iq(m, DEFAULT_AM_DEPTH if depth is None else depth)
    if modulation == Modulation.FM:
        m = sources.sample(source or ModSource.TONE, n_samples, sample_rate,
                            tone_hz=tone_hz, noise_seed=noise_seed)
        dev = DEFAULT_FM_DEVIATION_HZ if deviation_hz is None else deviation_hz
        return fm_iq(m, dev, sample_rate)
    if modulation == Modulation.NONE:
        raise ValueError("generate_iq() is for modulated emissions; NONE has no IQ")
    raise ValueError(f"unsupported modulation for IQ generation: {modulation!r}")


def summarize(iq, modulation: Modulation, source: Optional[ModSource],
              sample_rate: float) -> IQSummary:
    """Measure a generated buffer into an :class:`IQSummary` (depth/deviation)."""
    depth = measure_am_depth(iq) if modulation == Modulation.AM else None
    dev = measure_fm_deviation(iq, sample_rate) if modulation == Modulation.FM else None
    return IQSummary(
        modulation=modulation,
        source=source,
        n_samples=len(iq),
        sample_rate=float(sample_rate),
        depth=depth,
        deviation_hz=dev,
    )
