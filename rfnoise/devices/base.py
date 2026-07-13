"""Device abstraction layer.

Every supported radio is wrapped in an :class:`RFDevice` subclass so the noise
generation engine can drive any of them without knowing the hardware details.
A device advertises what it can do through :class:`DeviceCapabilities`, which is
also where the *auto-derived maximum broadcast bandwidth* lives -- the user does
not enter a max bandwidth per range; the device supplies it.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import FrozenSet, Optional, Tuple

from ..freq import format_freq


class DeviceError(Exception):
    """Base class for device-related errors."""


class TransmitNotSupported(DeviceError):
    """Raised when transmit is attempted on a receive-only device."""


# -- signal-generator axes --------------------------------------------------
# Three independent axes decouple *how the frequency moves* (Traversal) from
# *what rides on the carrier* (Modulation) and *what drives the modulation*
# (ModSource). Today's tool is exactly ``RANDOM_HOP x NONE x -``; later phases
# add sequential/sweep traversals and AM/FM. Devices advertise which values
# they support via :class:`DeviceCapabilities`.


class Modulation(enum.Enum):
    """What rides on the carrier. ``NONE`` is today's plain (CW/noise) output."""

    NONE = "none"
    AM = "am"
    FM = "fm"


class ModSource(enum.Enum):
    """What drives AM/FM modulation (Phase 3)."""

    TONE = "tone"
    NOISE = "noise"


class Traversal(enum.Enum):
    """How the centre frequency moves over time. ``RANDOM_HOP`` is today's."""

    RANDOM_HOP = "random_hop"
    SEQUENTIAL = "sequential"
    SWEEP_IN_BAND = "sweep_in_band"


@dataclass(frozen=True)
class SweepSpec:
    """A request to sweep across ``[start_hz, stop_hz]`` rather than sit on one band.

    Set on an :class:`Emission` by the engine when a coverage band is wider than
    the device can emit in a single burst. ``steps`` is how many discrete retune
    slices a *stepped* realisation should use; ``duration_s`` is the total time
    for the whole sweep (the hop's dwell). ``mode`` is ``"stepped"`` today;
    ``"continuous"`` (phase-continuous IQ chirp) arrives with numpy in Phase 3.
    """

    start_hz: int
    stop_hz: int
    steps: int
    duration_s: float
    mode: str = "stepped"

    @property
    def width_hz(self) -> int:
        return self.stop_hz - self.start_hz

    def step_bands(self) -> "list[tuple[int, int]]":
        """Divide ``[start, stop]`` into ``steps`` equal sub-bands, no gaps/overshoot."""
        n = max(1, self.steps)
        edges = [self.start_hz + round(i * self.width_hz / n) for i in range(n + 1)]
        return [(edges[i], edges[i + 1]) for i in range(n)]


@dataclass(frozen=True)
class Emission:
    """One thing to emit: a band, for a dwell, with optional modulation.

    The engine builds one :class:`Emission` per hop and hands it to
    :meth:`RFDevice.emit`. Frequency is expressed as ``start_hz``/``stop_hz``
    (matching the existing :meth:`RFDevice.broadcast` contract), not a
    centre+bandwidth. Modulation fields are all ``None``/``NONE`` today; they
    carry AM/FM parameters once Phase 3 lands. ``sweep`` is set when the band is
    wider than one burst and should be swept across the dwell.
    """

    start_hz: int
    stop_hz: int
    dwell_s: float
    power_dbm: Optional[float] = None
    modulation: Modulation = Modulation.NONE
    source: Optional[ModSource] = None
    deviation_hz: Optional[float] = None   # FM peak deviation
    depth: Optional[float] = None          # AM depth, 0..1
    tone_hz: Optional[float] = None        # source=TONE frequency
    sweep: Optional[SweepSpec] = None      # intra-band sweep (Phase 2)


@dataclass(frozen=True)
class TxBand:
    """A contiguous frequency span a device can output in a given mode.

    ``mode`` describes how the device produces a signal in this span (e.g.
    ``"sine"``/``"square"`` for the tinySA). It is informational for the engine
    but useful for drivers that switch output stages by frequency.
    """

    min_hz: int
    max_hz: int
    mode: str = ""

    def contains(self, hz: int) -> bool:
        return self.min_hz <= hz <= self.max_hz


@dataclass(frozen=True)
class DeviceCapabilities:
    """Static description of a device's transmit abilities.

    ``max_bandwidth_hz`` is the widest continuous signal the device can emit in
    one burst. ``None`` means the device has no fixed hardware cap (e.g. a CW
    generator like the tinySA that is limited only by its tuning range); in that
    case the engine falls back to the device's ``default_band_width``.
    """

    name: str
    can_transmit: bool
    tx_bands: Tuple[TxBand, ...]
    max_bandwidth_hz: Optional[int]
    default_band_width: int
    description: str = ""
    # Output level range the device can be commanded to, in dBm. ``None`` on
    # either bound means the device cannot control its output level.
    power_min_dbm: Optional[float] = None
    power_max_dbm: Optional[float] = None
    # -- signal-generator axes (declarative; the user never configures what the
    # hardware already knows). Defaults describe *today's* behavior so existing
    # devices need no change until they gain new abilities. --
    #: Traversal strategies this device can run. Every device can random-hop.
    supported_traversals: FrozenSet["Traversal"] = frozenset({Traversal.RANDOM_HOP})
    #: Modulations this device can emit. Today every device is CW/noise only.
    supported_modulations: FrozenSet["Modulation"] = frozenset({Modulation.NONE})
    #: Widest phase-continuous IQ chirp the device can emit in one tune (Phase
    #: 3); ``None`` means it has no IQ-generation path.
    instantaneous_bw_hz: Optional[int] = None
    #: How faithfully the device modulates: ``"iq"`` (arbitrary IQ),
    #: ``"fixed_tone"`` (crude built-in generator) or ``"none"``.
    modulation_fidelity: str = "none"

    @property
    def controls_power(self) -> bool:
        return self.power_min_dbm is not None and self.power_max_dbm is not None

    def clamp_power(self, dbm: float) -> float:
        """Clamp a requested dBm level into the device's supported range."""
        if not self.controls_power:
            return dbm
        return max(self.power_min_dbm, min(self.power_max_dbm, dbm))

    @property
    def freq_min_hz(self) -> Optional[int]:
        if not self.tx_bands:
            return None
        return min(b.min_hz for b in self.tx_bands)

    @property
    def freq_max_hz(self) -> Optional[int]:
        if not self.tx_bands:
            return None
        return max(b.max_hz for b in self.tx_bands)

    def supports_frequency(self, hz: int) -> bool:
        return any(b.contains(hz) for b in self.tx_bands)


class RFDevice(ABC):
    """Common interface implemented by every concrete device driver.

    Lifecycle: construct with options, then use as a context manager (``with``)
    or call :meth:`open`/:meth:`close` manually. While open, the engine calls
    :meth:`broadcast` repeatedly, once per frequency hop; the device is expected
    to keep its output active and simply retune, so there is no gap between hops.
    """

    #: Subclasses set this to their :class:`DeviceCapabilities`.
    capabilities: DeviceCapabilities

    def __init__(self, **options):
        self.options = options
        self._open = False

    # -- capability helpers -------------------------------------------------
    @property
    def name(self) -> str:
        return self.capabilities.name

    @property
    def can_transmit(self) -> bool:
        return self.capabilities.can_transmit

    def supports_frequency(self, hz: int) -> bool:
        return self.capabilities.supports_frequency(hz)

    def max_bandwidth_for(self, hz: int) -> Optional[int]:
        """Return the widest band the device may emit around ``hz``.

        Defaults to the static ``max_bandwidth_hz``; subclasses may override to
        make it frequency- or mode-dependent. ``None`` means "no fixed cap".
        """
        return self.capabilities.max_bandwidth_hz

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> None:
        if not self._open:
            self._on_open()
            self._open = True

    def close(self) -> None:
        if self._open:
            self._on_close()
            self._open = False

    def __enter__(self) -> "RFDevice":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- driver hooks -------------------------------------------------------
    def _on_open(self) -> None:  # pragma: no cover - trivial default
        pass

    def _on_close(self) -> None:  # pragma: no cover - trivial default
        pass

    def emit(self, emission: "Emission") -> None:
        """Emit one :class:`Emission`; the engine's single per-hop entry point.

        The base implementation ignores the modulation fields and forwards the
        band/dwell/power to :meth:`broadcast`. A stepped :class:`SweepSpec` is
        realised universally by retuning across its sub-bands -- dividing the
        dwell evenly and calling :meth:`broadcast` once per step, so every device
        can sweep a wide band with no extra backend. Plain (non-swept) emissions
        broadcast once, exactly as before. Devices with a native sweep (tinySA)
        or real modulation (Phase 3) override this.
        """
        sweep = emission.sweep
        if sweep is not None and sweep.steps > 1:
            step_dwell = emission.dwell_s / sweep.steps
            for step_start, step_stop in sweep.step_bands():
                self.broadcast(step_start, step_stop, step_dwell, emission.power_dbm)
            return
        self.broadcast(emission.start_hz, emission.stop_hz,
                       emission.dwell_s, emission.power_dbm)

    @abstractmethod
    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float,
                  power_dbm: Optional[float] = None) -> None:
        """Emit a signal covering ``start_hz``..``stop_hz`` for ``dwell_s``.

        Must return only after roughly ``dwell_s`` seconds have elapsed. The
        device should retune without stopping its output so consecutive calls
        produce a seamless hop. ``power_dbm`` is the requested output level; it
        is ``None`` when no level range is configured, in which case the device
        uses its default/fixed level.
        """

    def describe(self) -> str:
        caps = self.capabilities
        rng = "n/a"
        if caps.freq_min_hz is not None:
            rng = f"{format_freq(caps.freq_min_hz)} - {format_freq(caps.freq_max_hz)}"
        if caps.max_bandwidth_hz is None:
            bw = f"no hardware cap (default {format_freq(caps.default_band_width)})"
        else:
            bw = format_freq(caps.max_bandwidth_hz)
        if caps.controls_power:
            power = f"{caps.power_min_dbm:g} to {caps.power_max_dbm:g} dBm"
        else:
            power = "not adjustable"
        tx = "transmit" if caps.can_transmit else "RECEIVE ONLY"
        return (
            f"{caps.name} [{tx}]\n"
            f"  frequency range : {rng}\n"
            f"  max broadcast bw: {bw}\n"
            f"  output level    : {power}\n"
            f"  {caps.description}"
        )
