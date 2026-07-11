"""Device abstraction layer.

Every supported radio is wrapped in an :class:`RFDevice` subclass so the noise
generation engine can drive any of them without knowing the hardware details.
A device advertises what it can do through :class:`DeviceCapabilities`, which is
also where the *auto-derived maximum broadcast bandwidth* lives -- the user does
not enter a max bandwidth per range; the device supplies it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ..freq import format_freq


class DeviceError(Exception):
    """Base class for device-related errors."""


class TransmitNotSupported(DeviceError):
    """Raised when transmit is attempted on a receive-only device."""


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

    @abstractmethod
    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float) -> None:
        """Emit a signal covering ``start_hz``..``stop_hz`` for ``dwell_s``.

        Must return only after roughly ``dwell_s`` seconds have elapsed. The
        device should retune without stopping its output so consecutive calls
        produce a seamless hop.
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
        tx = "transmit" if caps.can_transmit else "RECEIVE ONLY"
        return (
            f"{caps.name} [{tx}]\n"
            f"  frequency range : {rng}\n"
            f"  max broadcast bw: {bw}\n"
            f"  {caps.description}"
        )
