"""The frequency-hopping noise generation engine."""

from __future__ import annotations

import random
import time
from typing import Callable, List, Optional

from .bands import Band, RandomBandSelector, build_bands
from .devices.base import RFDevice
from .freq import format_freq
from .model import Session
from .status import HopStatus


class ConfigurationError(Exception):
    """Raised when a session cannot be run on the chosen device."""


def validate(session: Session, device: RFDevice) -> List[Band]:
    """Validate a session against a device and return the pooled bands.

    Checks that the device can transmit, that at least one range is defined,
    and that every range lies fully within the device's transmit bands. Raises
    :class:`ConfigurationError` with an actionable message otherwise.
    """
    if not device.can_transmit:
        raise ConfigurationError(
            f"{device.name} cannot transmit (receive-only device)."
        )
    if not session.ranges:
        raise ConfigurationError("session has no frequency ranges defined.")

    for rng in session.ranges:
        if not device.supports_frequency(rng.lower_hz) or not device.supports_frequency(rng.upper_hz):
            fmin = device.capabilities.freq_min_hz
            fmax = device.capabilities.freq_max_hz
            raise ConfigurationError(
                f"range {format_freq(rng.lower_hz)}-{format_freq(rng.upper_hz)} "
                f"is outside {device.name}'s transmit range "
                f"({format_freq(fmin)}-{format_freq(fmax)})."
            )

    device_max = min(
        (device.max_bandwidth_for(r.lower_hz) for r in session.ranges
         if device.max_bandwidth_for(r.lower_hz) is not None),
        default=None,
    )
    bands = build_bands(
        session.ranges,
        device_max=device_max,
        device_default=device.capabilities.default_band_width,
        overlap=session.overlap,
    )
    if not bands:
        raise ConfigurationError("no broadcast bands could be built from ranges.")
    return bands


class NoiseGenerator:
    """Drives a device to hop across randomly-selected bands.

    Construct with a :class:`Session` and an open-able :class:`RFDevice`, then
    call :meth:`run`. The generator validates the configuration up front, opens
    the device, and loops pick -> broadcast(dwell) -> pick until a stop
    condition is met. Call :meth:`stop` (or send SIGINT) to end cleanly.
    """

    def __init__(self, device: RFDevice, session: Session):
        self.device = device
        self.session = session
        self.bands = validate(session, device)
        # One RNG shared by band selection and power draws -> reproducible.
        self.rng = random.Random(session.seed)
        self.selector = RandomBandSelector(self.bands, rng=self.rng)
        self.power_range = self._resolve_power_range()
        self._stopped = False
        self.hops = 0

    def _resolve_power_range(self):
        """Return the effective (min, max) dBm to draw from, or ``None``.

        Intersects the session's requested range with what the device can
        actually output. If the session asks for a level range the device
        cannot control, warns once and disables level control (not fatal).
        """
        if not self.session.has_power_range:
            return None
        caps = self.device.capabilities
        lo, hi = self.session.power_min_dbm, self.session.power_max_dbm
        if not caps.controls_power:
            print(f"warning: {self.device.name} cannot set output level; "
                  f"ignoring the {lo:g}..{hi:g} dBm strength range.")
            return None
        return (caps.clamp_power(lo), caps.clamp_power(hi))

    def _next_power(self):
        if self.power_range is None:
            return None
        return self.rng.uniform(self.power_range[0], self.power_range[1])

    def stop(self) -> None:
        self._stopped = True

    def plan(self, iterations: int) -> List[Band]:
        """Return the next ``iterations`` bands without transmitting (dry-run)."""
        return [self.selector.next() for _ in range(iterations)]

    def run(self, duration: Optional[float] = None,
            iterations: Optional[int] = None,
            on_hop: Optional[Callable[[HopStatus], None]] = None) -> int:
        """Run the hop loop.

        Stops after ``duration`` seconds and/or ``iterations`` hops, whichever
        comes first; if both are ``None`` it runs until :meth:`stop` or Ctrl-C.
        ``on_hop`` is invoked with a :class:`HopStatus` before each broadcast
        (so it reflects the band being transmitted during its dwell). Returns
        the number of hops performed.
        """
        self._stopped = False
        self.hops = 0
        start = time.monotonic()
        deadline = None if duration is None else start + duration
        self.device.open()
        try:
            while not self._stopped:
                if iterations is not None and self.hops >= iterations:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                band = self.selector.next()
                power = self._next_power()
                dwell = self.session.dwell_seconds
                if deadline is not None:
                    dwell = max(0.0, min(dwell, deadline - time.monotonic()))
                if on_hop is not None:
                    on_hop(HopStatus(
                        index=self.hops + 1,
                        start_hz=band.start_hz,
                        stop_hz=band.stop_hz,
                        power_dbm=power,
                        dwell_s=dwell,
                        elapsed_s=time.monotonic() - start,
                    ))
                self.device.broadcast(band.start_hz, band.stop_hz, dwell, power)
                self.hops += 1
        except KeyboardInterrupt:
            pass
        finally:
            self.device.close()
        return self.hops
