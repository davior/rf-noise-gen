"""The frequency-hopping noise generation engine."""

from __future__ import annotations

import math
import random
import time
from typing import Callable, List, Optional

from .bands import Band, build_bands, build_coverage_bands
from .devices.base import (
    Emission,
    Modulation,
    ModSource,
    RFDevice,
    SweepSpec,
    Traversal,
)
from .freq import format_freq
from .model import Session
from .status import HopStatus
from .tuning import (
    RandomPooledStrategy,
    SequentialSweepStrategy,
    SweepInBandStrategy,
)


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

    if session.traversal == Traversal.SWEEP_IN_BAND:
        # Sweep mode covers each range in device-uncapped chunks; the engine
        # steps across any chunk wider than one burst.
        bands = build_coverage_bands(session.ranges, overlap=session.overlap)
    else:
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
        # Pick the tuning strategy from the session. SequentialSweepStrategy
        # draws no randomness, so it never perturbs the shared power stream.
        if session.traversal == Traversal.SWEEP_IN_BAND:
            self.selector = SweepInBandStrategy(self.bands)
        elif session.traversal == Traversal.SEQUENTIAL:
            self.selector = SequentialSweepStrategy(self.bands)
        else:
            self.selector = RandomPooledStrategy(self.bands, rng=self.rng)
        self.power_range = self._resolve_power_range()
        # Resolve the requested modulation against the device once, warning and
        # falling back where the hardware can't honour it (same pattern as the
        # power range above). Result drives every Emission built in run().
        self.modulation, self.mod_source = self._resolve_modulation()
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

    def _resolve_modulation(self):
        """Return the effective ``(modulation, source)`` for this device.

        Mirrors :meth:`_resolve_power_range`: honour the session's request when
        the device supports it, otherwise warn once and fall back (never fatal):

        * modulation ∉ ``supported_modulations`` -> fall back to unmodulated CW.
        * a noise source on a device that can't do arbitrary IQ
          (``modulation_fidelity != "iq"``) -> fall back to a tone.
        """
        mod = self.session.modulation
        if mod == Modulation.NONE:
            return Modulation.NONE, None
        caps = self.device.capabilities
        if mod not in caps.supported_modulations:
            print(f"warning: {self.device.name} cannot emit {mod.value.upper()} "
                  f"modulation; falling back to unmodulated output.")
            return Modulation.NONE, None
        source = self.session.mod_source or ModSource.TONE
        if source == ModSource.NOISE and caps.modulation_fidelity != "iq":
            print(f"warning: {self.device.name} cannot modulate from a noise "
                  f"source (fidelity {caps.modulation_fidelity!r}); using a tone.")
            source = ModSource.TONE
        return mod, source

    def _sweep_for(self, band: Band, dwell: float) -> Optional[SweepSpec]:
        """Build a :class:`SweepSpec` for ``band`` if it must be swept, else None.

        Only sweep-in-band mode sweeps. A band no wider than the device's single
        burst (``max_bandwidth_for`` or, if uncapped like the tinySA, its
        ``default_band_width``) is emitted as one plain band; a wider one is
        divided into ``ceil(width / burst)`` steps to cover across the dwell.
        """
        if self.session.traversal != Traversal.SWEEP_IN_BAND:
            return None
        burst = (self.device.max_bandwidth_for(band.center_hz)
                 or self.device.capabilities.default_band_width)
        width = band.stop_hz - band.start_hz
        steps = max(1, math.ceil(width / burst)) if burst else 1
        if steps <= 1:
            return None
        return SweepSpec(start_hz=band.start_hz, stop_hz=band.stop_hz,
                         steps=steps, duration_s=dwell, mode="stepped")

    def stop(self) -> None:
        self._stopped = True

    def _pause(self, seconds: float, deadline: Optional[float]) -> None:
        """Sleep for ``seconds``, staying responsive to stop and the deadline.

        Sleeps in short slices so :meth:`stop`/Ctrl-C and a ``duration``
        deadline still end the run promptly instead of blocking for the whole
        pause. Clamps to the time left before ``deadline`` when one is set.
        """
        if seconds <= 0:
            return
        end = time.monotonic() + seconds
        if deadline is not None:
            end = min(end, deadline)
        while not self._stopped:
            left = end - time.monotonic()
            if left <= 0:
                break
            time.sleep(min(left, 0.1))

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
                        traversal=self.session.traversal.value,
                    ))
                self.device.emit(Emission(
                    start_hz=band.start_hz,
                    stop_hz=band.stop_hz,
                    dwell_s=dwell,
                    power_dbm=power,
                    modulation=self.modulation,
                    source=self.mod_source,
                    depth=self.session.depth,
                    deviation_hz=self.session.deviation_hz,
                    tone_hz=self.session.tone_hz,
                    sweep=self._sweep_for(band, dwell),
                ))
                self.hops += 1
                # Periodic pause: hold transmission after every N hops. Skipped
                # once iterations are exhausted (the loop-top check ends the run
                # next, so we don't pause after the final requested hop).
                if (self.session.has_pause
                        and self.hops % self.session.pause_every_hops == 0
                        and not (iterations is not None and self.hops >= iterations)):
                    self._pause(self.session.pause_seconds, deadline)
        except KeyboardInterrupt:
            pass
        finally:
            self.device.close()
        return self.hops
