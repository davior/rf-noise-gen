"""A software-only device used for testing and sandbox runs.

It performs no real I/O: it records every hop and (optionally) prints it, so the
engine, band-selection logic and interactive UI can be exercised end-to-end
without any radio attached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from ..freq import format_freq
from .base import DeviceCapabilities, RFDevice, TxBand


@dataclass
class HopRecord:
    start_hz: int
    stop_hz: int
    dwell_s: float
    power_dbm: Optional[float] = None

    @property
    def center_hz(self) -> int:
        return (self.start_hz + self.stop_hz) // 2

    @property
    def width_hz(self) -> int:
        return self.stop_hz - self.start_hz


class MockDevice(RFDevice):
    """Fake transmitter that logs hops instead of touching hardware.

    Options:
      * ``max_bandwidth_hz`` -- overrideable simulated hardware cap (default
        20 MHz, matching a HackRF-class radio).
      * ``verbose`` -- print each hop as it happens (default ``False``; the run
        status reporter is normally the single source of on-screen output).
      * ``sleep`` -- actually sleep for the dwell time (default ``True``);
        tests set this ``False`` to run instantly.
      * ``power_range`` -- ``(min_dbm, max_dbm)`` the mock reports it can output,
        or ``None`` to simulate a device that cannot control level.
    """

    def __init__(self, max_bandwidth_hz: int = 20_000_000, verbose: bool = False,
                 sleep: bool = True, power_range=(-120.0, 10.0), **options):
        super().__init__(**options)
        pmin, pmax = (None, None) if power_range is None else power_range
        self.capabilities = DeviceCapabilities(
            name="Mock Device",
            can_transmit=True,
            tx_bands=(TxBand(0, 6_000_000_000, "mock"),),
            max_bandwidth_hz=int(max_bandwidth_hz),
            default_band_width=1_000_000,
            description="Software-only test transmitter (no RF emitted).",
            power_min_dbm=pmin,
            power_max_dbm=pmax,
        )
        self.verbose = verbose
        self.sleep = sleep
        self.history: List[HopRecord] = []

    def _on_open(self) -> None:
        if self.verbose:
            print(f"[mock] opened ({format_freq(self.capabilities.max_bandwidth_hz)} max bw)")

    def _on_close(self) -> None:
        if self.verbose:
            print(f"[mock] closed after {len(self.history)} hops")

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float,
                  power_dbm=None) -> None:
        rec = HopRecord(int(start_hz), int(stop_hz), float(dwell_s), power_dbm)
        self.history.append(rec)
        if self.verbose:
            level = "" if power_dbm is None else f" @ {power_dbm:.1f} dBm"
            print(
                f"[mock] TX {format_freq(rec.center_hz):>10} "
                f"(band {format_freq(rec.start_hz)}-{format_freq(rec.stop_hz)}, "
                f"width {format_freq(rec.width_hz)}){level} for {dwell_s:.3f}s"
            )
        if self.sleep and dwell_s > 0:
            time.sleep(dwell_s)
