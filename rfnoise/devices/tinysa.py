"""tinySA Ultra driver (serial / USB CDC).

The tinySA Ultra's built-in signal generator emits a single CW carrier -- it
does not have a wide instantaneous bandwidth like an SDR. Two burst modes are
therefore supported:

* ``"sweep"`` (default): sweep the carrier across the selected band for the
  dwell time, emulating a wideband splatter across the slice.
* ``"cw"``: park a single tone at the band centre for the dwell time.

Output stage is chosen automatically by frequency:
  * sine   100 kHz - 800 MHz
  * square 800 MHz - 4.4 GHz
  * mixing 4.4 GHz - 5.4 GHz

.. warning::
   The exact serial command strings vary between tinySA firmware revisions and
   could not be verified against hardware in this environment. They are grouped
   in :data:`_COMMANDS` below; verify/adjust them against your firmware's
   ``help`` output before trusting the RF output. Requires ``pyserial``.
"""

from __future__ import annotations

import time
from typing import Optional

from .base import DeviceCapabilities, DeviceError, RFDevice, TxBand

# Serial commands, centralised so they are easy to correct per firmware.
# ``{freq}`` is an integer in Hz; ``{start}``/``{stop}`` likewise.
_COMMANDS = {
    "cw": "sweep {freq} {freq} 2\r",           # single point == fixed carrier
    "sweep": "sweep {start} {stop} 450\r",       # sweep across the band
    "level": "sweep gain {level}\r",             # output level (dBm-ish index)
    "output_on": "output on\r",
    "output_off": "output off\r",
}


def _mode_for(hz: int) -> str:
    if hz <= 800_000_000:
        return "sine"
    if hz <= 4_400_000_000:
        return "square"
    return "mixing"


class TinySAUltra(RFDevice):
    """Serial driver for the tinySA Ultra signal generator.

    Options:
      * ``port`` -- serial device path (e.g. ``/dev/ttyACM0``). Required to open.
      * ``mode`` -- ``"sweep"`` (default) or ``"cw"``.
      * ``level`` -- output level index passed to the device (default 0).
      * ``default_band_width`` -- band width used when no cap/override applies;
        defaults to 1 MHz in sweep mode, 100 kHz in cw mode.
      * ``baudrate`` -- serial baud (default 115200).
    """

    def __init__(self, port: Optional[str] = None, mode: str = "sweep",
                 level: int = 0, default_band_width: Optional[int] = None,
                 baudrate: int = 115200, **options):
        super().__init__(**options)
        if mode not in ("sweep", "cw"):
            raise ValueError("tinySA mode must be 'sweep' or 'cw'")
        self.port = port
        self.mode = mode
        self.level = level
        self.baudrate = baudrate
        if default_band_width is None:
            default_band_width = 1_000_000 if mode == "sweep" else 100_000
        self.capabilities = DeviceCapabilities(
            name="tinySA Ultra",
            can_transmit=True,
            tx_bands=(
                TxBand(100_000, 800_000_000, "sine"),
                TxBand(800_000_000, 4_400_000_000, "square"),
                TxBand(4_400_000_000, 5_400_000_000, "mixing"),
            ),
            # CW generator: no fixed instantaneous bandwidth -> None.
            max_bandwidth_hz=None,
            default_band_width=int(default_band_width),
            description=f"CW signal generator, {mode} burst mode.",
        )
        self._serial = None

    def _on_open(self) -> None:
        if not self.port:
            raise DeviceError("tinySA: no serial 'port' configured")
        try:
            import serial  # type: ignore
        except ImportError as exc:  # pragma: no cover - depends on env
            raise DeviceError(
                "tinySA requires pyserial (pip install rfnoise[hardware])"
            ) from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=1)
        self._send(_COMMANDS["level"].format(level=self.level))
        self._send(_COMMANDS["output_on"])

    def _on_close(self) -> None:
        if self._serial is not None:
            try:
                self._send(_COMMANDS["output_off"])
            finally:
                self._serial.close()
                self._serial = None

    def _send(self, command: str) -> None:
        if self._serial is None:  # pragma: no cover - guarded by open()
            raise DeviceError("tinySA: serial port not open")
        self._serial.write(command.encode("ascii"))
        self._serial.flush()

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float) -> None:
        if self.mode == "cw":
            center = (int(start_hz) + int(stop_hz)) // 2
            self._send(_COMMANDS["cw"].format(freq=center))
        else:
            self._send(_COMMANDS["sweep"].format(start=int(start_hz), stop=int(stop_hz)))
        if dwell_s > 0:
            time.sleep(dwell_s)
