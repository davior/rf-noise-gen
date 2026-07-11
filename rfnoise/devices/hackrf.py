"""HackRF One driver.

The HackRF One is a true wideband half-duplex transceiver: at its maximum
20 Msps sample rate it can emit roughly 20 MHz of instantaneous bandwidth,
tunable 1 MHz - 6 GHz. To make noise across a band we tune to the band centre,
set the sample rate to the band width (capped at 20 MHz) and stream complex
Gaussian noise samples.

Streaming is done by piping generated samples into the ``hackrf_transfer`` CLI.
The noise-sample generation (:func:`make_noise_samples`) is a pure function and
is unit-tested; the subprocess streaming path needs real hardware and could not
be exercised here.

.. note::
   Restarting ``hackrf_transfer`` on each hop introduces a small gap. A
   continuous-retune implementation via SoapySDR/pyhackrf is a possible future
   enhancement (see README).
"""

from __future__ import annotations

import subprocess
import time
from typing import Optional

from .base import DeviceCapabilities, DeviceError, RFDevice, TxBand

MAX_SAMPLE_RATE = 20_000_000  # 20 Msps -> ~20 MHz instantaneous bandwidth


def make_noise_samples(count: int, seed: Optional[int] = None) -> bytes:
    """Generate ``count`` interleaved 8-bit signed I/Q noise samples.

    Returns ``2 * count`` bytes (I,Q per sample) suitable for feeding to
    ``hackrf_transfer``. Pure and deterministic given ``seed`` so it can be
    tested without hardware.
    """
    import random as _random

    rng = _random.Random(seed)
    buf = bytearray(2 * count)
    for i in range(2 * count):
        # Uniform noise in signed 8-bit range; good enough for broadband hash.
        buf[i] = rng.randint(0, 255)
    return bytes(buf)


class HackRFOne(RFDevice):
    """Driver that streams broadband noise through ``hackrf_transfer``.

    Options:
      * ``txvga_gain`` -- TX VGA gain (0-47, default 30).
      * ``amp`` -- enable the TX RF amplifier (default ``False``).
      * ``binary`` -- path to ``hackrf_transfer`` (default found on ``PATH``).
    """

    def __init__(self, txvga_gain: int = 30, amp: bool = False,
                 binary: str = "hackrf_transfer", **options):
        super().__init__(**options)
        self.txvga_gain = txvga_gain
        self.amp = amp
        self.binary = binary
        self.capabilities = DeviceCapabilities(
            name="HackRF One",
            can_transmit=True,
            tx_bands=(TxBand(1_000_000, 6_000_000_000, "iq"),),
            max_bandwidth_hz=MAX_SAMPLE_RATE,
            default_band_width=MAX_SAMPLE_RATE,
            description="Wideband SDR transceiver, up to 20 MHz instantaneous bandwidth.",
        )
        self._proc: Optional[subprocess.Popen] = None

    def _on_close(self) -> None:
        self._stop_proc()

    def _stop_proc(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:  # pragma: no cover
                self._proc.kill()
            self._proc = None

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float) -> None:
        center = (int(start_hz) + int(stop_hz)) // 2
        width = max(1, int(stop_hz) - int(start_hz))
        sample_rate = min(width, MAX_SAMPLE_RATE)
        self._stop_proc()  # retune by restarting the stream
        cmd = [
            self.binary,
            "-f", str(center),
            "-s", str(sample_rate),
            "-x", str(self.txvga_gain),
            "-a", "1" if self.amp else "0",
            "-t", "-",  # read samples from stdin
        ]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except FileNotFoundError as exc:
            raise DeviceError(
                f"HackRF: '{self.binary}' not found; install hackrf tools"
            ) from exc

        deadline = time.monotonic() + dwell_s
        chunk = make_noise_samples(sample_rate // 10 or 1)  # ~0.1s of samples
        assert self._proc.stdin is not None
        try:
            while time.monotonic() < deadline:
                self._proc.stdin.write(chunk)
        except BrokenPipeError:  # pragma: no cover - hardware dependent
            pass
