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

from .base import (
    DeviceCapabilities,
    DeviceError,
    Emission,
    Modulation,
    RFDevice,
    TxBand,
)

MAX_SAMPLE_RATE = 20_000_000  # 20 Msps -> ~20 MHz instantaneous bandwidth

# The HackRF has no calibrated dBm output, only a 0-47 dB TX VGA gain. We expose
# a nominal dBm range and map it linearly onto that gain so a session's dBm
# strength range still works; the absolute dBm is approximate/uncalibrated.
POWER_MIN_DBM = -50.0
POWER_MAX_DBM = 5.0
MAX_TXVGA_GAIN = 47


def dbm_to_gain(dbm: float) -> int:
    """Map a nominal dBm level onto the HackRF TX VGA gain (0-47, clamped)."""
    span = POWER_MAX_DBM - POWER_MIN_DBM
    frac = (dbm - POWER_MIN_DBM) / span if span else 0.0
    return max(0, min(MAX_TXVGA_GAIN, int(round(frac * MAX_TXVGA_GAIN))))


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


def iq_to_int8(iq) -> bytes:
    """Convert a complex IQ array to interleaved signed 8-bit I/Q bytes.

    Scales by the peak magnitude so the strongest sample maps to full-scale
    (+/-127), then interleaves I,Q -- the format ``hackrf_transfer`` streams.
    Pure and numpy-based (requires the ``[dsp]`` extra), so it is unit-testable
    without hardware.
    """
    from ..modulation import require_numpy

    np = require_numpy()
    arr = np.asarray(iq, dtype=np.complex128)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = 127.0 / peak if peak > 0 else 0.0
    out = np.empty(2 * arr.size, dtype=np.int8)
    out[0::2] = np.clip(np.round(arr.real * scale), -127, 127).astype(np.int8)
    out[1::2] = np.clip(np.round(arr.imag * scale), -127, 127).astype(np.int8)
    return out.tobytes()


def make_modulated_samples(emission: Emission, sample_rate: int,
                           count: int) -> bytes:
    """Generate ``count`` interleaved 8-bit I/Q samples for a modulated emission.

    Synthesises the baseband IQ with the DSP core (:func:`generate_iq`) and
    converts it with :func:`iq_to_int8`. Requires the ``[dsp]`` extra.
    """
    from .. import modulation as dsp

    iq = dsp.generate_iq(
        emission.modulation, count, sample_rate,
        source=emission.source, depth=emission.depth,
        deviation_hz=emission.deviation_hz, tone_hz=emission.tone_hz,
    )
    return iq_to_int8(iq)


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
            power_min_dbm=POWER_MIN_DBM,
            power_max_dbm=POWER_MAX_DBM,
            # True IQ transmitter: we synthesise arbitrary AM/FM ourselves and
            # stream it (fidelity "iq"). Needs the [dsp] extra to generate IQ.
            supported_modulations=frozenset(
                {Modulation.NONE, Modulation.AM, Modulation.FM}
            ),
            instantaneous_bw_hz=MAX_SAMPLE_RATE,
            modulation_fidelity="iq",
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

    def emit(self, emission: Emission) -> None:
        """Emit one :class:`Emission`, streaming synthesised IQ when modulated.

        A plain (``NONE``) emission takes the base path (sweep-aware
        :meth:`broadcast`, streaming broadband noise). An AM/FM emission is
        generated as baseband IQ at the full sample rate, tuned to the band
        centre, and streamed like the noise path (per-hop restart). Generating
        IQ requires the ``[dsp]`` extra.
        """
        if emission.modulation == Modulation.NONE:
            super().emit(emission)
            return
        center = (int(emission.start_hz) + int(emission.stop_hz)) // 2
        # Modulated: use the full sample rate so the message/deviation fit
        # regardless of the slice width (AM/FM occupied bandwidth is set by the
        # depth/deviation, not the band width).
        sample_rate = MAX_SAMPLE_RATE
        gain = (self.txvga_gain if emission.power_dbm is None
                else dbm_to_gain(emission.power_dbm))
        chunk = make_modulated_samples(emission, sample_rate, sample_rate // 10 or 1)
        self._stream(center, sample_rate, gain, chunk, emission.dwell_s)

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float,
                  power_dbm: Optional[float] = None) -> None:
        center = (int(start_hz) + int(stop_hz)) // 2
        width = max(1, int(stop_hz) - int(start_hz))
        sample_rate = min(width, MAX_SAMPLE_RATE)
        gain = self.txvga_gain if power_dbm is None else dbm_to_gain(power_dbm)
        chunk = make_noise_samples(sample_rate // 10 or 1)  # ~0.1s of samples
        self._stream(center, sample_rate, gain, chunk, dwell_s)

    def _stream(self, center: int, sample_rate: int, gain: int, chunk: bytes,
                dwell_s: float) -> None:
        """(Re)start ``hackrf_transfer`` at ``center`` and stream ``chunk`` for the dwell.

        Retunes by restarting the transfer -- the small per-hop gap noted in the
        module docstring. Writes the pre-generated ``chunk`` repeatedly until the
        dwell elapses; a broken pipe (device/tool gone) ends the write quietly.
        """
        self._stop_proc()  # retune by restarting the stream
        cmd = [
            self.binary,
            "-f", str(center),
            "-s", str(sample_rate),
            "-x", str(gain),
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
        assert self._proc.stdin is not None
        try:
            while time.monotonic() < deadline:
                self._proc.stdin.write(chunk)
        except BrokenPipeError:  # pragma: no cover - hardware dependent
            pass
