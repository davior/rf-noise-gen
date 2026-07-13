"""Tests for stepped intra-band sweep (Phase 2)."""

import pytest

from rfnoise.bands import build_coverage_bands, coverage_bandwidth
from rfnoise.devices.base import Emission, SweepSpec
from rfnoise.devices.mock import MockDevice
from rfnoise.devices.tinysa import TinySAUltra
from rfnoise.engine import NoiseGenerator
from rfnoise.model import FrequencyRange, Session
from rfnoise.devices.base import Traversal


# -- SweepSpec.step_bands ---------------------------------------------------
def test_step_bands_cover_span_without_gaps_or_overshoot():
    spec = SweepSpec(start_hz=100_000_000, stop_hz=105_000_000, steps=5,
                     duration_s=0.5)
    bands = spec.step_bands()
    assert len(bands) == 5
    assert bands[0][0] == 100_000_000            # starts at the low edge
    assert bands[-1][1] == 105_000_000           # ends exactly at the high edge
    # Contiguous: each step starts where the previous ended.
    for (s0, s1), (n0, _n1) in zip(bands, bands[1:]):
        assert s1 == n0
    # No step exceeds the span.
    assert all(100_000_000 <= s0 < s1 <= 105_000_000 for s0, s1 in bands)


def test_step_bands_single_step_is_whole_span():
    spec = SweepSpec(0, 1_000, steps=1, duration_s=0.1)
    assert spec.step_bands() == [(0, 1_000)]


# -- coverage-band building (device-uncapped) -------------------------------
def test_coverage_bandwidth_uses_whole_range_without_override():
    assert coverage_bandwidth(FrequencyRange(0, 5_000_000)) == 5_000_000


def test_coverage_bandwidth_uses_override_when_set():
    assert coverage_bandwidth(FrequencyRange(0, 5_000_000, 1_000_000)) == 1_000_000


def test_build_coverage_bands_not_capped_by_device():
    # A single 50 MHz range with no override -> one 50 MHz coverage band, far
    # wider than any device burst. (build_bands would have split this to <=cap.)
    bands = build_coverage_bands([FrequencyRange(0, 50_000_000)])
    assert len(bands) == 1
    assert bands[0].width_hz == 50_000_000


# -- base emit() stepped realisation ----------------------------------------
def test_mock_emit_steps_across_a_swept_band():
    dev = MockDevice(verbose=False, sleep=False)
    spec = SweepSpec(100_000_000, 105_000_000, steps=5, duration_s=0.5)
    dev.emit(Emission(100_000_000, 105_000_000, dwell_s=0.5, power_dbm=-20.0,
                      sweep=spec))
    assert len(dev.history) == 5                          # one broadcast per step
    assert dev.history[0].start_hz == 100_000_000
    assert dev.history[-1].stop_hz == 105_000_000
    # Dwell is budgeted evenly across the steps and sums to the hop dwell.
    assert sum(r.dwell_s for r in dev.history) == pytest.approx(0.5)
    assert all(r.power_dbm == -20.0 for r in dev.history)


def test_mock_emit_without_sweep_is_a_single_broadcast():
    dev = MockDevice(verbose=False, sleep=False)
    dev.emit(Emission(100_000_000, 100_100_000, dwell_s=0.1))
    assert len(dev.history) == 1


# -- engine wiring ----------------------------------------------------------
def _sweep_session(**kw):
    base = dict(name="s", device="mock", traversal=Traversal.SWEEP_IN_BAND,
                ranges=[FrequencyRange(100_000_000, 105_000_000)],  # 5 MHz
                dwell_seconds=0.0, seed=1)
    base.update(kw)
    return Session(**base)


def test_engine_sweeps_a_band_wider_than_device_burst():
    # Device burst 1 MHz, coverage band 5 MHz -> 5 stepped emissions in one hop.
    dev = MockDevice(max_bandwidth_hz=1_000_000, verbose=False, sleep=False)
    gen = NoiseGenerator(dev, _sweep_session())
    assert len(gen.bands) == 1                    # one 5 MHz coverage band
    hops = gen.run(iterations=1)
    assert hops == 1                              # one coverage band == one hop
    assert len(dev.history) == 5                  # ...realised as 5 steps
    assert dev.history[0].start_hz == 100_000_000
    assert dev.history[-1].stop_hz == 105_000_000


def test_engine_does_not_step_a_band_within_burst():
    # Coverage band (5 MHz) fits inside the device burst (20 MHz) -> no stepping.
    dev = MockDevice(max_bandwidth_hz=20_000_000, verbose=False, sleep=False)
    gen = NoiseGenerator(dev, _sweep_session())
    gen.run(iterations=1)
    assert len(dev.history) == 1
    assert dev.history[0].width_hz == 5_000_000


def test_engine_step_count_follows_override_chunks():
    # Override splits the 5 MHz range into 1 MHz coverage chunks; each fits the
    # device burst, so every hop is a single (unstepped) emission.
    dev = MockDevice(max_bandwidth_hz=20_000_000, verbose=False, sleep=False)
    session = _sweep_session(
        ranges=[FrequencyRange(100_000_000, 105_000_000, max_bandwidth_hz=1_000_000)])
    gen = NoiseGenerator(dev, session)
    assert len(gen.bands) == 5                    # five 1 MHz coverage chunks
    gen.run(iterations=5)
    assert len(dev.history) == 5
    assert all(r.width_hz == 1_000_000 for r in dev.history)


# -- tinySA native sweep ----------------------------------------------------
class _FakeSerial:
    """Minimal pyserial stand-in that records writes (mirrors test_devices.py)."""

    def __init__(self):
        import serial
        self._serial_mod = serial
        self.in_buffer = b""
        self.writes = []
        self.closed = False

    def write(self, data):
        self.writes.append(data)
        self.in_buffer += b"ch> "
        return len(data)

    def flush(self):
        pass

    def read_until(self, expected):
        idx = self.in_buffer.find(expected)
        if idx == -1:
            drained, self.in_buffer = self.in_buffer, b""
            return drained
        end = idx + len(expected)
        drained, self.in_buffer = self.in_buffer[:end], self.in_buffer[end:]
        return drained

    def close(self):
        self.closed = True


def test_tinysa_realises_sweep_with_one_native_command():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    dev._serial = _FakeSerial()
    spec = SweepSpec(100_000_000, 105_000_000, steps=5, duration_s=0.0)
    dev.emit(Emission(100_000_000, 105_000_000, dwell_s=0.0, sweep=spec))
    # Exactly one command (not five step retunes), spanning the whole coverage band.
    assert len(dev._serial.writes) == 1
    cmd = dev._serial.writes[0]
    assert b"100000000" in cmd and b"105000000" in cmd
    assert cmd.startswith(b"sweep ")


def test_tinysa_plain_emission_falls_back_to_broadcast():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    dev._serial = _FakeSerial()
    dev.emit(Emission(100_000_000, 100_100_000, dwell_s=0.0))   # no sweep
    assert len(dev._serial.writes) == 1
