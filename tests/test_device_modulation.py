"""Phase 3 (device-wiring) tests: HackRF IQ conversion + tinySA fixed-tone AM/FM.

The HackRF helpers are pure numpy (skipped without the ``[dsp]`` extra); the
tinySA command path is exercised with a fake serial (skipped without pyserial),
mirroring the existing ``test_devices.py`` regression tests.
"""

import pytest

from rfnoise.devices.base import Emission, Modulation, ModSource
from rfnoise.devices.tinysa import TinySAUltra


# -- HackRF IQ conversion (numpy) -------------------------------------------

np = pytest.importorskip("numpy")

from rfnoise.devices.hackrf import (  # noqa: E402
    MAX_SAMPLE_RATE,
    HackRFOne,
    iq_to_int8,
    make_modulated_samples,
)


def test_iq_to_int8_scales_to_full_scale_and_interleaves():
    iq = np.array([1 + 0j, 0 + 1j, -1 + 0j])
    raw = iq_to_int8(iq)
    assert isinstance(raw, bytes)
    samples = np.frombuffer(raw, dtype=np.int8)
    assert len(samples) == 2 * len(iq)          # interleaved I,Q
    assert samples[0] == 127 and samples[1] == 0   # 1+0j -> full-scale I
    assert samples[2] == 0 and samples[3] == 127   # 0+1j -> full-scale Q
    assert samples[4] == -127 and samples[5] == 0  # -1+0j -> -full-scale I


def test_iq_to_int8_stays_in_signed_8bit_range():
    iq = np.exp(1j * np.linspace(0, 20 * np.pi, 5000)) * 3.7  # amplitude > 1
    samples = np.frombuffer(iq_to_int8(iq), dtype=np.int8)
    assert samples.min() >= -127 and samples.max() <= 127


def test_iq_to_int8_handles_all_zero_without_dividing_by_zero():
    raw = iq_to_int8(np.zeros(8, dtype=complex))
    samples = np.frombuffer(raw, dtype=np.int8)
    assert len(samples) == 16 and np.all(samples == 0)


def test_make_modulated_samples_length_matches_count():
    em = Emission(start_hz=100_000_000, stop_hz=100_100_000, dwell_s=0.0,
                  modulation=Modulation.AM, source=ModSource.TONE,
                  depth=0.5, tone_hz=1_000.0)
    raw = make_modulated_samples(em, sample_rate=2_000_000, count=1_000)
    assert len(raw) == 2 * 1_000  # I,Q per sample


def test_hackrf_emit_modulated_streams_iq_at_full_rate(monkeypatch):
    dev = HackRFOne()
    seen = {}
    monkeypatch.setattr(dev, "_stream",
                        lambda center, sr, gain, chunk, dwell: seen.update(
                            center=center, sr=sr, chunk=chunk))
    dev.emit(Emission(start_hz=100_000_000, stop_hz=100_200_000, dwell_s=0.0,
                      modulation=Modulation.FM, source=ModSource.TONE,
                      deviation_hz=5_000.0, tone_hz=1_000.0))
    assert seen["center"] == 100_100_000
    assert seen["sr"] == MAX_SAMPLE_RATE                     # modulated: full rate
    assert len(seen["chunk"]) == 2 * (MAX_SAMPLE_RATE // 10)  # ~0.1s of I,Q


def test_hackrf_emit_plain_uses_noise_path_at_band_width(monkeypatch):
    dev = HackRFOne()
    seen = {}
    monkeypatch.setattr(dev, "_stream",
                        lambda center, sr, gain, chunk, dwell: seen.update(sr=sr))
    dev.emit(Emission(start_hz=100_000_000, stop_hz=100_020_000, dwell_s=0.0))
    assert seen["sr"] == 20_000  # unmodulated: sample rate == slice width


# -- tinySA fixed-tone modulation (fake serial) -----------------------------

pytest.importorskip("serial")


class _FakeSerial:
    """Minimal serial stand-in: records writes, satisfies the drain read."""

    def __init__(self):
        self.writes = []
        self.in_buffer = b""
        self.closed = False

    def write(self, data):
        self.writes.append(data.decode("ascii"))
        self.in_buffer = b"ch> "
        return len(data)

    def flush(self):
        pass

    def read_until(self, expected):
        drained, self.in_buffer = self.in_buffer, b""
        return drained

    def close(self):
        self.closed = True


def _emit(dev, emission):
    dev._serial = _FakeSerial()
    dev.emit(emission)
    return dev._serial.writes


def test_tinysa_am_parks_carrier_then_enables_am():
    dev = TinySAUltra(port="/dev/null", mode="cw")
    writes = _emit(dev, Emission(
        start_hz=100_000_000, stop_hz=100_200_000, dwell_s=0.0,
        modulation=Modulation.AM, source=ModSource.TONE,
        depth=0.5, tone_hz=1_000.0,
    ))
    # Carrier parked at the band centre, then AM at 1 kHz / 50% depth.
    assert any("sweep 100100000 100100000" in w for w in writes)
    assert any("am 1000 50" in w for w in writes)


def test_tinysa_fm_issues_fm_command_with_deviation():
    dev = TinySAUltra(port="/dev/null", mode="cw")
    writes = _emit(dev, Emission(
        start_hz=100_000_000, stop_hz=100_200_000, dwell_s=0.0,
        modulation=Modulation.FM, source=ModSource.TONE,
        deviation_hz=8_000.0, tone_hz=2_000.0,
    ))
    assert any("fm 2000 8000" in w for w in writes)


def test_tinysa_modulation_uses_defaults_when_unset():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    writes = _emit(dev, Emission(
        start_hz=100_000_000, stop_hz=100_200_000, dwell_s=0.0,
        modulation=Modulation.AM, source=ModSource.TONE,
    ))
    assert any("am 1000 50" in w for w in writes)  # default tone + depth


def test_tinysa_plain_emission_still_sweeps():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    writes = _emit(dev, Emission(
        start_hz=100_000_000, stop_hz=100_200_000, dwell_s=0.0,
    ))
    assert any("sweep 100000000 100200000" in w for w in writes)
    assert not any(w.startswith("am") or w.startswith("fm") for w in writes)
