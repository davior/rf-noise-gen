"""Phase 3 DSP-core tests: AM depth, FM deviation, chirp linearity, sources.

These exercise the pure-numpy functions directly (no engine, no device). numpy
is required, so the whole module is skipped when the ``[dsp]`` extra is absent.
"""

import math

import pytest

np = pytest.importorskip("numpy")

from rfnoise import modulation as dsp
from rfnoise import sources
from rfnoise.devices.base import Modulation, ModSource


SAMPLE_RATE = 2_000_000
N = 200_000


def test_am_depth_recovered_from_envelope():
    # A tone spans [-1, 1], so the envelope (1 + depth*m) has depth = index.
    m = sources.sample(ModSource.TONE, N, SAMPLE_RATE, tone_hz=1_000.0)
    iq = dsp.am_iq(m, depth=0.4)
    assert dsp.measure_am_depth(iq) == pytest.approx(0.4, abs=1e-3)


def test_am_depth_zero_is_unmodulated_carrier():
    m = sources.sample(ModSource.TONE, N, SAMPLE_RATE, tone_hz=2_000.0)
    iq = dsp.am_iq(m, depth=0.0)
    assert dsp.measure_am_depth(iq) == pytest.approx(0.0, abs=1e-6)


def test_fm_peak_deviation_recovered():
    # Peak instantaneous frequency of exp(j 2pi dev integral(m)) is dev*max|m|.
    m = sources.sample(ModSource.TONE, N, SAMPLE_RATE, tone_hz=1_000.0)
    iq = dsp.fm_iq(m, deviation_hz=5_000.0, sample_rate=SAMPLE_RATE)
    assert dsp.measure_fm_deviation(iq, SAMPLE_RATE) == pytest.approx(5_000.0, rel=0.02)


def test_chirp_instantaneous_freq_is_linear_in_time():
    n = 100_000
    iq = dsp.chirp_iq(n, SAMPLE_RATE, start_hz=10_000.0, stop_hz=110_000.0)
    freq = dsp.instantaneous_freq(iq, SAMPLE_RATE)
    t = np.arange(len(freq)) / SAMPLE_RATE
    # Least-squares fit: slope ~ (stop-start)/duration, intercept ~ start.
    slope, intercept = np.polyfit(t, freq, 1)
    duration = n / SAMPLE_RATE
    assert slope == pytest.approx((110_000.0 - 10_000.0) / duration, rel=0.01)
    assert intercept == pytest.approx(10_000.0, abs=500.0)
    # Linear: residual to the fit is tiny relative to the swept span.
    residual = np.max(np.abs(freq - (slope * t + intercept)))
    assert residual < 0.01 * (110_000.0 - 10_000.0)


def test_tone_source_frequency():
    m = sources.sample(ModSource.TONE, N, SAMPLE_RATE, tone_hz=3_000.0)
    # Dominant spectral bin of the real tone corresponds to tone_hz.
    spectrum = np.abs(np.fft.rfft(m))
    peak_bin = int(np.argmax(spectrum))
    peak_hz = peak_bin * SAMPLE_RATE / N
    assert peak_hz == pytest.approx(3_000.0, abs=SAMPLE_RATE / N)


def test_tone_source_is_normalised():
    m = sources.sample(ModSource.TONE, N, SAMPLE_RATE, tone_hz=1_000.0)
    assert np.max(m) == pytest.approx(1.0, abs=1e-3)
    assert np.min(m) == pytest.approx(-1.0, abs=1e-3)


def test_noise_source_stays_in_range():
    m = sources.sample(ModSource.NOISE, N, SAMPLE_RATE)
    assert m.min() >= -1.0 and m.max() <= 1.0
    assert len(m) == N


def test_noise_source_is_independent_of_a_given_seed_stream():
    # Reproducible *for testing* when seeded, but this RNG is never the
    # schedule stream -- the point of Invariant 1.
    a = sources.sample(ModSource.NOISE, 1_000, SAMPLE_RATE, noise_seed=7)
    b = sources.sample(ModSource.NOISE, 1_000, SAMPLE_RATE, noise_seed=7)
    c = sources.sample(ModSource.NOISE, 1_000, SAMPLE_RATE, noise_seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_generate_iq_dispatches_and_summarises():
    iq = dsp.generate_iq(Modulation.AM, N, SAMPLE_RATE,
                         source=ModSource.TONE, depth=0.6, tone_hz=1_000.0)
    summary = dsp.summarize(iq, Modulation.AM, ModSource.TONE, SAMPLE_RATE)
    assert summary.modulation == Modulation.AM
    assert summary.depth == pytest.approx(0.6, abs=1e-3)
    assert summary.deviation_hz is None


def test_generate_iq_rejects_none():
    with pytest.raises(ValueError):
        dsp.generate_iq(Modulation.NONE, N, SAMPLE_RATE)


def test_require_numpy_error_message_points_at_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numpy":
            raise ImportError("no numpy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"\.\[dsp\]"):
        dsp.require_numpy()
