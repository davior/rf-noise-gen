"""Phase 3 engine-integration tests: modulation compositions + fallback on mock.

Exercises the full engine -> Emission -> MockDevice IQ path for AM/FM crossed
with tuning strategies, plus the warn-and-fall-back behaviour when a device
can't honour a requested modulation or source. numpy is required.
"""

from dataclasses import replace

import pytest

pytest.importorskip("numpy")

from rfnoise.devices.base import Modulation, ModSource
from rfnoise.devices.mock import MockDevice
from rfnoise.engine import NoiseGenerator
from rfnoise.model import FrequencyRange, Session


def _session(**kw):
    base = dict(
        device="mock",
        ranges=[FrequencyRange(100_000_000, 100_100_000)],
        dwell_seconds=0.0,
        seed=1,
    )
    base.update(kw)
    return Session(**base)


def test_random_am_tone_records_measured_depth():
    session = _session(modulation=Modulation.AM, mod_source=ModSource.TONE,
                       depth=0.5, tone_hz=1_000.0)
    dev = MockDevice(sleep=False)
    NoiseGenerator(dev, session).run(iterations=5)
    assert len(dev.history) == 5
    for rec in dev.history:
        assert rec.modulation == Modulation.AM
        assert rec.source == ModSource.TONE
        assert rec.depth == pytest.approx(0.5, abs=1e-2)
        assert rec.deviation_hz is None


def test_sequential_fm_noise_records_measured_deviation():
    session = _session(traversal="sequential", modulation=Modulation.FM,
                       mod_source=ModSource.NOISE, deviation_hz=8_000.0)
    dev = MockDevice(sleep=False)
    NoiseGenerator(dev, session).run(iterations=4)
    assert len(dev.history) == 4
    for rec in dev.history:
        assert rec.modulation == Modulation.FM
        assert rec.source == ModSource.NOISE
        # Broadband noise fills up to the peak deviation; measured <= requested.
        assert rec.deviation_hz is not None
        assert 0 < rec.deviation_hz <= 8_000.0 * 1.05
        assert rec.depth is None


def test_unmodulated_hop_takes_plain_path():
    # Default (NONE) modulation must not synthesise IQ or set summary fields.
    dev = MockDevice(sleep=False)
    NoiseGenerator(dev, _session()).run(iterations=3)
    assert all(r.modulation == Modulation.NONE for r in dev.history)
    assert all(r.depth is None and r.deviation_hz is None for r in dev.history)


def test_unsupported_modulation_warns_and_falls_back_to_cw(capsys):
    dev = MockDevice(sleep=False)
    # A device that only supports plain output.
    dev.capabilities = replace(
        dev.capabilities,
        supported_modulations=frozenset({Modulation.NONE}),
        modulation_fidelity="none",
    )
    session = _session(modulation=Modulation.FM, deviation_hz=5_000.0)
    gen = NoiseGenerator(dev, session)
    assert gen.modulation == Modulation.NONE
    warning = capsys.readouterr().out
    assert "cannot emit FM" in warning
    gen.run(iterations=2)
    assert all(r.modulation == Modulation.NONE for r in dev.history)


def test_noise_source_on_fixed_tone_device_falls_back_to_tone(capsys):
    dev = MockDevice(sleep=False)
    # A crude modulator: supports AM/FM but only from its built-in tone.
    dev.capabilities = replace(dev.capabilities, modulation_fidelity="fixed_tone")
    session = _session(modulation=Modulation.AM, mod_source=ModSource.NOISE,
                       depth=0.5)
    gen = NoiseGenerator(dev, session)
    assert gen.modulation == Modulation.AM
    assert gen.mod_source == ModSource.TONE
    assert "cannot modulate from a noise source" in capsys.readouterr().out


def test_modulation_choice_does_not_perturb_power_schedule():
    # Modulation params come from the session, not the RNG, so the seeded
    # power stream is identical with and without modulation.
    ranges = [FrequencyRange(100_000_000, 100_100_000)]
    plain = Session(device="mock", ranges=ranges, dwell_seconds=0.0, seed=42,
                    power_min_dbm=-40.0, power_max_dbm=-10.0)
    modded = Session(device="mock", ranges=ranges, dwell_seconds=0.0, seed=42,
                     power_min_dbm=-40.0, power_max_dbm=-10.0,
                     modulation=Modulation.AM, depth=0.5)
    dev_a, dev_b = MockDevice(sleep=False), MockDevice(sleep=False)
    NoiseGenerator(dev_a, plain).run(iterations=10)
    NoiseGenerator(dev_b, modded).run(iterations=10)
    assert [r.power_dbm for r in dev_a.history] == [r.power_dbm for r in dev_b.history]
