"""Phase 0 seam tests: Emission/emit path, capability defaults, tuning strategy.

These lock the new abstraction boundaries without asserting any *new* behavior:
the ``emit`` path must be identical to ``broadcast``, and every device must
declare today's behavior (random-hop, no modulation).
"""

import pytest

from rfnoise.bands import Band, build_bands
from rfnoise.devices import create_device, device_keys
from rfnoise.devices.base import (
    Emission,
    Modulation,
    ModSource,
    Traversal,
)
from rfnoise.devices.mock import MockDevice
from rfnoise.model import FrequencyRange
from rfnoise.tuning import RandomPooledStrategy, TuningStrategy


def test_emit_matches_broadcast_on_mock():
    # emit(Emission) must record exactly what the equivalent broadcast() does.
    a = MockDevice(verbose=False, sleep=False)
    a.emit(Emission(start_hz=100_000_000, stop_hz=100_100_000, dwell_s=0.0,
                    power_dbm=-33.0))
    b = MockDevice(verbose=False, sleep=False)
    b.broadcast(100_000_000, 100_100_000, 0.0, power_dbm=-33.0)
    assert a.history == b.history


def test_emit_ignores_modulation_fields_today():
    # A modulated Emission still forwards only band/dwell/power (Phase 0 is CW).
    dev = MockDevice(verbose=False, sleep=False)
    dev.emit(Emission(start_hz=1_000_000, stop_hz=1_010_000, dwell_s=0.0,
                      power_dbm=None, modulation=Modulation.FM,
                      source=ModSource.NOISE, deviation_hz=5_000.0))
    rec = dev.history[-1]
    assert (rec.start_hz, rec.stop_hz, rec.power_dbm) == (1_000_000, 1_010_000, None)


def test_every_device_declares_its_axes():
    # Every device can random-hop and always advertises plain (NONE) output.
    # Phase 3 gives the mock an IQ modulation path; the hardware drivers stay
    # NONE-only until their own wiring PR lands.
    for key in device_keys():
        caps = create_device(key).capabilities
        assert Traversal.RANDOM_HOP in caps.supported_traversals
        assert Modulation.NONE in caps.supported_modulations
        if key == "mock":
            assert {Modulation.AM, Modulation.FM} <= caps.supported_modulations
            assert caps.modulation_fidelity == "iq"
            assert caps.instantaneous_bw_hz is not None
        else:
            assert caps.supported_modulations == frozenset({Modulation.NONE})
            assert caps.instantaneous_bw_hz is None
            assert caps.modulation_fidelity == "none"


def test_random_band_selector_alias_still_importable():
    # Backwards-compat: the old name resolves to the new strategy class.
    from rfnoise.bands import RandomBandSelector

    assert RandomBandSelector is RandomPooledStrategy


def test_bands_module_rejects_unknown_attribute():
    import rfnoise.bands as bands

    with pytest.raises(AttributeError):
        bands.does_not_exist


def test_random_pooled_strategy_is_a_tuning_strategy():
    bands = build_bands([FrequencyRange(0, 100_000)], device_max=10_000,
                        device_default=10_000)
    strat = RandomPooledStrategy(bands, seed=1)
    assert isinstance(strat, TuningStrategy)
    assert len(strat) == len(bands)
    assert isinstance(strat.next(), Band)
