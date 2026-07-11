import pytest

from rfnoise.devices import create_device
from rfnoise.devices.mock import MockDevice
from rfnoise.engine import ConfigurationError, NoiseGenerator, validate
from rfnoise.model import FrequencyRange, Session


def _mock_session(**kwargs):
    defaults = dict(
        name="t",
        device="mock",
        ranges=[FrequencyRange(100_000, 200_000, 10_000)],
        dwell_seconds=0.0,
        seed=1,
    )
    defaults.update(kwargs)
    return Session(**defaults)


def _mock_device():
    return MockDevice(verbose=False, sleep=False)


def test_run_iterations():
    gen = NoiseGenerator(_mock_device(), _mock_session())
    hops = gen.run(iterations=20)
    assert hops == 20
    assert len(gen.device.history) == 20


def test_all_hops_within_ranges():
    gen = NoiseGenerator(_mock_device(), _mock_session())
    gen.run(iterations=50)
    for rec in gen.device.history:
        assert rec.start_hz >= 100_000
        assert rec.stop_hz <= 200_000


def test_bands_respect_device_max_bandwidth():
    # Device caps bandwidth at 5 kHz even though the range asks for nothing.
    device = MockDevice(max_bandwidth_hz=5_000, verbose=False, sleep=False)
    session = _mock_session(ranges=[FrequencyRange(0, 100_000)])
    gen = NoiseGenerator(device, session)
    assert all(b.width_hz <= 5_000 for b in gen.bands)


def test_range_override_narrows_below_device_max():
    device = MockDevice(max_bandwidth_hz=20_000_000, verbose=False, sleep=False)
    session = _mock_session(ranges=[FrequencyRange(0, 1_000_000, max_bandwidth_hz=50_000)])
    gen = NoiseGenerator(device, session)
    assert all(b.width_hz <= 50_000 for b in gen.bands)


def test_dry_run_plan_is_deterministic():
    gen = NoiseGenerator(_mock_device(), _mock_session())
    plan_a = gen.plan(5)
    gen2 = NoiseGenerator(_mock_device(), _mock_session())
    plan_b = gen2.plan(5)
    assert plan_a == plan_b


def test_receive_only_device_rejected():
    session = _mock_session(device="rtlsdr", ranges=[FrequencyRange(100_000_000, 100_100_000)])
    device = create_device("rtlsdr")
    with pytest.raises(ConfigurationError):
        validate(session, device)


def test_out_of_range_rejected():
    # tinySA tops out at 5.4 GHz; ask for 6 GHz.
    session = _mock_session(device="tinysa", ranges=[FrequencyRange(5_500_000_000, 5_600_000_000)])
    device = create_device("tinysa")
    with pytest.raises(ConfigurationError):
        validate(session, device)


def test_empty_ranges_rejected():
    device = _mock_device()
    with pytest.raises(ConfigurationError):
        validate(_mock_session(ranges=[]), device)


def test_duration_bounds_run():
    # With a tiny nonzero dwell and a short duration, run stops promptly.
    session = _mock_session(dwell_seconds=0.001)
    device = MockDevice(verbose=False, sleep=True)
    gen = NoiseGenerator(device, session)
    hops = gen.run(duration=0.05)
    assert hops >= 1
