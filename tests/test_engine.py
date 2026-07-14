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


def test_on_hop_called_with_status():
    gen = NoiseGenerator(_mock_device(), _mock_session())
    seen = []
    gen.run(iterations=5, on_hop=seen.append)
    assert len(seen) == 5
    assert seen[0].index == 1
    # center matches what the device broadcast
    assert seen[0].center_hz == gen.device.history[0].center_hz


def test_power_drawn_within_session_and_device_range():
    session = _mock_session(power_min_dbm=-60.0, power_max_dbm=-30.0)
    device = MockDevice(verbose=False, sleep=False)  # power range (-120, 10)
    gen = NoiseGenerator(device, session)
    gen.run(iterations=40)
    powers = [rec.power_dbm for rec in device.history]
    assert all(p is not None for p in powers)
    assert all(-60.0 <= p <= -30.0 for p in powers)


def test_power_clamped_to_device_capability():
    # Session asks for a wider range than the device supports; engine clamps.
    session = _mock_session(power_min_dbm=-200.0, power_max_dbm=200.0)
    device = MockDevice(verbose=False, sleep=False, power_range=(-50.0, 0.0))
    gen = NoiseGenerator(device, session)
    gen.run(iterations=30)
    assert all(-50.0 <= rec.power_dbm <= 0.0 for rec in device.history)


def test_power_is_reproducible_with_seed():
    def run_once():
        session = _mock_session(power_min_dbm=-60.0, power_max_dbm=-30.0, seed=99)
        device = MockDevice(verbose=False, sleep=False)
        NoiseGenerator(device, session).run(iterations=10)
        return [rec.power_dbm for rec in device.history]
    assert run_once() == run_once()


def test_pause_disabled_by_default():
    session = _mock_session()
    assert not session.has_pause


def test_pause_every_hops_still_completes_iterations():
    # A small pause every 2 hops must not change how many hops run.
    session = _mock_session(pause_seconds=0.01, pause_every_hops=2)
    device = MockDevice(verbose=False, sleep=False)
    gen = NoiseGenerator(device, session)
    hops = gen.run(iterations=6)
    assert hops == 6
    assert len(device.history) == 6


def test_pause_services_device_keep_alive():
    # The engine must service the device through a pause so a streaming device
    # (e.g. tinySA sweep) can't overflow its buffer while paused.
    session = _mock_session(dwell_seconds=0.0, pause_seconds=0.05, pause_every_hops=2)

    class _CountingMock(MockDevice):
        keep_alive_calls = 0

        def keep_alive(self):
            type(self).keep_alive_calls += 1

    device = _CountingMock(verbose=False, sleep=False)
    NoiseGenerator(device, session).run(iterations=4)
    assert _CountingMock.keep_alive_calls > 0


def test_pause_adds_wall_clock_time():
    # 4 hops with a pause every 2 hops -> one pause fires (after hop 2; the
    # pause after hop 4 is skipped as the final requested hop). ~0.1s minimum.
    session = _mock_session(dwell_seconds=0.0, pause_seconds=0.1, pause_every_hops=2)
    device = MockDevice(verbose=False, sleep=False)
    gen = NoiseGenerator(device, session)
    import time
    t0 = time.monotonic()
    gen.run(iterations=4)
    assert time.monotonic() - t0 >= 0.09


def test_pause_respects_duration_deadline():
    # A long pause must not blow past the run's duration bound.
    session = _mock_session(dwell_seconds=0.0, pause_seconds=10.0, pause_every_hops=1)
    device = MockDevice(verbose=False, sleep=False)
    gen = NoiseGenerator(device, session)
    import time
    t0 = time.monotonic()
    gen.run(duration=0.2)
    assert time.monotonic() - t0 < 1.0


def test_negative_pause_rejected():
    with pytest.raises(ValueError):
        _mock_session(pause_seconds=-1.0, pause_every_hops=2)
    with pytest.raises(ValueError):
        _mock_session(pause_seconds=1.0, pause_every_hops=-2)


def test_no_power_range_yields_none():
    gen = NoiseGenerator(_mock_device(), _mock_session())
    gen.run(iterations=5)
    assert all(rec.power_dbm is None for rec in gen.device.history)


def test_power_range_ignored_when_device_cannot_control(capsys):
    session = _mock_session(power_min_dbm=-60.0, power_max_dbm=-30.0)
    device = MockDevice(verbose=False, sleep=False, power_range=None)
    gen = NoiseGenerator(device, session)
    assert gen.power_range is None
    out = capsys.readouterr().out
    assert "cannot set output level" in out
    gen.run(iterations=3)
    assert all(rec.power_dbm is None for rec in device.history)
