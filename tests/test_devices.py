import pytest

from rfnoise.devices import create_device, device_keys, get_device_class
from rfnoise.devices.base import TransmitNotSupported
from rfnoise.devices.hackrf import (
    HackRFOne,
    MAX_TXVGA_GAIN,
    POWER_MAX_DBM,
    POWER_MIN_DBM,
    dbm_to_gain,
    make_noise_samples,
)
from rfnoise.devices.mock import MockDevice
from rfnoise.devices.rtlsdr import RTLSDR
from rfnoise.devices.tinysa import TinySAUltra


def test_registry_has_all_devices():
    assert set(device_keys()) == {"mock", "tinysa", "hackrf", "rtlsdr"}


def test_hackrf_auto_max_bandwidth():
    dev = HackRFOne()
    assert dev.capabilities.max_bandwidth_hz == 20_000_000
    assert dev.max_bandwidth_for(100_000_000) == 20_000_000
    assert dev.supports_frequency(100_000_000)
    assert not dev.supports_frequency(500_000)  # below 1 MHz


def test_tinysa_no_hardware_cap_but_has_default():
    sweep = TinySAUltra(mode="sweep")
    assert sweep.capabilities.max_bandwidth_hz is None
    assert sweep.capabilities.default_band_width == 1_000_000
    cw = TinySAUltra(mode="cw")
    assert cw.capabilities.default_band_width == 100_000


def test_tinysa_frequency_bands():
    dev = TinySAUltra()
    assert dev.supports_frequency(100_000)          # sine low end
    assert dev.supports_frequency(2_000_000_000)    # square
    assert dev.supports_frequency(5_000_000_000)    # mixing
    assert not dev.supports_frequency(6_000_000_000)  # above 5.4 GHz


def test_tinysa_rejects_bad_mode():
    with pytest.raises(ValueError):
        TinySAUltra(mode="bogus")


def test_rtlsdr_is_receive_only():
    dev = RTLSDR()
    assert dev.can_transmit is False
    with pytest.raises(TransmitNotSupported):
        dev.broadcast(100_000_000, 100_100_000, 0.0)


def test_make_noise_samples_shape_and_determinism():
    a = make_noise_samples(100, seed=5)
    b = make_noise_samples(100, seed=5)
    assert a == b
    assert len(a) == 200  # interleaved I/Q


def test_describe_runs_for_all_devices():
    for key in device_keys():
        text = get_device_class(key)().describe()
        assert isinstance(text, str) and text


def test_power_capabilities():
    # Firmware ``level -76..-6`` dBm range.
    assert TinySAUltra().capabilities.power_min_dbm == -76.0
    assert TinySAUltra().capabilities.power_max_dbm == -6.0
    assert HackRFOne().capabilities.controls_power is True
    assert RTLSDR().capabilities.controls_power is False
    assert MockDevice().capabilities.controls_power is True
    assert MockDevice(power_range=None).capabilities.controls_power is False


def test_clamp_power():
    caps = TinySAUltra().capabilities
    assert caps.clamp_power(0.0) == -6.0      # above max -> clamped
    assert caps.clamp_power(-200.0) == -76.0  # below min -> clamped
    assert caps.clamp_power(-50.0) == -50.0   # in range -> unchanged


def test_dbm_to_gain_monotonic_and_clamped():
    assert dbm_to_gain(POWER_MIN_DBM) == 0
    assert dbm_to_gain(POWER_MAX_DBM) == MAX_TXVGA_GAIN
    assert dbm_to_gain(-1000.0) == 0            # clamped low
    assert dbm_to_gain(1000.0) == MAX_TXVGA_GAIN  # clamped high
    mid = dbm_to_gain((POWER_MIN_DBM + POWER_MAX_DBM) / 2)
    assert 0 < mid < MAX_TXVGA_GAIN


def test_broadcast_accepts_power_dbm():
    dev = MockDevice(verbose=False, sleep=False)
    dev.broadcast(100_000_000, 100_100_000, 0.0, power_dbm=-33.0)
    assert dev.history[-1].power_dbm == -33.0


class _FakeSerial:
    """Minimal stand-in for pyserial.Serial for exercising ``_send``.

    ``echo_per_write`` bytes are queued into ``in_buffer`` on every write to
    mimic the tinySA shell echoing each command; ``read_until`` must drain them
    so the buffer never grows without bound. Set ``stall=True`` to make writes
    raise like a full-buffer write timeout.
    """

    def __init__(self, echo_per_write=32, stall=False):
        import serial

        self._serial_mod = serial
        self.echo_per_write = echo_per_write
        self.stall = stall
        self.in_buffer = b""
        self.writes = []
        self.closed = False

    def write(self, data):
        if self.stall:
            raise self._serial_mod.SerialTimeoutException("write timeout")
        self.writes.append(data)
        self.in_buffer += b"x" * self.echo_per_write + b"ch> "
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


def test_tinysa_drains_responses_so_buffer_does_not_grow():
    # Regression: unread shell responses used to fill the OS input buffer after
    # ~126 hops and wedge the next write. Draining must keep the buffer bounded.
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    dev._serial = _FakeSerial()
    for _ in range(500):
        dev.broadcast(100_000_000, 101_000_000, 0.0)
    assert len(dev._serial.in_buffer) < 1024  # bounded regardless of hop count


def test_tinysa_send_raises_deviceerror_on_write_stall():
    from rfnoise.devices.base import DeviceError

    dev = TinySAUltra(port="/dev/null", mode="sweep")
    dev._serial = _FakeSerial(stall=True)
    with pytest.raises(DeviceError):
        dev.broadcast(100_000_000, 101_000_000, 0.0)


def test_tinysa_close_does_not_raise_when_port_stalled():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    fake = _FakeSerial(stall=True)
    dev._serial = fake
    dev._open = True
    dev.close()  # must not propagate the stalled output-off write
    assert fake.closed is True
    assert dev._serial is None


class _StreamingSerial:
    """Fake serial that models a running sweep streaming scan data.

    ``feed`` queues bytes as if the firmware emitted them; ``in_waiting``/``read``
    let the dwell drain them. Exists to prove the dwell no longer sleeps blind
    (which let the buffer overflow after a couple of hops -> write stall).
    """

    def __init__(self):
        self.buffered = 0
        self.read_total = 0
        self.reset_calls = 0
        self.writes = []

    def feed(self, n):
        self.buffered += n

    @property
    def in_waiting(self):
        return self.buffered

    def read(self, n):
        n = min(n, self.buffered)
        self.buffered -= n
        self.read_total += n
        return b"x" * n

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def flush(self):
        pass

    def read_until(self, expected):
        return b"ch> "

    def reset_input_buffer(self):
        self.buffered = 0
        self.reset_calls += 1

    def close(self):
        self.closed = True


def test_tinysa_dwell_drains_streamed_data():
    # Regression: a running sweep streams data through the dwell. The dwell must
    # read and discard it (not sleep blind) so the buffer can't overflow.
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    fake = _StreamingSerial()
    fake.feed(5000)
    dev._serial = fake
    dev._dwell(0.02)
    assert fake.read_total >= 5000   # drained during the dwell
    assert fake.buffered == 0        # nothing left to carry over / overflow


def test_tinysa_zero_dwell_still_clears_buffer():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    fake = _StreamingSerial()
    fake.feed(500)
    dev._serial = fake
    dev._dwell(0.0)
    assert fake.buffered == 0 and fake.reset_calls >= 1


def test_tinysa_broadcast_stays_bounded_across_many_hops():
    dev = TinySAUltra(port="/dev/null", mode="sweep")
    fake = _StreamingSerial()
    dev._serial = fake
    for _ in range(200):
        fake.feed(300)               # each sweep leaves a burst behind
        dev.broadcast(400_000_000, 1_000_000_000, 0.0)
    assert fake.buffered == 0        # never accumulates hop after hop


class _EIOOnWriteSerial(_StreamingSerial):
    """Fake serial that raises EIO on write, mimicking a USB drop.

    ``succeed_first`` writes go through before ``fail_times`` raises begin, so a
    drop can be placed a few hops into a run rather than on the first write.
    """

    def __init__(self, fail_times=1, succeed_first=0):
        super().__init__()
        self.fail_times = fail_times
        self.succeed_first = succeed_first
        self._writes = 0

    def write(self, data):
        self._writes += 1
        if self._writes > self.succeed_first and self.fail_times > 0:
            self.fail_times -= 1
            raise OSError(5, "Input/output error")
        return super().write(data)


def test_tinysa_reconnects_on_io_error(monkeypatch):
    # A mid-run USB drop (EIO) must be recovered: reopen the port and retry the
    # write on the fresh handle instead of surfacing "(5, 'Input/output error')".
    dev = TinySAUltra(port="/dev/ttyACM0", mode="sweep")
    dev.reconnect_delay = 0
    dev._serial = _EIOOnWriteSerial(fail_times=1)
    fresh = _StreamingSerial()
    monkeypatch.setattr(dev, "_open_serial", lambda port: fresh)
    monkeypatch.setattr(dev, "_find_port", lambda: "/dev/ttyACM0")

    dev._send("sweep 1 2 450\r")

    assert dev._serial is fresh                                  # reconnected
    assert any(b"sweep 1 2 450" in w for w in fresh.writes)      # command retried
    assert any(b"mode output" in w for w in fresh.writes)        # generator mode re-armed


def test_tinysa_reconnect_gives_up_when_device_stays_gone(monkeypatch):
    from rfnoise.devices.base import DeviceError

    dev = TinySAUltra(port="/dev/ttyACM0", mode="sweep")
    dev.reconnect_delay = 0
    dev.reconnect_attempts = 3
    dev._serial = _EIOOnWriteSerial(fail_times=99)
    monkeypatch.setattr(dev, "_find_port", lambda: None)  # never comes back
    with pytest.raises(DeviceError, match="did not come back"):
        dev._send("x\r")


def test_tinysa_reconnect_disabled_fails_fast(monkeypatch):
    from rfnoise.devices.base import DeviceError

    dev = TinySAUltra(port="/dev/ttyACM0", mode="sweep", reconnect_attempts=0)
    dev._serial = _EIOOnWriteSerial(fail_times=99)
    with pytest.raises(DeviceError, match="reconnect disabled"):
        dev._send("x\r")


def test_tinysa_dwell_recovers_from_drop(monkeypatch):
    dev = TinySAUltra(port="/dev/ttyACM0", mode="sweep")
    dev.reconnect_delay = 0

    class _EIOOnReadSerial(_StreamingSerial):
        def read(self, n):
            raise OSError(5, "Input/output error")

    dropped = _EIOOnReadSerial()
    dropped.feed(500)                     # data waiting -> triggers a read
    dev._serial = dropped
    fresh = _StreamingSerial()
    monkeypatch.setattr(dev, "_open_serial", lambda port: fresh)
    monkeypatch.setattr(dev, "_find_port", lambda: "/dev/ttyACM0")

    dev._dwell(0.02)
    assert dev._serial is fresh           # recovered mid-dwell


def test_tinysa_run_continues_across_midrun_drop(monkeypatch):
    # The key end-to-end guarantee: if the device drops while a run is in
    # progress, the reconnect happens inside the device call so the engine's
    # hop loop keeps going -- the run completes all iterations, no restart.
    from rfnoise.engine import NoiseGenerator
    from rfnoise.model import FrequencyRange, Session

    dev = TinySAUltra(port="/dev/ttyACM0", mode="sweep")
    dev.reconnect_delay = 0

    made = []

    def fake_open(port):
        # The first handle (used for open + first hops) drops once a few writes
        # in; handles opened by reconnect are healthy.
        serial = (_EIOOnWriteSerial(fail_times=1, succeed_first=4)
                  if not made else _StreamingSerial())
        made.append(serial)
        return serial

    monkeypatch.setattr(dev, "_open_serial", fake_open)
    monkeypatch.setattr(dev, "_find_port", lambda: "/dev/ttyACM0")

    session = Session(device="tinysa", dwell_seconds=0.0, seed=1,
                      ranges=[FrequencyRange(400_000_000, 1_000_000_000)])
    hops = NoiseGenerator(dev, session).run(iterations=6)

    assert hops == 6            # ran to completion despite the mid-run drop
    assert len(made) >= 2       # a reconnect (second open) actually happened
