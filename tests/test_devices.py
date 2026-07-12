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
    assert TinySAUltra().capabilities.power_min_dbm == -110.0
    assert TinySAUltra().capabilities.power_max_dbm == -20.0
    assert HackRFOne().capabilities.controls_power is True
    assert RTLSDR().capabilities.controls_power is False
    assert MockDevice().capabilities.controls_power is True
    assert MockDevice(power_range=None).capabilities.controls_power is False


def test_clamp_power():
    caps = TinySAUltra().capabilities
    assert caps.clamp_power(0.0) == -20.0     # above max -> clamped
    assert caps.clamp_power(-200.0) == -110.0  # below min -> clamped
    assert caps.clamp_power(-50.0) == -50.0    # in range -> unchanged


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
