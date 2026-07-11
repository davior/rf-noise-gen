import pytest

from rfnoise.devices import create_device, device_keys, get_device_class
from rfnoise.devices.base import TransmitNotSupported
from rfnoise.devices.hackrf import HackRFOne, make_noise_samples
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
