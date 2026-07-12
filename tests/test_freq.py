import pytest

from rfnoise.freq import format_freq, parse_freq


@pytest.mark.parametrize("text,expected", [
    ("100", 100),
    ("100hz", 100),
    ("100k", 100_000),
    ("100kHz", 100_000),
    ("2.4M", 2_400_000),
    ("2.4MHz", 2_400_000),
    ("5.3GHz", 5_300_000_000),
    ("5.3 G", 5_300_000_000),
    ("  1.75 GHz ", 1_750_000_000),
])
def test_parse_freq(text, expected):
    assert parse_freq(text) == expected


def test_parse_freq_number():
    assert parse_freq(1000) == 1000
    assert parse_freq(1500.0) == 1500


@pytest.mark.parametrize("bad", ["", "abc", "10 furlongs", "-5", "1.2.3"])
def test_parse_freq_invalid(bad):
    with pytest.raises(ValueError):
        parse_freq(bad)


@pytest.mark.parametrize("hz,expected", [
    (100, "100 Hz"),
    (100_000, "100 kHz"),
    (2_400_000, "2.4 MHz"),
    (5_300_000_000, "5.3 GHz"),
])
def test_format_freq(hz, expected):
    assert format_freq(hz) == expected


def test_round_trip():
    for hz in (100_000, 2_400_000, 5_300_000_000):
        assert parse_freq(format_freq(hz)) == hz
