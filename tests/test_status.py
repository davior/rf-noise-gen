import io

from rfnoise.status import (
    HopStatus,
    LiveStatusReporter,
    LogStatusReporter,
    NullReporter,
    make_reporter,
)


def _status(power=-42.3):
    return HopStatus(
        index=7,
        start_hz=140_000_000,
        stop_hz=150_000_000,
        power_dbm=power,
        dwell_s=0.25,
        elapsed_s=3.0,
    )


def test_hopstatus_derived_fields():
    s = _status()
    assert s.center_hz == 145_000_000
    assert s.width_hz == 10_000_000


def test_hopstatus_line_contains_freq_and_power():
    line = _status().line()
    assert "145 MHz" in line
    assert "dBm" in line
    assert "hop 7" in line


def test_hopstatus_line_no_power():
    line = _status(power=None).line()
    assert "--" in line


def test_hopstatus_line_shows_modulation_only_when_active():
    plain = _status().line()
    assert "[am]" not in plain and "[fm]" not in plain
    modded = HopStatus(index=1, start_hz=140_000_000, stop_hz=150_000_000,
                       power_dbm=None, dwell_s=0.1, elapsed_s=1.0,
                       modulation="fm").line()
    assert "[fm]" in modded


def test_log_reporter_writes_line_per_hop():
    buf = io.StringIO()
    rep = LogStatusReporter(buf)
    rep.start()
    rep.update(_status())
    rep.finish(hops=1, elapsed_s=3.0)
    out = buf.getvalue()
    assert "145 MHz" in out
    assert "stopped after 1 hops" in out


def test_live_reporter_uses_carriage_return():
    buf = io.StringIO()
    rep = LiveStatusReporter(buf)
    rep.update(_status())
    assert buf.getvalue().startswith("\r")


def test_null_reporter_is_silent():
    buf = io.StringIO()
    rep = NullReporter()
    rep.start()
    rep.update(_status())
    rep.finish(1, 1.0)
    # NullReporter writes nowhere; buf stays empty.
    assert buf.getvalue() == ""


def test_make_reporter_modes():
    assert isinstance(make_reporter("quiet"), NullReporter)
    assert isinstance(make_reporter("log", io.StringIO()), LogStatusReporter)
    assert isinstance(make_reporter("live", io.StringIO()), LiveStatusReporter)
    # non-TTY stream under auto -> log
    assert isinstance(make_reporter("auto", io.StringIO()), LogStatusReporter)
