"""Tests for the GUI front-end's non-widget logic.

These exercise the pieces of :mod:`rfnoise.gui` that do not require Dear PyGui or
a display: the widget<->Session sync helpers, the live-plot decay model, and the
worker-thread run controller (driven against the real engine + mock device). The
Dear PyGui render loop itself is not started here.

``rfnoise.gui`` imports without the optional ``dearpygui`` extra (Dear PyGui is
only imported inside ``run_gui``), so these tests run in the default CI env.
"""

import time

import pytest

from rfnoise import gui
from rfnoise.model import FrequencyRange, Session
from rfnoise.status import HopStatus


# -- widget <-> session sync ------------------------------------------------
def _sample_session():
    return Session(
        name="demo",
        device="mock",
        device_options={"verbose": False},
        ranges=[
            FrequencyRange(100_000, 200_000, 10_000),
            FrequencyRange(433_000_000, 434_000_000),
        ],
        dwell_seconds=0.25,
        overlap=0.1,
        seed=7,
        power_min_dbm=-40.0,
        power_max_dbm=-10.0,
    )


def test_session_form_round_trip():
    original = _sample_session()
    values, rows = gui.session_to_form(original)
    rebuilt = gui.collect_session(values, rows, original.device, original.device_options)

    assert rebuilt.name == original.name
    assert rebuilt.dwell_seconds == original.dwell_seconds
    assert rebuilt.overlap == original.overlap
    assert rebuilt.seed == original.seed
    assert rebuilt.power_min_dbm == original.power_min_dbm
    assert rebuilt.power_max_dbm == original.power_max_dbm
    assert len(rebuilt.ranges) == 2
    assert rebuilt.ranges[0].lower_hz == 100_000
    assert rebuilt.ranges[0].upper_hz == 200_000
    assert rebuilt.ranges[0].max_bandwidth_hz == 10_000
    # The second range had no max bw -> stays automatic.
    assert rebuilt.ranges[1].max_bandwidth_hz is None


def test_collect_session_blank_seed_and_power():
    values = {"name": "x", "dwell": 0.5, "overlap": 0.0,
              "seed": "", "power_min": "", "power_max": ""}
    sess = gui.collect_session(values, [], "mock", {})
    assert sess.seed is None
    assert sess.power_min_dbm is None
    assert sess.power_max_dbm is None


def test_collect_session_partial_power_rejected():
    values = {"name": "x", "dwell": 0.5, "overlap": 0.0,
              "seed": "", "power_min": "-40", "power_max": ""}
    with pytest.raises(ValueError):
        gui.collect_session(values, [], "mock", {})


def test_collect_session_swaps_reversed_power():
    values = {"name": "x", "dwell": 0.5, "overlap": 0.0,
              "seed": "", "power_min": "-10", "power_max": "-40"}
    sess = gui.collect_session(values, [], "mock", {})
    assert sess.power_min_dbm == -40.0
    assert sess.power_max_dbm == -10.0


def test_parse_range_row_human_freq():
    rng = gui.parse_range_row("433M", "5.3GHz", "")
    assert rng.lower_hz == 433_000_000
    assert rng.upper_hz == 5_300_000_000
    assert rng.max_bandwidth_hz is None


def test_collect_session_bad_range_reports_index():
    rows = [{"lower": "100k", "upper": "200k", "max_bw": ""},
            {"lower": "nope", "upper": "200k", "max_bw": ""}]
    values = {"name": "x", "dwell": 0.5, "overlap": 0.0,
              "seed": "", "power_min": "", "power_max": ""}
    with pytest.raises(ValueError, match="range 2"):
        gui.collect_session(values, rows, "mock", {})


# -- decay plot model -------------------------------------------------------
def test_decay_alpha_and_prune():
    model = gui.DecayPlotModel(decay_window=10.0, tier_count=5)
    model.add(1, 100.0, now=0.0)
    model.add(2, 200.0, now=5.0)

    # At t=5 the first point is half-faded, the second is fresh.
    assert model.alpha_for(0.0, 5.0) == pytest.approx(0.5)
    assert model.alpha_for(5.0, 5.0) == pytest.approx(1.0)

    # Past the window everything is dropped.
    model.prune(now=100.0)
    assert not model._points


def test_decay_tiers_bucket_by_age():
    model = gui.DecayPlotModel(decay_window=10.0, tier_count=5)
    model.add(1, 100.0, now=0.0)   # oldest
    model.add(2, 200.0, now=9.9)   # freshest
    tiers = model.tiers(now=9.9)
    # Every live point lands in exactly one tier; total count is preserved.
    total = sum(len(xs) for _alpha, xs, _ys in tiers)
    assert total == 2
    # Freshest tier (index 0) holds the newest point.
    assert 2 in tiers[0][1]


def test_decay_tiers_stable_length():
    model = gui.DecayPlotModel(tier_count=6)
    assert len(model.tiers(now=0.0)) == 6


# -- plot axis extents (frequency vs strength view) ------------------------
def test_frequency_extent_spans_all_ranges():
    sess = Session(ranges=[
        FrequencyRange(433_000_000, 434_000_000),
        FrequencyRange(100_000, 200_000),
        FrequencyRange(2_400_000_000, 2_500_000_000),
    ])
    assert gui.frequency_extent(sess) == (100_000.0, 2_500_000_000.0)


def test_frequency_extent_none_without_ranges():
    assert gui.frequency_extent(Session(ranges=[])) is None


def test_power_extent_uses_session_range():
    sess = Session(power_min_dbm=-40.0, power_max_dbm=-10.0)
    assert gui.power_extent(sess) == (-40.0, -10.0)


def test_power_extent_defaults_without_range():
    assert gui.power_extent(Session()) == gui.DEFAULT_DBM_RANGE


def test_hop_plot_y_uses_power_or_baseline():
    assert gui.hop_plot_y(-33.5) == -33.5
    assert gui.hop_plot_y(None) == gui.NO_POWER_DBM


def test_dbm_ticks_span_endpoints():
    ticks = gui.dbm_ticks(-70.0, -15.0, count=6)
    assert len(ticks) == 6
    assert ticks[0] == ("-70", -70.0)
    assert ticks[-1] == ("-15", -15.0)
    # positions are monotonically increasing dBm values
    positions = [pos for _lbl, pos in ticks]
    assert positions == sorted(positions)


# -- run controller (real engine, mock device, worker thread) --------------
def _run_session():
    return Session(
        name="run",
        device="mock",
        device_options={"sleep": False},
        ranges=[FrequencyRange(100_000, 200_000, 10_000)],
        dwell_seconds=0.0,
        seed=1,
    )


def test_run_controller_start_stop_and_drain():
    controller = gui.RunController()
    controller.start(_run_session())
    # Let the worker produce some hops, then stop it.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and not controller.drain():
        time.sleep(0.01)
    controller.stop()
    assert not controller.running

    # Drained items are HopStatus objects the plot can consume.
    controller.start(_run_session())
    time.sleep(0.05)
    controller.stop()
    hops = controller.drain()
    for hop in hops:
        assert isinstance(hop, HopStatus)
        assert hop.center_hz > 0


def test_run_controller_setup_error_propagates():
    # A session with no ranges fails validation in NoiseGenerator's ctor.
    bad = Session(name="bad", device="mock", ranges=[])
    controller = gui.RunController()
    with pytest.raises(Exception):
        controller.start(bad)
    assert not controller.running


# -- display availability guard -------------------------------------------
def test_display_available_needs_env_on_linux(monkeypatch):
    monkeypatch.setattr(gui.sys, "platform", "linux")
    # Pretend any X connection would succeed so we isolate the env-var logic.
    monkeypatch.setattr(gui, "_can_open_x_display", lambda: True)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    assert gui.display_available() is False

    monkeypatch.setenv("DISPLAY", ":0")
    assert gui.display_available() is True

    # Wayland advertised but no X DISPLAY: still considered available.
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    assert gui.display_available() is True


def test_display_available_x_connection_refused(monkeypatch):
    """DISPLAY set but the X server refuses the connection -> unavailable."""
    monkeypatch.setattr(gui.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setattr(gui, "_can_open_x_display", lambda: False)
    assert gui.display_available() is False


def test_display_available_true_on_windows_macos(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for platform in ("win32", "darwin"):
        monkeypatch.setattr(gui.sys, "platform", platform)
        assert gui.display_available() is True


def test_run_gui_without_display_raises(monkeypatch):
    monkeypatch.setattr(gui, "display_available", lambda: False)
    with pytest.raises(gui.DisplayUnavailableError):
        gui.run_gui(Session())


def test_cli_gui_no_display(monkeypatch, capsys):
    from rfnoise import cli

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    # Pretend dearpygui is importable but no display exists.
    monkeypatch.setattr(gui, "display_available", lambda: False)

    class _Args:
        session = None

    rc = cli._cmd_gui(_Args())
    assert rc == 2
    err = capsys.readouterr().err
    assert "graphical display" in err
    assert "rfnoise ui" in err


# -- CLI wiring: missing optional dependency ------------------------------
def test_cli_gui_missing_dep(monkeypatch, capsys):
    import importlib.util

    from rfnoise import cli

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "dearpygui":
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    class _Args:
        session = None

    rc = cli._cmd_gui(_Args())
    assert rc == 2
    assert "pip install" in capsys.readouterr().err
