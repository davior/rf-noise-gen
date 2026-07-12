"""Dear PyGui graphical front-end for rfnoise.

A third front-end (alongside :mod:`rfnoise.cli` and :mod:`rfnoise.interactive`)
that drives the exact same UI-agnostic core: :class:`~rfnoise.engine.NoiseGenerator`,
:class:`~rfnoise.model.Session`, the :mod:`~rfnoise.devices` registry and
:mod:`~rfnoise.session` persistence. Nothing in the engine changes -- the GUI is
just another caller.

Dear PyGui is an *optional* dependency (``pip install -e ".[gui]"``); importing
this module without it raises ``ImportError``, which :mod:`rfnoise.cli` turns into
a friendly install hint. To keep that import path testable and to allow the
non-widget logic to be unit-tested without a display, the widget<->Session sync
helpers and the live-plot decay maths live in plain functions/classes below and
do not touch ``dpg`` until :func:`run_gui` actually builds the window.

Threading model
---------------
``NoiseGenerator.run()`` blocks and invokes its ``on_hop`` callback on the calling
thread. The GUI therefore runs the generator on a daemon worker thread; the
callback only pushes :class:`~rfnoise.status.HopStatus` objects onto a thread-safe
:class:`queue.Queue`. The Dear PyGui render loop drains that queue on the UI
thread once per frame and is the *only* place ``dpg`` state is mutated from hop
data. ``stop()`` is cooperative (it breaks at the next hop boundary, so it can lag
by up to one ``dwell_seconds``); the UI accounts for that when re-enabling Run and
on window close.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import session as session_store
from .devices import (
    DeviceError,
    create_device,
    device_keys,
    get_device_class,
)
from .engine import ConfigurationError, NoiseGenerator
from .freq import format_freq, parse_freq
from .model import FrequencyRange, Session
from .status import HopStatus


# ---------------------------------------------------------------------------
# Widget <-> Session sync (pure logic, no Dear PyGui -- unit-testable)
# ---------------------------------------------------------------------------
#
# A "form" is any mapping-like object exposing the current value of each named
# field (in the real GUI, a thin wrapper over ``dpg.get_value``/``set_value``;
# in tests, a plain dict). Keeping the translation here means the fiddly
# string<->number and range-table parsing is covered without opening a window.

# Per-device option field specs the option sub-form renders. Each entry is
# (key, label, kind, default) where kind is "int" | "float" | "bool" | "text".
DEVICE_OPTION_FIELDS: Dict[str, List[Tuple[str, str, str, Any]]] = {
    "mock": [
        ("max_bandwidth_hz", "max bandwidth (Hz)", "int", 20_000_000),
        ("verbose", "verbose logging", "bool", False),
    ],
    "tinysa": [
        ("port", "serial port", "text", ""),
        ("mode", "burst mode (sweep/cw)", "text", "sweep"),
        ("level", "output level (dBm)", "int", -30),
        ("baudrate", "baud rate", "int", 115200),
    ],
    "hackrf": [
        ("txvga_gain", "TX VGA gain (0-47)", "int", 30),
        ("amp", "enable TX amplifier", "bool", False),
    ],
    "rtlsdr": [],
}


def _parse_optional_int(raw: str) -> Optional[int]:
    """Parse a seed-like field: blank -> None, else an int (raises ValueError)."""
    raw = (raw or "").strip()
    if not raw:
        return None
    return int(raw)


def _parse_optional_float(raw: str) -> Optional[float]:
    raw = (raw or "").strip()
    if not raw:
        return None
    return float(raw)


def parse_range_row(lower: str, upper: str, max_bw: str) -> FrequencyRange:
    """Build a :class:`FrequencyRange` from three raw text fields.

    Frequencies accept the same human-friendly strings as the rest of rfnoise
    (``"433M"``, ``"5.3 GHz"``). ``max_bw`` blank means "use the device's
    automatic maximum". Raises ``ValueError`` on bad input (propagated to the
    UI as a status message).
    """
    lo = parse_freq(lower)
    hi = parse_freq(upper)
    bw = parse_freq(max_bw) if (max_bw or "").strip() else None
    return FrequencyRange(lo, hi, bw)


def collect_session(values: Dict[str, Any], rows: List[Dict[str, str]],
                    device: str, device_options: Dict[str, Any]) -> Session:
    """Assemble a :class:`Session` from raw form values.

    ``values`` holds the scalar fields (name/dwell/seed/overlap/power); ``rows``
    is the list of range rows (each a dict with ``lower``/``upper``/``max_bw``
    text). Raises ``ValueError`` if any field is malformed so the caller can
    surface a single clear message instead of a half-built session.
    """
    ranges: List[FrequencyRange] = []
    for i, row in enumerate(rows):
        try:
            ranges.append(parse_range_row(
                row.get("lower", ""), row.get("upper", ""), row.get("max_bw", "")))
        except ValueError as exc:
            raise ValueError(f"range {i + 1}: {exc}") from exc

    pmin = _parse_optional_float(values.get("power_min", ""))
    pmax = _parse_optional_float(values.get("power_max", ""))
    if (pmin is None) != (pmax is None):
        raise ValueError("set both min and max dBm, or leave both blank")
    if pmin is not None and pmax is not None and pmax < pmin:
        pmin, pmax = pmax, pmin

    return Session(
        name=(values.get("name") or "untitled").strip() or "untitled",
        device=device,
        device_options=dict(device_options),
        ranges=ranges,
        dwell_seconds=float(values.get("dwell", 0.5) or 0.5),
        overlap=float(values.get("overlap", 0.0) or 0.0),
        seed=_parse_optional_int(values.get("seed", "")),
        power_min_dbm=pmin,
        power_max_dbm=pmax,
    )


def session_to_form(session: Session) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    """Inverse of :func:`collect_session`: scalar values + range rows for a Session."""
    values = {
        "name": session.name,
        "dwell": session.dwell_seconds,
        "overlap": session.overlap,
        "seed": "" if session.seed is None else str(session.seed),
        "power_min": "" if session.power_min_dbm is None else f"{session.power_min_dbm:g}",
        "power_max": "" if session.power_max_dbm is None else f"{session.power_max_dbm:g}",
    }
    rows = [
        {
            "lower": format_freq(r.lower_hz),
            "upper": format_freq(r.upper_hz),
            "max_bw": "" if r.max_bandwidth_hz is None else format_freq(r.max_bandwidth_hz),
        }
        for r in session.ranges
    ]
    return values, rows


# ---------------------------------------------------------------------------
# Live-plot decay model (pure logic, no Dear PyGui -- unit-testable)
# ---------------------------------------------------------------------------
class DecayPlotModel:
    """Fading scatter of recently-played frequencies.

    Each hop is stored with the wall-clock time it was plotted. A point's alpha
    ramps from 1.0 (just played) to 0.0 over ``decay_window`` seconds; points
    older than the window are dropped so both the plot and this buffer stay
    bounded regardless of run length. Because a single ImPlot series cannot vary
    alpha per point, :meth:`tiers` buckets the live points into a fixed number of
    opacity bands, each of which the GUI renders as one scatter series.
    """

    def __init__(self, decay_window: float = 10.0, tier_count: int = 6):
        if decay_window <= 0:
            raise ValueError("decay_window must be positive")
        if tier_count < 1:
            raise ValueError("tier_count must be >= 1")
        self.decay_window = float(decay_window)
        self.tier_count = int(tier_count)
        # Each entry: (played_at, x, y). x/y are plot coordinates -- the GUI uses
        # x = center frequency (Hz), y = strength (dBm).
        self._points: List[Tuple[float, float, float]] = []

    def add(self, x: float, y: float, now: Optional[float] = None) -> None:
        self._points.append((time.monotonic() if now is None else now, float(x), float(y)))

    def prune(self, now: Optional[float] = None) -> None:
        """Drop points that have fully decayed."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.decay_window
        if self._points and self._points[0][0] < cutoff:
            self._points = [p for p in self._points if p[0] >= cutoff]

    def alpha_for(self, played_at: float, now: float) -> float:
        return max(0.0, 1.0 - (now - played_at) / self.decay_window)

    def tiers(self, now: Optional[float] = None) -> List[Tuple[float, List[float], List[float]]]:
        """Return ``tier_count`` bands as ``(alpha, xs, ys)``, freshest first.

        Bands are evenly spaced in alpha; a point falls in the band matching its
        current opacity. Empty bands are still returned (with empty coordinate
        lists) so the GUI can keep a stable set of series and just update data.
        """
        now = time.monotonic() if now is None else now
        self.prune(now)
        # Representative alpha for band i (i=0 freshest): centre of its slice.
        bands: List[Tuple[float, List[float], List[float]]] = [
            (((self.tier_count - i) - 0.5) / self.tier_count, [], [])
            for i in range(self.tier_count)
        ]
        for played_at, x, y in self._points:
            a = self.alpha_for(played_at, now)
            if a <= 0.0:
                continue
            # Map alpha in (0,1] to a band index 0..tier_count-1 (freshest=0).
            idx = int((1.0 - a) * self.tier_count)
            if idx >= self.tier_count:
                idx = self.tier_count - 1
            bands[idx][1].append(x)
            bands[idx][2].append(y)
        return bands

    def clear(self) -> None:
        self._points.clear()


# ---------------------------------------------------------------------------
# Plot axis extents (pure logic, no Dear PyGui -- unit-testable)
# ---------------------------------------------------------------------------
# The live plot is a fixed frequency-vs-strength view: X spans the configured
# ranges, Y spans the strength range. When a session has no power range every
# hop's dBm is None, so those points sit on a single baseline.
DEFAULT_DBM_RANGE: Tuple[float, float] = (-100.0, 10.0)
NO_POWER_DBM: float = 0.0  # y for hops with no strength info (no power range)


def frequency_extent(session: Session) -> Optional[Tuple[float, float]]:
    """(min lower, max upper) Hz across all ranges for the X axis, or None.

    ``None`` when the session has no ranges yet (the axis stays auto-scaled).
    """
    if not session.ranges:
        return None
    lo = float(min(r.lower_hz for r in session.ranges))
    hi = float(max(r.upper_hz for r in session.ranges))
    if hi <= lo:
        hi = lo + 1.0
    return (lo, hi)


def power_extent(session: Session) -> Tuple[float, float]:
    """(min, max) dBm for the Y axis: the session's power range, else a default."""
    if session.has_power_range:
        lo, hi = float(session.power_min_dbm), float(session.power_max_dbm)
        if hi <= lo:
            hi = lo + 1.0
        return (lo, hi)
    return DEFAULT_DBM_RANGE


def hop_plot_y(power_dbm: Optional[float]) -> float:
    """Y coordinate for a hop: its dBm, or the no-strength baseline if unknown."""
    return NO_POWER_DBM if power_dbm is None else float(power_dbm)


def dbm_ticks(lo: float, hi: float, count: int = 6) -> List[Tuple[str, float]]:
    """Evenly spaced ``(label, dBm)`` ticks spanning ``lo..hi`` for the Y axis.

    Bars are drawn as height-above-floor (see :func:`hop_plot_y` usage in the
    GUI), so the axis is relabelled with these real dBm values at their shifted
    positions.
    """
    count = max(2, count)
    step = (hi - lo) / (count - 1)
    return [(f"{lo + i * step:.0f}", lo + i * step) for i in range(count)]


# ---------------------------------------------------------------------------
# Run controller: engine on a worker thread, HopStatus over a queue
# ---------------------------------------------------------------------------
class RunController:
    """Owns a single background run and the queue the UI drains from.

    Kept UI-free so the start/stop/threading contract is exercised in tests with
    the real engine + mock device (no Dear PyGui). The UI thread calls
    :meth:`start`/:meth:`stop`/:meth:`drain`; the worker thread only feeds the
    queue via the ``on_hop`` callback.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[HopStatus]" = queue.Queue()
        self._gen: Optional[NoiseGenerator] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    @property
    def error(self) -> Optional[str]:
        return self._error

    def start(self, session: Session) -> None:
        """Build a device+generator and run it on a daemon thread.

        Raises on setup failure (bad config / device) so the caller can show the
        message and leave the UI in the stopped state.
        """
        if self.running:
            raise RuntimeError("a run is already in progress")
        opts = dict(session.device_options)
        if session.device == "mock":
            # Sleep for real dwell time so the live plot animates like hardware.
            opts.setdefault("sleep", True)
        device = create_device(session.device, **opts)  # may raise DeviceError
        gen = NoiseGenerator(device, session)            # may raise ConfigurationError
        self._gen = gen
        self._error = None
        # Fresh queue per run so a previous run's tail can't leak in.
        self._queue = queue.Queue()

        def _worker() -> None:
            try:
                gen.run(on_hop=self._queue.put)
            except Exception as exc:  # pragma: no cover - surfaced via error attr
                with self._lock:
                    self._error = str(exc)

        self._thread = threading.Thread(target=_worker, name="rfnoise-run", daemon=True)
        self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        """Ask the current run to stop and wait briefly for the worker to exit."""
        if self._gen is not None:
            self._gen.stop()
        t = self._thread
        if t is not None:
            t.join(timeout=join_timeout)

    def drain(self) -> List[HopStatus]:
        """Return all HopStatus objects queued since the last drain (UI thread)."""
        out: List[HopStatus] = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out


# ---------------------------------------------------------------------------
# The Dear PyGui window
# ---------------------------------------------------------------------------
class DisplayUnavailableError(RuntimeError):
    """Raised when the GUI cannot open because there is no graphical display."""


NO_DISPLAY_HINT = (
    "the GUI needs a graphical display, but none could be opened.\n"
    "\n"
    "  - DISPLAY / WAYLAND_DISPLAY may be unset, or the X server may be\n"
    "    rejecting the connection ('Authorization required').\n"
    "  - Over SSH, reconnect with X forwarding:  ssh -X user@host\n"
    "  - On a Wayland desktop, launch from a terminal inside that session so\n"
    "    DISPLAY and XAUTHORITY are inherited, or point XAUTHORITY at the\n"
    "    Xwayland cookie, e.g.:\n"
    "      export XAUTHORITY=$(ls /run/user/$(id -u)/.mutter-Xwaylandauth.* | head -1)\n"
    "\n"
    "No display? Use the text interface instead:\n"
    "  rfnoise ui              # interactive editor\n"
    "  rfnoise run <session>   # headless run"
)


def _env_has_display() -> bool:
    """True if the environment advertises a display server (env-var check only)."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _can_open_x_display() -> bool:
    """Try to actually open the X11 display named by ``$DISPLAY``.

    Dear PyGui's GLFW backend is X11-only and, when the server refuses the
    connection (missing/short xauth cookie -- 'Authorization required'), it
    aborts the *process* at the C level (``SIGABRT``), which no Python
    ``try/except`` can catch. We pre-flight the connection with libX11 so that
    failure becomes a clean, catchable error instead of a core dump. If libX11
    cannot be loaded we return ``True`` and let GLFW try.
    """
    import ctypes
    import ctypes.util

    for name in (ctypes.util.find_library("X11"), "libX11.so.6", "libX11.so"):
        if not name:
            continue
        try:
            x11 = ctypes.CDLL(name)
        except OSError:
            continue
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        dpy = x11.XOpenDisplay(None)  # None -> use $DISPLAY
        if dpy:
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            x11.XCloseDisplay(dpy)
            return True
        return False
    return True  # libX11 unavailable to probe with -- don't block, let GLFW try


def display_available() -> bool:
    """True if a graphical display can actually be opened.

    Windows/macOS always can. On Linux/BSD we need a display server advertised
    in the environment *and*, when it's X11, an X server that accepts the
    connection -- otherwise GLFW aborts with raw errors (or a core dump).
    """
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    if not _env_has_display():
        return False
    if os.environ.get("DISPLAY"):
        return _can_open_x_display()
    return True  # Wayland-only advertised; GLFW/XWayland will make the attempt


def run_gui(session: Optional[Session] = None) -> None:
    """Launch the graphical editor/runner. Blocks until the window closes.

    Importing Dear PyGui is deferred to here so ``import rfnoise.gui`` (and the
    unit tests above) work without the optional extra installed. Raises
    :class:`DisplayUnavailableError` if no graphical display is present.
    """
    if not display_available():
        raise DisplayUnavailableError(NO_DISPLAY_HINT)

    import dearpygui.dearpygui as dpg

    session = session or Session()

    controller = RunController()
    plot_model = DecayPlotModel()
    state: Dict[str, Any] = {"row_ids": [], "device": session.device}
    # Form values for the initial session -- used to seed the widgets at build
    # time (name/dwell/seed/power) and populate the range table.
    _initial_values, _initial_rows = session_to_form(session)

    dpg.create_context()

    # -- helpers bound to the live context --------------------------------
    def set_status(msg: str) -> None:
        dpg.set_value("status_text", msg)

    def current_device_options() -> Dict[str, Any]:
        opts: Dict[str, Any] = {}
        for key, _label, kind, _default in DEVICE_OPTION_FIELDS.get(state["device"], []):
            tag = f"devopt_{key}"
            if not dpg.does_item_exist(tag):
                continue
            val = dpg.get_value(tag)
            if kind == "int":
                try:
                    val = int(val)
                except (TypeError, ValueError):
                    continue
            opts[key] = val
        return opts

    def gather_session() -> Session:
        values = {
            "name": dpg.get_value("f_name"),
            "dwell": dpg.get_value("f_dwell"),
            "overlap": dpg.get_value("f_overlap"),
            "seed": dpg.get_value("f_seed"),
            "power_min": dpg.get_value("f_power_min"),
            "power_max": dpg.get_value("f_power_max"),
        }
        rows = []
        for row_id in state["row_ids"]:
            rows.append({
                "lower": dpg.get_value(f"{row_id}_lower"),
                "upper": dpg.get_value(f"{row_id}_upper"),
                "max_bw": dpg.get_value(f"{row_id}_maxbw"),
            })
        return collect_session(values, rows, state["device"], current_device_options())

    # -- range table ------------------------------------------------------
    def add_range_row(lower: str = "", upper: str = "", max_bw: str = "") -> None:
        row_id = f"row_{dpg.generate_uuid()}"
        state["row_ids"].append(row_id)
        with dpg.table_row(parent="ranges_table", tag=row_id):
            dpg.add_input_text(tag=f"{row_id}_lower", default_value=lower, width=120)
            dpg.add_input_text(tag=f"{row_id}_upper", default_value=upper, width=120)
            dpg.add_input_text(tag=f"{row_id}_maxbw", default_value=max_bw,
                               hint="auto", width=120)
            dpg.add_button(label="remove",
                           callback=lambda s, a, u=row_id: remove_range_row(u))

    def remove_range_row(row_id: str) -> None:
        if row_id in state["row_ids"]:
            state["row_ids"].remove(row_id)
        if dpg.does_item_exist(row_id):
            dpg.delete_item(row_id)

    def clear_ranges() -> None:
        for row_id in list(state["row_ids"]):
            remove_range_row(row_id)

    # -- device option sub-form ------------------------------------------
    def rebuild_device_options(preset: Optional[Dict[str, Any]] = None) -> None:
        dpg.delete_item("devopts_group", children_only=True)
        preset = preset or {}
        fields = DEVICE_OPTION_FIELDS.get(state["device"], [])
        try:
            desc = get_device_class(state["device"])().describe()
            dpg.add_text(desc.splitlines()[0], parent="devopts_group", wrap=440)
        except Exception:
            pass
        if not fields:
            if state["device"] == "rtlsdr":
                dpg.add_text("RTL-SDR is receive-only and cannot run a broadcast.",
                             parent="devopts_group", color=(230, 160, 60))
            return
        for key, label, kind, default in fields:
            value = preset.get(key, default)
            tag = f"devopt_{key}"
            if kind == "bool":
                dpg.add_checkbox(label=label, tag=tag, default_value=bool(value),
                                 parent="devopts_group")
            elif kind == "int":
                dpg.add_input_int(label=label, tag=tag, default_value=int(value),
                                  step=0, width=160, parent="devopts_group")
            else:  # text
                dpg.add_input_text(label=label, tag=tag, default_value=str(value),
                                   width=200, parent="devopts_group")

    def on_device_change(sender, app_data) -> None:
        state["device"] = app_data
        rebuild_device_options()

    # -- load a session into every widget --------------------------------
    def populate_from_session(sess: Session) -> None:
        values, rows = session_to_form(sess)
        dpg.set_value("f_name", values["name"])
        dpg.set_value("f_dwell", float(values["dwell"]))
        dpg.set_value("f_overlap", float(values["overlap"]))
        dpg.set_value("f_seed", values["seed"])
        dpg.set_value("f_power_min", values["power_min"])
        dpg.set_value("f_power_max", values["power_max"])
        state["device"] = sess.device
        dpg.set_value("f_device", sess.device)
        rebuild_device_options(sess.device_options)
        clear_ranges()
        for row in rows:
            add_range_row(row["lower"], row["upper"], row["max_bw"])
        apply_plot_axes(sess)

    # -- run / stop -------------------------------------------------------
    def set_running_ui(running: bool) -> None:
        dpg.set_item_label("run_button", "Stop" if running else "Run")

    def on_run_toggle() -> None:
        if controller.running:
            controller.stop()
            set_running_ui(False)
            set_status("stopped")
            return
        try:
            sess = gather_session()
        except ValueError as exc:
            set_status(f"cannot start: {exc}")
            return
        try:
            plot_model.clear()
            controller.start(sess)
        except (ConfigurationError, DeviceError, ValueError) as exc:
            set_status(f"cannot start: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive
            set_status(f"cannot start: {exc}")
            return
        apply_plot_axes(sess)
        set_running_ui(True)
        note = "" if sess.has_power_range else " (no power range: strength unset)"
        set_status(f"running '{sess.name}' on {sess.device}{note}")

    def on_validate() -> None:
        try:
            sess = gather_session()
            opts = dict(sess.device_options)
            if sess.device == "mock":
                opts.setdefault("sleep", False)
            device = create_device(sess.device, **opts)
            gen = NoiseGenerator(device, sess)
        except (ConfigurationError, DeviceError, ValueError, Exception) as exc:
            set_status(f"invalid: {exc}")
            return
        apply_plot_axes(sess)
        preview = gen.plan(min(10, max(1, len(gen.bands))))
        centers = ", ".join(format_freq(b.center_hz) for b in preview[:5])
        set_status(f"OK: {len(gen.bands)} bands. next: {centers} ...")

    # -- save / load dialogs ---------------------------------------------
    def on_save_selected(sender, app_data) -> None:
        path = app_data["file_path_name"]
        try:
            sess = gather_session()
            session_store.save(sess, path)
        except (ValueError, OSError) as exc:
            set_status(f"save failed: {exc}")
            return
        set_status(f"saved {path}")

    def on_load_selected(sender, app_data) -> None:
        path = app_data["file_path_name"]
        try:
            sess = session_store.load(path)
        except (ValueError, OSError, KeyError) as exc:
            set_status(f"load failed: {exc}")
            return
        populate_from_session(sess)
        set_status(f"loaded {path}")

    def apply_plot_axes(sess: Session) -> None:
        """Pin the X/Y axes to the session's frequency + strength extents.

        A static spectrum-style bar view: frequency on X across the ranges,
        strength (dBm) on Y. Bars are drawn as height above the strength floor
        (ImPlot bars rise from 0), so the Y axis is shifted and relabelled with
        real dBm values. Called at build and whenever the session changes.
        """
        fx = frequency_extent(sess)
        if fx is not None:
            lo, hi = fx
            span = hi - lo
            pad = span * 0.03 or 1.0
            dpg.set_axis_limits("plot_x", lo - pad, hi + pad)
            state["bar_weight"] = max(span * 0.006, 1.0)
        else:
            dpg.set_axis_limits_auto("plot_x")
            state["bar_weight"] = 1.0

        plo, phi = power_extent(sess)
        state["y_floor"] = plo  # bar heights are measured up from here
        ppad = (phi - plo) * 0.05 or 1.0
        # Shifted space: 0 == plo. Keep the same visible padding as real dBm.
        dpg.set_axis_limits("plot_y", -ppad, (phi - plo) + ppad)
        dpg.set_axis_ticks("plot_y",
                           tuple((lbl, pos - plo) for lbl, pos in dbm_ticks(plo, phi)))
        for i in range(plot_model.tier_count):
            series = f"decay_series_{i}"
            if dpg.does_item_exist(series):
                dpg.configure_item(series, weight=state["bar_weight"])

    # -- per-frame plot + queue drain (UI thread only) -------------------
    def refresh_plot() -> None:
        drained = controller.drain()
        for hop in drained:
            # x = center frequency, y = strength (dBm) -- baseline if no power.
            plot_model.add(hop.center_hz, hop_plot_y(hop.power_dbm))
        latest = drained[-1] if drained else None
        if latest is not None:
            dpg.set_value("status_text", latest.line())
        now = time.monotonic()
        floor = state.get("y_floor", 0.0)
        # Each tier has a fixed alpha (set once when its theme is built); bars
        # migrate between tiers as they age, so we only update series *data*.
        # y is shifted to a height above the strength floor (bars rise from 0).
        for i, (_alpha, xs, ys) in enumerate(plot_model.tiers(now)):
            series = f"decay_series_{i}"
            if dpg.does_item_exist(series):
                dpg.set_value(series, [list(xs), [y - floor for y in ys]])
        # Reflect a worker that stopped on its own (duration/iterations/error).
        if not controller.running and dpg.get_item_label("run_button") == "Stop":
            set_running_ui(False)
            if controller.error:
                set_status(f"run error: {controller.error}")

    # -- build the window -------------------------------------------------
    with dpg.window(tag="primary"):
        with dpg.group(horizontal=True):
            # Left: the editor.
            with dpg.child_window(width=470, autosize_y=True):
                dpg.add_text("Session")
                dpg.add_input_text(label="name", tag="f_name",
                                   default_value=session.name, width=260)
                dpg.add_input_float(label="dwell (s)", tag="f_dwell",
                                    default_value=session.dwell_seconds,
                                    width=160, step=0)
                dpg.add_input_float(label="overlap", tag="f_overlap",
                                    default_value=session.overlap, width=160, step=0)
                dpg.add_input_text(label="seed (blank=random)", tag="f_seed",
                                   default_value="" if session.seed is None else str(session.seed),
                                   width=160)
                with dpg.group(horizontal=True):
                    dpg.add_input_text(label="min dBm", tag="f_power_min",
                                       default_value=_initial_values["power_min"],
                                       width=90)
                    dpg.add_input_text(label="max dBm", tag="f_power_max",
                                       default_value=_initial_values["power_max"],
                                       width=90)

                dpg.add_separator()
                dpg.add_text("Device")
                dpg.add_combo(device_keys(), label="device", tag="f_device",
                              default_value=session.device, width=200,
                              callback=on_device_change)
                dpg.add_group(tag="devopts_group")

                dpg.add_separator()
                dpg.add_text("Frequency ranges (e.g. 433M, 5.3GHz; bw blank = auto)")
                with dpg.table(tag="ranges_table", header_row=True,
                               policy=dpg.mvTable_SizingFixedFit):
                    dpg.add_table_column(label="lower")
                    dpg.add_table_column(label="upper")
                    dpg.add_table_column(label="max bw")
                    dpg.add_table_column(label="")
                dpg.add_button(label="add range", callback=lambda: add_range_row())

                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Run", tag="run_button", callback=on_run_toggle)
                    dpg.add_button(label="Validate", callback=on_validate)
                    dpg.add_button(label="Save",
                                   callback=lambda: dpg.show_item("save_dialog"))
                    dpg.add_button(label="Load",
                                   callback=lambda: dpg.show_item("load_dialog"))

            # Right: live status + fading strength-vs-frequency plot.
            with dpg.child_window(autosize_x=True, autosize_y=True):
                dpg.add_text("", tag="status_text")
                with dpg.plot(label="Live spectrum -- strength vs frequency (fading)",
                              height=-1, width=-1):
                    dpg.add_plot_axis(dpg.mvXAxis, label="frequency (Hz)", tag="plot_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="strength (dBm)",
                                       tag="plot_y"):
                        # Dimmest (oldest) tier first so the freshest bars draw
                        # on top when they overlap at the same frequency.
                        for i in reversed(range(plot_model.tier_count)):
                            dpg.add_bar_series([], [], weight=1.0,
                                               tag=f"decay_series_{i}")

    # Per-tier themes: newest tiers brightest. Each tier has a fixed bar-fill
    # alpha; bars migrate between tiers as they age, so the column fades.
    for i in range(plot_model.tier_count):
        alpha = int(((plot_model.tier_count - i) / plot_model.tier_count) * 255)
        with dpg.theme(tag=f"decay_theme_{i}") as theme_id:
            with dpg.theme_component(dpg.mvBarSeries):
                dpg.add_theme_color(dpg.mvPlotCol_Fill, (80, 170, 255, alpha),
                                    category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (80, 170, 255, alpha),
                                    category=dpg.mvThemeCat_Plots)
        dpg.bind_item_theme(f"decay_series_{i}", theme_id)

    # File dialogs.
    with dpg.file_dialog(directory_selector=False, show=False, tag="save_dialog",
                         default_path=session_store.DEFAULT_SESSION_DIR,
                         default_filename=session.name or "session",
                         callback=on_save_selected, width=600, height=400):
        dpg.add_file_extension(".json")
    with dpg.file_dialog(directory_selector=False, show=False, tag="load_dialog",
                         default_path=session_store.DEFAULT_SESSION_DIR,
                         callback=on_load_selected, width=600, height=400):
        dpg.add_file_extension(".json")

    # Populate the editor from the initial session (builds option form + rows).
    rebuild_device_options(session.device_options)
    for row in _initial_rows:
        add_range_row(row["lower"], row["upper"], row["max_bw"])
    apply_plot_axes(session)

    dpg.set_frame_callback(1, lambda: None)  # ensure a first frame is scheduled

    # DISPLAY can be set yet unusable (broken X forwarding, dead server): GLFW
    # then aborts here. Turn that into the same actionable message.
    try:
        dpg.create_viewport(title="rfnoise", width=1000, height=640)
        dpg.setup_dearpygui()
        dpg.show_viewport()
    except Exception as exc:
        dpg.destroy_context()
        raise DisplayUnavailableError(f"{NO_DISPLAY_HINT}\n\n(display error: {exc})") from exc
    dpg.set_primary_window("primary", True)

    try:
        while dpg.is_dearpygui_running():
            refresh_plot()
            dpg.render_dearpygui_frame()
    finally:
        # Never leave a worker transmitting after the window is gone.
        if controller.running:
            controller.stop()
        dpg.destroy_context()
