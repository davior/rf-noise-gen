"""Menu-driven interactive interface for building and running sessions.

Stdlib only (uses ``input``), so it runs anywhere. Frequencies may be entered
human-friendly (``100k``, ``2.4M``, ``5.3GHz``). Sessions are persisted to JSON
files under the sessions directory and can be reopened later.
"""

from __future__ import annotations

import os
from typing import Optional

from . import session as session_store
from .devices import create_device, device_keys, get_device_class
from .engine import ConfigurationError, NoiseGenerator, validate
from .freq import format_freq, parse_freq
from .model import FrequencyRange, Session
from .status import make_reporter


# -- small input helpers ---------------------------------------------------
def _prompt(text: str, default: Optional[str] = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        raw = input(f"{text}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return raw or (default or "")


def _prompt_freq(text: str, default: Optional[int] = None) -> Optional[int]:
    dflt = format_freq(default) if default is not None else None
    while True:
        raw = _prompt(text, dflt)
        if not raw:
            return default
        try:
            return parse_freq(raw)
        except ValueError as exc:
            print(f"  ! {exc}")


def _prompt_float(text: str, default: float) -> float:
    while True:
        raw = _prompt(text, str(default))
        try:
            return float(raw)
        except ValueError:
            print("  ! enter a number")


def _confirm(text: str) -> bool:
    return _prompt(f"{text} (y/N)").lower().startswith("y")


# -- editor screens --------------------------------------------------------
def _device_defaults(key: str) -> dict:
    """Sensible default options per device for the interactive editor."""
    if key == "mock":
        return {}
    if key == "tinysa":
        return {"mode": "sweep", "port": ""}
    if key == "hackrf":
        return {"txvga_gain": 30, "amp": False}
    return {}


def _edit_ranges(session: Session) -> None:
    while True:
        print("\n-- Ranges --")
        if not session.ranges:
            print("  (none yet)")
        for i, rng in enumerate(session.ranges):
            print(f"  {i}: {rng}")
        print("  [a]dd  [d]elete  [b]ack")
        choice = _prompt("ranges>").lower()
        if choice in ("a", "add"):
            lower = _prompt_freq("  lower bound")
            upper = _prompt_freq("  upper bound")
            if lower is None or upper is None:
                print("  ! both bounds required")
                continue
            print("  max bandwidth per burst -- blank = use device's automatic max")
            bw = _prompt_freq("  max bandwidth (optional)")
            try:
                session.ranges.append(FrequencyRange(lower, upper, bw))
            except ValueError as exc:
                print(f"  ! {exc}")
        elif choice in ("d", "delete"):
            idx = _prompt("  index to delete")
            if idx.isdigit() and int(idx) < len(session.ranges):
                session.ranges.pop(int(idx))
            else:
                print("  ! invalid index")
        elif choice in ("b", "back", ""):
            return


def _choose_device(session: Session) -> None:
    keys = device_keys()
    print("\n-- Device --")
    for i, key in enumerate(keys):
        caps = get_device_class(key)().capabilities if key in ("mock", "rtlsdr") else None
        note = ""
        try:
            dev = create_device(key, **_device_defaults(key))
            caps = dev.capabilities
            if caps.max_bandwidth_hz is None:
                bw = f"auto (default {format_freq(caps.default_band_width)})"
            else:
                bw = format_freq(caps.max_bandwidth_hz)
            tx = "" if caps.can_transmit else " [RECEIVE ONLY]"
            note = f" - max bw {bw}{tx}"
        except Exception:
            pass
        marker = "*" if key == session.device else " "
        print(f"  {marker}{i}: {key}{note}")
    choice = _prompt("device (key or index)", session.device)
    if choice.isdigit() and int(choice) < len(keys):
        choice = keys[int(choice)]
    if choice not in keys:
        print("  ! unknown device")
        return
    session.device = choice
    session.device_options = _device_defaults(choice)
    _edit_device_options(session)


def _edit_device_options(session: Session) -> None:
    key = session.device
    opts = session.device_options
    if key == "tinysa":
        mode = _prompt("  tinySA burst mode (sweep/cw)", opts.get("mode", "sweep"))
        opts["mode"] = "cw" if mode.lower().startswith("c") else "sweep"
        opts["port"] = _prompt("  serial port (blank for later)", opts.get("port", ""))
    elif key == "hackrf":
        opts["txvga_gain"] = int(_prompt_float("  TX VGA gain (0-47)", opts.get("txvga_gain", 30)))
        opts["amp"] = _confirm("  enable TX amplifier?")
    if key == "rtlsdr":
        print("  note: RTL-SDR is receive-only and cannot run a broadcast session.")


def _edit_pause(session: Session) -> None:
    print("  periodic pause -- 0 hops disables it")
    every = _prompt_float("  pause every N hops (0 = never)",
                          float(session.pause_every_hops))
    session.pause_every_hops = max(0, int(every))
    if session.pause_every_hops > 0:
        session.pause_seconds = max(0.0, _prompt_float("  pause length (seconds)",
                                                       session.pause_seconds))
    else:
        session.pause_seconds = 0.0


def _edit_power(session: Session) -> None:
    print("  random broadcast strength (dBm) -- blank both to disable")
    lo = _prompt("  min dBm",
                 "" if session.power_min_dbm is None else str(session.power_min_dbm))
    hi = _prompt("  max dBm",
                 "" if session.power_max_dbm is None else str(session.power_max_dbm))
    if not lo.strip() and not hi.strip():
        session.power_min_dbm = session.power_max_dbm = None
        return
    try:
        pmin, pmax = float(lo), float(hi)
        if pmax < pmin:
            pmin, pmax = pmax, pmin
        session.power_min_dbm, session.power_max_dbm = pmin, pmax
    except ValueError:
        print("  ! enter numbers for both bounds (leaving strength unchanged)")


def _run_session(session: Session) -> None:
    opts = dict(session.device_options)
    if session.device == "mock":
        opts.setdefault("sleep", True)
    try:
        device = create_device(session.device, **opts)
    except Exception as exc:
        print(f"  ! could not create device: {exc}")
        return
    try:
        gen = NoiseGenerator(device, session)
    except ConfigurationError as exc:
        print(f"  ! {exc}")
        return
    print(f"\nBuilt {len(gen.bands)} bands. Press Ctrl-C to stop.")
    dur = _prompt("run for how many seconds? (blank = until Ctrl-C)")
    duration = float(dur) if dur else None
    reporter = make_reporter("auto")
    reporter.start()
    import time as _time
    t0 = _time.monotonic()
    try:
        hops = gen.run(duration=duration, on_hop=reporter.update)
    except Exception as exc:
        print(f"  ! run failed: {exc}")
        return
    reporter.finish(hops, _time.monotonic() - t0)


def _summary(session: Session) -> str:
    power = ""
    if session.has_power_range:
        power = f"  power={session.power_min_dbm:g}..{session.power_max_dbm:g}dBm"
    pause = ""
    if session.has_pause:
        pause = f"  pause={session.pause_seconds:g}s/{session.pause_every_hops}hops"
    return (
        f"name={session.name}  device={session.device}  "
        f"ranges={len(session.ranges)}  dwell={session.dwell_seconds}s{pause}{power}"
    )


def run_interactive(session: Optional[Session] = None) -> None:
    """Entry point for the interactive editor."""
    session = session or Session()
    print("RF Noise Generator - interactive session editor")
    while True:
        print(f"\n=== {_summary(session)} ===")
        print("  [1] name   [2] ranges   [3] device   [4] dwell/seed")
        print("  [5] save   [6] open     [7] run      [8] show")
        print("  [q] quit")
        choice = _prompt("menu>").lower()
        if choice == "1":
            session.name = _prompt("session name", session.name)
        elif choice == "2":
            _edit_ranges(session)
        elif choice == "3":
            _choose_device(session)
        elif choice == "4":
            session.dwell_seconds = _prompt_float("dwell seconds", session.dwell_seconds)
            seed = _prompt("random seed (blank = none)",
                           "" if session.seed is None else str(session.seed))
            session.seed = int(seed) if seed.strip() else None
            _edit_pause(session)
            _edit_power(session)
        elif choice == "5":
            path = session_store.default_path_for(session.name)
            path = _prompt("save to", path)
            written = session_store.save(session, path)
            print(f"  saved -> {written}")
        elif choice == "6":
            existing = session_store.list_sessions()
            if existing:
                print("  existing sessions:")
                for i, p in enumerate(existing):
                    print(f"    {i}: {os.path.basename(p)}")
            path = _prompt("open which (index or path)")
            if path.isdigit() and int(path) < len(existing):
                path = existing[int(path)]
            if path and os.path.exists(path):
                session = session_store.load(path)
                print(f"  loaded {session.name}")
            elif path:
                print("  ! not found")
        elif choice == "7":
            _run_session(session)
        elif choice == "8":
            print(f"  {_summary(session)}")
            for i, rng in enumerate(session.ranges):
                print(f"    range {i}: {rng}")
        elif choice in ("q", "quit", "exit"):
            if session.ranges and _confirm("save before quitting?"):
                path = session_store.default_path_for(session.name)
                session_store.save(session, _prompt("save to", path))
            return
