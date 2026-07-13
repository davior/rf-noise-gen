"""Command-line entry point for rfnoise."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from . import session as session_store
from .devices import create_device, device_keys, get_device_class
from .engine import ConfigurationError, NoiseGenerator
from .freq import format_freq
from .interactive import run_interactive
from .model import Session
from .status import make_reporter


def _cmd_list_devices(args) -> int:
    for key in device_keys():
        try:
            dev = get_device_class(key)()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"{key}: <error: {exc}>")
            continue
        print(f"[{key}]")
        print("  " + dev.describe().replace("\n", "\n  "))
        print()
    return 0


def _cmd_ui(args) -> int:
    session = None
    if args.session:
        session = session_store.load(args.session)
    run_interactive(session)
    return 0


def _cmd_new(args) -> int:
    run_interactive(Session())
    return 0


def _cmd_gui(args) -> int:
    # Dear PyGui is an optional extra imported lazily inside rfnoise.gui, so
    # check it is installed before we get far enough to fail cryptically.
    import importlib.util

    if importlib.util.find_spec("dearpygui") is None:
        print(
            'the GUI needs the optional "gui" extra; install it with:\n'
            '  pip install -e ".[gui]"',
            file=sys.stderr,
        )
        return 2
    session = session_store.load(args.session) if args.session else None
    from .gui import DisplayUnavailableError, run_gui

    try:
        run_gui(session)
    except DisplayUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_run(args) -> int:
    session = session_store.load(args.session)
    if args.device:
        session.device = args.device
    if args.pause_seconds is not None:
        session.pause_seconds = args.pause_seconds
    if args.pause_every is not None:
        session.pause_every_hops = args.pause_every
    opts = dict(session.device_options)
    if session.device == "mock":
        opts.setdefault("sleep", not args.dry_run)
    try:
        device = create_device(session.device, **opts)
        gen = NoiseGenerator(device, session)
    except (ConfigurationError, Exception) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        n = args.iterations or 10
        print(f"dry run: next {n} hops ({len(gen.bands)} bands in pool)")
        for i, band in enumerate(gen.plan(n)):
            print(f"  {i:>3}: {format_freq(band.center_hz):>10} "
                  f"({format_freq(band.start_hz)}-{format_freq(band.stop_hz)}, "
                  f"width {format_freq(band.width_hz)})")
        return 0

    mode = "quiet" if args.quiet else ("log" if args.log else "auto")
    reporter = make_reporter(mode)
    power = ""
    if session.has_power_range:
        power = f", power {session.power_min_dbm:g}..{session.power_max_dbm:g} dBm"
    pause = ""
    if session.has_pause:
        pause = (f", pause {session.pause_seconds:g}s every "
                 f"{session.pause_every_hops} hops")
    print(f"running '{session.name}' on {device.name}: "
          f"{len(gen.bands)} bands, dwell {session.dwell_seconds}s{power}{pause}")
    reporter.start()
    import time as _time
    t0 = _time.monotonic()
    hops = gen.run(duration=args.duration, iterations=args.iterations,
                   on_hop=reporter.update)
    reporter.finish(hops, _time.monotonic() - t0)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rfnoise",
        description="Frequency-hopping RF noise generator.",
    )
    sub = parser.add_subparsers(dest="command")

    p_ui = sub.add_parser("ui", help="launch the interactive session editor")
    p_ui.add_argument("session", nargs="?", help="session file to open")
    p_ui.set_defaults(func=_cmd_ui)

    p_new = sub.add_parser("new", help="create a new session interactively")
    p_new.set_defaults(func=_cmd_new)

    p_gui = sub.add_parser("gui", help="launch the graphical session editor")
    p_gui.add_argument("session", nargs="?", help="session file to open")
    p_gui.set_defaults(func=_cmd_gui)

    p_list = sub.add_parser("list-devices", help="show devices and auto max bandwidth")
    p_list.set_defaults(func=_cmd_list_devices)

    p_run = sub.add_parser("run", help="run a saved session")
    p_run.add_argument("session", help="session JSON file")
    p_run.add_argument("--device", choices=device_keys(), help="override device")
    p_run.add_argument("--duration", type=float, help="seconds to run")
    p_run.add_argument("--iterations", type=int, help="number of hops")
    p_run.add_argument("--pause-seconds", type=float,
                       help="override session pause length (seconds)")
    p_run.add_argument("--pause-every", type=int,
                       help="override session pause interval (hops); 0 disables")
    p_run.add_argument("--dry-run", action="store_true",
                       help="print the hop schedule without transmitting")
    p_run.add_argument("--log", action="store_true",
                       help="print one status line per hop instead of a live line")
    p_run.add_argument("--quiet", action="store_true",
                       help="suppress the live run status")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        # Default to the interactive UI.
        run_interactive(None)
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
