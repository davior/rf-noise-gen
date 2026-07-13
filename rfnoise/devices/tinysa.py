"""tinySA Ultra driver (serial / USB CDC).

The tinySA Ultra's built-in signal generator emits a single CW carrier -- it
does not have a wide instantaneous bandwidth like an SDR. Two burst modes are
supported:

* ``"sweep"`` (default): sweep the carrier across the selected band for the
  dwell time, emulating a wideband splatter across the slice.
* ``"cw"``: park a single tone at the band centre for the dwell time.

**The device must be put into generator (output) mode with ``mode output``**
before it will transmit; otherwise ``sweep`` just runs a spectrum *scan* (which
also floods the serial port). The output path is selected per hop:

  * ``output normal`` -- fundamental output, up to ~800 MHz
  * ``output mixer``  -- mixer/harmonic output, above ~800 MHz

.. note::
   Command strings here were checked against firmware ``tinySA4_v1.4`` (the
   ``help``/usage output: ``mode [low] input|output``,
   ``modulation off|am|fm|freq|depth|deviation 100..6000``, ``level -76..-6``,
   ``sweeptime 0.003..60``, ``output on|off|normal|mixer``). Other revisions may
   differ; they are grouped in :data:`_COMMANDS`. Requires ``pyserial``.
"""

from __future__ import annotations

import errno
import time
from typing import Optional

from .base import (
    DeviceCapabilities,
    DeviceError,
    Modulation,
    RFDevice,
    TxBand,
)

# Serial commands, centralised so they are easy to correct per firmware.
# ``{freq}``/``{start}``/``{stop}`` are integer Hz; ``{dbm}`` an integer level.
_COMMANDS = {
    "mode_output": "mode output\r",              # enter signal-generator mode
    "output_on": "output on\r",
    "output_off": "output off\r",
    "output_normal": "output normal\r",          # fundamental output path
    "output_mixer": "output mixer\r",            # mixer/harmonic output path
    "level": "level {dbm}\r",                    # output level, -76..-6 dBm
    "sweep": "sweep {start} {stop}\r",           # sweep the output across a band
    "cw": "sweep {freq} {freq}\r",               # zero-span == a fixed carrier
    "sweeptime": "sweeptime {seconds:.3f}\r",    # sweep duration, 0.003..60 s
    # -- fixed-tone modulation (firmware ``modulation`` command; tinySA has no
    # arbitrary-IQ path, so only a tone source is possible). ``{hz}`` is the
    # internal tone / FM deviation in Hz (100..6000); ``{value}`` the AM depth. --
    "mod_off": "modulation off\r",
    "mod_am": "modulation am\r",
    "mod_fm": "modulation fm\r",
    "mod_freq": "modulation freq {hz}\r",
    "mod_depth": "modulation depth {value}\r",
    "mod_deviation": "modulation deviation {hz}\r",
}

# Internal modulating-tone defaults when a session leaves them unset. Kept local
# so the pure-serial tinySA path never imports the numpy DSP core.
_DEFAULT_TONE_HZ = 1_000
_DEFAULT_AM_DEPTH_PCT = 50
_DEFAULT_FM_DEVIATION_HZ = 5_000
# Firmware limits for ``modulation freq``/``deviation`` (Hz).
_MOD_HZ_MIN = 100
_MOD_HZ_MAX = 6_000
# Frequency at/below which the fundamental ("normal") output path is used;
# above it the mixer path is needed.
_NORMAL_OUTPUT_MAX_HZ = 800_000_000

# Output level range of the tinySA Ultra signal generator (``level -76..-6``).
POWER_MIN_DBM = -76.0
POWER_MAX_DBM = -6.0

# The tinySA shell echoes each command and prints this prompt when it is ready
# for the next one. We read up to it after every command so the OS input buffer
# never fills (see ``_drain``). Kept as bytes since we compare against raw reads.
_PROMPT = b"ch> "

# Seconds a single serial write may block before we treat the port as stalled.
# Without this the pyserial default (``None``) lets a full-buffer write hang
# forever instead of raising.
_WRITE_TIMEOUT_S = 2.0

# OS errno values that mean "the device fell off the USB bus" (re-enumeration or
# a brown-out, e.g. RF from our own transmit coupling into the USB). The port
# node often reappears with the same name, so reopening it recovers the run.
_DISCONNECT_ERRNOS = frozenset({errno.EIO, errno.ENODEV, errno.ENXIO, errno.EBADF})

# Reconnect policy when the device drops mid-run.
_RECONNECT_ATTEMPTS = 6
_RECONNECT_DELAY_S = 0.5


def _import_serial(required: bool = False):
    """Return the pyserial module, ``None`` if absent (unless ``required``)."""
    try:
        import serial  # type: ignore
        return serial
    except ImportError as exc:  # pragma: no cover - depends on env
        if required:
            raise DeviceError(
                "tinySA requires pyserial (pip install rfnoise[hardware])"
            ) from exc
        return None


def _mode_for(hz: int) -> str:
    if hz <= 800_000_000:
        return "sine"
    if hz <= 4_400_000_000:
        return "square"
    return "mixing"


class TinySAUltra(RFDevice):
    """Serial driver for the tinySA Ultra signal generator.

    Options:
      * ``port`` -- serial device path (e.g. ``/dev/ttyACM0``). Required to open.
      * ``mode`` -- ``"sweep"`` (default) or ``"cw"``.
      * ``level`` -- output level index passed to the device (default 0).
      * ``default_band_width`` -- band width used when no cap/override applies;
        defaults to 1 MHz in sweep mode, 100 kHz in cw mode.
      * ``baudrate`` -- serial baud (default 115200).
    """

    def __init__(self, port: Optional[str] = None, mode: str = "sweep",
                 level: float = -30.0, default_band_width: Optional[int] = None,
                 baudrate: int = 115200, output_stage: str = "auto",
                 debug: bool = False,
                 reconnect_attempts: int = _RECONNECT_ATTEMPTS,
                 reconnect_delay: float = _RECONNECT_DELAY_S, **options):
        super().__init__(**options)
        if mode not in ("sweep", "cw"):
            raise ValueError("tinySA mode must be 'sweep' or 'cw'")
        if output_stage not in ("auto", "normal", "mixer"):
            raise ValueError("output_stage must be 'auto', 'normal' or 'mixer'")
        self.port = port
        self.mode = mode
        self.level = float(level)  # default output level in dBm
        self.baudrate = baudrate
        # Which RF output path to use. "auto" picks normal/mixer by frequency;
        # force "normal" or "mixer" if the auto boundary is wrong for your unit.
        self.output_stage = output_stage
        # When True, log every command sent + the device's response + timing to
        # stderr (for diagnosing which command the firmware rejects or stalls on).
        self.debug = bool(debug)
        # How hard to try to recover when the device drops off USB mid-run.
        # ``reconnect_attempts <= 0`` disables recovery (fail fast instead).
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_delay = float(reconnect_delay)
        self._reconnecting = False
        # State so we only re-send a command when it actually changes hop to hop
        # (fewer serial round-trips -> faster, less chance of overflow).
        self._stage: Optional[str] = None
        self._mod_on = False
        self._last_level: Optional[int] = None
        self._last_sweeptime: Optional[float] = None
        # Whether RF output is currently enabled. We keep TX OFF while sending
        # config commands, because the device's own radiated RF can couple into
        # its USB and wedge the serial link (write timeout) -- so all commands go
        # out with output off, then output is enabled just for the dwell.
        self._tx_on = False
        # A stray write timeout (transient EMI wedge) is retried this many times
        # after a short pause before it is treated as fatal.
        self._write_retries = 2
        self._write_retry_delay = 0.25
        if default_band_width is None:
            default_band_width = 1_000_000 if mode == "sweep" else 100_000
        self.capabilities = DeviceCapabilities(
            name="tinySA Ultra",
            can_transmit=True,
            tx_bands=(
                TxBand(100_000, 800_000_000, "sine"),
                TxBand(800_000_000, 4_400_000_000, "square"),
                TxBand(4_400_000_000, 5_400_000_000, "mixing"),
            ),
            # CW generator: no fixed instantaneous bandwidth -> None.
            max_bandwidth_hz=None,
            default_band_width=int(default_band_width),
            description=f"CW signal generator, {mode} burst mode.",
            power_min_dbm=POWER_MIN_DBM,
            power_max_dbm=POWER_MAX_DBM,
            # Crude built-in AM/FM from a fixed internal tone (fidelity
            # "fixed_tone"): no arbitrary IQ, so a noise source can't be honoured
            # -- the engine falls back to a tone. No instantaneous IQ bandwidth.
            supported_modulations=frozenset(
                {Modulation.NONE, Modulation.AM, Modulation.FM}
            ),
            modulation_fidelity="fixed_tone",
        )
        self._serial = None

    def _set_level(self, dbm: float) -> None:
        lvl = int(round(self.capabilities.clamp_power(dbm)))
        if lvl != self._last_level:
            self._send(_COMMANDS["level"].format(dbm=lvl))
            self._last_level = lvl

    def _open_serial(self, port: str):
        """Open the serial port (split out so reconnect/tests can reuse it)."""
        serial = _import_serial(required=True)
        return serial.Serial(
            port, self.baudrate, timeout=1, write_timeout=_WRITE_TIMEOUT_S
        )

    def _arm_output(self) -> None:
        """Enter generator mode and set a clean baseline (open and reconnect).

        ``mode output`` is essential: without it the device stays in analyzer
        mode and ``sweep`` runs a spectrum scan instead of transmitting. We also
        clear any leftover modulation and set the default level. The per-hop
        output path and ``output on`` are (re)issued by :meth:`broadcast` /
        :meth:`_modulate`, so the stage state is reset here to force a resend.
        """
        self._send(_COMMANDS["output_off"])
        self._tx_on = False
        self._send(_COMMANDS["mode_output"])
        self._send(_COMMANDS["mod_off"])
        self._mod_on = False
        self._stage = None
        self._last_level = None
        self._last_sweeptime = None
        self._set_level(self.level)

    def _stage_for(self, hz: int) -> str:
        """Return the output path ("normal"/"mixer") for a carrier frequency."""
        if self.output_stage != "auto":
            return self.output_stage
        return "normal" if hz <= _NORMAL_OUTPUT_MAX_HZ else "mixer"

    def _ensure_stage(self, hz: int) -> None:
        """Select the output path for ``hz`` if it differs from the last hop."""
        stage = self._stage_for(hz)
        if stage != self._stage:
            key = "output_normal" if stage == "normal" else "output_mixer"
            self._send(_COMMANDS[key])
            self._stage = stage

    def _set_sweeptime(self, dwell_s: float) -> None:
        """Set the firmware sweep duration to the dwell (clamped to 3ms..60s).

        Only re-sent when it changes -- the dwell is constant within a session,
        so this is effectively a one-time command.
        """
        if dwell_s <= 0:
            return
        seconds = round(max(0.003, min(60.0, float(dwell_s))), 3)
        if seconds != self._last_sweeptime:
            self._send(_COMMANDS["sweeptime"].format(seconds=seconds))
            self._last_sweeptime = seconds

    def _enable_output(self) -> None:
        """Turn RF output on (for the dwell) if not already transmitting."""
        if not self._tx_on:
            self._send(_COMMANDS["output_on"])
            self._tx_on = True

    def _disable_output(self) -> None:
        """Turn RF output off so config commands go out with the radio quiet.

        Transmitting while writing to the serial port risks an EMI-induced write
        stall (the radiated carrier couples into the USB). Keeping TX off during
        the command exchange avoids that; it also gives a genuinely quiet window
        during a periodic pause.
        """
        if self._tx_on:
            self._send(_COMMANDS["output_off"])
            self._tx_on = False

    def _clear_modulation(self) -> None:
        if self._mod_on:
            self._send(_COMMANDS["mod_off"])
            self._mod_on = False

    def _on_open(self) -> None:
        if not self.port:
            raise DeviceError("tinySA: no serial 'port' configured")
        self._serial = self._open_serial(self.port)
        self._arm_output()

    def _on_close(self) -> None:
        if self._serial is not None:
            try:
                # Best-effort: if the port has already stalled, don't let the
                # shutdown write hang -- surface nothing and still close the fd.
                self._send(_COMMANDS["output_off"])
            except DeviceError:
                pass
            finally:
                try:
                    self._serial.close()
                except Exception:  # pragma: no cover - already-dead port
                    pass
                self._serial = None

    def _classify_error(self, exc: Exception) -> str:
        """Bucket a serial error: ``"stall"``, ``"disconnect"`` or ``"other"``.

        A *stall* is a write timeout (buffer full, device not draining); a
        *disconnect* is the device dropping off USB (EIO/ENODEV, or a generic
        ``SerialException``) -- recoverable by reopening the port.
        """
        serial = _import_serial()
        if serial is not None and isinstance(exc, serial.SerialTimeoutException):
            return "stall"
        if isinstance(exc, OSError) and exc.errno in _DISCONNECT_ERRNOS:
            return "disconnect"
        if serial is not None and isinstance(exc, serial.SerialException):
            return "disconnect"
        return "other"

    def _find_port(self) -> Optional[str]:
        """Locate the device's port, preferring the configured path.

        After a re-enumeration the node usually reappears under the same name,
        but it can move (e.g. ``ttyACM0`` -> ``ttyACM1``); fall back to scanning
        for a tinySA / CDC-ACM port so recovery still works if it does.
        """
        import os

        if self.port and os.path.exists(self.port):
            return self.port
        try:
            from serial.tools import list_ports  # type: ignore
        except Exception:  # pragma: no cover - pyserial always ships this
            return None
        ports = list(list_ports.comports())
        for p in ports:
            text = f"{p.description or ''} {p.manufacturer or ''}".lower()
            if "tinysa" in text:
                return p.device
        for p in ports:
            dev = p.device or ""
            if "ACM" in dev or "usbmodem" in dev:
                return dev
        return None

    def _reconnect(self) -> None:
        """Recover from a mid-run USB drop by reopening and re-arming the port.

        Closes the stale handle, then retries find-port -> open -> re-arm output
        up to ``reconnect_attempts`` times. Raises :class:`DeviceError` if the
        device never comes back (truly unplugged / powered off).
        """
        if self.reconnect_attempts <= 0:
            raise DeviceError(
                "tinySA: device disconnected (I/O error); reconnect disabled. "
                "Check USB power/cable and RF shielding."
            )
        old, self._serial = self._serial, None
        if old is not None:
            try:
                old.close()
            except Exception:  # pragma: no cover - already-dead port
                pass
        self._reconnecting = True
        last: Optional[Exception] = None
        try:
            for _ in range(self.reconnect_attempts):
                time.sleep(self.reconnect_delay)
                port = self._find_port()
                if not port:
                    continue
                try:
                    self._serial = self._open_serial(port)
                    self.port = port
                    self._arm_output()
                except Exception as exc:  # not back yet / still noisy
                    last = exc
                    if self._serial is not None:
                        try:
                            self._serial.close()
                        except Exception:  # pragma: no cover
                            pass
                        self._serial = None
                    continue
                print(f"tinySA: reconnected on {port} after I/O error")
                return
        finally:
            self._reconnecting = False
        raise DeviceError(
            f"tinySA: device did not come back after {self.reconnect_attempts} "
            "reconnect attempts; check USB power/cable and RF shielding"
            + (f" (last error: {last})" if last else "")
        )

    def _write_flush(self, command: str) -> None:
        self._serial.write(command.encode("ascii"))
        self._serial.flush()

    def _safe_reset_input(self) -> None:
        ser = self._serial
        reset = getattr(ser, "reset_input_buffer", None) if ser is not None else None
        if reset is not None:
            try:
                reset()
            except Exception:  # pragma: no cover - best effort
                pass

    def _send(self, command: str) -> None:
        if self._serial is None:  # pragma: no cover - guarded by open()
            raise DeviceError("tinySA: serial port not open")
        t0 = time.monotonic()
        for attempt in range(self._write_retries + 1):
            try:
                self._write_flush(command)
                break
            except Exception as exc:
                kind = self._classify_error(exc)
                if kind == "disconnect" and not self._reconnecting:
                    # Device fell off USB (often our own RF): reopen and retry.
                    self._reconnect()
                    continue
                if kind == "stall" and attempt < self._write_retries \
                        and not self._reconnecting:
                    # Transient EMI wedge: pause, clear the port, and retry.
                    time.sleep(self._write_retry_delay)
                    self._safe_reset_input()
                    continue
                if kind == "stall":
                    raise DeviceError(
                        f"tinySA: serial write stalled/failed ({exc}); "
                        "the device stopped accepting data. If it recurs while "
                        "transmitting, it is RF coupling into the USB -- improve "
                        "shielding / use a powered hub."
                    ) from exc
                raise
        response = self._drain()
        if self.debug:
            import sys
            dt_ms = (time.monotonic() - t0) * 1000.0
            sys.stderr.write(
                f"[tinysa] >> {command.strip()!r}  ({dt_ms:.0f} ms)  "
                f"<< {response[:120]!r}\n"
            )
            sys.stderr.flush()

    def _drain(self) -> bytes:
        """Consume and return the shell's echo/response after a command.

        The tinySA echoes every command and follows it with a ``ch>`` prompt.
        If it is never read the OS input buffer (~4 KB) fills, back-pressures
        the port, and the next write blocks. Reading up to the prompt after each
        command keeps the buffer empty; the serial read timeout bounds the wait
        if a firmware revision omits the prompt. Returned bytes feed debug logs.
        """
        if self._serial is None:  # pragma: no cover - guarded by callers
            return b""
        data = self._serial.read_until(_PROMPT) or b""
        # Backstop: if a firmware revision's prompt/output doesn't match
        # ``ch> `` exactly, ``read_until`` stops at the read timeout with bytes
        # still queued. Clear them so nothing carries into the next write.
        reset = getattr(self._serial, "reset_input_buffer", None)
        if reset is not None:
            reset()
        return data

    def _dwell(self, seconds: float) -> None:
        """Hold for the dwell while transmitting -- with **no** serial I/O.

        In generator (``mode output``) mode the device does not stream, so there
        is nothing to drain; and because the carrier is radiating during the
        dwell, touching the serial port here is exactly when an EMI-induced
        write/read stall is most likely. So we simply sleep (the port stays
        idle), then the next hop turns output off before it writes anything.
        """
        if seconds > 0:
            time.sleep(seconds)

    def keep_alive(self) -> None:
        """Go quiet during a periodic pause: stop transmitting.

        The engine calls this while paused. Turning output off gives a genuinely
        idle window (the point of a duty-cycle pause) and keeps the carrier from
        coupling into the USB while we're not actively hopping. The next hop
        re-enables output.
        """
        if self._serial is None:
            return
        try:
            self._disable_output()
        except Exception as exc:
            self._recover_or_raise(exc)

    def _recover_or_raise(self, exc: Exception) -> None:
        """Reconnect on a disconnect error; re-raise anything else."""
        if self._classify_error(exc) == "disconnect" and not self._reconnecting:
            self._reconnect()
        else:
            raise exc

    def emit(self, emission) -> None:
        """Realise a native sweep or fixed-tone modulation; else plain broadcast.

        Three cases, in order:

        * **AM/FM** -- park a CW carrier at the band centre and enable the
          device's built-in fixed-tone modulator (:meth:`_modulate`). The tinySA
          has no arbitrary IQ path; the engine has already forced a noise source
          to a tone before we get here.
        * **stepped sweep** -- one native ``sweep {start} {stop}`` command sweeps
          the whole coverage span in firmware for the dwell.
        * **plain** -- fall back to :meth:`broadcast` via the base implementation.
        """
        if emission.modulation != Modulation.NONE:
            self._modulate(emission)
            return
        sweep = emission.sweep
        if sweep is not None and sweep.steps > 1:
            # One native firmware sweep covers the whole span for the dwell --
            # no Python step retunes.
            center = (int(sweep.start_hz) + int(sweep.stop_hz)) // 2
            self._disable_output()          # configure with the radio quiet
            self._ensure_stage(center)
            self._clear_modulation()
            if emission.power_dbm is not None:
                self._set_level(emission.power_dbm)
            self._send(_COMMANDS["sweep"].format(start=int(sweep.start_hz),
                                                 stop=int(sweep.stop_hz)))
            self._set_sweeptime(emission.dwell_s)
            self._enable_output()           # transmit for the dwell
            self._dwell(emission.dwell_s)
            return
        super().emit(emission)

    def _modulate(self, emission) -> None:
        """Park a carrier at the band centre and enable fixed-tone AM/FM.

        Uses the device's internal modulating tone (``tone_hz`` or a default,
        clamped to the firmware's 100..6000 Hz range); ``depth`` (AM) and
        ``deviation_hz`` (FM) fall back to defaults when unset. Realised with the
        firmware ``modulation`` command (see :data:`_COMMANDS`).
        """
        center = (int(emission.start_hz) + int(emission.stop_hz)) // 2
        self._disable_output()              # configure with the radio quiet
        self._ensure_stage(center)
        if emission.power_dbm is not None:
            self._set_level(emission.power_dbm)
        self._send(_COMMANDS["cw"].format(freq=center))  # park the carrier
        tone = int(emission.tone_hz) if emission.tone_hz else _DEFAULT_TONE_HZ
        tone = max(_MOD_HZ_MIN, min(_MOD_HZ_MAX, tone))
        self._send(_COMMANDS["mod_freq"].format(hz=tone))
        if emission.modulation == Modulation.AM:
            depth_pct = (int(round(emission.depth * 100)) if emission.depth is not None
                         else _DEFAULT_AM_DEPTH_PCT)
            self._send(_COMMANDS["mod_depth"].format(value=depth_pct))
            self._send(_COMMANDS["mod_am"])
        else:  # FM
            deviation = (int(emission.deviation_hz) if emission.deviation_hz
                         else _DEFAULT_FM_DEVIATION_HZ)
            deviation = max(_MOD_HZ_MIN, min(_MOD_HZ_MAX, deviation))
            self._send(_COMMANDS["mod_deviation"].format(hz=deviation))
            self._send(_COMMANDS["mod_fm"])
        self._mod_on = True
        self._enable_output()               # transmit for the dwell
        self._dwell(emission.dwell_s)

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float,
                  power_dbm: Optional[float] = None) -> None:
        center = (int(start_hz) + int(stop_hz)) // 2
        self._disable_output()              # configure with the radio quiet
        self._ensure_stage(center)
        self._clear_modulation()
        if power_dbm is not None:
            self._set_level(power_dbm)
        if self.mode == "cw":
            self._send(_COMMANDS["cw"].format(freq=center))
        else:
            self._send(_COMMANDS["sweep"].format(start=int(start_hz), stop=int(stop_hz)))
            self._set_sweeptime(dwell_s)
        self._enable_output()               # transmit for the dwell
        self._dwell(dwell_s)
