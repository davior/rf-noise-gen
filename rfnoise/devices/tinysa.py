"""tinySA Ultra driver (serial / USB CDC).

The tinySA Ultra's built-in signal generator emits a single CW carrier -- it
does not have a wide instantaneous bandwidth like an SDR. Two burst modes are
therefore supported:

* ``"sweep"`` (default): sweep the carrier across the selected band for the
  dwell time, emulating a wideband splatter across the slice.
* ``"cw"``: park a single tone at the band centre for the dwell time.

Output stage is chosen automatically by frequency:
  * sine   100 kHz - 800 MHz
  * square 800 MHz - 4.4 GHz
  * mixing 4.4 GHz - 5.4 GHz

.. warning::
   The exact serial command strings vary between tinySA firmware revisions and
   could not be verified against hardware in this environment. They are grouped
   in :data:`_COMMANDS` below; verify/adjust them against your firmware's
   ``help`` output before trusting the RF output. Requires ``pyserial``.
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
# ``{freq}`` is an integer in Hz; ``{start}``/``{stop}`` likewise; ``{dbm}`` is
# an integer output level in dBm within the device's -110..-20 dBm range.
_COMMANDS = {
    "cw": "sweep {freq} {freq} 2\r",           # single point == fixed carrier
    "sweep": "sweep {start} {stop} 450\r",       # sweep across the band
    "level": "level {dbm}\r",                    # output level in dBm
    "output_on": "output on\r",
    "output_off": "output off\r",
    # -- fixed-tone modulation (FIRMWARE-SPECIFIC; verify before trusting RF) --
    # The tinySA Ultra can impose a crude AM/FM on the parked CW carrier using
    # its own internal tone -- it has no arbitrary IQ path, so only a tone
    # source is possible. ``{tone}`` is the modulating tone in Hz, ``{depth}``
    # the AM depth in percent (0-100), ``{deviation}`` the FM peak deviation in
    # Hz. Command spellings vary by firmware; adjust against its ``help``.
    "am": "am {tone} {depth}\r",
    "fm": "fm {tone} {deviation}\r",
    "mod_off": "modulation off\r",
}

# Internal modulating-tone defaults when a session leaves them unset. Kept local
# so the pure-serial tinySA path never imports the numpy DSP core.
_DEFAULT_TONE_HZ = 1_000
_DEFAULT_AM_DEPTH_PCT = 50
_DEFAULT_FM_DEVIATION_HZ = 5_000

# Output level range of the tinySA Ultra signal generator.
POWER_MIN_DBM = -110.0
POWER_MAX_DBM = -20.0

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
                 baudrate: int = 115200,
                 reconnect_attempts: int = _RECONNECT_ATTEMPTS,
                 reconnect_delay: float = _RECONNECT_DELAY_S, **options):
        super().__init__(**options)
        if mode not in ("sweep", "cw"):
            raise ValueError("tinySA mode must be 'sweep' or 'cw'")
        self.port = port
        self.mode = mode
        self.level = float(level)  # default output level in dBm
        self.baudrate = baudrate
        # How hard to try to recover when the device drops off USB mid-run.
        # ``reconnect_attempts <= 0`` disables recovery (fail fast instead).
        self.reconnect_attempts = int(reconnect_attempts)
        self.reconnect_delay = float(reconnect_delay)
        self._reconnecting = False
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
        self._send(_COMMANDS["level"].format(dbm=int(round(self.capabilities.clamp_power(dbm)))))

    def _open_serial(self, port: str):
        """Open the serial port (split out so reconnect/tests can reuse it)."""
        serial = _import_serial(required=True)
        return serial.Serial(
            port, self.baudrate, timeout=1, write_timeout=_WRITE_TIMEOUT_S
        )

    def _arm_output(self) -> None:
        """Set the output level and enable output (used on open and reconnect)."""
        self._set_level(self.level)
        self._send(_COMMANDS["output_on"])

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

    def _send(self, command: str) -> None:
        if self._serial is None:  # pragma: no cover - guarded by open()
            raise DeviceError("tinySA: serial port not open")
        try:
            self._write_flush(command)
        except Exception as exc:
            kind = self._classify_error(exc)
            if kind == "stall":
                raise DeviceError(
                    f"tinySA: serial write stalled/failed ({exc}); "
                    "the device stopped accepting data."
                ) from exc
            if kind == "disconnect" and not self._reconnecting:
                # Device fell off USB (often our own RF): reopen and retry once.
                self._reconnect()
                try:
                    self._write_flush(command)
                except Exception as exc2:
                    raise DeviceError(
                        f"tinySA: write failed after reconnect ({exc2})"
                    ) from exc2
            else:
                raise
        self._drain()

    def _drain(self) -> None:
        """Consume the shell's echo/response so the input buffer never fills.

        The tinySA echoes every command and follows it with a ``ch>`` prompt.
        Nothing here needs that text, but if it is never read the OS input
        buffer (~4 KB) fills after a hundred-odd hops, back-pressures the port,
        and the next write blocks forever. Reading up to the prompt after each
        command keeps the buffer empty; the serial read timeout bounds the wait
        if a firmware revision omits the prompt.
        """
        if self._serial is None:  # pragma: no cover - guarded by callers
            return
        self._serial.read_until(_PROMPT)
        # Backstop: if a firmware revision's prompt/output doesn't match
        # ``ch> `` exactly, ``read_until`` stops at the read timeout with bytes
        # still queued. Clear them so nothing carries into the next write.
        reset = getattr(self._serial, "reset_input_buffer", None)
        if reset is not None:
            reset()

    def _dwell(self, seconds: float) -> None:
        """Wait out the dwell while draining anything the device streams.

        A running ``sweep`` streams scan data for the whole dwell. The old code
        slept through it reading nothing, so the OS input buffer filled (fast
        with a wide sweep -- a couple of hops), back-pressured the port, and the
        next write stalled with a ``Write timeout``. Here we read and discard
        pending bytes throughout the dwell (napping briefly when the port is
        idle so we don't busy-spin), keeping the buffer empty regardless of how
        much the firmware emits. With ``seconds <= 0`` we still clear anything
        buffered so it can't carry into the next command.
        """
        ser = self._serial
        if ser is None:  # pragma: no cover - guarded by callers
            return
        reset = getattr(ser, "reset_input_buffer", None)
        try:
            if seconds <= 0:
                if reset is not None:
                    reset()
                return
            end = time.monotonic() + seconds
            while True:
                left = end - time.monotonic()
                if left <= 0:
                    break
                waiting = getattr(ser, "in_waiting", 0)
                if waiting:
                    ser.read(waiting)
                else:
                    time.sleep(min(left, 0.02))
            if reset is not None:
                reset()
        except Exception as exc:
            # A drop while draining is recoverable -- reopen and let the next
            # hop continue on the fresh port.
            self._recover_or_raise(exc)

    def keep_alive(self) -> None:
        """Drain any streamed bytes while the engine is paused.

        During a periodic pause the engine stops emitting but the device's last
        ``sweep`` may keep streaming; without reading, the buffer fills over the
        pause and the next write stalls. Discard whatever is waiting. A drop
        during the pause is recovered so the run resumes cleanly.
        """
        ser = self._serial
        if ser is None:
            return
        try:
            waiting = getattr(ser, "in_waiting", 0)
            if waiting:
                ser.read(waiting)
            else:
                reset = getattr(ser, "reset_input_buffer", None)
                if reset is not None:
                    reset()
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
            if emission.power_dbm is not None:
                self._set_level(emission.power_dbm)
            self._send(_COMMANDS["sweep"].format(start=int(sweep.start_hz),
                                                 stop=int(sweep.stop_hz)))
            self._dwell(emission.dwell_s)
            return
        super().emit(emission)

    def _modulate(self, emission) -> None:
        """Park a carrier at the band centre and enable fixed-tone AM/FM.

        Uses the device's internal modulating tone (``tone_hz`` or a default);
        ``depth`` (AM) and ``deviation_hz`` (FM) fall back to sensible defaults
        when the session leaves them unset. Serial strings are the (unverified)
        entries in :data:`_COMMANDS`.
        """
        center = (int(emission.start_hz) + int(emission.stop_hz)) // 2
        if emission.power_dbm is not None:
            self._set_level(emission.power_dbm)
        self._send(_COMMANDS["cw"].format(freq=center))
        tone = int(emission.tone_hz) if emission.tone_hz else _DEFAULT_TONE_HZ
        if emission.modulation == Modulation.AM:
            depth_pct = (int(round(emission.depth * 100)) if emission.depth is not None
                         else _DEFAULT_AM_DEPTH_PCT)
            self._send(_COMMANDS["am"].format(tone=tone, depth=depth_pct))
        else:  # FM
            deviation = (int(emission.deviation_hz) if emission.deviation_hz
                         else _DEFAULT_FM_DEVIATION_HZ)
            self._send(_COMMANDS["fm"].format(tone=tone, deviation=deviation))
        self._dwell(emission.dwell_s)

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float,
                  power_dbm: Optional[float] = None) -> None:
        if power_dbm is not None:
            self._set_level(power_dbm)
        if self.mode == "cw":
            center = (int(start_hz) + int(stop_hz)) // 2
            self._send(_COMMANDS["cw"].format(freq=center))
        else:
            self._send(_COMMANDS["sweep"].format(start=int(start_hz), stop=int(stop_hz)))
        self._dwell(dwell_s)
