# RF Noise Generator

A script-based RF noise generator that broadcasts random signals by hopping
between randomly-chosen slices of user-defined frequency ranges. It dwells on
each slice for a configurable time and switches with no pause, driving any of
several RF devices through a common abstraction layer.

> ⚠️ **Transmitting RF is regulated.** Only transmit on frequencies you are
> licensed/authorized to use, and prefer a dummy load or a shielded enclosure.
> You are responsible for complying with the rules in your jurisdiction. The
> `mock` device emits nothing and is safe to run anywhere.

## Key features

- **Multiple frequency ranges**, hopped between at random — each hop is a random
  slice pooled across *all* ranges.
- **Auto-derived maximum broadcast bandwidth.** You don't enter a max bandwidth
  per range; the selected device supplies it (see below). You *may* still set a
  per-range override to go narrower.
- **Configurable dwell time** with seamless switching by default, plus an
  optional periodic pause (hold transmission for X seconds every N hops).
- **Random broadcast strength.** Optionally give a dBm range and each hop
  transmits at a random level drawn from it (see below).
- **Random band drift.** Optionally shift every band by a random amount when it
  fires (a fraction of its bandwidth), making the emission schedule far harder to
  predict while never straying outside the ranges you defined (see below).
- **Live run status.** While running, a status line shows the current frequency,
  band, output level, hop count and rate.
- **Pluggable device abstraction**: tinySA Ultra, HackRF One, RTL-SDR, plus a
  software `mock` for testing.
- **Interactive session editor** that persists to JSON you can reopen later.

## Auto max bandwidth per device

The "maximum bandwidth per broadcast" is the widest continuous signal a device
can emit in one burst. These are built in, so you never have to look them up:

| Device        | Transmit | Frequency range         | Auto max broadcast bandwidth | Output level | Modulation |
|---------------|----------|-------------------------|------------------------------|--------------|------------|
| HackRF One    | yes      | 1 MHz – 6 GHz           | **20 MHz** (20 Msps limit)   | -50…5 dBm (mapped to 0–47 dB gain, approx.) | AM/FM via arbitrary IQ (`iq`, needs `[dsp]`) |
| tinySA Ultra  | yes      | 100 kHz – 5.4 GHz       | CW generator → no fixed cap; uses a default band width (1 MHz sweep / 100 kHz cw) | -110…-20 dBm (calibrated) | AM/FM from a fixed internal tone (`fixed_tone`) |
| RTL-SDR       | **no**   | ~500 kHz – 1.766 GHz    | receive-only (cannot broadcast) | n/a | none (receive-only) |
| mock          | yes      | 0 – 6 GHz               | 20 MHz (configurable)        | -120…10 dBm (recorded only) | AM/FM via arbitrary IQ (`iq`, needs `[dsp]`) |

The engine picks the effective band width as
`min(range override if set, device hardware cap if any, else device default)`,
never wider than the range itself. Run `rfnoise list-devices` to print these.

**Modulation (AM/FM).** IQ devices (HackRF, mock) synthesise the waveform
themselves from a tone or noise source, so they need the `[dsp]` extra (numpy).
The tinySA has no IQ path: it applies a crude AM/FM from its own fixed internal
tone, so a *noise* source request falls back to a tone (with a warning) and the
tinySA path needs no numpy. A device asked for a modulation it can't emit falls
back to plain CW/noise with a warning. HackRF retunes per hop by restarting
`hackrf_transfer` (a small gap); gapless streaming is future work.

Sources: [HackRF docs](https://hackrf.readthedocs.io/en/latest/hackrf_one.html),
[tinySA](https://www.cnx-software.com/2025/12/15/tinysa-is-a-low-cost-handheld-spectrum-analyzer-with-built-in-signal-generator/),
[RTL-SDR](https://www.rtl-sdr.com/about-rtl-sdr/).

## Random broadcast strength (dBm)

Set an optional `power_min_dbm`/`power_max_dbm` on a session and each hop
transmits at a level drawn uniformly from that range (from the same seeded RNG,
so seeded runs stay reproducible). The level is applied per device:

- **tinySA Ultra** — set directly, its calibrated -110…-20 dBm output level.
- **HackRF One** — the dBm range is mapped onto its 0–47 dB TX gain; absolute
  dBm is **approximate/uncalibrated**.
- **mock** — recorded and displayed only.

The drawn level is always clamped into the device's supported range. If a
session sets a strength range but the device can't control its level, the range
is ignored with a warning.

## Periodic pause

By default hops are seamless. Set `pause_every_hops` and `pause_seconds` on a
session and the generator holds transmission for `pause_seconds` after every
`pause_every_hops` hops — useful for duty-cycling the transmitter or leaving a
quiet window for other equipment. The pause is active only when both values are
> 0; `pause_every_hops = 0` disables it. The pause stays responsive to Ctrl-C
and to `--duration`, so a run still stops promptly even mid-pause.

Override a saved session at run time with the `run` flags:

```
rfnoise run session.json --pause-every 10 --pause-seconds 2
```

## Random band drift

Normally every hop broadcasts on the exact boundaries of one of the fixed bands,
so an observer who learns your range/bandwidth layout knows precisely which
slices can ever go out. Set a `drift_fraction` on a session to make that
unpredictable: when a band fires it is shifted by a random offset up to
`± drift_fraction × bandwidth`, drawn from the same seeded RNG as everything else
(so seeded runs stay reproducible). `0.5` gives the classic **± bandwidth/2**
spread — a nominal 110–120 MHz band then goes out as anything from 105–115 up to
115–125 MHz.

The drift is **clamped to the band's parent range**: an offset can never push a
band past the `lower`/`upper` bounds it was cut from, so interior bands drift the
full `± reach` while bands touching a range edge drift only inward. A band that
already fills its whole range cannot move. Because drift only shifts the band
(preserving its width), it works for every emit mode automatically — the tinySA
sweep span or CW tone slides, and the HackRF centre retunes while its
instantaneous bandwidth is unchanged.

Leave `drift_fraction` blank / `None` / `0` to disable it (the default). The
`run` command also takes a `--drift FRACTION` override for quick experiments:

```bash
rfnoise run session.json --drift 0.5    # ± bandwidth/2 drift
rfnoise run session.json --drift 0      # force drift off
```

## Live run status

While a session runs you get a status line showing the current frequency being
broadcast, the band, the output level, hop count, elapsed time and hop rate:

```
⟳    145 MHz  band 140 MHz-150 MHz   -42.3 dBm  hop 128  02:11  4.1 hop/s
```

In a terminal it updates a single line in place; piped/non-TTY output (or
`--log`) prints one line per hop; `--quiet` suppresses it.

## Install

```bash
pip install -e .            # core (pure stdlib, mock + engine + UI)
pip install -e .[hardware]  # + pyserial for the tinySA driver
pip install -e .[gui]       # + dearpygui for the graphical editor
pip install -e .[dsp]       # + numpy for AM/FM/chirp modulation
pip install -e .[dev]       # + pytest
```

The HackRF driver shells out to `hackrf_transfer` (install the `hackrf`
system package). RTL-SDR support is receive-only and included for completeness.

## Finding the serial port (tinySA)

The **tinySA Ultra** connects as a USB CDC serial device, so you must tell
rfnoise which port it is (the interactive editor's *serial port* prompt, or the
`port` device option). The name differs per OS. The HackRF One does **not** need
this — `hackrf_transfer` auto-detects the device over USB.

**Cross-platform (recommended).** Once you've installed the `[hardware]` extra
you get pyserial, which ships a port lister that works everywhere:

```bash
python -m serial.tools.list_ports -v
```

Run it with the tinySA unplugged, then again plugged in, and the new entry is
your device. Look for a USB CDC / "tinySA" description; the first column is the
port name to use.

**Linux** — the port is usually `/dev/ttyACM0`:

```bash
ls /dev/ttyACM*          # list candidate ports
dmesg | tail             # right after plugging in, shows e.g. "ttyACM0"
```

If opening the port fails with a permission error, add yourself to the serial
group (`dialout` on Debian/Ubuntu, `uucp` on Arch), then log out and back in:

```bash
sudo usermod -aG dialout $USER
```

**macOS** — the port appears as `/dev/cu.usbmodem*` (use the `cu.` name, not
`tty.`). No driver is needed on modern macOS:

```bash
ls /dev/cu.usbmodem*
```

**Windows** — the device shows up as `COMx` (e.g. `COM3`). Find it in
**Device Manager → Ports (COM & LPT)**, or from PowerShell:

```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
# or, with descriptions:
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Description
```

Enter that value (e.g. `COM3`) as the port. Windows 10/11 supply the USB CDC
driver automatically; if the device shows as unknown, install the tinySA driver
per its documentation.

## Usage

Interactive editor (default when run with no arguments):

```bash
rfnoise            # or: rfnoise ui  /  python -m rfnoise
```

Menu: set a name, add ranges (enter bounds as `100k`, `2.4M`, `433.9MHz` …),
choose a device and its options, set dwell/seed and an optional strength range,
then **save** to a session file and **run**. Saved sessions live under
`sessions/` and can be reopened.

Graphical editor (optional, needs the `[gui]` extra):

```bash
pip install -e .[gui]
rfnoise gui                              # empty session
rfnoise gui examples/sample_session.json # open a saved session
```

The GUI (built on [Dear PyGui](https://github.com/hoffstadt/DearPyGui)) is a
third front-end on the same engine as the text `ui` — edit the session on the
left, hit **Run** to start hopping on a background thread, and watch a live
status line plus a spectrum-style **bar graph**: **frequency on X** (fixed to
the configured ranges), **strength (dBm) on Y**. Each burst raises a vertical
bar to its level, which then **sinks toward the floor and fades out** over the
*plot decay* window (adjustable above the plot, default 10 s) before vanishing.
(With no power range set, bars share one level and just show active
frequencies.)
**Save** / **Load** use the same JSON session files as everywhere else. The text
`ui` remains available and unchanged.

The GUI needs a graphical display. Over SSH, connect with X forwarding
(`ssh -X`); on a headless box use `rfnoise ui` or `rfnoise run` instead. Without
a display, `rfnoise gui` prints this guidance rather than a raw GLFW error.

#### Running the GUI remotely or under code-server

`rfnoise gui` opens a native OpenGL window, so it renders on a real display —
it **cannot** appear inside a browser-based IDE (VS Code `code-server`, etc.).
A few setups and their fixes:

- **Different user than the desktop session** (e.g. `code-server` runs as
  `devuser` but the desktop is user `gecko`). The GUI needs access to the
  desktop user's X cookie. Share it once:

  ```bash
  # as the desktop user (gecko), in a desktop terminal:
  xauth extract /tmp/xauth-share :1     # :1 = that session's DISPLAY
  chmod a+r /tmp/xauth-share

  # as the other user (devuser):
  export DISPLAY=:1
  export XAUTHORITY=/tmp/xauth-share
  rfnoise gui                           # window opens on the physical screen
  ```

  The window still appears on the desktop user's monitor — this only grants
  access, it does not move pixels into a browser. Delete `/tmp/xauth-share`
  when done.

- **Remote, and you want the GUI in a browser tab.** Use
  [Xpra](https://github.com/Xpra-org/xpra)'s HTML5 client, which runs the app in
  its own headless X server and serves just that window over HTTP:

  ```bash
  xpra start --start="rfnoise gui" \
       --bind-tcp=0.0.0.0:14500 --html=on --exit-with-children=yes
  # then open http://<host>:14500/ in your browser
  ```

- **No display at all.** Use the text UI — same engine, no window:
  `rfnoise ui` (interactive editor) or `rfnoise run <session>`.

Run a saved session headless:

```bash
rfnoise run examples/sample_session.json --duration 5
rfnoise run examples/sample_session.json --dry-run --iterations 10
rfnoise run examples/sample_session.json --log      # one status line per hop
rfnoise run examples/sample_session.json --quiet    # no status output
rfnoise list-devices
```

Or drive it from Python:

```python
from rfnoise import Session, FrequencyRange, NoiseGenerator
from rfnoise.devices import create_device

session = Session(
    name="fm-band",
    device="mock",
    ranges=[FrequencyRange(88_000_000, 108_000_000)],  # no bandwidth = device auto
    dwell_seconds=0.25,
    power_min_dbm=-60, power_max_dbm=-30,              # random strength per hop
    pause_seconds=2, pause_every_hops=10,             # pause 2s every 10 hops
    drift_fraction=0.5,                                # random +/- bw/2 band drift
    seed=42,
)
gen = NoiseGenerator(create_device("mock"), session)
gen.run(iterations=10, on_hop=lambda s: print(s.line()))
```

## Architecture

```
rfnoise/
  devices/       device abstraction + drivers (base, mock, tinysa, hackrf, rtlsdr)
  freq.py        human-friendly frequency parse/format
  model.py       FrequencyRange, Session
  bands.py       band splitting + coverage bands + drift offset
  tuning.py      tuning strategies (random-hop, sequential, sweep-in-band)
  modulation.py  AM/FM/chirp DSP core (numpy, optional [dsp] extra)
  sources.py     modulating sources (tone/noise) for AM/FM
  engine.py      NoiseGenerator: validation + hop/dwell loop + power/drift draws
  status.py      live/log run-status reporters (HopStatus)
  session.py     versioned JSON load/save
  interactive.py menu-driven session editor
  gui.py         Dear PyGui graphical editor (optional [gui] extra)
  cli.py         command-line entry point
```

To add a device, subclass `rfnoise.devices.base.RFDevice`, declare its
`DeviceCapabilities` (including `max_bandwidth_hz`), implement `broadcast()`,
and register it in `rfnoise/devices/__init__.py`.

## Hardware notes / limitations

- Real tinySA/HackRF/RTL-SDR I/O has **not** been verified on hardware here.
  - **tinySA**: serial command strings vary by firmware; they're centralized in
    `rfnoise/devices/tinysa.py` (`_COMMANDS`) — verify against your firmware's
    `help` output.
  - **HackRF**: retuning restarts `hackrf_transfer`, adding a small gap per hop.
    The noise-sample generator is unit-tested; the streaming path needs the
    device. A continuous-retune SoapySDR/pyhackrf path is a natural next step.
- The `mock` device and the whole engine/UI are fully exercised by the tests.

## Tests

```bash
pytest
```
