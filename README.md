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
- **Configurable dwell time** with seamless (no-pause) switching.
- **Random broadcast strength.** Optionally give a dBm range and each hop
  transmits at a random level drawn from it (see below).
- **Live run status.** While running, a status line shows the current frequency,
  band, output level, hop count and rate.
- **Pluggable device abstraction**: tinySA Ultra, HackRF One, RTL-SDR, plus a
  software `mock` for testing.
- **Interactive session editor** that persists to JSON you can reopen later.

## Auto max bandwidth per device

The "maximum bandwidth per broadcast" is the widest continuous signal a device
can emit in one burst. These are built in, so you never have to look them up:

| Device        | Transmit | Frequency range         | Auto max broadcast bandwidth | Output level |
|---------------|----------|-------------------------|------------------------------|--------------|
| HackRF One    | yes      | 1 MHz – 6 GHz           | **20 MHz** (20 Msps limit)   | -50…5 dBm (mapped to 0–47 dB gain, approx.) |
| tinySA Ultra  | yes      | 100 kHz – 5.4 GHz       | CW generator → no fixed cap; uses a default band width (1 MHz sweep / 100 kHz cw) | -110…-20 dBm (calibrated) |
| RTL-SDR       | **no**   | ~500 kHz – 1.766 GHz    | receive-only (cannot broadcast) | n/a |
| mock          | yes      | 0 – 6 GHz               | 20 MHz (configurable)        | -120…10 dBm (recorded only) |

The engine picks the effective band width as
`min(range override if set, device hardware cap if any, else device default)`,
never wider than the range itself. Run `rfnoise list-devices` to print these.

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
pip install -e .[dev]       # + pytest
```

The HackRF driver shells out to `hackrf_transfer` (install the `hackrf`
system package). RTL-SDR support is receive-only and included for completeness.

## Usage

Interactive editor (default when run with no arguments):

```bash
rfnoise            # or: rfnoise ui  /  python -m rfnoise
```

Menu: set a name, add ranges (enter bounds as `100k`, `2.4M`, `433.9MHz` …),
choose a device and its options, set dwell/seed and an optional strength range,
then **save** to a session file and **run**. Saved sessions live under
`sessions/` and can be reopened.

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
  bands.py       band splitting + random pooled selection
  engine.py      NoiseGenerator: validation + hop/dwell loop + power draws
  status.py      live/log run-status reporters (HopStatus)
  session.py     versioned JSON load/save
  interactive.py menu-driven session editor
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
