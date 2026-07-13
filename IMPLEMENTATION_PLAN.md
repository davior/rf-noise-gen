# rfnoise → Signal Generator: Revised Implementation Plan

## Context

`rfnoise` today does one thing: **random-hop broadband noise** across user-defined
frequency ranges, driving tinySA Ultra / HackRF One / RTL-SDR (RX-only) / mock through
a common device abstraction. Dave wrote a detailed plan to turn it into a general
**signal generator** with four features: (1) mode select noise-vs-sweep,
(2) sequential sweep across ranges, (3) AM/FM modulation, (4) sweep within a band.

That plan was written from the README, and explicitly asked to be verified against the
real source. This document is that verification: it keeps the plan's sound
architecture (a three-axis decomposition) and corrects the concrete assumptions that
don't match the code, so it can be handed to Claude Code phase-by-phase.

**Deliverable of this session:** commit this revised plan to branch
`claude/plan-review-feedback-7qdrrc` as `IMPLEMENTATION_PLAN.md` (implementation of the
phases themselves is separate, later work).

**Reproducibility scope (decided):** schedule-level only. The seed must reproduce the
frequency / power / strategy / random-modulation-*parameter* schedule bit-for-bit. The
raw noise/IQ *byte* stream does not need to be reproducible, so the noise generator
keeps its own independent RNG and must never draw from the schedule stream.

---

## What the original plan got right (keep)

- **Three-axis decomposition** — tuning strategy × modulation × modulating source — maps
  cleanly onto the real seams. `engine.py` fuses "what frequency"
  (`RandomBandSelector.next()`) with "emit" (`device.broadcast()`); both are single,
  replaceable call sites.
- **Dependency ordering** — stdlib phases (sequential + stepped sweep) before the numpy
  `[dsp]` phase (AM/FM/chirp). Correct; delivers 2 of 4 features with zero new deps.
- **Invariant 1 (per-component RNG sub-streams)** is the highest-risk correctness point
  and is correctly identified. See below for why it bites.
- **Ignore-with-warning** fallback pattern (mirroring how power ranges are already
  validated) is the right way to handle unsupported modulations.

## Corrections applied vs. the original plan

1. **Emit path is start/stop, not center/bandwidth.** Real signature:
   `RFDevice.broadcast(start_hz, stop_hz, dwell_s, power_dbm)` in
   `rfnoise/devices/base.py`. The engine passes `band.start_hz, band.stop_hz`; each
   device computes center/width itself. `Emission` must carry `start_hz/stop_hz`, not
   `center_hz/bandwidth_hz`.
2. **Extend `DeviceCapabilities`, don't replace it.** The real frozen dataclass has
   `tx_bands: Tuple[TxBand, ...]`, `default_band_width`, `clamp_power()`, and *computed*
   `freq_min_hz/freq_max_hz`. New fields get **added with defaults**; the proposed
   from-scratch rewrite would break every device constructor.
3. **No schema migration needed.** `rfnoise/session.py` already has `SCHEMA_VERSION = 1`,
   already absorbs missing fields via `.get()` defaults in `Session.from_dict`, and only
   rejects versions *newer* than supported. New optional fields with defaults are
   automatically backward-compatible — **do not** bump the version or write a migration
   function unless an existing field's meaning changes.
4. **"random-hop" is width-weighted, not uniform.** `build_bands()` (`rfnoise/bands.py`)
   flattens every range into a pool of slices; `RandomBandSelector.next()` does
   `rng.choice(pool)`, so wider ranges get proportionally more airtime. The golden test
   must lock in exactly this. Sequential sweep is a deliberately different distribution.
5. **Reproducibility is already only schedule-deep.** HackRF's `make_noise_samples()`
   uses its own local `random.Random(seed=None)`, unwired from `session.seed`. Matches
   the decided scope — keep noise RNG independent of the schedule stream.
6. **tinySA native sweep already half-exists.** `rfnoise/devices/tinysa.py` already has a
   `_COMMANDS["sweep"]` entry and issues `sweep {start} {stop} 450` in sweep mode.
   Phase 2's tinySA work is wiring, not net-new (still verify the sweep-time arg against
   firmware `help`).
7. **No golden *file* exists** — only two-runs-are-equal self-consistency tests. Add a
   checked-in golden fixture as the first commit of Phase 0.

**Opportunistic fixes:** `rfnoise/devices/mock.py` annotates `HopRecord.power_dbm:
Optional[...]` without importing `Optional` (latent bug); the `Session.overlap` field
exists but isn't exposed in the interactive editor.

---

## Invariants — do not break these

1. **Seeded reproducibility (schedule-level).** One `random.Random(session.seed)` is
   shared today between band selection *and* power draws in `engine.py`. Any new consumer
   of that stream (strategy shuffles, random modulation params) shifts existing runs
   unless it draws from its **own** deterministically-derived sub-stream
   (`random.Random(hash((seed, "label")) & 0xFFFFFFFF)` per component). The noise/IQ byte
   generator must stay entirely off this stream.
2. **`mock` is the safe default and emits nothing.** Every new path fully exercisable on
   `mock` with zero hardware.
3. **"Never have to look it up."** Devices declare their own capabilities; extend this to
   the new axes (`supported_modulations`, `supported_traversals`) rather than making the
   user configure what the hardware knows.
4. **Old sessions keep loading unchanged** — guaranteed for free by `from_dict` defaults
   (correction #3). Round-trip test `examples/sample_session.json`.
5. **Pure-stdlib core stays pure.** numpy enters **only** for IQ generation, behind a new
   `[dsp]` extra. Without numpy: noise, CW, sequential sweep, stepped intra-band sweep all
   work; AM/FM/chirp raise a clear "install `.[dsp]`" error.
6. **Regulatory posture preserved.** Keep transmit warnings and dummy-load guidance in
   README/CLI. Per-session frequency allow-list (Future Work) is the guardrail before any
   expanded real-TX use.

---

## Phase 0 — Seams (no behavior change)

Pure refactor; golden test green before and after.

**Tasks**
- **First commit: add a golden fixture.** Serialize `NoiseGenerator(session).plan(N)` for
  a fixed seed + representative multi-range session to a checked-in JSON; assert equality
  in `tests/test_golden.py`. This makes the rest of the refactor provably
  output-preserving. (Extends the existing `test_dry_run_plan_is_deterministic` pattern.)
- Extract `RandomBandSelector` selection into a `tuning.py` `TuningStrategy` interface
  with `RandomPooledStrategy` reproducing current behavior **exactly** (width-weighted
  `rng.choice(pool)`). `bands.py` keeps `build_bands`/`split_range` math only.
- Add new fields to the **existing** `DeviceCapabilities` (frozen dataclass, defaults so
  constructors keep working):
  ```python
  supported_modulations: frozenset[Modulation] = frozenset({Modulation.NONE})
  supported_traversals: frozenset[Traversal] = frozenset({Traversal.RANDOM_HOP})
  instantaneous_bw_hz: Optional[int] = None   # IQ chirp cap; None = n/a
  modulation_fidelity: str = "none"           # "iq" | "fixed_tone" | "none"
  ```
- Introduce an `Emission` dataclass carrying **start/stop** (not center):
  ```python
  @dataclass(frozen=True)
  class Emission:
      start_hz: int
      stop_hz: int
      dwell_s: float
      power_dbm: Optional[float] = None
      modulation: Modulation = Modulation.NONE
      source: Optional[ModSource] = None
      deviation_hz: Optional[float] = None   # FM
      depth: Optional[float] = None          # AM 0..1
      tone_hz: Optional[float] = None
      sweep: Optional[SweepSpec] = None
  ```
  Add `RFDevice.emit(Emission)` as the new abstract path; keep `broadcast(...)` as a thin
  concrete adapter that builds a `NONE`-modulation `Emission` and calls `emit`. One call
  site in the engine, four drivers — smallest safe diff, cleanest seam for later axes.
- Enums (`devices/base.py`): `Modulation{NONE,AM,FM}`, `ModSource{TONE,NOISE}`,
  `Traversal{RANDOM_HOP,SEQUENTIAL,SWEEP_IN_BAND}`.

**Done when:** golden test passes unchanged; `list-devices` prints correctly; all
existing tests green; zero behavior diff.

## Phase 1 — Sequential sweep (Features 1 & 2, stdlib)

- `SequentialSweepStrategy` in `tuning.py`: yield the flat band pool in deterministic
  ascending order, wrapping at end. Draws **no** RNG (so it can't perturb power draws —
  verify the golden test for random-hop is still green).
- Add `traversal: Traversal = Traversal.RANDOM_HOP` to `Session` (`model.py`) — remember
  all four touch points: dataclass field, `to_dict`, `from_dict`, `__post_init__` if
  validated.
- Engine picks the strategy from `session.traversal`; hop/dwell loop unchanged.
- Surface in `interactive.py` (menu item), `gui.py` (a `DEVICE_OPTION_FIELDS`-style enum
  widget + `collect_session`/`session_to_form`), `cli.py` (`--traversal` on `run`).
- `status.py`: show traversal mode (add field to `HopStatus` + `.line()` + engine
  construction).

**Tests** (`tests/test_tuning.py`): sequential order correct + wrap; random-hop golden
unchanged; per-range bandwidth overrides honored in both.

## Phase 2 — Stepped intra-band sweep (Feature 4, stdlib)

Cover a band wider than the device's single-burst cap by stepping center frequency across
it over the dwell. Works on every device (no IQ).

- `SweepSpec` dataclass: `start_hz, stop_hz, mode ("stepped"|"continuous"), steps,
  duration` (defaults to dwell).
- `SweepInBandStrategy` in `tuning.py`: when requested band width exceeds
  `min(range override, device max_bandwidth_hz)`, emit a stepped schedule across the band
  over the dwell; otherwise a single emission.
- Engine translates the step schedule into per-step `Emission`s, budgeting `dwell_seconds`
  across steps.
- tinySA: reuse the **existing** `_COMMANDS["sweep"]` path (`sweep start stop time`) — verify
  the sweep-time argument against firmware `help`.
- HackRF: stepped retune (restart `hackrf_transfer` per step for MVP — matches today's
  per-hop gap).
- mock: record the step sequence in `history`.
- Session/UI/GUI/CLI: expose steps/rate (continuous disabled until Phase 3). Config is
  **per-session with optional per-range override**, mirroring the existing
  `FrequencyRange.max_bandwidth_hz` override.

**Tests** (`tests/test_intra_band_sweep.py`): steps cover `[start, stop]` with no
gaps/overshoot; total time ≈ dwell; narrow band → single emission (no spurious stepping).

## Phase 3 — Modulation: AM / FM + continuous chirp (Feature 3, numpy `[dsp]`)

**3a. DSP core** (`modulation.py`, `sources.py`), pure-numpy, side-effect-free:
- AM: envelope `(1 + depth·m(t))` × `exp(j·2π·f_offset·t)`.
- FM: `exp(j·2π·Δf·∫m(t)dt)`.
- Continuous chirp (linear FM): phase = integral of a linear frequency ramp; capped at
  `instantaneous_bw_hz`.
- Sources produce `m(t)` normalized to [-1, 1]: `tone` (sine at `tone_hz`), `noise`
  (broadband for MVP). **Noise draws from its own independent RNG, never the schedule
  stream** (satisfies both Invariant 1 and the schedule-level reproducibility decision).

**3b. Device wiring** (all IQ paths require `[dsp]`):
- mock: generate IQ, record modulation params + a cheap measured summary
  (deviation/depth). Primary DSP test target.
- HackRF One: generate IQ ourselves, stream via `hackrf_transfer -t <file>`, per-hop
  restart (MVP). Map dBm→gain as today (approximate). Gapless streaming → Future Work.
- tinySA Ultra: crude **fixed-tone** AM/FM via signal-gen serial commands
  (`modulation_fidelity="fixed_tone"`, `supported_modulations={NONE,AM,FM}`). Cannot do
  arbitrary-source modulation → noise-source request warns and falls back. Centralize
  strings in `_COMMANDS`; verify against firmware.
- RTL-SDR: unchanged; `supported_modulations=frozenset()`.

**3c. Engine validation & fallback** (reuse the power-range validation pattern):
- modulation ∉ `supported_modulations` → warn, fall back to CW.
- source incompatible with `modulation_fidelity` → warn, fall back to device best.
- continuous chirp > `instantaneous_bw_hz` → warn, fall back to stepped (Phase 2).

**3d. Packaging:** add `[dsp] = ["numpy"]` to `pyproject.toml`; import numpy lazily inside
`modulation.py` with a clear ImportError pointing to `pip install -e .[dsp]`.

**Tests** (`tests/test_modulation.py`): FM deviation via peak of derivative of unwrapped
phase; AM depth via `(max-min)/(max+min)`; chirp instantaneous freq linear in time; tone
freq + noise reproducibility of *parameters*. `tests/test_engine_integration.py`:
compositions on mock (`sequential × fm × noise`, `random × am × tone`); fallback warnings
fire.

## Phase 4 — Integration & polish (no migration function needed)

- **Session schema:** stays at `SCHEMA_VERSION = 1`. New optional fields with defaults are
  already backward-compatible. Add a round-trip test proving `examples/sample_session.json`
  (a real v1 envelope) still loads → runs unchanged. Only bump the version if an existing
  field's interpretation changes.
- `interactive.py`: coherent flow for mode/traversal, intra-band sweep, modulation +
  source + params. Fold in the missing `overlap` editor while here.
- `gui.py` (Dear PyGui): enum controls for the three axes via the `DEVICE_OPTION_FIELDS`
  pattern; update `collect_session`/`session_to_form` (both directions). Spectrum bar
  graph (`DecayPlotModel`) can annotate swept/modulated emissions.
- `cli.py`: flags for the new axes; `list-devices` prints supported modulations,
  traversals, instantaneous BW, and fidelity.
- `status.py`: include modulation + sweep info in the live line and per-hop log.
- `README.md`: update the device capability table (per-device modulation support +
  fidelity), state the HackRF per-hop-restart limitation, keep all transmit warnings, and
  add `tuning.py`/`modulation.py`/`sources.py` to the Architecture listing.

**Fix opportunistically along the way:** import `Optional` in `mock.py`.

---

## PR breakdown (each independently green)

0. Phase 0 seams + golden fixture (pure refactor).
1. Sequential sweep (Features 1 & 2).
2. Stepped intra-band sweep (Feature 4, stdlib) + tinySA sweep wiring.
3. DSP core + mock modulation + `[dsp]` extra + DSP tests (no hardware).
4. HackRF IQ path + tinySA fixed-tone + engine fallback.
5. Session fields wired through front-ends + README + `list-devices` + `overlap` editor.

---

## Resolved open decisions

1. **Mock modulation without numpy?** No — `[dsp]`-gate it. Don't duplicate the DSP in
   `cmath`.
2. **`emit(Emission)` vs threading through `broadcast()`?** `emit(Emission)`; keep
   `broadcast()` as a thin adapter. One call site, four drivers, clean seam.
3. **Intra-band sweep config granularity?** Per-session with optional per-range override,
   mirroring the existing `FrequencyRange.max_bandwidth_hz`.

## Future Work (unchanged, parked)

Per-session frequency allow-list (F1) · scripted timeline/playlist (F2) · timestamped
JSONL/CSV log (F3) · TX→RX loopback self-test (F4) · band-limited noise +
phase-continuous switching (F5) · calibration table (F6) · SoapySDR/pyhackrf gapless
backend (F7) · more modulations/sources (F8) · more tuning strategies (F9).

---

## Verification (for this session's deliverable)

This session only writes the plan doc. To verify: confirm `IMPLEMENTATION_PLAN.md` is
committed to `claude/plan-review-feedback-7qdrrc` and its file/function references resolve
against the tree (`rfnoise/devices/base.py`, `rfnoise/engine.py`, `rfnoise/bands.py`,
`rfnoise/model.py`, `rfnoise/session.py`). Per-phase verification lives in each phase's
"Tests"/"Done when" above; every path is exercisable on `mock` with `pytest`.
