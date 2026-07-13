"""Tests for tuning strategies (Phase 1) and traversal wiring."""

import pytest

from rfnoise.bands import build_bands
from rfnoise.devices.base import Traversal
from rfnoise.devices.mock import MockDevice
from rfnoise.engine import NoiseGenerator
from rfnoise.model import FrequencyRange, Session
from rfnoise.tuning import (
    RandomPooledStrategy,
    SequentialSweepStrategy,
    TuningStrategy,
)


def _pool(ranges, device_max=10_000, device_default=10_000):
    return build_bands(ranges, device_max=device_max, device_default=device_default)


# -- SequentialSweepStrategy ------------------------------------------------
def test_sequential_is_ascending_and_covers_every_band_once():
    bands = _pool([FrequencyRange(0, 50_000)])  # 5 bands of 10 kHz
    strat = SequentialSweepStrategy(bands)
    assert isinstance(strat, TuningStrategy)
    got = [strat.next() for _ in range(len(bands))]
    starts = [b.start_hz for b in got]
    assert starts == sorted(starts)                 # ascending
    assert set(got) == set(bands)                   # every band exactly once


def test_sequential_wraps_after_last_band():
    bands = _pool([FrequencyRange(0, 30_000)])  # 3 bands
    strat = SequentialSweepStrategy(bands)
    seq = [strat.next() for _ in range(len(bands) + 2)]
    assert seq[len(bands)] == seq[0]                # wrapped back to the start
    assert seq[len(bands) + 1] == seq[1]


def test_sequential_orders_across_interleaved_ranges():
    # Ranges given high-then-low must still sweep globally low-to-high.
    bands = _pool([FrequencyRange(100_000, 120_000),
                   FrequencyRange(0, 20_000)])
    strat = SequentialSweepStrategy(bands)
    starts = [strat.next().start_hz for _ in range(len(bands))]
    assert starts == sorted(starts)


def test_sequential_is_deterministic():
    bands = _pool([FrequencyRange(0, 50_000)])
    a = [b.start_hz for b in (lambda s: [s.next() for _ in range(7)])(SequentialSweepStrategy(bands))]
    b = [b.start_hz for b in (lambda s: [s.next() for _ in range(7)])(SequentialSweepStrategy(bands))]
    assert a == b


def test_sequential_empty_pool_raises():
    with pytest.raises(ValueError):
        SequentialSweepStrategy([])


def test_sequential_honors_per_range_bandwidth_override():
    # A 50 kHz range capped to 10 kHz -> 5 bands, all <= 10 kHz, still ascending.
    bands = _pool([FrequencyRange(0, 50_000, max_bandwidth_hz=10_000)],
                  device_max=1_000_000)
    strat = SequentialSweepStrategy(bands)
    swept = [strat.next() for _ in range(len(bands))]
    assert all(b.width_hz <= 10_000 for b in swept)
    assert [b.start_hz for b in swept] == sorted(b.start_hz for b in swept)


# -- engine selects the strategy from the session ---------------------------
def _mock():
    return MockDevice(verbose=False, sleep=False, max_bandwidth_hz=10_000)


def _session(**kw):
    base = dict(name="t", device="mock",
                ranges=[FrequencyRange(0, 50_000)], dwell_seconds=0.0, seed=1)
    base.update(kw)
    return Session(**base)


def test_engine_uses_sequential_strategy_when_selected():
    gen = NoiseGenerator(_mock(), _session(traversal=Traversal.SEQUENTIAL))
    assert isinstance(gen.selector, SequentialSweepStrategy)
    gen.run(iterations=len(gen.bands))
    starts = [rec.start_hz for rec in gen.device.history]
    assert starts == sorted(starts)     # swept low-to-high


def test_engine_defaults_to_random_pooled_strategy():
    gen = NoiseGenerator(_mock(), _session())
    assert isinstance(gen.selector, RandomPooledStrategy)


def test_sequential_run_is_reproducible_with_seed():
    # The sweep order is fixed and the power draws come from the seeded RNG, so
    # two runs with the same seed produce an identical schedule.
    def run():
        gen = NoiseGenerator(
            _mock(),
            _session(traversal=Traversal.SEQUENTIAL,
                     power_min_dbm=-60.0, power_max_dbm=-20.0, seed=99),
        )
        gen.run(iterations=8)
        return [(r.start_hz, r.stop_hz, round(r.power_dbm, 9))
                for r in gen.device.history]

    assert run() == run()


# -- session persistence of the traversal field -----------------------------
def test_session_traversal_round_trips():
    s = Session(ranges=[FrequencyRange(0, 10_000)], traversal=Traversal.SEQUENTIAL)
    restored = Session.from_dict(s.to_dict())
    assert restored.traversal == Traversal.SEQUENTIAL


def test_session_accepts_traversal_as_string():
    s = Session(ranges=[FrequencyRange(0, 10_000)], traversal="sequential")
    assert s.traversal == Traversal.SEQUENTIAL


def test_old_session_without_traversal_defaults_to_random_hop():
    # A pre-Phase-1 session dict has no 'traversal' key.
    data = Session(ranges=[FrequencyRange(0, 10_000)]).to_dict()
    del data["traversal"]
    assert Session.from_dict(data).traversal == Traversal.RANDOM_HOP
