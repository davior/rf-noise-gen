import random

import pytest

from rfnoise.bands import (
    Band,
    RandomBandSelector,
    build_bands,
    drift_offset,
    effective_bandwidth,
    split_range,
)
from rfnoise.model import FrequencyRange


def test_split_range_sequential():
    rng = FrequencyRange(100_000, 200_000, 10_000)
    bands = split_range(rng, 10_000)
    assert len(bands) == 10
    assert bands[0].start_hz == 100_000
    assert bands[0].stop_hz == 110_000
    assert bands[-1].stop_hz == 200_000  # clipped to upper bound


def test_split_range_clips_final_band():
    rng = FrequencyRange(0, 25_000)
    bands = split_range(rng, 10_000)
    assert [b.stop_hz for b in bands] == [10_000, 20_000, 25_000]
    assert bands[-1].width_hz == 5_000


def test_split_range_width_never_exceeds_range():
    rng = FrequencyRange(0, 5_000)
    bands = split_range(rng, 10_000)  # bw wider than the range
    assert len(bands) == 1
    assert bands[0].start_hz == 0 and bands[0].stop_hz == 5_000


def test_split_range_overlap():
    rng = FrequencyRange(0, 30_000)
    bands = split_range(rng, 10_000, overlap=0.5)
    # step = 5000, so starts at 0,5k,10k,... each 10k wide
    assert bands[0].start_hz == 0 and bands[0].stop_hz == 10_000
    assert bands[1].start_hz == 5_000


def test_effective_bandwidth_precedence():
    rng = FrequencyRange(0, 100_000_000)
    # device cap applies when no override
    assert effective_bandwidth(rng, device_max=20_000_000, device_default=1_000_000) == 20_000_000
    # override narrows below device cap
    rng2 = FrequencyRange(0, 100_000_000, max_bandwidth_hz=5_000_000)
    assert effective_bandwidth(rng2, device_max=20_000_000, device_default=1_000_000) == 5_000_000
    # no cap and no override -> device default
    assert effective_bandwidth(rng, device_max=None, device_default=1_000_000) == 1_000_000


def test_effective_bandwidth_never_exceeds_range():
    rng = FrequencyRange(0, 3_000)
    assert effective_bandwidth(rng, device_max=20_000_000, device_default=1_000_000) == 3_000


def test_build_bands_pools_all_ranges():
    ranges = [FrequencyRange(0, 20_000), FrequencyRange(100_000, 130_000)]
    bands = build_bands(ranges, device_max=10_000, device_default=10_000)
    assert len(bands) == 2 + 3


def test_random_selector_is_seeded():
    ranges = [FrequencyRange(0, 100_000)]
    bands = build_bands(ranges, device_max=10_000, device_default=10_000)
    a = [RandomBandSelector(bands, seed=42).next() for _ in range(5)]
    b = [RandomBandSelector(bands, seed=42).next() for _ in range(5)]
    assert a == b


def test_random_selector_empty_pool():
    with pytest.raises(ValueError):
        RandomBandSelector([])


# -- band drift -------------------------------------------------------------
def test_split_range_records_parent_range():
    rng = FrequencyRange(100, 200, 10)
    bands = split_range(rng, 10)
    assert all(b.range_lower_hz == 100 and b.range_upper_hz == 200 for b in bands)


def test_drift_offset_interior_uses_full_reach():
    # [110,120] inside [100,200], reach = 0.5 * 10 = 5 -> delta in [-5, 5].
    band = Band(110, 120, 100, 200)
    r = random.Random(0)
    offs = [drift_offset(band, 0.5, r) for _ in range(2000)]
    assert all(-5 <= d <= 5 for d in offs)
    assert min(offs) <= -4 and max(offs) >= 4  # spans both directions


def test_drift_offset_low_edge_only_drifts_inward():
    # [100,110] touches the range floor -> can only move up: delta in [0, 5].
    band = Band(100, 110, 100, 200)
    r = random.Random(0)
    offs = [drift_offset(band, 0.5, r) for _ in range(2000)]
    assert all(0 <= d <= 5 for d in offs)


def test_drift_offset_high_edge_only_drifts_inward():
    # [190,200] touches the range ceiling -> can only move down: delta in [-5, 0].
    band = Band(190, 200, 100, 200)
    r = random.Random(0)
    offs = [drift_offset(band, 0.5, r) for _ in range(2000)]
    assert all(-5 <= d <= 0 for d in offs)


def test_drift_offset_never_leaves_range():
    rng = FrequencyRange(100, 200, 10)
    bands = split_range(rng, 10)
    r = random.Random(1)
    for band in bands:
        for _ in range(200):
            d = drift_offset(band, 0.5, r)
            assert band.start_hz + d >= 100
            assert band.stop_hz + d <= 200


def test_drift_offset_zero_fraction_is_off():
    band = Band(110, 120, 100, 200)
    r = random.Random(0)
    assert all(drift_offset(band, 0.0, r) == 0 for _ in range(50))


def test_drift_offset_band_filling_range_cannot_move():
    band = Band(100, 200, 100, 200)  # fills its whole range
    r = random.Random(0)
    assert all(drift_offset(band, 0.5, r) == 0 for _ in range(50))


def test_drift_offset_without_range_bounds_uses_reach_only():
    band = Band(0, 100)  # no parent range -> only the +/- reach cap applies
    r = random.Random(0)
    offs = [drift_offset(band, 0.5, r) for _ in range(2000)]
    assert all(-50 <= d <= 50 for d in offs)
    assert min(offs) <= -45 and max(offs) >= 45


def test_drift_offset_reproducible_with_seeded_rng():
    band = Band(110, 120, 100, 200)
    a = [drift_offset(band, 0.5, random.Random(0)) for _ in range(1)]
    b = [drift_offset(band, 0.5, random.Random(0)) for _ in range(1)]
    seq_a = [drift_offset(band, 0.5, r) for r in [random.Random(5)] for _ in range(10)]
    seq_b = [drift_offset(band, 0.5, r) for r in [random.Random(5)] for _ in range(10)]
    assert a == b and seq_a == seq_b
