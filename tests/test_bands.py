import pytest

from rfnoise.bands import (
    RandomBandSelector,
    build_bands,
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
