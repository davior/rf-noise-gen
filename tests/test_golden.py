"""Golden reproducibility test (Invariant 1).

Locks the exact schedule (band + power per hop) produced by the default
``random-hop x none`` behavior for a fixed seed. Any refactor that changes the
existing RNG stream -- e.g. a new stochastic component drawing from the shared
engine RNG instead of its own sub-stream -- will change these values and fail
here. The fixture in ``tests/golden/schedule_v1.json`` was captured before the
signal-generator refactor began.
"""

import json
import os

from rfnoise.devices.mock import MockDevice
from rfnoise.engine import NoiseGenerator
from rfnoise.model import FrequencyRange, Session

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "schedule_v1.json")


def _build():
    """Rebuild the exact device+session the golden fixture was captured from."""
    session = Session(
        name="golden",
        device="mock",
        ranges=[
            FrequencyRange(100_000, 500_000),
            FrequencyRange(1_000_000, 1_050_000, 10_000),
            FrequencyRange(2_400_000_000, 2_400_050_000),
        ],
        dwell_seconds=0.0,
        seed=20240713,
        power_min_dbm=-70.0,
        power_max_dbm=-25.0,
    )
    device = MockDevice(max_bandwidth_hz=100_000, verbose=False, sleep=False)
    return device, session


def test_default_schedule_matches_golden():
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)

    device, session = _build()
    gen = NoiseGenerator(device, session)
    gen.run(iterations=golden["iterations"])
    schedule = [[r.start_hz, r.stop_hz, r.power_dbm] for r in device.history]

    assert schedule == golden["schedule"]
