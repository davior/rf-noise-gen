import os

import pytest

from rfnoise import session as session_store
from rfnoise.model import FrequencyRange, Session


def _sample_session():
    return Session(
        name="test",
        device="mock",
        device_options={"verbose": False},
        ranges=[
            FrequencyRange(100_000, 200_000, 10_000),
            FrequencyRange(1_000_000, 2_000_000),  # no bandwidth -> device auto
        ],
        dwell_seconds=0.25,
        seed=7,
        pause_seconds=2.0,
        pause_every_hops=10,
        power_min_dbm=-60.0,
        power_max_dbm=-30.0,
    )


def test_save_load_round_trip(tmp_path):
    session = _sample_session()
    path = os.path.join(tmp_path, "s.json")
    session_store.save(session, path)
    loaded = session_store.load(path)
    assert loaded.to_dict() == session.to_dict()
    assert loaded.ranges[1].max_bandwidth_hz is None
    assert loaded.power_min_dbm == -60.0
    assert loaded.power_max_dbm == -30.0
    assert loaded.has_power_range
    assert loaded.pause_seconds == 2.0
    assert loaded.pause_every_hops == 10
    assert loaded.has_pause


def test_power_range_defaults_none():
    from rfnoise.model import Session
    s = Session()
    assert s.power_min_dbm is None and not s.has_power_range


def test_pause_defaults_disabled():
    s = Session()
    assert s.pause_seconds == 0.0 and s.pause_every_hops == 0
    assert not s.has_pause


def test_load_tolerates_missing_pause_keys(tmp_path):
    # Older session files without pause_* keys must still load (defaults).
    import json
    data = _sample_session().to_dict()
    del data["pause_seconds"]
    del data["pause_every_hops"]
    path = os.path.join(tmp_path, "legacy.json")
    with open(path, "w") as fh:
        json.dump({"schema_version": 1, "session": data}, fh)
    loaded = session_store.load(path)
    assert loaded.pause_seconds == 0.0 and loaded.pause_every_hops == 0
    assert not loaded.has_pause


def test_power_range_rejects_inverted():
    from rfnoise.model import Session
    with pytest.raises(ValueError):
        Session(power_min_dbm=-20.0, power_max_dbm=-60.0)


def test_drift_round_trip(tmp_path):
    session = _sample_session()
    session.drift_fraction = 0.5
    path = os.path.join(tmp_path, "drift.json")
    session_store.save(session, path)
    loaded = session_store.load(path)
    assert loaded.drift_fraction == 0.5
    assert loaded.has_drift
    assert loaded.to_dict() == session.to_dict()


def test_drift_defaults_none():
    s = Session()
    assert s.drift_fraction is None and not s.has_drift


def test_drift_rejects_negative():
    with pytest.raises(ValueError):
        Session(drift_fraction=-0.1)


def test_load_old_session_without_drift(tmp_path):
    # A payload predating the field still loads (drift stays off).
    import json
    data = _sample_session().to_dict()
    data.pop("drift_fraction", None)
    path = os.path.join(tmp_path, "legacy.json")
    with open(path, "w") as fh:
        json.dump({"schema_version": 1, "session": data}, fh)
    loaded = session_store.load(path)
    assert loaded.drift_fraction is None and not loaded.has_drift


def test_load_bare_dict(tmp_path):
    import json
    path = os.path.join(tmp_path, "bare.json")
    with open(path, "w") as fh:
        json.dump(_sample_session().to_dict(), fh)
    loaded = session_store.load(path)
    assert loaded.name == "test"


def test_load_rejects_future_schema(tmp_path):
    import json
    path = os.path.join(tmp_path, "future.json")
    with open(path, "w") as fh:
        json.dump({"schema_version": 999, "session": _sample_session().to_dict()}, fh)
    with pytest.raises(ValueError):
        session_store.load(path)


def test_list_sessions(tmp_path):
    session_store.save(_sample_session(), os.path.join(tmp_path, "a.json"))
    session_store.save(_sample_session(), os.path.join(tmp_path, "b.json"))
    found = session_store.list_sessions(str(tmp_path))
    assert len(found) == 2


def test_default_path_sanitises_name():
    path = session_store.default_path_for("My Session!", directory="/x")
    assert path.endswith("My_Session.json")


def test_range_rejects_bad_bounds():
    with pytest.raises(ValueError):
        FrequencyRange(200_000, 100_000)
