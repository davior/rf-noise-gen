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
    )


def test_save_load_round_trip(tmp_path):
    session = _sample_session()
    path = os.path.join(tmp_path, "s.json")
    session_store.save(session, path)
    loaded = session_store.load(path)
    assert loaded.to_dict() == session.to_dict()
    assert loaded.ranges[1].max_bandwidth_hz is None


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
