"""CLI tests for the Phase 4 modulation surfacing (flags + list-devices)."""

import json

from rfnoise.cli import main
from rfnoise.model import FrequencyRange, Session


def _write_session(tmp_path):
    sess = Session(name="cli", device="mock", dwell_seconds=0.0,
                   ranges=[FrequencyRange(100_000_000, 100_100_000)], seed=1)
    path = tmp_path / "s.json"
    path.write_text(json.dumps(sess.to_dict()))
    return str(path)


def test_run_applies_modulation_overrides(tmp_path, capsys):
    path = _write_session(tmp_path)
    rc = main(["run", path, "--modulation", "fm", "--source", "noise",
               "--deviation", "8000", "--iterations", "1", "--quiet"])
    assert rc == 0
    banner = capsys.readouterr().out
    assert "FM from noise" in banner


def test_run_plain_has_no_modulation_banner(tmp_path, capsys):
    path = _write_session(tmp_path)
    main(["run", path, "--iterations", "1", "--quiet"])
    out = capsys.readouterr().out
    assert "FM" not in out and "AM" not in out


def test_list_devices_prints_modulation(capsys):
    assert main(["list-devices"]) == 0
    out = capsys.readouterr().out
    assert "AM, FM" in out          # HackRF/mock advertise AM/FM
    assert "fixed_tone" in out      # tinySA fidelity
    assert "modulation" in out      # the describe() label
