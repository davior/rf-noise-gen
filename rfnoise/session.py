"""Load and save sessions as versioned JSON files."""

from __future__ import annotations

import json
import os
from typing import List

from .model import Session

SCHEMA_VERSION = 1
DEFAULT_SESSION_DIR = os.environ.get(
    "RFNOISE_SESSION_DIR",
    os.path.join(os.getcwd(), "sessions"),
)


def save(session: Session, path: str) -> str:
    """Write ``session`` to ``path`` as JSON. Returns the path written."""
    payload = {"schema_version": SCHEMA_VERSION, "session": session.to_dict()}
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def load(path: str) -> Session:
    """Load a session from a JSON file, tolerating older/looser payloads."""
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "session" in payload:
        version = payload.get("schema_version", SCHEMA_VERSION)
        if version > SCHEMA_VERSION:
            raise ValueError(
                f"session schema v{version} is newer than supported "
                f"v{SCHEMA_VERSION}; upgrade rfnoise."
            )
        data = payload["session"]
    else:
        # Bare session dict (no envelope) -- accept for convenience.
        data = payload
    return Session.from_dict(data)


def list_sessions(directory: str = DEFAULT_SESSION_DIR) -> List[str]:
    """Return sorted absolute paths of ``*.json`` sessions in ``directory``."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".json")
    )


def default_path_for(name: str, directory: str = DEFAULT_SESSION_DIR) -> str:
    """Build a filesystem path for a session ``name`` in ``directory``."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_")
    if not safe:
        safe = "session"
    return os.path.join(directory, f"{safe}.json")
