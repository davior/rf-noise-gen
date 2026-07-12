"""Device registry.

Maps short device keys (as stored in session files and used on the CLI) to the
concrete :class:`~rfnoise.devices.base.RFDevice` subclasses.
"""

from __future__ import annotations

from typing import Dict, List, Type

from .base import (
    DeviceCapabilities,
    DeviceError,
    RFDevice,
    TransmitNotSupported,
    TxBand,
)
from .hackrf import HackRFOne
from .mock import MockDevice
from .rtlsdr import RTLSDR
from .tinysa import TinySAUltra

_REGISTRY: Dict[str, Type[RFDevice]] = {
    "mock": MockDevice,
    "tinysa": TinySAUltra,
    "hackrf": HackRFOne,
    "rtlsdr": RTLSDR,
}


def device_keys() -> List[str]:
    """Return the registered device keys in a stable order."""
    return list(_REGISTRY.keys())


def get_device_class(key: str) -> Type[RFDevice]:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise DeviceError(
            f"unknown device {key!r}; choose one of: {', '.join(_REGISTRY)}"
        )


def create_device(key: str, **options) -> RFDevice:
    """Instantiate a device by key, passing through constructor options."""
    return get_device_class(key)(**options)


__all__ = [
    "DeviceCapabilities",
    "DeviceError",
    "RFDevice",
    "TransmitNotSupported",
    "TxBand",
    "HackRFOne",
    "MockDevice",
    "RTLSDR",
    "TinySAUltra",
    "device_keys",
    "get_device_class",
    "create_device",
]
