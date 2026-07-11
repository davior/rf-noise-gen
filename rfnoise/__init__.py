"""RF Noise Generator -- frequency-hopping RF noise across user-defined ranges.

Public API:
    * :class:`~rfnoise.model.FrequencyRange`, :class:`~rfnoise.model.Session`
    * :class:`~rfnoise.engine.NoiseGenerator`
    * device registry helpers in :mod:`rfnoise.devices`
"""

from __future__ import annotations

from .engine import ConfigurationError, NoiseGenerator
from .freq import format_freq, parse_freq
from .model import FrequencyRange, Session

__version__ = "0.1.0"

__all__ = [
    "FrequencyRange",
    "Session",
    "NoiseGenerator",
    "ConfigurationError",
    "parse_freq",
    "format_freq",
    "__version__",
]
