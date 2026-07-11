"""RTL-SDR driver -- receive only.

The RTL-SDR is a receive-only USB dongle (Realtek RTL2832U). It **cannot
transmit**, so it is included in the abstraction for completeness but any
attempt to broadcast raises :class:`TransmitNotSupported`. Its receive tuning
range and bandwidth are advertised for reference and for a possible monitor
role.
"""

from __future__ import annotations

from .base import DeviceCapabilities, RFDevice, TransmitNotSupported, TxBand

MAX_SAMPLE_RATE = 2_400_000  # stable RX rate (3.2 Msps is unstable)


class RTLSDR(RFDevice):
    """Receive-only SDR. Present in the registry; refuses to transmit.

    Options:
      * ``device_index`` -- RTL-SDR device index (default 0).
    """

    def __init__(self, device_index: int = 0, **options):
        super().__init__(**options)
        self.device_index = device_index
        self.capabilities = DeviceCapabilities(
            name="RTL-SDR",
            can_transmit=False,
            # Typical R820T2 tuning range; receive only.
            tx_bands=(TxBand(500_000, 1_766_000_000, "rx"),),
            max_bandwidth_hz=MAX_SAMPLE_RATE,
            default_band_width=MAX_SAMPLE_RATE,
            description="Receive-only dongle (RTL2832U); cannot broadcast.",
        )

    def broadcast(self, start_hz: int, stop_hz: int, dwell_s: float) -> None:
        raise TransmitNotSupported(
            "RTL-SDR is receive-only and cannot transmit RF signals."
        )
