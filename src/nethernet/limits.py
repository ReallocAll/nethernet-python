"""Optional, immutable limits for HTTP-signaled direct host connections."""

from dataclasses import dataclass, fields
from math import isfinite

from aioice import IceLimits
from aiortc import SctpLimits


@dataclass(frozen=True)
class ConnectionLimits:
    max_message_size: int = 2 * 16 * 1024 * 1024 + 1024
    send_bytes: int = 64 * 1024 * 1024 + 4096
    send_chunks: int = 65536
    send_messages: int = 512
    send_waiters: int = 64
    receive_messages: int = 256
    receive_bytes: int = 2 * (2 * 16 * 1024 * 1024 + 1024)
    incomplete_timeout: float = 10.0
    sctp_receive_bytes: int = 4 * 1024 * 1024
    sctp_receive_entries: int = 8192
    sctp_active_streams: int = 16
    udp_write_bytes: int = 512 * 1024
    udp_receive_bytes: int = 2 * 1024 * 1024
    udp_receive_datagrams: int = 2048
    http_sdp_bytes: int = 1024 * 1024
    ice_candidates: int = 64

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name == "incomplete_timeout":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError("Incomplete-message timeout must be positive and finite")
                if not isfinite(value) or value <= 0:
                    raise ValueError("Incomplete-message timeout must be positive and finite")
            elif type(value) is not int or value <= 0:
                raise ValueError("Connection limits must be positive integers")
        if self.max_message_size > 0x3FBFF01:
            raise ValueError("Message limit exceeds NetherNet framing capacity")

    def sctp_limits(self) -> SctpLimits:
        return SctpLimits(
            send_bytes=self.send_bytes,
            send_chunks=self.send_chunks,
            send_messages=self.send_messages,
            send_waiters=self.send_waiters,
            receive_bytes=self.sctp_receive_bytes,
            receive_entries=self.sctp_receive_entries,
            active_streams=self.sctp_active_streams,
            max_message_size=262144,
        )

    def ice_limits(self) -> IceLimits:
        return IceLimits(
            write_buffer_bytes=self.udp_write_bytes,
            receive_bytes=self.udp_receive_bytes,
            receive_datagrams=self.udp_receive_datagrams,
            candidates=self.ice_candidates,
        )
