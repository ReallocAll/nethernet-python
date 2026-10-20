"""aiortc PeerConnection wrapper — SPEC.md s6.1-s6.2, s4.

Wraps an aiortc ``RTCPeerConnection`` with the NetherNet specifics: the two data channels
(``ReliableDataChannel`` / ``UnreliableDataChannel``), application framing (Task 7), SDP
``max-message-size`` munging (so a full 262144-byte fragment is permitted — see
tasks/spike-aiortc.md), and CANDIDATEADD <-> ICE-candidate translation.

aiortc is non-trickle: ``setLocalDescription`` blocks until ICE gathering completes, so the
offer/answer returned here already embeds all candidates (SPEC.md s9.3). Inbound trickle
candidates are still accepted via ``add_remote_candidate`` (buffered until the remote
description is set, SPEC.md s5.3).

aiortc API verified against the official reference:
https://aiortc.readthedocs.io/en/latest/api.html

Note (RelayOnly, SPEC.md s4): aiortc's ``RTCConfiguration`` has no relay-only transport policy,
so RelayOnly cannot be fully enforced. As best-effort we filter our advertised candidates to
relay-only; remote/host candidate *usage* is not prevented by aiortc.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass

from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
from aiortc.stats import RTCTransportStats

from nethernet.errors import SendType
from nethernet.limits import ConnectionLimits
from nethernet.transport.framing import FRAGMENT_SIZE, MAX_PACKET_SIZE, PacketQueue

RELIABLE_LABEL = "ReliableDataChannel"  # SPEC.md s6.2
UNRELIABLE_LABEL = "UnreliableDataChannel"  # SPEC.md s6.2
MAX_MESSAGE_SIZE = 262144  # SPEC.md s6.1


@dataclass(frozen=True, slots=True)
class DtlsByteCounts:
    """Cumulative bytes at the DTLS transport's ICE-facing boundary."""

    bytes_sent: int
    bytes_received: int


class SendPermit:
    """One application-message reservation, transferred by synchronous commit."""

    def __init__(self, peer: PeerConnection, bound: int, reservation) -> None:
        self._peer = peer
        self._bound = bound
        self._reservation = reservation

    def send_nowait(self, data: bytes) -> None:
        try:
            if not isinstance(data, bytes) or len(data) > self._bound:
                raise ValueError("Application message exceeds its send reservation")
            if self._peer._closing:
                raise ConnectionError("NetherNet peer is closed")
            count = max(1, (len(data) + FRAGMENT_SIZE - 1) // FRAGMENT_SIZE)
            fragments = [
                bytes((count - index - 1,))
                + data[index * FRAGMENT_SIZE : (index + 1) * FRAGMENT_SIZE]
                for index in range(count)
            ]
            self._reservation.send_nowait(fragments)
        except BaseException as error:
            self._peer._fail(error)
            raise

    def release(self) -> None:
        self._reservation.release()


def set_max_message_size(sdp: str, value: int = MAX_MESSAGE_SIZE) -> str:
    """Force ``a=max-message-size`` in an SDP (replace if present, else insert) — SPEC.md s6.1."""
    if "a=max-message-size:" in sdp:
        return re.sub(r"a=max-message-size:\d+", f"a=max-message-size:{value}", sdp)
    out = []
    for line in sdp.splitlines(keepends=True):
        out.append(line)
        if line.startswith("a=sctp-port:"):
            newline = "\r\n" if line.endswith("\r\n") else "\n"
            out.append(f"a=max-message-size:{value}{newline}")
    return "".join(out)


def strip_candidates(sdp: str) -> tuple[str, list[str]]:
    """Split an SDP into a candidate-free SDP and the list of ``candidate:...`` strings.

    The LAN signaling Message payload is capped at 1140 bytes (SPEC.md s7.3), too small for an
    SDP with all candidates embedded. So the session sends the candidate-free offer/answer and
    streams candidates separately as CANDIDATEADD messages (trickle ICE, SPEC.md s9.3).
    """
    kept: list[str] = []
    candidates: list[str] = []
    for line in sdp.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("a=candidate:"):
            candidates.append(stripped[len("a=") :])  # "candidate:..."
        elif stripped == "a=end-of-candidates":
            continue  # we trickle candidates separately
        else:
            kept.append(line)
    return "".join(kept), candidates


def keep_only_relay_candidates(sdp: str) -> str:
    """Drop every ``a=candidate:`` line that is not ``typ relay`` (best-effort RelayOnly)."""
    out = [
        line
        for line in sdp.splitlines(keepends=True)
        if not (line.startswith("a=candidate:") and " typ relay" not in line)
    ]
    return "".join(out)


def candidate_to_message(candidate) -> str:  # noqa: ANN001 - aiortc RTCIceCandidate
    """Serialize an ICE candidate to CANDIDATEADD text (with the ``candidate:`` prefix)."""
    return "candidate:" + candidate_to_sdp(candidate)


def candidate_from_message(text: str):  # -> aiortc RTCIceCandidate
    """Parse CANDIDATEADD text into an ICE candidate (SPEC.md s5.4).

    aiortc's ``candidate_from_sdp`` expects the value without the ``candidate:`` prefix, so it is
    stripped here. The candidate is tagged with ``sdpMLineIndex = 0`` (SPEC.md s5.3: empty
    sdp-mid / mline-index 0) so aiortc can match it to the single bundled m-section.
    """
    value = text[len("candidate:") :] if text.startswith("candidate:") else text
    candidate = candidate_from_sdp(value)
    candidate.sdpMLineIndex = 0
    return candidate


class PeerConnection:
    """A NetherNet WebRTC peer: two data channels, framing, and candidate/SDP handling."""

    def __init__(
        self,
        *,
        is_dialer: bool,
        ice_servers: list[RTCIceServer] | None = None,
        relay_only: bool = False,
        on_packet: Callable[[bytes], None] | None = None,
        on_channels_open: Callable[[], None] | None = None,
        on_connection_state_change: Callable[[str], None] | None = None,
        limits: ConnectionLimits | None = None,
    ) -> None:
        if limits is not None and (ice_servers != [] or relay_only):
            raise ValueError("Bounded NetherNet requires explicit direct host ICE")
        configuration = RTCConfiguration(
            iceServers=ice_servers,
            sctpLimits=limits.sctp_limits() if limits is not None else None,
            iceLimits=limits.ice_limits() if limits is not None else None,
        )
        self._pc = RTCPeerConnection(configuration=configuration)
        self._limits = limits
        self._closing = False
        self._close_task: asyncio.Task | None = None
        self._remote_candidate_count = 0
        self._relay_only = relay_only
        self._on_packet = on_packet
        self._on_channels_open = on_channels_open
        self._on_connection_state_change = on_connection_state_change

        self._packets = PacketQueue(
            lambda fragment: self._channel_send(True, fragment),
            lambda fragment: self._channel_send(False, fragment),
            limits=limits,
            on_error=self._fail,
        )
        self._reliable: RTCDataChannel | None = None
        self._unreliable: RTCDataChannel | None = None
        self._reliable_open = False
        self._unreliable_open = False
        self._pending_reliable: list[bytes] = []
        self._pending_unreliable: list[bytes] = []
        self._channels_open_fired = False

        self._remote_description_set = False
        self._pending_remote_candidates: list = []

        @self._pc.on("connectionstatechange")
        def _on_connectionstatechange() -> None:
            if self._on_connection_state_change is not None:
                self._on_connection_state_change(self._pc.connectionState)

        if is_dialer:
            # The dialer MUST create exactly two data channels in its offer (SPEC.md s6.2).
            self._register_channel(self._pc.createDataChannel(RELIABLE_LABEL))
            self._register_channel(
                self._pc.createDataChannel(UNRELIABLE_LABEL, ordered=False, maxRetransmits=0)
            )
        else:

            @self._pc.on("datachannel")
            def _on_datachannel(channel: RTCDataChannel) -> None:
                self._register_channel(channel)

    def get_dtls_byte_counts(self) -> DtlsByteCounts:
        """Snapshot the pinned aiortc DTLS counters, including after transport close."""
        sctp = self._pc.sctp
        if sctp is None:
            return DtlsByteCounts(0, 0)
        report = sctp.transport._get_stats()
        if len(report) != 1:
            raise RuntimeError("Expected one DTLS transport statistics entry")
        stats = next(iter(report.values()))
        if (
            not isinstance(stats, RTCTransportStats)
            or stats.type != "transport"
            or type(stats.bytesSent) is not int
            or type(stats.bytesReceived) is not int
            or stats.bytesSent < 0
            or stats.bytesReceived < 0
        ):
            raise RuntimeError("Unexpected DTLS transport statistics entry")
        return DtlsByteCounts(stats.bytesSent, stats.bytesReceived)

    # -- SDP / negotiation -------------------------------------------------------------

    async def create_offer(self) -> str:
        """Create the offer and return its SDP (candidates embedded; max-message-size munged)."""
        await self._pc.setLocalDescription(await self._pc.createOffer())
        return self._local_sdp()

    async def create_answer(self, remote_offer_sdp: str) -> str:
        """Apply the remote offer and return our answer SDP."""
        self._validate_sdp(remote_offer_sdp)
        await self._pc.setRemoteDescription(RTCSessionDescription(remote_offer_sdp, "offer"))
        await self._flush_remote_candidates()
        await self._pc.setLocalDescription(await self._pc.createAnswer())
        return self._local_sdp()

    async def set_remote_answer(self, remote_answer_sdp: str) -> None:
        """Apply the remote answer (dialer side)."""
        self._validate_sdp(remote_answer_sdp)
        await self._pc.setRemoteDescription(RTCSessionDescription(remote_answer_sdp, "answer"))
        await self._flush_remote_candidates()

    async def add_remote_candidate(self, candidate_text: str) -> None:
        """Add an inbound trickle candidate, buffering until the remote description is set."""
        if self._limits is not None:
            if (
                len(candidate_text) > self._limits.http_sdp_bytes
                or self._remote_candidate_count >= self._limits.ice_candidates
            ):
                raise ValueError("NetherNet ICE candidate limit exceeded")
            self._remote_candidate_count += 1
        candidate = candidate_from_message(candidate_text)
        if self._remote_description_set:
            await self._pc.addIceCandidate(candidate)
        else:
            self._pending_remote_candidates.append(candidate)

    async def _flush_remote_candidates(self) -> None:
        self._remote_description_set = True
        for candidate in self._pending_remote_candidates:
            await self._pc.addIceCandidate(candidate)
        self._pending_remote_candidates.clear()

    def _local_sdp(self) -> str:
        sdp = set_max_message_size(self._pc.localDescription.sdp)
        if self._relay_only:
            sdp = keep_only_relay_candidates(sdp)
        self._validate_sdp(sdp)
        return sdp

    def _validate_sdp(self, sdp: str) -> None:
        if self._limits is None:
            return
        if (
            len(sdp) > self._limits.http_sdp_bytes
            or len(sdp.encode()) > self._limits.http_sdp_bytes
        ):
            raise ValueError("NetherNet SDP size limit exceeded")
        candidates = 0
        media = 0
        for line in sdp.splitlines():
            if line.startswith("a=candidate:"):
                candidates += 1
                if candidates > self._limits.ice_candidates:
                    raise ValueError("NetherNet ICE candidate limit exceeded")
            elif line.startswith("m="):
                if not line.startswith("m=application "):
                    raise ValueError("Bounded NetherNet requires a data-only SDP")
                media += 1
        if media != 1:
            raise ValueError("Bounded NetherNet requires one data transport")

    # -- data channels -----------------------------------------------------------------

    def send(self, data: bytes, send_type: SendType) -> bool:
        """Fragment and send an application packet; returns False if dropped (SPEC.md s6.3)."""
        if self._limits is not None:
            self._send_shape(len(data), send_type)
        try:
            return self._packets.push(data, send_type)
        except BaseException as error:
            self._fail(error)
            raise

    def _send_shape(self, wire_bound: int, reliability: SendType) -> tuple:
        maximum = self._limits.max_message_size if self._limits is not None else MAX_PACKET_SIZE
        if type(wire_bound) is not int or not 0 <= wire_bound <= maximum:
            raise ValueError("Invalid NetherNet application message size")
        if reliability not in (SendType.RELIABLE, SendType.UNRELIABLE):
            raise ValueError("Unsupported NetherNet reliability")
        if reliability != SendType.RELIABLE and (
            self._limits is not None or wire_bound > FRAGMENT_SIZE
        ):
            raise ValueError("This message requires reliable NetherNet delivery")
        if self._closing:
            raise ConnectionError("NetherNet peer is closed")
        channel = self._reliable if reliability == SendType.RELIABLE else self._unreliable
        if channel is None or channel.readyState != "open":
            raise ConnectionError("NetherNet data channel is not open")
        count = max(1, (wire_bound + FRAGMENT_SIZE - 1) // FRAGMENT_SIZE)
        sizes = (FRAGMENT_SIZE + 1,) * (count - 1) + (
            wire_bound - FRAGMENT_SIZE * (count - 1) + 1,
        )
        return channel, sizes

    async def reserve_send(
        self, wire_bound: int, reliability: SendType = SendType.RELIABLE
    ) -> SendPermit:
        channel, sizes = self._send_shape(wire_bound, reliability)
        reservation = await channel.transport.reserve_data_channel(channel, sizes)
        return SendPermit(self, wire_bound, reservation)

    def reserve_send_nowait(self, wire_bound: int, reliability: SendType) -> SendPermit:
        channel, sizes = self._send_shape(wire_bound, reliability)
        reservation = channel.transport.reserve_data_channel_nowait(channel, sizes)
        return SendPermit(self, wire_bound, reservation)

    def read(self, max_size: int | None = None) -> bytes | None:
        return self._packets.read(max_size)

    def peek(self) -> int | None:
        return self._packets.peek()

    def _register_channel(self, channel: RTCDataChannel) -> None:
        # Channel selection is by exact label (SPEC.md s6.2): only "ReliableDataChannel" is
        # the reliable/ordered queue; anything else is unreliable/unordered.
        is_reliable = channel.label == RELIABLE_LABEL
        if self._limits is not None:
            if channel.label not in (RELIABLE_LABEL, UNRELIABLE_LABEL) or (
                is_reliable
                and (
                    self._reliable is not None
                    or not channel.ordered
                    or channel.maxRetransmits is not None
                    or channel.maxPacketLifeTime is not None
                )
            ) or (not is_reliable and self._unreliable is not None):
                self._fail(ValueError("Incompatible NetherNet data channel"))
                return
        if is_reliable:
            self._reliable = channel
        else:
            self._unreliable = channel

        @channel.on("open")
        def _on_open() -> None:
            self._handle_channel_open(is_reliable)

        @channel.on("message")
        def _on_message(message: bytes | str) -> None:
            self._handle_channel_message(is_reliable, message)

        @channel.on("close")
        def _on_close() -> None:
            if not self._closing:
                self._fail(ConnectionError("NetherNet data channel closed"))

        if channel.readyState == "open":
            self._handle_channel_open(is_reliable)

    def _handle_channel_open(self, is_reliable: bool) -> None:
        if is_reliable:
            if self._reliable_open:
                return
            self._reliable_open = True
            channel, pending = self._reliable, self._pending_reliable
        else:
            if self._unreliable_open:
                return
            self._unreliable_open = True
            channel, pending = self._unreliable, self._pending_unreliable
        for fragment in pending:
            channel.send(fragment)
        pending.clear()
        if (
            self._reliable_open
            and self._unreliable_open
            and not self._channels_open_fired
            and self._on_channels_open is not None
        ):
            self._channels_open_fired = True
            self._on_channels_open()

    def _handle_channel_message(self, is_reliable: bool, message: bytes | str) -> None:
        if self._closing:
            return
        data = message if isinstance(message, (bytes, bytearray)) else str(message).encode()
        try:
            if is_reliable:
                self._packets.on_reliable_message(bytes(data))
            else:
                self._packets.on_unreliable_message(bytes(data))
            self._deliver()
        except ValueError as error:
            self._fail(error)

    def _deliver(self) -> None:
        if self._on_packet is None:
            return
        while (packet := self._packets.read()) is not None:
            self._on_packet(packet)

    def _channel_send(self, is_reliable: bool, fragment: bytes) -> None:
        channel = self._reliable if is_reliable else self._unreliable
        is_open = self._reliable_open if is_reliable else self._unreliable_open
        if channel is not None and is_open:
            channel.send(fragment)
        else:
            pending = self._pending_reliable if is_reliable else self._pending_unreliable
            pending.append(fragment)

    @property
    def connection_state(self) -> str:
        return self._pc.connectionState

    async def close(self) -> None:
        await asyncio.shield(self._request_close())

    def _request_close(self) -> asyncio.Task:
        if self._close_task is None:
            self._closing = True
            self._packets.close()
            self._pending_reliable.clear()
            self._pending_unreliable.clear()
            self._pending_remote_candidates.clear()
            self._close_task = asyncio.create_task(self._pc.close())
            self._close_task.add_done_callback(self._close_done)
        return self._close_task

    @staticmethod
    def _close_done(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    def _fail(self, error: BaseException) -> None:
        if not self._closing:
            self._request_close()
            if self._on_connection_state_change is not None:
                self._on_connection_state_change("failed")
