"""Public async API — see docs/api-design.md.

Module-level ``connect()`` / ``serve()`` / ``discover()`` returning a single ``Connection``
(both roles; NetherNet is symmetric P2P) and a ``Server``. This mirrors how BDS wraps NetherNet
(a ``Connector`` producing per-connection ``NetworkPeer`` objects), expressed in idiomatic
asyncio (websockets-style): ``await send`` / ``await recv`` / ``async for`` / ``async with``.

It is a thin layer over the internal :class:`~nethernet.transport_api.Transport` engine: a
``Future`` resolves the connect/accept, and an ``asyncio.Queue`` per connection carries received
packets. The byte-exact protocol code underneath is untouched.
"""

from __future__ import annotations

import asyncio
import inspect
import math
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from nethernet.constants import (
    DEFAULT_APPLICATION_ID,
    DEFAULT_BROADCAST_PORT,
    DEFAULT_NEGOTIATION_TIMEOUT,
)
from nethernet.discovery.lan import Address
from nethernet.errors import (
    ConnectionClosed,
    ConnectionFailed,
    SendType,
    SessionError,
)
from nethernet.identity import (
    IdentityEnvelope,
    IdentitySigner,
    InvalidIdentity,
    ServerIdentity,
    extract_identity,
    insert_identity,
    strip_identity,
    verify_server_identity,
)
from nethernet.limits import ConnectionLimits
from nethernet.network_id import NetworkID, new_connection_id
from nethernet.session import Session, SessionState
from nethernet.signaling.http import HttpSignalingServer, _check_sdp_size, request_answer
from nethernet.transport_api import Transport

# Cap on packets buffered for an incoming connection before its Connection object exists (an app
# packet can in principle arrive between a channel opening and both channels being open).
_MAX_PENDING_PACKETS = 256


@dataclass(frozen=True)
class DiscoveredHost:
    """A host found via LAN discovery (docs/api-design.md s3.4)."""

    network_id: NetworkID
    advertisement: bytes  # opaque application data (SPEC.md s7.3 Response)
    address: Address  # source address; connect() reuses it for unicast signaling


class _Closed:
    """Sentinel queued on a connection's receive queue to wake a waiting ``recv()`` on close."""


_CLOSED = _Closed()


class Connection:
    """One connection to a peer (BDS ``WebRTCNetworkPeer``); a single class for both roles.

    ``endpoint`` is None for HTTP-signaled connections, which have no LAN transport to own.
    """

    def __init__(
        self, session: Session, endpoint: _Endpoint | None, *, owns_endpoint: bool = False,
        limits: ConnectionLimits | None = None,
    ) -> None:
        self._session = session
        self._endpoint = endpoint
        self._owns_endpoint = owns_endpoint
        self._recv_queue: asyncio.Queue[bytes | _Closed] = asyncio.Queue()
        self._connected: asyncio.Future[None] = asyncio.get_event_loop().create_future()
        self._closed = asyncio.Event()
        self._closed_error: SessionError | None = None
        self._limits = limits
        self._recv_bytes = 0
        self._close_task: asyncio.Task | None = None

    # -- identity (SPEC.md s3, s9) -----------------------------------------------------

    @property
    def remote_id(self) -> NetworkID:
        return self._session.remote_id

    @property
    def local_id(self) -> NetworkID:
        return self._session.local_id

    @property
    def connection_id(self) -> int:
        return self._session.connection_id

    @property
    def state(self) -> SessionState:
        return self._session.state

    # -- message I/O (SPEC.md s6.2-s6.3) -----------------------------------------------

    async def send(self, data: bytes, reliability: SendType = SendType.RELIABLE) -> None:
        """Send an application packet; raises :class:`ConnectionClosed` if not connected."""
        if not isinstance(data, bytes):
            raise TypeError("NetherNet messages must be bytes")
        if self._limits is not None:
            if self._session.state is not SessionState.CONNECTED:
                raise ConnectionClosed(self._closed_error or SessionError.NONE)
            permit = self._session._pc.reserve_send_nowait(len(data), reliability)
        else:
            permit = await self.reserve_send(len(data), reliability)
        try:
            permit.send_nowait(data)
        finally:
            permit.release()

    async def reserve_send(self, wire_bound: int, reliability: SendType = SendType.RELIABLE):
        """Reserve capacity for a complete message before encoding its payload."""
        if self._session.state is not SessionState.CONNECTED:
            raise ConnectionClosed(self._closed_error or SessionError.NONE)
        permit = await self._session._pc.reserve_send(wire_bound, reliability)
        if self._session.state is not SessionState.CONNECTED:
            permit.release()
            raise ConnectionClosed(self._closed_error or SessionError.NONE)
        return permit

    async def recv(self) -> bytes:
        """Return the next complete application packet (SPEC.md s6.4 FIFO).

        Raises :class:`ConnectionClosed` once the connection is closed and the queue is drained.
        """
        item = await self._recv_queue.get()
        if isinstance(item, _Closed):
            self._recv_queue.put_nowait(_CLOSED)  # sticky: subsequent recv() also raises
            raise ConnectionClosed(self._closed_error or SessionError.NONE)
        self._recv_bytes -= len(item)
        return item

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        try:
            return await self.recv()
        except ConnectionClosed as exc:
            if exc.error == SessionError.NONE:
                raise StopAsyncIteration from None
            raise

    # -- lifecycle (SPEC.md s9.5) ------------------------------------------------------

    async def close(self, error: SessionError = SessionError.NONE) -> None:
        await asyncio.shield(self._request_close(error))

    def _request_close(self, error: SessionError = SessionError.NONE) -> asyncio.Task:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close(error))
            self._close_task.add_done_callback(self._close_done)
        return self._close_task

    async def _close(self, error: SessionError) -> None:
        try:
            await self._session.aclose(error)
        finally:
            if self._owns_endpoint:
                await self._endpoint.aclose()

    @staticmethod
    def _close_done(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    async def _join_failed_connect(self) -> None:
        task = self._request_close(SessionError.GENERIC_FAILURE)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()

    async def wait_closed(self) -> SessionError:
        await self._closed.wait()
        await asyncio.shield(self._request_close(self._closed_error or SessionError.NONE))
        return self._closed_error or SessionError.NONE

    async def __aenter__(self) -> Connection:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- internal: driven by the endpoint's engine callbacks ----------------------------

    def _mark_connected(self) -> None:
        if not self._connected.done():
            self._connected.set_result(None)

    def _mark_closed(self, error: SessionError) -> None:
        if self._closed_error is not None:
            return
        self._closed_error = error
        if not self._connected.done():
            self._connected.set_exception(ConnectionFailed(error))
            self._connected.exception()  # mark retrieved; awaiting the future still raises
        self._recv_queue.put_nowait(_CLOSED)
        self._closed.set()
        self._request_close(error)

    def _enqueue(self, data: bytes) -> None:
        if self._closed_error is not None:
            return
        if self._limits is not None and (
            self._recv_queue.qsize() >= self._limits.receive_messages
            or self._recv_bytes + len(data) > self._limits.receive_bytes
        ):
            self._session.close(SessionError.GENERIC_FAILURE)
            return
        self._recv_bytes += len(data)
        self._recv_queue.put_nowait(data)

    async def _await_connected(self, timeout: float | None) -> None:
        # timeout=None blocks until the session opens (or raises ConnectionFailed if it closes
        # first); a timeout cancels the wait and reports the negotiation timeout.
        try:
            await asyncio.wait_for(self._connected, timeout)
        except TimeoutError:
            raise ConnectionFailed(
                SessionError.NEGOTIATION_TIMEOUT_WAITING_FOR_RESPONSE
            ) from None


class _Endpoint:
    """Internal: wraps one :class:`Transport`, routing engine callbacks to per-connection objects.

    BDS analog: ``NetherNetConnector`` (owns the ``connId -> peer`` map and the event queue).
    """

    def __init__(self, local_id: NetworkID, **transport_cfg: object) -> None:
        self._conns: dict[int, Connection] = {}
        self._pending_packets: dict[int, list[bytes]] = {}
        self._incoming: asyncio.Queue[Connection] = asyncio.Queue()
        self._discovered: asyncio.Queue[DiscoveredHost] = asyncio.Queue()
        self._transport = Transport(
            local_id,
            on_session=self._on_session_open,
            on_session_close=self._on_session_close,
            on_packet=self._on_packet,
            on_host_discovered=self._on_host_discovered,
            **transport_cfg,  # type: ignore[arg-type]
        )

    # -- lifecycle ---------------------------------------------------------------------

    async def start(self) -> None:
        await self._transport.start()

    async def aclose(self) -> None:
        # Closing the transport closes every live session, which marks each Connection closed
        # through the on_session_close callback below — no separate sweep needed here.
        await self._transport.aclose()

    @property
    def bound_port(self) -> int | None:
        return self._transport.bound_port

    @property
    def local_id(self) -> NetworkID:
        return self._transport.local_id

    def start_broadcasting(self) -> None:
        self._transport.start_broadcasting()

    def broadcast_request(self) -> None:
        self._transport._lan.broadcast_request()

    def set_advertisement(self, advertisement: bytes | None) -> None:
        self._transport.set_advertisement(advertisement)

    async def connect(self, remote_id: NetworkID, *, timeout: float | None) -> Connection:
        session = await self._transport.connect(remote_id)
        conn = self._register(session, owns_endpoint=True)
        await conn._await_connected(timeout)
        return conn

    def remember_peer(self, network_id: NetworkID, addr: Address) -> None:
        self._transport.remember_peer(network_id, addr)

    # -- engine callbacks --------------------------------------------------------------

    def _register(self, session: Session, *, owns_endpoint: bool) -> Connection:
        conn = Connection(session, self, owns_endpoint=owns_endpoint)
        self._conns[session.connection_id] = conn
        for data in self._pending_packets.pop(session.connection_id, []):
            conn._enqueue(data)
        return conn

    def _on_session_open(self, session: Session) -> None:
        conn = self._conns.get(session.connection_id)
        if conn is None:  # incoming connection
            conn = self._register(session, owns_endpoint=False)
            self._incoming.put_nowait(conn)
        conn._mark_connected()

    def _on_session_close(self, session: Session, error: SessionError) -> None:
        conn = self._conns.pop(session.connection_id, None)
        if conn is not None:
            conn._mark_closed(error)

    def _on_packet(self, session: Session, data: bytes) -> None:
        conn = self._conns.get(session.connection_id)
        if conn is not None:
            conn._enqueue(data)
            return
        buffer = self._pending_packets.setdefault(session.connection_id, [])
        if len(buffer) < _MAX_PENDING_PACKETS:
            buffer.append(data)

    def _on_host_discovered(self, network_id: NetworkID, data: bytes, addr: Address) -> None:
        self._discovered.put_nowait(DiscoveredHost(network_id, data, addr))


class Server:
    """Accepts incoming connections, one ``handler`` task each (BDS ServerNetherNetConnector)."""

    def __init__(
        self,
        endpoint: _Endpoint,
        handler: Callable[[Connection], Awaitable[None]],
        *,
        broadcast: bool,
    ) -> None:
        self._endpoint = endpoint
        self._handler = handler
        self._broadcast = broadcast
        self._accept_task: asyncio.Task | None = None
        self._handler_tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()
        self._close_task: asyncio.Task | None = None

    async def start(self) -> Server:
        await self._endpoint.start()
        if self._broadcast:
            self._endpoint.start_broadcasting()
        self._accept_task = asyncio.create_task(self._accept_loop())
        return self

    async def _accept_loop(self) -> None:
        while True:
            conn = await self._endpoint._incoming.get()
            task = asyncio.create_task(self._run_handler(conn))
            self._handler_tasks.add(task)
            task.add_done_callback(self._handler_tasks.discard)

    async def _run_handler(self, conn: Connection) -> None:
        try:
            await self._handler(conn)
        except ConnectionClosed:
            pass  # remote disconnect is a normal end of a handler
        finally:
            await conn.close()

    async def serve_forever(self) -> None:
        await self._closed.wait()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed.set()
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(Connection._close_done)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        if self._accept_task is not None:
            self._accept_task.cancel()
            await asyncio.gather(self._accept_task, return_exceptions=True)
        tasks = list(self._handler_tasks)
        for task in tasks:
            task.cancel()
        await self._endpoint.aclose()
        await asyncio.gather(*tasks, return_exceptions=True)

    def set_advertisement(self, advertisement: bytes | None) -> None:
        self._endpoint.set_advertisement(advertisement)

    @property
    def local_id(self) -> NetworkID:
        return self._endpoint.local_id

    @property
    def bound_port(self) -> int | None:
        return self._endpoint.bound_port

    @property
    def connections(self) -> list[Connection]:
        return list(self._endpoint._conns.values())

    async def __aenter__(self) -> Server:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


# --- module-level entry points (docs/api-design.md s3.1) ---


async def connect(
    remote: NetworkID | DiscoveredHost,
    *,
    local_id: NetworkID | None = None,
    application_id: int = DEFAULT_APPLICATION_ID,
    port: int = DEFAULT_BROADCAST_PORT,
    bind_host: str = "",
    broadcast_host: str = "255.255.255.255",
    broadcast_port: int | None = None,
    ice_servers: list | None = None,
    relay_only: bool = False,
    negotiation_timeout: float = DEFAULT_NEGOTIATION_TIMEOUT,
    timeout: float | None = None,
) -> Connection:
    """Dial a remote peer and return a connected :class:`Connection`.

    BDS analog: ``ClientNetherNetConnector``. Resolves once the session is CONNECTED; raises
    :class:`ConnectionFailed` on negotiation timeout / ICE failure.
    """
    endpoint = _Endpoint(
        local_id or NetworkID.p2p(new_connection_id()),
        application_id=application_id,
        port=port,
        bind_host=bind_host,
        broadcast_host=broadcast_host,
        broadcast_port=broadcast_port,
        ice_servers=ice_servers,
        relay_only=relay_only,
        negotiation_timeout=negotiation_timeout,
    )
    await endpoint.start()
    try:
        if isinstance(remote, DiscoveredHost):
            endpoint.remember_peer(remote.network_id, remote.address)
            remote_id = remote.network_id
        else:
            remote_id = remote
        return await endpoint.connect(remote_id, timeout=timeout)
    except BaseException:
        await endpoint.aclose()
        raise


def serve(
    handler: Callable[[Connection], Awaitable[None]],
    local_id: NetworkID,
    *,
    advertisement: bytes | None = None,
    application_id: int = DEFAULT_APPLICATION_ID,
    port: int = DEFAULT_BROADCAST_PORT,
    bind_host: str = "",
    broadcast_host: str = "255.255.255.255",
    broadcast_port: int | None = None,
    broadcast: bool = True,
    ice_servers: list | None = None,
    relay_only: bool = False,
    negotiation_timeout: float = DEFAULT_NEGOTIATION_TIMEOUT,
) -> Server:
    """Accept incoming connections (BDS ``ServerNetherNetConnector::host``).

    ``handler(connection)`` is awaited once per connection in its own task. Returns a
    :class:`Server` async context manager; ``await server.serve_forever()`` runs until closed.
    """
    endpoint = _Endpoint(
        local_id,
        advertisement=advertisement,
        application_id=application_id,
        port=port,
        bind_host=bind_host,
        broadcast_host=broadcast_host,
        broadcast_port=broadcast_port,
        ice_servers=ice_servers,
        relay_only=relay_only,
        negotiation_timeout=negotiation_timeout,
    )
    return Server(endpoint, handler, broadcast=broadcast)


async def discover(
    *,
    local_id: NetworkID | None = None,
    application_id: int = DEFAULT_APPLICATION_ID,
    port: int = DEFAULT_BROADCAST_PORT,
    bind_host: str = "",
    broadcast_host: str = "255.255.255.255",
    broadcast_port: int | None = None,
    timeout: float = 2.0,
) -> AsyncIterator[DiscoveredHost]:
    """Broadcast a Request and async-iterate discovered hosts (BDS ``NetherNetServerLocator``).

    Yields each distinct responder as its Response arrives, for up to ``timeout`` seconds.
    """
    endpoint = _Endpoint(
        local_id or NetworkID.p2p(new_connection_id()),
        application_id=application_id,
        port=port,
        bind_host=bind_host,
        broadcast_host=broadcast_host,
        broadcast_port=broadcast_port,
    )
    await endpoint.start()
    seen: set[NetworkID] = set()
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    try:
        endpoint.broadcast_request()
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            try:
                host = await asyncio.wait_for(endpoint._discovered.get(), timeout=remaining)
            except TimeoutError:
                return
            if host.network_id in seen:
                continue
            seen.add(host.network_id)
            yield host
    finally:
        await endpoint.aclose()


# --- HTTP signaling (NetherNet Onboarding Guide s4) — connect_http() / serve_http() ---


@dataclass(frozen=True)
class IncomingOffer:
    """An offer received over HTTP signaling, before any answer is generated."""

    network_id_text: str  # raw URL path segment; treat as opaque (Guide s9)
    network_id: NetworkID  # best-effort parsed form (may be unset)
    sdp: str  # offer SDP with a=identity already stripped
    identity: IdentityEnvelope | None  # client assertion; validating it is the app's policy


def _make_http_session(
    *,
    local_id: NetworkID,
    remote_id: NetworkID,
    is_dialer: bool,
    on_open: Callable[[Connection], None],
    on_close: Callable[[Connection, SessionError], None],
    ice_servers: list | None,
    relay_only: bool,
    negotiation_timeout: float,
    limits: ConnectionLimits | None = None,
) -> Connection:
    """A Session/Connection pair with no LAN transport (full ICE: nothing to trickle)."""
    slot: list[Connection] = []
    session = Session(
        connection_id=new_connection_id(),
        local_id=local_id,
        remote_id=remote_id,
        is_dialer=is_dialer,
        send_signal=lambda message: None,
        on_open=lambda session: on_open(slot[0]),
        on_close=lambda session, error: on_close(slot[0], error),
        on_packet=lambda session, data: slot[0]._enqueue(data),
        ice_servers=ice_servers,
        relay_only=relay_only,
        negotiation_timeout=negotiation_timeout,
        limits=limits,
    )
    conn = Connection(session, None, limits=limits)
    slot.append(conn)
    return conn


async def connect_http(
    server_url: str,
    *,
    local_id: NetworkID | None = None,
    ice_servers: list | None = None,
    relay_only: bool = False,
    negotiation_timeout: float = DEFAULT_NEGOTIATION_TIMEOUT,
    capability_check: bool = True,
    on_server_identity: Callable[[ServerIdentity], Awaitable[None] | None] | None = None,
    identity_signer: IdentitySigner | None = None,
    timeout: float | None = None,
    limits: ConnectionLimits | None = None,
    sign_offer: Callable[[str], str] | None = None,
) -> Connection:
    """Dial a host through HTTP signaling (Onboarding Guide s4, s7).

    Full ICE: the complete offer goes in one ``POST {server_url}/v1/join/{local_id}`` and the
    answer comes back in the response — no trickle, no LAN discovery. Requires the ``http``
    extra (aiohttp).

    ``on_server_identity`` receives the structurally verified server assertion (Guide s5.2);
    trust policy — TLS vs TOFU pinning of ``key_digest`` — is the caller's, and raising from
    the callback aborts the connection. Without it, any ``a=identity`` is stripped unverified.
    ``identity_signer`` signs the offer; BDS rejects an offer without ``a=identity``.
    ``identity_signer`` and ``sign_offer`` are mutually exclusive.
    """
    if identity_signer is not None and sign_offer is not None:
        raise ValueError("identity_signer and sign_offer are mutually exclusive")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Connection timeout must be positive and finite")
    local = local_id or NetworkID.p2p(new_connection_id())
    conn = _make_http_session(
        local_id=local,
        remote_id=NetworkID.unset(),
        is_dialer=True,
        on_open=Connection._mark_connected,
        on_close=Connection._mark_closed,
        ice_servers=ice_servers,
        relay_only=relay_only,
        negotiation_timeout=negotiation_timeout,
        limits=limits,
    )
    session = conn._session
    try:
        deadline = timeout if timeout is not None else (
            negotiation_timeout if limits is not None else None
        )
        async with asyncio.timeout(deadline):
            offer_sdp = await session.offer()
            if identity_signer is not None or sign_offer is not None:
                if any(line.strip().startswith("a=identity:") for line in offer_sdp.splitlines()):
                    raise InvalidIdentity("Offer already contains an identity assertion")
                if identity_signer is not None:
                    signed = identity_signer.sign(offer_sdp)
                else:
                    assert sign_offer is not None
                    signed = sign_offer(offer_sdp)
                if inspect.isawaitable(signed):
                    if inspect.iscoroutine(signed):
                        signed.close()
                    raise TypeError("Offer signer must return a string synchronously")
                if not isinstance(signed, str):
                    raise TypeError("Offer signer must return a string")
                if identity_signer is None:
                    if "\r" in signed or "\n" in signed:
                        raise InvalidIdentity("Identity envelope must contain one encoded value")
                    signed = insert_identity(offer_sdp, signed)
                if limits is not None:
                    _check_sdp_size(signed, limits)
                identities = 0
                in_media = False
                for line in signed.splitlines():
                    if line.startswith("m="):
                        in_media = True
                    if line.strip().startswith("a=identity:"):
                        if in_media or not line.startswith("a=identity:"):
                            raise InvalidIdentity("Offer identity must be session-level")
                        identities += 1
                if identities != 1:
                    raise InvalidIdentity("Signed offer must contain exactly one identity")
                stripped, _ = extract_identity(signed)
                if stripped != offer_sdp:
                    raise InvalidIdentity("Offer signer changed non-identity SDP")
                offer_sdp = signed
            answer_sdp = await request_answer(
                server_url,
                str(local),
                offer_sdp,
                capability_check=capability_check,
                timeout=negotiation_timeout,
                limits=limits,
            )
            if on_server_identity is not None:
                answer_sdp, server_identity = verify_server_identity(answer_sdp)
                result = on_server_identity(server_identity)
                if inspect.isawaitable(result):
                    await result
            else:
                answer_sdp = strip_identity(answer_sdp)
            await session.accept_answer(answer_sdp)
            await conn._await_connected(None)
        return conn
    except BaseException:
        await conn._join_failed_connect()
        raise


class HttpServer:
    """Accepts connections via HTTP signaling; one ``handler`` task per connection.

    The partner-flow analog of :class:`Server`: no LAN discovery or broadcast — clients learn
    the signaling URL out of band (Guide s3).
    """

    def __init__(
        self,
        handler: Callable[[Connection], Awaitable[None]],
        local_id: NetworkID,
        *,
        host: str,
        port: int,
        ssl_context: ssl.SSLContext | None,
        identity_signer: IdentitySigner | None,
        validate_offer: Callable[[IncomingOffer], Awaitable[None] | None] | None,
        ice_servers: list | None,
        relay_only: bool,
        negotiation_timeout: float,
        limits: ConnectionLimits | None = None,
    ) -> None:
        self._handler = handler
        self._local_id = local_id
        self._identity_signer = identity_signer
        self._validate_offer = validate_offer
        self._ice_servers = ice_servers
        self._relay_only = relay_only
        self._negotiation_timeout = negotiation_timeout
        self._limits = limits
        self._signaling = HttpSignalingServer(
            self._answer_offer, host=host, port=port, ssl_context=ssl_context, limits=limits
        )
        self._conns: dict[int, Connection] = {}
        self._handler_tasks: set[asyncio.Task] = set()
        self._closed = asyncio.Event()
        self._closing_conns: set[Connection] = set()
        self._close_task: asyncio.Task | None = None

    async def start(self) -> HttpServer:
        await self._signaling.start()
        return self

    # -- signaling callback ------------------------------------------------------------

    async def _answer_offer(self, network_id_text: str, offer_sdp: str) -> str:
        if self._closed.is_set():
            raise ConnectionFailed(SessionError.GENERIC_FAILURE)
        offer_sdp, envelope = extract_identity(offer_sdp)  # malformed -> InvalidIdentity -> 400
        remote_id = NetworkID.parse(network_id_text)
        if self._validate_offer is not None:
            result = self._validate_offer(
                IncomingOffer(network_id_text, remote_id, offer_sdp, envelope)
            )
            if inspect.isawaitable(result):
                await result
        conn = _make_http_session(
            local_id=self._local_id,
            remote_id=remote_id,
            is_dialer=False,
            on_open=self._handle_open,
            on_close=self._handle_close,
            ice_servers=self._ice_servers,
            relay_only=self._relay_only,
            negotiation_timeout=self._negotiation_timeout,
            limits=self._limits,
        )
        self._conns[conn.connection_id] = conn
        try:
            async with asyncio.timeout(
                self._negotiation_timeout if self._limits is not None else None
            ):
                answer_sdp = await conn._session.answer(offer_sdp)
            if self._identity_signer is not None:
                answer_sdp = self._identity_signer.sign(answer_sdp)
            if self._limits is not None:
                _check_sdp_size(answer_sdp, self._limits)
        except BaseException:
            self._conns.pop(conn.connection_id, None)
            await conn._join_failed_connect()
            raise
        return answer_sdp

    def _handle_open(self, conn: Connection) -> None:
        conn._mark_connected()
        task = asyncio.create_task(self._run_handler(conn))
        self._handler_tasks.add(task)
        task.add_done_callback(self._handler_tasks.discard)

    def _handle_close(self, conn: Connection, error: SessionError) -> None:
        self._conns.pop(conn.connection_id, None)
        conn._mark_closed(error)
        self._closing_conns.add(conn)
        conn._close_task.add_done_callback(lambda task: self._closing_conns.discard(conn))

    async def _run_handler(self, conn: Connection) -> None:
        try:
            await self._handler(conn)
        except ConnectionClosed:
            pass  # remote disconnect is a normal end of a handler
        finally:
            await conn.close()

    # -- lifecycle ---------------------------------------------------------------------

    async def serve_forever(self) -> None:
        await self._closed.wait()

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closed.set()
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(Connection._close_done)
        await asyncio.shield(self._close_task)

    async def _close(self) -> None:
        await self._signaling.aclose()
        # Close sessions first so handlers end via ConnectionClosed; then reap stragglers.
        conns = set(self._conns.values()) | self._closing_conns
        await asyncio.gather(*(conn.close() for conn in conns))
        self._conns.clear()
        for task in list(self._handler_tasks):
            task.cancel()
        await asyncio.gather(*self._handler_tasks, return_exceptions=True)

    @property
    def local_id(self) -> NetworkID:
        return self._local_id

    @property
    def bound_port(self) -> int | None:
        return self._signaling.bound_port

    @property
    def connections(self) -> list[Connection]:
        return list(self._conns.values())

    async def __aenter__(self) -> HttpServer:
        return await self.start()

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


def serve_http(
    handler: Callable[[Connection], Awaitable[None]],
    local_id: NetworkID | None = None,
    *,
    host: str = "0.0.0.0",
    port: int = 0,
    ssl_context: ssl.SSLContext | None = None,
    identity_signer: IdentitySigner | None = None,
    validate_offer: Callable[[IncomingOffer], Awaitable[None] | None] | None = None,
    ice_servers: list | None = None,
    relay_only: bool = False,
    negotiation_timeout: float = DEFAULT_NEGOTIATION_TIMEOUT,
    limits: ConnectionLimits | None = None,
) -> HttpServer:
    """Accept connections via HTTP signaling (Onboarding Guide s4). Requires ``nethernet[http]``.

    ``handler(connection)`` is awaited once per connection in its own task. ``validate_offer``
    is the authorization hook — it sees the raw network id, the stripped offer SDP, and the
    client's identity envelope (validating the token is the application's policy); raise
    :class:`~nethernet.signaling.http.SignalingRejected` to refuse. ``identity_signer`` adds
    the server assertion that real Minecraft clients require on every answer (Guide s5.2).
    """
    return HttpServer(
        handler,
        local_id or NetworkID.p2p(new_connection_id()),
        host=host,
        port=port,
        ssl_context=ssl_context,
        identity_signer=identity_signer,
        validate_offer=validate_offer,
        ice_servers=ice_servers,
        relay_only=relay_only,
        negotiation_timeout=negotiation_timeout,
        limits=limits,
    )
