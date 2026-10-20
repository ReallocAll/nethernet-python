import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from aiohttp import web

import nethernet
import nethernet.api as api
from nethernet.session_manager import SessionManager
from nethernet.signaling.http import request_answer
from nethernet.transport.peer_connection import PeerConnection
from nethernet.transport_api import Transport


async def test_manager_joins_a_session_removed_during_logical_close():
    release = asyncio.Event()
    closing = asyncio.create_task(release.wait())
    session = SimpleNamespace(connection_id=1, _close_task=closing)

    async def aclose():
        await asyncio.shield(closing)

    session.aclose = aclose
    manager = SessionManager(local_id=nethernet.NetworkID.p2p(1), send_signal=lambda *args: None)
    manager._sessions[1] = session
    manager._handle_session_close(session, nethernet.SessionError.NONE)
    stopping = asyncio.create_task(manager.aclose())
    try:
        await asyncio.sleep(0)
        assert not stopping.done()
    finally:
        release.set()
        await stopping
        await closing


async def test_transport_stops_signaling_admission_and_joins_existing_work():
    transport = Transport(nethernet.NetworkID.p2p(1))
    entered = asyncio.Event()
    finished = asyncio.Event()

    async def dispatch(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    transport._manager.handle_signal = dispatch
    transport._on_lan_signal(nethernet.NetworkID.p2p(2), "test")
    await entered.wait()
    try:
        await transport.aclose()
        assert finished.is_set()
        transport._on_lan_signal(nethernet.NetworkID.p2p(2), "late")
        assert not transport._tasks
    finally:
        tasks = list(transport._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_http_request_and_chunked_response_caps():
    requests = []

    async def join(request):
        requests.append(await request.read())
        response = web.StreamResponse(headers={"Content-Type": "application/sdp"})
        await response.prepare(request)
        await response.write(b"v=0\n1234")
        await response.write(b"5")
        return response

    app = web.Application()
    app.router.add_post("/v1/join/{id}", join)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    limits = nethernet.ConnectionLimits(http_sdp_bytes=8)
    try:
        with pytest.raises(nethernet.ConnectionFailed):
            await request_answer(url, "1", "123456789", capability_check=False, limits=limits)
        assert requests == []
        with pytest.raises(nethernet.ConnectionFailed):
            await request_answer(url, "1", "v=0\n", capability_check=False, limits=limits)
        assert requests == [b"v=0\n"]
    finally:
        await runner.cleanup()


async def test_http_server_caps_body_before_offer_callback():
    offers = []

    async def handler(conn):
        pass

    async with nethernet.serve_http(
        handler, host="127.0.0.1", ice_servers=[],
        limits=nethernet.ConnectionLimits(http_sdp_bytes=8), validate_offer=offers.append,
    ) as server:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                f"http://127.0.0.1:{server.bound_port}/v1/join/1", data=b"123456789"
            ) as response:
                assert response.status == 400
        assert offers == []


async def test_candidate_cap_precedes_aiortc_sdp_parsing():
    peer = PeerConnection(is_dialer=True, ice_servers=[], limits=nethernet.ConnectionLimits())
    apply_sdp = AsyncMock()
    peer._pc.setRemoteDescription = apply_sdp
    sdp = "v=0\r\nm=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n" + (
        "a=candidate:1 1 udp 1 127.0.0.1 1234 typ host\r\n" * 65
    )
    try:
        with pytest.raises(ValueError, match="candidate"):
            await peer.set_remote_answer(sdp)
        assert not apply_sdp.called
    finally:
        await peer.close()


async def test_total_connect_deadline_includes_offer_gathering_and_joins_close(monkeypatch):
    peers = []

    async def offer(peer):
        peers.append(peer)
        await asyncio.Event().wait()

    monkeypatch.setattr(PeerConnection, "create_offer", offer)
    with pytest.raises(TimeoutError):
        await nethernet.connect_http(
            "http://127.0.0.1:1", ice_servers=[], limits=nethernet.ConnectionLimits(), timeout=0.01,
        )
    assert len(peers) == 1
    assert peers[0].connection_state == "closed"
    assert peers[0]._close_task.done()


async def test_cancelled_partial_http_connect_joins_client_and_server_work(monkeypatch):
    entered = asyncio.Event()
    finished = asyncio.Event()
    peers = []
    create_offer = PeerConnection.create_offer

    async def offer(peer):
        peers.append(peer)
        return await create_offer(peer)

    async def validate(offer):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    monkeypatch.setattr(PeerConnection, "create_offer", offer)
    server = nethernet.serve_http(
        lambda conn: None, host="127.0.0.1", ice_servers=[],
        limits=nethernet.ConnectionLimits(), validate_offer=validate,
    )
    await server.start()
    dialing = asyncio.create_task(nethernet.connect_http(
        f"http://127.0.0.1:{server.bound_port}", ice_servers=[],
        limits=nethernet.ConnectionLimits(), timeout=3,
    ))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        dialing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dialing
        assert peers[0].connection_state == "closed"
    finally:
        dialing.cancel()
        await asyncio.gather(dialing, return_exceptions=True)
        await server.aclose()
    assert finished.is_set()
    assert not server._signaling._requests


async def test_oversized_signed_answer_joins_actual_peer_teardown_before_http_error(monkeypatch):
    sessions = []
    entered = asyncio.Event()
    release = asyncio.Event()
    create_session = api._make_http_session

    def make_session(**kwargs):
        conn = create_session(**kwargs)
        if not kwargs["is_dialer"]:
            sessions.append(conn)
            close = conn._session._pc.close

            async def joined_close():
                await close()
                entered.set()
                await release.wait()

            conn._session._pc.close = joined_close
        return conn

    monkeypatch.setattr(api, "_make_http_session", make_session)
    signer = nethernet.IdentitySigner(
        nethernet.generate_operator_key(), domain="test.example", claims={"padding": "x" * 4096}
    )
    limits = nethernet.ConnectionLimits(http_sdp_bytes=2048)

    async def handler(conn):
        await conn.wait_closed()

    async with nethernet.serve_http(
        handler, host="127.0.0.1", ice_servers=[], limits=limits, identity_signer=signer
    ) as server:
        dialing = asyncio.create_task(nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=5,
        ))
        closing = asyncio.create_task(entered.wait())
        try:
            await asyncio.wait([dialing, closing], timeout=2, return_when=asyncio.FIRST_COMPLETED)
            assert closing.done(), "HTTP error completed before server teardown began"
            assert not dialing.done(), "HTTP error completed before owned teardown joined"
            release.set()
            with pytest.raises(nethernet.ConnectionFailed):
                await asyncio.wait_for(dialing, 2)
            assert len(sessions) == 1
            conn = sessions[0]
            assert conn._session._pc.connection_state == "closed"
            assert conn._session._close_task.done()
            assert conn._close_task.done()
        finally:
            release.set()
            closing.cancel()
            dialing.cancel()
            await asyncio.gather(closing, dialing, return_exceptions=True)
