import asyncio
from dataclasses import replace

import pytest

import nethernet
from nethernet.api import Connection
from nethernet.session import SessionState
from nethernet.transport.framing import PacketQueue
from nethernet.transport.peer_connection import PeerConnection


class ClosingSession:
    state = SessionState.CONNECTED

    def __init__(self):
        self.conn = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.complete = False

    def close(self, error=nethernet.SessionError.NONE):
        self.state = SessionState.CLOSED
        self.conn._mark_closed(error)

    async def aclose(self, error=nethernet.SessionError.NONE):
        self.close(error)
        self.entered.set()
        await self.release.wait()
        self.complete = True


def connection(**limits):
    session = ClosingSession()
    conn = Connection(session, None, limits=nethernet.ConnectionLimits(**limits))
    session.conn = conn
    return conn, session


async def test_cancelled_close_and_wait_closed_join_the_same_teardown():
    conn, session = connection()
    first = asyncio.create_task(conn.close())
    await session.entered.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    waiting = asyncio.create_task(conn.wait_closed())
    await asyncio.sleep(0)
    assert not waiting.done()
    session.release.set()
    assert await asyncio.wait_for(waiting, 1) == nethernet.SessionError.NONE
    assert session.complete


async def test_completed_receive_byte_cap_preserves_already_queued_messages():
    conn, session = connection(receive_bytes=4)
    conn._enqueue(b"1234")
    conn._enqueue(b"5")
    assert await conn.recv() == b"1234"
    with pytest.raises(nethernet.ConnectionClosed) as caught:
        await conn.recv()
    assert caught.value.error != nethernet.SessionError.NONE
    session.release.set()
    await conn.wait_closed()


async def test_zero_payload_receive_messages_cannot_bypass_entry_cap():
    conn, session = connection(receive_messages=2)
    for _ in range(3):
        conn._enqueue(b"")
    assert await conn.recv() == b""
    assert await conn.recv() == b""
    with pytest.raises(nethernet.ConnectionClosed):
        await conn.recv()
    session.release.set()
    await conn.wait_closed()


async def test_receive_credit_is_released_on_dequeue():
    conn, session = connection(receive_bytes=4, receive_messages=1)
    for payload in (b"1234", b"", b"5678"):
        conn._enqueue(payload)
        assert await conn.recv() == payload
    session.release.set()
    await conn.close()


@pytest.mark.parametrize("second", [b"\x02b", b"\x00b", b""])
async def test_reliable_countdown_rejects_missing_or_invalid_fragments(second):
    queue = PacketQueue(lambda data: None, lambda data: None, limits=nethernet.ConnectionLimits())
    try:
        queue.on_reliable_message(b"\x02a")
        with pytest.raises(ValueError):
            queue.on_reliable_message(second)
        assert queue.read() is None
    finally:
        queue.close()


async def test_incomplete_message_byte_cap_is_checked_before_append():
    queue = PacketQueue(
        lambda data: None, lambda data: None,
        limits=nethernet.ConnectionLimits(max_message_size=4),
    )
    try:
        queue.on_reliable_message(b"\x01abcd")
        with pytest.raises(ValueError):
            queue.on_reliable_message(b"\x00e")
        assert queue.read() is None
    finally:
        queue.close()


async def test_incomplete_deadline_fails_without_a_followup_fragment():
    expired = asyncio.Event()
    queue = PacketQueue(
        lambda data: None, lambda data: None,
        limits=nethernet.ConnectionLimits(incomplete_timeout=0.01),
        on_error=lambda error: expired.set(),
    )
    try:
        queue.on_reliable_message(b"\x01abc")
        await asyncio.wait_for(expired.wait(), 1)
        assert queue.read() is None
    finally:
        queue.close()


async def test_bounded_framing_preserves_header_only_empty_application_payload():
    queue = PacketQueue(lambda data: None, lambda data: None, limits=nethernet.ConnectionLimits())
    queue.on_reliable_message(b"\x00")
    assert queue.read() == b""
    queue.close()


async def test_direct_configuration_preserves_explicit_empty_ice_servers():
    peer = PeerConnection(is_dialer=True, ice_servers=[], limits=nethernet.ConnectionLimits())
    try:
        ice = peer._pc.sctp.transport.transport._connection
        assert ice.stun_server is None
        assert ice.turn_server is None
        ice.data_received(b"" , 1)
        assert ice._queue.qsize() == 1
    finally:
        await peer.close()


async def test_bounded_peer_rejects_implicit_stun_and_relay_before_creation():
    with pytest.raises(ValueError):
        PeerConnection(is_dialer=True, limits=nethernet.ConnectionLimits())
    with pytest.raises(ValueError):
        PeerConnection(is_dialer=True, ice_servers=[], relay_only=True, limits=nethernet.ConnectionLimits())


async def test_send_reports_oversize_and_unsupported_reliability_without_partial_send():
    received = []

    async def handler(conn):
        async for message in conn:
            received.append(message)

    limits = replace(nethernet.ConnectionLimits(), max_message_size=4)
    async with nethernet.serve_http(handler, host="127.0.0.1", ice_servers=[], limits=limits) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3,
        )
        try:
            with pytest.raises(ValueError):
                await conn.send(b"12345")
            with pytest.raises(ValueError):
                await conn.send(b"x", nethernet.SendType.UNRELIABLE)
            await conn.send(b"ok")
            for _ in range(100):
                if received:
                    break
                await asyncio.sleep(0.01)
            assert received == [b"ok"]
        finally:
            await conn.close()


async def test_ordinary_send_refuses_full_storage_then_succeeds_after_ack_release():
    async def echo(conn):
        async for data in conn:
            await conn.send(data)

    limits = nethernet.ConnectionLimits(max_message_size=63, send_bytes=64)
    async with nethernet.serve_http(
        echo, host="127.0.0.1", ice_servers=[], limits=limits
    ) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        try:
            held = await conn.reserve_send(63)
            held.release()
            await conn.send(b"x" * 63)
            sending = asyncio.create_task(conn.send(b"later"))
            try:
                await asyncio.sleep(0)
                assert sending.done(), "ordinary send must not wait with an uncharged payload"
                with pytest.raises(BufferError):
                    await sending
            finally:
                if not sending.done():
                    sending.cancel()
                    await asyncio.gather(sending, return_exceptions=True)
            assert conn.state == SessionState.CONNECTED
            assert await asyncio.wait_for(conn.recv(), 2) == b"x" * 63
            held = await asyncio.wait_for(conn.reserve_send(2), 2)
            held.release()
            await conn.send(b"ok")
            assert await asyncio.wait_for(conn.recv(), 2) == b"ok"
        finally:
            await conn.close()
