import asyncio
import hashlib

import pytest

import nethernet


async def echo(conn):
    async for data in conn:
        await conn.send(data)


async def test_whole_message_reservation_cancellation_before_and_after_grant():
    limits = nethernet.ConnectionLimits(max_message_size=63, send_bytes=64)
    async with nethernet.serve_http(echo, host="127.0.0.1", ice_servers=[], limits=limits) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        try:
            for after_grant in (False, True):
                held = await asyncio.wait_for(conn.reserve_send(63), 2)
                waiting = asyncio.create_task(conn.reserve_send(63))
                await asyncio.sleep(0)
                assert not waiting.done()
                if after_grant:
                    held.release()
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
                held.release()
                held.release()
                permit = await asyncio.wait_for(conn.reserve_send(63), 2)
                permit.send_nowait(b"refunded")
                permit.release()
                assert await asyncio.wait_for(conn.recv(), 2) == b"refunded"
        finally:
            await conn.close()


async def test_failure_after_possible_send_side_effect_closes_and_releases_credit():
    limits = nethernet.ConnectionLimits()
    async with nethernet.serve_http(echo, host="127.0.0.1", ice_servers=[], limits=limits) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        permit = await conn.reserve_send(16)
        commit = permit._reservation.send_nowait
        failure = OSError("injected failure after enqueue")

        def send_then_fail(messages):
            commit(messages)
            raise failure

        permit._reservation.send_nowait = send_then_fail
        with pytest.raises(OSError) as caught:
            permit.send_nowait(b"possibly sent")
        assert caught.value is failure
        permit.release()
        assert await asyncio.wait_for(conn.wait_closed(), 3) != nethernet.SessionError.NONE
        sctp = conn._session._pc._pc.sctp
        assert (sctp._send_bytes, sctp._send_chunks, sctp._send_messages) == (0, 0, 0)
        with pytest.raises(nethernet.ConnectionClosed):
            await conn.send(b"no retry")


async def test_slow_consumer_overflow_is_terminal_and_preserves_complete_queue():
    limits = nethernet.ConnectionLimits(receive_messages=1)

    async def send_three(conn):
        for data in (b"first", b"second", b"third"):
            await conn.send(data)
        await conn.wait_closed()

    async with nethernet.serve_http(send_three, host="127.0.0.1", ice_servers=[], limits=limits) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        try:
            assert await asyncio.wait_for(conn.wait_closed(), 3) != nethernet.SessionError.NONE
            assert await conn.recv() == b"first"
            with pytest.raises(nethernet.ConnectionClosed):
                await conn.recv()
        finally:
            await conn.close()


async def test_complete_maximum_wire_message_is_legal_over_actual_http_udp():
    limits = nethernet.ConnectionLimits()
    payload = b"m" * 33555456
    expected = hashlib.sha256(payload).digest()
    peers = []

    async def digest(conn):
        peers.append(conn)
        data = await conn.recv()
        assert len(data) == 33555456
        await conn.send(hashlib.sha256(data).digest())
        await conn.wait_closed()

    async with nethernet.serve_http(digest, host="127.0.0.1", ice_servers=[], limits=limits) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        try:
            permit = await conn.reserve_send(33555456)
            permit.send_nowait(payload)
            permit.release()
            try:
                assert await asyncio.wait_for(conn.recv(), 30) == expected
            except Exception as error:
                for peer in [conn, *peers]:
                    sctp = peer._session._pc._pc.sctp
                    ice = sctp.transport.transport._connection
                    error.add_note(
                        f"SCTP failure={sctp._failure!r}; ICE failure={ice._failure!r}; "
                        f"framing failure={peer._session._pc._packets._failure!r}"
                    )
                raise
        finally:
            await conn.close()
