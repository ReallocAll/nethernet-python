import asyncio
from dataclasses import FrozenInstanceError, replace

import pytest
from aiortc.stats import RTCStatsReport

import nethernet
from nethernet.session import Session


def make_connection(*, is_dialer):
    session = Session(
        connection_id=1,
        local_id=nethernet.NetworkID.p2p(1),
        remote_id=nethernet.NetworkID.p2p(2),
        is_dialer=is_dialer,
        send_signal=lambda message: None,
        ice_servers=[],
    )
    return nethernet.Connection(session, None)


def assert_matches_transport(conn, snapshot):
    dtls = conn._session._pc._pc.sctp.transport
    stats, = dtls._get_stats().values()
    assert (snapshot.bytes_sent, snapshot.bytes_received) == (stats.bytesSent, stats.bytesReceived)


@pytest.mark.parametrize("bounded", [False, True])
async def test_real_dtls_counts_include_handshake_exchange_and_survive_close(bounded):
    incoming = asyncio.Queue()

    async def echo(conn):
        incoming.put_nowait(conn)
        async for data in conn:
            await conn.send(data)

    limits = nethernet.ConnectionLimits() if bounded else None
    async with nethernet.serve_http(
        echo, host="127.0.0.1", ice_servers=[], limits=limits
    ) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[], limits=limits, timeout=3
        )
        remote = await asyncio.wait_for(incoming.get(), 2)
        try:
            handshake = conn.get_dtls_byte_counts()
            assert isinstance(handshake, nethernet.DtlsByteCounts)
            assert handshake.bytes_sent > 0
            assert handshake.bytes_received > 0
            assert not hasattr(handshake, "__dict__")
            with pytest.raises(FrozenInstanceError):
                handshake.bytes_sent = 0
            with pytest.raises(FrozenInstanceError):
                handshake.bytes_received = 0
            assert_matches_transport(conn, handshake)
            original = (handshake.bytes_sent, handshake.bytes_received)

            previous = handshake
            for payload in (b"first application message", b"second" * 1000):
                await conn.send(payload)
                assert await asyncio.wait_for(conn.recv(), 2) == payload
                current = conn.get_dtls_byte_counts()
                assert current.bytes_sent > previous.bytes_sent
                assert current.bytes_received > previous.bytes_received
                assert_matches_transport(conn, current)
                previous = current
            assert (handshake.bytes_sent, handshake.bytes_received) == original
            assert_matches_transport(remote, remote.get_dtls_byte_counts())

            await conn.close()
            await remote.close()
            final = conn.get_dtls_byte_counts()
            assert final.bytes_sent >= previous.bytes_sent
            assert final.bytes_received >= previous.bytes_received
            assert_matches_transport(conn, final)
            await asyncio.sleep(0)
            assert conn.get_dtls_byte_counts() == final
            assert (handshake.bytes_sent, handshake.bytes_received) == original
        finally:
            await conn.close()
            await remote.close()


@pytest.mark.parametrize("is_dialer", [False, True])
async def test_new_connection_counts_are_zero_before_dtls_traffic(is_dialer):
    conn = make_connection(is_dialer=is_dialer)
    try:
        if not is_dialer:
            assert conn._session._pc._pc.sctp is None
        assert conn.get_dtls_byte_counts() == nethernet.DtlsByteCounts(0, 0)
    finally:
        await conn.close()


async def test_dtls_counts_reject_reads_outside_the_connection_loop():
    conn = make_connection(is_dialer=True)

    async def read_from_another_loop():
        with pytest.raises(RuntimeError):
            conn.get_dtls_byte_counts()

    def read_without_a_loop():
        with pytest.raises(RuntimeError):
            conn.get_dtls_byte_counts()

    try:
        await asyncio.to_thread(lambda: asyncio.run(read_from_another_loop()))
        await asyncio.to_thread(read_without_a_loop)
    finally:
        await conn.close()


@pytest.mark.parametrize(
    "invalid", ["missing", "unexpected", "ambiguous", "negative", "non_integer"]
)
async def test_existing_dtls_transport_rejects_invalid_statistics(monkeypatch, invalid):
    conn = make_connection(is_dialer=True)
    dtls = conn._session._pc._pc.sctp.transport
    stats, = dtls._get_stats().values()
    reports = {
        "missing": RTCStatsReport(),
        "unexpected": {stats.id: object()},
        "ambiguous": {stats.id: stats, "other": replace(stats, id="other")},
        "negative": {stats.id: replace(stats, bytesSent=-1)},
        "non_integer": {stats.id: replace(stats, bytesReceived="0")},
    }
    monkeypatch.setattr(dtls, "_get_stats", lambda: reports[invalid])
    try:
        with pytest.raises(RuntimeError, match="DTLS.*stat"):
            conn.get_dtls_byte_counts()
    finally:
        await conn.close()
