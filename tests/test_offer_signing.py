import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import nethernet
import nethernet.api as api
from nethernet.identity import (
    build_envelope,
    insert_identity,
    sign_fingerprints,
    verify_fingerprints,
)
from nethernet.transport.peer_connection import PeerConnection


@pytest.mark.parametrize("provider", ["sign_offer", "identity_signer"])
async def test_offer_signer_receives_gathered_sdp_and_inserts_one_assertion(provider):
    client_key = nethernet.generate_operator_key()
    server_key = nethernet.generate_operator_key()
    server_signer = nethernet.IdentitySigner(server_key, domain="server.example")
    signed = []
    offers = []
    identities = []

    def sign(offer):
        assert "a=candidate:" in offer
        assert "a=identity:" not in offer
        signed.append(offer)
        return build_envelope(
            "offline.endstone.invalid", "test-token", sign_fingerprints(offer, client_key)
        )

    def full_sign(offer):
        assert "a=candidate:" in offer and "a=identity:" not in offer
        signed.append(offer)
        return nethernet.IdentitySigner(client_key, domain="offline.endstone.invalid").sign(offer)

    def validate(offer):
        assert offer.identity.domain == "offline.endstone.invalid"
        if provider == "sign_offer":
            assert offer.identity.token == "test-token"
        verify_fingerprints(offer.sdp, offer.identity.fingerprints, client_key.public_key())
        offers.append(offer)

    async def echo(conn):
        async for data in conn:
            await conn.send(data)

    async with nethernet.serve_http(
        echo, host="127.0.0.1", ice_servers=[], limits=nethernet.ConnectionLimits(),
        identity_signer=server_signer, validate_offer=validate,
    ) as server:
        conn = await nethernet.connect_http(
            f"http://127.0.0.1:{server.bound_port}", ice_servers=[],
            limits=nethernet.ConnectionLimits(), timeout=3,
            on_server_identity=identities.append,
            **(
                {"sign_offer": sign} if provider == "sign_offer"
                else {"identity_signer": SimpleNamespace(sign=full_sign)}
            ),
        )
        try:
            await conn.send(b"signed")
            assert await conn.recv() == b"signed"
        finally:
            await conn.close()
    assert len(signed) == len(offers) == len(identities) == 1


@pytest.mark.parametrize("kind", ["exception", "nonstring", "awaitable", "preexisting", "oversize"])
async def test_invalid_signer_result_or_existing_assertion_never_posts_and_closes(
    monkeypatch, kind,
):
    peers = []
    make_offer = PeerConnection.create_offer
    posting = AsyncMock()
    monkeypatch.setattr(api, "request_answer", posting)

    async def offer(peer):
        peers.append(peer)
        sdp = await make_offer(peer)
        if kind == "preexisting":
            sdp += "a=identity:existing\r\n"
        return sdp

    def sign(sdp):
        if kind == "exception":
            raise LookupError("signing failed")
        if kind == "nonstring":
            return 1
        if kind == "awaitable":
            return asyncio.sleep(0)
        if kind == "oversize":
            return "a" * (1024 * 1024)
        raise AssertionError("signer must not run for an existing assertion")

    monkeypatch.setattr(PeerConnection, "create_offer", offer)
    with pytest.raises((LookupError, TypeError, ValueError, nethernet.NetherNetError)):
        await nethernet.connect_http(
            "http://127.0.0.1:1", ice_servers=[], limits=nethernet.ConnectionLimits(),
            sign_offer=sign, timeout=3,
        )
    assert not posting.called
    assert peers[0].connection_state == "closed"
    assert peers[0]._close_task.done()


async def test_dual_signers_rejected_before_session_or_provider_work(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no session or signing work is allowed")

    monkeypatch.setattr(api, "_make_http_session", forbidden)
    with pytest.raises(ValueError, match="mutually exclusive"):
        await nethernet.connect_http(
            "http://127.0.0.1:1", sign_offer=forbidden,
            identity_signer=SimpleNamespace(sign=forbidden),
        )


@pytest.mark.parametrize("kind", [
    "missing", "duplicate", "media", "malformed", "modified", "oversize", "utf8_oversize",
    "nonstring", "awaitable", "exception", "preexisting",
])
async def test_invalid_full_sdp_signer_never_posts_and_joins_cleanup(monkeypatch, kind):
    peers = []
    create_offer = PeerConnection.create_offer
    posting = AsyncMock()
    monkeypatch.setattr(api, "request_answer", posting)
    signer = nethernet.IdentitySigner(nethernet.generate_operator_key(), domain="test.example")
    calls = []

    async def offer(peer):
        peers.append(peer)
        sdp = await create_offer(peer)
        return sdp + "a=identity:existing\r\n" if kind == "preexisting" else sdp

    def sign(sdp):
        calls.append(sdp)
        if kind == "exception":
            raise LookupError("signing failed")
        if kind == "nonstring":
            return 1
        if kind == "awaitable":
            return asyncio.sleep(0)
        if kind == "missing":
            return sdp
        if kind == "malformed":
            return insert_identity(sdp, "malformed")
        if kind == "oversize":
            return "x" * (1024 * 1024 + 1)
        if kind == "utf8_oversize":
            return "é" * (512 * 1024 + 1)
        signed = signer.sign(sdp)
        if kind == "duplicate":
            return signer.sign(signed)
        if kind == "media":
            return sdp + next(
                line for line in signed.splitlines(keepends=True) if line.startswith("a=identity:")
            )
        if kind == "modified":
            return signed.replace("s=-", "s=modified", 1)
        raise AssertionError("pre-existing identity must reject before signing")

    monkeypatch.setattr(PeerConnection, "create_offer", offer)
    with pytest.raises((LookupError, TypeError, ValueError, nethernet.NetherNetError)):
        await nethernet.connect_http(
            "http://127.0.0.1:1", ice_servers=[], limits=nethernet.ConnectionLimits(),
            identity_signer=SimpleNamespace(sign=sign), timeout=3,
        )
    assert len(calls) == (0 if kind == "preexisting" else 1)
    assert not posting.called
    assert peers[0].connection_state == "closed"
    assert peers[0]._close_task.done()
