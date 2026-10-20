"""HTTP signaling channel — NetherNet Onboarding Guide s4.

The partner flow: the dialer POSTs a full-ICE SDP offer to ``/v1/join/{networkId}`` and
receives the full-ICE answer in the response body; ``GET /v1/join`` is the capability check.
Requires the ``http`` extra (aiohttp).

aiohttp API verified against the official reference:
https://docs.aiohttp.org/en/stable/web_reference.html
https://docs.aiohttp.org/en/stable/client_reference.html
"""

from __future__ import annotations

import asyncio
import ssl
from collections.abc import Awaitable, Callable

try:
    import aiohttp
    from aiohttp import web
except ImportError:  # pragma: no cover - exercised only without the http extra
    aiohttp = None
    web = None

from nethernet.errors import ConnectionFailed, NetherNetError, SessionError
from nethernet.limits import ConnectionLimits

SDP_CONTENT_TYPE = "application/sdp"

# answer_offer(network_id, offer_sdp) -> answer_sdp; network_id is the raw URL path segment.
AnswerOffer = Callable[[str, str], Awaitable[str]]


class SignalingRejected(NetherNetError):  # noqa: N818 - matches ConnectionFailed et al.
    """Reject an offer from an ``answer_offer`` / ``validate_offer`` callback (HTTP 403)."""

    def __init__(self, reason: str = "rejected", status: int = 403) -> None:
        self.status = status
        super().__init__(reason)


def _require_aiohttp() -> None:
    if aiohttp is None:
        raise NetherNetError("HTTP signaling requires aiohttp — install nethernet[http]")


def _parse_answer(body: str) -> str:
    """Return the answer SDP, or raise :class:`ConnectionFailed` if the body is an error code.

    A rejecting host answers 200 ``application/sdp`` with a bare decimal ``ESessionError``
    rather than an SDP body — BDS returns ``37`` (``IDENTITY_NOT_ALLOWED``) for an offer with
    no ``a=identity`` assertion. An SDP answer always opens with the ``v=`` version line.
    """
    text = body.strip()
    if text.startswith("v="):
        return body
    if text.isdigit():
        try:
            error = SessionError(int(text))
        except ValueError:
            error = SessionError.SIGNALING_UNKNOWN_ERROR
        raise ConnectionFailed(error)
    raise ConnectionFailed(SessionError.SIGNALING_PARSING_FAILURE)


async def request_answer(
    server_url: str,
    network_id: str,
    offer_sdp: str,
    *,
    capability_check: bool = True,
    timeout: float = 10.0,
    limits: ConnectionLimits | None = None,
) -> str:
    """POST the offer to ``{server_url}/v1/join/{network_id}`` and return the answer SDP.

    Exactly one POST per attempt, no retries (Guide s4). Raises :class:`ConnectionFailed` with
    ``NO_SIGNALING_CHANNEL`` if the capability check fails, ``SIGNALING_FAILED_TO_SEND`` on
    a non-2xx or failed POST, and the host's own :class:`SessionError` when the 2xx body is an
    error code instead of an answer.
    """
    _require_aiohttp()
    if limits is not None:
        _check_sdp_size(offer_sdp, limits)
    base = server_url.rstrip("/")
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(
        timeout=client_timeout, auto_decompress=limits is None
    ) as http:
        if capability_check:
            try:
                async with http.get(f"{base}/v1/join") as response:
                    supported = response.status // 100 == 2
            except (aiohttp.ClientError, OSError):
                supported = False
            if not supported:
                raise ConnectionFailed(SessionError.NO_SIGNALING_CHANNEL)
        try:
            async with http.post(
                f"{base}/v1/join/{network_id}",
                data=offer_sdp.encode(),
                headers={"Content-Type": SDP_CONTENT_TYPE},
            ) as response:
                if response.status // 100 != 2:
                    raise ConnectionFailed(SessionError.SIGNALING_FAILED_TO_SEND)
                if limits is not None:
                    if response.headers.get("Content-Encoding", "identity") != "identity":
                        raise ConnectionFailed(SessionError.SIGNALING_PARSING_FAILURE)
                    body = await _read_sdp(response.content, limits.http_sdp_bytes)
                    return _parse_answer(body)
                return _parse_answer(await response.text())
        except (aiohttp.ClientError, OSError):
            raise ConnectionFailed(SessionError.SIGNALING_FAILED_TO_SEND) from None


def _check_sdp_size(sdp: str, limits: ConnectionLimits) -> None:
    if len(sdp) > limits.http_sdp_bytes or len(sdp.encode()) > limits.http_sdp_bytes:
        raise ConnectionFailed(SessionError.SIGNALING_PARSING_FAILURE)


async def _read_sdp(content, limit: int) -> str:
    data = bytearray()
    while True:
        chunk = await content.read(min(65536, limit + 1 - len(data)))
        if not chunk:
            break
        if len(data) + len(chunk) > limit:
            raise ConnectionFailed(SessionError.SIGNALING_PARSING_FAILURE)
        data.extend(chunk)
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        raise ConnectionFailed(SessionError.SIGNALING_PARSING_FAILURE) from None


class HttpSignalingServer:
    """Serves the two signaling endpoints; answers come from an ``answer_offer`` callback.

    ``GET /v1/join`` returns 2xx (capability check). ``POST /v1/join/{network_id}`` passes the
    body to the callback and returns its answer as ``application/sdp``; a raised
    :class:`SignalingRejected` maps to its HTTP status, any other :class:`NetherNetError`
    (e.g. a malformed identity envelope) to 400.
    """

    def __init__(
        self,
        answer_offer: AnswerOffer,
        *,
        host: str = "0.0.0.0",
        port: int = 0,
        ssl_context: ssl.SSLContext | None = None,
        limits: ConnectionLimits | None = None,
    ) -> None:
        _require_aiohttp()
        self._answer_offer = answer_offer
        self._host = host
        self._port = port
        self._ssl_context = ssl_context
        self._runner: web.AppRunner | None = None
        self.bound_port: int | None = None
        self._limits = limits
        self._site: web.TCPSite | None = None
        self._requests: set[asyncio.Task] = set()
        self._closing = False
        self._close_task: asyncio.Task | None = None

    async def start(self) -> None:
        app = web.Application(
            client_max_size=self._limits.http_sdp_bytes if self._limits is not None else 1024**2
        )
        app.router.add_get("/v1/join", self._handle_capability)
        app.router.add_post("/v1/join/{network_id}", self._handle_join)
        self._runner = web.AppRunner(app, auto_decompress=self._limits is None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port, ssl_context=self._ssl_context)
        self._site = site
        await site.start()
        addresses = self._runner.addresses
        self.bound_port = addresses[0][1] if addresses else None

    async def aclose(self) -> None:
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
            self._close_task.add_done_callback(self._close_done)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _close_done(task: asyncio.Task) -> None:
        if not task.cancelled():
            task.exception()

    async def _close(self) -> None:
        if self._site is not None:
            await self._site.stop()
            self._site = None
        tasks = list(self._requests)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_capability(self, request: web.Request) -> web.Response:
        return web.Response()

    async def _handle_join(self, request: web.Request) -> web.Response:
        if self._closing:
            return web.Response(status=503)
        task = asyncio.current_task()
        self._requests.add(task)
        try:
            if self._limits is not None:
                if request.headers.get("Content-Encoding", "identity") != "identity":
                    return web.Response(status=415)
                offer_sdp = await _read_sdp(request.content, self._limits.http_sdp_bytes)
            else:
                offer_sdp = await request.text()
            answer = await self._answer_offer(request.match_info["network_id"], offer_sdp)
            if self._limits is not None:
                _check_sdp_size(answer, self._limits)
        except SignalingRejected as exc:
            return web.Response(status=exc.status, text=str(exc))
        except NetherNetError as exc:
            return web.Response(status=400, text=str(exc))
        finally:
            self._requests.discard(task)
        return web.Response(text=answer, content_type=SDP_CONTENT_TYPE)
