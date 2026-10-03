import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.responses import JSONResponse, StreamingResponse

from core.inference.attached import arbiter
from routes import inference


def test_local_admission_failure_is_retryable(monkeypatch):
    monkeypatch.setattr(arbiter, "free_for_local", AsyncMock(return_value = arbiter.ArbiterResult(error = "oMLX is still loading")))
    with pytest.raises(HTTPException) as error:
        asyncio.run(inference._admit_attached_local_load())
    assert error.value.status_code == 503
    assert error.value.detail == "oMLX is still loading"
    assert error.value.headers == {"Retry-After": "15"}


@pytest.mark.parametrize("resident", [True, False])
def test_local_load_exit_retains_hold_for_residency_or_releases_failure(monkeypatch, resident):
    lease = SimpleNamespace(watch_until = AsyncMock(), release = AsyncMock())
    monkeypatch.setattr(inference, "_attached_local_memory_active", lambda: resident)
    asyncio.run(inference._finish_attached_local_load(lease))
    if resident:
        lease.watch_until.assert_awaited_once_with(inference._attached_local_memory_active)
        lease.release.assert_not_awaited()
    else:
        lease.release.assert_awaited_once()


@pytest.mark.parametrize("kind", ["active", "loading", "managed", "llama"])
def test_residency_covers_loading_and_managed_workers(monkeypatch, kind):
    from core.inference import llama_cpp
    monkeypatch.setattr(llama_cpp, "chat_load_active", lambda: False)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: SimpleNamespace(is_active = kind == "llama"))
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: SimpleNamespace(
        active_model_name = "m" if kind == "active" else None,
        loading_models = {"m"} if kind == "loading" else set(),
        _managed_engine = object() if kind == "managed" else None))
    assert inference._attached_local_memory_active()


@pytest.mark.parametrize("failure", [False, True])
def test_omlx_nonstream_or_failure_releases_hold(monkeypatch, failure):
    lease = SimpleNamespace(release = AsyncMock())
    monkeypatch.setattr(arbiter, "before_omlx_use", AsyncMock(return_value = arbiter.ArbiterResult(lease = lease)))
    proxy = AsyncMock(side_effect = RuntimeError("failed")) if failure else AsyncMock(return_value = JSONResponse({}))
    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    if failure:
        with pytest.raises(RuntimeError):
            asyncio.run(inference._attached_omlx_proxy(None, None, None))
    else:
        asyncio.run(inference._attached_omlx_proxy(None, None, None))
    lease.release.assert_awaited_once()


@pytest.mark.parametrize("failure", [False, True])
def test_stream_hold_covers_asgi_lifetime_even_before_first_chunk(monkeypatch, failure):
    lease = SimpleNamespace(release = AsyncMock())
    monkeypatch.setattr(arbiter, "before_omlx_use", AsyncMock(return_value = arbiter.ArbiterResult(lease = lease)))
    async def body():
        assert not lease.release.await_count
        yield b"hello"
    monkeypatch.setattr(inference, "_proxy_to_external_provider", AsyncMock(return_value = StreamingResponse(body())))
    async def serve(self, *args):
        assert not lease.release.await_count
        if failure:
            raise asyncio.CancelledError()
        async for _ in self.body_iterator:
            assert not lease.release.await_count
    monkeypatch.setattr(StreamingResponse, "__call__", serve)
    async def exercise():
        response = await inference._attached_omlx_proxy(None, None, None)
        assert not lease.release.await_count
        if failure:
            with pytest.raises(asyncio.CancelledError):
                await response({}, None, None)
        else:
            await response({}, None, None)
    asyncio.run(exercise())
    lease.release.assert_awaited_once()
