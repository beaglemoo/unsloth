"""Admission and proxy behavior without peer-engine holds."""
import asyncio
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


def test_local_admission_success_needs_no_hold(monkeypatch):
    result = arbiter.ArbiterResult(acted = True, actions = ("Unloaded oMLX",))
    unload = AsyncMock(return_value = result)
    monkeypatch.setattr(arbiter, "free_for_local", unload)
    assert asyncio.run(inference._admit_attached_local_load()) == result
    unload.assert_awaited_once_with("local_load")


@pytest.mark.parametrize("stream", [False, True])
def test_omlx_proxy_returns_original_response(monkeypatch, stream):
    async def body():
        yield b"hello"
    response = StreamingResponse(body()) if stream else JSONResponse({})
    monkeypatch.setattr(arbiter, "before_omlx_use", AsyncMock(return_value = arbiter.ArbiterResult()))
    proxy = AsyncMock(return_value = response)
    monkeypatch.setattr(inference, "_proxy_to_external_provider", proxy)
    assert asyncio.run(inference._attached_omlx_proxy(None, None, None)) is response
    proxy.assert_awaited_once_with(None, None, None)
