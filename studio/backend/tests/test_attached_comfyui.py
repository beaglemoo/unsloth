import asyncio
import json

import httpx
import pytest

from core.inference.attached import arbiter


@pytest.mark.parametrize("queue,expected", [
    ({"queue_running": [], "queue_pending": []}, ["GET", "POST"]),
    ({"queue_running": [1], "queue_pending": []}, ["GET"]),
    ({"queue_running": [], "queue_pending": [1]}, ["GET"]),
    ({}, ["GET"]),
    ([], ["GET"]),
])
def test_comfyui_free_only_when_queue_is_confirmed_empty(monkeypatch, queue, expected):
    monkeypatch.setenv("STUDIO_COMFYUI_URL", "http://mock")
    calls = []
    def handler(request):
        calls.append(request.method)
        assert request.extensions["timeout"]["read"] == 3
        if request.method == "GET":
            assert request.url.path == "/queue"
            return httpx.Response(200, json = queue)
        assert request.url.path == "/free"
        assert json.loads(request.content) == {"unload_models": True, "free_memory": True}
        return httpx.Response(200)
    asyncio.run(arbiter.free_comfyui(transport = httpx.MockTransport(handler)))
    assert calls == expected


@pytest.mark.parametrize("failure", ["connect", "timeout", "http", "json"])
def test_comfyui_failure_never_blocks(monkeypatch, failure):
    monkeypatch.setenv("STUDIO_COMFYUI_URL", "http://mock")
    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("offline")
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout")
        return httpx.Response(500) if failure == "http" else httpx.Response(200, content = b"bad")
    asyncio.run(arbiter.free_comfyui(transport = httpx.MockTransport(handler)))


def test_comfyui_disabled_makes_no_requests(monkeypatch):
    monkeypatch.setenv("STUDIO_COMFYUI_URL", "")
    def handler(request):
        pytest.fail("disabled ComfyUI must not be contacted")
    asyncio.run(arbiter.free_comfyui(transport = httpx.MockTransport(handler)))
