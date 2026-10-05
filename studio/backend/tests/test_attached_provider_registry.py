# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The oMLX and legacy engine registry entries: hidden, loopback, and wired to the right body shape."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference import providers as providers_mod
from core.inference.external_provider import ExternalProviderClient

SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "min_p",
    "repetition_penalty",
)

SSE = (
    'data: {"choices":[{"index":0,"delta":{"reasoning_content":"thinking"}}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{"content":"answer"}}]}\n\n'
    "data: [DONE]\n\n"
)


def _stream(provider_type: str, base_url: str, model: str, **kwargs):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, content = SSE, headers = {"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(provider_type = provider_type, base_url = base_url, api_key = "")
    lines: list[str] = []

    async def run() -> None:
        try:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}], model = model, **kwargs
            ):
                lines.append(line)
        finally:
            await mock_client.aclose()

    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = mock_client
    try:
        loop.run_until_complete(run())
    finally:
        ep_mod._http_client = previous
        loop.close()
    return captured, lines


def test_registry_entries_are_hidden_loopback_and_not_managed():
    omlx = providers_mod.get_provider_info("omlx")
    assert omlx["display_name"] == "oMLX"
    assert omlx["base_url"] == "http://127.0.0.1:8843/v1"
    for info in (omlx,):
        assert info["hidden"] is True
        assert not info.get("managed")
        assert (
            info["supports_streaming"] and info["supports_vision"] and info["supports_tool_calling"]
        )
    assert omlx["supports_chat_template_kwargs"] is True
    assert providers_mod.get_connectable_provider_info("omlx") is omlx


def test_both_are_template_applying_and_request_usage():
    for provider_type in ("omlx",):
        assert provider_type in ep_mod._TEMPLATE_APPLYING_PROVIDERS
        assert provider_type in ep_mod._USAGE_STREAM_OPTION_PROVIDERS


def test_omlx_streams_reasoning_and_carries_sampling():
    captured, lines = _stream(
        "omlx",
        "http://127.0.0.1:8843/v1",
        "swift-1.5-27b:fast",
        temperature = 0.6,
        top_p = 0.9,
        enable_thinking = False,
    )
    assert captured["url"] == "http://127.0.0.1:8843/v1/chat/completions"
    body = captured["body"]
    assert body["model"] == "swift-1.5-27b:fast"
    assert body["temperature"] == 0.6 and body["top_p"] == 0.9
    assert body["stream_options"] == {"include_usage": True}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    joined = "\n".join(lines)
    assert '"reasoning_content": "thinking"' in joined or '"reasoning_content":"thinking"' in joined
    assert "answer" in joined






@pytest.mark.parametrize("provider_type", ["omlx", "legacy"])
def test_colon_and_slash_free_model_ids_pass_through_verbatim(provider_type):
    captured, _ = _stream(provider_type, "http://127.0.0.1:8843/v1", "swift-1.5-27b:medium")
    assert captured["body"]["model"] == "swift-1.5-27b:medium"
