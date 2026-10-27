# SPDX-License-Identifier: AGPL-3.0-only
"""Testing a connection that reaches the attached oMLX is arbitrated like any other request to it.

Responses, speech, chat and System One probes are inference: they can make oMLX load a model, so they are admitted
and held. A ``/models`` listing sends none and stays open.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from models.providers import ProviderTestRequest
from routes import providers
from utils.attached_engines_settings import DEFAULT_CONFIG

OMLX = "http://127.0.0.1:8843/v1"
ELSEWHERE = "http://127.0.0.1:9999/v1"


class FakeClient:
    """Records every request and what the arbiter held at that moment."""

    models = [{"id": "swift"}]
    models_error: Exception | None = None
    speech_error: Exception | None = RuntimeError("no /audio/speech")
    log: list

    def __init__(self, **kwargs):
        pass

    def _note(self, name):
        FakeClient.log.append((name, arbiter.omlx_holds()))

    async def list_models(self):
        self._note("models")
        if FakeClient.models_error:
            raise FakeClient.models_error
        return FakeClient.models

    async def list_decision_models(self):
        self._note("decision_models")
        return FakeClient.models

    async def create_speech(self, **kwargs):
        self._note("speech")
        if FakeClient.speech_error:
            raise FakeClient.speech_error

    async def chat_completion(self, **kwargs):
        self._note("chat")
        return {}

    async def create_decision(self, *args, **kwargs):
        self._note("decision")
        return {"answers": {"outage": {"type": "noul", "noul": 0.5}}}

    async def stream_chat_completion(self, **kwargs):
        self._note("responses")
        yield 'data: {"choices": [{"delta": {}}]}'
        yield "data: [DONE]"

    async def close(self):
        pass


@pytest.fixture
def studio(monkeypatch):
    state = {"training": False}
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(arbiter, "_training_active", lambda: state["training"])
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    monkeypatch.setattr(arbiter, "_comfyui_refusal", _no_refusal)
    FakeClient.log = []
    FakeClient.models = [{"id": "swift"}]
    FakeClient.models_error = None
    FakeClient.speech_error = RuntimeError("no /audio/speech")
    monkeypatch.setattr(providers, "ExternalProviderClient", FakeClient)
    monkeypatch.setattr(providers, "resolve_provider_api_key_or_400", lambda *a, **k: "")
    monkeypatch.setattr(providers, "validate_provider_base_url", lambda url: url)
    arbiter._omlx_holds.clear()
    yield state
    arbiter._omlx_holds.clear()


async def _no_refusal(*args, **kwargs):
    return None


def test_connection(**fields):
    payload = ProviderTestRequest(provider_type = "custom", base_url = OMLX, model_id = "swift", **fields)
    return asyncio.run(providers.test_provider(payload, "user", False))


test_connection.__test__ = False  # a helper, not a test


def names():
    return [name for name, _ in FakeClient.log]


def test_a_responses_probe_to_the_omlx_origin_is_held_for_its_whole_run(studio):
    result = test_connection(api_type = "responses")
    assert result.success, result.message
    assert FakeClient.log == [("models", 0), ("responses", 1)]
    assert arbiter.omlx_holds() == 0


def test_a_responses_probe_is_refused_while_studio_trains_and_sends_nothing(studio):
    studio["training"] = True
    result = test_connection(api_type = "responses")
    assert not result.success
    assert "Studio is training" in result.message
    assert "responses" not in names()
    assert arbiter.omlx_holds() == 0


def test_a_listing_only_test_sends_no_inference_and_needs_no_admission(studio):
    studio["training"] = True
    result = test_connection()
    assert result.success
    assert names() == ["models"]


def test_the_speech_and_chat_fallbacks_share_one_admission_and_one_hold(studio):
    FakeClient.models_error = RuntimeError("no /models")
    result = test_connection()
    assert result.success, result.message
    assert FakeClient.log == [("models", 0), ("speech", 1), ("chat", 1)]
    assert arbiter.omlx_holds() == 0


def test_the_fallbacks_are_refused_while_studio_trains(studio):
    studio["training"] = True
    FakeClient.models_error = RuntimeError("no /models")
    result = test_connection()
    assert not result.success and "Studio is training" in result.message
    assert names() == ["models"]


def test_a_system_one_probe_is_admitted_and_held(studio):
    result = test_connection(api_type = "systemone")
    assert result.success, result.message
    assert FakeClient.log == [("decision", 1)]
    assert arbiter.omlx_holds() == 0


def test_a_system_one_probe_is_refused_while_studio_trains(studio):
    studio["training"] = True
    result = test_connection(api_type = "systemone")
    assert not result.success and "Studio is training" in result.message
    assert names() == []


def test_a_hold_is_released_when_a_probe_fails(studio):
    FakeClient.models_error = RuntimeError("no /models")

    async def boom(**kwargs):
        raise RuntimeError("chat failed")

    FakeClient.chat_completion = lambda self, **kwargs: boom()
    result = test_connection()
    assert not result.success
    assert arbiter.omlx_holds() == 0


def test_another_destination_is_not_arbitrated(studio):
    studio["training"] = True
    payload = ProviderTestRequest(
        provider_type = "custom", base_url = ELSEWHERE, model_id = "swift", api_type = "responses"
    )
    result = asyncio.run(providers.test_provider(payload, "user", False))
    assert result.success
    assert FakeClient.log == [("models", 0), ("responses", 0)]


def test_a_concurrent_local_admission_sees_the_probe_hold(studio, monkeypatch):
    """While the Responses probe runs, a local load is refused with oMLX's busy message."""
    seen = []
    real = FakeClient.stream_chat_completion

    async def probing(self, **kwargs):
        result = await arbiter.free_for_local("local_load")
        seen.append(result.error)
        async for line in real(self, **kwargs):
            yield line

    monkeypatch.setattr(FakeClient, "stream_chat_completion", probing)
    assert test_connection(api_type = "responses").success
    assert seen and "busy" in seen[0].lower()
