# SPDX-License-Identifier: AGPL-3.0-only
"""Every external-provider route to the attached oMLX is arbitrated, once."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import ATTACHED_OMLX_ID, arbiter
from models.inference import ChatCompletionRequest
from routes import inference
from utils.attached_engines_settings import DEFAULT_CONFIG

PASSED = 418  # raised by the stubbed credential step: admission let the request through


@pytest.fixture
def studio(monkeypatch):
    """Studio state the arbiter reads, and a proxy that stops at the credential step."""
    state = {"training": False, "local": False}
    config = replace(DEFAULT_CONFIG, enabled = True)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(arbiter, "_training_active", lambda: state["training"])
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: state["local"])
    gate = AsyncMock(wraps = arbiter.before_omlx_use)
    monkeypatch.setattr(arbiter, "before_omlx_use", gate)

    def stop(*args, **kwargs):
        raise HTTPException(status_code = PASSED, detail = "past admission")

    monkeypatch.setattr(inference, "resolve_provider_api_key_or_400", stop)
    state["gate"] = gate
    state["saved"] = {}
    monkeypatch.setattr(inference.providers_db, "get_provider", lambda pid: state["saved"].get(pid))
    return state


def proxy(**fields):
    payload = ChatCompletionRequest(
        messages = [{"role": "user", "content": "hi"}], external_model = "swift", **fields
    )
    request = NS(scope = {}, state = NS(), headers = {}, url = NS(path = "/v1/chat/completions"))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(inference._proxy_to_external_provider(payload, request, "user"))
    return caught.value


def saved_connection(**fields):
    return {
        "id": "userconn", "display_name": "My oMLX", "provider_type": "custom",
        "base_url": "http://127.0.0.1:8843/v1", "is_enabled": True,
        "api_type": "chat_completions", "reasoning_config": None, **fields,
    }


@pytest.mark.parametrize("busy", ["training", "local"])
def test_direct_omlx_type_without_provider_id_is_refused_while_busy(studio, busy):
    studio[busy] = True
    error = proxy(provider_type = "omlx")
    assert error.status_code == 503
    assert error.headers == {"Retry-After": "15"}
    studio["gate"].assert_awaited_once()


def test_direct_omlx_type_passes_when_studio_is_idle(studio):
    assert proxy(provider_type = "omlx").status_code == PASSED
    studio["gate"].assert_awaited_once()


@pytest.mark.parametrize("busy", ["training", "local"])
def test_unsaved_custom_connection_to_the_omlx_origin_is_refused_while_busy(studio, busy):
    studio[busy] = True
    error = proxy(provider_type = "custom", provider_base_url = "http://localhost:8843/v1/")
    assert error.status_code == 503
    studio["gate"].assert_awaited_once()


@pytest.mark.parametrize("busy", ["training", "local"])
def test_saved_custom_connection_to_the_omlx_origin_is_refused_while_busy(studio, busy):
    studio[busy] = True
    studio["saved"]["userconn"] = saved_connection()
    error = proxy(provider_id = "userconn")
    assert error.status_code == 503
    studio["gate"].assert_awaited_once()


def test_saved_custom_connection_to_the_omlx_origin_passes_when_idle(studio):
    studio["saved"]["userconn"] = saved_connection()
    assert proxy(provider_id = "userconn").status_code == PASSED
    studio["gate"].assert_awaited_once()


def test_other_destinations_are_not_arbitrated(studio):
    studio["training"] = True
    studio["saved"]["userconn"] = saved_connection(base_url = "http://127.0.0.1:8080/v1")
    assert proxy(provider_id = "userconn").status_code == PASSED
    assert proxy(provider_type = "custom", provider_base_url = "http://192.168.2.5:8843/v1").status_code == PASSED
    studio["gate"].assert_not_awaited()


def test_attached_provider_is_admitted_exactly_once(studio):
    studio["saved"][ATTACHED_OMLX_ID] = saved_connection(id = ATTACHED_OMLX_ID, provider_type = "omlx")
    assert proxy(provider_id = ATTACHED_OMLX_ID).status_code == PASSED
    studio["gate"].assert_awaited_once()


def test_attached_provider_is_refused_while_training(studio):
    studio["training"] = True
    studio["saved"][ATTACHED_OMLX_ID] = saved_connection(id = ATTACHED_OMLX_ID, provider_type = "omlx")
    assert proxy(provider_id = ATTACHED_OMLX_ID).status_code == 503
    studio["gate"].assert_awaited_once()


@pytest.mark.parametrize("busy", ["training", "local"])
def test_speech_connection_to_the_omlx_origin_is_refused_while_busy(studio, busy):
    from models.inference import AudioSpeechRequest

    studio[busy] = True
    studio["saved"]["userconn"] = saved_connection()
    body = AudioSpeechRequest(input = "hi", model = "kokoro", voice = "af", provider_id = "userconn")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(inference._external_tts_speech(body, NS(headers = {}, state = NS(), scope = {})))
    assert caught.value.status_code == 503
    studio["gate"].assert_awaited_once()


@pytest.mark.parametrize("provider_type,base_url,expected", [
    ("omlx", None, True),
    ("custom", "http://127.0.0.1:8843/v1", True),
    ("custom", "http://localhost:8843", True),
    ("custom", "HTTP://LOCALHOST:8843/v1/", True),
    ("custom", "http://[::1]:8843/v1", True),
    ("custom", "http://127.0.0.1:8844/v1", False),
    ("custom", "https://127.0.0.1:8843/v1", False),
    ("custom", "http://192.168.2.5:8843/v1", False),
    ("custom", "not a url", False),
    ("custom", None, False),
])
def test_destination_match(studio, provider_type, base_url, expected):
    assert arbiter.targets_omlx(provider_type, base_url) is expected


def test_destination_follows_the_configured_omlx_url(studio, monkeypatch):
    config = replace(DEFAULT_CONFIG, enabled = True, omlx_url = "http://192.168.2.9:9000")
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    assert arbiter.targets_omlx("custom", "http://192.168.2.9:9000/v1")
    assert not arbiter.targets_omlx("custom", "http://127.0.0.1:8843/v1")


def test_disabled_attached_engines_never_block_the_provider(studio, monkeypatch):
    studio["training"] = True
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    assert proxy(provider_type = "omlx").status_code == PASSED
