# SPDX-License-Identifier: AGPL-3.0-only
"""RAG embeddings through a saved connection that targets the attached oMLX are admitted like any oMLX call.

``core/rag/embed_connection.py`` builds its own ``ExternalProviderClient`` and posts from a worker thread, so it never
met ``_admit_attached_omlx_provider``. oMLX loads its embedding model on first use: the request must be refused while
Studio holds memory, and must hold oMLX against local loads and ComfyUI jobs while it runs.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from core.rag import config, embed_connection, embeddings
from utils.attached_engines_settings import DEFAULT_CONFIG

_PROVIDERS = [
    {
        "id": "p-omlx",
        "provider_type": "custom",
        "display_name": "Local oMLX",
        "base_url": "http://127.0.0.1:8843/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
    {
        "id": "p-attached",
        "provider_type": "omlx",
        "display_name": "Attached",
        "base_url": "http://localhost:8843/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
    {
        "id": "p-local",
        "provider_type": "custom",
        "display_name": "Local other",
        "base_url": "http://127.0.0.1:9000/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
    {
        "id": "p-remote",
        "provider_type": "custom",
        "display_name": "Remote",
        "base_url": "http://scheduler.local:8843/v1",
        "is_enabled": True,
        "api_type": "chat_completions",
        "extra_headers": {},
    },
]


class World:
    def __init__(self):
        self.training = False
        self.local = False
        self.requests = []
        self.holds_seen = []


@pytest.fixture
def world(monkeypatch):
    state = World()
    from storage import credential_secrets, providers_db

    monkeypatch.setattr(providers_db, "list_providers", lambda: [dict(p) for p in _PROVIDERS])
    monkeypatch.setattr(credential_secrets, "get_provider_api_key", lambda pid: "")
    embeddings._reset_backend()
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(arbiter, "_training_active", lambda: state.training)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: state.local)
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: None)
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda config: (None, []))

    async def no_stt():
        return None

    monkeypatch.setattr(arbiter, "_stt_blocker", no_stt)
    arbiter._omlx_holds.clear()
    arbiter._media_holds.clear()
    arbiter._comfyui_holds.clear()
    arbiter._media_probe_future = None

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    arbiter.remember_loop(loop)
    yield state
    arbiter.remember_loop(None)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)
    loop.close()
    arbiter._omlx_holds.clear()
    embeddings._reset_backend()


def _backend(world):
    def handler(request: httpx.Request) -> httpx.Response:
        world.requests.append(request)
        world.holds_seen.append(arbiter.omlx_holds())
        body = json.loads(request.content)
        data = [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json = {"data": data})

    backend = embed_connection.ConnectionEmbeddingBackend()
    backend._http = httpx.Client(transport = httpx.MockTransport(handler))
    return backend


@pytest.mark.parametrize("name", ["Local oMLX", "Attached"])
def test_an_omlx_connection_holds_a_claim_while_the_request_runs_and_releases_it(world, name):
    backend = _backend(world)
    out = backend.encode(["a", "b"], model_name = f"connection:{name}/Qwen3-Embedding")
    assert out.shape == (2, 2)
    assert world.holds_seen == [1]
    assert arbiter.omlx_holds() == 0


def test_every_batch_is_admitted_and_released(world, monkeypatch):
    monkeypatch.setattr(config, "EMBED_BATCH", 1)
    backend = _backend(world)
    backend.encode(["a", "b", "c"], model_name = "connection:Local oMLX/Qwen3-Embedding")
    assert world.holds_seen == [1, 1, 1] and arbiter.omlx_holds() == 0


@pytest.mark.parametrize("busy", ["training", "local"])
def test_a_refusal_raises_at_once_and_sends_nothing(world, busy, monkeypatch):
    sleeps = []
    monkeypatch.setattr(embed_connection.time, "sleep", lambda s: sleeps.append(s))
    setattr(world, busy, True)
    backend = _backend(world)
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = "Studio .*oMLX"):
        backend.encode(["a"], model_name = "connection:Local oMLX/Qwen3-Embedding")
    assert world.requests == [] and sleeps == [] and arbiter.omlx_holds() == 0


def test_a_failed_request_still_releases_the_claim(world, monkeypatch):
    monkeypatch.setattr(embed_connection.time, "sleep", lambda s: None)
    backend = embed_connection.ConnectionEmbeddingBackend()

    def boom(request):
        raise httpx.ConnectError("refused")

    backend._http = httpx.Client(transport = httpx.MockTransport(boom))
    with pytest.raises(embed_connection.ConnectionEmbedderError):
        backend.encode(["a"], model_name = "connection:Local oMLX/Qwen3-Embedding")
    assert arbiter.omlx_holds() == 0


def test_a_held_embedding_request_keeps_local_loads_and_comfyui_jobs_out(world, monkeypatch):
    seen = {}

    def handler(request):
        async def competitors():
            seen["local"] = (await arbiter.free_for_local("local_load")).error
            seen["comfyui"] = (await arbiter.before_comfyui_job()).error

        asyncio.run_coroutine_threadsafe(competitors(), arbiter._server_loop).result(10)
        return httpx.Response(200, json = {"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    backend = embed_connection.ConnectionEmbeddingBackend()
    backend._http = httpx.Client(transport = httpx.MockTransport(handler))
    backend.encode(["a"], model_name = "connection:Local oMLX/Qwen3-Embedding")
    assert seen["local"] and seen["comfyui"], seen


@pytest.mark.parametrize("name", ["Local other", "Remote"])
def test_other_connections_never_ask_the_arbiter(world, name, monkeypatch):
    world.training = True  # would refuse an oMLX call
    called = []
    monkeypatch.setattr(arbiter, "before_omlx_use_from_thread", lambda *a, **k: called.append(1))
    backend = _backend(world)
    assert backend.encode(["a"], model_name = f"connection:{name}/bge").shape == (1, 2)
    assert called == [] and world.holds_seen == [0]


def test_with_attached_engines_off_nothing_is_asked_and_no_loop_is_needed(world, monkeypatch):
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    arbiter.remember_loop(None)
    world.training = True
    backend = _backend(world)
    assert backend.encode(["a"], model_name = "connection:Local oMLX/Qwen3-Embedding").shape == (1, 2)
    assert world.holds_seen == [0]


def test_without_a_server_loop_the_request_fails_closed(world):
    arbiter.remember_loop(None)
    backend = _backend(world)
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = "not running"):
        backend.encode(["a"], model_name = "connection:Local oMLX/Qwen3-Embedding")
    assert world.requests == []


def test_a_caller_on_the_server_loop_fails_closed_instead_of_deadlocking(world):
    async def on_the_loop():
        return arbiter.before_omlx_use_from_thread(arbiter.new_omlx_hold(), asyncio.get_running_loop())

    result = asyncio.run(on_the_loop())
    assert result.error and "own event loop" in result.error


def test_a_bad_reference_is_reported_by_the_request_not_the_admission(world):
    backend = _backend(world)
    with pytest.raises(embed_connection.ConnectionEmbedderError, match = "No saved connection"):
        backend.encode(["a"], model_name = "connection:Nope/bge")
