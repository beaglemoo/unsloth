# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's own /images/load and /video/load go through the attached-engine arbiter as local loads."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

import routes.inference as inference_routes
import routes.video as video_routes
from core.inference.attached import arbiter
from core.inference.attached.comfyui_client import ComfyuiClient
from utils.attached_engines_settings import DEFAULT_CONFIG

from .test_diffusion_routes import client as image_client_fixture  # noqa: F401
from .test_video_routes import _healthy_diffusers, client as video_client_fixture  # noqa: F401

IMAGE_LOAD = ("/api/inference/images/load", {
    "model_path": "unsloth/Z-Image-Turbo-GGUF",
    "gguf_filename": "z-image-turbo-Q4_K_S.gguf",
    "base_repo": "unsloth/Z-Image-base",
})
VIDEO_LOAD = ("/api/inference/video/load", {
    "model_path": "unsloth/LTX-2.3-GGUF",
    "gguf_filename": "ltx-2.3-distilled-Q4_K_M.gguf",
})
IMAGE_STATUS = "/api/inference/images/status"
VIDEO_STATUS = "/api/inference/video/status"


@pytest.fixture
def image_client(image_client_fixture):
    return image_client_fixture


@pytest.fixture
def video_client(video_client_fixture):
    return video_client_fixture


@pytest.fixture(params = ["image", "video"])
def media(request):
    """(client, load path, load body, status path, route module guard name) for each media kind."""
    kind = request.param
    client = request.getfixturevalue(f"{kind}_client")
    path, body = IMAGE_LOAD if kind == "image" else VIDEO_LOAD
    status = IMAGE_STATUS if kind == "image" else VIDEO_STATUS
    guard = (inference_routes, "_guard_diffusion_load_against_training") if kind == "image" else (
        video_routes, "_guard_video_load_against_training"
    )
    return kind, client, path, body, status, guard


def test_a_load_is_admitted_once_as_a_local_load(media, monkeypatch):
    _, client, path, body, status, _ = media
    admit = AsyncMock(return_value = arbiter.ArbiterResult())
    monkeypatch.setattr(arbiter, "free_for_local", admit)
    response = client.post(path, json = body)
    assert response.status_code == 200, response.text
    admit.assert_awaited_once_with("local_load")
    assert client.get(status).json()["loaded"] is True


def test_a_refused_admission_is_a_retryable_503_and_loads_nothing(media, monkeypatch):
    _, client, path, body, status, _ = media
    monkeypatch.setattr(
        arbiter, "free_for_local",
        AsyncMock(return_value = arbiter.ArbiterResult(skipped = "error", error = "ComfyUI is generating")),
    )
    response = client.post(path, json = body)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    assert response.json()["detail"] == "ComfyUI is generating"
    assert client.get(status).json()["loaded"] is False


def test_a_bad_pick_never_reaches_the_arbiter(media, monkeypatch):
    kind, client, path, body, _, _ = media
    admit = AsyncMock(return_value = arbiter.ArbiterResult())
    monkeypatch.setattr(arbiter, "free_for_local", admit)
    bad = dict(body, model_path = "evil/not-a-trusted-repo")
    response = client.post(path, json = bad)
    assert response.status_code >= 400
    admit.assert_not_awaited()


def test_training_refuses_before_the_arbiter_is_asked(media, monkeypatch):
    _, client, path, body, _, (module, name) = media
    admit = AsyncMock(return_value = arbiter.ArbiterResult())
    monkeypatch.setattr(arbiter, "free_for_local", admit)

    def training():
        raise HTTPException(status_code = 409, detail = "Can't load while training is running")

    monkeypatch.setattr(module, name, training)
    assert client.post(path, json = body).status_code == 409
    admit.assert_not_awaited()


@pytest.fixture
def world(monkeypatch):
    """Real arbiter with fake oMLX and two fake ComfyUIs; the flag is on."""
    state = {"omlx": [], "free": [], "busy": False}
    monkeypatch.delenv("STUDIO_COMFYUI_URL", raising = False)
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            state["omlx"].append("unload_all")
            return ["chat"]

    def handler(request: httpx.Request) -> httpx.Response:
        port = request.url.port
        if request.url.path == "/queue":
            running = [[0, "job", {}, {}, []]] if state["busy"] and port == 8844 else []
            return httpx.Response(200, json = {"queue_running": running, "queue_pending": []})
        if request.url.path == "/free":
            state["free"].append(port)
            return httpx.Response(200)
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(arbiter, "ComfyuiClient", lambda url, transport_ = None, **kw: ComfyuiClient(url, transport = transport))
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    arbiter._notices.clear()
    return state


def test_a_load_unloads_omlx_and_frees_idle_comfyui_first(media, world):
    _, client, path, body, status, _ = media
    response = client.post(path, json = body)
    assert response.status_code == 200, response.text
    assert world["omlx"] == ["unload_all"]
    assert sorted(world["free"]) == [8188, 8844]
    assert client.get(status).json()["loaded"] is True
    assert "Unloaded oMLX" in arbiter.recent_notices()[0]["actions"][0]


def test_a_load_is_refused_while_comfyui_generates(media, world):
    _, client, path, body, status, _ = media
    world["busy"] = True
    response = client.post(path, json = body)
    assert response.status_code == 503 and response.headers["Retry-After"] == "15"
    assert "ComfyUI is generating" in response.json()["detail"]
    assert world["omlx"] == [] and world["free"] == []
    assert client.get(status).json()["loaded"] is False


def test_the_flag_off_leaves_loads_untouched(media, world, monkeypatch):
    _, client, path, body, status, _ = media
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    assert client.post(path, json = body).status_code == 200
    assert world["omlx"] == [] and world["free"] == []
