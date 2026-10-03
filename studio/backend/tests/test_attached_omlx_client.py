# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached import AttachedEngineError  # noqa: E402
from core.inference.attached.omlx_client import OmlxClient, OmlxModel  # noqa: E402

SWIFT_DIR = "ukisai--Swift-1.5-27B-oQ4e-mtp"
SWIFT_PATH = f"/Users/test/.mtplx/models/{SWIFT_DIR}"
EMBED_ID = "Qwen3-Embedding-8B-4bit-DWQ"


def _row(model_id, path, **over):
    row = {
        "id": model_id,
        "model_path": path,
        "loaded": False,
        "is_loading": False,
        "estimated_size": 17820265635,
        "pinned": False,
        "engine_type": "vlm",
        "is_helper": False,
    }
    row.update(over)
    return row


# Shapes captured from a live oMLX /v1/models/status and /v1/models.
STATUS = {
    "final_ceiling": 31174935238,
    "current_model_memory": 4470007249,
    "model_count": 6,
    "loaded_count": 1,
    "models": [
        _row(
            EMBED_ID,
            f"/Users/test/.mtplx/models/mlx-community/{EMBED_ID}",
            loaded = True,
            pinned = True,
            engine_type = "embedding",
            estimated_size = 4470007249,
        ),
        _row(SWIFT_DIR, SWIFT_PATH, model_alias = "swift-1.5-27b"),
        _row(
            "z-lab--Qwen3.8-27B-DFlash2",
            "/Users/test/.mtplx/models/z-lab--Qwen3.8-27B-DFlash2",
            engine_type = "batched",
            is_helper = True,
            estimated_size = 4041258790,
        ),
        _row("swift-1.5-27b:fast", SWIFT_PATH),
        _row("swift-1.5-27b:medium", SWIFT_PATH),
        _row("swift-1.5-27b:low", SWIFT_PATH),
    ],
}
MODELS = {
    "object": "list",
    "data": [
        {"id": model_id, "object": "model", "owned_by": "omlx"}
        for model_id in (
            EMBED_ID,
            "swift-1.5-27b",
            "swift-1.5-27b:fast",
            "swift-1.5-27b:medium",
            "swift-1.5-27b:low",
        )
    ],
}


class Server:
    def __init__(
        self,
        status = STATUS,
        models = MODELS,
    ):
        self.status = status
        self.models = models
        self.calls: list[tuple[str, str]] = []
        self.raw_paths: list[str] = []
        self.timeouts: list[dict] = []
        self.post_status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        self.raw_paths.append(request.url.raw_path.decode())
        self.timeouts.append(dict(request.extensions.get("timeout", {})))
        if request.method == "POST":
            return httpx.Response(self.post_status, json = {"success": True})
        if request.url.path == "/v1/models/status":
            return httpx.Response(200, json = self.status)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json = self.models)
        return httpx.Response(404)

    def client(self) -> OmlxClient:
        return OmlxClient("http://127.0.0.1:8843/", transport = httpx.MockTransport(self))


def run(coro):
    return asyncio.run(coro)


def test_status_parses_live_shape():
    status = run(Server().client().status())
    assert status.reachable is True
    assert status.memory_bytes == 4470007249
    assert status.ceiling_bytes == 31174935238
    assert len(status.models) == 6
    embed = status.models[0]
    assert (embed.id, embed.pinned, embed.loaded, embed.engine_type) == (
        EMBED_ID,
        True,
        True,
        "embedding",
    )
    assert status.models[1].model_alias == "swift-1.5-27b"
    assert status.models[2].is_helper is True


def test_status_unreachable_never_raises():
    def boom(request):
        raise httpx.ConnectError("refused", request = request)

    client = OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(boom))
    status = run(client.status())
    assert status.reachable is False
    assert status.models == ()
    assert run(client.chat_model_ids()) is None
    assert run(client.loaded_ids()) == []
    assert run(client.unload_all()) == []


def test_status_http_error_and_bad_json_are_unreachable():
    for response in (httpx.Response(500), httpx.Response(200, content = b"not json")):
        client = OmlxClient(
            "http://127.0.0.1:8843", transport = httpx.MockTransport(lambda r, resp = response: resp)
        )
        assert run(client.status()).reachable is False


def test_chat_model_ids_drop_embedding_and_helpers_keep_aliases():
    ids = run(Server().client().chat_model_ids())
    assert ids == [
        "swift-1.5-27b",
        "swift-1.5-27b:fast",
        "swift-1.5-27b:medium",
        "swift-1.5-27b:low",
    ]


def test_chat_model_ids_drops_helper_listed_on_v1_models():
    models = {"data": [{"id": "z-lab--Qwen3.8-27B-DFlash2"}, {"id": "swift-1.5-27b"}]}
    assert run(Server(models = models).client().chat_model_ids()) == ["swift-1.5-27b"]


def test_chat_model_ids_falls_back_to_name_when_status_down():
    server = Server()
    original = server.__call__

    def handler(request):
        if request.url.path == "/v1/models/status":
            return httpx.Response(503)
        return original(request)

    client = OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(handler))
    ids = run(client.chat_model_ids())
    assert EMBED_ID not in ids
    assert "swift-1.5-27b:fast" in ids


def test_loaded_ids():
    assert run(Server().client().loaded_ids()) == [EMBED_ID]


def test_resolve_dir_maps_alias_and_profiles_to_directory():
    models = run(Server().client().status()).models
    resolve = OmlxClient.resolve_dir
    assert resolve(list(models), "swift-1.5-27b") == SWIFT_DIR
    assert resolve(list(models), "swift-1.5-27b:fast") == SWIFT_DIR
    assert resolve(list(models), "swift-1.5-27b:low") == SWIFT_DIR
    assert resolve(list(models), SWIFT_DIR) == SWIFT_DIR
    assert resolve(list(models), EMBED_ID) == EMBED_ID
    assert resolve(list(models), "no-such-model") is None


def test_resolve_dir_without_basename_match_prefers_non_profile_id():
    models = [
        OmlxModel(id = "alias:fast", model_path = "/m/x"),
        OmlxModel(id = "plain-name", model_path = "/m/x"),
    ]
    assert OmlxClient.resolve_dir(models, "alias:fast") == "plain-name"


def test_load_posts_dir_id_with_long_timeout():
    server = Server()
    run(server.client().load(SWIFT_DIR))
    assert server.calls == [("POST", f"/admin/api/models/{SWIFT_DIR}/load")]
    assert server.timeouts[0]["read"] == 300.0
    assert server.timeouts[0]["connect"] == 2.0


def test_unload_quotes_colon_ids():
    server = Server()
    run(server.client().unload("swift-1.5-27b:fast"))
    assert server.raw_paths == ["/v1/models/swift-1.5-27b%3Afast/unload"]


def test_load_and_unload_raise_on_failure():
    server = Server()
    server.post_status = 500
    with pytest.raises(AttachedEngineError):
        run(server.client().load(SWIFT_DIR))
    with pytest.raises(AttachedEngineError):
        run(server.client().unload(EMBED_ID))

    def refused(request):
        raise httpx.ConnectError("refused", request = request)

    with pytest.raises(AttachedEngineError):
        run(OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(refused)).load("x"))


def test_unload_all_includes_pinned_and_dedupes_by_directory():
    status = {
        **STATUS,
        "models": [
            {**m, "loaded": True} if m["id"] in (SWIFT_DIR, "swift-1.5-27b:fast") else m
            for m in STATUS["models"]
        ],
    }
    server = Server(status = status)
    unloaded = run(server.client().unload_all())
    assert unloaded == [EMBED_ID, SWIFT_DIR]
    posts = [path for method, path in server.calls if method == "POST"]
    assert posts == [f"/v1/models/{EMBED_ID}/unload", f"/v1/models/{SWIFT_DIR}/unload"]


def test_unload_all_continues_after_one_failure():
    status = {
        **STATUS,
        "models": [{**m, "loaded": m["id"] in (EMBED_ID, SWIFT_DIR)} for m in STATUS["models"]],
    }
    server = Server(status = status)
    original = server.__call__

    def handler(request):
        if request.method == "POST" and EMBED_ID in request.url.path:
            return httpx.Response(500)
        return original(request)

    client = OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(handler))
    assert run(client.unload_all()) == [SWIFT_DIR]


def test_chat_model_ids_reuses_a_supplied_status():
    server = Server()
    client = server.client()
    status = run(client.status())
    server.calls.clear()
    ids = run(client.chat_model_ids(status = status))
    assert "swift-1.5-27b" in ids and EMBED_ID not in ids
    assert server.calls == [("GET", "/v1/models")]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, content = b"not json"),
        httpx.Response(200, json = {"object": "list"}),
        httpx.Response(200, json = {"data": "nope"}),
        httpx.Response(200, json = ["not", "an", "object"]),
    ],
)
def test_chat_model_ids_is_none_not_empty_when_the_listing_fails(response):
    def handler(request):
        if request.url.path == "/v1/models":
            return response
        return httpx.Response(200, json = STATUS)

    client = OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(handler))
    assert run(client.chat_model_ids()) is None


def test_chat_model_ids_is_none_on_timeout_and_empty_list_for_a_real_empty_catalog():
    def slow(request):
        raise httpx.ReadTimeout("slow", request = request)

    client = OmlxClient("http://127.0.0.1:8843", transport = httpx.MockTransport(slow))
    assert run(client.chat_model_ids()) is None
    assert run(Server(models = {"data": []}).client().chat_model_ids()) == []
