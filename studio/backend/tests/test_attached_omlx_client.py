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
        self.post_json = {"success": True}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        self.raw_paths.append(request.url.raw_path.decode())
        self.timeouts.append(dict(request.extensions.get("timeout", {})))
        if request.method == "POST":
            if self.post_json is None:
                return httpx.Response(self.post_status, text = "busy")
            return httpx.Response(self.post_status, json = self.post_json)
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
    with pytest.raises(AttachedEngineError, match = "Cannot verify"):
        run(client.unload_all())


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


def test_unload_waits_longer_than_other_calls():
    server = Server()
    run(server.client().unload(EMBED_ID))
    run(server.client().status())
    assert server.timeouts[0]["read"] == 25.0
    assert server.timeouts[0]["connect"] == 2.0
    assert server.timeouts[1]["read"] == 5.0


def _busy_server():
    server = Server()
    server.post_status = 409
    server.post_json = {"error": {"type": "model_busy", "message": "busy"}}
    return server


def test_unload_409_model_busy_is_a_busy_error_with_a_clear_message():
    from core.inference.attached import AttachedEngineBusy

    with pytest.raises(AttachedEngineBusy, match = "busy serving another client; retry shortly"):
        run(_busy_server().client().unload(EMBED_ID))


def test_unload_409_other_type_stays_a_plain_error():
    from core.inference.attached import AttachedEngineBusy

    server = _busy_server()
    server.post_json = {"error": {"type": "model_loading"}}
    with pytest.raises(AttachedEngineError, match = "HTTP 409") as caught:
        run(server.client().unload(EMBED_ID))
    assert not isinstance(caught.value, AttachedEngineBusy)
    server.post_json = None
    with pytest.raises(AttachedEngineError, match = "HTTP 409"):
        run(server.client().unload(EMBED_ID))


def test_unload_all_surfaces_a_busy_model():
    from core.inference.attached import AttachedEngineBusy

    with pytest.raises(AttachedEngineBusy):
        run(_busy_server().client().unload_all())


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
    original = server.__call__

    def handler(request):
        response = original(request)
        if request.method == "POST":
            target = request.url.path.split("/")[3]
            target_path = next(m["model_path"] for m in status["models"] if m["id"] == target)
            for model in status["models"]:
                if model["model_path"] == target_path:
                    model["loaded"] = False
        return response

    client = OmlxClient("http://mock", transport = httpx.MockTransport(handler))
    unloaded = run(client.unload_all())
    assert unloaded == [EMBED_ID, SWIFT_DIR]
    posts = [path for method, path in server.calls if method == "POST"]
    assert posts == [f"/v1/models/{EMBED_ID}/unload", f"/v1/models/{SWIFT_DIR}/unload"]


def test_unload_all_reports_failure():
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
    with pytest.raises(AttachedEngineError, match = "HTTP 500"):
        run(client.unload_all())


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


def _loaded_status(**swift_over):
    rows = [dict(r) for r in STATUS["models"]]
    for row in rows:
        if row["id"] == SWIFT_DIR:
            row.update(loaded = True, **swift_over)
    return {**STATUS, "models": rows}


class TtlServer(Server):
    def __init__(self, *, global_ttl = 600, own_ttl = None, **kwargs):
        super().__init__(**kwargs)
        self.global_ttl = global_ttl
        self.own_ttl = own_ttl
        self.put_bodies: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "PUT":
            self.calls.append((request.method, request.url.path))
            self.put_bodies.append(__import__("json").loads(request.content))
            return httpx.Response(200, json = {"requires_reload": True})
        if request.url.path == "/admin/api/global-settings":
            return httpx.Response(
                200, json = {"idle_timeout": {"idle_timeout_seconds": self.global_ttl}}
            )
        if request.url.path == "/admin/api/models":
            return httpx.Response(
                200,
                json = {
                    "models": [
                        {
                            "id": SWIFT_DIR,
                            "model_context_length": 262144,
                            "settings": {"max_context_window": 131072, "ttl_seconds": self.own_ttl},
                        }
                    ]
                },
            )
        return super().__call__(request)

    def client(self) -> OmlxClient:
        return OmlxClient("http://127.0.0.1:8843/", transport = httpx.MockTransport(self))


def test_idle_remaining_counts_down_from_last_access():
    import time

    status = run(
        TtlServer(status = _loaded_status(last_access = time.time() - 100)).client().status()
    )
    swift = next(m for m in status.models if m.id == SWIFT_DIR)
    assert swift.ttl_s == 600
    assert 495 <= swift.idle_remaining_s <= 500


def test_per_model_ttl_wins_and_remaining_never_goes_negative():
    import time

    server = TtlServer(own_ttl = 60, status = _loaded_status(last_access = time.time() - 500))
    swift = next(m for m in run(server.client().status()).models if m.id == SWIFT_DIR)
    assert (swift.ttl_s, swift.idle_remaining_s) == (60, 0.0)


def test_profile_row_uses_directory_ttl_and_last_access():
    import time

    server = TtlServer(own_ttl = 300, status = _loaded_status(last_access = time.time() - 50))
    rows = {m.id: m for m in run(server.client().status()).models}
    assert rows["swift-1.5-27b:fast"].idle_remaining_s is None  # not loaded itself


def test_pinned_unloaded_and_ttl_less_models_have_no_countdown():
    import time

    pinned = run(TtlServer(status = _loaded_status(pinned = True, last_access = time.time())).client().status())
    assert next(m for m in pinned.models if m.id == SWIFT_DIR).idle_remaining_s is None
    embed = next(m for m in pinned.models if m.id == EMBED_ID)
    assert embed.idle_remaining_s is None and embed.pinned
    none = run(TtlServer(global_ttl = None, status = _loaded_status(last_access = time.time())).client().status())
    assert next(m for m in none.models if m.id == SWIFT_DIR).idle_remaining_s is None
    cold = run(TtlServer().client().status())
    assert all(m.idle_remaining_s is None for m in cold.models)


def test_status_survives_unreachable_ttl_endpoints():
    import time

    status = run(Server(status = _loaded_status(last_access = time.time())).client().status())
    assert status.reachable is True
    assert all(m.idle_remaining_s is None for m in status.models)


def test_set_context_puts_the_setting_and_reports_requires_reload():
    server = TtlServer()
    assert run(server.client().set_context(SWIFT_DIR, 65536)) is True
    assert run(server.client().set_context(SWIFT_DIR, None)) is True
    assert server.put_bodies == [{"max_context_window": 65536}, {"max_context_window": None}]
    assert ("PUT", f"/admin/api/models/{SWIFT_DIR}/settings") in server.calls


def test_set_context_failure_raises():
    def handler(request):
        return httpx.Response(400, json = {"detail": "bad"})

    client = OmlxClient("http://x", transport = httpx.MockTransport(handler))
    with pytest.raises(AttachedEngineError):
        run(client.set_context(SWIFT_DIR, 8192))


@pytest.mark.parametrize("loaded,loading", [(True, False), (False, True)])
def test_unload_all_verifies_accepted_drain_and_loading_residency(loaded, loading):
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(202, json = {"status": "draining"})
        if request.url.path == "/v1/models/status":
            return httpx.Response(200, json = {"models": [_row("m", "/m", loaded = loaded, is_loading = loading)]})
        return httpx.Response(404)
    client = OmlxClient("http://mock", transport = httpx.MockTransport(handler))
    with pytest.raises(AttachedEngineError, match = "still loaded or loading"):
        run(client.unload_all(timeout_s = 0))
    assert ("POST", "/v1/models/m/unload") in calls
    assert calls.count(("GET", "/v1/models/status")) == 2


def test_unload_all_waits_for_drain_to_complete(monkeypatch):
    reads = 0
    sleeps = []
    async def sleep(delay):
        sleeps.append(delay)
    monkeypatch.setattr("core.inference.attached.omlx_client.asyncio.sleep", sleep)
    def handler(request):
        nonlocal reads
        if request.method == "POST":
            return httpx.Response(202)
        if request.url.path == "/v1/models/status":
            reads += 1
            return httpx.Response(200, json = {"models": [_row("m", "/m", is_loading = reads < 3)]})
        return httpx.Response(404)
    client = OmlxClient("http://mock", transport = httpx.MockTransport(handler))
    assert run(client.unload_all()) == ["m"]
    assert sleeps == [0.2]
