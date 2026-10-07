# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import types
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import routes.attached_engines as routes  # noqa: E402
from auth import policy  # noqa: E402
from auth.authentication import get_current_subject  # noqa: E402
from core.inference.attached import ATTACHED_OMLX_ID, arbiter  # noqa: E402
from core.inference.attached.comfyui_client import ComfyuiClient  # noqa: E402
from core.inference.attached.omlx_client import OmlxClient  # noqa: E402
from routes.provider_credentials import provider_config_guard  # noqa: E402
from storage import providers_db  # noqa: E402
from utils.attached_engines_settings import AttachedEnginesConfig  # noqa: E402

SWIFT_DIR = "ukisai--Swift-1.5-27B-oQ4e-mtp"
SWIFT_PATH = f"/m/{SWIFT_DIR}"
EMBED = "Qwen3-Embedding-8B-4bit-DWQ"


def _row(model_id, path, **over):
    row = {
        "id": model_id,
        "model_path": path,
        "loaded": False,
        "estimated_size": 1000,
        "pinned": False,
        "engine_type": "vlm",
        "is_helper": False,
    }
    row.update(over)
    return row


class Fake:
    """oMLX behind a MockTransport."""

    def __init__(self):
        self.omlx_up = True
        self.calls: list[str] = []
        self.chat_ids = [
            "swift-1.5-27b",
            "swift-1.5-27b:fast",
            "swift-1.5-27b:low",
        ]
        self.loaded = {EMBED}
        self.post_status = 200
        self.post_json = {}
        self.models_fail = False
        self.ctx_setting = 131072
        self.native = 262144
        # Studio's ComfyUI (8844) and the StoryPress peer (8188); nothing here ever reaches a real port.
        self.comfy = {
            8844: dict(up = True, running = [], pending = [], free_status = 200),
            8188: dict(up = True, running = [], pending = [], free_status = 200),
        }
        self.comfy_calls: list[tuple[int, str, str, object]] = []
        self.comfy_models = {"diffusion_models": ["qwen.safetensors"], "loras": ["a.safetensors"]}

    def omlx(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"omlx {request.method} {request.url.raw_path.decode()}")
        if not self.omlx_up:
            raise httpx.ConnectError("refused", request = request)
        path = request.url.path
        if request.method == "PUT" and path.endswith("/settings"):
            self.put_body = json.loads(request.content)
            self.ctx_setting = self.put_body["max_context_window"]
            return httpx.Response(200, json = {"requires_reload": False})
        if path == "/admin/api/models":
            return httpx.Response(
                200,
                json = {
                    "models": [
                        {
                            "id": SWIFT_DIR,
                            "model_context_length": self.native,
                            "settings": {"max_context_window": self.ctx_setting},
                        },
                        {"id": EMBED, "model_context_length": 40960, "settings": {}},
                    ]
                },
            )
        if path == "/admin/api/global-settings":
            return httpx.Response(200, json = {"idle_timeout": {"idle_timeout_seconds": 600}})
        if request.method == "POST":
            if path.endswith("/unload"):
                self.loaded.discard(path.split("/")[3])
            if path.endswith("/load"):
                self.loaded.add(path.split("/")[4])
            return httpx.Response(self.post_status, json = self.post_json)
        if path == "/v1/models/status":
            rows = [
                _row(
                    EMBED,
                    f"/m/{EMBED}",
                    engine_type = "embedding",
                    pinned = True,
                    loaded = EMBED in self.loaded,
                ),
                _row(
                    SWIFT_DIR,
                    SWIFT_PATH,
                    model_alias = "swift-1.5-27b",
                    loaded = SWIFT_DIR in self.loaded,
                    max_context_window = self.ctx_setting or self.native,
                    model_context_length = self.native,
                ),
                _row("swift-1.5-27b:fast", SWIFT_PATH),
                _row("swift-1.5-27b:low", SWIFT_PATH),
            ]
            return httpx.Response(
                200,
                json = {"models": rows, "current_model_memory": 5, "final_ceiling": 50},
            )
        if path == "/v1/models":
            if self.models_fail:
                return httpx.Response(503)
            ids = [EMBED, *self.chat_ids]
            return httpx.Response(200, json = {"data": [{"id": i} for i in ids]})
        return httpx.Response(404)


def _comfy_handler(state: "Fake"):
    def handle(request: httpx.Request) -> httpx.Response:
        port = request.url.port
        body = json.loads(request.content) if request.content else None
        state.comfy_calls.append((port, request.method, request.url.path, body))
        comfy = state.comfy[port]
        if not comfy["up"]:
            raise httpx.ConnectError("refused", request = request)
        path = request.url.path
        item = lambda pid, n, extra: [n, pid, {}, extra, []]  # noqa: E731
        if path == "/queue" and request.method == "GET":
            return httpx.Response(
                200,
                json = {
                    "queue_running": [item(pid, i, {"unsloth": {"template": "t"}} if pid == "mine" else {}) for i, pid in enumerate(comfy["running"])],
                    "queue_pending": [item(pid, 10 + i, {}) for i, pid in enumerate(comfy["pending"])],
                },
            )
        if path == "/system_stats":
            return httpx.Response(
                200,
                json = {
                    "system": {"comfyui_version": "0.39.0", "ram_total": 64, "ram_free": 20},
                    "devices": [{"name": "mps", "type": "mps", "vram_total": 64, "vram_free": 20}],
                },
            )
        if path.startswith("/models/"):
            return httpx.Response(200, json = state.comfy_models.get(path.rsplit("/", 1)[-1], []))
        if request.method == "POST" and path in {"/free", "/interrupt", "/queue"}:
            return httpx.Response(comfy["free_status"] if path == "/free" else 200)
        return httpx.Response(404)

    return handle


def _config(**over) -> AttachedEnginesConfig:
    base = dict(
        enabled = True,
        omlx_url = "http://127.0.0.1:8843",
        scan_denylist = (),
        arbitrate_local_loads = True,
    )
    base.update(over)
    return AttachedEnginesConfig(**base)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    state = Fake()
    config = {"value": _config()}
    monkeypatch.setattr(routes, "get_config", lambda: config["value"])
    monkeypatch.setattr(arbiter, "get_config", lambda: config["value"])
    state.config = config
    for module in (routes, arbiter):
        monkeypatch.setattr(
            module,
            "OmlxClient",
            lambda url: OmlxClient(url, transport = httpx.MockTransport(state.omlx)),
        )
    monkeypatch.delenv("STUDIO_COMFYUI_URL", raising = False)
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path / "engines"))
    state.engines_home = tmp_path / "engines"
    state.engines_home.mkdir()
    comfy_transport = httpx.MockTransport(_comfy_handler(state))
    for module in (routes, arbiter):
        monkeypatch.setattr(
            module, "ComfyuiClient", lambda url, transport = None: ComfyuiClient(url, transport = comfy_transport)
        )
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    arbiter._notices.clear()
    arbiter._busy_cache.clear()
    return state


@pytest.fixture
def client(fake):
    app = FastAPI()
    app.include_router(routes.router, prefix = "/api/engines/attached")
    app.dependency_overrides[get_current_subject] = lambda: "owner"
    with TestClient(app) as test_client:
        yield test_client


def _wait_for(predicate, timeout = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_every_route_but_sync_is_404_when_flag_off(client, fake):
    fake.config["value"] = _config(enabled = False)
    for method, path, body in (
        ("get", "/status", None),
        ("post", "/omlx/load", {"model_id": "x"}),
        ("post", "/omlx/unload", {"model_id": "x"}),
        ("post", "/omlx/unload-all", None),
        ("post", "/omlx/context/get", {"model_id": "x"}),
        ("post", "/omlx/context", {"model_id": "x", "max_context_window": 8192}),
        ("post", "/prepare", {"provider": "omlx"}),
        ("post", "/comfyui/free", None),
        ("get", "/comfyui/queue", None),
        ("post", "/comfyui/queue/cancel", {"prompt_id": "x"}),
        ("get", "/comfyui/models", None),
    ):
        response = client.request(method.upper(), f"/api/engines/attached{path}", json = body)
        assert response.status_code == 404, path
    assert fake.calls == [] and fake.comfy_calls == []


def test_routes_require_owner_and_authentication(fake):
    deps = [dep.dependency for dep in routes.router.dependencies]
    assert policy.require_owner in deps and get_current_subject in deps

    app = FastAPI()
    app.include_router(routes.router, prefix = "/api/engines/attached")

    async def not_owner():
        raise HTTPException(status_code = 403, detail = "Only the installation owner can do this")

    app.dependency_overrides[get_current_subject] = lambda: "member"
    app.dependency_overrides[policy.require_owner] = not_owner
    with TestClient(app) as test_client:
        assert test_client.get("/api/engines/attached/status").status_code == 403
        assert test_client.post("/api/engines/attached/sync").status_code == 403
        for method, path in (
            ("post", "/comfyui/free"),
            ("get", "/comfyui/queue"),
            ("post", "/comfyui/queue/cancel"),
            ("get", "/comfyui/models"),
        ):
            assert test_client.request(method.upper(), f"/api/engines/attached{path}").status_code == 403
    assert fake.calls == [] and fake.comfy_calls == []


def test_status_reports_omlx_hash_and_notices(client, fake):
    arbiter._record("training", ["Unloaded oMLX"])
    body = client.get("/api/engines/attached/status").json()
    assert body["enabled"] is True
    assert body["omlx"]["reachable"] is True
    assert body["omlx"]["memory_bytes"] == 5 and body["omlx"]["ceiling_bytes"] == 50
    assert body["omlx"]["chat_model_ids"] == fake.chat_ids
    assert len(body["models_hash"]) == 16
    assert [n["reason"] for n in body["notices"]] == ["training"]
    later = client.get("/api/engines/attached/status", params = {"since": time.time() + 5}).json()
    assert later["notices"] == []
    fake.chat_ids.append("swift-1.5-27b:medium")
    assert client.get("/api/engines/attached/status").json()["models_hash"] != body["models_hash"]


def test_status_with_engines_down_never_errors(client, fake):
    fake.omlx_up = False
    body = client.get("/api/engines/attached/status").json()
    assert body["omlx"]["reachable"] is False and body["omlx"]["chat_model_ids"] == []


def test_sync_seeds_rows_with_live_models(client, fake):
    response = client.post("/api/engines/attached/sync")
    assert response.status_code == 200
    omlx = providers_db.get_provider(ATTACHED_OMLX_ID)
    assert omlx["provider_type"] == "omlx"
    assert omlx["base_url"] == "http://127.0.0.1:8843/v1"
    assert omlx["models"] == fake.chat_ids
    assert EMBED not in omlx["available_models"]
    assert omlx["is_enabled"] == 1


def test_sync_keeps_user_choices_and_adds_new_models(client, fake):
    client.post("/api/engines/attached/sync")
    providers_db.update_provider(ATTACHED_OMLX_ID, models = ["swift-1.5-27b"], is_enabled = False)
    fake.chat_ids.append("swift-1.5-27b:medium")
    assert client.post("/api/engines/attached/sync").json()["rows"]["omlx"] == "updated"
    row = providers_db.get_provider(ATTACHED_OMLX_ID)
    assert row["models"] == ["swift-1.5-27b", "swift-1.5-27b:medium"]
    assert row["available_models"] == fake.chat_ids
    assert row["is_enabled"] == 0  # never re-enabled by a sync


def test_sync_does_not_overwrite_a_choice_saved_while_it_is_reading(client, fake, monkeypatch):
    client.post("/api/engines/attached/sync")
    entered = threading.Event()
    release = threading.Event()
    original_get = providers_db.get_provider
    calls = 0

    def delayed_get(row_id):
        nonlocal calls
        calls += 1
        if row_id == ATTACHED_OMLX_ID and calls == 1:
            entered.set()
            assert release.wait(1)
        return original_get(row_id)

    monkeypatch.setattr(providers_db, "get_provider", delayed_get)

    async def run():
        sync = asyncio.create_task(
            routes._upsert_row(
                ATTACHED_OMLX_ID,
                "omlx",
                "oMLX",
                "http://127.0.0.1:8843/v1",
                fake.chat_ids,
            )
        )
        assert await asyncio.to_thread(entered.wait, 1)

        async def choose_one_model():
            async with provider_config_guard(ATTACHED_OMLX_ID):
                await asyncio.to_thread(
                    providers_db.update_provider,
                    ATTACHED_OMLX_ID,
                    models = ["swift-1.5-27b:fast"],
                )

        choice = asyncio.create_task(choose_one_model())
        release.set()
        await asyncio.gather(sync, choice)

    asyncio.run(run())
    assert providers_db.get_provider(ATTACHED_OMLX_ID)["models"] == ["swift-1.5-27b:fast"]


def test_sync_with_engine_down_keeps_models_and_enabled_flag(client, fake):
    client.post("/api/engines/attached/sync")
    fake.omlx_up = False
    assert client.post("/api/engines/attached/sync").json()["rows"]["omlx"] == "kept_models"
    row = providers_db.get_provider(ATTACHED_OMLX_ID)
    assert row["models"] == fake.chat_ids
    assert row["is_enabled"] == 1


def test_sync_updates_base_url_from_config(client, fake):
    client.post("/api/engines/attached/sync")
    fake.config["value"] = _config(omlx_url = "http://127.0.0.1:9999")
    client.post("/api/engines/attached/sync")
    assert providers_db.get_provider(ATTACHED_OMLX_ID)["base_url"] == "http://127.0.0.1:9999/v1"


def test_sync_deletes_rows_when_flag_off(client, fake):
    client.post("/api/engines/attached/sync")
    providers_db.create_provider(
        id = "userrow", provider_type = "custom", display_name = "Mine", base_url = "http://x/v1"
    )
    fake.config["value"] = _config(enabled = False)
    calls_before = len(fake.calls)
    response = client.post("/api/engines/attached/sync")
    assert providers_db.get_provider(ATTACHED_OMLX_ID) is None
    assert providers_db.get_provider("userrow") is not None
    assert len(fake.calls) == calls_before


def test_omlx_load_unknown_model_and_failures(client, fake):
    assert (
        client.post("/api/engines/attached/omlx/load", json = {"model_id": "nope"}).status_code == 404
    )
    fake.post_status = 500
    assert (
        client.post(
            "/api/engines/attached/omlx/load", json = {"model_id": "swift-1.5-27b"}
        ).status_code
        == 502
    )
    fake.omlx_up = False
    assert (
        client.post(
            "/api/engines/attached/omlx/load", json = {"model_id": "swift-1.5-27b"}
        ).status_code
        == 502
    )


def test_omlx_unload_and_unload_all(client, fake):
    fake.loaded = {EMBED, SWIFT_DIR}
    response = client.post("/api/engines/attached/omlx/unload", json = {"model_id": "swift-1.5-27b"})
    assert response.json() == {"unloaded": SWIFT_DIR}
    assert fake.loaded == {EMBED}
    response = client.post("/api/engines/attached/omlx/unload-all")
    assert response.json() == {"unloaded": [EMBED]}
    assert fake.loaded == set()


def test_busy_omlx_unload_is_a_retryable_503(client, fake):
    fake.loaded = {EMBED, SWIFT_DIR}
    fake.post_status = 409
    fake.post_json = {"error": {"type": "model_busy"}}
    for path, body in (("/omlx/unload", {"model_id": "swift-1.5-27b"}), ("/omlx/unload-all", None)):
        response = client.post(f"/api/engines/attached{path}", json = body)
        assert response.status_code == 503
        assert response.headers["Retry-After"] == "15"
        assert "busy serving another client" in response.json()["detail"]


def test_ids_with_colons_only_ever_travel_in_bodies(client, fake):
    client.post("/api/engines/attached/omlx/unload", json = {"model_id": "swift-1.5-27b:low"})
    assert not any(
        ":" in path for path in (r.rsplit(" ", 1)[-1] for r in fake.calls if "unload" in r)
    )


def test_prepare_rejects_unknown_provider(client):
    assert (
        client.post("/api/engines/attached/prepare", json = {"provider": "ollama"}).status_code == 422
    )


def test_router_is_mounted_before_the_generic_engines_router():
    source = (_BACKEND / "main.py").read_text()
    attached = source.index("attached_engines_router, prefix")
    generic = source.index('app.include_router(engines_router, prefix = "/api/engines"')
    assert attached < generic


def test_sync_preserves_saved_models_when_a_catalog_fetch_fails(client, fake):
    client.post("/api/engines/attached/sync")
    providers_db.update_provider(ATTACHED_OMLX_ID, models = ["swift-1.5-27b"])
    before_omlx = providers_db.get_provider(ATTACHED_OMLX_ID)
    fake.models_fail = True
    body = client.post("/api/engines/attached/sync").json()
    assert body["rows"] == {"omlx": "kept_models"}
    after = providers_db.get_provider(ATTACHED_OMLX_ID)
    assert after["models"] == before_omlx["models"]
    assert after["available_models"] == before_omlx["available_models"]
    assert providers_db.get_provider(ATTACHED_OMLX_ID)["models"] == ["swift-1.5-27b"]


def test_sync_clears_models_when_a_reachable_engine_really_lists_none(client, fake):
    client.post("/api/engines/attached/sync")
    fake.chat_ids.clear()
    client.post("/api/engines/attached/sync")
    assert providers_db.get_provider(ATTACHED_OMLX_ID)["available_models"] == []


def test_status_reports_empty_lists_when_a_catalog_fetch_fails(client, fake):
    fake.models_fail = True
    body = client.get("/api/engines/attached/status").json()
    assert body["omlx"]["reachable"] is True
    assert body["omlx"]["chat_model_ids"] == []


def test_status_exposes_engine_failure_when_the_reader_is_available(client, monkeypatch):
    failures = types.ModuleType("core.inference.attached.failures")
    failures.read_failure = lambda name: {"reason": f"{name} failed"}
    monkeypatch.setitem(sys.modules, "core.inference.attached.failures", failures)

    body = client.get("/api/engines/attached/status").json()

    assert body["omlx"]["failure"] == {"reason": "omlx failed"}


def test_omlx_context_get_resolves_alias_and_reports_all_three_values(client, fake):
    body = client.post(
        "/api/engines/attached/omlx/context/get", json = {"model_id": "swift-1.5-27b:fast"}
    ).json()
    assert body == {
        "model_id": "swift-1.5-27b:fast",
        "dir": SWIFT_DIR,
        "max_context_window": 131072,
        "native_max": 262144,
        "effective": 131072,
    }


def test_omlx_context_set_puts_to_the_directory_and_returns_the_new_state(client, fake):
    body = client.post(
        "/api/engines/attached/omlx/context",
        json = {"model_id": "swift-1.5-27b", "max_context_window": 65536},
    ).json()
    assert fake.put_body == {"max_context_window": 65536}
    assert any(c == f"omlx PUT /admin/api/models/{SWIFT_DIR}/settings" for c in fake.calls)
    assert body["max_context_window"] == 65536 and body["effective"] == 65536
    assert body["requires_reload"] is False


def test_omlx_context_null_resets_to_default(client, fake):
    body = client.post(
        "/api/engines/attached/omlx/context",
        json = {"model_id": "swift-1.5-27b", "max_context_window": None},
    ).json()
    assert fake.put_body == {"max_context_window": None}
    assert body["max_context_window"] is None and body["effective"] == 262144


def test_omlx_context_validates_against_native_and_floor(client, fake):
    url = "/api/engines/attached/omlx/context"
    over = client.post(url, json = {"model_id": "swift-1.5-27b", "max_context_window": 300000})
    assert over.status_code == 422 and "262144" in over.json()["detail"]
    under = client.post(url, json = {"model_id": "swift-1.5-27b", "max_context_window": 100})
    assert under.status_code == 422
    assert not any(" PUT " in c for c in fake.calls)


def test_omlx_context_unknown_model_and_engine_down(client, fake):
    url = "/api/engines/attached/omlx/context/get"
    assert client.post(url, json = {"model_id": "nope"}).status_code == 404
    fake.omlx_up = False
    assert client.post(url, json = {"model_id": "swift-1.5-27b"}).status_code == 502


@pytest.mark.parametrize("enabled", [True, False])
def test_sync_deletes_legacy_row_idempotently_and_keeps_user_rows(client, fake, enabled):
    legacy_id = "attachedds400001"
    providers_db.create_provider(
        id = legacy_id, provider_type = "dwarfstar", display_name = "Legacy", base_url = "http://localhost/v1"
    )
    providers_db.create_provider(
        id = "userrow", provider_type = "custom", display_name = "Mine", base_url = "http://localhost/v1"
    )
    fake.config["value"] = _config(enabled = enabled)
    for _ in range(2):
        assert client.post("/api/engines/attached/sync").status_code == 200
        assert providers_db.get_provider(legacy_id) is None
        assert providers_db.get_provider("userrow") is not None
    assert (providers_db.get_provider(ATTACHED_OMLX_ID) is not None) == enabled


def test_legacy_row_delete_waits_for_the_provider_config_guard(fake):
    legacy_id = "attachedds400001"
    providers_db.create_provider(
        id = legacy_id, provider_type = "dwarfstar", display_name = "Legacy", base_url = "http://localhost/v1"
    )
    fake.config["value"] = _config(enabled = True)

    async def run():
        async with provider_config_guard(legacy_id):
            sync = asyncio.create_task(routes.attached_sync())
            await asyncio.sleep(0.2)
            assert not sync.done()
            assert providers_db.get_provider(legacy_id) is not None
        await sync
        return sync.result()

    assert asyncio.run(run())["enabled"] is True
    assert providers_db.get_provider(legacy_id) is None


def test_legacy_provider_type_can_be_listed_before_sync(fake, monkeypatch):
    from routes import providers
    legacy_id = "attachedds400001"
    providers_db.create_provider(
        id = legacy_id, provider_type = "dwarfstar", display_name = "Legacy", base_url = "http://localhost/v1"
    )
    monkeypatch.setattr(providers.credential_secrets, "has_secret", lambda *args: False)
    assert providers._provider_response(providers_db.get_provider(legacy_id)).display_name == "Legacy"


# --- ComfyUI ----------------------------------------------------------------------------------

BASE = "/api/engines/attached"


def test_status_comfyui_block_when_up(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    fake.comfy[8844]["pending"] = ["a", "b"]
    fake.comfy[8188]["pending"] = ["sp"]
    body = client.get(f"{BASE}/status").json()["comfyui"]
    assert body["url"] == "http://127.0.0.1:8844" and body["reachable"] is True
    assert body["version"] == "0.39.0"
    assert (body["queue_running"], body["queue_pending"]) == (1, 2)
    assert body["devices"][0]["name"] == "mps" and body["ram_free"] == 20 and body["ram_total"] == 64
    assert body["failure"] is None and body["helper_wanted"] is None
    assert body["peers"] == [{"url": "http://127.0.0.1:8188", "reachable": True, "busy": True}]


def test_status_comfyui_block_when_down_never_errors(client, fake):
    fake.comfy[8844]["up"] = False
    fake.comfy[8188]["up"] = False
    response = client.get(f"{BASE}/status")
    assert response.status_code == 200
    body = response.json()["comfyui"]
    assert body["reachable"] is False and body["version"] is None
    assert (body["queue_running"], body["queue_pending"], body["devices"]) == (0, 0, [])
    assert body["peers"] == [{"url": "http://127.0.0.1:8188", "reachable": False, "busy": False}]
    assert response.json()["omlx"]["reachable"] is True


def test_status_comfyui_down_and_omlx_down_are_independent(client, fake):
    fake.omlx_up = False
    body = client.get(f"{BASE}/status").json()
    assert body["omlx"]["reachable"] is False and body["comfyui"]["reachable"] is True


def test_status_comfyui_failure_marker_and_helper_wanted(client, fake):
    fake.comfy[8844]["up"] = False
    (fake.engines_home / "comfyui.fail").write_text(
        json.dumps({"reason": "port 8844 (ComfyUI) is held by StoryPress's ComfyUI (pid 7)", "ts": 5, "count": 2})
    )
    (fake.engines_home / "desktop.json").write_text(
        json.dumps({"engines_enabled": True, "helpers": {"comfyui": True}})
    )
    body = client.get(f"{BASE}/status").json()["comfyui"]
    assert body["failure"]["count"] == 2 and "StoryPress" in body["failure"]["reason"]
    assert body["helper_wanted"] is True and body["reachable"] is False


def test_status_comfyui_probes_the_configured_urls_and_only_reads(client, fake):
    fake.config["value"] = _config(comfyui_peer_urls = ())
    body = client.get(f"{BASE}/status").json()["comfyui"]
    assert body["peers"] == []
    assert {(c[1], c[2]) for c in fake.comfy_calls} == {("GET", "/queue"), ("GET", "/system_stats")}
    assert all(c[0] == 8844 for c in fake.comfy_calls)


def test_status_never_writes_to_the_peer(client, fake):
    client.get(f"{BASE}/status")
    assert {(c[1], c[2]) for c in fake.comfy_calls if c[0] == 8188} == {("GET", "/queue")}


def test_comfyui_free(client, fake):
    response = client.post(f"{BASE}/comfyui/free")
    assert response.status_code == 200 and response.json() == {"freed": True, "deferred": False}
    assert (8844, "POST", "/free", {"unload_models": True, "free_memory": True}) in fake.comfy_calls
    fake.comfy[8844]["running"] = ["x"]
    assert client.post(f"{BASE}/comfyui/free").json() == {"freed": True, "deferred": True}
    assert not any(c[0] == 8188 for c in fake.comfy_calls)


def test_comfyui_free_unreachable_and_refused(client, fake):
    fake.comfy[8844]["free_status"] = 500
    assert client.post(f"{BASE}/comfyui/free").status_code == 502
    fake.comfy[8844]["up"] = False
    response = client.post(f"{BASE}/comfyui/free")
    assert response.status_code == 502 and response.json()["detail"] == "ComfyUI is not reachable."


def test_comfyui_queue_lists_jobs_with_the_studio_marker(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    fake.comfy[8844]["pending"] = ["web1"]
    body = client.get(f"{BASE}/comfyui/queue").json()
    assert body["running"] == [{"prompt_id": "mine", "number": 0, "state": "running", "studio": {"template": "t"}}]
    assert body["pending"] == [{"prompt_id": "web1", "number": 10, "state": "pending", "studio": None}]


def test_comfyui_queue_unreachable_is_502(client, fake):
    fake.comfy[8844]["up"] = False
    assert client.get(f"{BASE}/comfyui/queue").status_code == 502


def test_cancel_interrupts_a_running_job_by_id(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    fake.comfy[8844]["pending"] = ["web1"]
    response = client.post(f"{BASE}/comfyui/queue/cancel", json = {"prompt_id": "mine"})
    assert response.json() == {"cancelled": "mine", "state": "running"}
    posts = [c for c in fake.comfy_calls if c[1] == "POST"]
    assert posts == [(8844, "POST", "/interrupt", {"prompt_id": "mine"})]


def test_cancel_deletes_a_pending_job(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    fake.comfy[8844]["pending"] = ["web1"]
    response = client.post(f"{BASE}/comfyui/queue/cancel", json = {"prompt_id": "web1"})
    assert response.json() == {"cancelled": "web1", "state": "pending"}
    assert [c for c in fake.comfy_calls if c[1] == "POST"] == [(8844, "POST", "/queue", {"delete": ["web1"]})]


def test_cancel_unknown_job_is_404_and_peer_jobs_are_never_touched(client, fake):
    fake.comfy[8188]["running"] = ["sp"]
    assert client.post(f"{BASE}/comfyui/queue/cancel", json = {"prompt_id": "sp"}).status_code == 404
    assert [c for c in fake.comfy_calls if c[1] == "POST"] == []
    assert client.post(f"{BASE}/comfyui/queue/cancel", json = {"prompt_id": ""}).status_code == 422
    fake.comfy[8844]["up"] = False
    assert client.post(f"{BASE}/comfyui/queue/cancel", json = {"prompt_id": "x"}).status_code == 502


def test_comfyui_models_lists_the_fixed_folders(client, fake):
    body = client.get(f"{BASE}/comfyui/models").json()
    assert list(body["folders"]) == ["diffusion_models", "checkpoints", "text_encoders", "vae", "loras"]
    assert body["folders"]["diffusion_models"] == ["qwen.safetensors"]
    assert body["folders"]["loras"] == ["a.safetensors"] and body["folders"]["vae"] == []


def test_comfyui_models_unreachable_is_502(client, fake):
    fake.comfy[8844]["up"] = False
    assert client.get(f"{BASE}/comfyui/models").status_code == 502


def test_omlx_load_frees_idle_comfyui_first(client, fake):
    response = client.post(f"{BASE}/omlx/load", json = {"model_id": "swift-1.5-27b"})
    assert response.status_code == 200
    assert "Freed ComfyUI memory" in " ".join(response.json()["actions"])
    assert [c[0] for c in fake.comfy_calls if c[1] == "POST"] == [8844, 8188]


def test_omlx_load_is_a_retryable_503_while_comfyui_generates(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    response = client.post(f"{BASE}/omlx/load", json = {"model_id": "swift-1.5-27b"})
    assert response.status_code == 503 and response.headers["Retry-After"] == "15"
    assert "ComfyUI is generating" in response.json()["detail"]
    assert not any(c.startswith("omlx POST") for c in fake.calls)
    assert [c for c in fake.comfy_calls if c[1] == "POST"] == []


def test_prepare_frees_idle_comfyui_and_is_refused_while_it_generates(client, fake):
    response = client.post(f"{BASE}/prepare", json = {"provider": "omlx"})
    assert response.status_code == 200 and response.json()["actions"]
    arbiter._busy_cache.clear()
    fake.comfy[8844]["pending"] = ["a"]
    assert client.post(f"{BASE}/prepare", json = {"provider": "omlx"}).status_code == 503


def test_context_calls_are_refused_while_comfyui_generates_but_never_free_it(client, fake):
    fake.comfy[8844]["running"] = ["mine"]
    response = client.post(f"{BASE}/omlx/context/get", json = {"model_id": "swift-1.5-27b"})
    assert response.status_code == 503
