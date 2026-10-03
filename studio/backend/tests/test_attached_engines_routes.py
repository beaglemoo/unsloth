# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
import time
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
from core.inference.attached import ATTACHED_DS4_ID, ATTACHED_OMLX_ID, arbiter  # noqa: E402
from core.inference.attached.ds4_client import Ds4Client  # noqa: E402
from core.inference.attached.omlx_client import OmlxClient  # noqa: E402
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
    """Both engines behind MockTransports."""

    def __init__(self):
        self.omlx_up = True
        self.ds4_up = True
        self.ds4_loaded = False
        self.ds4_pid = None
        self.calls: list[str] = []
        self.chat_ids = [
            "swift-1.5-27b",
            "swift-1.5-27b:fast",
            "swift-1.5-27b:low",
        ]
        self.loaded = {EMBED}
        self.post_status = 200

    def omlx(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"omlx {request.method} {request.url.raw_path.decode()}")
        if not self.omlx_up:
            raise httpx.ConnectError("refused", request = request)
        path = request.url.path
        if request.method == "POST":
            if path.endswith("/unload"):
                self.loaded.discard(path.split("/")[3])
            if path.endswith("/load"):
                self.loaded.add(path.split("/")[4])
            return httpx.Response(self.post_status, json = {})
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
                ),
                _row("swift-1.5-27b:fast", SWIFT_PATH),
                _row("swift-1.5-27b:low", SWIFT_PATH),
            ]
            return httpx.Response(
                200,
                json = {"models": rows, "current_model_memory": 5, "final_ceiling": 50},
            )
        if path == "/v1/models":
            ids = [EMBED, *self.chat_ids]
            return httpx.Response(200, json = {"data": [{"id": i} for i in ids]})
        return httpx.Response(404)

    def ds4(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(f"ds4 {request.method} {request.url.path}")
        if not self.ds4_up:
            raise httpx.ConnectError("refused", request = request)
        if request.url.path == "/admin/start":
            self.ds4_loaded, self.ds4_pid = True, 7
            return httpx.Response(200, json = {"status": "ok"})
        if request.url.path == "/admin/stop":
            self.ds4_loaded, self.ds4_pid = False, None
            return httpx.Response(200, json = {"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json = {"data": [{"id": "qwen3.8-flash-next"}, {"id": "qwen3.8-flash-next-chat"}]},
            )
        return httpx.Response(
            200,
            json = {
                "loaded": self.ds4_loaded,
                "pid": self.ds4_pid,
                "in_flight": 0,
                "stats": {"live": None, "last": None, "totals": {}},
                "config": {"ds4_start_timeout": 120.0},
            },
        )


def _config(**over) -> AttachedEnginesConfig:
    base = dict(
        enabled = True,
        omlx_url = "http://127.0.0.1:8843",
        ds4_url = "http://127.0.0.1:8001",
        scan_denylist = (),
        arbitrate_local_loads = True,
        prewarm_ds4_on_select = True,
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
        monkeypatch.setattr(
            module,
            "Ds4Client",
            lambda url: Ds4Client(url, transport = httpx.MockTransport(state.ds4)),
        )
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()
    arbiter._notices.clear()
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
        ("post", "/ds4/start", None),
        ("post", "/ds4/stop", None),
        ("post", "/prepare", {"provider": "omlx"}),
    ):
        response = client.request(method.upper(), f"/api/engines/attached{path}", json = body)
        assert response.status_code == 404, path
    assert fake.calls == []


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
    assert fake.calls == []


def test_status_reports_both_engines_hash_and_notices(client, fake):
    arbiter._record("training", ["Stopped DwarfStar"], 1)
    body = client.get("/api/engines/attached/status").json()
    assert body["enabled"] is True
    assert body["omlx"]["reachable"] is True
    assert body["omlx"]["memory_bytes"] == 5 and body["omlx"]["ceiling_bytes"] == 50
    assert body["omlx"]["chat_model_ids"] == fake.chat_ids
    assert body["ds4"]["reachable"] is True and body["ds4"]["loaded"] is False
    assert len(body["models_hash"]) == 16
    assert [n["reason"] for n in body["notices"]] == ["training"]
    later = client.get("/api/engines/attached/status", params = {"since": time.time() + 5}).json()
    assert later["notices"] == []
    fake.chat_ids.append("swift-1.5-27b:medium")
    assert client.get("/api/engines/attached/status").json()["models_hash"] != body["models_hash"]


def test_status_with_engines_down_never_errors(client, fake):
    fake.omlx_up = fake.ds4_up = False
    body = client.get("/api/engines/attached/status").json()
    assert body["omlx"]["reachable"] is False and body["omlx"]["chat_model_ids"] == []
    assert body["ds4"]["reachable"] is False


def test_sync_seeds_rows_with_live_models(client, fake):
    response = client.post("/api/engines/attached/sync")
    assert response.status_code == 200
    assert response.json()["rows"] == {"omlx": "created", "dwarfstar": "created"}
    omlx = providers_db.get_provider(ATTACHED_OMLX_ID)
    ds4 = providers_db.get_provider(ATTACHED_DS4_ID)
    assert omlx["provider_type"] == "omlx"
    assert omlx["base_url"] == "http://127.0.0.1:8843/v1"
    assert omlx["models"] == fake.chat_ids
    assert EMBED not in omlx["available_models"]
    assert ds4["provider_type"] == "dwarfstar"
    assert ds4["base_url"] == "http://127.0.0.1:8001/v1"
    assert ds4["models"] == ["qwen3.8-flash-next", "qwen3.8-flash-next-chat"]
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
    assert response.json() == {"enabled": False, "deleted": [ATTACHED_OMLX_ID, ATTACHED_DS4_ID]}
    assert providers_db.get_provider(ATTACHED_OMLX_ID) is None
    assert providers_db.get_provider(ATTACHED_DS4_ID) is None
    assert providers_db.get_provider("userrow") is not None
    assert len(fake.calls) == calls_before


def test_omlx_load_resolves_profile_id_to_directory_and_stops_ds4_first(client, fake):
    fake.ds4_loaded, fake.ds4_pid = True, 7
    response = client.post(
        "/api/engines/attached/omlx/load", json = {"model_id": "swift-1.5-27b:fast"}
    )
    assert response.status_code == 200
    assert response.json()["loaded"] == SWIFT_DIR
    assert response.json()["actions"] == ["Stopped DwarfStar"]
    stop = fake.calls.index("ds4 POST /admin/stop")
    load = fake.calls.index(f"omlx POST /admin/api/models/{SWIFT_DIR}/load")
    assert stop < load
    assert SWIFT_DIR in fake.loaded


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


def test_ids_with_colons_only_ever_travel_in_bodies(client, fake):
    client.post("/api/engines/attached/omlx/unload", json = {"model_id": "swift-1.5-27b:low"})
    assert not any(
        ":" in path for path in (r.rsplit(" ", 1)[-1] for r in fake.calls if "unload" in r)
    )


def test_ds4_start_is_a_background_task(client, fake):
    response = client.post("/api/engines/attached/ds4/start")
    assert response.json() == {"starting": True, "loaded": False, "reachable": True}
    assert _wait_for(lambda: "ds4 POST /admin/start" in fake.calls)
    again = client.post("/api/engines/attached/ds4/start").json()
    assert again["loaded"] is True or again["starting"] is True
    assert fake.calls.count("ds4 POST /admin/start") == 1


def test_ds4_start_when_unreachable_does_not_spawn(client, fake):
    fake.ds4_up = False
    assert client.post("/api/engines/attached/ds4/start").json()["reachable"] is False


def test_ds4_stop(client, fake):
    fake.ds4_loaded, fake.ds4_pid = True, 7
    body = client.post("/api/engines/attached/ds4/stop").json()
    assert body["loaded"] is False and body["reachable"] is True


def test_prepare_omlx_stops_ds4(client, fake):
    fake.ds4_loaded, fake.ds4_pid = True, 7
    body = client.post("/api/engines/attached/prepare", json = {"provider": "omlx"}).json()
    assert body["actions"] == ["Stopped DwarfStar"]
    assert fake.ds4_loaded is False


def test_prepare_dwarfstar_prewarms_only_when_enabled(client, fake):
    body = client.post("/api/engines/attached/prepare", json = {"provider": "dwarfstar"}).json()
    assert body["prewarm"] is True and body["starting"] is True
    assert _wait_for(lambda: "ds4 POST /admin/start" in fake.calls)

    fake.ds4_loaded, fake.ds4_pid = False, None
    fake.calls.clear()
    fake.config["value"] = _config(prewarm_ds4_on_select = False)
    body = client.post("/api/engines/attached/prepare", json = {"provider": "dwarfstar"}).json()
    assert body == {"provider": "dwarfstar", "starting": False, "prewarm": False}
    assert fake.calls == []


def test_prepare_rejects_unknown_provider(client):
    assert (
        client.post("/api/engines/attached/prepare", json = {"provider": "ollama"}).status_code == 422
    )


def test_router_is_mounted_before_the_generic_engines_router():
    source = (_BACKEND / "main.py").read_text()
    attached = source.index("attached_engines_router, prefix")
    generic = source.index('app.include_router(engines_router, prefix = "/api/engines"')
    assert attached < generic
