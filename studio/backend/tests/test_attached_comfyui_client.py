# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached.comfyui_client import (
    ComfyGraphError,
    ComfyQueue,
    ComfyuiClient,
    ComfyuiError,
    parse_queue,
    system_version,
)


def _client(handler) -> ComfyuiClient:
    return ComfyuiClient("http://comfy/", transport = httpx.MockTransport(handler))


def run(coro):
    return asyncio.run(coro)


def _item(number, prompt_id, extra = None):
    return [number, prompt_id, {"1": {}}, extra if extra is not None else {}, ["9"]]


def test_queue_parses_ids_markers_and_numbers():
    body = {
        "queue_running": [_item(3, "a", {"unsloth": {"template": "t"}})],
        "queue_pending": [_item(4, "b"), _item(5, "c", {"unsloth": "not a dict"})],
    }

    def handler(request):
        assert request.url.path == "/queue" and request.extensions["timeout"]["read"] == 3
        return httpx.Response(200, json = body)

    queue = run(_client(handler).queue())
    assert queue.reachable and not queue.malformed and queue.busy
    assert queue.running_ids == ("a",) and queue.pending_ids == ("b", "c")
    assert queue.studio == {"a": {"template": "t"}}
    assert queue.numbers == {"a": 3, "b": 4, "c": 5}


def test_empty_queue_is_idle():
    queue = run(_client(lambda r: httpx.Response(200, json = {"queue_running": [], "queue_pending": []})).queue())
    assert queue.reachable and not queue.busy


@pytest.mark.parametrize(
    "body",
    [
        [],
        {},
        {"queue_running": [], "queue_pending": "x"},
        {"queue_running": [[1]], "queue_pending": []},
        {"queue_running": [[1, 5]], "queue_pending": []},
        {"queue_running": ["x"], "queue_pending": []},
    ],
)
def test_malformed_queue_is_unreachable_for_display_and_busy_for_admission(body):
    queue = run(_client(lambda r: httpx.Response(200, json = body)).queue())
    assert not queue.reachable and queue.malformed and queue.busy


def test_non_json_queue_is_uncertain_so_busy_for_admission():
    queue = run(_client(lambda r: httpx.Response(200, content = b"nope")).queue())
    assert queue == ComfyQueue(reachable = False, uncertain = True)
    assert queue.busy and queue.state == "unknown"


def _faulty(failure):
    def handler(request):
        if failure == "connect":
            raise httpx.ConnectError("offline")
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        if failure == "connect-timeout":
            raise httpx.ConnectTimeout("slow")
        if failure == "protocol":
            raise httpx.RemoteProtocolError("reset")
        if failure == "json":
            return httpx.Response(200, content = b"<html>")
        return httpx.Response(503 if failure == "http" else 500)

    return handler


def test_connection_refused_is_confirmed_not_running():
    queue = run(_client(_faulty("connect")).queue())
    assert not queue.reachable and not queue.uncertain and not queue.malformed and not queue.busy
    assert queue.state == "down"


@pytest.mark.parametrize("failure", ["timeout", "connect-timeout", "protocol", "http", "http500", "json"])
def test_a_server_that_may_be_running_but_gave_no_queue_is_uncertain_and_busy(failure):
    queue = run(_client(_faulty(failure)).queue())
    assert not queue.reachable and queue.uncertain and queue.busy
    assert queue.state == "unknown"


def test_queue_states():
    idle = run(_client(lambda r: httpx.Response(200, json = {"queue_running": [], "queue_pending": []})).queue())
    busy = run(_client(lambda r: httpx.Response(200, json = {"queue_running": [[1, "a"]], "queue_pending": []})).queue())
    assert (idle.state, busy.state) == ("idle", "busy")
    assert run(_client(lambda r: httpx.Response(200, json = [])).queue()).state == "unknown"


def test_parse_queue_accepts_tuples_of_extra_data_less_entries():
    queue = parse_queue({"queue_running": [[1, "a"]], "queue_pending": []})
    assert queue.running_ids == ("a",) and queue.studio == {}


def test_system_stats_and_version():
    stats = {"system": {"comfyui_version": "0.39.0", "ram_free": 5}, "devices": [{"name": "mps"}]}
    client = _client(lambda r: httpx.Response(200, json = stats))
    assert run(client.system_stats()) == stats
    assert run(client.version()) == "0.39.0"


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, json = [1]), httpx.Response(200, content = b"x")])
def test_system_stats_degrades_to_none(response):
    client = _client(lambda r: response)
    assert run(client.system_stats()) is None
    assert run(client.version()) is None


def test_version_missing_or_blank():
    assert system_version({"system": {}}) is None
    assert system_version({"system": {"comfyui_version": ""}}) is None
    assert system_version(None) is None


def test_folders_and_models():
    def handler(request):
        if request.url.path == "/models":
            return httpx.Response(200, json = ["vae", "loras", 3])
        if request.url.path == "/models/loras":
            return httpx.Response(200, json = ["a.safetensors", "b.safetensors"])
        if request.url.path == "/models/gone":
            return httpx.Response(404)
        if request.url.path == "/models/bad":
            return httpx.Response(200, json = {"x": 1})
        return httpx.Response(500)

    client = _client(handler)
    assert run(client.folders()) == ["vae", "loras"]
    assert run(client.models("loras")) == ["a.safetensors", "b.safetensors"]
    assert run(client.models("gone")) == []
    with pytest.raises(ComfyuiError, match = "unexpected"):
        run(client.models("bad"))
    with pytest.raises(ComfyuiError, match = "HTTP 500"):
        run(client.models("other"))


def test_models_unreachable_raises():
    def handler(request):
        raise httpx.ConnectError("offline")

    with pytest.raises(ComfyuiError, match = "unreachable"):
        run(_client(handler).models("vae"))
    with pytest.raises(ComfyuiError, match = "unreachable"):
        run(_client(handler).folders())


def test_control_posts():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, json.loads(request.content)))
        assert request.extensions["timeout"]["read"] == 3
        return httpx.Response(200)

    client = _client(handler)
    run(client.free())
    run(client.free(unload_models = False, free_memory = True))
    run(client.interrupt("p1"))
    run(client.interrupt())
    run(client.delete_pending(["a", "b"]))
    run(client.clear_pending())
    assert seen == [
        ("POST", "/free", {"unload_models": True, "free_memory": True}),
        ("POST", "/free", {"unload_models": False, "free_memory": True}),
        ("POST", "/interrupt", {"prompt_id": "p1"}),
        ("POST", "/interrupt", {}),
        ("POST", "/queue", {"delete": ["a", "b"]}),
        ("POST", "/queue", {"clear": True}),
    ]


def test_control_post_errors_raise():
    with pytest.raises(ComfyuiError, match = "HTTP 500"):
        run(_client(lambda r: httpx.Response(500)).free())

    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(ComfyuiError, match = "unreachable"):
        run(_client(handler).interrupt("x"))


def test_submit_success_sends_graph_client_id_and_extra():
    def handler(request):
        assert request.url.path == "/prompt" and request.extensions["timeout"]["read"] == 30
        body = json.loads(request.content)
        assert body == {"prompt": {"1": {"class_type": "X"}}, "client_id": "cid", "extra_data": {"unsloth": {"template": "t"}}}
        return httpx.Response(200, json = {"prompt_id": "pid", "number": 7, "node_errors": {}})

    graph = {"1": {"class_type": "X"}}
    assert run(_client(handler).submit(graph, "cid", {"unsloth": {"template": "t"}})) == ("pid", 7)


def test_submit_without_extra_omits_extra_data():
    def handler(request):
        assert "extra_data" not in json.loads(request.content)
        return httpx.Response(200, json = {"prompt_id": "pid"})

    assert run(_client(handler).submit({}, "cid")) == ("pid", 0)


def test_submit_400_raises_graph_error_with_node_errors():
    body = {
        "error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
        "node_errors": {"6": {"errors": [{"type": "value_not_in_list"}]}},
    }
    with pytest.raises(ComfyGraphError) as caught:
        run(_client(lambda r: httpx.Response(400, json = body)).submit({}, "cid"))
    assert "failed validation" in str(caught.value)
    assert caught.value.node_errors == body["node_errors"]
    assert isinstance(caught.value, ComfyuiError)


@pytest.mark.parametrize("response", [httpx.Response(400, content = b"junk"), httpx.Response(400, json = {"error": "plain"})])
def test_submit_400_with_odd_bodies_still_raises_graph_error(response):
    with pytest.raises(ComfyGraphError) as caught:
        run(_client(lambda r: response).submit({}, "cid"))
    assert caught.value.node_errors == {}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, content = b"x"),
        httpx.Response(200, json = {}),
        httpx.Response(200, json = {"prompt_id": ""}),
        httpx.Response(200, json = []),
    ],
)
def test_submit_other_failures_raise_comfyui_error(response):
    with pytest.raises(ComfyuiError) as caught:
        run(_client(lambda r: response).submit({}, "cid"))
    assert not isinstance(caught.value, ComfyGraphError)


def test_submit_unreachable_raises():
    def handler(request):
        raise httpx.ConnectError("offline")

    with pytest.raises(ComfyuiError, match = "unreachable"):
        run(_client(handler).submit({}, "cid"))


def test_history_returns_entry_or_none():
    def handler(request):
        if request.url.path == "/history/p1":
            return httpx.Response(200, json = {"p1": {"outputs": {}}})
        return httpx.Response(200, json = {})

    client = _client(handler)
    assert run(client.history("p1")) == {"outputs": {}}
    assert run(client.history("p2")) is None


def test_view_returns_bytes_and_passes_params():
    def handler(request):
        assert request.url.path == "/view"
        assert dict(request.url.params) == {"filename": "a b.png", "subfolder": "", "type": "temp"}
        return httpx.Response(200, content = b"\x89PNG")

    assert run(_client(handler).view("a b.png", "", "temp")) == b"\x89PNG"
    with pytest.raises(ComfyuiError, match = "HTTP 404"):
        run(_client(lambda r: httpx.Response(404)).view("x"))


def test_object_info():
    assert run(_client(lambda r: httpx.Response(200, json = {"KSampler": {}})).object_info()) == {"KSampler": {}}
    with pytest.raises(ComfyuiError, match = "unexpected"):
        run(_client(lambda r: httpx.Response(200, json = [])).object_info())


def test_client_ignores_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9")
    monkeypatch.setenv("ALL_PROXY", "http://proxy.invalid:9")
    assert run(_client(lambda r: httpx.Response(200, json = {"queue_running": [], "queue_pending": []})).queue()).reachable
