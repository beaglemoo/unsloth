# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached.ds4_client import Ds4Client  # noqa: E402

CONFIG = {"ds4_start_timeout": 120.0, "ds4_idle_seconds": 300.0}

# Shapes captured from a live ds4 on-demand launcher /admin/status.
COLD = {
    "loaded": False,
    "pid": None,
    "uptime_seconds": None,
    "last_activity_seconds_ago": 356059.7,
    "in_flight": 0,
    "idle_seconds_remaining": None,
    "stats": {
        "live": None,
        "last": None,
        "totals": {"requests": 0, "completion_tokens": 0, "errors": 0},
    },
    "config": CONFIG,
}
WARM = {
    "loaded": True,
    "pid": 4242,
    "uptime_seconds": 61.5,
    "last_activity_seconds_ago": 3.0,
    "in_flight": 1,
    "idle_seconds_remaining": None,
    "stats": {
        "live": {"gen_tokens": 40, "gen_tps": 31.25, "elapsed_s": 2.0, "phase": "decode"},
        "last": {
            "model": "qwen3.8-flash-next",
            "completion_tokens": 120,
            "ttft_ms": 850.5,
            "prefill_tps": 400.0,
            "gen_tps": 28.5,
            "duration_s": 5.0,
            "finished_at": 1790000000.0,
        },
        "totals": {"requests": 3, "completion_tokens": 400, "errors": 1},
    },
    "config": {**CONFIG, "ds4_start_timeout": 90.0},
}
STARTING = {**COLD, "pid": 4243}
MODELS = {
    "object": "list",
    "data": [
        {"id": "qwen3.8-flash-next", "loaded": False},
        {"id": "qwen3.8-flash-next-chat", "loaded": False},
        {"id": "qwen3.8-flash-next-reasoner", "loaded": False},
    ],
}


def run(coro):
    return asyncio.run(coro)


def _client(handler):
    return Ds4Client("http://127.0.0.1:8001/", transport = httpx.MockTransport(handler))


def _static(status):
    def handler(request):
        if request.url.path == "/admin/status":
            return httpx.Response(200, json = status)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json = MODELS)
        return httpx.Response(404)

    return handler


def test_cold_status():
    status = run(_client(_static(COLD)).status())
    assert status.reachable is True
    assert (status.loaded, status.starting, status.pid) == (False, False, None)
    assert status.live_tps is None and status.last_gen_tps is None
    assert status.totals == {"requests": 0, "completion_tokens": 0, "errors": 0}
    assert status.start_timeout_s == 120.0


def test_starting_is_pid_without_loaded():
    status = run(_client(_static(STARTING)).status())
    assert (status.loaded, status.starting, status.pid) == (False, True, 4243)


def test_warm_status_reports_throughput():
    status = run(_client(_static(WARM)).status())
    assert (status.loaded, status.starting, status.pid) == (True, False, 4242)
    assert status.uptime_s == 61.5
    assert status.in_flight == 1
    assert status.live_tps == 31.25
    assert status.last_gen_tps == 28.5
    assert status.last_ttft_ms == 850.5
    assert status.start_timeout_s == 90.0


def test_unreachable_never_raises():
    def boom(request):
        raise httpx.ConnectError("refused", request = request)

    client = _client(boom)
    status = run(client.status())
    assert status.reachable is False and status.loaded is False and status.starting is False
    assert run(client.model_ids()) == []
    assert run(client.start()).reachable is False
    assert run(client.stop()).reachable is False


def test_model_ids():
    assert run(_client(_static(COLD)).model_ids()) == [
        "qwen3.8-flash-next",
        "qwen3.8-flash-next-chat",
        "qwen3.8-flash-next-reasoner",
    ]


def test_start_uses_timeout_from_launcher_plus_slack_and_returns_fresh_status():
    seen = {}
    state = {"started": False}

    def handler(request):
        if request.url.path == "/admin/start":
            seen["timeout"] = dict(request.extensions["timeout"])
            state["started"] = True
            return httpx.Response(200, json = {"status": "ok", "loaded": True, "pid": 4242})
        if request.url.path == "/admin/status":
            return httpx.Response(200, json = WARM if state["started"] else COLD)
        return httpx.Response(404)

    client = _client(handler)
    assert run(client.status()).loaded is False
    status = run(client.start())
    assert status.loaded is True and status.error is None
    assert seen["timeout"]["read"] == 120.0 + 15.0
    assert seen["timeout"]["connect"] == 2.0


def test_start_503_reports_error_instead_of_raising():
    def handler(request):
        if request.url.path == "/admin/start":
            return httpx.Response(503, json = {"detail": "ds4 failed to start"})
        return httpx.Response(200, json = COLD)

    status = run(_client(handler).start())
    assert status.reachable is True and status.loaded is False
    assert status.error == "HTTP 503"


def test_stop_posts_and_returns_status():
    posts = []
    state = {"stopped": False}

    def handler(request):
        if request.method == "POST":
            posts.append(request.url.path)
            state["stopped"] = True
            return httpx.Response(200, json = {"status": "ok", "loaded": False})
        return httpx.Response(200, json = COLD if state["stopped"] else WARM)

    status = run(_client(handler).stop())
    assert posts == ["/admin/stop"]
    assert status.loaded is False and status.error is None
