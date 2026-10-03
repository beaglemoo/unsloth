# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.attached import arbiter  # noqa: E402
from core.inference.attached.ds4_client import Ds4Client  # noqa: E402
from core.inference.attached.omlx_client import OmlxClient  # noqa: E402
from utils.attached_engines_settings import AttachedEnginesConfig  # noqa: E402

OMLX_URL = "http://127.0.0.1:8843"
DS4_URL = "http://127.0.0.1:8001"
SWIFT_DIR = "ukisai--Swift-1.5-27B-oQ4e-mtp"
EMBED = "Qwen3-Embedding-8B-4bit-DWQ"


def _config(**over) -> AttachedEnginesConfig:
    base = dict(
        enabled = True,
        omlx_url = OMLX_URL,
        ds4_url = DS4_URL,
        scan_denylist = (),
        arbitrate_local_loads = True,
        prewarm_ds4_on_select = True,
    )
    base.update(over)
    return AttachedEnginesConfig(**base)


class Engines:
    """Both engines behind MockTransports, with one shared, ordered call log."""

    def __init__(
        self,
        *,
        ds4_loaded = True,
        ds4_in_flight = 0,
        omlx_up = True,
        ds4_up = True,
    ):
        self.log: list[str] = []
        self.ds4_loaded = ds4_loaded
        self.ds4_pid: int | None = 4242 if ds4_loaded else None
        self.ds4_in_flight = ds4_in_flight
        self.omlx_up = omlx_up
        self.ds4_up = ds4_up
        self.omlx_loaded = {EMBED, SWIFT_DIR}

    def ds4_handler(self, request: httpx.Request) -> httpx.Response:
        self.log.append(f"ds4 {request.method} {request.url.path}")
        if not self.ds4_up:
            raise httpx.ConnectError("refused", request = request)
        if request.method == "POST" and request.url.path == "/admin/stop":
            self.ds4_loaded, self.ds4_pid, self.ds4_in_flight = False, None, 0
            return httpx.Response(200, json = {"status": "ok", "loaded": False})
        return httpx.Response(
            200,
            json = {
                "loaded": self.ds4_loaded,
                "pid": self.ds4_pid,
                "in_flight": self.ds4_in_flight,
                "stats": {"live": None, "last": None, "totals": {}},
                "config": {"ds4_start_timeout": 120.0},
            },
        )

    def omlx_handler(self, request: httpx.Request) -> httpx.Response:
        self.log.append(f"omlx {request.method} {request.url.path}")
        if not self.omlx_up:
            raise httpx.ConnectError("refused", request = request)
        if request.method == "POST":
            model = request.url.path.split("/")[3]
            self.omlx_loaded.discard(model)
            return httpx.Response(200, json = {})
        models = [
            {
                "id": model,
                "model_path": f"/m/{model}",
                "loaded": model in self.omlx_loaded,
                "pinned": model == EMBED,
                "engine_type": "embedding" if model == EMBED else "vlm",
            }
            for model in (EMBED, SWIFT_DIR)
        ]
        return httpx.Response(200, json = {"models": models})


@pytest.fixture
def engines(monkeypatch):
    state = Engines()
    install(monkeypatch, state)
    return state


def install(
    monkeypatch,
    state: Engines,
    config: AttachedEnginesConfig | None = None,
):
    monkeypatch.setattr(arbiter, "get_config", lambda: config or _config())
    monkeypatch.setattr(
        arbiter,
        "OmlxClient",
        lambda url: OmlxClient(url, transport = httpx.MockTransport(state.omlx_handler)),
    )
    monkeypatch.setattr(
        arbiter,
        "Ds4Client",
        lambda url: Ds4Client(url, transport = httpx.MockTransport(state.ds4_handler)),
    )
    arbiter._notices.clear()
    arbiter._invalidate_ds4_cache()


def run(coro):
    return asyncio.run(coro)


def test_flag_off_makes_zero_http_calls(monkeypatch):
    state = Engines()
    install(monkeypatch, state, _config(enabled = False))
    assert run(arbiter.free_for_local("training")).skipped == "disabled"
    assert run(arbiter.free_for_local("local_load")).skipped == "disabled"
    assert run(arbiter.before_omlx_use()).skipped == "disabled"
    assert arbiter.free_for_local_from_thread("training", asyncio.new_event_loop()).skipped == (
        "disabled"
    )
    assert state.log == []
    assert arbiter.recent_notices() == []


def test_arbitrate_local_loads_off_skips_local_but_not_omlx_use(monkeypatch):
    state = Engines()
    install(monkeypatch, state, _config(arbitrate_local_loads = False))
    assert run(arbiter.free_for_local("local_load")).skipped == "arbitration_off"
    assert state.log == []
    assert run(arbiter.before_omlx_use()).acted is True
    assert state.ds4_loaded is False


def test_ds4_is_stopped_before_omlx_unload_and_pinned_models_go_too(engines):
    result = run(arbiter.free_for_local("local_load"))
    stop = engines.log.index("ds4 POST /admin/stop")
    first_unload = next(i for i, entry in enumerate(engines.log) if "omlx POST" in entry)
    assert stop < first_unload
    assert engines.omlx_loaded == set()
    assert result.acted is True
    assert result.actions[0] == "Stopped DwarfStar"
    assert result.actions[1].startswith("Unloaded oMLX:")
    assert EMBED in result.actions[1]


def test_notice_recorded_with_in_flight_count_and_since_filter(monkeypatch):
    state = Engines(ds4_in_flight = 2)
    install(monkeypatch, state)
    before = time.time() - 1
    run(arbiter.free_for_local("training"))
    notices = arbiter.recent_notices()
    assert len(notices) == 1
    notice = notices[0]
    assert notice["reason"] == "training"
    assert notice["in_flight_killed"] == 2
    assert notice["actions"][0] == "Stopped DwarfStar"
    assert notice["ts"] > before
    assert arbiter.recent_notices(since = notice["ts"]) == []
    assert len(arbiter.recent_notices(since = before)) == 1


def test_nothing_to_free_records_no_notice(monkeypatch):
    state = Engines(ds4_loaded = False)
    state.omlx_loaded = set()
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("local_load"))
    assert result.acted is False and result.actions == ()
    assert arbiter.recent_notices() == []
    assert not any("POST" in entry for entry in state.log)


def test_omlx_unreachable_still_returns_and_ds4_is_still_stopped(monkeypatch):
    state = Engines(omlx_up = False)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("local_load"))
    assert state.ds4_loaded is False
    assert result.actions == ("Stopped DwarfStar",)


def test_both_engines_unreachable_is_a_quiet_noop(monkeypatch):
    state = Engines(omlx_up = False, ds4_up = False)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("training"))
    assert result.acted is False and result.skipped is None
    assert arbiter.recent_notices() == []


def test_unexpected_exception_never_propagates(monkeypatch):
    state = Engines()
    install(monkeypatch, state)

    def boom(url):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(arbiter, "OmlxClient", boom)
    assert run(arbiter.free_for_local("local_load")).skipped == "error"
    monkeypatch.setattr(arbiter, "Ds4Client", boom)
    assert run(arbiter.before_omlx_use()).skipped == "error"


def test_ds4_stop_failure_is_reported_not_raised(monkeypatch):
    state = Engines()
    install(monkeypatch, state)
    original = state.ds4_handler

    def handler(request):
        if request.method == "POST":
            state.log.append("ds4 POST /admin/stop")
            return httpx.Response(500)
        return original(request)

    monkeypatch.setattr(state, "ds4_handler", handler)
    result = run(arbiter.free_for_local("local_load"))
    assert result.actions[0] == "DwarfStar stop failed"
    assert result.in_flight_killed == 0


def test_before_omlx_use_stops_loaded_or_starting_ds4(monkeypatch):
    state = Engines(ds4_loaded = False)
    state.ds4_pid = 99  # spawned, not serving yet: starting
    install(monkeypatch, state)
    result = run(arbiter.before_omlx_use())
    assert result.actions == ("Stopped DwarfStar",)
    assert "ds4 POST /admin/stop" in state.log
    assert arbiter.recent_notices()[0]["reason"] == "omlx_use"
    assert not any(entry.startswith("omlx") for entry in state.log)


def test_before_omlx_use_caches_ds4_status_for_two_seconds(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def three_calls():
        for _ in range(3):
            await arbiter.before_omlx_use()

    run(three_calls())
    assert state.log.count("ds4 GET /admin/status") == 1
    arbiter._ds4_cache = (time.monotonic() - 5, arbiter._ds4_cache[1])
    run(arbiter.before_omlx_use())
    assert state.log.count("ds4 GET /admin/status") == 2


def test_concurrent_callers_are_serialised(monkeypatch):
    state = Engines()
    install(monkeypatch, state)

    async def both():
        return await asyncio.gather(
            arbiter.free_for_local("training"), arbiter.free_for_local("local_load")
        )

    first, second = run(both())
    assert [first.acted, second.acted].count(True) == 1
    assert state.log.count("ds4 POST /admin/stop") == 1


def test_notice_ring_is_bounded_at_twenty(monkeypatch):
    install(monkeypatch, Engines())
    for index in range(30):
        arbiter._record("training", [f"action {index}"], 0)
    notices = arbiter.recent_notices()
    assert len(notices) == 20
    assert notices[0]["actions"] == ["action 10"]


def test_from_thread_runs_on_the_given_loop(monkeypatch):
    state = Engines()
    install(monkeypatch, state)
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def serve():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target = serve, daemon = True)
    thread.start()
    ready.wait(5)
    try:
        result = arbiter.free_for_local_from_thread("training", loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()
    assert result.acted is True
    assert state.ds4_loaded is False
    assert arbiter.recent_notices()[0]["reason"] == "training"


def test_from_loop_thread_refuses_instead_of_deadlocking(monkeypatch):
    state = Engines()
    install(monkeypatch, state)

    async def call():
        return arbiter.free_for_local_from_thread("training", asyncio.get_running_loop())

    assert run(call()).skipped == "error"
    assert state.log == []


def test_hooks_are_wired_at_each_entry_point():
    import inspect

    import routes.inference as inf_mod
    import routes.training as training_mod

    gated = inspect.getsource(inf_mod.load_model_gated)
    assert gated.index("free_for_local(") < gated.index("async with inference_lifecycle_gate()")
    assert "free_for_local(" in inspect.getsource(inf_mod.load_model_for_preview)
    assert "free_for_local(" in inspect.getsource(inf_mod._maybe_auto_switch_model)
    chat = inspect.getsource(inf_mod.produce_openai_chat_completions)
    assert chat.index("ATTACHED_OMLX_ID") < chat.index("before_omlx_use()")
    assert chat.index("before_omlx_use()") < chat.index("return await _proxy_to_external_provider(")
    assert "free_for_local_from_thread(" in inspect.getsource(training_mod.start_training)
