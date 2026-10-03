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
        ds4_starting = False,
        omlx_up = True,
        ds4_up = True,
    ):
        self.log: list[str] = []
        self.ds4_loaded = ds4_loaded
        self.ds4_starting = ds4_starting
        self.ds4_pid: int | None = 4242 if ds4_loaded else None
        self.ds4_in_flight = ds4_in_flight
        self.omlx_up = omlx_up
        self.ds4_up = ds4_up
        self.omlx_loaded = {EMBED, SWIFT_DIR}
        self.omlx_loading: set[str] = set()
        self.holds: dict[str, dict] = {}
        self.hold_bodies: list[dict] = []
        self.next_hold = 0

    def ds4_handler(self, request: httpx.Request) -> httpx.Response:
        self.log.append(f"ds4 {request.method} {request.url.path}")
        if not self.ds4_up:
            raise httpx.ConnectError("refused", request = request)
        if request.method == "POST" and request.url.path == "/admin/hold":
            body = __import__("json").loads(request.content)
            self.hold_bodies.append(body)
            self.next_hold += 1
            hold_id = f"hold-{self.next_hold}"
            self.holds[hold_id] = body
            return httpx.Response(200, json = {"hold_id": hold_id})
        if request.method == "DELETE" and request.url.path.startswith("/admin/hold/"):
            self.holds.pop(request.url.path.rsplit("/", 1)[-1], None)
            return httpx.Response(200, json = {"status": "ok"})
        if request.method == "POST" and request.url.path == "/admin/start":
            if self.holds:
                return httpx.Response(503, headers = {"Retry-After": "15"}, json = {"error": "held"})
            self.ds4_loaded, self.ds4_pid = True, 4242
            return httpx.Response(200, json = {"status": "ok"})
        if request.method == "POST" and request.url.path == "/admin/stop":
            if self.ds4_in_flight or self.ds4_starting:
                return httpx.Response(409, json = {"error": "busy"})
            self.ds4_loaded, self.ds4_pid, self.ds4_in_flight = False, None, 0
            return httpx.Response(200, json = {"status": "ok", "loaded": False})
        return httpx.Response(
            200,
            json = {
                "loaded": self.ds4_loaded,
                "starting": self.ds4_starting,
                "pid": self.ds4_pid,
                "in_flight": self.ds4_in_flight,
                "holds": [dict(id = hold_id, **hold) for hold_id, hold in self.holds.items()],
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
            self.omlx_loading.discard(model)
            return httpx.Response(200, json = {})
        models = [
            {
                "id": model,
                "model_path": f"/m/{model}",
                "loaded": model in self.omlx_loaded,
                "is_loading": model in self.omlx_loading,
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
    async def no_comfyui():
        return None
    monkeypatch.setattr(arbiter, "free_comfyui", no_comfyui)
    arbiter._notices.clear()
    arbiter._leases.clear()


async def _clean_result(coro):
    result = await coro
    if isinstance(result, arbiter.ArbiterResult) and result.lease is not None:
        await result.lease.release()
        await asyncio.sleep(0)
    return result


def run(coro):
    return asyncio.run(_clean_result(coro))


async def _release_and_settle(lease):
    await lease.release()
    await asyncio.sleep(0)


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


def test_busy_ds4_blocks_training_without_killing_the_reply(monkeypatch):
    state = Engines(ds4_in_flight = 2)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("training"))
    assert result.error is not None
    assert state.ds4_in_flight == 2
    assert "ds4 POST /admin/stop" not in state.log
    assert arbiter.recent_notices() == []


def test_nothing_to_free_records_no_notice(monkeypatch):
    state = Engines(ds4_loaded = False)
    state.omlx_loaded = set()
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("local_load"))
    assert result.acted is False and result.actions == ()
    assert arbiter.recent_notices() == []
    assert state.log[0] == "ds4 POST /admin/hold"
    assert "omlx GET /v1/models/status" in state.log
    assert state.log[-1] == "ds4 DELETE /admin/hold/hold-1"


def test_omlx_unreachable_blocks_local_admission(monkeypatch):
    state = Engines(omlx_up = False)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("local_load"))
    assert result.error is not None
    assert state.ds4_loaded is False
    assert "Cannot verify oMLX residency" in result.error


def test_ds4_unreachable_blocks_training(monkeypatch):
    state = Engines(omlx_up = False, ds4_up = False)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("training"))
    assert result.error is not None
    assert "Could not hold DwarfStar" in result.error
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
        if request.method == "POST" and request.url.path == "/admin/stop":
            state.log.append("ds4 POST /admin/stop")
            return httpx.Response(500)
        return original(request)

    monkeypatch.setattr(state, "ds4_handler", handler)
    result = run(arbiter.free_for_local("local_load"))
    assert result.error is not None
    assert "eviction failed" in result.error


def test_starting_ds4_with_null_pid_blocks_omlx_admission(monkeypatch):
    state = Engines(ds4_loaded = False, ds4_starting = True)
    install(monkeypatch, state)
    result = run(arbiter.before_omlx_use())
    assert result.error is not None
    assert "starting or replying" in result.error
    assert "ds4 POST /admin/stop" not in state.log
    assert not any(entry.startswith("omlx") for entry in state.log)


def test_ds4_starting_race_after_hold_blocks_omlx_admission(monkeypatch):
    """A start observed after the hold was created still blocks oMLX use."""
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)
    original = state.ds4_handler

    def handler(request):
        if request.method == "POST" and request.url.path == "/admin/hold":
            response = original(request)
            state.ds4_starting = True
            return response
        return original(request)

    monkeypatch.setattr(state, "ds4_handler", handler)
    result = run(arbiter.before_omlx_use())
    assert result.error is not None
    assert state.log[:2] == ["ds4 POST /admin/hold", "ds4 GET /admin/status"]


def test_concurrent_callers_are_serialised(monkeypatch):
    state = Engines()
    install(monkeypatch, state)

    async def both():
        first = await arbiter.before_omlx_use()
        second = await arbiter.free_for_local("training")
        await first.lease.release()
        await asyncio.sleep(0)
        return first, second

    first, second = asyncio.run(both())
    assert first.lease is not None
    assert second.error is not None
    assert state.log.count("ds4 POST /admin/stop") == 1


@pytest.mark.parametrize("reason", ["training", "local_load"])
def test_hold_is_taken_before_residency_check_and_blocks_ds4_start(monkeypatch, reason):
    state = Engines(ds4_loaded = False)
    state.omlx_loaded = set()
    install(monkeypatch, state)

    async def admit_then_try_cold_start():
        result = await arbiter.free_for_local(reason)
        assert result.lease is not None
        async with httpx.AsyncClient(
            base_url = DS4_URL, transport = httpx.MockTransport(state.ds4_handler)
        ) as client:
            response = await client.post("/admin/start")
        await result.lease.release()
        await asyncio.sleep(0)
        return response

    response = asyncio.run(admit_then_try_cold_start())
    assert state.log[0] == "ds4 POST /admin/hold"
    assert state.hold_bodies == [{"reason": f"Studio {reason}", "ttl_s": 120}]
    assert response.status_code == 503
    assert state.holds == {}


def test_hold_renewal_creates_replacement_before_deleting_old_hold(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def renew():
        lease = await arbiter.HoldLease.create(DS4_URL, "training")
        await lease.renew()
        await lease.release()
        await asyncio.sleep(0)

    asyncio.run(renew())
    assert state.hold_bodies == [
        {"reason": "Studio training", "ttl_s": 120},
        {"reason": "Studio training", "ttl_s": 120},
    ]
    assert state.log == [
        "ds4 POST /admin/hold",
        "ds4 POST /admin/hold",
        "ds4 DELETE /admin/hold/hold-1",
        "ds4 DELETE /admin/hold/hold-2",
    ]
    assert state.holds == {}


def test_lifetime_watch_releases_hold_when_residency_is_already_gone(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def watch():
        lease = await arbiter.HoldLease.create(DS4_URL, "local_load")
        await lease.watch_until(lambda: False)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return lease

    lease = asyncio.run(watch())
    assert lease.closed is True
    assert state.holds == {}
    assert state.log[-1] == "ds4 DELETE /admin/hold/hold-1"


def test_failed_local_admission_releases_its_hold(monkeypatch):
    state = Engines(ds4_loaded = False, omlx_up = False)
    install(monkeypatch, state)
    result = run(arbiter.free_for_local("local_load"))
    assert result.error is not None
    assert state.holds == {}
    assert state.log[-1] == "ds4 DELETE /admin/hold/hold-1"


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
        if result.lease is not None:
            asyncio.run_coroutine_threadsafe(_release_and_settle(result.lease), loop).result(5)
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

    # The local-load eviction lives only inside the real-load path, never at the entry points: those
    # also see refused loads and no-op reloads.
    for entry in (
        inf_mod.load_model_gated,
        inf_mod.load_model_for_preview,
        inf_mod._maybe_auto_switch_model,
    ):
        assert "free_for_local(" not in inspect.getsource(entry), entry.__name__
    chat = inspect.getsource(inf_mod.produce_openai_chat_completions)
    assert chat.index("ATTACHED_OMLX_ID") < chat.index("_attached_omlx_proxy")
    proxy = inspect.getsource(inf_mod._attached_omlx_proxy)
    assert "before_omlx_use" in proxy
    assert "lease.release" in proxy
    assert "free_for_local_from_thread(" in inspect.getsource(training_mod.start_training)


def test_local_load_eviction_sits_after_admission_and_the_noop_check_in_both_load_paths():
    import inspect

    import routes.inference as inf_mod

    source = inspect.getsource(inf_mod._load_model_impl)
    marker = "await _admit_attached_local_load()"
    assert source.count(marker) == 2
    gguf_hook = source.index(marker)
    standard_hook = source.index(marker, gguf_hook + 1)
    # GGUF: after the no-op reuse return and the confirmed-reload cancel, before any teardown.
    assert source.index("return reused") < gguf_hook
    assert source.rindex("on_reload_confirmed(cancel = True)", 0, gguf_hook) > source.index(
        "return reused"
    )
    assert gguf_hook < source.index(
        "unsloth_backend.unload_model, unsloth_backend.active_model_name"
    )
    assert gguf_hook < source.index("_run_gguf_load_attempt(")
    # Standard path: after its own already_loaded return, before the llama teardown.
    assert source.index('status = "already_loaded"') < standard_hook
    assert source.rindex("on_reload_confirmed(cancel = True)", 0, standard_hook) > gguf_hook
    assert standard_hook < source.index("await _unload_llama_before_standard_load(llama_backend)")


@pytest.mark.parametrize("failure", [409, 503])
def test_a_refused_load_does_not_free_the_engines(monkeypatch, failure):
    """A load refused at the gate (e.g. active generations, 409) must leave ds4 and oMLX alone."""
    from fastapi import HTTPException

    import routes.inference as inf_mod
    from models.inference import LoadRequest

    state = Engines()
    install(monkeypatch, state)

    async def refuse(*args, **kwargs):
        raise HTTPException(status_code = failure, detail = "refused")

    monkeypatch.setattr(inf_mod, "_run_tracked_load_model_impl", refuse)
    with pytest.raises(HTTPException) as err:
        run(inf_mod.load_model_gated(LoadRequest(model_path = "org/A"), object(), "tester"))
    assert err.value.status_code == failure
    assert state.log == []
    assert state.ds4_loaded is True
    assert arbiter.recent_notices() == []


def test_training_eviction_and_chat_prewarm_interleave_under_hold(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def exercise():
        entered = asyncio.Event()
        resume = asyncio.Event()
        original = arbiter.OmlxClient

        class DelayedUnload:
            def __init__(self, url):
                self.client = original(url)

            async def unload_all(self):
                entered.set()
                await resume.wait()
                return await self.client.unload_all()

        monkeypatch.setattr(arbiter, "OmlxClient", DelayedUnload)
        admission = asyncio.create_task(arbiter.free_for_local("training"))
        await entered.wait()
        # Precisely in the eviction/pre-spawn gap, a chat selection prewarms ds4.
        async with httpx.AsyncClient(base_url = DS4_URL, transport = httpx.MockTransport(state.ds4_handler)) as client:
            response = await client.post("/admin/start")
        assert response.status_code == 503
        assert state.holds and not state.ds4_loaded
        resume.set()
        result = await admission
        assert result.error is None
        await result.lease.release()

    asyncio.run(exercise())


def test_cold_start_before_hold_blocks_omlx_even_without_child_pid(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def exercise():
        entered = asyncio.Event()
        resume = asyncio.Event()
        original = state.ds4_handler

        async def delayed_hold(request):
            if request.url.path == "/admin/hold" and request.method == "POST":
                entered.set()
                await resume.wait()
            return original(request)

        monkeypatch.setattr(state, "ds4_handler", delayed_hold)
        admission = asyncio.create_task(arbiter.before_omlx_use())
        await entered.wait()
        # A cold start already owns admission, before its first awaited eviction.
        state.ds4_starting = True
        state.ds4_pid = None
        resume.set()
        result = await admission
        assert result.error is not None and "starting" in result.error
        assert result.lease is None
        assert not state.holds
        assert not any(entry.startswith("omlx") for entry in state.log)

    asyncio.run(exercise())


def test_manual_unload_releases_only_local_holds_when_residency_is_clear(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def exercise():
        local = await arbiter.HoldLease.create(DS4_URL, "local_load")
        training = await arbiter.HoldLease.create(DS4_URL, "training")
        await arbiter.release_local_if_idle(lambda: True)
        assert not local.closed
        await arbiter.release_local_if_idle(lambda: False)
        assert local.closed and not training.closed
        await training.release()

    asyncio.run(exercise())


def test_renewal_scheduler_runs_at_sixty_seconds_and_retries_without_dropping_hold(monkeypatch):
    state = Engines(ds4_loaded = False)
    install(monkeypatch, state)

    async def exercise():
        first_sleep = asyncio.Event()
        allow_renewal = asyncio.Event()
        retried = asyncio.Event()
        durations = []
        original_sleep = asyncio.sleep

        async def sleep(delay):
            durations.append(delay)
            if len(durations) == 1:
                first_sleep.set()
                await allow_renewal.wait()
            else:
                retried.set()
                await asyncio.Event().wait()

        monkeypatch.setattr(arbiter.asyncio, "sleep", sleep)
        lease = await arbiter.HoldLease.create(DS4_URL, "training")
        await first_sleep.wait()
        original_id = lease.hold_id
        original_handler = state.ds4_handler

        def fail_replacement(request):
            if request.method == "POST" and request.url.path == "/admin/hold":
                return httpx.Response(503)
            return original_handler(request)

        monkeypatch.setattr(state, "ds4_handler", fail_replacement)
        allow_renewal.set()
        await retried.wait()
        assert durations == [60, 5]
        assert lease.hold_id == original_id and original_id in state.holds
        assert lease.error is not None
        await lease.release()
        await original_sleep(0)

    asyncio.run(exercise())
