# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ComfyUI as a peer in the memory arbiter, in both directions. Every ComfyUI is a MockTransport."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import AttachedEngineBusy, AttachedEngineError, arbiter
from core.inference.attached.comfyui_client import ComfyuiClient
from utils.attached_engines_settings import DEFAULT_CONFIG

OWN = "http://127.0.0.1:8844"
PEER = "http://127.0.0.1:8188"


class World:
    """Two ComfyUIs (Studio's and StoryPress's), oMLX, and what Studio is running."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []
        self.comfy = {
            OWN: dict(up = True, running = 0, pending = 0, malformed = False),
            PEER: dict(up = True, running = 0, pending = 0, malformed = False),
        }
        self.omlx_calls: list[str] = []
        self.omlx_error: Exception | None = None
        self.omlx_loaded = ["chat"]
        self.omlx_reachable = True
        self.training = False
        self.local = False
        self.media: str | None = None

    def handler(self, url):
        state = self.comfy[url]

        def handle(request: httpx.Request) -> httpx.Response:
            self.calls.append((url, request.method, request.url.path))
            if not state["up"]:
                raise httpx.ConnectError("refused", request = request)
            if request.url.path == "/queue" and request.method == "GET":
                if state["malformed"]:
                    return httpx.Response(200, json = {"queue_running": "?"})
                item = lambda n: [n, f"p{n}", {}, {}, []]  # noqa: E731
                return httpx.Response(
                    200,
                    json = {
                        "queue_running": [item(i) for i in range(state["running"])],
                        "queue_pending": [item(10 + i) for i in range(state["pending"])],
                    },
                )
            if request.url.path == "/free":
                return httpx.Response(200)
            return httpx.Response(404)

        return handle

    def posts(self, url):
        return [c for c in self.calls if c[0] == url and c[1] == "POST"]

    def probes(self, url):
        return [c for c in self.calls if c[0] == url and c[1] == "GET"]


@pytest.fixture
def world(monkeypatch, tmp_path):
    w = World()
    config = {"value": replace(DEFAULT_CONFIG, enabled = True)}
    w.config = config
    monkeypatch.delenv("STUDIO_COMFYUI_URL", raising = False)
    monkeypatch.setattr(arbiter, "get_config", lambda: config["value"])
    arbiter._notices.clear()
    arbiter._busy_cache.clear()

    def make_comfy(url, *, transport = None):
        return ComfyuiClient(url, transport = transport or httpx.MockTransport(w.handler(url.rstrip("/"))))

    monkeypatch.setattr(arbiter, "ComfyuiClient", make_comfy)

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            w.omlx_calls.append("unload_all")
            if w.omlx_error:
                raise w.omlx_error
            return list(w.omlx_loaded)

        async def status(self):
            return types.SimpleNamespace(reachable = w.omlx_reachable)

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(arbiter, "_training_active", lambda: w.training)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: w.local)
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: w.media)
    monkeypatch.setenv("UNSLOTH_ENGINES_HOME", str(tmp_path))
    return w


def run(coro):
    return asyncio.run(coro)


def set_config(world, **over):
    world.config["value"] = replace(world.config["value"], **over)


# --- free_comfyui -----------------------------------------------------------------------------


def test_free_comfyui_frees_own_and_peer_when_idle(world):
    assert run(arbiter.free_comfyui()) == [OWN, PEER]
    assert [c[2] for c in world.posts(OWN)] == ["/free"] and [c[2] for c in world.posts(PEER)] == ["/free"]


def test_free_comfyui_skips_busy_unreachable_and_unreadable(world):
    world.comfy[OWN]["running"] = 1
    world.comfy[PEER]["up"] = False
    assert run(arbiter.free_comfyui()) == []
    world.comfy[OWN].update(running = 0, malformed = True)
    world.comfy[PEER].update(up = True, pending = 2)
    assert run(arbiter.free_comfyui()) == []
    assert world.posts(OWN) == [] and world.posts(PEER) == []


def test_free_comfyui_peer_down_still_frees_own(world):
    world.comfy[PEER]["up"] = False
    assert run(arbiter.free_comfyui()) == [OWN]


def test_free_comfyui_does_not_probe_a_url_twice(world):
    set_config(world, comfyui_peer_urls = (OWN, PEER))
    assert run(arbiter.free_comfyui()) == [OWN, PEER]
    assert len(world.probes(OWN)) == 1


def test_free_comfyui_env_override_is_the_only_target(world, monkeypatch):
    monkeypatch.setenv("STUDIO_COMFYUI_URL", PEER + "/")
    assert run(arbiter.free_comfyui()) == [PEER]
    assert world.calls and all(c[0] == PEER for c in world.calls)


def test_free_comfyui_empty_env_disables_it(world, monkeypatch):
    monkeypatch.setenv("STUDIO_COMFYUI_URL", "")
    assert run(arbiter.free_comfyui()) == []
    assert world.calls == []


def test_peers_only_ever_get_a_queue_read_and_an_idle_free(world):
    run(arbiter.free_comfyui())
    run(arbiter.before_comfyui_job())
    assert {(c[1], c[2]) for c in world.calls if c[0] == PEER} <= {("GET", "/queue"), ("POST", "/free")}


# --- Studio local work (training, chat loads, image and video loads) --------------------------


def test_local_load_is_refused_while_comfyui_generates(world):
    world.comfy[OWN]["running"] = 1
    result = run(arbiter.free_for_local("local_load"))
    with pytest.raises(arbiter.AttachedAdmissionError, match = "ComfyUI is generating"):
        result.require_clear()
    assert world.omlx_calls == [] and world.posts(OWN) == [] and world.posts(PEER) == []


def test_training_start_is_refused_while_comfyui_is_queued(world):
    world.comfy[OWN]["pending"] = 1
    result = run(arbiter.free_for_local("training"))
    assert result.error and "ComfyUI is generating" in result.error


def test_unreadable_own_queue_counts_as_busy_for_a_local_load(world):
    world.comfy[OWN]["malformed"] = True
    assert run(arbiter.free_for_local("local_load")).error


def test_local_load_proceeds_when_comfyui_is_down(world):
    world.comfy[OWN]["up"] = False
    world.comfy[PEER]["up"] = False
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert result.actions == ("Unloaded oMLX: chat",)


def test_local_load_frees_comfyui_both_and_unloads_omlx(world):
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert world.omlx_calls == ["unload_all"]
    assert [len(world.posts(u)) for u in (OWN, PEER)] == [1, 1]
    assert result.actions == ("Unloaded oMLX: chat", f"Freed ComfyUI memory: {OWN}, {PEER}")
    assert arbiter.recent_notices()[0]["actions"] == list(result.actions)


def test_a_busy_peer_does_not_refuse_a_local_load_it_is_just_not_freed(world):
    world.comfy[PEER]["running"] = 1
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert world.posts(PEER) == [] and len(world.posts(OWN)) == 1


def test_arbitrate_comfyui_off_never_refuses_a_local_load(world):
    set_config(world, arbitrate_comfyui = False)
    world.comfy[OWN]["running"] = 1
    run(arbiter.free_for_local("local_load")).require_clear()
    assert world.omlx_calls == ["unload_all"]


def test_a_switched_off_omlx_that_is_down_does_not_block_local_work(world):
    (Path(world_home := __import__("os").environ["UNSLOTH_ENGINES_HOME"]) / "desktop.json").write_text(
        json.dumps({"engines_enabled": True, "helpers": {"omlx": False}})
    )
    world.omlx_error = AttachedEngineError("Cannot verify oMLX residency: engine is unreachable.")
    world.omlx_reachable = False
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert world_home and not any("Unloaded oMLX" in a for a in result.actions)


def test_a_wanted_omlx_that_is_down_still_blocks_local_work(world):
    (Path(__import__("os").environ["UNSLOTH_ENGINES_HOME"]) / "desktop.json").write_text(
        json.dumps({"engines_enabled": True})
    )
    world.omlx_error = AttachedEngineError("Cannot verify oMLX residency: engine is unreachable.")
    world.omlx_reachable = False
    assert run(arbiter.free_for_local("local_load")).error


def test_an_oMLX_that_is_up_but_failing_to_unload_blocks_even_when_switched_off(world):
    (Path(__import__("os").environ["UNSLOTH_ENGINES_HOME"]) / "desktop.json").write_text(
        json.dumps({"engines_enabled": True, "helpers": {"omlx": False}})
    )
    world.omlx_error = AttachedEngineError("oMLX models are still loaded or loading: x")
    world.omlx_reachable = True
    assert run(arbiter.free_for_local("local_load")).error


# --- oMLX use and load ------------------------------------------------------------------------


def test_omlx_use_is_refused_while_comfyui_generates(world):
    world.comfy[OWN]["running"] = 1
    result = run(arbiter.before_omlx_use())
    with pytest.raises(arbiter.AttachedAdmissionError, match = "ComfyUI is generating"):
        result.require_clear()
    assert result.skipped == "error"


def test_omlx_use_is_refused_for_queued_work_and_unreadable_queue(world):
    world.comfy[OWN]["pending"] = 1
    assert run(arbiter.before_omlx_use()).error
    arbiter._busy_cache.clear()
    world.comfy[OWN].update(pending = 0, malformed = True)
    assert run(arbiter.before_omlx_use()).error


def test_omlx_use_is_not_refused_when_comfyui_arbitration_is_off(world):
    set_config(world, arbitrate_comfyui = False)
    world.comfy[OWN]["running"] = 1
    run(arbiter.before_omlx_use()).require_clear()
    assert world.calls == []


def test_omlx_use_ignores_a_busy_peer_and_a_down_comfyui(world):
    world.comfy[PEER]["running"] = 1
    run(arbiter.before_omlx_use()).require_clear()
    world.comfy[OWN]["up"] = False
    arbiter._busy_cache.clear()
    run(arbiter.before_omlx_use()).require_clear()
    assert world.posts(OWN) == [] and world.posts(PEER) == []
    assert all(c[0] == OWN for c in world.calls)


def test_omlx_use_probes_comfyui_at_most_once_a_second(world, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(arbiter.time, "monotonic", lambda: clock[0])
    run(arbiter.before_omlx_use())
    run(arbiter.before_omlx_use())
    assert len(world.probes(OWN)) == 1
    clock[0] += 1.5
    run(arbiter.before_omlx_use())
    assert len(world.probes(OWN)) == 2


def test_omlx_use_does_not_free_comfyui(world):
    run(arbiter.before_omlx_use()).require_clear()
    assert world.posts(OWN) == [] and world.posts(PEER) == []


def test_omlx_use_keeps_the_training_refusal_ahead_of_comfyui(world):
    world.training = True
    world.comfy[OWN]["running"] = 1
    assert "Studio is training" in run(arbiter.before_omlx_use()).error


def test_omlx_load_frees_idle_comfyui(world):
    result = run(arbiter.before_omlx_load())
    result.require_clear()
    assert result.acted and result.actions == (f"Freed ComfyUI memory: {OWN}, {PEER}",)
    assert arbiter.recent_notices()[0]["reason"] == "omlx_load"
    assert world.omlx_calls == []


def test_omlx_load_is_refused_while_comfyui_generates_and_frees_nothing(world):
    world.comfy[OWN]["running"] = 1
    result = run(arbiter.before_omlx_load())
    assert result.error and "ComfyUI is generating" in result.error
    assert world.posts(OWN) == [] and world.posts(PEER) == []


def test_omlx_load_frees_the_idle_peer_when_own_is_down(world):
    world.comfy[OWN]["up"] = False
    assert run(arbiter.before_omlx_load()).actions == (f"Freed ComfyUI memory: {PEER}",)


def test_omlx_load_with_comfyui_arbitration_off_frees_nothing(world):
    set_config(world, arbitrate_comfyui = False)
    result = run(arbiter.before_omlx_load())
    result.require_clear()
    assert not result.acted and world.calls == []


def test_omlx_load_with_nothing_to_free_is_a_plain_admission(world):
    world.comfy[OWN]["up"] = False
    world.comfy[PEER]["up"] = False
    result = run(arbiter.before_omlx_load())
    result.require_clear()
    assert not result.acted and arbiter.recent_notices() == []


def test_omlx_load_disabled_is_skipped(world):
    set_config(world, enabled = False)
    assert run(arbiter.before_omlx_load()).skipped == "disabled"
    assert world.calls == []


def test_omlx_load_refused_for_a_local_model_never_frees(world):
    world.local = True
    assert run(arbiter.before_omlx_load()).error
    assert world.calls == []


# --- a Studio ComfyUI job ---------------------------------------------------------------------


def test_comfyui_job_refusal_order(world):
    world.training = world.local = True
    world.media = "has an image model loaded"
    world.comfy[PEER]["running"] = 1
    for expected in ("is training", "has a local model loaded", "has an image model loaded"):
        result = run(arbiter.before_comfyui_job())
        assert expected in (result.error or ""), expected
        if expected == "is training":
            world.training = False
        elif expected == "has a local model loaded":
            world.local = False
        else:
            world.media = None
    result = run(arbiter.before_comfyui_job())
    assert "generating" in result.error and PEER in result.error
    assert world.omlx_calls == [] and world.posts(OWN) == [] and world.posts(PEER) == []


def test_comfyui_job_refuses_for_video_resident(world):
    world.media = "has a video model loaded"
    assert "video model" in run(arbiter.before_comfyui_job()).error


def test_comfyui_job_unloads_omlx_and_frees_idle_peers_but_not_its_own(world):
    result = run(arbiter.before_comfyui_job())
    result.require_clear()
    assert world.omlx_calls == ["unload_all"]
    assert world.posts(OWN) == [] and len(world.posts(PEER)) == 1
    assert result.actions == ("Unloaded oMLX: chat", f"Freed ComfyUI memory: {PEER}")
    assert arbiter.recent_notices()[0]["reason"] == "comfyui_job"


def test_comfyui_job_with_nothing_to_do_records_nothing(world):
    world.omlx_loaded = []
    world.comfy[PEER]["up"] = False
    result = run(arbiter.before_comfyui_job())
    result.require_clear()
    assert not result.acted and arbiter.recent_notices() == []


def test_comfyui_job_oMLX_busy_becomes_an_error(world):
    world.omlx_error = AttachedEngineBusy("oMLX is busy serving another client; retry shortly.")
    result = run(arbiter.before_comfyui_job())
    with pytest.raises(arbiter.AttachedAdmissionError, match = "busy serving another client"):
        result.require_clear()
    assert world.posts(PEER) == []


def test_comfyui_job_oMLX_eviction_failure_becomes_an_error(world):
    world.omlx_error = RuntimeError("oMLX remains resident")
    assert "remains resident" in run(arbiter.before_comfyui_job()).error


def test_comfyui_job_with_arbitration_off_or_disabled_is_skipped(world):
    set_config(world, arbitrate_comfyui = False)
    assert run(arbiter.before_comfyui_job()).skipped == "arbitration_off"
    set_config(world, arbitrate_comfyui = True, enabled = False)
    assert run(arbiter.before_comfyui_job()).skipped == "disabled"
    assert world.calls == [] and world.omlx_calls == []


def test_comfyui_job_does_not_refuse_for_its_own_busy_queue(world):
    world.comfy[OWN]["running"] = 1
    run(arbiter.before_comfyui_job()).require_clear()


# --- Studio media residency probe -------------------------------------------------------------


def test_media_probe_reads_loaded_backends_only(monkeypatch):
    for name in ("core.inference.diffusion", "core.inference.sd_cpp_backend", "core.inference.video"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(
            _diffusion_backend = None, _sd_cpp_backend = None, _backend = None))
    assert arbiter._studio_media_resident() is None

    class Backend:
        def __init__(self, loaded):
            self.loaded = loaded

        def status(self):
            if self.loaded is None:
                raise RuntimeError("boom")
            return {"loaded": self.loaded}

    sys.modules["core.inference.diffusion"]._diffusion_backend = Backend(False)
    sys.modules["core.inference.video"]._backend = Backend(None)
    assert arbiter._studio_media_resident() is None
    sys.modules["core.inference.sd_cpp_backend"]._sd_cpp_backend = Backend(True)
    assert arbiter._studio_media_resident() == "has an image model loaded"
    sys.modules["core.inference.sd_cpp_backend"]._sd_cpp_backend = None
    sys.modules["core.inference.video"]._backend = Backend(True)
    assert arbiter._studio_media_resident() == "has a video model loaded"


def test_media_probe_never_imports_or_constructs(monkeypatch):
    for name in ("core.inference.diffusion", "core.inference.sd_cpp_backend", "core.inference.video"):
        monkeypatch.delitem(sys.modules, name, raising = False)
    assert arbiter._studio_media_resident() is None
    assert not any(n in sys.modules for n in ("core.inference.diffusion", "core.inference.video"))


# --- idle free --------------------------------------------------------------------------------


@pytest.fixture
def idle(world, monkeypatch):
    sleeps: list[int] = []
    gate = {"release": None}

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        await gate["release"].wait()
        # One release wakes one sleep: a real sleep always yields to the loop.
        gate["release"].clear()

    monkeypatch.setattr(arbiter, "_IDLE_SLEEP", fake_sleep)
    world.sleeps, world.gate = sleeps, gate
    return world


async def _settle():
    for _ in range(20):
        await asyncio.sleep(0)


def test_idle_free_fires_once_after_the_idle_period(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        assert arbiter.schedule_comfyui_idle_free()
        await _settle()
        assert idle.sleeps == [300] and idle.posts(OWN) == []
        idle.gate["release"].set()
        await _settle()
        assert len(idle.posts(OWN)) == 1 and idle.posts(PEER) == []
        await _settle()
        assert len(idle.posts(OWN)) == 1
        assert arbiter.recent_notices()[0]["reason"] == "comfyui_idle"

    run(go())


def test_idle_free_waits_another_period_while_work_is_queued(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        idle.comfy[OWN]["running"] = 1
        arbiter.schedule_comfyui_idle_free()
        await _settle()
        idle.gate["release"].set()
        await _settle()
        assert idle.posts(OWN) == [] and len(idle.sleeps) == 2
        idle.comfy[OWN]["running"] = 0
        idle.gate["release"].set()
        await _settle()
        assert len(idle.posts(OWN)) == 1

    run(go())


def test_a_new_job_cancels_the_idle_free(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        arbiter.schedule_comfyui_idle_free()
        await _settle()
        task = arbiter._idle_free
        (await arbiter.before_comfyui_job()).require_clear()
        await _settle()
        assert task.cancelled() and arbiter._idle_free is None
        idle.gate["release"].set()
        await _settle()
        assert idle.posts(OWN) == []

    run(go())


def test_rescheduling_replaces_the_timer(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        arbiter.schedule_comfyui_idle_free()
        first = arbiter._idle_free
        await _settle()
        arbiter.schedule_comfyui_idle_free()
        await _settle()
        assert first.cancelled() and arbiter._idle_free is not first
        arbiter.cancel_comfyui_idle_free()
        await _settle()

    run(go())


@pytest.mark.parametrize("over", [dict(comfyui_idle_free_s = 0), dict(arbitrate_comfyui = False), dict(enabled = False)])
def test_idle_free_off_schedules_nothing(idle, over):
    set_config(idle, **over)

    async def go():
        idle.gate["release"] = asyncio.Event()
        assert arbiter.schedule_comfyui_idle_free() is False
        assert arbiter._idle_free is None

    run(go())


def test_idle_free_does_not_raise_when_comfyui_vanished(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        idle.comfy[OWN]["up"] = False
        arbiter.schedule_comfyui_idle_free()
        await _settle()
        idle.gate["release"].set()
        await _settle()
        assert idle.posts(OWN) == [] and arbiter._idle_free is None

    run(go())
