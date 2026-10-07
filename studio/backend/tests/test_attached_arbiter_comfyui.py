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
FAULTS = ["http", "timeout", "json"]


class World:
    """Two ComfyUIs (Studio's and StoryPress's), oMLX, and what Studio is running."""

    def __init__(self):
        self.calls: list[tuple[str, str, str]] = []
        self.comfy = {
            OWN: dict(up = True, running = 0, pending = 0, malformed = False, fault = None),
            PEER: dict(up = True, running = 0, pending = 0, malformed = False, fault = None),
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
                if state["fault"] == "http":
                    return httpx.Response(503)
                if state["fault"] == "timeout":
                    raise httpx.ReadTimeout("slow", request = request)
                if state["fault"] == "json":
                    return httpx.Response(200, content = b"<html>")
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
    arbiter._media_probe_future = None
    arbiter._media_holds.clear()
    arbiter._comfyui_holds.clear()

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


_REAL_MEDIA_PROBE = arbiter._studio_media_resident


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


@pytest.mark.parametrize("what", ["running", "pending"])
@pytest.mark.parametrize("reason", ["local_load", "training"])
def test_a_generating_peer_refuses_local_work_and_is_neither_freed_nor_interrupted(world, what, reason):
    world.comfy[PEER][what] = 1
    result = run(arbiter.free_for_local(reason))
    assert result.error and PEER in result.error and "generating" in result.error
    assert world.omlx_calls == []
    assert world.posts(PEER) == [] and world.posts(OWN) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_an_uncertain_peer_refuses_local_work(world, fault):
    world.comfy[PEER]["fault"] = fault
    result = run(arbiter.free_for_local("local_load"))
    assert result.error and PEER in result.error
    assert world.omlx_calls == [] and world.posts(PEER) == [] and world.posts(OWN) == []


def test_a_down_peer_does_not_refuse_local_work_and_an_idle_one_is_freed(world):
    world.comfy[PEER]["up"] = False
    run(arbiter.free_for_local("local_load")).require_clear()
    assert world.posts(PEER) == [] and len(world.posts(OWN)) == 1
    world.comfy[PEER]["up"] = True
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert len(world.posts(PEER)) == 1 and PEER in result.actions[-1]


def test_a_peer_busy_with_arbitration_off_never_refuses(world):
    set_config(world, arbitrate_comfyui = False)
    world.comfy[PEER]["running"] = 1
    run(arbiter.free_for_local("local_load")).require_clear()
    run(arbiter.before_omlx_use()).require_clear()
    assert world.posts(PEER) == []


def test_the_env_override_is_the_only_comfyui_probed_for_admission(world, monkeypatch):
    world.comfy[PEER]["running"] = 1
    monkeypatch.setenv("STUDIO_COMFYUI_URL", OWN)
    run(arbiter.free_for_local("local_load")).require_clear()
    run(arbiter.before_omlx_use()).require_clear()
    assert all(c[0] == OWN for c in world.calls)


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


# --- an uncertain queue answer is not an idle ComfyUI -----------------------------------------

@pytest.mark.parametrize("fault", FAULTS)
def test_an_uncertain_own_queue_refuses_a_local_load_and_frees_nothing(world, fault):
    world.comfy[OWN]["fault"] = fault
    result = run(arbiter.free_for_local("local_load"))
    assert result.error and "ComfyUI" in result.error
    assert world.omlx_calls == [] and world.posts(OWN) == [] and world.posts(PEER) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_an_uncertain_own_queue_refuses_omlx_use_and_load(world, fault):
    world.comfy[OWN]["fault"] = fault
    assert run(arbiter.before_omlx_use()).error
    arbiter._busy_cache.clear()
    assert run(arbiter.before_omlx_load()).error
    assert world.posts(OWN) == [] and world.posts(PEER) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_an_uncertain_answer_is_not_cached_as_idle_or_busy(world, fault):
    world.comfy[OWN]["fault"] = fault
    assert run(arbiter.before_omlx_use()).error
    assert OWN not in arbiter._busy_cache
    world.comfy[OWN]["fault"] = None
    run(arbiter.before_omlx_use()).require_clear()


def test_a_cached_idle_answer_is_not_overwritten_by_a_later_fault_within_the_ttl(world):
    run(arbiter.before_omlx_use()).require_clear()
    world.comfy[OWN]["fault"] = "http"
    run(arbiter.before_omlx_use()).require_clear()
    assert len(world.probes(OWN)) == 1


def test_connection_refused_is_not_running_and_admits(world):
    world.comfy[OWN]["up"] = False
    run(arbiter.before_omlx_use()).require_clear()
    result = run(arbiter.free_for_local("local_load"))
    result.require_clear()
    assert result.actions == ("Unloaded oMLX: chat", f"Freed ComfyUI memory: {PEER}")


@pytest.mark.parametrize("fault", FAULTS)
def test_free_comfyui_never_frees_an_uncertain_server(world, fault):
    world.comfy[OWN]["fault"] = fault
    assert run(arbiter.free_comfyui()) == [PEER]
    assert world.posts(OWN) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_comfyui_job_is_refused_next_to_an_uncertain_peer(world, fault):
    world.comfy[PEER]["fault"] = fault
    assert run(arbiter.before_comfyui_job()).error
    assert world.omlx_calls == [] and world.posts(PEER) == []


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


@pytest.mark.parametrize("what", ["running", "pending"])
def test_omlx_use_and_load_are_refused_while_a_peer_generates(world, what):
    world.comfy[PEER][what] = 1
    for admit in (arbiter.before_omlx_use, arbiter.before_omlx_load):
        arbiter._busy_cache.clear()
        result = run(admit())
        assert result.error and PEER in result.error and "to use oMLX" in result.error
    assert world.posts(OWN) == [] and world.posts(PEER) == []


@pytest.mark.parametrize("fault", FAULTS)
def test_omlx_use_is_refused_next_to_an_uncertain_peer(world, fault):
    world.comfy[PEER]["fault"] = fault
    assert PEER in run(arbiter.before_omlx_use()).error


def test_omlx_use_ignores_a_down_comfyui_and_a_down_peer(world):
    world.comfy[OWN]["up"] = False
    run(arbiter.before_omlx_use()).require_clear()
    world.comfy[PEER]["up"] = False
    arbiter._busy_cache.clear()
    run(arbiter.before_omlx_use()).require_clear()
    assert world.posts(OWN) == [] and world.posts(PEER) == []


def test_omlx_use_probes_the_peer_at_most_once_a_second_too(world, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(arbiter.time, "monotonic", lambda: clock[0])
    run(arbiter.before_omlx_use())
    run(arbiter.before_omlx_use())
    assert len(world.probes(PEER)) == 1
    world.comfy[PEER]["running"] = 1
    clock[0] += 1.5
    assert run(arbiter.before_omlx_use()).error


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


# --- the ComfyUI job reservation ---------------------------------------------------------------


def _admit_job(world):
    hold = arbiter.new_comfyui_hold()
    run(arbiter.before_comfyui_job(hold)).require_clear()
    assert hold.active
    return hold


def test_an_admitted_job_holds_the_memory_until_released(world):
    hold = _admit_job(world)

    async def attempts():
        return (
            await arbiter.before_omlx_use(),
            await arbiter.before_omlx_load(),
            await arbiter.free_for_local("local_load"),
            await arbiter.free_for_local("training"),
        )

    results = run(attempts())
    assert all("ComfyUI job starting" in r.error for r in results), [r.error for r in results]
    assert "retry shortly to use oMLX" in results[0].error
    assert "before loading a local model" in results[2].error
    hold.release()
    hold.release()  # idempotent
    assert not arbiter._comfyui_held()
    assert all(r.error is None for r in run(attempts()))


def test_a_refused_media_load_does_not_leave_its_own_claim_behind(world):
    _admit_job(world)
    media = arbiter.new_media_hold("image")
    assert run(arbiter.free_for_local("local_load", media_hold = media)).error
    assert not media.active and arbiter._media_held() is None


def test_a_refused_or_skipped_job_registers_nothing(world):
    world.comfy[PEER]["running"] = 1
    hold = arbiter.new_comfyui_hold()
    assert run(arbiter.before_comfyui_job(hold)).error and not hold.active and not arbiter._comfyui_held()
    world.comfy[PEER]["running"] = 0
    world.training = True
    assert run(arbiter.before_comfyui_job(hold)).error and not hold.active
    world.training = False
    world.omlx_error = AttachedEngineBusy("busy")
    assert run(arbiter.before_comfyui_job(hold)).error and not hold.active
    world.omlx_error = None
    set_config(world, arbitrate_comfyui = False)
    assert run(arbiter.before_comfyui_job(hold)).skipped == "arbitration_off" and not hold.active


def test_the_reservation_only_binds_while_comfyui_arbitration_is_on(world):
    hold = _admit_job(world)
    set_config(world, arbitrate_comfyui = False)
    assert run(arbiter.before_omlx_use()).error is None
    assert run(arbiter.free_for_local("local_load")).error is None
    hold.release()


def test_concurrent_admissions_serialise_with_the_job_admission(world):
    async def race():
        hold = arbiter.new_comfyui_hold()
        job, chat = await asyncio.gather(arbiter.before_comfyui_job(hold), arbiter.before_omlx_use())
        return job, chat

    job, chat = run(race())
    job.require_clear()
    # The chat request ran first (oMLX admitted, then unloaded by the job) or saw the claim: never both unaware.
    assert chat.error is None or "ComfyUI job starting" in chat.error


def test_idle_free_does_not_fire_while_a_job_is_reserved(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        hold = arbiter.new_comfyui_hold()
        (await arbiter.before_comfyui_job(hold)).require_clear()
        assert arbiter.schedule_comfyui_idle_free()
        await _settle()
        idle.gate["release"].set()
        await _settle()
        assert idle.posts(OWN) == [] and len(idle.sleeps) == 2
        hold.release()
        idle.gate["release"].set()
        await _settle()
        assert len(idle.posts(OWN)) == 1

    run(go())


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


# --- oMLX is refused beside a Studio image or video model, resident, loading or about to load --


@pytest.mark.parametrize("media", ["has an image model loaded", "has a video model loaded", "is loading an image model"])
def test_omlx_use_and_load_are_refused_while_studio_has_media(world, media):
    world.media = media
    for admit in (arbiter.before_omlx_use, arbiter.before_omlx_load):
        arbiter._busy_cache.clear()
        result = run(admit())
        assert result.error and media in result.error and "to use oMLX" in result.error
    assert world.posts(OWN) == [] and world.posts(PEER) == [] and world.calls == []


def test_the_media_refusal_follows_the_local_arbitration_setting(world):
    world.media = "has an image model loaded"
    set_config(world, arbitrate_local_loads = False)
    run(arbiter.before_omlx_use()).require_clear()


def test_a_media_load_replacing_a_media_model_is_not_refused_by_it(world):
    world.media = "has an image model loaded"
    hold = arbiter.new_media_hold("image")
    run(arbiter.free_for_local("local_load", media_hold = hold)).require_clear()
    hold.release()


def test_a_slow_media_probe_is_a_refusal_not_an_admission(world, monkeypatch):
    import threading

    release = threading.Event()
    monkeypatch.setattr(arbiter, "_MEDIA_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: release.wait(5) and None)
    try:
        assert run(arbiter.before_omlx_use()).error
    finally:
        release.set()
        arbiter._media_probe_future.result(timeout = 5)


def _probe_threads() -> int:
    import threading

    return sum(t.name.startswith("arbiter-media-probe") for t in threading.enumerate())


def test_a_stuck_backend_lock_never_grows_the_probe_threads(world, backends, monkeypatch):
    import threading

    lock = threading.Lock()
    entered = []

    class Stuck:
        def status(self):
            return {"loaded": False}

        def loading_repo_ids(self):
            entered.append(1)
            with lock:
                return ()

    backends["core.inference.sd_cpp_backend"]._sd_cpp_backend = Stuck()
    monkeypatch.setattr(arbiter, "_MEDIA_PROBE_TIMEOUT_S", 0.05)
    lock.acquire()
    try:
        before = _probe_threads()
        for _ in range(30):
            result = run(arbiter.before_omlx_use())
            assert result.error and "busy loading or unloading" in result.error
        assert len(entered) == 1
        assert _probe_threads() <= max(before, 1)
        assert run(arbiter.before_omlx_load()).error and len(entered) == 1
        assert world.omlx_calls == [] and world.calls == []
    finally:
        lock.release()
    arbiter._media_probe_future.result(timeout = 5)
    run(arbiter.before_omlx_use()).require_clear()
    assert len(entered) == 2


def test_a_held_media_claim_refuses_with_no_thread_hop(world, backends, monkeypatch):
    hold = arbiter.new_media_hold("image")
    run(arbiter.free_for_local("local_load", media_hold = hold)).require_clear()

    def no_hop(*args, **kwargs):
        raise AssertionError("a held claim must not reach the probe pool")

    monkeypatch.setattr(arbiter._media_probe_pool, "submit", no_hop)
    for admit in (arbiter.before_omlx_use, arbiter.before_omlx_load, arbiter.before_comfyui_job):
        assert "is loading an image model" in run(admit()).error
    hold.release()


def test_the_probe_never_holds_admission_longer_than_its_limit(world, backends, monkeypatch):
    import threading
    import time

    release = threading.Event()
    backends["core.inference.video"]._backend = types.SimpleNamespace(
        status = lambda: {"loaded": False}, loading_repo_ids = lambda: release.wait(5) and ()
    )
    monkeypatch.setattr(arbiter, "_MEDIA_PROBE_TIMEOUT_S", 0.2)
    started = time.monotonic()
    try:
        assert run(arbiter.before_omlx_use()).error
        assert time.monotonic() - started < 1.5
    finally:
        release.set()
        arbiter._media_probe_future.result(timeout = 5)


def _backend(loaded, loading = None, *, boom = False):
    class Backend:
        def status(self):
            return {"loaded": loaded}

        if loading is not None:
            def loading_repo_ids(self):
                if boom:
                    raise RuntimeError("boom")
                return loading

    return Backend()


@pytest.fixture
def backends(monkeypatch):
    for name in ("core.inference.diffusion", "core.inference.sd_cpp_backend", "core.inference.video"):
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(
            _diffusion_backend = None, _sd_cpp_backend = None, _backend = None))
    monkeypatch.setattr(arbiter, "_studio_media_resident", _REAL_MEDIA_PROBE)
    arbiter._media_holds.clear()
    yield sys.modules
    arbiter._media_holds.clear()


def test_the_probe_reports_a_loading_image_or_video_model(backends):
    assert _REAL_MEDIA_PROBE() is None
    backends["core.inference.diffusion"]._diffusion_backend = _backend(False, ())
    assert _REAL_MEDIA_PROBE() is None
    backends["core.inference.diffusion"]._diffusion_backend = _backend(False, ("org/model",))
    assert _REAL_MEDIA_PROBE() == "is loading an image model"
    backends["core.inference.diffusion"]._diffusion_backend = None
    backends["core.inference.sd_cpp_backend"]._sd_cpp_backend = _backend(False, ("org/model",))
    assert _REAL_MEDIA_PROBE() == "is loading an image model"
    backends["core.inference.sd_cpp_backend"]._sd_cpp_backend = None
    backends["core.inference.video"]._backend = _backend(False, ("org/video",))
    assert _REAL_MEDIA_PROBE() == "is loading a video model"


def test_the_probe_prefers_loaded_and_survives_a_backend_that_cannot_say(backends):
    backends["core.inference.video"]._backend = _backend(True, ("x",))
    assert _REAL_MEDIA_PROBE() == "has a video model loaded"
    backends["core.inference.video"]._backend = _backend(False, ("x",), boom = True)
    assert _REAL_MEDIA_PROBE() is None


def test_omlx_is_refused_while_a_media_backend_is_loading(world, backends):
    backends["core.inference.diffusion"]._diffusion_backend = _backend(False, ("org/model",))
    result = run(arbiter.before_omlx_use())
    assert result.error and "is loading an image model" in result.error


def test_an_admitted_media_load_holds_omlx_out_until_released(world, backends):
    hold = arbiter.new_media_hold("image")
    result = run(arbiter.free_for_local("local_load", media_hold = hold))
    result.require_clear()
    assert world.omlx_calls == ["unload_all"]
    assert "is loading an image model" in run(arbiter.before_omlx_use()).error
    assert "is loading an image model" in run(arbiter.before_omlx_load()).error
    assert "ComfyUI job" in run(arbiter.before_comfyui_job()).error
    hold.release()
    hold.release()
    run(arbiter.before_omlx_use()).require_clear()


def test_a_video_hold_names_the_video_model(world, backends):
    hold = arbiter.new_media_hold("video")
    run(arbiter.free_for_local("local_load", media_hold = hold)).require_clear()
    assert "is loading a video model" in run(arbiter.before_omlx_use()).error


def test_a_refused_media_admission_takes_no_claim(world, backends):
    world.comfy[OWN]["running"] = 1
    hold = arbiter.new_media_hold("image")
    assert run(arbiter.free_for_local("local_load", media_hold = hold)).error
    assert not hold.active and arbiter._media_holds == set()
    world.comfy[OWN]["running"] = 0
    run(arbiter.before_omlx_use()).require_clear()


def test_a_media_admission_that_fails_to_unload_omlx_gives_the_claim_back(world, backends):
    world.omlx_error = AttachedEngineError("oMLX models are still loaded or loading: x")
    hold = arbiter.new_media_hold("image")
    assert run(arbiter.free_for_local("local_load", media_hold = hold)).error
    assert not hold.active and arbiter._media_holds == set()


def test_arbitration_off_takes_no_claim(world, backends):
    set_config(world, arbitrate_local_loads = False)
    hold = arbiter.new_media_hold("image")
    assert run(arbiter.free_for_local("local_load", media_hold = hold)).skipped == "arbitration_off"
    assert arbiter._media_holds == set()


def test_a_concurrent_omlx_request_cannot_slip_in_after_a_media_admission(world, backends):
    async def go(order):
        hold = arbiter.new_media_hold("image")
        coros = {
            "media": arbiter.free_for_local("local_load", media_hold = hold),
            "omlx": arbiter.before_omlx_use(),
        }
        tasks = {name: asyncio.ensure_future(coros[name]) for name in order}
        return {name: await task for name, task in tasks.items()}, hold

    results, hold = run(go(["media", "omlx"]))
    results["media"].require_clear()
    assert results["omlx"].error and "is loading an image model" in results["omlx"].error
    hold.release()
    world.omlx_calls.clear()
    # oMLX first: it was admitted before the media load existed, and the media load then unloads it.
    results, hold = run(go(["omlx", "media"]))
    results["omlx"].require_clear()
    results["media"].require_clear()
    assert world.omlx_calls == ["unload_all"]
    hold.release()


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


def test_idle_free_does_not_free_an_uncertain_server_and_tries_again_later(idle):
    async def go():
        idle.gate["release"] = asyncio.Event()
        idle.comfy[OWN]["fault"] = "http"
        arbiter.schedule_comfyui_idle_free()
        await _settle()
        idle.gate["release"].set()
        await _settle()
        assert idle.posts(OWN) == [] and len(idle.sleeps) == 2
        idle.comfy[OWN]["fault"] = None
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
