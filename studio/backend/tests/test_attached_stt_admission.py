# SPDX-License-Identifier: AGPL-3.0-only
"""Dictation (STT) sidecars take part in attached-engine arbitration, in both directions.

A sidecar that is loading or resident counts as local memory, so oMLX and ComfyUI are refused next to it, and a
sidecar load that allocates is admitted as a local load, with a claim that spans its startup.
"""
from __future__ import annotations

import asyncio
import functools
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import routes.inference as inference
from core.inference import stt_registry
from core.inference.attached import arbiter
from utils.attached_engines_settings import DEFAULT_CONFIG

MODEL = "openai/whisper-small"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def idle_studio(monkeypatch):
    """No chat model, slot or training; no dictation sidecar until a test says so."""
    import core.inference.llama_cpp as llama_mod
    from core.inference import model_slots

    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: NS(is_active = False))
    monkeypatch.setattr(llama_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(inference, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(inference, "_resident_npu_model", lambda: None, raising = False)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    state = {"stt": {"model": None, "engine": None, "device": None, "loading": False}}
    monkeypatch.setattr(stt_registry, "resident", lambda: dict(state["stt"]))
    return state


STT_STATES = {
    "resident": {"model": MODEL, "engine": "transformers", "device": "cuda", "loading": False},
    "resident_cpu": {"model": "ggml-small", "engine": "gguf", "device": "whisper.cpp", "loading": False},
    "loading": {"model": None, "engine": "mtmd", "device": None, "loading": True},
}


# --- direction 1: oMLX and ComfyUI are refused next to a dictation sidecar -----------------------


def test_an_idle_dictation_stack_leaves_the_memory_probe_clear(idle_studio):
    assert not inference._attached_local_memory_active()


@pytest.mark.parametrize("state", list(STT_STATES))
def test_a_loading_or_resident_sidecar_counts_as_local_memory(idle_studio, state):
    idle_studio["stt"] = STT_STATES[state]
    assert inference._attached_local_memory_active()


def test_a_broken_stt_registry_never_breaks_the_probe(idle_studio, monkeypatch):
    def boom():
        raise RuntimeError("cannot import the sidecar")

    monkeypatch.setattr(stt_registry, "resident", boom)
    assert not inference._attached_local_memory_active()


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))


@pytest.mark.parametrize("state", list(STT_STATES))
def test_omlx_use_is_refused_while_dictation_is_loading_or_resident(idle_studio, enabled, state):
    idle_studio["stt"] = STT_STATES[state]
    result = run(arbiter.before_omlx_use())
    assert result.error and "local model loaded" in result.error


@pytest.mark.parametrize("state", list(STT_STATES))
def test_a_comfyui_job_is_refused_while_dictation_is_loading_or_resident(idle_studio, enabled, state):
    idle_studio["stt"] = STT_STATES[state]
    result = run(arbiter.before_comfyui_job())
    assert result.error and "local model loaded" in result.error


# --- direction 2: a dictation load that allocates is admitted as a local load --------------------


@pytest.fixture
def world(monkeypatch, enabled):
    """Real arbiter, fake oMLX; records what the load saw."""
    state = {"omlx": [], "unload_error": None}

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            if state["unload_error"]:
                raise RuntimeError(state["unload_error"])
            state["omlx"].append("unload_all")
            return ["chat"]

    async def no_comfyui():
        return []

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(arbiter, "free_comfyui", no_comfyui)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    arbiter._media_holds.clear()
    arbiter._notices.clear()
    yield state
    arbiter._media_holds.clear()


@pytest.fixture
def stt(monkeypatch, world):
    """The dictation routes with a fake sidecar and a fake lifecycle that records the claim it ran under."""
    seen = {"loads": [], "claim": []}
    sidecar = NS(loaded_model = None, device = None)

    def load_stt(model, engine, cancel_event = None, **kwargs):
        seen["loads"].append((model, engine, list(world["omlx"])))
        seen["claim"].append(arbiter._media_held())
        sidecar.loaded_model = model

    async def watcher(request, sidecar_, cancel_event):
        return None

    async def stop(task):
        return None

    monkeypatch.setattr(inference.account_access, "managed_account", lambda: False)
    monkeypatch.setattr(inference, "_resolve_serving_stt_engine", lambda engine: "transformers")
    monkeypatch.setattr(inference, "_stt_sidecar_for", lambda engine: sidecar)
    monkeypatch.setattr(inference, "_stt_resolved_model_id", lambda model, engine: model)
    monkeypatch.setattr(inference, "_prepare_runtime_fallback_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(inference, "_stt_lifecycle", lambda: (load_stt, lambda *a, **k: []))
    monkeypatch.setattr(inference, "_await_stt_disconnect_then_cancel", watcher)
    monkeypatch.setattr(inference, "_stop_local_disconnect_cancel_watcher", stop)
    seen["sidecar"] = sidecar
    return seen


def stt_load():
    payload = NS(model = MODEL, engine = "transformers", device = None)
    return run(inference.stt_load(payload, NS(), "user"))


def transcribe():
    return run(inference._transcribe_audio_result(b"audio", MODEL, None, False))


@pytest.fixture(params = ["stt_load", "transcribe"])
def entry(request, stt, monkeypatch):
    if request.param == "transcribe":
        stt["sidecar"].transcribe = lambda raw, model, language, fast, *a, **k: {"text": "hi"}
    return {"stt_load": stt_load, "transcribe": transcribe}[request.param]


def test_a_dictation_load_unloads_omlx_first_and_holds_it_out_during_startup(entry, stt, world):
    entry()
    assert world["omlx"] == ["unload_all"]
    assert stt["loads"] == [(MODEL, "transformers", ["unload_all"])]
    assert stt["claim"] == ["is loading a dictation model"]
    assert arbiter._media_holds == set()


def test_a_refused_admission_is_a_retryable_503_and_starts_no_sidecar(entry, stt, world):
    world["unload_error"] = "oMLX remains resident"
    with pytest.raises(HTTPException) as caught:
        entry()
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "15"}
    assert stt["loads"] == []
    assert arbiter._media_holds == set()


def test_a_failing_load_releases_the_claim(entry, stt, world, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("worker died")

    monkeypatch.setattr(inference, "_stt_lifecycle", lambda: (boom, lambda *a, **k: []))
    with pytest.raises(HTTPException):
        entry()
    assert world["omlx"] == ["unload_all"]
    assert arbiter._media_holds == set()


def test_a_model_that_is_already_resident_is_not_admitted_again(entry, stt, world):
    stt["sidecar"].loaded_model = MODEL
    entry()
    assert world["omlx"] == []
    assert stt["claim"] == [None]


def test_a_dictation_load_is_left_alone_when_arbitration_is_off(entry, stt, world, monkeypatch):
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    entry()
    assert world["omlx"] == []
    assert len(stt["loads"]) == 1


def test_oMLX_use_is_refused_while_the_dictation_claim_is_held(world, idle_studio):
    """A load that has been admitted but whose sidecar has not published its loading state yet."""
    hold = arbiter.new_media_hold("stt")
    result = run(arbiter.free_for_local("local_load", media_hold = hold))
    assert not result.error
    try:
        refused = run(arbiter.before_omlx_use())
        assert refused.error and "dictation model" in refused.error
        job = run(arbiter.before_comfyui_job())
        assert job.error and "dictation model" in job.error
    finally:
        hold.release()
    assert not run(arbiter.before_omlx_use()).error
