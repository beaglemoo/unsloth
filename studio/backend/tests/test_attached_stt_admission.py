# SPDX-License-Identifier: AGPL-3.0-only
"""Dictation (STT) sidecars take part in attached-engine arbitration, in both directions.

oMLX and ComfyUI admission unloads an idle sidecar and is refused only while one is loading or transcribing (or
cannot be unloaded). A sidecar load that allocates is admitted as a local load, with a claim that spans its startup.
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


_REAL_RELEASE = arbiter._release_idle_stt

IDLE = {"model": None, "engine": None, "device": None, "loading": False}
RESIDENT = {"model": MODEL, "engine": "transformers", "device": "cuda", "loading": False}
LOADING = {"model": None, "engine": "mtmd", "device": None, "loading": True}


@pytest.fixture
def dictation(monkeypatch):
    """A fake dictation registry with the real release logic. ``busy`` models a transcription in flight: like the
    sidecars, ``unload(wait=False)`` leaves a busy sidecar resident."""
    state = {"stt": dict(IDLE), "busy": False, "unloads": [], "unload_fails": False}

    def resident():
        return dict(state["stt"])

    def unload(engines = None, *, wait = True, expected_model = None):
        state["unloads"].append(wait)
        if state["unload_fails"]:
            return ["transformers"]
        if not state["busy"] and not state["stt"]["loading"]:
            state["stt"] = dict(IDLE)
        return []

    monkeypatch.setattr(stt_registry, "resident", resident)
    monkeypatch.setattr(stt_registry, "unload", unload)
    monkeypatch.setattr(arbiter, "_release_idle_stt", _REAL_RELEASE)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    return state


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))


# --- direction 1: oMLX and ComfyUI evict an idle dictation sidecar, and wait for a busy one -------


def test_the_memory_probe_does_not_count_dictation(dictation):
    """Dictation is not a reason to refuse by itself; admission frees it or refuses on its own state."""
    dictation["stt"] = dict(RESIDENT)
    assert not inference._attached_local_memory_active()


def test_an_idle_resident_sidecar_is_unloaded_and_omlx_is_admitted(dictation, enabled):
    dictation["stt"] = dict(RESIDENT)
    result = run(arbiter.before_omlx_use())
    assert not result.error
    assert dictation["unloads"] == [False]
    assert dictation["stt"] == IDLE


def test_an_idle_resident_sidecar_is_unloaded_and_a_comfyui_job_is_admitted(dictation, enabled, monkeypatch):
    async def none(*args, **kwargs):
        return []

    monkeypatch.setattr(arbiter, "_unload_omlx", none)
    monkeypatch.setattr(arbiter, "_free_urls", none)
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda config: (None, []))
    dictation["stt"] = dict(RESIDENT)
    result = run(arbiter.before_comfyui_job())
    assert not result.error
    assert dictation["unloads"] == [False]
    assert dictation["stt"] == IDLE


def test_no_sidecar_means_nothing_is_unloaded(dictation, enabled):
    assert not run(arbiter.before_omlx_use()).error
    assert dictation["unloads"] == []


def test_a_transcription_in_flight_refuses_omlx(dictation, enabled):
    dictation["stt"], dictation["busy"] = dict(RESIDENT), True
    result = run(arbiter.before_omlx_use())
    assert result.error and "transcribing" in result.error
    assert dictation["stt"] == RESIDENT


def test_a_transcription_in_flight_refuses_a_comfyui_job(dictation, enabled):
    dictation["stt"], dictation["busy"] = dict(RESIDENT), True
    result = run(arbiter.before_comfyui_job())
    assert result.error and "transcribing" in result.error


@pytest.mark.parametrize("job", ["omlx", "comfyui"])
def test_a_loading_sidecar_refuses_and_is_left_alone(dictation, enabled, job):
    dictation["stt"] = dict(LOADING)
    result = run(arbiter.before_omlx_use() if job == "omlx" else arbiter.before_comfyui_job())
    assert result.error and "loading a dictation model" in result.error
    assert dictation["unloads"] == []


@pytest.mark.parametrize("job", ["omlx", "comfyui"])
def test_a_failed_eviction_fails_closed(dictation, enabled, job):
    dictation["stt"], dictation["unload_fails"] = dict(RESIDENT), True
    result = run(arbiter.before_omlx_use() if job == "omlx" else arbiter.before_comfyui_job())
    assert result.error and "could not unload its dictation model" in result.error


@pytest.mark.parametrize("job", ["omlx", "comfyui"])
def test_a_registry_that_raises_fails_closed(dictation, enabled, monkeypatch, job):
    def boom(*args, **kwargs):
        raise RuntimeError("sidecar import failed")

    monkeypatch.setattr(stt_registry, "resident", boom)
    result = run(arbiter.before_omlx_use() if job == "omlx" else arbiter.before_comfyui_job())
    assert result.error and "could not unload its dictation model" in result.error


def test_the_omlx_proxy_turns_a_busy_sidecar_into_a_retryable_503(dictation, enabled):
    import routes.attached_engines as attached_routes

    dictation["stt"], dictation["busy"] = dict(RESIDENT), True
    with pytest.raises(HTTPException) as caught:
        run(attached_routes._admit_omlx_use())
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "15"}


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


def test_oMLX_use_is_refused_while_the_dictation_claim_is_held(world, dictation):
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
