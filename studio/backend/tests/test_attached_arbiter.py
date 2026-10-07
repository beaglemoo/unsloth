# SPDX-License-Identifier: AGPL-3.0-only
"""Shared-memory admission requires only the attached oMLX engine."""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from utils.attached_engines_settings import DEFAULT_CONFIG


@pytest.fixture
def admission(monkeypatch):
    calls = []
    config = replace(DEFAULT_CONFIG, enabled = True)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    arbiter._notices.clear()

    async def free_comfy():
        calls.append("free comfy")

    class Engine:
        def __init__(self, url):
            assert url == config.omlx_url

        async def unload_all(self):
            calls.append("unload omlx")
            return ["chat", "pinned-embedding"]

    monkeypatch.setattr(arbiter, "free_comfyui", free_comfy)
    monkeypatch.setattr(arbiter, "OmlxClient", Engine)
    return calls


@pytest.mark.parametrize("reason", ["local_load", "training"])
def test_local_work_proceeds_after_omlx_unload(admission, reason):
    async def work():
        result = await arbiter.free_for_local(reason)
        result.require_clear()
        admission.append(reason)
        return result

    result = asyncio.run(work())
    assert admission == ["free comfy", "unload omlx", reason]
    assert result.acted
    assert result.actions == ("Unloaded oMLX: chat, pinned-embedding",)
    assert arbiter.recent_notices()[0]["reason"] == reason


@pytest.mark.parametrize("reason", ["local_load", "training"])
def test_failed_omlx_eviction_blocks_local_work(admission, monkeypatch, reason):
    class Unreachable:
        def __init__(self, url):
            pass

        async def unload_all(self):
            raise RuntimeError("oMLX remains resident")

    monkeypatch.setattr(arbiter, "OmlxClient", Unreachable)
    result = asyncio.run(arbiter.free_for_local(reason))
    with pytest.raises(arbiter.AttachedAdmissionError, match = "remains resident"):
        result.require_clear()
    assert not result.acted


@pytest.mark.parametrize("config,skip", [
    (DEFAULT_CONFIG, "disabled"),
    (replace(DEFAULT_CONFIG, enabled = True, arbitrate_local_loads = False), "arbitration_off"),
])
def test_disabled_arbitration_never_contacts_engine(admission, monkeypatch, config, skip):
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    result = asyncio.run(arbiter.free_for_local("local_load"))
    assert result.skipped == skip
    assert admission == []


@pytest.fixture
def idle_studio(monkeypatch):
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)


def test_using_omlx_requires_no_peer_eviction(admission, idle_studio):
    result = asyncio.run(arbiter.before_omlx_use())
    result.require_clear()
    assert admission == []
    assert not result.acted


def test_thread_admission_rejects_waiting_on_its_own_loop(admission):
    async def work():
        return arbiter.free_for_local_from_thread("training", asyncio.get_running_loop())

    result = asyncio.run(work())
    assert "own event loop" in result.error
    assert admission == []


def test_omlx_use_is_refused_while_training(admission, monkeypatch):
    monkeypatch.setattr(arbiter, "_training_active", lambda: True)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    result = asyncio.run(arbiter.before_omlx_use())
    with pytest.raises(arbiter.AttachedAdmissionError, match = "Studio is training"):
        result.require_clear()
    assert result.skipped == "error"
    assert admission == []


def test_omlx_use_is_refused_while_a_local_model_is_resident(admission, monkeypatch):
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: True)
    result = asyncio.run(arbiter.before_omlx_use())
    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        result.require_clear()
    assert admission == []


def test_a_failing_residency_probe_refuses_omlx_use(admission, monkeypatch):
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)

    def broken():
        raise RuntimeError("probe failed")

    monkeypatch.setattr(arbiter, "_local_memory_active", broken)
    result = asyncio.run(arbiter.before_omlx_use())
    assert "probe failed" in result.error


def test_omlx_use_gate_is_off_when_local_arbitration_is_off(admission, monkeypatch):
    config = replace(DEFAULT_CONFIG, enabled = True, arbitrate_local_loads = False)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(arbiter, "_training_active", lambda: True)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: True)
    asyncio.run(arbiter.before_omlx_use()).require_clear()


def test_omlx_use_gate_does_not_block_local_admission(admission, monkeypatch):
    # Training and local loads free oMLX; only Studio's own oMLX use is gated.
    monkeypatch.setattr(arbiter, "_training_active", lambda: True)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: True)
    result = asyncio.run(arbiter.free_for_local("training"))
    result.require_clear()
    assert "unload omlx" in admission


def test_omlx_refusal_is_a_retryable_503_on_the_proxy_and_admin_paths(admission, monkeypatch):
    import routes.attached_engines as attached_routes

    monkeypatch.setattr(arbiter, "_training_active", lambda: True)
    with pytest.raises(Exception) as caught:
        asyncio.run(attached_routes._admit_omlx_use())
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "15"}
    assert "Studio is training" in caught.value.detail


def test_local_memory_probe_covers_loaded_loading_and_managed_engines(monkeypatch):
    import routes.inference as inf_mod
    from types import SimpleNamespace as NS
    import core.inference.llama_cpp as llama_mod

    def probe(llama_active = False, load = False, **backend):
        monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: NS(is_active = llama_active))
        monkeypatch.setattr(llama_mod, "chat_load_active", lambda: load)
        monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: NS(**backend) if backend else None)
        return inf_mod._attached_local_memory_active()

    assert not probe()
    assert probe(llama_active = True)
    assert probe(load = True)
    assert probe(active_model_name = "m")
    assert probe(loading_models = ("m",))
    assert probe(_managed_engine = object())


def _extra_slot(*, active_model = None, llama_active = False, loading_models = ()):
    from types import SimpleNamespace as NS

    return NS(
        llama = NS(is_active = llama_active),
        orchestrator = NS(active_model_name = active_model, loading_models = loading_models),
    )


@pytest.fixture
def idle_primary(monkeypatch):
    import routes.inference as inf_mod
    import core.inference.llama_cpp as llama_mod
    from core.inference import model_slots
    from types import SimpleNamespace as NS

    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: NS(is_active = False))
    monkeypatch.setattr(llama_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    return model_slots


def _extra_slot_states():
    return {
        "resident": lambda ms: ms.slots.append(_extra_slot(active_model = "kept")),
        "resident_cpu_llama": lambda ms: ms.slots.append(_extra_slot(llama_active = True)),
        "idle_but_retained": lambda ms: ms.slots.append(_extra_slot()),
        "loading": lambda ms: setattr(ms, "loading", (_extra_slot(), "loading-model")),
        "slot_orchestrator_loading": lambda ms: ms.slots.append(_extra_slot(loading_models = ("m",))),
        "stuck": lambda ms: ms.stuck.append(_extra_slot()),
    }


def test_idle_primary_and_no_slots_leaves_omlx_open(admission, idle_primary):
    import routes.inference as inf_mod

    assert not inf_mod._attached_local_memory_active()
    asyncio.run(arbiter.before_omlx_use()).require_clear()


@pytest.mark.parametrize("state", list(_extra_slot_states()))
def test_extra_slot_residency_refuses_omlx_use(admission, idle_primary, state):
    import routes.inference as inf_mod

    _extra_slot_states()[state](idle_primary)
    assert inf_mod._attached_local_memory_active()
    result = asyncio.run(arbiter.before_omlx_use())
    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        result.require_clear()
    assert admission == []


def test_extra_slot_residency_is_a_retryable_503_on_the_proxy_path(admission, idle_primary):
    import routes.inference as inf_mod
    import routes.attached_engines as attached_routes

    idle_primary.stuck.append(_extra_slot())
    with pytest.raises(Exception) as caught:
        asyncio.run(attached_routes._admit_omlx_use())
    assert caught.value.status_code == 503
    assert caught.value.headers == {"Retry-After": "15"}
    assert inf_mod._attached_local_memory_active()


def test_primary_residency_is_read_from_the_primary_not_the_routed_slot(admission, monkeypatch):
    import routes.inference as inf_mod
    import core.inference.llama_cpp as llama_mod
    from core.inference import model_slots
    from core.inference.orchestrator import routed_slot
    from types import SimpleNamespace as NS

    primary = NS(is_active = True)
    routed = NS(is_active = False)
    monkeypatch.setattr(
        inf_mod, "get_llama_cpp_backend", lambda: routed if routed_slot.get() is not None else primary
    )
    monkeypatch.setattr(llama_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    token = routed_slot.set(object())
    try:
        assert inf_mod._attached_local_memory_active()
    finally:
        routed_slot.reset(token)


def test_thread_admission_timeout_blocks_training_and_cancels(admission, monkeypatch):
    import concurrent.futures

    class Future:
        cancelled = False

        def result(self, timeout):
            raise concurrent.futures.TimeoutError()

        def cancel(self):
            Future.cancelled = True

    def submit(coro, loop):
        coro.close()
        return Future()

    monkeypatch.setattr(arbiter.asyncio, "run_coroutine_threadsafe", submit)
    result = arbiter.free_for_local_from_thread("training", asyncio.new_event_loop())
    assert "timed out" in result.error
    assert Future.cancelled


def test_thread_admission_timeout_before_future_exists_is_not_unbound(admission, monkeypatch):
    import concurrent.futures

    def raising(coro, loop):
        coro.close()
        raise concurrent.futures.TimeoutError()

    monkeypatch.setattr(arbiter.asyncio, "run_coroutine_threadsafe", raising)
    result = arbiter.free_for_local_from_thread("training", asyncio.new_event_loop())
    assert result.skipped == "error"
    assert "UnboundLocalError" not in result.error and "referenced before assignment" not in result.error
    assert result.error.startswith("Shared-memory admission failed")


@pytest.mark.parametrize("local", [True, False])
def test_config_failure_is_a_retryable_error_not_an_exception(admission, monkeypatch, local):
    def broken():
        raise RuntimeError("settings unreadable")

    monkeypatch.setattr(arbiter, "get_config", broken)
    result = asyncio.run(arbiter._admit("training", local = local))
    assert result.skipped == "error"
    assert "settings unreadable" in result.error
    with pytest.raises(arbiter.AttachedAdmissionError):
        result.require_clear()
