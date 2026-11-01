# SPDX-License-Identifier: AGPL-3.0-only
"""Attached admission stays closed while Studio is still tearing a model down.

A media backend clears its loaded and loading state before it joins the builder, waits for the generation and frees
memory, and a failed chat unload clears ``active_model_name`` while its worker lives on. In both windows the memory is
not free yet, so oMLX and ComfyUI must keep getting the retryable 503.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from utils.attached_engines_settings import DEFAULT_CONFIG

_MEDIA_MODULES = (
    ("core.inference.diffusion", "_diffusion_backend"),
    ("core.inference.sd_cpp_backend", "_sd_cpp_backend"),
    ("core.inference.mlx_diffusion_backend", "_mlx_diffusion_backend"),
    ("core.inference.video", "_backend"),
)


async def _no_blocker():
    return None


class _Chat:
    def __init__(self, name = None):
        self.active_model_name = name
        self.models = {name: {"is_mlx": True}} if name else {}
        self.loading_models = ()
        self._managed_engine = None

    def is_worker_alive(self):
        return False


@pytest.fixture
def world(monkeypatch):
    import core.inference.llama_cpp as llama_mod
    import routes.inference as inf_mod
    from core.inference import model_slots

    config = replace(DEFAULT_CONFIG, enabled = True, arbitrate_comfyui = False)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_stt_blocker", _no_blocker)
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: NS(is_active = False, is_loaded = False))
    monkeypatch.setattr(llama_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)
    monkeypatch.setattr(arbiter, "_media_probe_future", None)
    arbiter._media_holds.clear()
    arbiter._omlx_holds.clear()
    from core.inference import mlx_diffusion_backend as real_mlx_module

    state = NS(chat = _Chat(), stubs = {}, mlx_module = real_mlx_module)
    for module, attribute in _MEDIA_MODULES:
        stub = types.SimpleNamespace(**{attribute: None})
        monkeypatch.setitem(sys.modules, module, stub)
        state.stubs[attribute] = stub
    monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: state.chat)
    yield state
    arbiter._media_holds.clear()
    arbiter._omlx_holds.clear()


def _admit():
    return asyncio.run(arbiter.before_omlx_use())


# -- media teardown ----------------------------------------------------------


def test_mlx_image_unload_blocks_admission_until_the_builder_is_joined_and_memory_freed(world):
    freed = []
    backend = world.mlx_module.MlxDiffusionBackend()
    backend._runtime = NS(free = lambda: freed.append(True))
    world.stubs["_mlx_diffusion_backend"]._mlx_diffusion_backend = backend

    # A cancelled load whose builder cannot be interrupted: unload() joins it before it reports the memory free.
    release_builder = threading.Event()
    builder = threading.Thread(target = release_builder.wait, daemon = True)
    builder.start()
    backend._load_thread = builder
    backend._loading = NS(repo_id = "org/model", base_repo = None, fetch_repo = None, error = None)
    backend._free_pending = True

    unloaded = threading.Thread(target = backend.unload, daemon = True)
    unloaded.start()
    try:
        deadline = threading.Event()
        for _ in range(200):
            if backend._unloading:
                break
            deadline.wait(0.01)
        # The state the old probe read as clear: nothing loaded, nothing loading, but still draining.
        assert backend._unloading == 1
        assert backend.status()["loaded"] is False
        assert backend.loading_repo_ids() == ()
        assert backend.draining_repo_ids() == ("org/model",)
        with pytest.raises(arbiter.AttachedAdmissionError, match = "still unloading an image model"):
            _admit().require_clear()
    finally:
        release_builder.set()
        unloaded.join(5)
    assert not unloaded.is_alive()
    assert freed == [True]
    _admit().require_clear()


@pytest.mark.parametrize(
    "attribute, counter",
    [
        ("_diffusion_backend", "_unload_waiters"),
        ("_sd_cpp_backend", "_stopping_servers"),
        ("_backend", "_teardown_waiters"),
    ],
)
def test_other_media_backends_block_admission_while_their_teardown_counter_is_up(world, attribute, counter):
    backend = NS(**{counter: 0}, status = lambda: {"loaded": False}, loading_repo_ids = lambda: ())
    world.stubs[attribute].__dict__[attribute] = backend
    _admit().require_clear()
    setattr(backend, counter, 1)
    with pytest.raises(arbiter.AttachedAdmissionError, match = "still unloading"):
        _admit().require_clear()
    setattr(backend, counter, 0)
    _admit().require_clear()


def test_a_draining_diffusers_load_blocks_admission(world):
    draining = [("org/model",)]
    backend = NS(
        status = lambda: {"loaded": False},
        loading_repo_ids = lambda: (),
        draining_repo_ids = lambda: draining[0],
    )
    world.stubs["_diffusion_backend"]._diffusion_backend = backend
    with pytest.raises(arbiter.AttachedAdmissionError, match = "still unloading"):
        _admit().require_clear()
    draining[0] = ()
    _admit().require_clear()


# -- failed chat unload ------------------------------------------------------


class _Proc:
    pid = 424242

    def __init__(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout = None):
        pass


class _BrokenQueue:
    """The command queue's pipe is gone, so the unload command cannot be written."""

    def put(self, *_a, **_k):
        raise OSError("broken pipe")

    def get(self, *_a, **_k):
        import queue

        raise queue.Empty

    get_nowait = get


@pytest.fixture
def orchestrator(world, monkeypatch):
    import core.inference.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.time, "sleep", lambda *_a, **_k: None)
    orch = orch_mod.InferenceOrchestrator()
    orch._proc = _Proc()
    orch._cmd_queue = _BrokenQueue()
    orch._resp_queue = _BrokenQueue()
    orch.models["/models/Qwen-MLX"] = {"is_mlx": True}
    orch.active_model_name = "/models/Qwen-MLX"
    world.chat = orch
    yield orch
    # The orchestrator's atexit hook must find no worker to shut down.
    orch._proc = None


def test_a_failed_chat_unload_keeps_oMLX_out_until_the_worker_is_gone(world, orchestrator):
    import routes.inference as inf_mod

    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        _admit().require_clear()

    # The queue write fails: unload_model() reports False, drops the model name and leaves the worker running.
    assert orchestrator.unload_model("/models/Qwen-MLX") is False
    assert orchestrator.active_model_name is None
    assert orchestrator.is_worker_alive()
    assert inf_mod._attached_local_memory_active()
    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        _admit().require_clear()

    # The worker exits (or is killed): its weights are gone, so oMLX is admissible again.
    orchestrator._proc.alive = False
    assert not inf_mod._attached_local_memory_active()
    _admit().require_clear()


def test_a_worker_kept_alive_after_a_successful_unload_does_not_block_oMLX(world, orchestrator, monkeypatch):
    import routes.inference as inf_mod

    # The unload command is acknowledged; the worker stays up idle (for example a sidecar keeps it).
    class _Queue(_BrokenQueue):
        def put(self, *_a, **_k):
            pass

    orchestrator._cmd_queue = _Queue()
    monkeypatch.setattr(orchestrator, "_wait_response", lambda *_a, **_k: {"type": "unloaded"})
    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", lambda *_a, **_k: False)
    orchestrator.models["/models/other"] = {"is_mlx": True}
    assert orchestrator.unload_model("/models/Qwen-MLX") is True
    assert orchestrator.is_worker_alive()
    assert not inf_mod._attached_local_memory_active()
    _admit().require_clear()


def test_a_later_successful_unload_clears_the_failed_unload_claim(world, orchestrator, monkeypatch):
    import routes.inference as inf_mod

    assert orchestrator.unload_model("/models/Qwen-MLX") is False
    assert inf_mod._attached_local_memory_active()

    class _Queue(_BrokenQueue):
        def put(self, *_a, **_k):
            pass

    orchestrator._cmd_queue = _Queue()
    orchestrator.models["/models/Qwen-MLX"] = {"is_mlx": True}
    orchestrator.active_model_name = "/models/Qwen-MLX"
    monkeypatch.setattr(orchestrator, "_wait_response", lambda *_a, **_k: {"type": "unloaded"})
    monkeypatch.setattr(orchestrator, "_shutdown_subprocess", lambda *_a, **_k: False)
    assert orchestrator.unload_model("/models/Qwen-MLX") is True
    assert not inf_mod._attached_local_memory_active()
