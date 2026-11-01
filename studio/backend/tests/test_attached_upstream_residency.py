# SPDX-License-Identifier: AGPL-3.0-only
"""oMLX admission follows live residency through upstream's coexistence and idle-unload features.

#15693 keeps an image or video model loaded beside a chat model on Apple Silicon, so the chat model going away no
longer means Studio holds nothing. #15680 idle-unloads an MLX chat model, after which nothing may keep oMLX out.
"""
from __future__ import annotations

import asyncio
import sys
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


class _Media:
    def __init__(self, loaded):
        self.loaded = loaded

    def status(self):
        return {"loaded": self.loaded}


class _Chat:
    """The orchestrator-shaped primary: the fields ``_attached_local_memory_active`` reads."""

    def __init__(self, name = None):
        self.active_model_name = name
        self.models = {name: {"is_mlx": True}} if name else {}
        self.loading_models = ()
        self._managed_engine = None


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
    state = NS(chat = _Chat(), stubs = {})
    for module, attribute in _MEDIA_MODULES:
        stub = types.SimpleNamespace(**{attribute: None})
        monkeypatch.setitem(sys.modules, module, stub)
        state.stubs[attribute] = stub
    monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: state.chat)
    yield state
    arbiter._media_holds.clear()
    arbiter._omlx_holds.clear()


async def _no_blocker():
    return None


def _set_media(state, attribute, backend):
    setattr(state.stubs[attribute], attribute, backend)


def _admit():
    return asyncio.run(arbiter.before_omlx_use())


def test_media_left_beside_chat_keeps_omlx_out_until_it_is_unloaded(world):
    import routes.inference as inf_mod

    # #15693: the MLX image engine stays resident beside the chat model.
    world.chat = _Chat("/models/Qwen-MLX")
    _set_media(world, "_mlx_diffusion_backend", _Media(True))
    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        _admit().require_clear()

    # The chat model goes away; the media model still holds unified memory, so oMLX stays out.
    world.chat = _Chat()
    assert not inf_mod._attached_local_memory_active()
    with pytest.raises(arbiter.AttachedAdmissionError, match = "image model loaded"):
        _admit().require_clear()

    _set_media(world, "_mlx_diffusion_backend", _Media(False))
    _admit().require_clear()


@pytest.mark.parametrize(
    "attribute, noun",
    [
        ("_diffusion_backend", "image model"),
        ("_sd_cpp_backend", "image model"),
        ("_backend", "video model"),
    ],
)
def test_every_media_engine_left_beside_chat_counts_as_resident(world, attribute, noun):
    world.chat = _Chat("/models/Qwen-MLX")
    _set_media(world, attribute, _Media(True))
    world.chat = _Chat()
    with pytest.raises(arbiter.AttachedAdmissionError, match = f"{noun} loaded"):
        _admit().require_clear()


def test_idle_unload_of_an_mlx_chat_model_makes_omlx_admissible_again(world, monkeypatch):
    # #15680: the idle loop frees an API-loaded MLX primary; the arbiter reads live state, so oMLX is open again.
    import routes.inference as inf_mod
    from core.inference import llama_keepwarm as kw
    from utils import openai_auto_switch_settings as settings

    unloads = []

    class Orchestrator(_Chat):
        _openai_advertised_id = None
        _loaded_by_user_action = False

        def save_prompt_cache_for_resume(self, should_abort):
            return None

        def unload_model(self, name):
            unloads.append(name)
            self.models.pop(name, None)
            self.active_model_name = None
            return True

    world.chat = Orchestrator("/models/Qwen-MLX")
    llama = NS(
        is_active = False,
        is_loaded = False,
        model_identifier = None,
        _loaded_by_user_action = False,
        hf_variant = None,
    )
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: llama)
    monkeypatch.setattr(inf_mod, "release_chat_gpu_claim", lambda: None)
    monkeypatch.setattr(settings, "get_auto_unload_idle_seconds", lambda: 0.005)
    monkeypatch.setattr(settings, "get_auto_unload_api_only", lambda: False)
    monkeypatch.setattr(settings, "get_auto_unload_keep_kv", lambda: False)
    monkeypatch.setattr(kw, "_inflight", 0)
    monkeypatch.setattr(kw, "_pending", 0)
    monkeypatch.setattr(kw, "_last_active", kw.time.monotonic() - 3600)
    monkeypatch.setattr(kw, "_last_unloaded_model", None)
    monkeypatch.setattr(kw, "_last_unloaded_mlx", False)
    monkeypatch.setattr(kw, "_kv_resume", None)

    assert inf_mod._attached_local_memory_active()
    with pytest.raises(arbiter.AttachedAdmissionError, match = "local model loaded"):
        _admit().require_clear()

    async def drive():
        task = asyncio.create_task(kw.idle_unload_loop(poll_seconds = 0.02))
        for _ in range(500):
            if unloads:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(drive())

    assert unloads == ["/models/Qwen-MLX"]
    assert not inf_mod._attached_local_memory_active()
    _admit().require_clear()
