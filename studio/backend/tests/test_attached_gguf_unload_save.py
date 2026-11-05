# SPDX-License-Identifier: AGPL-3.0-only
"""A GGUF unload from the picker saves its chat cache first (up to 45 s, upstream #16493).

The server stays up for the whole save, so the model must keep counting as a local resident: oMLX and ComfyUI jobs
are refused from the first byte of the save until the teardown finished.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import core.inference.chat_swap_kv as chat_swap_kv
import core.inference.llama_cpp as llama_mod
from core.inference import model_slots
from core.inference.attached import arbiter
from models.inference import UnloadRequest
from utils.attached_engines_settings import DEFAULT_CONFIG

from test_active_generations import _route_gate, _stub_unload_backends  # noqa: E402


async def _no_blocker():
    return None


def _no_media():
    return None


@pytest.fixture
def world(monkeypatch):
    _route_gate()
    config = replace(DEFAULT_CONFIG, enabled = True)
    monkeypatch.setattr(arbiter, "get_config", lambda: config)
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_stt_blocker", _no_blocker)
    monkeypatch.setattr(arbiter, "_studio_media_resident", _no_media)
    monkeypatch.setattr(arbiter, "_media_probe_future", None)
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda cfg: (None, ()))
    monkeypatch.setattr(llama_mod, "chat_load_active", lambda: False)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(model_slots, "stuck", [])
    monkeypatch.setattr(model_slots, "loading", None)

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            return []

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    arbiter._omlx_holds.clear()
    arbiter._media_holds.clear()
    arbiter._comfyui_holds.clear()
    yield
    arbiter._omlx_holds.clear()
    arbiter._media_holds.clear()
    arbiter._comfyui_holds.clear()


def test_the_model_stays_resident_and_refuses_oMLX_and_ComfyUI_through_the_save_and_the_teardown(
    world, monkeypatch
):
    import routes.inference as inf_mod

    llama = NS(is_active = True, is_loaded = True, model_identifier = "org/A-GGUF")
    seen = {}

    async def observe(label):
        seen[label] = {
            "resident": inf_mod._attached_local_memory_active(),
            "omlx": (await arbiter.before_omlx_use()).error,
            "comfyui": (await arbiter.before_comfyui_job()).error,
        }

    async def save():
        await observe("save")

    def teardown():
        seen["teardown_started"] = inf_mod._attached_local_memory_active()
        llama.is_active = llama.is_loaded = False

    llama.unload_model = teardown
    _stub_unload_backends(
        monkeypatch,
        llama = llama,
        backend = NS(
            get_loading_model = lambda: None,
            unload_model = lambda path: None,
            active_model_name = None,
            models = {},
            loading_models = (),
            _managed_engine = None,
        ),
    )
    monkeypatch.setattr(inf_mod, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(chat_swap_kv, "save_chat_kv_before_unload", save)

    async def drive():
        await inf_mod.unload_model(UnloadRequest(model_path = "org/A-GGUF"), "tester")
        await observe("after")

    asyncio.run(drive())

    assert seen["save"]["resident"] is True
    assert seen["save"]["omlx"], "oMLX must be refused while the chat cache is being saved"
    assert seen["save"]["comfyui"], "a ComfyUI job must be refused while the chat cache is being saved"
    assert seen["teardown_started"] is True
    assert seen["after"] == {"resident": False, "omlx": None, "comfyui": None}
