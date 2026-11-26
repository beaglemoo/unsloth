# SPDX-License-Identifier: AGPL-3.0-only
"""The chat image swap takes part in attached-engine arbitration for its whole length.

The image load (``load_diffusion_model_gated``) and the chat reload (``_load_model_impl``) are admitted on their own,
but chat is out of memory between the two and a resident chat model is what keeps oMLX out. The swap therefore takes a
media hold before chat is evicted and releases it after chat is back; a refusal changes nothing.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference import chat_media_swap as swap
from core.inference.attached import arbiter
from utils.attached_engines_settings import DEFAULT_CONFIG

from .test_chat_media_swap import KV, FakeChat, FakeImage, Log, _trip


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def world(monkeypatch):
    """The real arbiter with a fake oMLX; the flag is on and Studio holds no local model while chat is out."""
    state = {"omlx_unloads": 0, "chat_resident": True}
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(arbiter, "_training_active", lambda: False)
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: state["chat_resident"])
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: None)

    async def no_comfyui(*args, **kwargs):
        return []

    async def no_stt():
        return None

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            state["omlx_unloads"] += 1
            return ["swift"]

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(arbiter, "free_comfyui", no_comfyui)
    monkeypatch.setattr(arbiter, "_stt_blocker", no_stt)
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda config: (None, []))
    arbiter._media_holds.clear()
    arbiter._omlx_holds.clear()
    arbiter._media_probe_future = None
    yield state
    arbiter._media_holds.clear()
    arbiter._omlx_holds.clear()


class WatchedChat(FakeChat):
    """Records whether oMLX would be admitted at each point where chat is out of memory."""

    def __init__(self, log, world, probes, **kw):
        super().__init__(log, **kw)
        self.world = world
        self.probes = probes

    async def _probe(self, where):
        self.world["chat_resident"] = self.resident
        result = await arbiter.before_omlx_use()
        self.probes.append((where, result.error))

    async def unload(self, snap):
        await super().unload(snap)
        await self._probe("after_unload")

    async def reload(self, snap, kv):
        await self._probe("before_reload")
        await super().reload(snap, kv)
        self.world["chat_resident"] = True


class WatchedImage(FakeImage):
    def __init__(self, log, world, probes, **kw):
        super().__init__(log, **kw)
        self.world = world
        self.probes = probes

    async def unload(self):
        await super().unload()
        result = await arbiter.before_omlx_use()
        self.probes.append(("after_image_unload", result.error))


def _swap(world, **kw):
    log, probes = Log(), []
    chat = WatchedChat(log, world, probes)
    media = WatchedImage(log, world, probes, chat = chat, **kw)
    trip = _trip(chat, media, admit = swap._admit_attached_swap)
    return trip, chat, media, log, probes


def test_oMLX_is_refused_for_the_whole_swap_and_admitted_again_after_it(world):
    trip, chat, media, log, probes = _swap(world)
    result = run(trip.run())
    assert result.startswith("Generated")
    assert [where for where, _ in probes] == ["after_unload", "after_image_unload", "before_reload"]
    assert all(error and "image model" in error for _, error in probes), probes
    assert chat.resident and arbiter._media_holds == set()
    world["chat_resident"] = True
    assert run(arbiter.before_omlx_use()).error  # chat is resident again: refused for the usual reason
    world["chat_resident"] = False
    assert run(arbiter.before_omlx_use()).error is None


def test_the_claim_is_taken_before_chat_is_touched(world):
    trip, chat, media, log, probes = _swap(world)
    seen = {}
    real_save = chat.save_kv

    async def save_kv(snap, should_abort):
        seen["claim"] = arbiter._media_held()
        return await real_save(snap, should_abort)

    chat.save_kv = save_kv
    run(trip.run())
    assert seen == {"claim": "is loading an image model"}
    assert world["omlx_unloads"] == 1


def test_a_failed_render_still_restores_chat_and_releases_the_claim(world):
    trip, chat, media, log, probes = _swap(world, fail = {"generate"})
    result = run(trip.run())
    assert result.startswith("Error: image generation failed")
    assert chat.resident and chat.loaded_with[2] == KV
    assert arbiter._media_holds == set()


def test_a_refused_admission_leaves_chat_loaded_and_registers_nothing(world, monkeypatch):
    from fastapi import HTTPException

    async def refuse(media_hold = None):
        raise HTTPException(status_code = 503, detail = "oMLX is serving a request", headers = {"Retry-After": "15"})

    import routes.inference as inference

    monkeypatch.setattr(inference, "_admit_attached_local_load", refuse)
    trip, chat, media, log, probes = _swap(world)
    with pytest.raises(swap.SwapUnavailable, match = "shared memory could not be freed for the image model.*oMLX is serving"):
        run(trip.run())
    assert log == [] and chat.resident and arbiter._media_holds == set()


def test_a_refusal_from_the_real_arbiter_leaves_chat_alone(world):
    hold = arbiter.new_omlx_hold()
    world["chat_resident"] = False
    assert run(arbiter.before_omlx_use(hold)).error is None
    world["chat_resident"] = True
    trip, chat, media, log, probes = _swap(world)
    with pytest.raises(swap.SwapUnavailable, match = "shared memory could not be freed"):
        run(trip.run())
    assert log == [] and chat.resident and arbiter._media_holds == set()
    assert world["omlx_unloads"] == 0
    hold.release()


def test_with_attached_engines_off_the_swap_registers_nothing(world, monkeypatch):
    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    trip, chat, media, log, probes = _swap(world)
    seen = {}
    real_save = chat.save_kv

    async def save_kv(snap, should_abort):
        seen["claim"] = arbiter._media_held()
        return await real_save(snap, should_abort)

    chat.save_kv = save_kv
    assert run(trip.run()).startswith("Generated")
    assert seen == {"claim": None} and arbiter._media_holds == set()
    assert world["omlx_unloads"] == 0
