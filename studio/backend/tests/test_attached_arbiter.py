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


def test_using_omlx_requires_no_peer_eviction(admission):
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
