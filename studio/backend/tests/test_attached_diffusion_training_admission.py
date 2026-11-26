# SPDX-License-Identifier: AGPL-3.0-only
"""Diffusion (Images) LoRA training, the LTX families included, takes part in attached-engine arbitration.

It starts from ``POST /diffusion/start``, not ``start_training``, so ``_free_vram_for_training`` never ran for it: oMLX
stayed loaded beside the trainer, and oMLX could be used while it ran because ``arbiter._training_active`` read only
the LLM trainer.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.inference.attached import arbiter
from utils.attached_engines_settings import DEFAULT_CONFIG

from .test_diffusion_training import _BODY, client  # noqa: F401

START = "/api/train/diffusion/start"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def order(client, monkeypatch):  # noqa: F811
    """Records the order of the admission, the GPU free and the service calls."""
    import routes.training as tr

    seen: list = []

    async def admit(reason, **kwargs):
        seen.append(("admit", reason, list(client._fake.calls)))
        return arbiter.ArbiterResult()

    monkeypatch.setattr(arbiter, "free_for_local", admit)
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: seen.append(("free",)))
    return seen


def test_a_diffusion_training_start_is_admitted_as_training_after_it_reserves_and_before_it_frees(client, order):
    response = client.post(START, json = _BODY)
    assert response.status_code == 200, response.text
    assert [entry[0] for entry in order] == ["admit", "free"]
    assert order[0][1] == "training"
    assert order[0][2][0] == "reserve" and "start" not in order[0][2]


def test_a_refusal_is_a_retryable_503_that_frees_and_starts_nothing(client, order, monkeypatch):
    async def refuse(reason, **kwargs):
        order.append(("admit", reason))
        return arbiter.ArbiterResult(skipped = "error", error = "oMLX is serving a request")

    monkeypatch.setattr(arbiter, "free_for_local", refuse)
    response = client.post(START, json = _BODY)
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "15"
    assert "oMLX is serving a request" in response.json()["detail"]
    assert ("free",) not in order
    assert "start" not in client._fake.calls and "unreserve" in client._fake.calls


def test_a_foreign_export_refuses_before_oMLX_is_unloaded(client, order, monkeypatch):
    import routes.training as tr

    def foreign():
        raise RuntimeError("Another account's export is running.")

    monkeypatch.setattr(tr, "_refuse_a_foreign_export", foreign)
    response = client.post(START, json = _BODY)
    assert response.status_code == 409
    assert order == []


def test_nothing_is_asked_of_oMLX_while_attached_engines_are_off(client, monkeypatch):
    """The real arbiter, flag off: the start behaves as it did before the hook."""
    import routes.training as tr

    monkeypatch.setattr(arbiter, "get_config", lambda: DEFAULT_CONFIG)
    unloads = []

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            unloads.append(1)
            return []

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    assert client.post(START, json = _BODY).status_code == 200
    assert unloads == []


def test_the_real_arbiter_unloads_oMLX_for_a_diffusion_training_start(client, monkeypatch):
    import routes.training as tr

    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda config: (None, []))
    unloads = []

    class Omlx:
        def __init__(self, url):
            pass

        async def unload_all(self):
            unloads.append(1)
            return ["swift"]

    async def no_comfyui(*args, **kwargs):
        return []

    monkeypatch.setattr(arbiter, "OmlxClient", Omlx)
    monkeypatch.setattr(arbiter, "free_comfyui", no_comfyui)
    monkeypatch.setattr(tr, "_free_gpu_for_diffusion_training", lambda: None)
    arbiter._omlx_holds.clear()
    assert client.post(START, json = _BODY).status_code == 200
    assert unloads == [1]


# --- the other direction: oMLX and ComfyUI are refused while diffusion training runs ----------------------------


@pytest.fixture
def studio(monkeypatch):
    import core.training as training
    import core.training.diffusion_training_service as service_module

    state = NS(llm = False, diffusion = False)
    monkeypatch.setattr(training, "get_training_backend", lambda: NS(is_training_active = lambda: state.llm))
    monkeypatch.setattr(
        service_module, "_service", NS(is_active = lambda: state.diffusion), raising = False
    )
    monkeypatch.setattr(arbiter, "get_config", lambda: replace(DEFAULT_CONFIG, enabled = True))
    monkeypatch.setattr(arbiter, "_local_memory_active", lambda: False)
    monkeypatch.setattr(arbiter, "_studio_media_resident", lambda: None)
    monkeypatch.setattr(arbiter, "_comfyui_targets", lambda config: (None, []))
    monkeypatch.setattr(arbiter, "_stt_blocker", AsyncMock(return_value = None))
    arbiter._omlx_holds.clear()
    arbiter._media_probe_future = None
    return state


def test_a_running_diffusion_training_refuses_oMLX_and_a_comfyui_job(studio):
    studio.diffusion = True
    assert arbiter._training_active()
    omlx = run(arbiter.before_omlx_use())
    assert omlx.error and "is training" in omlx.error
    job = run(arbiter.before_comfyui_job())
    assert job.error and "is training" in job.error
    studio.diffusion = False
    assert not arbiter._training_active()
    assert run(arbiter.before_omlx_use()).error is None


def test_an_llm_training_run_still_counts(studio):
    studio.llm = True
    assert arbiter._training_active()


def test_a_service_that_was_never_built_is_not_training(studio, monkeypatch):
    import core.training.diffusion_training_service as service_module

    monkeypatch.setattr(service_module, "_service", None)
    assert not arbiter._training_active()
    monkeypatch.delitem(sys.modules, "core.training.diffusion_training_service")
    assert not arbiter._training_active()
