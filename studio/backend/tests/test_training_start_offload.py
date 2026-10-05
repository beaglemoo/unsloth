# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""start_training runs off the event loop, since its before_spawn unload can block for seconds."""

import asyncio
import contextlib
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import routes.training as tr
from models import TrainingStartRequest


class _FakeBackend:
    def __init__(self, result = True):
        self._result = result
        self.start_thread = None
        self.hook = None
        self.current_job_id = None

    def is_training_active(self):
        return False

    def start_training(
        self,
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        # before_spawn runs synchronously here, so this is the hook's thread.
        self.start_thread = threading.current_thread()
        self.hook = before_spawn
        self.current_job_id = job_id
        return self._result


class _HookBackend(_FakeBackend):
    def __init__(self, result = True):
        super().__init__(result)
        self.active = False
        self.trainer = SimpleNamespace(
            training_progress = SimpleNamespace(error = "spawn failed")
        )

    def is_training_active(self):
        return self.active

    def start_training(self, job_id, *, before_spawn = None, **kwargs):
        self.start_thread = threading.current_thread()
        self.hook = before_spawn
        self.current_job_id = job_id
        before_spawn()
        self.active = self._result
        return self._result


def _route_setup(monkeypatch, backend):
    monkeypatch.setattr(tr, "get_training_backend", lambda: backend)
    monkeypatch.setattr(tr, "_diffusion_training_active", lambda: False)
    monkeypatch.setattr(tr, "_diffusion_gpu_admission", contextlib.nullcontext)
    monkeypatch.setattr(
        tr,
        "_reject_untrainable_model_request",
        lambda request, *_args: SimpleNamespace(
            model_name = request.model_name,
            cached_model_pin = None,
            model_local_path = None,
        ),
    )
    monkeypatch.setattr(tr, "_preflight_hf_dataset_request", lambda *_args: None)
    monkeypatch.setattr("utils.hardware.ensure_hardware_detected", lambda: None)


def _request() -> TrainingStartRequest:
    return TrainingStartRequest(
        model_name = "unsloth/tiny-model",
        training_type = "LoRA/QLoRA",
        format_type = "alpaca",
        hf_dataset = "org/data",
        load_in_4bit = False,
        # Skip the YAML trust_remote_code lookup (needs the model catalog on disk).
        trust_remote_code = True,
    )


def test_start_route_offloads_blocking_start(monkeypatch):
    fake = _FakeBackend()
    _route_setup(monkeypatch, fake)

    async def _run():
        return threading.current_thread(), await tr.start_training(
            request = _request(), current_subject = "test-user", via_api_key = False
        )

    loop_thread, resp = asyncio.run(_run())

    assert resp.status == "queued", resp
    assert fake.hook is not None
    assert fake.start_thread is not None
    assert fake.start_thread is not loop_thread


@pytest.mark.parametrize("spawned,expected", [(True, "queued"), (False, "error")])
def test_training_admission_proceeds_without_peer_holds(monkeypatch, spawned, expected):
    from core.inference.attached import arbiter

    backend = _HookBackend(result = spawned)
    _route_setup(monkeypatch, backend)
    calls = []

    def admit(reason, loop):
        # Admission runs before the training worker begins using memory.
        assert backend.active is False
        calls.append(reason)
        return arbiter.ArbiterResult(acted = True, actions = ("Unloaded oMLX",))

    monkeypatch.setattr(arbiter, "free_for_local_from_thread", admit)
    response = asyncio.run(tr.start_training(
        request = _request(), current_subject = "test-user", via_api_key = False
    ))
    assert response.status == expected
    assert calls == ["training"]
    assert backend.active is spawned


def test_training_admission_failure_returns_retryable_clear_message(monkeypatch):
    from core.inference.attached import arbiter
    from core.inference.attached.arbiter import AttachedAdmissionError

    backend = _HookBackend()
    _route_setup(monkeypatch, backend)
    monkeypatch.setattr(
        arbiter,
        "free_for_local_from_thread",
        lambda *_args: SimpleNamespace(
            require_clear = lambda: (_ for _ in ()).throw(
                AttachedAdmissionError("oMLX is still loading")
            ),
        ),
    )

    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            tr.start_training(request = _request(), current_subject = "test-user", via_api_key = False)
        )

    error = caught.value
    assert error.status_code == 503
    assert error.headers["Retry-After"] == "15"
    assert "residency is cleared" in error.detail


def test_backend_start_guard_blocks_overlapping_starts():
    # Offloaded routes can overlap start_training; the reservation lets exactly one proceed.
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    first_entered = threading.Event()
    release_first = threading.Event()
    results = {}

    def _slow_impl(
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        first_entered.set()
        release_first.wait(timeout = 5.0)
        return True

    backend._start_training_with_lifecycle_reserved = _slow_impl

    def _first():
        results["first"] = backend.start_training("job-a")

    t = threading.Thread(target = _first, daemon = True)
    t.start()
    assert first_entered.wait(timeout = 5.0)
    results["second"] = backend.start_training("job-b")
    release_first.set()
    t.join(timeout = 5.0)

    assert results["first"] is True
    assert results["second"] is False
    assert backend._spawn_in_progress is False
    assert backend._new_job_spawn_id is None


def test_is_training_active_true_during_start_reservation():
    # Reserved-but-not-spawned must read active, or other pipelines race for freed VRAM.
    from core.training.training import TrainingBackend

    backend = TrainingBackend()
    assert backend.is_training_active() is False

    entered = threading.Event()
    release = threading.Event()
    captured = {}

    def _slow_impl(
        job_id,
        *,
        before_spawn = None,
        **kwargs,
    ):
        entered.set()
        release.wait(timeout = 5.0)
        return True

    backend._start_training_with_lifecycle_reserved = _slow_impl

    t = threading.Thread(target = lambda: backend.start_training("job-a"), daemon = True)
    t.start()
    assert entered.wait(timeout = 5.0)
    captured["in_window"] = backend.is_training_active()
    release.set()
    t.join(timeout = 5.0)

    assert captured["in_window"] is True
    assert backend.is_training_active() is False
