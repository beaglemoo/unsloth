# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory admission for oMLX and Studio local workloads."""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional

import httpx

from core.inference.attached import AttachedEngineError
from core.inference.attached.omlx_client import OmlxClient
from loggers import get_logger
from utils.attached_engines_settings import get_config

logger = get_logger(__name__)
Reason = Literal["training", "local_load"]
_THREAD_WAIT_S = 360.0
_notices: deque[dict[str, Any]] = deque(maxlen = 20)
_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


class AttachedAdmissionError(AttachedEngineError):
    blocks_training = True


@dataclass(frozen = True)
class ArbiterResult:
    acted: bool = False
    actions: tuple[str, ...] = ()
    in_flight_killed: int = 0
    skipped: Optional[str] = None
    error: Optional[str] = None

    def require_clear(self) -> None:
        if self.error:
            raise AttachedAdmissionError(self.error)


def _lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = _locks[loop] = asyncio.Lock()
    return lock


def _record(reason: str, actions: list[str], in_flight_killed: int = 0) -> None:
    _notices.append(dict(ts = time.time(), reason = reason, actions = list(actions), in_flight_killed = in_flight_killed))


def recent_notices(since: float = 0) -> list[dict]:
    return [dict(notice) for notice in _notices if notice["ts"] > since]


async def free_comfyui(*, transport = None) -> None:
    url = os.environ.get("STUDIO_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    if not url:
        return
    try:
        async with httpx.AsyncClient(base_url = url, timeout = 3.0, trust_env = False, transport = transport) as client:
            response = await client.get("/queue")
            response.raise_for_status()
            queue = response.json()
            if not isinstance(queue, dict) or not isinstance(queue.get("queue_running"), list) or not isinstance(queue.get("queue_pending"), list):
                return
            if queue["queue_running"] or queue["queue_pending"]:
                return
            (await client.post("/free", json = {"unload_models": True, "free_memory": True})).raise_for_status()
    except Exception as exc:
        logger.info("ComfyUI memory release skipped: %s", type(exc).__name__)


def _training_active() -> bool:
    from core.training import get_training_backend

    return bool(get_training_backend().is_training_active())


def _local_memory_active() -> bool:
    from routes.inference import _attached_local_memory_active

    return _attached_local_memory_active()


def _local_workload() -> Optional[str]:
    """What Studio is running that oMLX must not share memory with, if anything."""
    if _training_active():
        return "is training"
    if _local_memory_active():
        return "has a local model loaded"
    return None


async def _admit(reason: str, *, local: bool) -> ArbiterResult:
    try:
        config = get_config()
        if not config.enabled:
            return ArbiterResult(skipped = "disabled")
        if local and not config.arbitrate_local_loads:
            return ArbiterResult(skipped = "arbitration_off")
        async with _lock():
            if not local and config.arbitrate_local_loads:
                busy = _local_workload()
                if busy:
                    raise AttachedAdmissionError(
                        f"Studio {busy}; finish or unload it to use oMLX."
                    )
            if local:
                await free_comfyui()
                unloaded = await OmlxClient(config.omlx_url).unload_all()
                actions = [f"Unloaded oMLX: {', '.join(unloaded)}"] if unloaded else []
                if actions:
                    _record(reason, actions)
                return ArbiterResult(acted = bool(actions), actions = tuple(actions))
            return ArbiterResult()
    except Exception as exc:
        message = f"Cannot free shared memory for {reason}: {exc}"
        logger.warning("%s", message)
        return ArbiterResult(skipped = "error", error = message)


async def free_for_local(reason: Reason) -> ArbiterResult:
    return await _admit(reason, local = True)


async def before_omlx_use() -> ArbiterResult:
    return await _admit("omlx_use", local = False)


def free_for_local_from_thread(reason: Reason, loop: asyncio.AbstractEventLoop) -> ArbiterResult:
    try:
        if not get_config().enabled:
            return ArbiterResult(skipped = "disabled")
        try:
            if asyncio.get_running_loop() is loop:
                return ArbiterResult(
                    skipped = "error",
                    error = "Memory admission cannot wait on its own event loop.",
                )
        except RuntimeError:
            pass
        future = asyncio.run_coroutine_threadsafe(free_for_local(reason), loop)
        try:
            return future.result(timeout = _THREAD_WAIT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            return ArbiterResult(skipped = "error", error = "Shared-memory admission timed out; training was blocked.")
    except Exception as exc:
        return ArbiterResult(skipped = "error", error = f"Shared-memory admission failed: {exc}")
