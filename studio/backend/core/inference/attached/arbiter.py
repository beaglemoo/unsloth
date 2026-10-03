# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory arbiter for the attached engines. oMLX weights and the ds4 server share unified memory with whatever Studio loads or trains, so the engines are evicted before a local GGUF load or a training run, and ds4 is stopped before oMLX serves a chat. Everything here is a strict no-op unless the ``attached_engines`` flag is on, and it never raises into its caller: an engine that is down, slow or erroring is logged and skipped, since a failed eviction must not block the load or training the user asked for."""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional

from core.inference.attached.ds4_client import Ds4Client, Ds4Status
from core.inference.attached.omlx_client import OmlxClient
from loggers import get_logger
from utils.attached_engines_settings import get_config

logger = get_logger(__name__)

Reason = Literal["training", "local_load"]

_DS4_STATUS_TTL_S = 2.0
_THREAD_WAIT_S = 360.0

_notices: deque[dict[str, Any]] = deque(maxlen = 20)
_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)
_ds4_cache: Optional[tuple[float, Ds4Status]] = None


@dataclass(frozen = True)
class ArbiterResult:
    acted: bool = False
    actions: tuple[str, ...] = ()
    in_flight_killed: int = 0
    # Why nothing was done: "disabled", "arbitration_off" or "error"; None when it ran.
    skipped: Optional[str] = None


def _lock() -> asyncio.Lock:
    # One lock per event loop: an asyncio.Lock binds to the loop that first contends it.
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = _locks[loop] = asyncio.Lock()
    return lock


def _record(reason: str, actions: list[str], in_flight_killed: int) -> None:
    _notices.append(
        {
            "ts": time.time(),
            "reason": reason,
            "actions": list(actions),
            "in_flight_killed": in_flight_killed,
        }
    )


def recent_notices(since: float = 0) -> list[dict]:
    """Eviction notices newer than ``since`` (epoch seconds), oldest first."""
    return [dict(notice) for notice in list(_notices) if notice["ts"] > since]


async def _ds4_status(client: Ds4Client, *, fresh: bool) -> Ds4Status:
    global _ds4_cache
    now = time.monotonic()
    if not fresh and _ds4_cache is not None and now - _ds4_cache[0] < _DS4_STATUS_TTL_S:
        return _ds4_cache[1]
    status = await client.status()
    _ds4_cache = (time.monotonic(), status)
    return status


def _invalidate_ds4_cache() -> None:
    global _ds4_cache
    _ds4_cache = None


async def _stop_ds4(url: str, *, fresh: bool) -> tuple[list[str], int]:
    """Stop ds4 when it is loaded or starting. Returns (actions, in-flight requests killed)."""
    client = Ds4Client(url)
    status = await _ds4_status(client, fresh = fresh)
    if not status.reachable or not (status.loaded or status.starting):
        return [], 0
    killed = status.in_flight
    stopped = await client.stop()
    _invalidate_ds4_cache()
    if stopped.error is not None or stopped.loaded or stopped.starting:
        logger.warning("DwarfStar did not stop cleanly (%s)", stopped.error or "still running")
        return ["DwarfStar stop failed"], 0
    return ["Stopped DwarfStar"], killed


async def free_for_local(reason: Reason) -> ArbiterResult:
    """Free both engines for a local load or training run: stop ds4, then unload every loaded oMLX model, pinned ones included."""
    try:
        config = get_config()
        if not config.enabled:
            return ArbiterResult(skipped = "disabled")
        if not config.arbitrate_local_loads:
            return ArbiterResult(skipped = "arbitration_off")
        async with _lock():
            actions, killed = await _stop_ds4(config.ds4_url, fresh = True)
            unloaded = await OmlxClient(config.omlx_url).unload_all()
            if unloaded:
                actions.append(f"Unloaded oMLX: {', '.join(unloaded)}")
            if actions:
                _record(reason, actions, killed)
                logger.info("Attached engines freed for %s: %s", reason, "; ".join(actions))
            return ArbiterResult(
                acted = bool(actions), actions = tuple(actions), in_flight_killed = killed
            )
    except Exception as exc:  # noqa: BLE001 -- eviction must never fail the caller's load or training
        logger.warning("Attached engines could not be freed for %s: %s", reason, exc)
        return ArbiterResult(skipped = "error")


async def before_omlx_use() -> ArbiterResult:
    """Stop ds4 when it is loaded or starting, before oMLX serves a request."""
    try:
        config = get_config()
        if not config.enabled:
            return ArbiterResult(skipped = "disabled")
        async with _lock():
            actions, killed = await _stop_ds4(config.ds4_url, fresh = False)
            if actions:
                _record("omlx_use", actions, killed)
                logger.info("DwarfStar stopped before oMLX use: %s", "; ".join(actions))
            return ArbiterResult(
                acted = bool(actions), actions = tuple(actions), in_flight_killed = killed
            )
    except Exception as exc:  # noqa: BLE001 -- see free_for_local
        logger.warning("DwarfStar could not be stopped before oMLX use: %s", exc)
        return ArbiterResult(skipped = "error")


def free_for_local_from_thread(reason: Reason, loop: asyncio.AbstractEventLoop) -> ArbiterResult:
    """``free_for_local`` for a worker thread (the training spawn hook): runs it on the app's loop so the single lock holds, and waits. Never raises."""
    try:
        if not get_config().enabled:
            return ArbiterResult(skipped = "disabled")
        try:
            if asyncio.get_running_loop() is loop:
                # Waiting on the loop from the loop's own thread would deadlock it.
                logger.warning(
                    "Freeing attached engines for %s skipped: called on the loop", reason
                )
                return ArbiterResult(skipped = "error")
        except RuntimeError:
            pass
        future = asyncio.run_coroutine_threadsafe(free_for_local(reason), loop)
        try:
            return future.result(timeout = _THREAD_WAIT_S)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("Freeing attached engines for %s timed out", reason)
            return ArbiterResult(skipped = "error")
    except Exception as exc:  # noqa: BLE001 -- see free_for_local
        logger.warning("Attached engines could not be freed for %s: %s", reason, exc)
        return ArbiterResult(skipped = "error")
