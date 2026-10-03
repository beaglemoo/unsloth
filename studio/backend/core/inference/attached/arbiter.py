# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Fail-closed memory admission for engines sharing Studio's unified memory."""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
import time
import weakref
from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Optional

import httpx

from core.inference.attached import AttachedEngineError
from core.inference.attached.ds4_client import Ds4Client
from core.inference.attached.omlx_client import OmlxClient
from loggers import get_logger
from utils.attached_engines_settings import get_config

logger = get_logger(__name__)
Reason = Literal["training", "local_load"]
_THREAD_WAIT_S = 360.0
HOLD_TTL_S = 120
HOLD_RENEW_S = 60
_notices: deque[dict[str, Any]] = deque(maxlen = 20)
_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_leases: set[HoldLease] = set()


class AttachedAdmissionError(AttachedEngineError):
    # The training backend must not swallow this before_spawn failure.
    blocks_training = True


class HoldLease:
    """A crash-safe ds4 hold, renewed without a gap until its owner ends.

    The lifetime watcher runs independently of HTTP clients/polling, including
    failed spawns, idle model eviction and disconnects. Pending loads keep their
    lease until teardown completes, then arm the residency watcher.
    """

    def __init__(self, client: Ds4Client, hold_id: str, reason: str):
        self.client, self.hold_id, self.reason = client, hold_id, reason
        self.closed = False
        self.error: Optional[str] = None
        self._mutex = asyncio.Lock()
        self._watcher: Optional[asyncio.Task] = None
        self._renewer = asyncio.create_task(self._renew(), name = f"ds4-hold-{reason}")
        _leases.add(self)

    @classmethod
    async def create(cls, url: str, reason: str) -> HoldLease:
        client = Ds4Client(url)
        return cls(client, await client.hold(f"Studio {reason}", HOLD_TTL_S), reason)

    async def renew(self) -> None:
        async with self._mutex:
            if self.closed:
                return
            replacement = await self.client.hold(f"Studio {self.reason}", HOLD_TTL_S)
            previous, self.hold_id = self.hold_id, replacement
            # A failed delete leaves only a bounded extra hold, never an admission gap.
            try:
                await self.client.release_hold(previous)
            except AttachedEngineError as exc:
                logger.warning("%s", exc)
            self.error = None

    async def _renew(self) -> None:
        delay = HOLD_RENEW_S
        while not self.closed:
            await asyncio.sleep(delay)
            try:
                await self.renew()
                delay = HOLD_RENEW_S
            except AttachedEngineError as exc:
                self.error = str(exc)
                logger.error("Studio %s hold renewal failed: %s", self.reason, exc)
                # Retry while the original TTL is still valid. Local admissions
                # also consult _leases and never bypass a failed renewal.
                delay = 5

    async def watch_until(self, predicate) -> None:
        if self.closed or self._watcher is not None:
            return

        async def watch():
            while not self.closed:
                try:
                    active = predicate()
                    if inspect.isawaitable(active):
                        active = await active
                    if not active:
                        await self.release()
                        return
                except Exception:
                    # Unknown residency is not proof it is safe to drop the
                    # hold. Retry so a transient failure cannot strand it.
                    logger.exception("Studio %s residency probe failed", self.reason)
                await asyncio.sleep(1)

        self._watcher = asyncio.create_task(watch(), name = f"ds4-residency-{self.reason}")

    async def release(self) -> None:
        async with self._mutex:
            if self.closed:
                return
            self.closed = True
            _leases.discard(self)
            for task in (self._renewer, self._watcher):
                if task is not None and task is not asyncio.current_task():
                    task.cancel()
            try:
                await self.client.release_hold(self.hold_id)
            except AttachedEngineError as exc:
                logger.warning("%s (expires within %ss)", exc, HOLD_TTL_S)


@dataclass(frozen = True)
class ArbiterResult:
    acted: bool = False
    actions: tuple[str, ...] = ()
    in_flight_killed: int = 0
    skipped: Optional[str] = None
    error: Optional[str] = None
    lease: Optional[HoldLease] = None

    def require_clear(self) -> None:
        if self.error:
            raise AttachedAdmissionError(self.error)


def _lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _locks.get(loop)
    if lock is None:
        lock = _locks[loop] = asyncio.Lock()
    return lock


def _record(reason: str, actions: list[str], in_flight_killed: int) -> None:
    _notices.append(dict(ts = time.time(), reason = reason, actions = list(actions),
                         in_flight_killed = in_flight_killed))


def recent_notices(since: float = 0) -> list[dict]:
    return [dict(notice) for notice in list(_notices) if notice["ts"] > since]


async def free_comfyui(*, transport = None) -> None:
    """Best effort only: leave queued/running work alone, never block admission."""
    url = os.environ.get("STUDIO_COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    if not url:
        return
    try:
        async with httpx.AsyncClient(base_url = url, timeout = 3.0,
                                     trust_env = False, transport = transport) as client:
            response = await client.get("/queue")
            response.raise_for_status()
            queue = response.json()
            # Missing/malformed queue data is not evidence that ComfyUI is idle.
            if (not isinstance(queue, dict)
                or not isinstance(queue.get("queue_running"), list)
                or not isinstance(queue.get("queue_pending"), list)):
                return
            if queue["queue_running"] or queue["queue_pending"]:
                logger.info("Leaving ComfyUI resident: running or queued work")
                return
            response = await client.post("/free", json = {"unload_models": True, "free_memory": True})
            response.raise_for_status()
    except Exception as exc:
        logger.info("ComfyUI memory release skipped: %s", type(exc).__name__)


async def _stop_ds4(url: str) -> tuple[list[str], int]:
    client = Ds4Client(url)
    status = await client.status()
    if not status.reachable:
        raise AttachedAdmissionError("Cannot verify DwarfStar residency: launcher is unreachable.")
    if status.starting or status.in_flight:
        raise AttachedAdmissionError("DwarfStar is starting or replying. Wait until it is idle, then retry.")
    actions = []
    if status.loaded:
        status = await client.stop(if_idle = True)
        if status.error:
            raise AttachedAdmissionError(f"DwarfStar eviction failed ({status.error}). Retry when it is idle.")
        actions.append("Stopped DwarfStar")
    # A second read catches a start that raced the initial idle status. The
    # ds4 hold prevents any *new* start; one already starting must block us.
    status = await client.status()
    if not status.reachable or status.loaded or status.starting:
        raise AttachedAdmissionError("DwarfStar is still loaded or starting; memory admission was blocked.")
    return actions, 0


async def _admit(reason: str, *, local: bool) -> ArbiterResult:
    lease = None
    actions: list[str] = []
    try:
        config = get_config()
        if not config.enabled:
            return ArbiterResult(skipped = "disabled")
        if local and not config.arbitrate_local_loads:
            return ArbiterResult(skipped = "arbitration_off")
        async with _lock():
            conflicts = [l for l in _leases if not l.closed and
                         ((local and l.reason == "omlx_use") or
                          (not local and l.reason in ("training", "local_load")))]
            if conflicts:
                raise AttachedAdmissionError("Studio is using shared memory for " + conflicts[0].reason + ". Retry after it finishes or unloads.")
            lease = await HoldLease.create(config.ds4_url, reason)
            if local:
                await free_comfyui()
            actions, killed = await _stop_ds4(config.ds4_url)
            if local:
                unloaded = await OmlxClient(config.omlx_url).unload_all()
                if unloaded:
                    actions.append(f"Unloaded oMLX: {', '.join(unloaded)}")
            if actions:
                _record(reason, actions, killed)
            return ArbiterResult(acted = bool(actions), actions = tuple(actions),
                                 in_flight_killed = killed, lease = lease)
    except BaseException as exc:
        if lease is not None:
            await lease.release()
        if not isinstance(exc, Exception):
            raise
        message = f"Cannot free shared memory for {reason}: {exc}"
        logger.warning("%s", message)
        return ArbiterResult(acted = bool(actions), actions = tuple(actions),
                             skipped = "error", error = message)


async def release_local_if_idle(predicate) -> None:
    """Manual unload releases immediately; the watcher covers automatic eviction."""
    leases = [lease for lease in _leases if lease.reason == "local_load" and not lease.closed]
    if leases and not predicate():
        for lease in leases:
            await lease.release()


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
                return ArbiterResult(skipped = "error", error = "Memory admission cannot wait on its own event loop.")
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
