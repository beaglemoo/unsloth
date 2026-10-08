# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Memory admission for oMLX and Studio local workloads."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Literal, Optional
from urllib.parse import urlsplit

from core.inference.attached import AttachedEngineBusy, AttachedEngineError, desktop_settings
from core.inference.attached.comfyui_client import ComfyuiClient
from core.inference.attached.failures import engines_home
from core.inference.attached.omlx_client import OmlxClient
from loggers import get_logger
from utils.attached_engines_settings import AttachedEnginesConfig, get_config

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


_COMFYUI_BUSY_TTL_S = 1.0
# url -> (monotonic time, state: down/idle/busy); only the per-chat-request probe reads it.
_busy_cache: dict[str, tuple[float, str]] = {}


def _comfyui_targets(config: AttachedEnginesConfig) -> tuple[Optional[str], list[str]]:
    """Studio's own ComfyUI and its peers. Env ``STUDIO_COMFYUI_URL`` replaces all of them: set means only that URL, empty means none."""
    override = os.environ.get("STUDIO_COMFYUI_URL")
    if override is not None:
        url = override.strip().rstrip("/")
        return (url or None), []
    peers = [url for url in config.comfyui_peer_urls if url != config.comfyui_url]
    return config.comfyui_url, peers


async def _free_if_idle(url: str, transport = None) -> bool:
    """Free one ComfyUI's memory, only when its queue is confirmed empty. Never raises."""
    try:
        client = ComfyuiClient(url, transport = transport)
        queue = await client.queue()
        if not queue.reachable or queue.busy:
            return False
        await client.free()
        return True
    except Exception as exc:
        logger.info("ComfyUI memory release skipped: %s", type(exc).__name__)
        return False


async def _free_urls(urls: list[str], transport = None) -> list[str]:
    if not urls:
        return []
    results = await asyncio.gather(*(_free_if_idle(url, transport) for url in urls))
    return [url for url, freed in zip(urls, results) if freed]


async def free_comfyui(*, transport = None) -> list[str]:
    """Free Studio's ComfyUI and its peers (StoryPress) where idle; returns the URLs freed. Failures never raise."""
    own, peers = _comfyui_targets(get_config() if os.environ.get("STUDIO_COMFYUI_URL") is None else None)
    return await _free_urls([url for url in (own, *peers) if url], transport)


async def _comfyui_state(url: str, *, cached: bool = False) -> str:
    """``down`` (confirmed not running), ``idle``, ``busy`` or ``unknown`` (no usable answer from a server that may be up).
    ``unknown`` is never cached: a blip must not pin an answer for the TTL."""
    now = time.monotonic()
    if cached:
        hit = _busy_cache.get(url)
        if hit is not None and now - hit[0] < _COMFYUI_BUSY_TTL_S:
            return hit[1]
    try:
        state = (await ComfyuiClient(url).queue()).state
    except Exception:
        return "unknown"
    if cached and state != "unknown":
        _busy_cache[url] = (now, state)
    return state


async def _comfyui_refusal(config: AttachedEnginesConfig, action: str, *, cached: bool = False) -> Optional[str]:
    """Why Studio must not proceed with ``action`` while ComfyUI runs: Studio's own ComfyUI or any configured peer
    (StoryPress) is generating, or did not answer its queue check. A busy server is never interrupted or freed."""
    own, peers = _comfyui_targets(config)
    urls = [url for url in (own, *peers) if url]
    if not urls:
        return None
    states = await asyncio.gather(*(_comfyui_state(url, cached = cached) for url in urls))
    for url, state in zip(urls, states):
        if state not in ("busy", "unknown"):
            continue
        label = "ComfyUI" if url == own else f"The ComfyUI at {url}"
        if state == "busy":
            return f"{label} is generating; wait for or cancel the job {action}."
        return f"{label} did not report its queue, so it may be generating; retry shortly {action}."
    return None


class MediaHold:
    """Studio's claim on a media load from admission until the backend publishes its own loading state.

    ``begin_load`` returns before the model is resident, and the route validates and selects an engine
    between admission and ``begin_load``. Without a claim an oMLX request admitted in that window would
    reload beside the media model. The claim is taken under the arbiter lock, so a concurrent oMLX
    admission either ran first (the media load is then refused or has unloaded it) or sees the claim.
    ``release`` is idempotent; the route calls it in a ``finally``.
    """

    __slots__ = ("kind", "active")

    def __init__(self, kind: str):
        self.kind = kind
        self.active = False

    def release(self) -> None:
        with _media_holds_lock:
            self.active = False
            _media_holds.discard(self)


_media_holds: set[MediaHold] = set()
_media_holds_lock = threading.Lock()


def new_media_hold(kind: str) -> MediaHold:
    """An unregistered claim for an ``image`` or ``video`` load; admission registers it."""
    return MediaHold(kind)


def _register_media_hold(hold: MediaHold) -> None:
    with _media_holds_lock:
        hold.active = True
        _media_holds.add(hold)


def _media_held() -> Optional[str]:
    with _media_holds_lock:
        kinds = {hold.kind for hold in _media_holds}
    if not kinds:
        return None
    return "is loading a video model" if kinds == {"video"} else "is loading an image model"


class ComfyJobHold:
    """Studio's claim on a ComfyUI job from admission until ComfyUI is confirmed done with it.

    ``before_comfyui_job`` returns before the websocket is open and the prompt is submitted, so for that
    gap (and for the time a cancel or timeout needs to take effect) ComfyUI's queue says nothing. The
    claim is registered under the arbiter lock on admission, so an oMLX or local-load admission either ran
    first (the job then unloaded or refused it) or sees the claim and is refused. ``release`` is
    idempotent; the job runner calls it in a ``finally`` and from its reaper.
    """

    __slots__ = ("active",)

    def __init__(self):
        self.active = False

    def release(self) -> None:
        with _comfyui_holds_lock:
            self.active = False
            _comfyui_holds.discard(self)


_comfyui_holds: set[ComfyJobHold] = set()
_comfyui_holds_lock = threading.Lock()


def new_comfyui_hold() -> ComfyJobHold:
    """An unregistered claim for a Studio ComfyUI job; ``before_comfyui_job`` registers it on admission."""
    return ComfyJobHold()


def _register_comfyui_hold(hold: ComfyJobHold) -> None:
    with _comfyui_holds_lock:
        hold.active = True
        _comfyui_holds.add(hold)


def _comfyui_held() -> bool:
    with _comfyui_holds_lock:
        return bool(_comfyui_holds)


def _studio_media_resident() -> Optional[str]:
    """Which Studio image or video model is resident, loading, or about to load, if any.
    Reads loaded modules only: nothing is imported or constructed."""
    for module_name, attribute, noun in (
        ("core.inference.diffusion", "_diffusion_backend", "an image model"),
        ("core.inference.sd_cpp_backend", "_sd_cpp_backend", "an image model"),
        ("core.inference.video", "_backend", "a video model"),
    ):
        backend = getattr(sys.modules.get(module_name), attribute, None)
        if backend is None:
            continue
        try:
            if backend.status().get("loaded"):
                return f"has {noun} loaded"
            loading_repo_ids = getattr(backend, "loading_repo_ids", None)
            if callable(loading_repo_ids) and loading_repo_ids():
                return f"is loading {noun}"
        except Exception:
            continue
    return _media_held()


_MEDIA_PROBE_TIMEOUT_S = 3.0
_MEDIA_BUSY = "is busy loading or unloading an image or video model"
# One worker and one probe in flight: a backend lock held through a long load can strand at most this thread,
# never one per request, and the default executor the routes share is never used.
_media_probe_pool = concurrent.futures.ThreadPoolExecutor(max_workers = 1, thread_name_prefix = "arbiter-media-probe")
_media_probe_future: Optional[concurrent.futures.Future] = None


async def _media_resident_async() -> Optional[str]:
    """``_studio_media_resident`` for admission, bounded to ``_MEDIA_PROBE_TIMEOUT_S`` and never queueing threads.

    A held claim answers at once with no thread hop. Otherwise the probe runs on a dedicated single-worker pool
    (the backends' loading probes take their own lock, which a load can hold). While an earlier probe is still
    running, or this one does not answer in time, media counts as busy: a retryable refusal, not an admission.
    """
    global _media_probe_future
    held = _media_held()
    if held:
        return held
    pending = _media_probe_future
    if pending is not None and not pending.done():
        return _MEDIA_BUSY
    future = _media_probe_future = _media_probe_pool.submit(_studio_media_resident)
    try:
        return await asyncio.wait_for(asyncio.wrap_future(future), _MEDIA_PROBE_TIMEOUT_S)
    except asyncio.TimeoutError:
        return _MEDIA_BUSY


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


async def _unload_omlx(config: AttachedEnginesConfig) -> list[str]:
    """Unload every oMLX model. An oMLX the user switched off in the desktop shell and that does not answer holds nothing, so it is not a failure."""
    client = OmlxClient(config.omlx_url)
    try:
        return await client.unload_all()
    except AttachedEngineBusy:
        raise
    except AttachedEngineError:
        try:
            switched_off = desktop_settings.helper_wanted("omlx") is False
            if switched_off and not (await client.status()).reachable:
                return []
        except Exception:
            pass
        raise


def _freed_action(urls: list[str]) -> str:
    return f"Freed ComfyUI memory: {', '.join(urls)}"


async def _admit(reason: str, *, local: bool, media_hold: Optional[MediaHold] = None) -> ArbiterResult:
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
                media = await _media_resident_async()
                if media:
                    raise AttachedAdmissionError(f"Studio {media}; unload it to use oMLX.")
            if not local and config.arbitrate_comfyui:
                if _comfyui_held():
                    raise AttachedAdmissionError("Studio has a ComfyUI job starting; retry shortly to use oMLX.")
                refusal = await _comfyui_refusal(config, "to use oMLX", cached = True)
                if refusal:
                    raise AttachedAdmissionError(refusal)
            if local:
                if config.arbitrate_comfyui:
                    if _comfyui_held():
                        raise AttachedAdmissionError(
                            "Studio has a ComfyUI job starting; wait for it to finish before loading a local model."
                        )
                    refusal = await _comfyui_refusal(config, "before loading a local model")
                    if refusal:
                        raise AttachedAdmissionError(refusal)
                if media_hold is not None:
                    _register_media_hold(media_hold)
                freed = await free_comfyui() or []
                unloaded = await _unload_omlx(config)
                actions = [f"Unloaded oMLX: {', '.join(unloaded)}"] if unloaded else []
                if freed:
                    actions.append(_freed_action(freed))
                if actions:
                    _record(reason, actions)
                return ArbiterResult(acted = bool(actions), actions = tuple(actions))
            return ArbiterResult()
    except Exception as exc:
        if media_hold is not None:
            media_hold.release()
        message = f"Cannot free shared memory for {reason}: {exc}"
        logger.warning("%s", message)
        return ArbiterResult(skipped = "error", error = message)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _origin(url: str) -> Optional[tuple[str, str, int]]:
    try:
        parts = urlsplit(str(url).strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if not host or scheme not in _DEFAULT_PORTS:
        return None
    host = host.rstrip(".").lower()
    return scheme, "loopback" if host in _LOOPBACK_HOSTS else host, port or _DEFAULT_PORTS[scheme]


def targets_omlx(provider_type: Optional[str], base_url: Optional[str]) -> bool:
    """True when a provider request reaches the attached oMLX: its type, or a base URL on its origin."""
    if provider_type == "omlx":
        return True
    if not base_url:
        return False
    target = _origin(base_url)
    return target is not None and target == _origin(get_config().omlx_url)


async def free_for_local(reason: Reason, *, media_hold: Optional[MediaHold] = None) -> ArbiterResult:
    """Admit local work. A Studio image or video load passes its ``MediaHold``, registered under the arbiter lock on
    success so oMLX is refused from this moment until the route releases it (or the backend reports the load)."""
    return await _admit(reason, local = True, media_hold = media_hold)


async def before_omlx_use() -> ArbiterResult:
    return await _admit("omlx_use", local = False)


async def before_omlx_load() -> ArbiterResult:
    """``before_omlx_use``, then free idle ComfyUI memory (Studio's and its peers') to make room for the load."""
    result = await before_omlx_use()
    if result.error or result.skipped:
        return result
    try:
        if not get_config().arbitrate_comfyui:
            return result
        async with _lock():
            freed = await free_comfyui() or []
    except Exception as exc:
        logger.info("ComfyUI memory release skipped: %s", type(exc).__name__)
        return result
    if not freed:
        return result
    actions = (*result.actions, _freed_action(freed))
    _record("omlx_load", list(actions))
    return replace(result, acted = True, actions = actions)


async def before_comfyui_job(hold: Optional[ComfyJobHold] = None) -> ArbiterResult:
    """Admit a Studio-submitted ComfyUI job: refuse next to training or any resident Studio model or a busy peer, then make room by unloading oMLX and freeing idle peers.

    On success ``hold`` (when given) is registered under the arbiter lock, so oMLX and local-load admission
    refuse from this moment until the caller releases it. Nothing is registered on a refusal or an error."""
    try:
        config = get_config()
        if not config.enabled:
            return ArbiterResult(skipped = "disabled")
        if not config.arbitrate_comfyui:
            return ArbiterResult(skipped = "arbitration_off")
        async with _lock():
            cancel_comfyui_idle_free()
            busy = _local_workload() or await _media_resident_async()
            if busy:
                raise AttachedAdmissionError(f"Studio {busy}; finish or unload it to run a ComfyUI job.")
            _, peers = _comfyui_targets(config)
            for peer in peers:
                state = await _comfyui_state(peer)
                if state == "busy":
                    raise AttachedAdmissionError(
                        f"The ComfyUI at {peer} is generating; wait for it to finish."
                    )
                if state == "unknown":
                    raise AttachedAdmissionError(
                        f"The ComfyUI at {peer} did not report its queue, so it may be generating; retry shortly."
                    )
            unloaded = await _unload_omlx(config)
            freed = await _free_urls(peers)
            actions = [f"Unloaded oMLX: {', '.join(unloaded)}"] if unloaded else []
            if freed:
                actions.append(_freed_action(freed))
            if actions:
                _record("comfyui_job", actions)
            if hold is not None:
                _register_comfyui_hold(hold)
            return ArbiterResult(acted = bool(actions), actions = tuple(actions))
    except Exception as exc:
        message = f"Cannot free shared memory for a ComfyUI job: {exc}"
        logger.warning("%s", message)
        return ArbiterResult(skipped = "error", error = message)


_IDLE_SLEEP = asyncio.sleep
_idle_free: Optional[asyncio.Task] = None
# Epoch seconds at which the pending idle free next looks at ComfyUI (None when none is pending). The
# desktop tray cannot ask the authenticated backend, so the same value is published as a marker file,
# like the engines' ``*.fail`` markers: ``{"free_at": epoch seconds}``.
_idle_free_at: Optional[float] = None
IDLE_MARKER = "comfyui-idle.json"


def _publish_idle_deadline(at: Optional[float]) -> None:
    global _idle_free_at
    _idle_free_at = at
    try:
        path = engines_home() / IDLE_MARKER
        if at is None:
            path.unlink(missing_ok = True)
            return
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_name(f".{IDLE_MARKER}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"free_at": at}), encoding = "utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("ComfyUI idle marker not updated: %s", type(exc).__name__)


def comfyui_idle_free_at() -> Optional[float]:
    """Epoch seconds of the next idle free, or None when none is pending."""
    task = _idle_free
    return _idle_free_at if task is not None and not task.done() else None


def cancel_comfyui_idle_free() -> None:
    """Drop the pending idle free, from any thread."""
    global _idle_free
    task, _idle_free = _idle_free, None
    if task is not None and not task.done():
        task.get_loop().call_soon_threadsafe(task.cancel)
    if task is not None:
        _publish_idle_deadline(None)


async def _idle_free_after(seconds: int) -> None:
    """After ``seconds`` of no Studio job, free Studio's ComfyUI once its queue is empty. Work in the queue pushes it back by another period."""
    global _idle_free
    try:
        while True:
            _publish_idle_deadline(time.time() + seconds)
            await _IDLE_SLEEP(seconds)
            async with _lock():
                config = get_config()
                own, _ = _comfyui_targets(config)
                if not own or not config.arbitrate_comfyui:
                    return
                queue = await ComfyuiClient(own).queue()
                if queue.state == "down":
                    return
                if queue.busy or _comfyui_held():
                    continue
                if await _free_urls([own]):
                    _record("comfyui_idle", [f"Freed ComfyUI memory after {seconds} s idle"])
                return
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info("ComfyUI idle free skipped: %s", type(exc).__name__)
    finally:
        if _idle_free is asyncio.current_task():
            _idle_free = None
            _publish_idle_deadline(None)


def schedule_comfyui_idle_free() -> bool:
    """(Re)start the idle timer; call after a Studio ComfyUI job, from the event loop. False when the setting is off."""
    global _idle_free
    cancel_comfyui_idle_free()
    config = get_config()
    if not config.enabled or not config.arbitrate_comfyui or config.comfyui_idle_free_s <= 0:
        return False
    _idle_free = asyncio.get_running_loop().create_task(_idle_free_after(config.comfyui_idle_free_s))
    _publish_idle_deadline(time.time() + config.comfyui_idle_free_s)
    return True


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
