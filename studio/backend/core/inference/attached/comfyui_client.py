# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async client for a ComfyUI server (Studio's own helper on 8844, or a peer such as StoryPress on 8188).

``queue`` and ``system_stats`` never raise. ``queue`` tells three cases apart:

- confirmed not running (connection refused, nothing listening): ``reachable = False``, not busy;
- answering with a queue: ``reachable = True``;
- uncertain (an HTTP error status, a read timeout, invalid JSON, or a body that is not a queue):
  ``reachable = False`` with ``uncertain = True``. The display treats it as unreachable, admission
  treats it as busy because we cannot prove it idle.
Everything else raises ``ComfyuiError`` so a caller can tell a refusal from success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import quote

import httpx

from core.inference.attached import AttachedEngineError
from loggers import get_logger

logger = get_logger(__name__)

_STATUS_TIMEOUT = httpx.Timeout(3.0, connect = 2.0)
_SUBMIT_TIMEOUT = httpx.Timeout(30.0, connect = 2.0)


class ComfyuiError(AttachedEngineError):
    """ComfyUI is unreachable or refused a request."""


class ComfyuiNotRunning(ComfyuiError):
    """Nothing is listening at the URL (connection refused): the server is confirmed not running."""


class ComfyGraphError(ComfyuiError):
    """ComfyUI rejected a graph (HTTP 400); ``node_errors`` is its per-node detail."""

    def __init__(self, message: str, node_errors: Optional[dict] = None):
        super().__init__(message)
        self.node_errors = node_errors or {}


@dataclass(frozen = True)
class ComfyQueue:
    reachable: bool
    running_ids: tuple[str, ...] = ()
    pending_ids: tuple[str, ...] = ()
    # True when the server answered but the body was not a queue: not reachable for display, busy for admission.
    malformed: bool = False
    # True when a server that may be running did not give a usable answer (HTTP error, timeout, bad JSON).
    uncertain: bool = False
    # prompt_id -> the ``extra_data.unsloth`` marker of jobs Studio submitted.
    studio: dict = field(default_factory = dict)
    # prompt_id -> queue number (position key) as ComfyUI reports it.
    numbers: dict = field(default_factory = dict)

    @property
    def busy(self) -> bool:
        return self.malformed or self.uncertain or bool(self.running_ids or self.pending_ids)

    @property
    def state(self) -> str:
        """``down`` (confirmed not running), ``unknown`` (cannot tell), ``busy`` or ``idle``."""
        if self.uncertain or self.malformed:
            return "unknown"
        if not self.reachable:
            return "down"
        return "busy" if self.busy else "idle"


def _items(value: Any) -> Optional[list[tuple[str, Any, Any]]]:
    """``(prompt_id, number, unsloth marker)`` per queue entry, or None when the shape is wrong."""
    if not isinstance(value, list):
        return None
    out = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            return None
        prompt_id = entry[1]
        if not isinstance(prompt_id, str) or not prompt_id:
            return None
        extra = entry[3] if len(entry) > 3 and isinstance(entry[3], dict) else {}
        marker = extra.get("unsloth")
        out.append((prompt_id, entry[0], marker if isinstance(marker, dict) else None))
    return out


def parse_queue(body: Any) -> ComfyQueue:
    if not isinstance(body, dict):
        return ComfyQueue(reachable = False, malformed = True)
    running, pending = _items(body.get("queue_running")), _items(body.get("queue_pending"))
    if running is None or pending is None:
        return ComfyQueue(reachable = False, malformed = True)
    studio = {pid: marker for pid, _, marker in (*running, *pending) if marker is not None}
    numbers = {pid: number for pid, number, _ in (*running, *pending)}
    return ComfyQueue(
        reachable = True,
        running_ids = tuple(pid for pid, _, _ in running),
        pending_ids = tuple(pid for pid, _, _ in pending),
        studio = studio,
        numbers = numbers,
    )


def system_version(stats: Optional[dict]) -> Optional[str]:
    """``system.comfyui_version`` from a ``/system_stats`` body."""
    system = stats.get("system") if isinstance(stats, dict) else None
    value = system.get("comfyui_version") if isinstance(system, dict) else None
    return value if isinstance(value, str) and value else None


class ComfyuiClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    def _client(self, timeout: httpx.Timeout = _STATUS_TIMEOUT) -> httpx.AsyncClient:
        # trust_env off: an environment proxy must never see loopback engine traffic.
        return httpx.AsyncClient(
            base_url = self.base_url,
            timeout = timeout,
            transport = self._transport,
            trust_env = False,
        )

    async def _get_json(self, path: str, timeout: httpx.Timeout = _STATUS_TIMEOUT, **params: Any) -> Any:
        try:
            async with self._client(timeout) as client:
                response = await client.get(path, params = params or None)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise ComfyuiError(f"ComfyUI returned HTTP {exc.response.status_code} for {path}") from exc
        except httpx.ConnectError as exc:
            raise ComfyuiNotRunning(f"ComfyUI is unreachable ({type(exc).__name__})") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise ComfyuiError(f"ComfyUI is unreachable ({type(exc).__name__})") from exc

    async def _post(self, path: str, body: dict, timeout: httpx.Timeout = _STATUS_TIMEOUT) -> httpx.Response:
        try:
            async with self._client(timeout) as client:
                response = await client.post(path, json = body)
                response.raise_for_status()
                return response
        except httpx.HTTPStatusError as exc:
            raise ComfyuiError(f"ComfyUI returned HTTP {exc.response.status_code} for {path}") from exc
        except httpx.HTTPError as exc:
            raise ComfyuiError(f"ComfyUI is unreachable ({type(exc).__name__})") from exc

    async def queue(self) -> ComfyQueue:
        try:
            body = await self._get_json("/queue")
        except ComfyuiNotRunning:
            return ComfyQueue(reachable = False)
        except ComfyuiError:
            return ComfyQueue(reachable = False, uncertain = True)
        return parse_queue(body)

    async def system_stats(self) -> Optional[dict]:
        try:
            body = await self._get_json("/system_stats")
        except ComfyuiError:
            return None
        return body if isinstance(body, dict) else None

    async def version(self) -> Optional[str]:
        stats = await self.system_stats()
        return system_version(stats)

    async def folders(self) -> list[str]:
        body = await self._get_json("/models")
        if not isinstance(body, list):
            raise ComfyuiError("ComfyUI returned an unexpected /models payload")
        return [name for name in body if isinstance(name, str)]

    async def models(self, folder: str) -> list[str]:
        """File names in a model folder; an unknown folder (HTTP 404) is empty."""
        try:
            body = await self._get_json(f"/models/{quote(folder, safe = '')}")
        except ComfyuiError as exc:
            if "HTTP 404" in str(exc):
                return []
            raise
        if not isinstance(body, list):
            raise ComfyuiError("ComfyUI returned an unexpected model listing")
        return [name for name in body if isinstance(name, str)]

    async def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        await self._post("/free", {"unload_models": unload_models, "free_memory": free_memory})

    async def interrupt(self, prompt_id: Optional[str] = None) -> None:
        await self._post("/interrupt", {"prompt_id": prompt_id} if prompt_id else {})

    async def cancel_job(self, prompt_id: str) -> Optional[bool]:
        """``POST /api/jobs/{id}/cancel``: ComfyUI cancels the job atomically, whether it is pending or running.

        True when it acted on a job that was pending or running, False for one that already finished or is
        unknown, None when this ComfyUI has no such endpoint (the caller falls back to queue delete + interrupt).
        """
        try:
            response = await self._post(f"/api/jobs/{quote(prompt_id, safe = '')}/cancel", {})
        except ComfyuiError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise
        try:
            cancelled = response.json()["cancelled"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ComfyuiError("ComfyUI returned an unexpected cancel payload") from exc
        if not isinstance(cancelled, bool):
            raise ComfyuiError("ComfyUI returned an unexpected cancel payload")
        return cancelled

    async def delete_pending(self, ids: list[str]) -> None:
        await self._post("/queue", {"delete": list(ids)})

    async def clear_pending(self) -> None:
        await self._post("/queue", {"clear": True})

    async def submit(
        self, graph: dict, client_id: str, extra: Optional[dict] = None, prompt_id: Optional[str] = None
    ) -> tuple[str, int]:
        """Queue a graph. ``prompt_id`` (a canonical lowercase UUID) lets the caller name the job up front,
        so it can still cancel it when this call times out after ComfyUI queued the prompt."""
        body: dict[str, Any] = {"prompt": graph, "client_id": client_id}
        if prompt_id:
            body["prompt_id"] = prompt_id
        if extra:
            body["extra_data"] = extra
        try:
            async with self._client(_SUBMIT_TIMEOUT) as client:
                response = await client.post("/prompt", json = body)
        except httpx.HTTPError as exc:
            raise ComfyuiError(f"ComfyUI is unreachable ({type(exc).__name__})") from exc
        if response.status_code == 400:
            try:
                detail = response.json()
            except ValueError:
                detail = None
            error = detail.get("error") if isinstance(detail, dict) else None
            message = (
                error.get("message") or error.get("type") if isinstance(error, dict) else error
            )
            node_errors = detail.get("node_errors") if isinstance(detail, dict) else None
            raise ComfyGraphError(
                str(message or "ComfyUI rejected the graph"),
                node_errors if isinstance(node_errors, dict) else None,
            )
        if response.status_code >= 400:
            raise ComfyuiError(f"ComfyUI returned HTTP {response.status_code} for /prompt")
        try:
            data = response.json()
            prompt_id, number = data["prompt_id"], data.get("number", 0)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise ComfyuiError("ComfyUI returned an unexpected /prompt payload") from exc
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyuiError("ComfyUI returned an unexpected /prompt payload")
        return prompt_id, number if isinstance(number, int) and not isinstance(number, bool) else 0

    async def history(self, prompt_id: str) -> Optional[dict]:
        body = await self._get_json(f"/history/{quote(prompt_id, safe = '')}", _SUBMIT_TIMEOUT)
        entry = body.get(prompt_id) if isinstance(body, dict) else None
        return entry if isinstance(entry, dict) else None

    async def view(self, filename: str, subfolder: str = "", type: str = "output") -> bytes:
        try:
            async with self._client(_SUBMIT_TIMEOUT) as client:
                response = await client.get(
                    "/view", params = {"filename": filename, "subfolder": subfolder, "type": type}
                )
                response.raise_for_status()
                return response.content
        except httpx.HTTPStatusError as exc:
            raise ComfyuiError(f"ComfyUI returned HTTP {exc.response.status_code} for /view") from exc
        except httpx.HTTPError as exc:
            raise ComfyuiError(f"ComfyUI is unreachable ({type(exc).__name__})") from exc

    async def object_info(self) -> dict:
        """The node catalogue (large; used only to validate imported graphs)."""
        body = await self._get_json("/object_info", _SUBMIT_TIMEOUT)
        if not isinstance(body, dict):
            raise ComfyuiError("ComfyUI returned an unexpected /object_info payload")
        return body
