# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async client for the oMLX management API. ``status`` never raises: an engine that is down reports ``reachable = False``. ``load``, ``unload`` raise ``AttachedEngineError`` so a caller can tell a refusal from success."""

from __future__ import annotations

import asyncio
import posixpath
import re
import time
from dataclasses import dataclass, replace
from typing import Any, Optional
from urllib.parse import quote

import httpx

from core.inference.attached import AttachedEngineBusy, AttachedEngineError
from loggers import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(5.0, connect = 2.0)
_LOAD_TIMEOUT = httpx.Timeout(300.0, connect = 2.0)
# oMLX waits up to 20 s for in-flight requests before answering a graceful unload (200 or 409 model_busy).
_UNLOAD_TIMEOUT = httpx.Timeout(5.0, connect = 2.0, read = 25.0)
BUSY_MESSAGE = "oMLX is busy serving another client; retry shortly."
_EMBEDDING_ID = re.compile(r"embed", re.IGNORECASE)


@dataclass(frozen = True)
class OmlxModel:
    id: str
    model_path: str = ""
    loaded: bool = False
    is_loading: bool = False
    estimated_size: int = 0
    pinned: bool = False
    engine_type: Optional[str] = None
    is_helper: bool = False
    model_alias: Optional[str] = None
    last_access: Optional[float] = None
    # Resolved prompt cap, native length and output cap from /v1/models/status.
    max_context_window: Optional[int] = None
    model_context_length: Optional[int] = None
    max_tokens: Optional[int] = None
    # Idle unload: the TTL that applies and the seconds left before it fires. None while
    # unloaded, loading, pinned, or when no TTL applies.
    ttl_s: Optional[int] = None
    idle_remaining_s: Optional[float] = None


@dataclass(frozen = True)
class OmlxContext:
    dir: str
    # The per-model setting; None means the default applies.
    max_context_window: Optional[int]
    native_max: Optional[int]
    effective: Optional[int]


@dataclass(frozen = True)
class OmlxStatus:
    reachable: bool
    models: tuple[OmlxModel, ...] = ()
    memory_bytes: int = 0
    ceiling_bytes: int = 0
    error: Optional[str] = None


def _error_type(response: httpx.Response) -> Optional[str]:
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None
    kind = error.get("type") if isinstance(error, dict) else None
    return kind if isinstance(kind, str) else None


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _pos_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _epoch(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        return None
    return float(value)


def _parse_model(row: dict[str, Any]) -> Optional[OmlxModel]:
    model_id = row.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    alias = row.get("model_alias")
    return OmlxModel(
        id = model_id,
        model_path = row.get("model_path") if isinstance(row.get("model_path"), str) else "",
        loaded = bool(row.get("loaded")),
        is_loading = bool(row.get("is_loading")),
        estimated_size = _int(row.get("estimated_size")),
        pinned = bool(row.get("pinned")),
        engine_type = row.get("engine_type") if isinstance(row.get("engine_type"), str) else None,
        is_helper = bool(row.get("is_helper")),
        model_alias = alias if isinstance(alias, str) and alias else None,
        last_access = _epoch(row.get("last_access")),
        max_context_window = _pos_int(row.get("max_context_window")),
        model_context_length = _pos_int(row.get("model_context_length")),
        max_tokens = _pos_int(row.get("max_tokens")),
    )


def _row_for(models: tuple[OmlxModel, ...] | list[OmlxModel], api_id: str) -> Optional[OmlxModel]:
    for model in models:
        if model.id == api_id:
            return model
    for model in models:
        if model.model_alias == api_id:
            return model
    return None


class OmlxClient:
    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._transport = transport

    def _client(self, timeout: httpx.Timeout = _TIMEOUT) -> httpx.AsyncClient:
        # trust_env off: an environment proxy must never see loopback engine traffic.
        return httpx.AsyncClient(
            base_url = self.base_url,
            timeout = timeout,
            transport = self._transport,
            trust_env = False,
        )

    async def status(self) -> OmlxStatus:
        try:
            async with self._client() as client:
                response = await client.get("/v1/models/status")
                response.raise_for_status()
                body = response.json()
            rows = body.get("models") if isinstance(body, dict) else None
            if not isinstance(rows, list):
                raise ValueError("unexpected /v1/models/status payload")
        except (httpx.HTTPError, ValueError) as exc:
            return OmlxStatus(reachable = False, error = type(exc).__name__)
        models = tuple(m for m in (_parse_model(r) for r in rows if isinstance(r, dict)) if m)
        models = await self._with_idle_remaining(models)
        return OmlxStatus(
            reachable = True,
            models = models,
            memory_bytes = _int(body.get("current_model_memory")),
            ceiling_bytes = _int(body.get("final_ceiling")),
        )

    async def _ttl_sources(self) -> tuple[Optional[int], dict[str, Optional[int]]]:
        """The global idle timeout and each model directory's own ``ttl_seconds``. Best effort: ``(None, {})`` on any failure."""
        try:
            async with self._client() as client:
                settings, listing = await asyncio.gather(
                    client.get("/admin/api/global-settings"), client.get("/admin/api/models")
                )
                settings.raise_for_status()
                listing.raise_for_status()
                idle = settings.json().get("idle_timeout")
                global_ttl = _pos_int(idle.get("idle_timeout_seconds")) if isinstance(idle, dict) else None
                per_model: dict[str, Optional[int]] = {}
                for row in listing.json().get("models") or []:
                    if isinstance(row, dict) and isinstance(row.get("id"), str):
                        own = row.get("settings")
                        per_model[row["id"]] = (
                            _pos_int(own.get("ttl_seconds")) if isinstance(own, dict) else None
                        )
                return global_ttl, per_model
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None, {}

    async def _with_idle_remaining(self, models: tuple[OmlxModel, ...]) -> tuple[OmlxModel, ...]:
        """Fill ``ttl_s`` / ``idle_remaining_s`` on loaded, unpinned models. Mirrors ``EnginePool.check_ttl_expirations``: the per-model TTL wins over the global one, and the clock runs from ``last_access``."""
        if not any(m.loaded and not m.pinned for m in models):
            return models
        global_ttl, per_model = await self._ttl_sources()
        now = time.time()
        out = []
        for model in models:
            if not model.loaded or model.pinned or model.is_loading:
                out.append(model)
                continue
            dir_id = self.resolve_dir(list(models), model.id) or model.id
            ttl = per_model.get(dir_id) or per_model.get(model.id) or global_ttl
            last = model.last_access
            if last is None:
                dir_row = _row_for(models, dir_id)
                last = dir_row.last_access if dir_row else None
            if ttl is None or last is None:
                out.append(model)
                continue
            out.append(replace(model, ttl_s = ttl, idle_remaining_s = max(0.0, ttl - (now - last))))
        return tuple(out)

    async def context(self, dir_id: str) -> OmlxContext:
        """The per-model context cap for a model directory id, with the native length and the value in force."""
        try:
            async with self._client() as client:
                listing, status = await asyncio.gather(
                    client.get("/admin/api/models"), client.get("/v1/models/status")
                )
                listing.raise_for_status()
                status.raise_for_status()
                rows = listing.json().get("models") or []
                status_rows = status.json().get("models") or []
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise AttachedEngineError(f"oMLX is unreachable ({type(exc).__name__})") from exc
        row = next((r for r in rows if isinstance(r, dict) and r.get("id") == dir_id), None)
        if row is None:
            raise AttachedEngineError("Model not found on oMLX.")
        own = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        live = next(
            (r for r in status_rows if isinstance(r, dict) and r.get("id") == dir_id), {}
        )
        native = _pos_int(row.get("model_context_length"))
        setting = _pos_int(own.get("max_context_window"))
        return OmlxContext(
            dir = dir_id,
            max_context_window = setting,
            native_max = native,
            effective = _pos_int(live.get("max_context_window")) or setting or native,
        )

    async def set_context(self, dir_id: str, max_context_window: Optional[int]) -> bool:
        """Set (or with None, clear) the model's context cap. It applies immediately; returns oMLX's ``requires_reload``."""
        path = f"/admin/api/models/{quote(dir_id, safe = '')}/settings"
        try:
            async with self._client() as client:
                response = await client.put(path, json = {"max_context_window": max_context_window})
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPStatusError as exc:
            raise AttachedEngineError(
                f"oMLX returned HTTP {exc.response.status_code} for settings"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise AttachedEngineError(f"oMLX is unreachable ({type(exc).__name__})") from exc
        return bool(body.get("requires_reload")) if isinstance(body, dict) else False

    async def chat_model_ids(self, *, status: Optional[OmlxStatus] = None) -> Optional[list[str]]:
        """Ids the server lists on ``/v1/models`` minus embedding and helper (drafter) models, or None when the listing could not be fetched (timeout, HTTP error, malformed body), which is not the same as an empty catalog. Pass an already fetched ``status`` to spare a second round trip."""
        try:
            async with self._client() as client:
                response = await client.get("/v1/models")
                response.raise_for_status()
                data = response.json().get("data")
                if not isinstance(data, list):
                    raise ValueError("unexpected /v1/models payload")
            ids = [
                row["id"]
                for row in data
                if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"]
            ]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return None
        status = status if status is not None else await self.status()
        kept = []
        for model_id in ids:
            row = _row_for(status.models, model_id) if status.reachable else None
            if row is not None:
                if row.engine_type == "embedding" or row.is_helper:
                    continue
            elif _EMBEDDING_ID.search(model_id):
                continue
            kept.append(model_id)
        return kept

    async def loaded_ids(self) -> list[str]:
        status = await self.status()
        return [m.id for m in status.models if m.loaded]

    async def load(self, dir_id: str) -> None:
        await self._post(f"/admin/api/models/{quote(dir_id, safe = '')}/load", _LOAD_TIMEOUT)

    async def unload(self, model_id: str) -> None:
        await self._post(f"/v1/models/{quote(model_id, safe = '')}/unload", _UNLOAD_TIMEOUT)

    async def unload_all(self, *, timeout_s: float = 10.0) -> list[str]:
        """Unload pinned and loading models too, and verify that draining has finished.

        A 202 only acknowledges a drain; it is not proof that weights are gone.
        Fail closed if the status cannot be read or residency remains at the deadline.
        """
        status = await self.status()
        if not status.reachable:
            raise AttachedEngineError("Cannot verify oMLX residency: engine is unreachable.")
        targets: list[str] = []
        for model in status.models:
            if not (model.loaded or model.is_loading):
                continue
            target = self.resolve_dir(list(status.models), model.id) or model.id
            if target not in targets:
                targets.append(target)
        for target in targets:
            await self.unload(target)
        deadline = time.monotonic() + timeout_s
        while True:
            status = await self.status()
            if not status.reachable:
                raise AttachedEngineError("Cannot verify oMLX residency after unload.")
            resident = [m.id for m in status.models if m.loaded or m.is_loading]
            if not resident:
                return targets
            if time.monotonic() >= deadline:
                raise AttachedEngineError(
                    "oMLX models are still loaded or loading: " + ", ".join(resident)
                )
            await asyncio.sleep(0.2)

    async def _post(self, path: str, timeout: httpx.Timeout) -> None:
        try:
            async with self._client(timeout) as client:
                response = await client.post(path)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 409 and _error_type(exc.response) == "model_busy":
                raise AttachedEngineBusy(BUSY_MESSAGE) from exc
            raise AttachedEngineError(
                f"oMLX returned HTTP {exc.response.status_code} for {path.rsplit('/', 1)[-1]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AttachedEngineError(f"oMLX is unreachable ({type(exc).__name__})") from exc

    @staticmethod
    def resolve_dir(models: list[OmlxModel], api_id: str) -> str | None:
        """Map an id from ``/v1/models`` (directory name, alias or ``alias:profile``) to the model directory id the admin load endpoint accepts, via the shared ``model_path``."""
        row = _row_for(models, api_id)
        if row is None:
            return None
        if not row.model_path:
            return row.id
        siblings = [m for m in models if m.model_path == row.model_path]
        base = posixpath.basename(row.model_path.rstrip("/"))
        for model in siblings:
            if model.id == base:
                return model.id
        for model in siblings:
            if ":" not in model.id:
                return model.id
        return row.id
