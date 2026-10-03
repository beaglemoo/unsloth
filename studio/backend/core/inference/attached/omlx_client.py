# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Async client for the oMLX management API. ``status`` never raises: an engine that is down reports ``reachable = False``. ``load``, ``unload`` raise ``AttachedEngineError`` so a caller can tell a refusal from success."""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote

import httpx

from core.inference.attached import AttachedEngineError
from loggers import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(5.0, connect = 2.0)
_LOAD_TIMEOUT = httpx.Timeout(300.0, connect = 2.0)
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


@dataclass(frozen = True)
class OmlxStatus:
    reachable: bool
    models: tuple[OmlxModel, ...] = ()
    memory_bytes: int = 0
    ceiling_bytes: int = 0
    error: Optional[str] = None


def _int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


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
        return OmlxStatus(
            reachable = True,
            models = models,
            memory_bytes = _int(body.get("current_model_memory")),
            ceiling_bytes = _int(body.get("final_ceiling")),
        )

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
        await self._post(f"/v1/models/{quote(model_id, safe = '')}/unload", _TIMEOUT)

    async def unload_all(self) -> list[str]:
        """Unload every loaded model, pinned included. Returns the directory ids actually unloaded."""
        status = await self.status()
        targets: list[str] = []
        for model in status.models:
            if not model.loaded:
                continue
            target = self.resolve_dir(list(status.models), model.id) or model.id
            if target not in targets:
                targets.append(target)
        unloaded: list[str] = []
        for target in targets:
            try:
                await self.unload(target)
            except AttachedEngineError as exc:
                logger.warning("oMLX unload of %s failed: %s", target, exc)
                continue
            unloaded.append(target)
        return unloaded

    async def _post(self, path: str, timeout: httpx.Timeout) -> None:
        try:
            async with self._client(timeout) as client:
                response = await client.post(path)
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
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
